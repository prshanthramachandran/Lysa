"""
Image filters / preprocessing routes.

Provides non-destructive preview and destructive "bake" operations for:
    - Gaussian blur
    - Median filter
    - Unsharp mask
    - Rolling-ball background subtraction

Preview endpoint returns a PNG — frontend paints it into an offscreen canvas
and swaps the image source without touching the stored data.

Bake endpoint applies the same pipeline to `entry["display_array"]` in the
shared store, pushing the previous array onto a per-image undo stack so the
operation can be reverted.
"""

from __future__ import annotations

import io
import threading
from typing import List, Optional

import numpy as np
from PIL import Image
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .. import store
from .. import pyramid
from ..image_processing import array_to_png_bytes

router = APIRouter(prefix="/api", tags=["filters"])


# ---------------------------------------------------------------------------
# Parameter models
# ---------------------------------------------------------------------------

class FilterParams(BaseModel):
    """All filter params are optional — enabled=False skips the step."""
    gaussian_enabled: bool = False
    gaussian_sigma: float = 1.0

    median_enabled: bool = False
    median_radius: int = 1

    unsharp_enabled: bool = False
    unsharp_radius: float = 1.0
    unsharp_amount: float = 1.0

    rollingball_enabled: bool = False
    rollingball_radius: int = 25


class BatchFilterParams(BaseModel):
    params: FilterParams
    image_ids: List[str]


# ---------------------------------------------------------------------------
# Undo snapshot cache (display_array snapshots, per image_id)
# ---------------------------------------------------------------------------

_undo_stacks: dict = {}
_undo_lock = threading.Lock()
_MAX_UNDO_DEPTH = 5


def _push_undo(image_id: str, arr: np.ndarray) -> None:
    with _undo_lock:
        stack = _undo_stacks.setdefault(image_id, [])
        stack.append(arr.copy())
        while len(stack) > _MAX_UNDO_DEPTH:
            stack.pop(0)


def _pop_undo(image_id: str) -> Optional[np.ndarray]:
    with _undo_lock:
        stack = _undo_stacks.get(image_id)
        if not stack:
            return None
        return stack.pop()


def _undo_depth(image_id: str) -> int:
    with _undo_lock:
        return len(_undo_stacks.get(image_id, []))


# ---------------------------------------------------------------------------
# Filter pipeline (uint8 in, uint8 out — RGB preserved by per-channel apply)
# ---------------------------------------------------------------------------

def _apply_single_channel(ch: np.ndarray, params: FilterParams) -> np.ndarray:
    """Run the enabled filters on a 2-D uint8 array in order."""
    # skimage/scipy lazy-imported so module import is cheap
    from scipy import ndimage as ndi
    from skimage import filters as skfilt
    from skimage.restoration import rolling_ball

    out = ch.astype(np.float32)

    # 1) Gaussian blur
    if params.gaussian_enabled and params.gaussian_sigma > 0:
        out = ndi.gaussian_filter(out, sigma=float(params.gaussian_sigma))

    # 2) Median filter (radius 1 -> 3x3 footprint)
    if params.median_enabled and params.median_radius > 0:
        size = int(params.median_radius) * 2 + 1
        out = ndi.median_filter(out, size=size)

    # 3) Unsharp mask — uses skimage helper on 0..1 float then rescale
    if params.unsharp_enabled and params.unsharp_amount > 0:
        norm = np.clip(out / 255.0, 0.0, 1.0)
        sharp = skfilt.unsharp_mask(
            norm,
            radius=float(params.unsharp_radius),
            amount=float(params.unsharp_amount),
            preserve_range=False,
        )
        out = np.clip(sharp * 255.0, 0.0, 255.0)

    # 4) Rolling-ball background subtraction
    if params.rollingball_enabled and params.rollingball_radius > 0:
        # Work on uint8 range for speed; radius is in pixels.
        try:
            background = rolling_ball(
                out.astype(np.uint8), radius=int(params.rollingball_radius)
            )
            out = out - background.astype(np.float32)
            out = np.clip(out, 0.0, 255.0)
        except Exception as exc:
            raise HTTPException(
                500, f"Rolling-ball failed (radius too large for image?): {exc}"
            )

    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _apply_pipeline(display: np.ndarray, params: FilterParams) -> np.ndarray:
    """Apply the filter pipeline to a display-ready uint8 array (2-D or 3-D)."""
    if display.ndim == 2:
        return _apply_single_channel(display, params)
    # 3-D: filter each channel independently, keep alpha untouched if present
    channels = []
    nchan = display.shape[2]
    for c in range(min(nchan, 3)):
        channels.append(_apply_single_channel(display[..., c], params))
    out = np.stack(channels, axis=-1)
    if nchan == 4:
        out = np.concatenate([out, display[..., 3:4]], axis=-1)
    return out


def _any_enabled(params: FilterParams) -> bool:
    return (
        params.gaussian_enabled
        or params.median_enabled
        or params.unsharp_enabled
        or params.rollingball_enabled
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/images/{image_id}/filter/preview")
def filter_preview(image_id: str, params: FilterParams):
    """
    Return the filtered display image as PNG. Does NOT mutate the store.
    Frontend paints this into an offscreen canvas to get an ImageData.
    """
    if not store.contains(image_id):
        raise HTTPException(404, f"Unknown image_id: {image_id}")

    display = store.load_display_array(image_id)
    if not _any_enabled(params):
        filtered = display
    else:
        filtered = _apply_pipeline(display, params)
    png = array_to_png_bytes(filtered)
    return StreamingResponse(io.BytesIO(png), media_type="image/png")


@router.post("/images/{image_id}/filter/apply")
def filter_apply(image_id: str, params: FilterParams):
    """
    Bake the filter into the stored display_array. Pushes the previous array
    onto the undo stack so it can be reverted with /filter/undo.
    Returns { undo_depth } so the UI can enable/disable the Undo button.
    """
    if not store.contains(image_id):
        raise HTTPException(404, f"Unknown image_id: {image_id}")

    if not _any_enabled(params):
        return {"image_id": image_id, "undo_depth": _undo_depth(image_id), "noop": True}

    entry = store.get(image_id)
    current = store.load_display_array(image_id)
    _push_undo(image_id, current)
    filtered = _apply_pipeline(current, params)
    entry["display_array"] = filtered
    pyramid.clear_image(image_id)
    return {
        "image_id": image_id,
        "undo_depth": _undo_depth(image_id),
        "applied": True,
    }


@router.post("/images/{image_id}/filter/undo")
def filter_undo(image_id: str):
    """Revert the most recent baked filter operation for this image."""
    if not store.contains(image_id):
        raise HTTPException(404, f"Unknown image_id: {image_id}")
    prev = _pop_undo(image_id)
    if prev is None:
        raise HTTPException(400, "Nothing to undo")
    entry = store.get(image_id)
    entry["display_array"] = prev
    pyramid.clear_image(image_id)
    return {"image_id": image_id, "undo_depth": _undo_depth(image_id), "reverted": True}


@router.get("/images/{image_id}/filter/undo-depth")
def filter_undo_depth(image_id: str):
    return {"image_id": image_id, "undo_depth": _undo_depth(image_id)}


@router.post("/filter/batch-apply")
def filter_batch_apply(payload: BatchFilterParams):
    """
    Bake the same filter pipeline into every requested image's display_array.
    Each image gets its own undo entry. Skips unknown image_ids rather than
    aborting the whole batch.
    """
    if not _any_enabled(payload.params):
        return {"ok": 0, "skipped": len(payload.image_ids), "results": []}

    results = []
    ok = 0
    for iid in payload.image_ids:
        try:
            if not store.contains(iid):
                results.append({"image_id": iid, "status": "unknown"})
                continue
            entry = store.get(iid)
            current = store.load_display_array(iid)
            _push_undo(iid, current)
            filtered = _apply_pipeline(current, payload.params)
            entry["display_array"] = filtered
            pyramid.clear_image(iid)
            results.append({"image_id": iid, "status": "ok"})
            ok += 1
        except HTTPException as exc:
            results.append({"image_id": iid, "status": "error", "detail": exc.detail})
        except Exception as exc:  # noqa: BLE001
            results.append({"image_id": iid, "status": "error", "detail": str(exc)})
    return {"ok": ok, "total": len(payload.image_ids), "results": results}
