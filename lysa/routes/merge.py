"""
Channel merge routes — legacy 2-channel and N-channel merge.
"""

from __future__ import annotations

import io
import uuid

import numpy as np
from PIL import Image
from fastapi import APIRouter, HTTPException

from ..models import MergeParams, MergeNParams
from .. import store

router = APIRouter(prefix="/api", tags=["merge"])

# Mapping of colour names to RGB channel indices
_CHANNEL_MAP = {
    "red": [0], "green": [1], "blue": [2],
    "cyan": [1, 2], "magenta": [0, 2], "yellow": [0, 1], "white": [0, 1, 2],
}


def _do_merge(channel_specs: list[dict], blend_mode: str, merge_name: str | None) -> dict:
    """
    Core merge logic for N images.

    Each *spec* in *channel_specs* must have keys: image_id, channel, weight.
    Returns {"image_id": ..., "metadata": ...} after storing the result.
    """
    # Convert each source to grayscale float64
    grays = []
    for spec in channel_specs:
        arr = store.load_display_array(spec["image_id"]).copy()
        g = np.mean(arr[:, :, :3].astype(np.float64), axis=2) if len(arr.shape) == 3 else arr.astype(np.float64)
        grays.append(g)

    # Pad all to the same (max) size
    h = max(g.shape[0] for g in grays)
    w = max(g.shape[1] for g in grays)
    grays = [_pad(g, h, w) for g in grays]

    # Per-channel window/level + gamma (applied to grayscale BEFORE weight),
    # then weight multiply. Specs use 0..255 because grays are in display-array
    # range. Defaults (cl_min=0, cl_max=255, gamma=1) are pass-through.
    for i, spec in enumerate(channel_specs):
        cl_min = float(spec.get("cl_min", 0.0))
        cl_max = float(spec.get("cl_max", 255.0))
        gamma = max(1e-3, float(spec.get("gamma", 1.0)))
        rng = max(cl_max - cl_min, 1e-6)
        v = np.clip((grays[i] - cl_min) / rng, 0.0, 1.0)
        if gamma != 1.0:
            v = np.power(v, 1.0 / gamma)
        grays[i] = np.clip(v * 255.0 * float(spec["weight"]), 0, 255)

    # Composite
    if blend_mode == "max":
        composite = _blend_max(grays, channel_specs, h, w)
    elif blend_mode == "average":
        composite = _blend_average(grays, channel_specs, h, w)
    else:
        composite = _blend_additive(grays, channel_specs, h, w)

    composite = np.clip(composite, 0, 255).astype(np.uint8)

    # Build name
    names = [store.get(s["image_id"])["name"] for s in channel_specs]
    if not merge_name:
        merge_name = f"Merge({'+'.join(names)})"

    # Store result
    image_id = uuid.uuid4().hex[:8]
    save_path = store.UPLOAD_DIR / f"{image_id}.png"
    img = Image.fromarray(composite)
    img.save(save_path)

    buf = io.BytesIO()
    img.save(buf, format="PNG")

    metadata = {
        "filename": merge_name,
        "width": w, "height": h,
        "mode": "RGB", "channels": 3, "bands": ["R", "G", "B"],
        "dtype": "uint8", "size_bytes": buf.tell(),
        "min_value": int(composite.min()),
        "max_value": int(composite.max()),
        "mean_value": round(float(composite.mean()), 2),
        "sources": [s["image_id"] for s in channel_specs],
        "channel_assignments": [s["channel"] for s in channel_specs],
        "blend_mode": blend_mode,
    }

    store.put(image_id, {
        "path": str(save_path), "name": merge_name,
        "metadata": metadata, "array": composite,
    })
    return {"image_id": image_id, "metadata": metadata}


# --- Blend helpers ---

def _pad(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    result = np.zeros((target_h, target_w), dtype=np.float64)
    result[:arr.shape[0], :arr.shape[1]] = arr
    return result


def _assign(comp: np.ndarray, gray: np.ndarray, ch_name: str):
    for idx in _CHANNEL_MAP.get(ch_name, [1]):
        comp[:, :, idx] += gray


def _blend_additive(grays, specs, h, w):
    composite = np.zeros((h, w, 3), dtype=np.float64)
    for i, spec in enumerate(specs):
        _assign(composite, grays[i], spec["channel"])
    return composite


def _blend_max(grays, specs, h, w):
    layers = []
    for i, spec in enumerate(specs):
        layer = np.zeros((h, w, 3), dtype=np.float64)
        _assign(layer, grays[i], spec["channel"])
        layers.append(layer)
    composite = layers[0]
    for layer in layers[1:]:
        composite = np.maximum(composite, layer)
    return composite


def _blend_average(grays, specs, h, w):
    composite = np.zeros((h, w, 3), dtype=np.float64)
    count = np.zeros((h, w, 3), dtype=np.float64)
    for i, spec in enumerate(specs):
        layer = np.zeros((h, w, 3), dtype=np.float64)
        _assign(layer, grays[i], spec["channel"])
        mask = layer > 0
        composite += layer
        count += mask.astype(np.float64)
    return composite / np.maximum(count, 1)


# --- Endpoints ---

@router.post("/merge")
async def merge_images(params: MergeParams):
    """Legacy 2-image merge endpoint."""
    specs = [
        {"image_id": params.image_id_1, "channel": params.channel_1, "weight": params.weight_1,
         "cl_min": 0.0, "cl_max": 255.0, "gamma": 1.0},
        {"image_id": params.image_id_2, "channel": params.channel_2, "weight": params.weight_2,
         "cl_min": 0.0, "cl_max": 255.0, "gamma": 1.0},
    ]
    return _do_merge(specs, params.blend_mode, params.name)


@router.post("/merge-n")
async def merge_n_images(params: MergeNParams):
    """N-channel merge endpoint."""
    if len(params.channels) < 2:
        raise HTTPException(400, "Need at least 2 images to merge")
    specs = [
        {
            "image_id": c.image_id,
            "channel": c.channel,
            "weight": c.weight,
            "cl_min": c.cl_min,
            "cl_max": c.cl_max,
            "gamma": c.gamma,
        }
        for c in params.channels
    ]
    return _do_merge(specs, params.blend_mode, params.name)


class MergeInplaceParams(MergeNParams):
    """Re-render an existing composite in place using updated channel specs.
    The target image's pixels and metadata are replaced; its image_id is kept
    so client-side references (open tabs, ROIs, etc.) stay valid."""
    target_image_id: str


@router.post("/merge-n/inplace")
async def merge_n_inplace(params: MergeInplaceParams):
    """Re-merge using new per-channel specs and overwrite the target image."""
    if len(params.channels) < 2:
        raise HTTPException(400, "Need at least 2 images to merge")
    if not store.contains(params.target_image_id):
        raise HTTPException(404, f"Target image not found: {params.target_image_id}")
    specs = [
        {
            "image_id": c.image_id,
            "channel": c.channel,
            "weight": c.weight,
            "cl_min": c.cl_min,
            "cl_max": c.cl_max,
            "gamma": c.gamma,
        }
        for c in params.channels
    ]
    fresh = _do_merge(specs, params.blend_mode, params.name)
    new_id = fresh["image_id"]
    new_entry = store.get(new_id)
    # Move the new entry's pixels onto the target id, preserving the original
    # name so the sidebar card doesn't churn, and keeping the source list
    # for the channel-adjustments panel.
    target_entry = store.get(params.target_image_id)
    target_entry["array"] = new_entry["array"]
    target_entry["display_array"] = new_entry.get("display_array")
    new_meta = dict(new_entry["metadata"])
    new_meta["filename"] = target_entry["metadata"].get("filename", new_meta.get("filename"))
    target_entry["metadata"] = new_meta
    target_entry["path"] = new_entry["path"]
    # Drop the throwaway store entry we just created.
    store.delete(new_id)
    # Invalidate any tile pyramid cached for the target so OSD re-fetches.
    try:
        from .. import pyramid
        pyramid.clear_image(params.target_image_id)
    except Exception:
        pass
    return {"image_id": params.target_image_id, "metadata": new_meta}
