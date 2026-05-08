"""
Leica LIF file loading routes — lazy / register-only.

Loading a LIF no longer decodes any pixel data. We:
  1. Open the file via the long-lived `lif_handles` pool.
  2. Enumerate sub-images and read their headers (sizes, dtype, pixel size).
  3. Register a lightweight store entry per requested sub-image. The entry's
     `array` and `display_array` are left empty — they materialise on first
     `/raw` request via the LIF-aware path in `store.load_array`.

Net effect: a 1.5 GB LIF with 20 sub-images is registered in <1 s with peak
RSS in the low 100s of MB, and only the sub-images the user actually opens
ever cost real memory.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional, List

import numpy as np
from fastapi import APIRouter, HTTPException

from ..models import LifLoadParams
from .. import store
from .. import lif_handles

router = APIRouter(prefix="/api", tags=["lif"])


# ---------------------------------------------------------------------------
# Header → metadata dict
# ---------------------------------------------------------------------------

def _build_metadata(
    lif_img,
    lif_path: Path,
    img_name: str,
    z_default: int,
    display_stem: str,
) -> dict:
    """Cheap header-only metadata. Stats come from the thumbnail."""
    sizes = dict(lif_img.sizes)
    width = int(sizes.get("X", 0))
    height = int(sizes.get("Y", 0))
    n_channels = int(sizes.get("C", 1))

    bands = ["L"] if n_channels == 1 else ["R", "G", "B"][:n_channels]
    if n_channels == 2:
        bands = bands + ["Z"]  # padded with zeros in read_full_plane

    px = lif_handles.extract_pixel_size(lif_img)

    dtype_str = str(getattr(lif_img, "dtype", "uint16"))
    bit_depth = int(np.dtype(dtype_str).itemsize * 8) if dtype_str else 16

    return {
        "filename": f"{display_stem}/{img_name}",
        "width": width,
        "height": height,
        "mode": "L" if n_channels == 1 else "RGB",
        "channels": n_channels,
        "bands": bands,
        "dtype": dtype_str,
        "bit_depth": bit_depth,
        "size_bytes": int(getattr(lif_img, "nbytes", 0)),
        # Stats are filled in below from the thumbnail
        "min_value": 0,
        "max_value": 0,
        "mean_value": 0.0,
        "percentile_1": 0.0,
        "percentile_99": 0.0,
        "pixel_size_x": px["pixel_size_x"],
        "pixel_size_y": px["pixel_size_y"],
        "pixel_size_unit": px["pixel_size_unit"],
        "lif_source": str(lif_path),
        "lif_image_name": img_name,
        "lif_sizes": sizes,
        "lif_dims": tuple(lif_img.dims),
        "lif_z_index": z_default,
    }


def _register_lif_image(
    lif_path: Path,
    idx: int,
    lif_img,
    display_stem: str,
) -> dict:
    """Build a virtual store entry for one LIF sub-image. No pixel decode."""
    img_name = lif_img.name or f"LIF_image_{idx}"
    sizes = dict(lif_img.sizes)
    z_default = (sizes.get("Z", 1) // 2) if sizes.get("Z", 1) > 1 else 0

    metadata = _build_metadata(lif_img, lif_path, img_name, z_default, display_stem)

    # One cheap thumbnail read fills in min/max/mean/p1/p99 — way better than
    # leaving the metadata blank, and 100x cheaper than a full asarray().
    try:
        thumb = lif_handles.read_thumbnail_raw(str(lif_path), idx, max_dim=256)
        mn, mx, mean, p1, p99 = lif_handles.thumbnail_stats(thumb)
        metadata.update({
            "min_value": int(mn),
            "max_value": int(mx),
            "mean_value": round(mean, 2),
            "percentile_1": round(p1, 2),
            "percentile_99": round(p99, 2),
        })
    except Exception:
        # If the thumbnail read fails for some weird sub-image, register
        # anyway with zero stats; the user can still try to open it.
        pass

    image_id = uuid.uuid4().hex[:8]
    entry = {
        "path": f"lif://{lif_path}#{idx}",
        "name": metadata["filename"],
        "metadata": metadata,
        "array": None,
        "display_array": None,
        "source": "lif",
        "lif_path": str(lif_path),
        "lif_index": idx,
        "lif_z": z_default,
        "lif_t": 0,
        "lif_m": 0,
    }
    store.put(image_id, entry)
    return {"image_id": image_id, "metadata": metadata}


# ---------------------------------------------------------------------------
# Shared loader (called from upload and load-lif)
# ---------------------------------------------------------------------------

async def _load_lif_internal(
    lif_path_str: str,
    indices: Optional[List[int]] = None,
    display_name: Optional[str] = None,
    name_pattern: Optional[str] = None,
) -> dict:
    """Register sub-images from a LIF file.

    `display_name` is the original user-facing filename (e.g.
    `20260506_MfnG_lines_roottip_t6.lif`); when uploads are renamed to a uuid
    on the way to disk, the caller passes the original name so sidebar
    entries stay readable. When loaded via /api/load-lif with a real path,
    we just use that path's stem.
    """
    lif_path = Path(lif_path_str)
    if not lif_path.is_file():
        raise HTTPException(400, f"File not found: {lif_path_str}")

    display_stem = Path(display_name).stem if display_name else lif_path.stem

    try:
        imgs = lif_handles.list_images(str(lif_path))
    except ImportError:
        raise HTTPException(500, "liffile library not installed. Run: pip install liffile[all]")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Failed to open LIF: {exc}")

    image_list = [
        {
            "index": i,
            "name": img.name,
            "sizes": dict(img.sizes),
            "dtype": str(getattr(img, "dtype", "unknown")),
        }
        for i, img in enumerate(imgs)
    ]

    load_indices = indices if indices is not None else list(range(len(imgs)))

    # Apply optional name-glob filter — case-insensitive fnmatch against the
    # sub-image name. Useful for "*Merged*" to skip per-tile sub-images on
    # tile-scan LIFs and load only the stitched composites.
    if name_pattern:
        import fnmatch
        pat = name_pattern.lower()
        load_indices = [
            idx for idx in load_indices
            if 0 <= idx < len(imgs) and fnmatch.fnmatch((imgs[idx].name or "").lower(), pat)
        ]

    results = []
    for idx in load_indices:
        if idx < 0 or idx >= len(imgs):
            results.append({"error": f"Index {idx} out of range", "filename": f"index_{idx}"})
            continue
        try:
            results.append(_register_lif_image(lif_path, idx, imgs[idx], display_stem))
        except Exception as exc:  # noqa: BLE001
            results.append({"error": str(exc), "filename": imgs[idx].name})

    return {
        "loaded": sum(1 for r in results if "image_id" in r),
        "errors": sum(1 for r in results if "error" in r),
        "images": results,
        "available_images": image_list,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/load-lif")
async def load_lif_file(params: LifLoadParams):
    """Register sub-images from a LIF without decoding pixel data."""
    return await _load_lif_internal(
        params.path,
        indices=params.image_indices,
        name_pattern=params.name_pattern,
    )


@router.get("/lif-info")
async def lif_info(path: str):
    """List sub-images in a LIF file. Pure header read, no decode."""
    lif_path = Path(path)
    if not lif_path.is_file():
        raise HTTPException(400, f"File not found: {path}")
    try:
        imgs = lif_handles.list_images(str(lif_path))
    except ImportError:
        raise HTTPException(500, "liffile library not installed. Run: pip install liffile[all]")

    image_list = [
        {
            "index": i,
            "name": img.name,
            "sizes": dict(img.sizes),
            "dtype": str(getattr(img, "dtype", "unknown")),
        }
        for i, img in enumerate(imgs)
    ]
    return {"path": str(lif_path), "num_images": len(image_list), "images": image_list}
