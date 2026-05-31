"""
Core image processing utilities — normalisation, format conversion, ingestion.

This module is intentionally free of FastAPI or request/response concerns.
Every function takes numpy arrays or plain values and returns results.
"""

import io
import os
import uuid
import base64
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Optional heavy-duty loaders for robust ingestion of large / unusual files.
# Imported defensively so the module still works if one is missing — the
# fallback chain in load_image_array() simply skips whatever isn't present.
try:
    import tifffile
except Exception:  # pragma: no cover - environment-dependent
    tifffile = None
try:
    import cv2
except Exception:  # pragma: no cover - environment-dependent
    cv2 = None


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_to_uint8(
    arr: np.ndarray,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> np.ndarray:
    """
    Map any-bit-depth array to uint8 [0, 255] using percentile scaling.

    Critical for 16-bit microscopy images where the actual signal may only
    occupy a small fraction of the 0–65535 range.
    """
    if arr.dtype == np.uint8:
        return arr

    arr_float = arr.astype(np.float64)
    p_low = np.percentile(arr_float, percentile_low)
    p_high = np.percentile(arr_float, percentile_high)

    if p_high <= p_low:
        p_low, p_high = float(arr_float.min()), float(arr_float.max())
    if p_high <= p_low:
        return np.zeros_like(arr, dtype=np.uint8)

    scaled = (arr_float - p_low) / (p_high - p_low) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------

def array_to_png_bytes(arr: np.ndarray) -> bytes:
    """Convert a numpy array to PNG bytes — always returns 3-channel RGB.

    Channel handling:
      - 1ch (HxW)            → broadcast to RGB
      - 2ch (HxWx2)          → channel 0 to R, channel 1 to G, B = 0
      - 3ch (HxWx3)          → directly RGB
      - 4+ch (HxWxN, N>=4)   → first 3 channels as RGB (drops alpha-or-beyond
                                — important for fluorescence + brightfield
                                LIFs where channel 4 is independent signal,
                                NOT alpha; the frontend channelMap UI lets
                                the user remap which source channel feeds
                                each RGB slot.)
    """
    if arr.dtype != np.uint8:
        arr = normalize_to_uint8(arr)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        c = arr.shape[2]
        if c == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif c == 2:
            zeros = np.zeros_like(arr[..., :1])
            arr = np.concatenate([arr, zeros], axis=-1)
        elif c >= 4:
            arr = arr[..., :3]
        # c == 3 → use as-is
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def array_to_base64(arr: np.ndarray) -> str:
    """Convert a numpy array to a base64-encoded PNG string."""
    return base64.b64encode(array_to_png_bytes(arr)).decode("utf-8")


# ---------------------------------------------------------------------------
# Pixel-size extraction
# ---------------------------------------------------------------------------

def _resolution_to_pixel_size(x_res, y_res, res_unit) -> dict:
    """Convert TIFF XResolution/YResolution (+ ResolutionUnit) to µm/pixel.

    res_unit follows the TIFF spec: 2 = inch, 3 = centimeter. Anything else
    yields no result. Values may be rationals as (num, den) tuples or plain
    numbers.
    """
    result = {"pixel_size_x": None, "pixel_size_y": None, "pixel_size_unit": None}

    def _ratio(v):
        if isinstance(v, tuple):
            return v[0] / v[1] if len(v) == 2 and v[1] else v[0]
        return v

    x = _ratio(x_res)
    y = _ratio(y_res) if y_res else x
    if not x or float(x) <= 0 or res_unit not in (2, 3):
        return result
    x = float(x)
    y = float(y or x)
    if res_unit == 3:        # pixels/cm → µm/pixel
        result["pixel_size_x"] = round(1e4 / x, 6)
        result["pixel_size_y"] = round(1e4 / y, 6)
    else:                    # pixels/inch → µm/pixel
        result["pixel_size_x"] = round(25400.0 / x, 6)
        result["pixel_size_y"] = round(25400.0 / y, 6)
    result["pixel_size_unit"] = "µm"
    return result


def _imagej_desc_pixel_size(desc) -> dict:
    """Parse pixel spacing from an ImageJ ImageDescription tag.

    ImageJ stores e.g. ``unit=micron\\nspacing=0.5``. Returns the standard
    pixel-size dict (all-None if nothing parseable).
    """
    result = {"pixel_size_x": None, "pixel_size_y": None, "pixel_size_unit": None}
    if isinstance(desc, bytes):
        desc = desc.decode("utf-8", errors="ignore")
    if not desc or ("unit=" not in desc and "spacing=" not in desc):
        return result
    import re
    spacing_match = re.search(r"spacing\s*=\s*([0-9.eE+-]+)", desc)
    if not spacing_match:
        return result
    unit_match = re.search(r"unit\s*=\s*(\S+)", desc)
    unit_str = unit_match.group(1) if unit_match else "µm"
    unit_map = {"micron": "µm", "um": "µm", "µm": "µm",
                "nm": "nm", "mm": "mm", "cm": "cm", "m": "m"}
    spacing = float(spacing_match.group(1))
    result["pixel_size_unit"] = unit_map.get(unit_str.lower(), unit_str)
    result["pixel_size_x"] = round(spacing, 6)
    result["pixel_size_y"] = round(spacing, 6)
    return result


def extract_pixel_size(img: Image.Image, file_path: str = "") -> dict:
    """Extract physical pixel size from a PIL image's TIFF / ImageJ metadata.

    Returns {pixel_size_x, pixel_size_y, pixel_size_unit} (all-None if
    unknown). Never raises — metadata extraction must not break ingestion.
    """
    result = {"pixel_size_x": None, "pixel_size_y": None, "pixel_size_unit": None}
    try:
        tag = getattr(img, "tag_v2", None) or getattr(img, "tag", None)
        if not tag:
            return result
        # XResolution=282, YResolution=283, ResolutionUnit=296
        result = _resolution_to_pixel_size(
            tag.get(282), tag.get(283), tag.get(296, 2))
        if result["pixel_size_x"] is None:  # ImageDescription=270
            result = _imagej_desc_pixel_size(tag.get(270, ""))
    except Exception:
        pass  # Metadata extraction should never crash ingestion
    return result


def extract_pixel_size_from_tiff(file_path: str) -> dict:
    """Extract pixel size straight from a TIFF on disk via tifffile.

    Used when the image was decoded by a non-PIL backend (BigTIFF, tiled,
    unusual compression), so no PIL tag dict is available. Returns all-None
    if tifffile is missing or the tags aren't present.
    """
    result = {"pixel_size_x": None, "pixel_size_y": None, "pixel_size_unit": None}
    if tifffile is None:
        return result
    try:
        with tifffile.TiffFile(file_path) as tif:
            tags = tif.pages[0].tags

            def _val(name, code):
                t = tags.get(name)
                if t is None:
                    t = tags.get(code)
                return t.value if t is not None else None

            unit = _val("ResolutionUnit", 296)
            result = _resolution_to_pixel_size(
                _val("XResolution", 282),
                _val("YResolution", 283),
                int(unit) if unit else 2)
            if result["pixel_size_x"] is None:
                result = _imagej_desc_pixel_size(
                    _val("ImageDescription", 270) or "")
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def _normalize_channel_axis(arr: np.ndarray) -> np.ndarray:
    """Move a channel-first axis to the end so downstream sees HxW or HxWxC.

    tifffile can return planar / channel-first arrays (C, H, W) for some
    multi-channel TIFFs. We only reorder when it's unambiguous — a small
    leading axis (2–4) with a clearly larger trailing axis — so genuine
    Z-stacks and ordinary HxWxC arrays are left untouched.
    """
    if arr.ndim == 3 and arr.shape[0] in (2, 3, 4) and arr.shape[2] > 4:
        arr = np.moveaxis(arr, 0, -1)
    return np.ascontiguousarray(arr)


def load_image_array(file_path: str):
    """Load an image to a numpy array with a robust multi-backend fallback.

    Order: Pillow (richest metadata) → tifffile (BigTIFF / tiled / unusual
    compression that Pillow rejects) → OpenCV (IMREAD_UNCHANGED, preserves
    bit depth). The first backend that succeeds wins.

    Returns (array, pil_image_or_None, loader_name). Raises RuntimeError
    only if every available backend fails.
    """
    errors = []

    # 1. Pillow — keep the PIL handle for its metadata / tag access.
    try:
        img = Image.open(file_path)
        arr = np.array(img)
        if arr.size == 0:
            raise ValueError("decoded an empty array")
        return arr, img, "pillow"
    except Exception as e:
        errors.append(f"pillow: {e}")

    # 2. tifffile — the heavy lifter for large / odd TIFFs.
    if tifffile is not None:
        try:
            arr = _normalize_channel_axis(tifffile.imread(file_path))
            if arr.size == 0:
                raise ValueError("decoded an empty array")
            return arr, None, "tifffile"
        except Exception as e:
            errors.append(f"tifffile: {e}")

    # 3. OpenCV — last resort; returns BGR(A), so swap to RGB(A).
    if cv2 is not None:
        try:
            arr = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise ValueError("cv2.imread returned None")
            if arr.ndim == 3 and arr.shape[2] == 3:
                arr = arr[..., ::-1]
            elif arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[..., [2, 1, 0, 3]]
            return np.ascontiguousarray(arr), None, "opencv"
        except Exception as e:
            errors.append(f"opencv: {e}")

    raise RuntimeError(
        f"Could not load image {file_path!r}. Tried: {'; '.join(errors)}")


def ingest_image(file_path: str, filename: str) -> dict:
    """
    Open an image file, compute metadata, normalise for display, and return
    a dict ready for insertion into the store.

    Uses load_image_array()'s fallback chain so large / odd TIFFs and other
    formats Pillow can't decode still ingest correctly.

    Returns {"image_id": str, "metadata": dict, "entry": dict}.
    """
    ext = Path(file_path).suffix.lower()
    arr, img, loader = load_image_array(file_path)
    display_arr = normalize_to_uint8(arr)

    # Percentiles on original data (for auto-contrast info)
    if arr.ndim == 2:
        flat = arr.flatten().astype(np.float64)
    else:
        flat = np.mean(arr[:, :, :3].astype(np.float64), axis=2).flatten()

    p1 = float(np.percentile(flat, 1))
    p99 = float(np.percentile(flat, 99))

    # Geometry / channel info — from the PIL handle when we have one,
    # otherwise derived from the decoded array shape.
    if img is not None:
        width, height = img.width, img.height
        mode = img.mode
        bands = list(img.getbands())
        channels = len(bands)
    else:
        height, width = arr.shape[0], arr.shape[1]
        channels = 1 if arr.ndim == 2 else arr.shape[2]
        mode = {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}.get(
            channels, f"{channels}CH")
        bands = list(mode) if channels <= 4 else [
            f"C{i}" for i in range(channels)]

    # Physical pixel size: PIL tags first; fall back to reading the TIFF
    # directly when PIL didn't decode it or carried no resolution tags.
    px_info = (extract_pixel_size(img, file_path) if img is not None else
               {"pixel_size_x": None, "pixel_size_y": None,
                "pixel_size_unit": None})
    if px_info["pixel_size_x"] is None and ext in {".tif", ".tiff"}:
        tif_px = extract_pixel_size_from_tiff(file_path)
        if tif_px["pixel_size_x"] is not None:
            px_info = tif_px

    # Keep integer ranges as ints; round floats so JSON stays compact.
    is_int = np.issubdtype(arr.dtype, np.integer)
    min_value = int(arr.min()) if is_int else round(float(arr.min()), 4)
    max_value = int(arr.max()) if is_int else round(float(arr.max()), 4)

    image_id = uuid.uuid4().hex[:8]

    metadata = {
        "filename": filename,
        "width": int(width),
        "height": int(height),
        "mode": mode,
        "channels": int(channels),
        "bands": bands,
        "dtype": str(arr.dtype),
        "bit_depth": int(arr.dtype.itemsize * 8),
        "size_bytes": os.path.getsize(file_path),
        "min_value": min_value,
        "max_value": max_value,
        "mean_value": round(float(arr.mean()), 2),
        "percentile_1": round(p1, 2),
        "percentile_99": round(p99, 2),
        "pixel_size_x": px_info["pixel_size_x"],
        "pixel_size_y": px_info["pixel_size_y"],
        "pixel_size_unit": px_info["pixel_size_unit"],
        # Provenance of the scale: "dpi" when recovered from image resolution
        # tags at ingest, else "none". A manual ruler calibration overwrites
        # this with "manual" via the /calibrate endpoint.
        "pixel_size_source": "dpi" if px_info["pixel_size_x"] else "none",
        "loader": loader,
    }

    entry = {
        "path": str(file_path),
        "name": filename,
        "metadata": metadata,
        "array": arr,
        "display_array": display_arr,
    }

    return {"image_id": image_id, "metadata": metadata, "entry": entry}
