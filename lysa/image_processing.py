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

def extract_pixel_size(img: Image.Image, file_path: str = "") -> dict:
    """
    Try to extract physical pixel size from image metadata.

    Returns a dict with:
        pixel_size_x: float or None  (physical units per pixel)
        pixel_size_y: float or None
        pixel_size_unit: str or None  ("µm", "nm", "mm", "cm", "m")
    """
    result = {"pixel_size_x": None, "pixel_size_y": None, "pixel_size_unit": None}

    try:
        # --- TIFF resolution tags ---
        tag = getattr(img, "tag_v2", None) or getattr(img, "tag", None)
        if tag:
            x_res = tag.get(282)  # XResolution (pixels per unit)
            y_res = tag.get(283)  # YResolution
            res_unit = tag.get(296, 2)  # ResolutionUnit: 1=none, 2=inch, 3=cm

            if x_res and res_unit in (2, 3):
                # x_res = pixels per unit-of-resolution
                if isinstance(x_res, tuple):
                    x_res = x_res[0] / x_res[1] if len(x_res) == 2 else x_res[0]
                if isinstance(y_res, tuple):
                    y_res = y_res[0] / y_res[1] if len(y_res) == 2 else y_res[0]

                if x_res and x_res > 0:
                    if res_unit == 3:  # centimeters
                        # pixels/cm → µm/pixel
                        result["pixel_size_x"] = round(1e4 / float(x_res), 6)
                        result["pixel_size_y"] = round(1e4 / float(y_res or x_res), 6)
                        result["pixel_size_unit"] = "µm"
                    elif res_unit == 2:  # inches
                        # pixels/inch → µm/pixel
                        result["pixel_size_x"] = round(25400.0 / float(x_res), 6)
                        result["pixel_size_y"] = round(25400.0 / float(y_res or x_res), 6)
                        result["pixel_size_unit"] = "µm"

        # --- ImageJ description tag ---
        if result["pixel_size_x"] is None and tag:
            desc = tag.get(270, "")
            if isinstance(desc, bytes):
                desc = desc.decode("utf-8", errors="ignore")
            if "unit=" in desc or "spacing=" in desc:
                import re
                # ImageJ often stores: unit=micron\nspacing=0.5
                unit_match = re.search(r"unit\s*=\s*(\S+)", desc)
                spacing_match = re.search(r"spacing\s*=\s*([0-9.eE+-]+)", desc)
                if spacing_match:
                    spacing = float(spacing_match.group(1))
                    unit_str = unit_match.group(1) if unit_match else "µm"
                    # Normalise unit
                    unit_map = {"micron": "µm", "um": "µm", "µm": "µm",
                                "nm": "nm", "mm": "mm", "cm": "cm", "m": "m"}
                    result["pixel_size_unit"] = unit_map.get(unit_str.lower(), unit_str)
                    result["pixel_size_x"] = round(spacing, 6)
                    result["pixel_size_y"] = round(spacing, 6)

    except Exception:
        pass  # Metadata extraction should never crash ingestion

    return result


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_image(file_path: str, filename: str) -> dict:
    """
    Open an image file, compute metadata, normalise for display, and return
    a dict ready for insertion into the store.

    Returns {"image_id": str, "metadata": dict, "entry": dict}.
    """
    img = Image.open(file_path)
    arr = np.array(img)
    display_arr = normalize_to_uint8(arr)

    # Percentiles on original data (for auto-contrast info)
    if len(arr.shape) == 2:
        flat = arr.flatten().astype(np.float64)
    else:
        flat = np.mean(arr[:, :, :3].astype(np.float64), axis=2).flatten()

    p1 = float(np.percentile(flat, 1))
    p99 = float(np.percentile(flat, 99))

    # Extract physical pixel size from metadata (TIFF tags, ImageJ, etc.)
    px_info = extract_pixel_size(img, file_path)

    image_id = uuid.uuid4().hex[:8]

    metadata = {
        "filename": filename,
        "width": img.width,
        "height": img.height,
        "mode": img.mode,
        "channels": len(img.getbands()),
        "bands": list(img.getbands()),
        "dtype": str(arr.dtype),
        "bit_depth": int(arr.dtype.itemsize * 8),
        "size_bytes": os.path.getsize(file_path),
        "min_value": int(arr.min()),
        "max_value": int(arr.max()),
        "mean_value": round(float(arr.mean()), 2),
        "percentile_1": round(p1, 2),
        "percentile_99": round(p99, 2),
        "pixel_size_x": px_info["pixel_size_x"],
        "pixel_size_y": px_info["pixel_size_y"],
        "pixel_size_unit": px_info["pixel_size_unit"],
    }

    entry = {
        "path": str(file_path),
        "name": filename,
        "metadata": metadata,
        "array": arr,
        "display_array": display_arr,
    }

    return {"image_id": image_id, "metadata": metadata, "entry": entry}
