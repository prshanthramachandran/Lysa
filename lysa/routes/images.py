"""
Image CRUD routes — upload, list, delete, raw, thumbnail.
"""

import hashlib
import io
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, Response

from ..image_processing import ingest_image, array_to_png_bytes
from .. import store

router = APIRouter(prefix="/api", tags=["images"])

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".lif"}


def _content_addressed_path(content: bytes, ext: str) -> Path:
    """Return uploads/<sha1[:12]><ext>. Same content → same path, so re-uploading
    the same file (common for the 1.3 GB LIF re-drag) reuses the existing copy
    instead of accumulating duplicates."""
    digest = hashlib.sha1(content).hexdigest()[:12]
    return store.UPLOAD_DIR / f"{digest}{ext}"


@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    name_pattern: Optional[str] = Form(None),
):
    """Upload an image file and store it.

    `name_pattern` is honored only for LIF uploads — a glob like ``*Merged*``
    that filters which sub-images get registered. Ignored for single-image
    formats.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format: {ext}")

    content = await file.read()
    save_path = _content_addressed_path(content, ext)
    if not save_path.exists():
        with open(save_path, "wb") as f:
            f.write(content)

    # LIF files need special handling. Pass the user's original filename so
    # sidebar entries read like "20260506_MfnG_lines_roottip_t6/Col_2"
    # instead of the upload-side uuid stem; pass the optional name_pattern
    # so drag-drop can filter sub-images server-side.
    if ext == ".lif":
        from .lif import _load_lif_internal
        return await _load_lif_internal(
            str(save_path),
            display_name=file.filename,
            name_pattern=name_pattern,
        )

    result = ingest_image(str(save_path), file.filename)
    store.put(result["image_id"], result["entry"])
    return {"image_id": result["image_id"], "metadata": result["metadata"]}


@router.get("/images")
async def list_images():
    """List all uploaded images."""
    return store.list_all()


@router.delete("/images/{image_id}")
async def delete_image(image_id: str):
    """Remove an image from the store."""
    store.delete(image_id)
    return {"status": "deleted"}


@router.get("/images/{image_id}/raw")
async def get_raw_image(image_id: str):
    """Serve the display-ready (8-bit) image as PNG."""
    display = store.load_display_array(image_id)
    png_data = array_to_png_bytes(display)
    return StreamingResponse(io.BytesIO(png_data), media_type="image/png")


@router.get("/images/{image_id}/raw_data")
async def get_raw_data(image_id: str):
    """Serve the original pixel data preserving native bit depth.

    Returns raw little-endian bytes with shape/dtype in response headers so
    the client can window/level 16-bit data without the precision loss from
    the percentile-stretched uint8 PNG path.

    Response headers:
        X-Lysa-Width, X-Lysa-Height — image dimensions in pixels
        X-Lysa-Channels — 1, 3, or 4
        X-Lysa-Dtype — numpy dtype string ("uint8", "uint16", "float32", ...)
        X-Lysa-Data-Min, X-Lysa-Data-Max — actual data range across the array
    """
    if not store.contains(image_id):
        raise HTTPException(404, "Image not found")
    arr = store.load_array(image_id)
    if arr is None:
        raise HTTPException(404, "No pixel data available")

    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2:
        h, w = arr.shape
        channels = 1
    elif arr.ndim == 3:
        h, w, channels = arr.shape
    else:
        raise HTTPException(500, f"Unsupported array shape: {arr.shape}")

    dtype_str = str(arr.dtype)
    # Force little-endian on the wire (most clients expect this)
    if arr.dtype.byteorder == ">":
        arr = arr.astype(arr.dtype.newbyteorder("<"))

    headers = {
        "Content-Type": "application/octet-stream",
        "X-Lysa-Width": str(w),
        "X-Lysa-Height": str(h),
        "X-Lysa-Channels": str(channels),
        "X-Lysa-Dtype": dtype_str,
        "X-Lysa-Data-Min": str(int(arr.min()) if np.issubdtype(arr.dtype, np.integer) else float(arr.min())),
        "X-Lysa-Data-Max": str(int(arr.max()) if np.issubdtype(arr.dtype, np.integer) else float(arr.max())),
        # Expose custom headers so JS can read them via fetch().response.headers
        "Access-Control-Expose-Headers":
            "X-Lysa-Width, X-Lysa-Height, X-Lysa-Channels, X-Lysa-Dtype, X-Lysa-Data-Min, X-Lysa-Data-Max",
    }
    return Response(content=arr.tobytes(), headers=headers, media_type="application/octet-stream")


@router.get("/images/{image_id}/thumbnail")
async def get_thumbnail(image_id: str, size: int = 150):
    """Get a resized thumbnail (always 8-bit RGB).

    Handles arrays with 1, 2, 3, 4, or 5+ channels — PIL only accepts a
    narrow set of layouts. The LIF loader now returns native channel counts
    (no more 2→3 padding), so the conversion to PNG-friendly RGB happens
    here.
    """
    display = store.load_display_array(image_id)
    if display.ndim == 2:
        display = np.stack([display, display, display], axis=-1)
    elif display.ndim == 3:
        c = display.shape[2]
        if c == 1:
            display = np.repeat(display, 3, axis=-1)
        elif c == 2:
            zeros = np.zeros_like(display[..., :1])
            display = np.concatenate([display, zeros], axis=-1)
        elif c >= 4:
            display = display[..., :3]
        # c == 3 → use as-is
    img = Image.fromarray(display)
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
