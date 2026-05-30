"""
Image store — centralised in-memory storage and retrieval for loaded images.

Each entry in the store holds:
    path           – filesystem path to the saved file
    name           – human-readable filename
    metadata       – dict of image properties (width, height, dtype, etc.)
    array          – original numpy array (any dtype)
    display_array  – 8-bit normalised numpy array (for display)
"""

from pathlib import Path
from typing import Optional
import os

import numpy as np
from fastapi import HTTPException

from .image_processing import normalize_to_uint8

# Upload directory (created once at import time)
UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# The store itself: {image_id: dict}
_store: dict = {}


# --- Public API ---

def get(image_id: str) -> dict:
    """Return the store entry for *image_id*, or raise 404."""
    if image_id not in _store:
        raise HTTPException(404, "Image not found")
    return _store[image_id]


def put(image_id: str, entry: dict) -> None:
    """Insert or overwrite an entry."""
    _store[image_id] = entry


def delete(image_id: str) -> None:
    """Remove an image and optionally delete the file from uploads/."""
    entry = get(image_id)
    path = entry["path"]
    if os.path.exists(path) and str(UPLOAD_DIR) in path:
        os.remove(path)
    del _store[image_id]
    # Flush any cached pyramid levels
    try:
        from . import pyramid
        pyramid.clear_image(image_id)
    except Exception:
        pass


def list_all() -> list:
    """Return lightweight summaries for every stored image."""
    return [
        {"image_id": iid, "name": e["name"], "metadata": e["metadata"]}
        for iid, e in _store.items()
    ]


def contains(image_id: str) -> bool:
    return image_id in _store


# --- Array helpers ---

def load_array(image_id: str) -> np.ndarray:
    """Return the *original* numpy array, loading from source on first access.

    For regular files: open with PIL and materialise once. For LIF virtual
    entries: decode the currently-selected (channel, Z, T, M) plane via the
    LIF handle pool — never the whole stack.
    """
    entry = get(image_id)
    if entry.get("array") is not None:
        return entry["array"]

    if entry.get("source") == "lif":
        from . import lif_handles
        arr = lif_handles.read_full_plane(
            entry["lif_path"],
            entry["lif_index"],
            z=entry.get("lif_z"),
            t=entry.get("lif_t", 0),
            m=entry.get("lif_m", 0),
        )
        entry["array"] = arr
        return arr

    # Use the same robust multi-backend loader as ingestion so a file that
    # only decodes via tifffile/OpenCV (BigTIFF, tiled, odd compression)
    # reloads correctly on cache-miss instead of failing on bare Image.open().
    from .image_processing import load_image_array  # deferred: avoid circular import
    arr, _img, _loader = load_image_array(entry["path"])
    entry["array"] = arr
    return entry["array"]


def load_display_array(image_id: str) -> np.ndarray:
    """Return the 8-bit display-ready array, computing it lazily."""
    entry = get(image_id)
    if entry.get("display_array") is None:
        entry["display_array"] = normalize_to_uint8(load_array(image_id))
    return entry["display_array"]
