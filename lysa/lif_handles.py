"""
Long-lived `liffile.LifFile` handle pool.

`LifFile` opens the .lif on construction and reads its index — cheap. We keep
one open handle per absolute path so we can serve thousands of `frame()` calls
for tens of sub-images without re-opening the file each time.

Frame reads are protected by a per-path lock because the underlying file
position is stateful inside `liffile`.
"""

from __future__ import annotations

import os
import threading
from typing import Dict, List, Tuple

import numpy as np

# liffile is imported lazily so module import stays cheap
_LifFile = None
_LifImage = None


def _lazy_import():
    global _LifFile, _LifImage
    if _LifFile is None:
        from liffile import LifFile, LifImage  # type: ignore
        _LifFile = LifFile
        _LifImage = LifImage
    return _LifFile


_handles: Dict[str, "object"] = {}            # abs_path -> LifFile
_image_lists: Dict[str, List["object"]] = {}  # abs_path -> [LifImage, ...] (order = file order)
_locks: Dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _path_lock(path: str) -> threading.RLock:
    """Reentrant — read_plane holds it across a nested open_lif() call."""
    with _locks_guard:
        lk = _locks.get(path)
        if lk is None:
            lk = threading.RLock()
            _locks[path] = lk
        return lk


def _abspath(path: str) -> str:
    return os.path.abspath(path)


def open_lif(path: str):
    """Open or return the cached LifFile handle for *path*."""
    LifFile = _lazy_import()
    abspath = _abspath(path)
    with _path_lock(abspath):
        h = _handles.get(abspath)
        if h is None:
            h = LifFile(abspath)
            _handles[abspath] = h
            _image_lists[abspath] = list(h.images)
        return h


def list_images(path: str) -> List["object"]:
    """Return the ordered list of LifImage objects for *path*."""
    open_lif(path)
    return _image_lists[_abspath(path)]


def get_image(path: str, index: int):
    """Return the LifImage at positional *index* in *path*."""
    imgs = list_images(path)
    if index < 0 or index >= len(imgs):
        raise IndexError(f"LIF index {index} out of range (have {len(imgs)})")
    return imgs[index]


def close(path: str) -> None:
    abspath = _abspath(path)
    with _path_lock(abspath):
        h = _handles.pop(abspath, None)
        _image_lists.pop(abspath, None)
        if h is not None:
            try:
                h.close()
            except Exception:
                pass


def close_all() -> None:
    for p in list(_handles.keys()):
        close(p)


# ---------------------------------------------------------------------------
# Plane reads
# ---------------------------------------------------------------------------

def default_plane_indices(lif_img) -> Dict[str, int]:
    """
    Pick a sensible default for the non-spatial dims so we get one 2D plane.
    - Z: middle slice (most informative for stacks, no projection cost)
    - M / T: 0
    - C / S: not specified — frame() returns the full innermost 2 dims +
      optional sample axis, so we leave channel handling to the caller.
    """
    sizes = dict(lif_img.sizes)
    out: Dict[str, int] = {}
    if "Z" in sizes and sizes["Z"] > 1:
        out["Z"] = sizes["Z"] // 2
    for d in ("M", "T"):
        if d in sizes:
            out[d] = 0
    return out


def read_plane(
    path: str,
    index: int,
    channel: int = 0,
    z: int | None = None,
    t: int = 0,
    m: int = 0,
) -> np.ndarray:
    """
    Decode a single 2D plane from a LIF sub-image.

    Returns an HxW uint16 (or whatever the file's dtype is) numpy array.
    Holds the per-path lock for the duration of the read.
    """
    abspath = _abspath(path)
    with _path_lock(abspath):
        img = get_image(abspath, index)
        sizes = dict(img.sizes)

        idx: Dict[str, int] = {}
        if "C" in sizes:
            idx["C"] = max(0, min(channel, sizes["C"] - 1))
        if "Z" in sizes:
            idx["Z"] = (sizes["Z"] // 2) if z is None else max(0, min(z, sizes["Z"] - 1))
        if "T" in sizes:
            idx["T"] = max(0, min(t, sizes["T"] - 1))
        if "M" in sizes:
            idx["M"] = max(0, min(m, sizes["M"] - 1))

        plane = img.frame(**idx)
        return np.ascontiguousarray(plane)


def read_full_plane(
    path: str,
    index: int,
    z: int | None = None,
    t: int = 0,
    m: int = 0,
) -> np.ndarray:
    """
    Read all channels at a single Z/T/M position, stacked into HxW (1ch) or
    HxWxC (multi-channel). Returns the **native** channel count — no padding,
    no truncation. Downstream code (PNG encoder, frontend channel-map UI)
    handles the case where C != 3.

    For LIF datasets this matters: a brightfield channel as the 4th channel
    of a 4-channel acquisition is independent signal, not RGBA alpha. Padding
    2-channel to 3 (with a zero blue) used to be done here for the legacy
    display pipeline, but that decision now lives on the frontend via the
    channelMap state.
    """
    abspath = _abspath(path)
    with _path_lock(abspath):
        img = get_image(abspath, index)
        sizes = dict(img.sizes)
        n_channels = sizes.get("C", 1)
        planes = []
        for c in range(n_channels):
            planes.append(read_plane(abspath, index, channel=c, z=z, t=t, m=m))
        if n_channels == 1:
            return planes[0]
        return np.stack(planes, axis=-1)


def read_thumbnail_raw(
    path: str,
    index: int,
    max_dim: int = 256,
) -> np.ndarray:
    """Strided thumbnail of channel 0 in the source dtype (e.g. uint16)."""
    plane = read_plane(path, index, channel=0)
    h, w = plane.shape[:2]
    sy = max(1, h // max_dim)
    sx = max(1, w // max_dim)
    return np.ascontiguousarray(plane[::sy, ::sx])


def thumbnail_stats(thumb: np.ndarray) -> Tuple[float, float, float, float, float]:
    """(min, max, mean, p1, p99) on the source-dtype thumbnail."""
    f = thumb.astype(np.float32)
    return (
        float(f.min()),
        float(f.max()),
        float(f.mean()),
        float(np.percentile(f, 1)),
        float(np.percentile(f, 99)),
    )


# ---------------------------------------------------------------------------
# Pixel-size extraction (kept here so it lives next to the LIF code)
# ---------------------------------------------------------------------------

_UNIT_TO_UM = {"m": 1e6, "mm": 1e3, "µm": 1.0, "um": 1.0, "nm": 1e-3}


def extract_pixel_size(lif_img) -> dict:
    """
    Physical pixel size from LIF metadata, normalised to µm/px.

    Reads the `DimensionDescription` elements in the image's XML — DimID 1
    is X, 2 is Y, 3 is Z. Each has Length (in `Unit`) and NumberOfElements.

    Avoids `lif_img.coords`, which on liffile 2026 calls `numpy.astype(...)`
    that doesn't exist before NumPy 2.0.
    """
    result = {"pixel_size_x": None, "pixel_size_y": None, "pixel_size_unit": None}
    try:
        xml = getattr(lif_img, "xml_element", None)
        if xml is None:
            return result
        for dim in xml.iter("DimensionDescription"):
            dim_id = dim.attrib.get("DimID")
            if dim_id not in ("1", "2"):
                continue
            try:
                n = int(dim.attrib.get("NumberOfElements", "0"))
                length = float(dim.attrib.get("Length", "0") or 0)
            except ValueError:
                continue
            if n <= 1 or length == 0:
                continue
            unit = dim.attrib.get("Unit") or "m"
            scale = _UNIT_TO_UM.get(unit, 1e6)  # default: assume metres
            spacing_um = abs(length) * scale / (n - 1)
            if dim_id == "1":
                result["pixel_size_x"] = round(spacing_um, 6)
            elif dim_id == "2":
                result["pixel_size_y"] = round(spacing_um, 6)
            result["pixel_size_unit"] = "µm"
    except Exception:
        pass
    return result
