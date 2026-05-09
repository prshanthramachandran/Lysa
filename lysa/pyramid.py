"""
Image pyramid builder and tile server backing store.

Builds a multi-resolution pyramid for each image and serves tiles in a
DeepZoom-compatible layout so OpenSeadragon can consume them directly.

Design notes
------------
- Pyramid levels are created lazily, the first time a level is requested.
  Level 0 = full resolution. Level L = full resolution / 2**L.
- Levels are cached in memory as uint8 RGB numpy arrays. The full image and
  the first couple of levels dominate memory; levels past ~4 are tiny and
  essentially free.
- Tiles are rendered on the fly from the cached level arrays. We do not
  persist tiles to disk — PNG encoding of a 512x512 uint8 tile is <5ms on
  modern hardware and avoids disk bloat. If this ever becomes a bottleneck
  the tile rendering can be wrapped in an LRU cache trivially.
- DeepZoom layout: the client asks for level indices where level 0 is the
  smallest (1x1 pixel) and level (N-1) is the full image. We translate.
"""

from __future__ import annotations

import io
import math
import threading
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from .image_processing import normalize_to_uint8
from . import store

# Per-image pyramid caches. _levels[image_id][level_down] = uint8 RGB array
# where level_down=0 is full resolution and increasing values halve each axis.
_levels: Dict[str, Dict[int, np.ndarray]] = {}
# v3 per-channel pyramid cache: _channel_levels[image_id][channel_idx][level_down]
# = uint8 grayscale 2-D array. Built lazily, one channel at a time, the first
# time a channel-specific tile is requested. Linearly scaled (not percentile
# stretched) so the shader's per-channel windowing math is well-defined.
_channel_levels: Dict[str, Dict[int, Dict[int, np.ndarray]]] = {}
# Cached per-channel scale factor (data_max in native dtype) so the shader
# can convert between native-range slider values and the pyramid's 0..255 range.
_channel_max: Dict[str, Dict[int, float]] = {}
# Cached per-channel (p1, p99) in native dtype units. Used to seed the
# frontend's default contrast window so freshly-loaded images render with
# the same percentile-stretched look as the legacy thumbnail.
_channel_percentiles: Dict[str, Dict[int, Tuple[float, float]]] = {}
# Per-image build locks so concurrent tile requests for the same image
# coalesce on a single build, while different images don't serialize.
_build_locks: Dict[str, threading.Lock] = {}
_build_locks_guard = threading.Lock()

TILE_SIZE = 512
TILE_OVERLAP = 1  # DeepZoom requires a small overlap to hide seams


def _get_build_lock(image_id: str) -> threading.Lock:
    with _build_locks_guard:
        lock = _build_locks.get(image_id)
        if lock is None:
            lock = threading.Lock()
            _build_locks[image_id] = lock
        return lock


# ---------------------------------------------------------------------------
# Pyramid building
# ---------------------------------------------------------------------------

def _ensure_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    """Return an HxWx3 uint8 array, regardless of input channel count.

    Multi-channel microscopy arrays (4+ channels for fluorescence + brightfield)
    are clipped to the first 3 channels for the tile pyramid; the user's
    Channel Mapping panel handles arbitrary source-channel routing client-side.
    """
    if arr.dtype != np.uint8:
        arr = normalize_to_uint8(arr)
    if arr.ndim == 2:
        return np.ascontiguousarray(np.stack([arr, arr, arr], axis=-1))
    c = arr.shape[-1]
    if c == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif c == 2:
        zeros = np.zeros_like(arr[..., :1])
        arr = np.concatenate([arr, zeros], axis=-1)
    elif c >= 4:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------------
# v3: per-channel pyramid (one grayscale pyramid per source channel)
# ---------------------------------------------------------------------------

def _channel_to_uint8(arr: np.ndarray, channel_idx: int) -> Tuple[np.ndarray, float, float, float]:
    """Extract one source channel from a HxW or HxWxC array. Returns
    (uint8_array, native_max, p1, p99) where:
      - uint8_array: 2-D grayscale image linearly scaled by native_max
      - native_max: the channel's actual maximum in its source dtype
      - p1, p99: 1st/99th percentile of the channel in native units
                 (used by the frontend to seed sensible default contrast).

    Linear scaling preserves the relationship between native pixel values
    and uint8 storage so the shader's clMin/clMax windowing produces the
    same result regardless of pyramid level.
    """
    if arr.ndim == 2:
        ch = arr
    elif arr.ndim == 3:
        c = arr.shape[2]
        if channel_idx < 0 or channel_idx >= c:
            # Out of range — return a black tile rather than 500-ing
            ch = np.zeros(arr.shape[:2], dtype=arr.dtype)
        else:
            ch = arr[..., channel_idx]
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    # Subsample for percentile computation on large images — np.percentile
    # is O(n log n) and a 1939×1025 channel is ~2M elements; sampling every
    # ~10th pixel cuts that by an order of magnitude with negligible impact
    # on a 1/99 percentile.
    flat = ch.ravel()
    if flat.size > 200_000:
        stride = max(1, flat.size // 200_000)
        sample = flat[::stride]
    else:
        sample = flat

    if ch.dtype == np.uint8:
        p1 = float(np.percentile(sample, 1))
        p99 = float(np.percentile(sample, 99))
        return np.ascontiguousarray(ch), 255.0, p1, p99

    # Use the channel's actual maximum so dim signal isn't lost to a wide
    # dtype's range. Tracked alongside the pyramid so the shader can
    # translate its 0..255 sample back to native units for slider math.
    cmax = float(ch.max())
    if cmax <= 0:
        return np.zeros(ch.shape, dtype=np.uint8), 1.0, 0.0, 0.0
    p1 = float(np.percentile(sample, 1))
    p99 = float(np.percentile(sample, 99))
    scaled = (ch.astype(np.float32) / cmax * 255.0).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(scaled), cmax, p1, p99


def _downsample_2x_gray(arr: np.ndarray) -> np.ndarray:
    """2x LANCZOS downsample for a 2-D uint8 array.

    LANCZOS preserves edge sharpness much better than the simple 2×2 box
    average we used to do — after 10+ cascaded downsamples (which is
    what you see at fit-zoom for a 36 MP image) the difference is
    dramatic. Matches the resampling Root Measure uses for display.
    """
    h, w = arr.shape
    if h < 2 or w < 2:
        return arr
    pil = Image.fromarray(arr, mode="L")
    out = pil.resize((max(1, w // 2), max(1, h // 2)), Image.LANCZOS)
    return np.asarray(out, dtype=np.uint8)


def _build_channel_levels(image_id: str, channel_idx: int) -> Dict[int, np.ndarray]:
    """Build the full per-channel pyramid for `(image_id, channel_idx)`.

    Holds the per-image build lock so concurrent tile requests coalesce.
    """
    lock = _get_build_lock(image_id)
    with lock:
        cache = _channel_levels.setdefault(image_id, {}).get(channel_idx)
        if cache is not None and 0 in cache:
            return cache

        # Always build from the original (full bit-depth) array, not from
        # store.load_display_array() — we want native channel values.
        from . import store
        arr = store.load_array(image_id)
        base, cmax, p1, p99 = _channel_to_uint8(arr, channel_idx)
        _channel_max.setdefault(image_id, {})[channel_idx] = cmax
        _channel_percentiles.setdefault(image_id, {})[channel_idx] = (p1, p99)

        cache = {0: base}
        cur = base
        h, w = cur.shape
        ld = 0
        while max(h, w) > 1:
            cur = _downsample_2x_gray(cur)
            ld += 1
            cache[ld] = cur
            h, w = cur.shape
            if h == 0 or w == 0:
                break
        _channel_levels[image_id][channel_idx] = cache
        return cache


def get_channel_level(image_id: str, channel_idx: int, level_down: int) -> np.ndarray:
    cache = _channel_levels.get(image_id, {}).get(channel_idx)
    if cache is None or level_down not in cache:
        cache = _build_channel_levels(image_id, channel_idx)
    # _num_levels uses ceil(log2(max_dim))+1, but our integer-cropped
    # halving may stop one level shy of that (e.g. 1941px stops at 1×1
    # after 10 halvings, not 11). Clamp to the deepest built level so
    # OSD's smallest-DZ-level request resolves to the 1×1 thumbnail
    # instead of throwing KeyError.
    if level_down not in cache:
        level_down = max(cache.keys())
    return cache[level_down]


def get_channel_max(image_id: str, channel_idx: int) -> float:
    """Return the native-dtype max used to scale this channel into uint8.
    Triggers a pyramid build if the channel hasn't been seen yet."""
    cm = _channel_max.get(image_id, {}).get(channel_idx)
    if cm is None:
        _build_channel_levels(image_id, channel_idx)
        cm = _channel_max.get(image_id, {}).get(channel_idx, 1.0)
    return cm


def get_channel_percentiles(image_id: str, channel_idx: int) -> Tuple[float, float]:
    """Return (p1, p99) of the channel's pixel values in native units.
    Triggers a pyramid build if the channel hasn't been seen yet."""
    cps = _channel_percentiles.get(image_id, {}).get(channel_idx)
    if cps is None:
        _build_channel_levels(image_id, channel_idx)
        cps = _channel_percentiles.get(image_id, {}).get(channel_idx, (0.0, 1.0))
    return cps


def get_channel_tile_png(
    image_id: str,
    channel_idx: int,
    dz_level: int,
    col: int,
    row: int,
) -> bytes:
    """Render a single grayscale DeepZoom tile for one channel."""
    entry = store.get(image_id)
    meta = entry.get("metadata", {})
    full_w = int(meta.get("width"))
    full_h = int(meta.get("height"))
    num_levels = _num_levels(full_w, full_h)

    if dz_level < 0 or dz_level >= num_levels:
        raise ValueError(f"dz_level {dz_level} out of range [0,{num_levels-1}]")

    level_down = (num_levels - 1) - dz_level
    lvl_w, lvl_h = _level_dims(full_w, full_h, dz_level, num_levels)
    level_arr = get_channel_level(image_id, channel_idx, level_down)
    ah, aw = level_arr.shape
    lvl_w = min(lvl_w, aw)
    lvl_h = min(lvl_h, ah)

    x0 = col * TILE_SIZE - (TILE_OVERLAP if col > 0 else 0)
    y0 = row * TILE_SIZE - (TILE_OVERLAP if row > 0 else 0)
    x1 = (col + 1) * TILE_SIZE + TILE_OVERLAP
    y1 = (row + 1) * TILE_SIZE + TILE_OVERLAP
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(lvl_w, x1); y1 = min(lvl_h, y1)

    if x1 <= x0 or y1 <= y0:
        img = Image.new("L", (1, 1), 0)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    tile = level_arr[y0:y1, x0:x1]
    img = Image.fromarray(tile, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def _downsample_2x(arr: np.ndarray) -> np.ndarray:
    """2x LANCZOS downsample for uint8 RGB.

    Matches Root Measure's display resampling — preserves edge sharpness
    through many cascaded downsamples. Costs ~5× more than box-filter
    averaging (~3-5s vs 0.5s for a full 36 MP pyramid build), but the
    visual difference at fit-zoom is dramatic.
    """
    h, w = arr.shape[:2]
    if h < 2 or w < 2:
        return arr
    pil = Image.fromarray(arr)
    out = pil.resize((max(1, w // 2), max(1, h // 2)), Image.LANCZOS)
    return np.asarray(out, dtype=np.uint8)


def _build_all_levels(image_id: str) -> Dict[int, np.ndarray]:
    """
    Load the full-resolution image once and downsample iteratively until
    max(w, h) <= 1. Returns a dict {level_down: array}.

    Holds the per-image build lock for the duration so concurrent tile
    requests for the same image coalesce on a single build.
    """
    lock = _get_build_lock(image_id)
    with lock:
        # Re-check inside the lock — another thread may have just finished.
        cache = _levels.get(image_id)
        if cache is not None and 0 in cache:
            return cache

        base = _ensure_rgb_uint8(store.load_display_array(image_id))
        cache = {0: base}
        cur = base
        h, w = cur.shape[:2]
        ld = 0
        while max(h, w) > 1:
            cur = _downsample_2x(cur)
            ld += 1
            cache[ld] = cur
            h, w = cur.shape[:2]
            if h == 0 or w == 0:
                break
        _levels[image_id] = cache
        return cache


def get_level(image_id: str, level_down: int) -> np.ndarray:
    """
    Return the cached RGB uint8 array for `level_down` (0 = full res,
    1 = half, 2 = quarter, ...). Builds the entire pyramid on first call.
    """
    cache = _levels.get(image_id)
    if cache is None or level_down not in cache:
        cache = _build_all_levels(image_id)
    # See get_channel_level — clamp to the deepest built level rather
    # than KeyError-ing on the off-by-one between _num_levels and what
    # integer-cropped halving actually produces.
    if level_down not in cache:
        level_down = max(cache.keys())
    return cache[level_down]


def ensure_built(image_id: str) -> None:
    """Eagerly build the pyramid for `image_id` if not already cached."""
    if image_id not in _levels or 0 not in _levels[image_id]:
        _build_all_levels(image_id)


def clear_image(image_id: str) -> None:
    lock = _get_build_lock(image_id)
    with lock:
        _levels.pop(image_id, None)
        _channel_levels.pop(image_id, None)
        _channel_max.pop(image_id, None)
        _channel_percentiles.pop(image_id, None)
    with _build_locks_guard:
        _build_locks.pop(image_id, None)


# ---------------------------------------------------------------------------
# DeepZoom geometry
# ---------------------------------------------------------------------------

def _num_levels(width: int, height: int) -> int:
    """DeepZoom level count: ceil(log2(max(w, h))) + 1."""
    m = max(width, height)
    if m <= 1:
        return 1
    return int(math.ceil(math.log2(m))) + 1


def _level_dims(width: int, height: int, dz_level: int, num_levels: int) -> Tuple[int, int]:
    """Dimensions at a given DeepZoom level (0 = smallest, num_levels-1 = full)."""
    scale = 2 ** (num_levels - 1 - dz_level)
    return (
        max(1, math.ceil(width / scale)),
        max(1, math.ceil(height / scale)),
    )


def dzi_info(image_id: str) -> dict:
    """Return the info dict OpenSeadragon needs (DZI-compatible).

    Eagerly builds the pyramid so subsequent tile requests are served from
    in-memory cache without serializing on a cold-start build.
    """
    ensure_built(image_id)
    entry = store.get(image_id)
    meta = entry.get("metadata", {})
    w = int(meta.get("width"))
    h = int(meta.get("height"))
    return {
        "Image": {
            "xmlns": "http://schemas.microsoft.com/deepzoom/2008",
            "Format": "png",
            "Overlap": TILE_OVERLAP,
            "TileSize": TILE_SIZE,
            "Size": {"Width": w, "Height": h},
        },
        "num_levels": _num_levels(w, h),
        "width": w,
        "height": h,
        "tile_size": TILE_SIZE,
        "overlap": TILE_OVERLAP,
    }


def get_tile_png(
    image_id: str,
    dz_level: int,
    col: int,
    row: int,
) -> bytes:
    """
    Render a single DeepZoom tile as PNG bytes.

    OpenSeadragon's DeepZoom convention:
      - dz_level 0 = smallest (1x1 area)
      - dz_level num_levels-1 = full resolution
      - tile indices (col, row) are in tiles of size TILE_SIZE (+ overlap)
    """
    entry = store.get(image_id)
    meta = entry.get("metadata", {})
    full_w = int(meta.get("width"))
    full_h = int(meta.get("height"))
    num_levels = _num_levels(full_w, full_h)

    if dz_level < 0 or dz_level >= num_levels:
        raise ValueError(f"dz_level {dz_level} out of range [0,{num_levels-1}]")

    # How many times to downsample from full res to reach this level.
    level_down = (num_levels - 1) - dz_level
    # At this level the image has these dimensions:
    lvl_w, lvl_h = _level_dims(full_w, full_h, dz_level, num_levels)

    level_arr = get_level(image_id, level_down)
    # The cached level may be slightly larger than lvl_w x lvl_h due to our
    # integer-halving downsample (we crop odd dims). Clamp to cached size.
    ah, aw = level_arr.shape[:2]
    lvl_w = min(lvl_w, aw)
    lvl_h = min(lvl_h, ah)

    # Tile window in level coordinates (with overlap on all sides except
    # the outer borders).
    x0 = col * TILE_SIZE - (TILE_OVERLAP if col > 0 else 0)
    y0 = row * TILE_SIZE - (TILE_OVERLAP if row > 0 else 0)
    x1 = (col + 1) * TILE_SIZE + TILE_OVERLAP
    y1 = (row + 1) * TILE_SIZE + TILE_OVERLAP

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(lvl_w, x1)
    y1 = min(lvl_h, y1)

    if x1 <= x0 or y1 <= y0:
        # Empty tile — return 1x1 transparent
        img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    tile = level_arr[y0:y1, x0:x1]
    img = Image.fromarray(tile)
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)  # fast compression
    return buf.getvalue()
