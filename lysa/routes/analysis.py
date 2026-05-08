"""
Analysis routes — histogram, ROI stats, line profile, measurement,
thresholding, edge detection, filters.
"""

import math
import uuid

import numpy as np
from fastapi import APIRouter, HTTPException
from scipy import ndimage
from skimage import filters, measure

from ..models import (
    AdjustmentParams, ROIParams, LineProfileParams, PolylineProfileParams,
    ThresholdParams, MeasurementParams, RotateParams, CropParams, AngledCropParams,
    ProfileExportParams, ShapedROIParams, PointStatsParams,
    ROIMeasurementExportParams, PointSetExportParams,
    MeasurementExportParams, ROIStatisticsExportParams,
    AnnotationSaveParams, AnnotationLoadParams,
)
from ..image_processing import array_to_base64
from .. import store
from .. import pyramid

router = APIRouter(prefix="/api", tags=["analysis"])


# ---------------------------------------------------------------------------
# Adjustments
# ---------------------------------------------------------------------------

@router.post("/images/{image_id}/adjust")
async def adjust_image(image_id: str, params: AdjustmentParams):
    """Apply brightness, contrast, gamma adjustments (server-side)."""
    arr = store.load_display_array(image_id).copy().astype(np.float64)

    # Channel isolation
    if params.channel and len(arr.shape) == 3:
        ch_map = {"red": 0, "green": 1, "blue": 2}
        if params.channel == "gray":
            arr = np.mean(arr[:, :, :3], axis=2)
        elif params.channel in ch_map:
            ch = ch_map[params.channel]
            gray = np.zeros_like(arr)
            gray[:, :, ch] = arr[:, :, ch]
            arr = gray

    # Brightness → Contrast → Gamma → Invert
    arr = arr + params.brightness
    mean = np.mean(arr)
    arr = (arr - mean) * params.contrast + mean
    arr = np.clip(arr / 255.0, 0, 1)
    arr = np.power(arr, 1.0 / params.gamma) * 255.0
    if params.invert:
        arr = 255.0 - arr

    return {"image": array_to_base64(np.clip(arr, 0, 255).astype(np.uint8))}


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------

@router.get("/images/{image_id}/histogram")
async def get_histogram(image_id: str, bins: int = 256):
    """Compute per-channel histograms."""
    display = store.load_display_array(image_id)
    histograms: dict = {}

    if len(display.shape) == 2:
        hist, _ = np.histogram(display.flatten(), bins=bins, range=(0, 256))
        histograms["gray"] = hist.tolist()
    elif len(display.shape) == 3:
        for i, name in enumerate(["red", "green", "blue"][:display.shape[2]]):
            hist, _ = np.histogram(display[:, :, i].flatten(), bins=bins, range=(0, 256))
            histograms[name] = hist.tolist()
        lum = np.mean(display[:, :, :3], axis=2)
        hist, _ = np.histogram(lum.flatten(), bins=bins, range=(0, 256))
        histograms["luminance"] = hist.tolist()

    orig = store.load_array(image_id)
    return {
        "histograms": histograms,
        "bins": bins,
        "original_dtype": str(orig.dtype),
        "original_range": [int(orig.min()), int(orig.max())],
    }


# ---------------------------------------------------------------------------
# ROI Statistics
# ---------------------------------------------------------------------------

def _build_roi_mask(shape: str, h: int, w: int, params: ShapedROIParams) -> np.ndarray:
    """Build a boolean 2D mask for the requested ROI shape."""
    mask = np.zeros((h, w), dtype=bool)

    if shape == "rect":
        if params.width is None or params.height is None or params.x is None or params.y is None:
            raise HTTPException(400, "Rectangle ROI requires x, y, width, height")
        y1 = max(0, min(int(params.y), h))
        y2 = max(0, min(int(params.y) + int(params.height), h))
        x1 = max(0, min(int(params.x), w))
        x2 = max(0, min(int(params.x) + int(params.width), w))
        mask[y1:y2, x1:x2] = True

    elif shape == "ellipse":
        if params.width is None or params.height is None or params.x is None or params.y is None:
            raise HTTPException(400, "Ellipse ROI requires x, y, width, height (bounding box)")
        x0 = float(params.x); y0 = float(params.y)
        bw = float(params.width); bh = float(params.height)
        if bw <= 0 or bh <= 0:
            return mask
        cx = x0 + bw / 2.0
        cy = y0 + bh / 2.0
        a = bw / 2.0  # x-radius
        b = bh / 2.0  # y-radius
        # Compute on the bounding box only for efficiency
        x1 = max(0, int(math.floor(x0)))
        y1 = max(0, int(math.floor(y0)))
        x2 = min(w, int(math.ceil(x0 + bw)))
        y2 = min(h, int(math.ceil(y0 + bh)))
        if x2 <= x1 or y2 <= y1:
            return mask
        yy, xx = np.ogrid[y1:y2, x1:x2]
        ellipse = ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1.0
        mask[y1:y2, x1:x2] = ellipse

    elif shape == "polygon":
        if not params.points or len(params.points) < 3:
            raise HTTPException(400, "Polygon ROI requires at least 3 points")
        from skimage.draw import polygon as sk_polygon
        rows = np.array([p[1] for p in params.points], dtype=np.float64)
        cols = np.array([p[0] for p in params.points], dtype=np.float64)
        rr, cc = sk_polygon(rows, cols, shape=(h, w))
        mask[rr, cc] = True

    else:
        raise HTTPException(400, f"Unknown ROI shape: {shape}")

    return mask


def _stats_from_mask(arr: np.ndarray, mask: np.ndarray,
                     pixel_size: float | None, pixel_unit: str | None) -> dict:
    """Compute per-ROI statistics given a 2D mask."""
    n_pixels = int(mask.sum())
    if n_pixels == 0:
        raise HTTPException(400, "Empty ROI (no pixels selected)")

    if arr.ndim == 2:
        region = arr[mask]
    else:
        # Luminance for overall stats, plus per-channel
        region = arr[mask]  # shape (N, C)
    region_flat = region if region.ndim == 1 else region.mean(axis=-1)

    stats: dict = {
        "area_pixels": n_pixels,
        "min": round(float(region_flat.min()), 2),
        "max": round(float(region_flat.max()), 2),
        "mean": round(float(region_flat.mean()), 2),
        "std": round(float(region_flat.std()), 2),
        "median": round(float(np.median(region_flat)), 2),
        "integrated_density": round(float(region_flat.sum()), 2),
        "raw_integrated_density": round(float(region_flat.sum()), 2),
    }

    # Physical area
    if pixel_size and pixel_size > 0 and pixel_unit:
        area_phys = n_pixels * (pixel_size ** 2)
        stats["area_physical"] = round(float(area_phys), 4)
        stats["area_unit"] = f"{pixel_unit}²"
        # Integrated density per physical unit ("IntDen" in Fiji = mean * area_phys)
        stats["integrated_density_physical"] = round(float(stats["mean"] * area_phys), 4)

    # Per-channel stats (full set: mean, std, min, max, median, integrated density)
    if arr.ndim == 3:
        channels = ["red", "green", "blue"][: arr.shape[2]]
        for i, ch in enumerate(channels):
            ch_data = arr[:, :, i][mask]
            if ch_data.size == 0:
                continue
            stats[f"{ch}_mean"] = round(float(ch_data.mean()), 2)
            stats[f"{ch}_std"] = round(float(ch_data.std()), 2)
            stats[f"{ch}_min"] = round(float(ch_data.min()), 2)
            stats[f"{ch}_max"] = round(float(ch_data.max()), 2)
            stats[f"{ch}_median"] = round(float(np.median(ch_data)), 2)
            stats[f"{ch}_integrated_density"] = round(float(ch_data.sum()), 2)
            if pixel_size and pixel_size > 0 and pixel_unit:
                area_phys = n_pixels * (pixel_size ** 2)
                stats[f"{ch}_integrated_density_physical"] = round(float(stats[f"{ch}_mean"] * area_phys), 4)

    # Histogram (of luminance / gray)
    hist, _ = np.histogram(region_flat, bins=64, range=(0, 256))
    stats["histogram"] = hist.tolist()
    return stats


@router.post("/images/{image_id}/roi-stats")
async def roi_statistics(image_id: str, roi: ROIParams):
    """Legacy rectangular ROI endpoint (kept for backward compat)."""
    arr = store.load_display_array(image_id)
    h, w = arr.shape[:2]
    shaped = ShapedROIParams(
        shape="rect", x=roi.x, y=roi.y, width=roi.width, height=roi.height,
    )
    mask = _build_roi_mask("rect", h, w, shaped)
    return _stats_from_mask(arr, mask, None, None)


@router.post("/images/{image_id}/shaped-roi-stats")
async def shaped_roi_statistics(image_id: str, params: ShapedROIParams):
    """
    Compute statistics for a rectangle, ellipse, or polygon ROI.
    Returns area in pixels (and physical units if pixel_size/unit given),
    mean/std/min/max/median, and integrated density (raw and physical).
    """
    arr = store.load_display_array(image_id)
    h, w = arr.shape[:2]
    mask = _build_roi_mask(params.shape, h, w, params)
    return _stats_from_mask(arr, mask, params.pixel_size, params.pixel_unit)


@router.post("/images/{image_id}/point-stats")
async def point_statistics(image_id: str, params: PointStatsParams):
    """
    Sample intensity at a list of points.

    For each point: returns pixel coordinates, physical coordinates (if pixel_size given),
    and the intensity value (single pixel if radius == 0, or the mean over a disk
    of the requested radius otherwise). Also computes inter-point distances between
    successive points (pixels and physical units).
    """
    if not params.points:
        raise HTTPException(400, "No points provided")

    arr = store.load_display_array(image_id)
    h, w = arr.shape[:2]
    r = max(0, int(params.radius))
    has_physical = bool(params.pixel_size and params.pixel_size > 0 and params.pixel_unit)

    def sample(px: float, py: float) -> dict:
        ix = int(round(px))
        iy = int(round(py))
        out: dict = {"x": ix, "y": iy}
        if 0 <= ix < w and 0 <= iy < h:
            if r == 0:
                val = arr[iy, ix]
                if arr.ndim == 2:
                    out["intensity"] = round(float(val), 2)
                else:
                    out["intensity"] = round(float(val.mean()), 2)
                    for i, ch in enumerate(["red", "green", "blue"][: arr.shape[2]]):
                        out[f"{ch}"] = int(val[i])
            else:
                y1 = max(0, iy - r); y2 = min(h, iy + r + 1)
                x1 = max(0, ix - r); x2 = min(w, ix + r + 1)
                yy, xx = np.ogrid[y1:y2, x1:x2]
                disk = (yy - iy) ** 2 + (xx - ix) ** 2 <= r ** 2
                if arr.ndim == 2:
                    out["intensity"] = round(float(arr[y1:y2, x1:x2][disk].mean()), 2)
                else:
                    region = arr[y1:y2, x1:x2]
                    lum = region.mean(axis=-1)
                    out["intensity"] = round(float(lum[disk].mean()), 2)
                    for i, ch in enumerate(["red", "green", "blue"][: arr.shape[2]]):
                        out[f"{ch}"] = round(float(region[:, :, i][disk].mean()), 2)
        else:
            out["intensity"] = None
        if has_physical:
            out["x_phys"] = round(ix * params.pixel_size, 4)
            out["y_phys"] = round(iy * params.pixel_size, 4)
        return out

    samples = [sample(float(p[0]), float(p[1])) for p in params.points]

    # Inter-point distances (between successive points)
    distances: list = []
    cum_dist_px = 0.0
    for i in range(1, len(params.points)):
        x1, y1 = params.points[i - 1]
        x2, y2 = params.points[i]
        d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        cum_dist_px += d
        entry = {
            "from_index": i - 1,
            "to_index": i,
            "distance_pixels": round(float(d), 3),
            "cumulative_pixels": round(float(cum_dist_px), 3),
        }
        if has_physical:
            entry["distance_physical"] = round(float(d * params.pixel_size), 4)
            entry["cumulative_physical"] = round(float(cum_dist_px * params.pixel_size), 4)
        distances.append(entry)

    result: dict = {
        "points": samples,
        "distances": distances,
        "count": len(samples),
    }
    if has_physical:
        result["pixel_size"] = params.pixel_size
        result["pixel_unit"] = params.pixel_unit
        result["total_length_physical"] = round(float(cum_dist_px * params.pixel_size), 4)
    result["total_length_pixels"] = round(float(cum_dist_px), 3)
    return result


# ---------------------------------------------------------------------------
# Line Profile
# ---------------------------------------------------------------------------

@router.post("/images/{image_id}/line-profile")
async def line_profile(image_id: str, params: LineProfileParams):
    """Compute intensity profile along a simple two-point line (legacy)."""
    poly_params = PolylineProfileParams(
        points=[[params.x1, params.y1], [params.x2, params.y2]],
        line_width=params.line_width,
    )
    return await polyline_profile(image_id, poly_params)


@router.post("/images/{image_id}/polyline-profile")
async def polyline_profile(image_id: str, params: PolylineProfileParams):
    """
    Compute intensity profile along a multi-segment polyline.

    Supports adjustable line_width: when >1, pixels are averaged across a
    band perpendicular to the line direction (Fiji-style "fat line" averaging).

    Returns per-channel profiles, cumulative distance axis, and segment info.
    """
    arr = store.load_display_array(image_id)
    points = params.points
    line_width = max(1, min(20, params.line_width))

    if len(points) < 2:
        raise HTTPException(400, "Need at least 2 points")

    # Build coordinate arrays for each segment, then concatenate
    all_x, all_y, all_dist = [], [], []
    cum_dist = 0.0
    segment_lengths = []

    for seg_idx in range(len(points) - 1):
        x1, y1 = points[seg_idx]
        x2, y2 = points[seg_idx + 1]
        seg_len = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if seg_len < 1:
            segment_lengths.append(0)
            continue

        n_samples = max(2, int(seg_len))
        xs = np.linspace(x1, x2, n_samples, endpoint=(seg_idx == len(points) - 2))
        ys = np.linspace(y1, y2, n_samples, endpoint=(seg_idx == len(points) - 2))
        ds = np.linspace(cum_dist, cum_dist + seg_len, n_samples, endpoint=(seg_idx == len(points) - 2))

        all_x.append(xs)
        all_y.append(ys)
        all_dist.append(ds)
        cum_dist += seg_len
        segment_lengths.append(round(seg_len, 2))

    if not all_x:
        raise HTTPException(400, "Zero-length polyline")

    x_coords = np.concatenate(all_x)
    y_coords = np.concatenate(all_y)
    distances = np.concatenate(all_dist)
    total_length = round(float(cum_dist), 2)

    # --- Extract profiles, optionally with line width averaging ---
    def _extract_profile(channel_2d: np.ndarray) -> list:
        """Sample a single 2D channel along the polyline with width averaging."""
        ch = channel_2d.astype(np.float64)

        if line_width <= 1:
            vals = ndimage.map_coordinates(ch, [y_coords, x_coords], order=1)
            return vals.tolist()

        # "Fat line" averaging: sample across perpendicular offsets
        # Compute per-point normal direction from local tangent
        dx = np.gradient(x_coords)
        dy = np.gradient(y_coords)
        mag = np.sqrt(dx ** 2 + dy ** 2)
        mag[mag == 0] = 1  # avoid division by zero
        nx = -dy / mag     # perpendicular normal x
        ny =  dx / mag     # perpendicular normal y

        half_w = (line_width - 1) / 2.0
        offsets = np.linspace(-half_w, half_w, line_width)
        accumulated = np.zeros(len(x_coords), dtype=np.float64)

        for off in offsets:
            ox = x_coords + nx * off
            oy = y_coords + ny * off
            accumulated += ndimage.map_coordinates(ch, [oy, ox], order=1, mode='nearest')

        return (accumulated / line_width).tolist()

    profiles: dict = {}
    if len(arr.shape) == 2:
        profiles["gray"] = _extract_profile(arr)
    else:
        for i, ch_name in enumerate(["red", "green", "blue"][:arr.shape[2]]):
            profiles[ch_name] = _extract_profile(arr[:, :, i])
        lum = np.mean(arr[:, :, :3].astype(np.float64), axis=2)
        profiles["luminance"] = _extract_profile(lum)

    return {
        "profiles": profiles,
        "length_pixels": total_length,
        "distances": distances.tolist(),        # cumulative distance axis
        "coordinates": {
            "x": x_coords.tolist(),
            "y": y_coords.tolist(),
        },
        "segments": segment_lengths,
        "line_width": line_width,
        "num_points": len(points),
    }


# ---------------------------------------------------------------------------
# Distance Measurement
# ---------------------------------------------------------------------------

@router.post("/images/{image_id}/measure")
async def measure_distance(image_id: str, params: MeasurementParams):
    """Measure Euclidean distance between two points."""
    dx = params.x2 - params.x1
    dy = params.y2 - params.y1
    dist_px = math.sqrt(dx ** 2 + dy ** 2)
    return {
        "distance_pixels": round(dist_px, 2),
        "distance_scaled": round(dist_px * params.pixel_size, 4),
        "unit": params.pixel_unit,
        "angle_degrees": round(math.degrees(math.atan2(dy, dx)), 2),
        "dx": dx,
        "dy": dy,
    }


# ---------------------------------------------------------------------------
# Thresholding / Segmentation
# ---------------------------------------------------------------------------

@router.post("/images/{image_id}/threshold")
async def threshold_image(image_id: str, params: ThresholdParams):
    """Apply thresholding and label connected components."""
    arr = store.load_display_array(image_id)
    gray = np.mean(arr[:, :, :3], axis=2).astype(np.uint8) if len(arr.shape) == 3 else arr.copy()

    if params.method == "otsu":
        thresh_val = filters.threshold_otsu(gray)
        binary = gray > thresh_val
    elif params.method == "manual":
        thresh_val = params.value if params.value is not None else 128
        binary = gray > thresh_val
    elif params.method == "adaptive":
        thresh_val = filters.threshold_local(gray, block_size=params.block_size)
        binary = gray > thresh_val
        thresh_val = float(np.mean(thresh_val))
    else:
        raise HTTPException(400, f"Unknown method: {params.method}")

    labeled = measure.label(binary)
    regions = measure.regionprops(labeled)
    region_data = [
        {
            "label": int(r.label), "area": int(r.area),
            "centroid": [round(r.centroid[0], 1), round(r.centroid[1], 1)],
            "bbox": list(r.bbox),
            "perimeter": round(float(r.perimeter), 2),
            "eccentricity": round(float(r.eccentricity), 4),
        }
        for r in regions[:100]
    ]

    # Green overlay for thresholded pixels
    overlay = np.zeros((*binary.shape, 4), dtype=np.uint8)
    overlay[binary, 1] = 255
    overlay[binary, 2] = 100
    overlay[binary, 3] = 140

    return {
        "threshold_value": round(float(thresh_val), 2),
        "method": params.method,
        "num_regions": len(regions),
        "regions": region_data,
        "overlay": array_to_base64(overlay),
        "binary": array_to_base64((binary.astype(np.uint8) * 255)),
    }


# ---------------------------------------------------------------------------
# Edge Detection
# ---------------------------------------------------------------------------

@router.get("/images/{image_id}/edges")
async def detect_edges(image_id: str, method: str = "canny", sigma: float = 1.0):
    """Detect edges using Canny, Sobel, or Laplacian."""
    arr = store.load_display_array(image_id)
    gray = (np.mean(arr[:, :, :3], axis=2) if len(arr.shape) == 3 else arr.copy()).astype(np.float64) / 255.0

    if method == "canny":
        edges = filters.farid(gray)
        try:
            from skimage.feature import canny
            edges = canny(gray, sigma=sigma).astype(np.float64)
        except ImportError:
            pass
    elif method == "sobel":
        edges = filters.sobel(gray)
    elif method == "laplacian":
        edges = np.abs(filters.laplace(gray))
    else:
        raise HTTPException(400, f"Unknown method: {method}")

    edges = (edges / edges.max() * 255).astype(np.uint8) if edges.max() > 0 else edges.astype(np.uint8)
    return {"image": array_to_base64(edges), "method": method}


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

@router.get("/images/{image_id}/filter")
async def apply_filter(image_id: str, filter_type: str = "gaussian", size: int = 3):
    """Apply spatial filters (gaussian, median, sharpen)."""
    arr = store.load_display_array(image_id).copy()

    def _apply_per_channel(func):
        if len(arr.shape) == 3:
            for c in range(arr.shape[2]):
                arr[:, :, c] = func(arr[:, :, c])
        else:
            return func(arr)
        return arr

    if filter_type == "gaussian":
        result = _apply_per_channel(
            lambda ch: ndimage.gaussian_filter(ch.astype(float), sigma=size / 2).astype(np.uint8)
        )
    elif filter_type == "median":
        result = _apply_per_channel(lambda ch: ndimage.median_filter(ch, size=size))
    elif filter_type == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        result = _apply_per_channel(
            lambda ch: np.clip(ndimage.convolve(ch.astype(float), kernel), 0, 255).astype(np.uint8)
        )
    else:
        raise HTTPException(400, f"Unknown filter: {filter_type}")

    return {"image": array_to_base64(result)}


# ---------------------------------------------------------------------------
# Image Transformations
# ---------------------------------------------------------------------------

@router.post("/images/{image_id}/rotate")
async def rotate_image(image_id: str, params: RotateParams):
    """Rotate an image and create a new copy in the store."""
    arr = store.load_array(image_id)
    entry = store.get(image_id)

    # Rotate using scipy
    rotated = ndimage.rotate(arr, params.angle, reshape=params.expand, order=1)

    # Create display version
    from ..image_processing import normalize_to_uint8
    display = normalize_to_uint8(rotated)

    new_id = uuid.uuid4().hex[:8]
    name = f"{entry['name']}_rot{int(params.angle)}"

    metadata = {
        "filename": name,
        "width": rotated.shape[1],
        "height": rotated.shape[0],
        "mode": entry["metadata"].get("mode", "RGB"),
        "channels": entry["metadata"].get("channels", 3),
        "bands": entry["metadata"].get("bands", ["R","G","B"]),
        "dtype": str(rotated.dtype),
        "bit_depth": int(rotated.dtype.itemsize * 8),
        "size_bytes": rotated.nbytes,
        "min_value": int(rotated.min()),
        "max_value": int(rotated.max()),
        "mean_value": round(float(rotated.mean()), 2),
        "percentile_1": round(float(np.percentile(rotated.astype(np.float64), 1)), 2),
        "percentile_99": round(float(np.percentile(rotated.astype(np.float64), 99)), 2),
    }

    store.put(new_id, {
        "path": entry["path"],
        "name": name,
        "metadata": metadata,
        "array": rotated,
        "display_array": display,
    })

    return {"image_id": new_id, "metadata": metadata}


@router.post("/images/{image_id}/crop")
async def crop_image(image_id: str, params: CropParams):
    """Crop a region and create a new image."""
    arr = store.load_array(image_id)
    entry = store.get(image_id)
    h, w = arr.shape[:2]

    y1 = max(0, min(params.y, h))
    y2 = max(0, min(params.y + params.height, h))
    x1 = max(0, min(params.x, w))
    x2 = max(0, min(params.x + params.width, w))

    if y2 <= y1 or x2 <= x1:
        raise HTTPException(400, "Invalid crop region")

    cropped = arr[y1:y2, x1:x2].copy()

    from ..image_processing import normalize_to_uint8
    display = normalize_to_uint8(cropped)

    new_id = uuid.uuid4().hex[:8]
    name = f"{entry['name']}_crop"

    metadata = {
        "filename": name,
        "width": cropped.shape[1],
        "height": cropped.shape[0],
        "mode": entry["metadata"].get("mode", "RGB"),
        "channels": entry["metadata"].get("channels", 3),
        "bands": entry["metadata"].get("bands", ["R","G","B"]),
        "dtype": str(cropped.dtype),
        "bit_depth": int(cropped.dtype.itemsize * 8),
        "size_bytes": cropped.nbytes,
        "min_value": int(cropped.min()),
        "max_value": int(cropped.max()),
        "mean_value": round(float(cropped.mean()), 2),
        "percentile_1": round(float(np.percentile(cropped.astype(np.float64), 1)), 2),
        "percentile_99": round(float(np.percentile(cropped.astype(np.float64), 99)), 2),
    }

    store.put(new_id, {
        "path": entry["path"],
        "name": name,
        "metadata": metadata,
        "array": cropped,
        "display_array": display,
    })

    return {"image_id": new_id, "metadata": metadata}


@router.post("/images/{image_id}/rotate-inplace")
async def rotate_image_inplace(image_id: str, params: RotateParams):
    """Rotate an image in-place (modifies the existing image, no copy)."""
    arr = store.load_array(image_id)
    entry = store.get(image_id)

    rotated = ndimage.rotate(arr, params.angle, reshape=params.expand, order=1)

    from ..image_processing import normalize_to_uint8
    display = normalize_to_uint8(rotated)

    # Update the existing entry in-place
    entry["array"] = rotated
    entry["display_array"] = display
    entry["metadata"]["width"] = rotated.shape[1]
    entry["metadata"]["height"] = rotated.shape[0]
    entry["metadata"]["min_value"] = int(rotated.min())
    entry["metadata"]["max_value"] = int(rotated.max())
    entry["metadata"]["mean_value"] = round(float(rotated.mean()), 2)
    pyramid.clear_image(image_id)

    return {"image_id": image_id, "metadata": entry["metadata"]}


@router.post("/images/{image_id}/crop-save")
async def crop_and_save(image_id: str, params: CropParams):
    """Crop a region, create a new image, and save it to disk next to the original."""
    from PIL import Image as PILImage
    import os
    from pathlib import Path

    arr = store.load_array(image_id)
    entry = store.get(image_id)
    h, w = arr.shape[:2]

    y1 = max(0, min(params.y, h))
    y2 = max(0, min(params.y + params.height, h))
    x1 = max(0, min(params.x, w))
    x2 = max(0, min(params.x + params.width, w))

    if y2 <= y1 or x2 <= x1:
        raise HTTPException(400, "Invalid crop region")

    cropped = arr[y1:y2, x1:x2].copy()

    from ..image_processing import normalize_to_uint8
    display = normalize_to_uint8(cropped)

    # Determine save path: same folder as original
    orig_path = entry.get("path", "")
    if orig_path and os.path.exists(os.path.dirname(orig_path)):
        parent = Path(orig_path).parent
        stem = Path(orig_path).stem
        ext = Path(orig_path).suffix or ".png"
        save_name = f"{stem}_crop{ext}"
        save_path = parent / save_name
        counter = 1
        while save_path.exists():
            save_name = f"{stem}_crop_{counter}{ext}"
            save_path = parent / save_name
            counter += 1
    else:
        save_path = store.UPLOAD_DIR / f"crop_{uuid.uuid4().hex[:8]}.png"
        save_name = save_path.name

    # Save to disk — use TIFF for non-uint8 to preserve bit depth
    if cropped.dtype != np.uint8:
        save_path = save_path.with_suffix('.tif')
        save_name = save_path.name
    pil_img = PILImage.fromarray(cropped)
    pil_img.save(str(save_path))

    new_id = uuid.uuid4().hex[:8]
    metadata = {
        "filename": save_name,
        "width": cropped.shape[1],
        "height": cropped.shape[0],
        "mode": entry["metadata"].get("mode", "RGB"),
        "channels": entry["metadata"].get("channels", 3),
        "bands": entry["metadata"].get("bands", ["R", "G", "B"]),
        "dtype": str(cropped.dtype),
        "bit_depth": int(cropped.dtype.itemsize * 8),
        "size_bytes": cropped.nbytes,
        "min_value": int(cropped.min()),
        "max_value": int(cropped.max()),
        "mean_value": round(float(cropped.mean()), 2),
        "percentile_1": round(float(np.percentile(cropped.astype(np.float64), 1)), 2),
        "percentile_99": round(float(np.percentile(cropped.astype(np.float64), 99)), 2),
    }

    store.put(new_id, {
        "path": str(save_path),
        "name": save_name,
        "metadata": metadata,
        "array": cropped,
        "display_array": display,
    })

    return {"image_id": new_id, "metadata": metadata, "saved_path": str(save_path)}


@router.post("/images/{image_id}/crop-angled")
async def crop_angled(image_id: str, params: AngledCropParams):
    """
    Straighten & crop using 4 corner points.

    Simple, robust approach (like Fiji):
    1. Compute the rotation angle from the top edge (corner 1→2)
    2. Rotate the ENTIRE image by -angle to make the selection axis-aligned
    3. Transform the 4 corner coordinates into rotated image space
    4. Crop the axis-aligned bounding box from the rotated image
    """
    from PIL import Image as PILImage
    import os
    from pathlib import Path

    if len(params.corners) != 4:
        raise HTTPException(400, "Exactly 4 corner points required")

    arr = store.load_array(image_id)
    entry = store.get(image_id)

    pts = np.array(params.corners, dtype=np.float64)

    # Use corners in the order the user clicked them:
    # points[0]→points[1] defines the "top edge" (the direction to straighten)
    # points[1]→points[2] defines the "right edge"
    # points[2]→points[3] defines the "bottom edge"
    # points[3]→points[0] defines the "left edge"
    p0, p1, p2, p3 = pts[0], pts[1], pts[2], pts[3]

    # Step 1: Compute angle from the first edge (points 1→2)
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    angle_rad = math.atan2(dy, dx)
    angle_deg = round(math.degrees(angle_rad), 2)

    # Step 2: Rotate the entire image to straighten the selected edge.
    #
    # scipy.ndimage.rotate(arr, θ) forward-maps input points via:
    #   out_row = cos(θ)·in_row − sin(θ)·in_col   (relative to center)
    #   out_col = sin(θ)·in_row + cos(θ)·in_col
    #
    # For the edge vector (dx, dy) = (Δcol, Δrow), the output Δrow becomes:
    #   cos(θ)·dy − sin(θ)·dx
    # Setting this to zero (horizontal) gives θ = atan2(dy, dx) = angle_rad.
    # So we pass angle_deg directly (not negated).
    rotated = ndimage.rotate(arr, angle_deg, reshape=True, order=1, cval=0)

    # Step 3: Transform the 4 corners into the rotated coordinate system
    h_orig, w_orig = arr.shape[:2]
    h_rot, w_rot = rotated.shape[:2]
    cx_orig = w_orig / 2.0
    cy_orig = h_orig / 2.0
    cx_rot = w_rot / 2.0
    cy_rot = h_rot / 2.0

    # Forward map in (x=col, y=row) for scipy angle θ:
    #   out_x = cos(θ)·px + sin(θ)·py
    #   out_y = −sin(θ)·px + cos(θ)·py
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    def transform_point(x, y):
        """Transform a point from original image coords to rotated image coords."""
        px = x - cx_orig
        py = y - cy_orig
        rx =  px * cos_a + py * sin_a
        ry = -px * sin_a + py * cos_a
        return rx + cx_rot, ry + cy_rot

    corners_rot = np.array([
        transform_point(p0[0], p0[1]),
        transform_point(p1[0], p1[1]),
        transform_point(p2[0], p2[1]),
        transform_point(p3[0], p3[1]),
    ])

    # Step 4: Crop the axis-aligned bounding box of the transformed corners
    x_min = max(0, int(math.floor(corners_rot[:, 0].min())))
    x_max = min(w_rot, int(math.ceil(corners_rot[:, 0].max())))
    y_min = max(0, int(math.floor(corners_rot[:, 1].min())))
    y_max = min(h_rot, int(math.ceil(corners_rot[:, 1].max())))

    if x_max <= x_min or y_max <= y_min:
        raise HTTPException(400, "Crop region too small after rotation")

    cropped = rotated[y_min:y_max, x_min:x_max].copy()

    from ..image_processing import normalize_to_uint8
    display = normalize_to_uint8(cropped)

    new_id = uuid.uuid4().hex[:8]
    base_name = entry['name']
    name = f"{base_name}_straightened"

    saved_path = None
    if params.save_to_disk:
        orig_path = entry.get("path", "")
        if orig_path and os.path.exists(os.path.dirname(orig_path)):
            parent = Path(orig_path).parent
            stem = Path(orig_path).stem
            ext = Path(orig_path).suffix or ".png"
            save_name = f"{stem}_straightened{ext}"
            save_path = parent / save_name
            counter = 1
            while save_path.exists():
                save_name = f"{stem}_straightened_{counter}{ext}"
                save_path = parent / save_name
                counter += 1
        else:
            save_path = store.UPLOAD_DIR / f"straightened_{new_id}.png"
            save_name = save_path.name

        if cropped.dtype != np.uint8:
            save_path = save_path.with_suffix('.tif')
            save_name = save_path.name
        pil_img = PILImage.fromarray(cropped)
        pil_img.save(str(save_path))
        saved_path = str(save_path)
        name = save_name

    metadata = {
        "filename": name,
        "width": cropped.shape[1],
        "height": cropped.shape[0],
        "mode": entry["metadata"].get("mode", "RGB"),
        "channels": entry["metadata"].get("channels", 3),
        "bands": entry["metadata"].get("bands", ["R", "G", "B"]),
        "dtype": str(cropped.dtype),
        "bit_depth": int(cropped.dtype.itemsize * 8),
        "size_bytes": cropped.nbytes,
        "min_value": int(cropped.min()),
        "max_value": int(cropped.max()),
        "mean_value": round(float(cropped.mean()), 2),
        "percentile_1": round(float(np.percentile(cropped.astype(np.float64), 1)), 2),
        "percentile_99": round(float(np.percentile(cropped.astype(np.float64), 99)), 2),
    }

    store.put(new_id, {
        "path": saved_path or entry["path"],
        "name": name,
        "metadata": metadata,
        "array": cropped,
        "display_array": display,
    })

    result = {"image_id": new_id, "metadata": metadata, "angle": angle_deg}
    if saved_path:
        result["saved_path"] = saved_path
    return result


# ---------------------------------------------------------------------------
# Line Profile Export (Excel)
# ---------------------------------------------------------------------------

@router.post("/export-profiles")
async def export_profiles(params: ProfileExportParams):
    """
    Export line profile data as a multi-sheet Excel file.

    Sheet 1 — Metadata: image info, adjustments, ROI details, scale bar
    Sheet 2+ — Profile Data: distance (physical if pixel size known), per-channel intensities
    """
    import os
    from datetime import datetime
    from pathlib import Path
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    if not params.entries:
        raise HTTPException(400, "No profile data to export")

    wb = Workbook()

    # --- Styling ---
    header_font = Font(name='Arial', bold=True, size=11, color='FFFFFF')
    header_fill = PatternFill('solid', fgColor='2B5797')
    subheader_font = Font(name='Arial', bold=True, size=10, color='2B5797')
    label_font = Font(name='Arial', bold=True, size=10)
    data_font = Font(name='Arial', size=10)
    thin_border = Border(
        bottom=Side(style='thin', color='D0D0D0')
    )

    has_physical = params.pixelSize and params.pixelSize > 0 and params.pixelSizeUnit
    px_size = params.pixelSize or 1.0
    px_unit = params.pixelSizeUnit or 'px'

    # =========================================================================
    #  Sheet 1: Metadata
    # =========================================================================
    ws_meta = wb.active
    ws_meta.title = "Metadata"
    ws_meta.sheet_properties.tabColor = '2B5797'

    def write_section(ws, row, title):
        ws.cell(row=row, column=1, value=title).font = subheader_font
        return row + 1

    def write_kv(ws, row, key, value):
        c1 = ws.cell(row=row, column=1, value=key)
        c1.font = label_font
        c2 = ws.cell(row=row, column=2, value=str(value) if value is not None else '')
        c2.font = data_font
        c1.border = thin_border
        c2.border = thin_border
        return row + 1

    r = 1
    # Header row
    for col_idx, txt in enumerate(['Parameter', 'Value'], 1):
        c = ws_meta.cell(row=r, column=col_idx, value=txt)
        c.font = header_font
        c.fill = header_fill
        c.alignment = Alignment(horizontal='center')
    r += 1

    # Export info
    r = write_section(ws_meta, r, '── Export Info ──')
    r = write_kv(ws_meta, r, 'Export Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    r = write_kv(ws_meta, r, 'Number of Profiles', len(params.entries))
    r += 1

    # Scale bar / pixel size
    r = write_section(ws_meta, r, '── Scale ──')
    if has_physical:
        r = write_kv(ws_meta, r, 'Pixel Size', f'{px_size} {px_unit}/px')
        r = write_kv(ws_meta, r, 'Distance Unit', px_unit)
    else:
        r = write_kv(ws_meta, r, 'Pixel Size', 'Not available')
        r = write_kv(ws_meta, r, 'Distance Unit', 'pixels')
    r += 1

    # Per-profile info
    for i, entry in enumerate(params.entries):
        r = write_section(ws_meta, r, f'── Profile {i+1} ──')
        r = write_kv(ws_meta, r, 'Image Name', entry.imageName)
        r = write_kv(ws_meta, r, 'Image ID', entry.imageId)

        # Image metadata from store
        stored = store.get(entry.imageId)
        if stored:
            m = stored.get('metadata', {})
            r = write_kv(ws_meta, r, 'Image Dimensions', f'{m.get("width", "?")} x {m.get("height", "?")}')
            r = write_kv(ws_meta, r, 'Bit Depth', m.get('bit_depth', '?'))
            r = write_kv(ws_meta, r, 'Dtype', m.get('dtype', '?'))
            if m.get('pixel_size_x'):
                r = write_kv(ws_meta, r, 'Image Pixel Size (from metadata)',
                             f'{m["pixel_size_x"]} {m.get("pixel_size_unit", "")}')

        r = write_kv(ws_meta, r, 'Line Width (px)', entry.lineWidth)
        length_display = (f'{entry.lengthPixels * px_size:.2f} {px_unit}'
                          if has_physical
                          else f'{entry.lengthPixels:.2f} px')
        r = write_kv(ws_meta, r, 'Total Length', length_display)
        r = write_kv(ws_meta, r, 'Waypoints', entry.numPoints)
        r = write_kv(ws_meta, r, 'Segments', len(entry.segments))
        segs_str = ', '.join(f'{s:.1f}' for s in entry.segments if s > 0)
        r = write_kv(ws_meta, r, 'Segment Lengths (px)', segs_str)
        if entry.polylinePoints:
            pts_str = ' → '.join(f'({p[0]:.0f},{p[1]:.0f})' for p in entry.polylinePoints)
            r = write_kv(ws_meta, r, 'Polyline Coordinates', pts_str)
        r = write_kv(ws_meta, r, 'Channels', ', '.join(entry.profiles.keys()))

        # Adjustments
        if params.adjustments and entry.imageId in params.adjustments:
            adj = params.adjustments[entry.imageId]
            r = write_kv(ws_meta, r, 'Contrast Min', adj.get('clMin', ''))
            r = write_kv(ws_meta, r, 'Contrast Max', adj.get('clMax', ''))
            r = write_kv(ws_meta, r, 'Gamma', adj.get('gamma', ''))
            r = write_kv(ws_meta, r, 'Colormap', adj.get('colormap', 'none'))
            r = write_kv(ws_meta, r, 'Inverted', adj.get('invert', False))
            r = write_kv(ws_meta, r, 'Interpolation', adj.get('interpolation', ''))
        r += 1

    ws_meta.column_dimensions['A'].width = 32
    ws_meta.column_dimensions['B'].width = 60

    # =========================================================================
    #  Sheet 2: Profile Data
    # =========================================================================
    ws_data = wb.create_sheet("Profile Data")
    ws_data.sheet_properties.tabColor = '4ECDC4'

    # Build headers
    headers = []
    if has_physical:
        headers.append(f'Distance ({px_unit})')
    else:
        headers.append('Distance (px)')

    multi = len(params.entries) > 1
    for entry in params.entries:
        prefix = (entry.imageName.replace('/', '_').replace('\\', '_') + ' — ') if multi else ''
        for ch in entry.profiles.keys():
            headers.append(f'{prefix}{ch}')

    # Write header row
    for col, h in enumerate(headers, 1):
        c = ws_data.cell(row=1, column=col, value=h)
        c.font = header_font
        c.fill = header_fill
        c.alignment = Alignment(horizontal='center')

    # Find max length
    max_len = max(len(e.distances) for e in params.entries)

    # Write data rows
    for i in range(max_len):
        row = i + 2
        # Distance from first entry (they should all share the same scale)
        if i < len(params.entries[0].distances):
            dist_px = params.entries[0].distances[i]
            dist_val = dist_px * px_size if has_physical else dist_px
        else:
            dist_val = ''
        ws_data.cell(row=row, column=1, value=round(dist_val, 4) if isinstance(dist_val, float) else dist_val).font = data_font

        col = 2
        for entry in params.entries:
            for ch_name, ch_data in entry.profiles.items():
                val = ch_data[i] if i < len(ch_data) else ''
                ws_data.cell(row=row, column=col, value=round(val, 4) if isinstance(val, float) else val).font = data_font
                col += 1

    # Auto-width for data columns
    ws_data.column_dimensions['A'].width = 18
    for col_idx in range(2, len(headers) + 1):
        from openpyxl.utils import get_column_letter
        ws_data.column_dimensions[get_column_letter(col_idx)].width = 14

    # Freeze top row
    ws_data.freeze_panes = 'A2'

    # =========================================================================
    #  Determine save path
    # =========================================================================
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Try to save next to the original image
    save_dir = None
    save_dir_reason = ""
    if params.saveDirectory and os.path.isdir(params.saveDirectory):
        save_dir = Path(params.saveDirectory)
        save_dir_reason = "user-specified"
    else:
        # Use the first image's directory
        first_entry = store.get(params.entries[0].imageId)
        if first_entry:
            # Prefer LIF source file's directory if present (path points to temp tif)
            lif_src = (first_entry.get('metadata') or {}).get('lif_source')
            if lif_src and os.path.isfile(lif_src):
                save_dir = Path(lif_src).parent
                save_dir_reason = "lif source parent"
            else:
                orig_path = first_entry.get('path', '')
                if orig_path:
                    parent = os.path.dirname(orig_path)
                    # Skip if it's the temp upload dir
                    try:
                        is_upload = os.path.samefile(parent, str(store.UPLOAD_DIR)) if parent and os.path.isdir(parent) else False
                    except Exception:
                        is_upload = False
                    if parent and os.path.isdir(parent) and not is_upload:
                        save_dir = Path(parent)
                        save_dir_reason = "image parent"

    if not save_dir:
        save_dir = Path(store.UPLOAD_DIR)
        save_dir_reason = "fallback: upload dir"

    # Ensure writable
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise HTTPException(500, f"Cannot create save directory {save_dir}: {e}")
    if not os.access(str(save_dir), os.W_OK):
        raise HTTPException(500, f"Save directory not writable: {save_dir}")

    # Build descriptive filename
    img_name = params.entries[0].imageName.replace('/', '_').replace('\\', '_')
    # Truncate long names
    if len(img_name) > 40:
        img_name = img_name[:40]
    base_name = f'{img_name}_line_profiles_{timestamp}'
    save_path = save_dir / f'{base_name}.xlsx'

    # Auto-increment if file exists
    counter = 1
    while save_path.exists():
        save_path = save_dir / f'{base_name}_{counter}.xlsx'
        counter += 1

    try:
        wb.save(str(save_path))
    except Exception as e:
        raise HTTPException(500, f"Failed to write Excel file to {save_path}: {e}")

    if not save_path.exists():
        raise HTTPException(500, f"Excel file was not created at {save_path}")

    return {
        "saved_path": str(save_path),
        "filename": save_path.name,
        "save_dir": str(save_dir),
        "save_dir_reason": save_dir_reason,
        "num_profiles": len(params.entries),
        "num_data_points": max_len,
        "has_physical_units": has_physical,
    }


# ---------------------------------------------------------------------------
# ROI Measurement Export (Excel)
# ---------------------------------------------------------------------------

def _resolve_export_dir(first_image_id: str, requested: str | None):
    """Mirror of the save-dir resolution used for profile export."""
    import os
    from pathlib import Path

    save_dir = None
    reason = ""
    if requested and os.path.isdir(requested):
        save_dir = Path(requested)
        reason = "user-specified"
    else:
        first_entry = store.get(first_image_id)
        if first_entry:
            lif_src = (first_entry.get('metadata') or {}).get('lif_source')
            if lif_src and os.path.isfile(lif_src):
                save_dir = Path(lif_src).parent
                reason = "lif source parent"
            else:
                orig_path = first_entry.get('path', '')
                if orig_path:
                    parent = os.path.dirname(orig_path)
                    try:
                        is_upload = os.path.samefile(parent, str(store.UPLOAD_DIR)) if parent and os.path.isdir(parent) else False
                    except Exception:
                        is_upload = False
                    if parent and os.path.isdir(parent) and not is_upload:
                        save_dir = Path(parent)
                        reason = "image parent"
    if not save_dir:
        save_dir = Path(store.UPLOAD_DIR)
        reason = "fallback: upload dir"
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise HTTPException(500, f"Cannot create save directory {save_dir}: {e}")
    if not os.access(str(save_dir), os.W_OK):
        raise HTTPException(500, f"Save directory not writable: {save_dir}")
    return save_dir, reason


@router.post("/export-roi-measurements")
async def export_roi_measurements(params: ROIMeasurementExportParams):
    """
    Export completed ROI measurements (rect / ellipse / polygon) as an Excel file.

    Sheet 1 — Metadata: export info, scale, per-image details, adjustments
    Sheet 2 — Measurements: one row per ROI with all numeric statistics
    Sheet 3 — Coordinates: full rect/ellipse bbox or polygon vertices
    """
    import os
    from datetime import datetime
    from pathlib import Path
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    if not params.entries:
        raise HTTPException(400, "No ROI measurements to export")

    wb = Workbook()

    # --- Styling ---
    header_font = Font(name='Arial', bold=True, size=11, color='FFFFFF')
    header_fill = PatternFill('solid', fgColor='2B5797')
    subheader_font = Font(name='Arial', bold=True, size=10, color='2B5797')
    label_font = Font(name='Arial', bold=True, size=10)
    data_font = Font(name='Arial', size=10)
    thin_border = Border(bottom=Side(style='thin', color='D0D0D0'))

    has_physical = params.pixelSize and params.pixelSize > 0 and params.pixelSizeUnit
    px_size = params.pixelSize or 1.0
    px_unit = params.pixelSizeUnit or 'px'

    # =========================================================================
    #  Sheet 1: Metadata
    # =========================================================================
    ws_meta = wb.active
    ws_meta.title = "Metadata"
    ws_meta.sheet_properties.tabColor = '2B5797'

    def write_section(ws, row, title):
        ws.cell(row=row, column=1, value=title).font = subheader_font
        return row + 1

    def write_kv(ws, row, key, value):
        c1 = ws.cell(row=row, column=1, value=key); c1.font = label_font
        c2 = ws.cell(row=row, column=2, value=str(value) if value is not None else '')
        c2.font = data_font
        c1.border = thin_border; c2.border = thin_border
        return row + 1

    r = 1
    for col_idx, txt in enumerate(['Parameter', 'Value'], 1):
        c = ws_meta.cell(row=r, column=col_idx, value=txt)
        c.font = header_font; c.fill = header_fill
        c.alignment = Alignment(horizontal='center')
    r += 1

    r = write_section(ws_meta, r, '── Export Info ──')
    r = write_kv(ws_meta, r, 'Export Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    r = write_kv(ws_meta, r, 'Number of ROIs', len(params.entries))
    shape_counts: dict = {}
    for e in params.entries:
        shape_counts[e.shape] = shape_counts.get(e.shape, 0) + 1
    r = write_kv(ws_meta, r, 'Shape breakdown',
                 ', '.join(f'{k}: {v}' for k, v in shape_counts.items()))
    r += 1

    r = write_section(ws_meta, r, '── Scale ──')
    if has_physical:
        r = write_kv(ws_meta, r, 'Pixel Size', f'{px_size} {px_unit}/px')
        r = write_kv(ws_meta, r, 'Area Unit', f'{px_unit}²')
    else:
        r = write_kv(ws_meta, r, 'Pixel Size', 'Not available (pixels only)')
    r += 1

    # Per-image metadata
    seen_images = set()
    for e in params.entries:
        if e.imageId in seen_images:
            continue
        seen_images.add(e.imageId)
        r = write_section(ws_meta, r, f'── Image: {e.imageName} ──')
        r = write_kv(ws_meta, r, 'Image ID', e.imageId)
        stored = store.get(e.imageId)
        if stored:
            m = stored.get('metadata', {})
            r = write_kv(ws_meta, r, 'Dimensions', f'{m.get("width", "?")} x {m.get("height", "?")}')
            r = write_kv(ws_meta, r, 'Bit Depth', m.get('bit_depth', '?'))
            r = write_kv(ws_meta, r, 'Dtype', m.get('dtype', '?'))
            if m.get('pixel_size_x'):
                r = write_kv(ws_meta, r, 'Image Pixel Size (metadata)',
                             f'{m["pixel_size_x"]} {m.get("pixel_size_unit", "")}')
        if params.adjustments and e.imageId in params.adjustments:
            adj = params.adjustments[e.imageId] or {}
            r = write_kv(ws_meta, r, 'Contrast Min', adj.get('clMin', ''))
            r = write_kv(ws_meta, r, 'Contrast Max', adj.get('clMax', ''))
            r = write_kv(ws_meta, r, 'Gamma', adj.get('gamma', ''))
            r = write_kv(ws_meta, r, 'Colormap', adj.get('colormap', 'none'))
            r = write_kv(ws_meta, r, 'Inverted', adj.get('invert', False))
        r += 1

    ws_meta.column_dimensions['A'].width = 32
    ws_meta.column_dimensions['B'].width = 60

    # =========================================================================
    #  Sheet 2: Measurements
    # =========================================================================
    ws_data = wb.create_sheet("Measurements")
    ws_data.sheet_properties.tabColor = '4ECDC4'

    headers = [
        '#', 'Label', 'Image', 'Shape',
        'Area (px)',
    ]
    if has_physical:
        headers.append(f'Area ({px_unit}²)')
    headers += ['Min', 'Max', 'Mean', 'Std', 'Median', 'IntDen (raw)']
    if has_physical:
        headers.append(f'IntDen ({px_unit}²)')
    # Full per-channel set: mean, std, min, max, median, IntDen for R/G/B
    for ch in ('R', 'G', 'B'):
        headers += [f'{ch} Mean', f'{ch} Std', f'{ch} Min', f'{ch} Max', f'{ch} Median', f'{ch} IntDen']

    for col, h in enumerate(headers, 1):
        c = ws_data.cell(row=1, column=col, value=h)
        c.font = header_font; c.fill = header_fill
        c.alignment = Alignment(horizontal='center')

    def g(d, k):
        v = d.get(k) if isinstance(d, dict) else None
        return v if v is not None else ''

    for i, e in enumerate(params.entries):
        row = i + 2
        s = e.stats or {}
        vals: list = [
            i + 1, e.label or f'ROI {i+1}', e.imageName, e.shape,
            g(s, 'area_pixels'),
        ]
        if has_physical:
            vals.append(g(s, 'area_physical'))
        vals += [
            g(s, 'min'), g(s, 'max'), g(s, 'mean'), g(s, 'std'), g(s, 'median'),
            g(s, 'integrated_density'),
        ]
        if has_physical:
            vals.append(g(s, 'integrated_density_physical'))
        for ch in ('red', 'green', 'blue'):
            vals += [
                g(s, f'{ch}_mean'), g(s, f'{ch}_std'),
                g(s, f'{ch}_min'), g(s, f'{ch}_max'),
                g(s, f'{ch}_median'), g(s, f'{ch}_integrated_density'),
            ]
        for col, v in enumerate(vals, 1):
            c = ws_data.cell(row=row, column=col, value=v)
            c.font = data_font

    ws_data.column_dimensions['A'].width = 4
    ws_data.column_dimensions['B'].width = 16
    ws_data.column_dimensions['C'].width = 24
    ws_data.column_dimensions['D'].width = 10
    for col_idx in range(5, len(headers) + 1):
        ws_data.column_dimensions[get_column_letter(col_idx)].width = 14
    ws_data.freeze_panes = 'A2'

    # =========================================================================
    #  Sheet 3: Coordinates (bbox for rect/ellipse, vertices for polygon)
    # =========================================================================
    ws_coords = wb.create_sheet("Coordinates")
    ws_coords.sheet_properties.tabColor = 'FFB347'
    coord_headers = ['#', 'Label', 'Image', 'Shape', 'X', 'Y', 'Width', 'Height', 'Polygon Vertices']
    for col, h in enumerate(coord_headers, 1):
        c = ws_coords.cell(row=1, column=col, value=h)
        c.font = header_font; c.fill = header_fill
        c.alignment = Alignment(horizontal='center')
    for i, e in enumerate(params.entries):
        row = i + 2
        rect = e.rect or {}
        pts_str = ''
        if e.points:
            pts_str = ' → '.join(f'({p[0]:.0f},{p[1]:.0f})' for p in e.points)
        vals = [
            i + 1, e.label or f'ROI {i+1}', e.imageName, e.shape,
            rect.get('x', ''), rect.get('y', ''),
            rect.get('width', ''), rect.get('height', ''),
            pts_str,
        ]
        for col, v in enumerate(vals, 1):
            ws_coords.cell(row=row, column=col, value=v).font = data_font
    for col_idx, w in enumerate([4, 16, 24, 10, 10, 10, 10, 10, 60], 1):
        ws_coords.column_dimensions[get_column_letter(col_idx)].width = w
    ws_coords.freeze_panes = 'A2'

    # =========================================================================
    #  Save
    # =========================================================================
    save_dir, reason = _resolve_export_dir(params.entries[0].imageId, params.saveDirectory)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    img_name = params.entries[0].imageName.replace('/', '_').replace('\\', '_')
    if len(img_name) > 40:
        img_name = img_name[:40]
    base = f'{img_name}_roi_measurements_{timestamp}'
    save_path = save_dir / f'{base}.xlsx'
    counter = 1
    while save_path.exists():
        save_path = save_dir / f'{base}_{counter}.xlsx'
        counter += 1
    try:
        wb.save(str(save_path))
    except Exception as e:
        raise HTTPException(500, f"Failed to write Excel file to {save_path}: {e}")
    if not save_path.exists():
        raise HTTPException(500, f"Excel file was not created at {save_path}")

    return {
        "saved_path": str(save_path),
        "filename": save_path.name,
        "save_dir": str(save_dir),
        "save_dir_reason": reason,
        "num_rois": len(params.entries),
        "has_physical_units": bool(has_physical),
    }


# ---------------------------------------------------------------------------
# Point Set Export (Excel)
# ---------------------------------------------------------------------------

@router.post("/export-point-sets")
async def export_point_sets(params: PointSetExportParams):
    """
    Export saved multi-point sets as an Excel file.

    Sheet 1 — Metadata: export info, scale, per-image details, adjustments
    Sheet 2 — Summary: one row per set with count, total length, coordinates
    Sheet 3 — Points: one row per individual point with position, intensity, cumulative distance
    """
    from datetime import datetime
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    if not params.entries:
        raise HTTPException(400, "No point sets to export")

    wb = Workbook()

    header_font = Font(name='Arial', bold=True, size=11, color='FFFFFF')
    header_fill = PatternFill('solid', fgColor='2B5797')
    subheader_font = Font(name='Arial', bold=True, size=10, color='2B5797')
    label_font = Font(name='Arial', bold=True, size=10)
    data_font = Font(name='Arial', size=10)
    thin_border = Border(bottom=Side(style='thin', color='D0D0D0'))

    has_physical = params.pixelSize and params.pixelSize > 0 and params.pixelSizeUnit
    px_size = params.pixelSize or 1.0
    px_unit = params.pixelSizeUnit or 'px'

    # =========================================================================
    #  Sheet 1: Metadata
    # =========================================================================
    ws_meta = wb.active
    ws_meta.title = "Metadata"
    ws_meta.sheet_properties.tabColor = '2B5797'

    def write_section(ws, row, title):
        ws.cell(row=row, column=1, value=title).font = subheader_font
        return row + 1

    def write_kv(ws, row, key, value):
        c1 = ws.cell(row=row, column=1, value=key); c1.font = label_font
        c2 = ws.cell(row=row, column=2, value=str(value) if value is not None else '')
        c2.font = data_font
        c1.border = thin_border; c2.border = thin_border
        return row + 1

    r = 1
    for col_idx, txt in enumerate(['Parameter', 'Value'], 1):
        c = ws_meta.cell(row=r, column=col_idx, value=txt)
        c.font = header_font; c.fill = header_fill
        c.alignment = Alignment(horizontal='center')
    r += 1

    total_points = sum(len(e.points) for e in params.entries)

    r = write_section(ws_meta, r, '── Export Info ──')
    r = write_kv(ws_meta, r, 'Export Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    r = write_kv(ws_meta, r, 'Number of Sets', len(params.entries))
    r = write_kv(ws_meta, r, 'Total Points', total_points)
    r += 1

    r = write_section(ws_meta, r, '── Scale ──')
    if has_physical:
        r = write_kv(ws_meta, r, 'Pixel Size', f'{px_size} {px_unit}/px')
        r = write_kv(ws_meta, r, 'Distance Unit', px_unit)
    else:
        r = write_kv(ws_meta, r, 'Pixel Size', 'Not available (pixels only)')
    r += 1

    seen_images = set()
    for e in params.entries:
        if e.imageId in seen_images:
            continue
        seen_images.add(e.imageId)
        r = write_section(ws_meta, r, f'── Image: {e.imageName} ──')
        r = write_kv(ws_meta, r, 'Image ID', e.imageId)
        stored = store.get(e.imageId)
        if stored:
            m = stored.get('metadata', {})
            r = write_kv(ws_meta, r, 'Dimensions', f'{m.get("width", "?")} x {m.get("height", "?")}')
            r = write_kv(ws_meta, r, 'Bit Depth', m.get('bit_depth', '?'))
            r = write_kv(ws_meta, r, 'Dtype', m.get('dtype', '?'))
            if m.get('pixel_size_x'):
                r = write_kv(ws_meta, r, 'Image Pixel Size (metadata)',
                             f'{m["pixel_size_x"]} {m.get("pixel_size_unit", "")}')
        if params.adjustments and e.imageId in params.adjustments:
            adj = params.adjustments[e.imageId] or {}
            r = write_kv(ws_meta, r, 'Contrast Min', adj.get('clMin', ''))
            r = write_kv(ws_meta, r, 'Contrast Max', adj.get('clMax', ''))
            r = write_kv(ws_meta, r, 'Gamma', adj.get('gamma', ''))
            r = write_kv(ws_meta, r, 'Colormap', adj.get('colormap', 'none'))
            r = write_kv(ws_meta, r, 'Inverted', adj.get('invert', False))
        r += 1

    ws_meta.column_dimensions['A'].width = 32
    ws_meta.column_dimensions['B'].width = 60

    # =========================================================================
    #  Sheet 2: Summary (one row per set)
    # =========================================================================
    ws_sum = wb.create_sheet("Summary")
    ws_sum.sheet_properties.tabColor = '4ECDC4'

    sum_headers = ['#', 'Label', 'Image', 'Num Points', 'Total Length (px)']
    if has_physical:
        sum_headers.append(f'Total Length ({px_unit})')
    sum_headers.append('Coordinates (px)')

    for col, h in enumerate(sum_headers, 1):
        c = ws_sum.cell(row=1, column=col, value=h)
        c.font = header_font; c.fill = header_fill
        c.alignment = Alignment(horizontal='center')

    for i, e in enumerate(params.entries):
        row = i + 2
        stats = e.stats or {}
        total_px = stats.get('total_length_pixels', '')
        total_phys = stats.get('total_length_physical', '') if has_physical else None
        coords_str = ' → '.join(f'({p[0]:.0f},{p[1]:.0f})' for p in e.points)
        vals = [i + 1, e.label or f'Set {i+1}', e.imageName, len(e.points), total_px]
        if has_physical:
            vals.append(total_phys)
        vals.append(coords_str)
        for col, v in enumerate(vals, 1):
            ws_sum.cell(row=row, column=col, value=v).font = data_font

    for col_idx, w in enumerate([4, 16, 24, 12, 16, 18, 80], 1):
        if col_idx <= len(sum_headers):
            ws_sum.column_dimensions[get_column_letter(col_idx)].width = w
    ws_sum.freeze_panes = 'A2'

    # =========================================================================
    #  Sheet 3: Points (one row per individual point)
    # =========================================================================
    ws_pts = wb.create_sheet("Points")
    ws_pts.sheet_properties.tabColor = 'FFB347'

    pt_headers = ['Set', 'Image', 'Point #', 'X (px)', 'Y (px)']
    if has_physical:
        pt_headers += [f'X ({px_unit})', f'Y ({px_unit})']
    pt_headers += ['Intensity', 'R', 'G', 'B',
                   'Dist from prev (px)']
    if has_physical:
        pt_headers.append(f'Dist from prev ({px_unit})')
    pt_headers.append('Cumulative (px)')
    if has_physical:
        pt_headers.append(f'Cumulative ({px_unit})')

    for col, h in enumerate(pt_headers, 1):
        c = ws_pts.cell(row=1, column=col, value=h)
        c.font = header_font; c.fill = header_fill
        c.alignment = Alignment(horizontal='center')

    row = 2
    for e in params.entries:
        stats = e.stats or {}
        samples = stats.get('points') or stats.get('samples') or []
        distances = stats.get('distances') or []
        # Map distances by to_index for easy lookup
        dist_by_to = {d.get('to_index'): d for d in distances}
        for i in range(len(e.points)):
            p = e.points[i]
            smp = samples[i] if i < len(samples) else {}
            vals = [
                e.label or '',
                e.imageName,
                i + 1,
                smp.get('x', int(round(p[0]))),
                smp.get('y', int(round(p[1]))),
            ]
            if has_physical:
                vals += [smp.get('x_phys', ''), smp.get('y_phys', '')]
            vals += [
                smp.get('intensity', ''),
                smp.get('red', ''), smp.get('green', ''), smp.get('blue', ''),
            ]
            if i == 0:
                vals += ['']
                if has_physical:
                    vals += ['']
                vals += [0]
                if has_physical:
                    vals += [0]
            else:
                d = dist_by_to.get(i, {})
                vals += [d.get('distance_pixels', '')]
                if has_physical:
                    vals += [d.get('distance_physical', '')]
                vals += [d.get('cumulative_pixels', '')]
                if has_physical:
                    vals += [d.get('cumulative_physical', '')]
            for col, v in enumerate(vals, 1):
                ws_pts.cell(row=row, column=col, value=v).font = data_font
            row += 1

    for col_idx in range(1, len(pt_headers) + 1):
        ws_pts.column_dimensions[get_column_letter(col_idx)].width = 14
    ws_pts.column_dimensions['A'].width = 12
    ws_pts.column_dimensions['B'].width = 24
    ws_pts.freeze_panes = 'A2'

    # =========================================================================
    #  Save
    # =========================================================================
    save_dir, reason = _resolve_export_dir(params.entries[0].imageId, params.saveDirectory)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    img_name = params.entries[0].imageName.replace('/', '_').replace('\\', '_')
    if len(img_name) > 40:
        img_name = img_name[:40]
    base = f'{img_name}_point_sets_{timestamp}'
    save_path = save_dir / f'{base}.xlsx'
    counter = 1
    while save_path.exists():
        save_path = save_dir / f'{base}_{counter}.xlsx'
        counter += 1
    try:
        wb.save(str(save_path))
    except Exception as exc:
        raise HTTPException(500, f"Failed to write Excel file to {save_path}: {exc}")
    if not save_path.exists():
        raise HTTPException(500, f"Excel file was not created at {save_path}")

    return {
        "saved_path": str(save_path),
        "filename": save_path.name,
        "save_dir": str(save_dir),
        "save_dir_reason": reason,
        "num_sets": len(params.entries),
        "num_points": total_points,
        "has_physical_units": bool(has_physical),
    }


# ---------------------------------------------------------------------------
# Measurement Export (angles + freehand curves)
# ---------------------------------------------------------------------------

@router.post("/export-measurements")
async def export_measurements(params: MeasurementExportParams):
    """
    Export saved angle + freehand curve measurements as a multi-sheet xlsx.

    Sheet 1 — Metadata: export info, scale, per-image details
    Sheet 2 — Angles: one row per angle (vertex, arm endpoints, angle, arm lengths)
    Sheet 3 — Curves: one row per curve (length, point count, endpoints)
    Sheet 4 — Curve Points: per-point rows with cumulative distance and intensity (if profile present)
    """
    from datetime import datetime
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    if not params.entries:
        raise HTTPException(400, "No measurements to export")

    wb = Workbook()

    header_font = Font(name='Arial', bold=True, size=11, color='FFFFFF')
    header_fill = PatternFill('solid', fgColor='2B5797')
    subheader_font = Font(name='Arial', bold=True, size=10, color='2B5797')
    label_font = Font(name='Arial', bold=True, size=10)
    data_font = Font(name='Arial', size=10)
    thin_border = Border(bottom=Side(style='thin', color='D0D0D0'))

    has_physical = bool(params.pixelSize and params.pixelSize > 0 and params.pixelSizeUnit)
    px_size = params.pixelSize or 1.0
    px_unit = params.pixelSizeUnit or 'px'

    angles = [e for e in params.entries if e.type == 'angle']
    curves = [e for e in params.entries if e.type == 'curve']

    # --- Sheet 1: Metadata ---
    ws_meta = wb.active
    ws_meta.title = "Metadata"
    ws_meta.sheet_properties.tabColor = '2B5797'

    def write_section(ws, row, title):
        ws.cell(row=row, column=1, value=title).font = subheader_font
        return row + 1

    def write_kv(ws, row, key, value):
        c1 = ws.cell(row=row, column=1, value=key); c1.font = label_font
        c2 = ws.cell(row=row, column=2, value=str(value) if value is not None else '')
        c2.font = data_font
        c1.border = thin_border; c2.border = thin_border
        return row + 1

    r = 1
    for col_idx, txt in enumerate(['Parameter', 'Value'], 1):
        c = ws_meta.cell(row=r, column=col_idx, value=txt)
        c.font = header_font; c.fill = header_fill
        c.alignment = Alignment(horizontal='center')
    r += 1

    r = write_section(ws_meta, r, '── Export Info ──')
    r = write_kv(ws_meta, r, 'Export Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    r = write_kv(ws_meta, r, 'Total Measurements', len(params.entries))
    r = write_kv(ws_meta, r, 'Angles', len(angles))
    r = write_kv(ws_meta, r, 'Curves', len(curves))
    r += 1

    r = write_section(ws_meta, r, '── Scale ──')
    if has_physical:
        r = write_kv(ws_meta, r, 'Pixel Size', f'{px_size} {px_unit}/px')
        r = write_kv(ws_meta, r, 'Distance Unit', px_unit)
    else:
        r = write_kv(ws_meta, r, 'Pixel Size', 'Not available (pixels only)')
    r += 1

    seen_images = set()
    for e in params.entries:
        if e.imageId in seen_images:
            continue
        seen_images.add(e.imageId)
        name = e.imageName or e.imageId
        r = write_section(ws_meta, r, f'── Image: {name} ──')
        r = write_kv(ws_meta, r, 'Image ID', e.imageId)
        stored = store.get(e.imageId)
        if stored:
            m = stored.get('metadata', {})
            r = write_kv(ws_meta, r, 'Dimensions', f'{m.get("width", "?")} x {m.get("height", "?")}')
            r = write_kv(ws_meta, r, 'Bit Depth', m.get('bit_depth', '?'))
            r = write_kv(ws_meta, r, 'Dtype', m.get('dtype', '?'))
        r += 1

    ws_meta.column_dimensions['A'].width = 32
    ws_meta.column_dimensions['B'].width = 60

    # --- Sheet 2: Angles ---
    ws_ang = wb.create_sheet("Angles")
    ws_ang.sheet_properties.tabColor = 'FFCC66'
    ang_headers = ['#', 'Label', 'Image', 'Angle (°)',
                   'Vertex X', 'Vertex Y',
                   'Arm1 X', 'Arm1 Y', 'Arm2 X', 'Arm2 Y',
                   'Arm1 Length (px)', 'Arm2 Length (px)']
    if has_physical:
        ang_headers += [f'Arm1 Length ({px_unit})', f'Arm2 Length ({px_unit})']
    for col, h in enumerate(ang_headers, 1):
        c = ws_ang.cell(row=1, column=col, value=h)
        c.font = header_font; c.fill = header_fill
        c.alignment = Alignment(horizontal='center')
    for i, e in enumerate(angles):
        row = i + 2
        v = e.vertex or [None, None]
        a1 = e.arm1 or [None, None]
        a2 = e.arm2 or [None, None]
        vals = [
            i + 1, e.label or f'Angle {i+1}', e.imageName or e.imageId,
            e.angle_degrees,
            v[0], v[1], a1[0], a1[1], a2[0], a2[1],
            e.arm1_length_px, e.arm2_length_px,
        ]
        if has_physical:
            vals += [
                (e.arm1_length_px * px_size) if e.arm1_length_px is not None else None,
                (e.arm2_length_px * px_size) if e.arm2_length_px is not None else None,
            ]
        for col, val in enumerate(vals, 1):
            ws_ang.cell(row=row, column=col, value=val).font = data_font
    for col_idx in range(1, len(ang_headers) + 1):
        ws_ang.column_dimensions[get_column_letter(col_idx)].width = 14
    ws_ang.column_dimensions['A'].width = 5
    ws_ang.column_dimensions['B'].width = 18
    ws_ang.column_dimensions['C'].width = 22
    ws_ang.freeze_panes = 'A2'

    # --- Sheet 3: Curves ---
    ws_cur = wb.create_sheet("Curves")
    ws_cur.sheet_properties.tabColor = '66DDFF'
    cur_headers = ['#', 'Label', 'Image', 'Num Points',
                   'Length (px)']
    if has_physical:
        cur_headers.append(f'Length ({px_unit})')
    cur_headers += ['Start X', 'Start Y', 'End X', 'End Y']
    for col, h in enumerate(cur_headers, 1):
        c = ws_cur.cell(row=1, column=col, value=h)
        c.font = header_font; c.fill = header_fill
        c.alignment = Alignment(horizontal='center')
    for i, e in enumerate(curves):
        row = i + 2
        pts = e.points or []
        start = pts[0] if pts else [None, None]
        end = pts[-1] if pts else [None, None]
        vals = [
            i + 1, e.label or f'Curve {i+1}', e.imageName or e.imageId,
            len(pts), e.length_px,
        ]
        if has_physical:
            vals.append((e.length_px * px_size) if e.length_px is not None else None)
        vals += [start[0], start[1], end[0], end[1]]
        for col, val in enumerate(vals, 1):
            ws_cur.cell(row=row, column=col, value=val).font = data_font
    for col_idx in range(1, len(cur_headers) + 1):
        ws_cur.column_dimensions[get_column_letter(col_idx)].width = 14
    ws_cur.column_dimensions['A'].width = 5
    ws_cur.column_dimensions['B'].width = 18
    ws_cur.column_dimensions['C'].width = 22
    ws_cur.freeze_panes = 'A2'

    # --- Sheet 4: Curve Points ---
    ws_pts = wb.create_sheet("Curve Points")
    ws_pts.sheet_properties.tabColor = 'B0E0E6'
    pt_headers = ['Curve', 'Image', 'Point #', 'X (px)', 'Y (px)', 'Cumulative (px)']
    if has_physical:
        pt_headers.append(f'Cumulative ({px_unit})')
    pt_headers += ['Intensity', 'R', 'G', 'B']
    for col, h in enumerate(pt_headers, 1):
        c = ws_pts.cell(row=1, column=col, value=h)
        c.font = header_font; c.fill = header_fill
        c.alignment = Alignment(horizontal='center')

    row = 2
    for c_idx, e in enumerate(curves):
        pts = e.points or []
        profile = e.profile or {}
        distances = profile.get('distances') or []
        profiles_dict = profile.get('profiles') or {}
        lum = profiles_dict.get('luminance') or profiles_dict.get('gray') or []
        red = profiles_dict.get('red') or []
        green = profiles_dict.get('green') or []
        blue = profiles_dict.get('blue') or []
        n_prof = len(distances)
        # Output one row per saved point; interpolate profile index
        for i, p in enumerate(pts):
            if n_prof > 0:
                # Map saved point index to profile index
                frac = i / max(1, len(pts) - 1)
                pi = min(n_prof - 1, int(round(frac * (n_prof - 1))))
                cum = distances[pi] if pi < n_prof else ''
                intensity = lum[pi] if pi < len(lum) else ''
                r_v = red[pi] if pi < len(red) else ''
                g_v = green[pi] if pi < len(green) else ''
                b_v = blue[pi] if pi < len(blue) else ''
            else:
                cum = intensity = r_v = g_v = b_v = ''
            vals = [
                e.label or f'Curve {c_idx+1}',
                e.imageName or e.imageId,
                i + 1,
                p[0], p[1],
                cum,
            ]
            if has_physical:
                vals.append(cum * px_size if isinstance(cum, (int, float)) else '')
            vals += [intensity, r_v, g_v, b_v]
            for col, val in enumerate(vals, 1):
                ws_pts.cell(row=row, column=col, value=val).font = data_font
            row += 1

    for col_idx in range(1, len(pt_headers) + 1):
        ws_pts.column_dimensions[get_column_letter(col_idx)].width = 14
    ws_pts.freeze_panes = 'A2'

    # --- Save ---
    save_dir, reason = _resolve_export_dir(params.entries[0].imageId, params.saveDirectory)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    img_name = (params.entries[0].imageName or params.entries[0].imageId).replace('/', '_').replace('\\', '_')
    if len(img_name) > 40:
        img_name = img_name[:40]
    base = f'{img_name}_measurements_{timestamp}'
    save_path = save_dir / f'{base}.xlsx'
    counter = 1
    while save_path.exists():
        save_path = save_dir / f'{base}_{counter}.xlsx'
        counter += 1
    try:
        wb.save(str(save_path))
    except Exception as exc:
        raise HTTPException(500, f"Failed to write Excel file to {save_path}: {exc}")
    if not save_path.exists():
        raise HTTPException(500, f"Excel file was not created at {save_path}")

    return {
        "saved_path": str(save_path),
        "filename": save_path.name,
        "save_dir": str(save_dir),
        "save_dir_reason": reason,
        "num_measurements": len(params.entries),
        "num_angles": len(angles),
        "num_curves": len(curves),
        "has_physical_units": has_physical,
    }


# ---------------------------------------------------------------------------
# ROI Statistics Export (extended — histograms + scatter charts)
# ---------------------------------------------------------------------------

@router.post("/export-roi-statistics")
async def export_roi_statistics(params: ROIStatisticsExportParams):
    """
    Export extended ROI statistics: per-ROI histograms (R/G/B/luminance) plus
    a scatter chart of ROI area vs mean intensity. Uses openpyxl BarChart and
    ScatterChart for embedded visualizations.
    """
    from datetime import datetime
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
    from openpyxl.chart import BarChart, ScatterChart, Reference, Series
    from openpyxl.chart.marker import Marker

    if not params.entries:
        raise HTTPException(400, "No ROIs to export")

    wb = Workbook()

    header_font = Font(name='Arial', bold=True, size=11, color='FFFFFF')
    header_fill = PatternFill('solid', fgColor='2B5797')
    subheader_font = Font(name='Arial', bold=True, size=10, color='2B5797')
    data_font = Font(name='Arial', size=10)

    has_physical = bool(params.pixelSize and params.pixelSize > 0 and params.pixelSizeUnit)
    px_size = params.pixelSize or 1.0
    px_unit = params.pixelSizeUnit or 'px'

    # --- Sheet 1: Metadata ---
    ws_meta = wb.active
    ws_meta.title = "Metadata"
    ws_meta.sheet_properties.tabColor = '2B5797'
    ws_meta.cell(row=1, column=1, value='Parameter').font = header_font
    ws_meta.cell(row=1, column=1).fill = header_fill
    ws_meta.cell(row=1, column=2, value='Value').font = header_font
    ws_meta.cell(row=1, column=2).fill = header_fill
    meta_rows = [
        ('Export Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
        ('Number of ROIs', len(params.entries)),
        ('Pixel Size', f'{px_size} {px_unit}/px' if has_physical else 'Not available'),
        ('Histogram Bins', 64),
    ]
    for i, (k, v) in enumerate(meta_rows):
        ws_meta.cell(row=i + 2, column=1, value=k).font = data_font
        ws_meta.cell(row=i + 2, column=2, value=str(v)).font = data_font
    ws_meta.column_dimensions['A'].width = 24
    ws_meta.column_dimensions['B'].width = 40

    # --- Sheet 2: Summary ---
    ws_sum = wb.create_sheet("Summary")
    ws_sum.sheet_properties.tabColor = '4ECDC4'
    sum_headers = ['#', 'Label', 'Image', 'Shape', 'Area (px)']
    if has_physical:
        sum_headers.append(f'Area ({px_unit}²)')
    sum_headers += ['Mean', 'Std', 'Min', 'Max', 'Median', 'IntDen']
    for col, h in enumerate(sum_headers, 1):
        c = ws_sum.cell(row=1, column=col, value=h)
        c.font = header_font; c.fill = header_fill
        c.alignment = Alignment(horizontal='center')
    for i, e in enumerate(params.entries):
        row = i + 2
        s = e.stats or {}
        vals = [
            i + 1, e.label or f'ROI {i+1}', e.imageName, e.shape,
            s.get('area_pixels'),
        ]
        if has_physical:
            vals.append(s.get('area_physical'))
        vals += [s.get('mean'), s.get('std'), s.get('min'), s.get('max'),
                 s.get('median'), s.get('integrated_density')]
        for col, v in enumerate(vals, 1):
            ws_sum.cell(row=row, column=col, value=v).font = data_font
    for col_idx in range(1, len(sum_headers) + 1):
        ws_sum.column_dimensions[get_column_letter(col_idx)].width = 14
    ws_sum.column_dimensions['B'].width = 18
    ws_sum.column_dimensions['C'].width = 22
    ws_sum.freeze_panes = 'A2'

    # --- Sheet 3: Histograms (per ROI, with embedded BarChart) ---
    ws_hist = wb.create_sheet("Histograms")
    ws_hist.sheet_properties.tabColor = 'FFB347'

    NBINS = 64
    edges = np.linspace(0, 256, NBINS + 1)
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()

    current_row = 1
    for i, e in enumerate(params.entries):
        # Compute histogram by rebuilding the ROI mask and sampling the display array
        try:
            disp = store.load_display_array(e.imageId)
        except Exception:
            continue
        if disp is None:
            continue
        h_img, w_img = disp.shape[:2]

        # Build ShapedROIParams-like object for _build_roi_mask
        class _P:
            pass
        pm = _P()
        pm.x = pm.y = pm.width = pm.height = None
        pm.points = None
        if e.shape in ('rect', 'ellipse') and e.rect:
            pm.x = int(e.rect.get('x', 0))
            pm.y = int(e.rect.get('y', 0))
            pm.width = int(e.rect.get('width', 0))
            pm.height = int(e.rect.get('height', 0))
        elif e.shape == 'polygon' and e.points:
            pm.points = [[float(p[0]), float(p[1])] for p in e.points]
        try:
            mask = _build_roi_mask(e.shape, h_img, w_img, pm)  # type: ignore
        except Exception:
            continue
        if not mask.any():
            continue

        # Header block
        ws_hist.cell(row=current_row, column=1,
                     value=f'ROI {i+1}: {e.label or ""} — {e.imageName or e.imageId}').font = subheader_font
        current_row += 1
        ws_hist.cell(row=current_row, column=1, value='Bin Center').font = header_font
        ws_hist.cell(row=current_row, column=1).fill = header_fill

        channel_names: list = []
        channel_data: list = []
        if disp.ndim == 3 and disp.shape[2] >= 3:
            for ci, cname in enumerate(['Red', 'Green', 'Blue']):
                hvals, _ = np.histogram(disp[:, :, ci][mask], bins=edges)
                channel_names.append(cname)
                channel_data.append(hvals.tolist())
            lum = np.mean(disp[:, :, :3], axis=2)
            hvals, _ = np.histogram(lum[mask], bins=edges)
            channel_names.append('Luminance')
            channel_data.append(hvals.tolist())
        else:
            gray = disp if disp.ndim == 2 else disp[:, :, 0]
            hvals, _ = np.histogram(gray[mask], bins=edges)
            channel_names.append('Gray')
            channel_data.append(hvals.tolist())

        for col, cname in enumerate(channel_names, start=2):
            c = ws_hist.cell(row=current_row, column=col, value=cname)
            c.font = header_font; c.fill = header_fill
            c.alignment = Alignment(horizontal='center')
        data_start_row = current_row + 1

        for bi in range(NBINS):
            ws_hist.cell(row=data_start_row + bi, column=1, value=round(centers[bi], 2)).font = data_font
            for col_i, vals in enumerate(channel_data):
                ws_hist.cell(row=data_start_row + bi, column=2 + col_i, value=int(vals[bi])).font = data_font

        # Embed BarChart for this ROI
        chart = BarChart()
        chart.type = "col"
        chart.style = 10
        chart.title = f'Histogram — {e.label or f"ROI {i+1}"}'
        chart.y_axis.title = 'Count'
        chart.x_axis.title = 'Intensity'
        chart.height = 7
        chart.width = 15

        data_ref = Reference(
            ws_hist,
            min_col=2, max_col=1 + len(channel_names),
            min_row=current_row, max_row=data_start_row + NBINS - 1,
        )
        cats_ref = Reference(
            ws_hist,
            min_col=1, max_col=1,
            min_row=data_start_row, max_row=data_start_row + NBINS - 1,
        )
        chart.add_data(data_ref, titles_from_data=True)
        chart.set_categories(cats_ref)

        # Anchor chart to the right of the data block
        anchor_col = get_column_letter(len(channel_names) + 3)
        ws_hist.add_chart(chart, f'{anchor_col}{current_row}')

        current_row = data_start_row + NBINS + 2  # advance past data + spacing

    for col_idx in range(1, 8):
        ws_hist.column_dimensions[get_column_letter(col_idx)].width = 14

    # --- Sheet 4: Scatter (Area vs Mean) ---
    ws_sc = wb.create_sheet("Scatter")
    ws_sc.sheet_properties.tabColor = 'CC99FF'

    sc_headers = ['#', 'Label', 'Image',
                  f'Area ({px_unit}²)' if has_physical else 'Area (px)',
                  'Mean', 'Std']
    for col, h in enumerate(sc_headers, 1):
        c = ws_sc.cell(row=1, column=col, value=h)
        c.font = header_font; c.fill = header_fill
        c.alignment = Alignment(horizontal='center')

    scatter_rows = 0
    for i, e in enumerate(params.entries):
        s = e.stats or {}
        if has_physical and s.get('area_physical') is not None:
            area_val = s.get('area_physical')
        else:
            area_val = s.get('area_pixels')
        if area_val is None or s.get('mean') is None:
            continue
        row = scatter_rows + 2
        ws_sc.cell(row=row, column=1, value=i + 1).font = data_font
        ws_sc.cell(row=row, column=2, value=e.label or f'ROI {i+1}').font = data_font
        ws_sc.cell(row=row, column=3, value=e.imageName).font = data_font
        ws_sc.cell(row=row, column=4, value=area_val).font = data_font
        ws_sc.cell(row=row, column=5, value=s.get('mean')).font = data_font
        ws_sc.cell(row=row, column=6, value=s.get('std')).font = data_font
        scatter_rows += 1

    for col_idx in range(1, len(sc_headers) + 1):
        ws_sc.column_dimensions[get_column_letter(col_idx)].width = 16

    if scatter_rows >= 1:
        chart = ScatterChart()
        chart.title = 'ROI Area vs Mean Intensity'
        chart.style = 13
        chart.x_axis.title = sc_headers[3]
        chart.y_axis.title = 'Mean Intensity'
        chart.height = 10
        chart.width = 18

        x_ref = Reference(ws_sc, min_col=4, min_row=2, max_row=1 + scatter_rows)
        y_ref = Reference(ws_sc, min_col=5, min_row=2, max_row=1 + scatter_rows)
        series = Series(y_ref, x_ref, title='ROIs')
        try:
            series.marker = Marker(symbol='circle', size=8)
            series.graphicalProperties.line.noFill = True
        except Exception:
            pass
        chart.series.append(series)
        ws_sc.add_chart(chart, 'H2')

    # --- Save ---
    save_dir, reason = _resolve_export_dir(params.entries[0].imageId, params.saveDirectory)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    img_name = (params.entries[0].imageName or params.entries[0].imageId).replace('/', '_').replace('\\', '_')
    if len(img_name) > 40:
        img_name = img_name[:40]
    base = f'{img_name}_roi_statistics_{timestamp}'
    save_path = save_dir / f'{base}.xlsx'
    counter = 1
    while save_path.exists():
        save_path = save_dir / f'{base}_{counter}.xlsx'
        counter += 1
    try:
        wb.save(str(save_path))
    except Exception as exc:
        raise HTTPException(500, f"Failed to write Excel file to {save_path}: {exc}")
    if not save_path.exists():
        raise HTTPException(500, f"Excel file was not created at {save_path}")

    return {
        "saved_path": str(save_path),
        "filename": save_path.name,
        "save_dir": str(save_dir),
        "save_dir_reason": reason,
        "num_rois": len(params.entries),
        "has_physical_units": has_physical,
    }


# ---------------------------------------------------------------------------
# Annotations (JSON sidecar save/load)
# ---------------------------------------------------------------------------

def _annotation_sidecar_for_image(image_id: str, image_path_hint: str | None):
    """Compute the sidecar .annotations.json path for a given image.

    Prefers the real image path on disk (so annotations live next to the file).
    Falls back to a hint string if the store doesn't have a path.
    """
    import os
    from pathlib import Path

    image_path = None
    if store.contains(image_id):
        entry = store.get(image_id)
        image_path = entry.get("path")
    if not image_path:
        image_path = image_path_hint
    if not image_path:
        return None
    p = Path(image_path)
    # <stem>.annotations.json next to the image
    return p.with_suffix("").parent / f"{p.stem}.annotations.json"


@router.post("/annotations/save")
def save_annotations(params: AnnotationSaveParams):
    """Write a <image_stem>.annotations.json sidecar next to each image."""
    import json
    from pathlib import Path

    saved = []
    errors = []
    fallback_dir = None
    if params.saveDirectory:
        fallback_dir = Path(params.saveDirectory).expanduser()
        try:
            fallback_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise HTTPException(500, f"Could not create save directory: {exc}")

    for image_id, group in (params.annotations or {}).items():
        entries = group.get("entries", []) if isinstance(group, dict) else []
        if not entries:
            continue
        image_path_hint = group.get("imagePath") if isinstance(group, dict) else None
        sidecar = _annotation_sidecar_for_image(image_id, image_path_hint)

        # If sidecar path is not writable or unknown, fall back to saveDirectory.
        target = None
        if sidecar is not None:
            try:
                sidecar.parent.mkdir(parents=True, exist_ok=True)
                target = sidecar
            except Exception:
                target = None
        if target is None and fallback_dir is not None:
            name = (group.get("imageName") if isinstance(group, dict) else None) or image_id
            stem = Path(name).stem or image_id
            target = fallback_dir / f"{stem}.annotations.json"
        if target is None:
            errors.append({"image_id": image_id, "error": "no writable location"})
            continue

        payload = {
            "image_id": image_id,
            "image_name": group.get("imageName") if isinstance(group, dict) else None,
            "entries": entries,
        }
        try:
            with open(target, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            saved.append({"image_id": image_id, "path": str(target), "count": len(entries)})
        except Exception as exc:
            errors.append({"image_id": image_id, "error": str(exc)})

    return {
        "saved_count": len(saved),
        "saved": saved,
        "errors": errors,
    }


@router.post("/annotations/load")
def load_annotations(params: AnnotationLoadParams):
    """Load annotation sidecar JSONs for the given open image ids, when present."""
    import json

    result = {}
    errors = []
    for image_id in params.image_ids:
        sidecar = _annotation_sidecar_for_image(image_id, None)
        if sidecar is None or not sidecar.exists():
            continue
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data.get("entries") if isinstance(data, dict) else None
            if isinstance(entries, list):
                result[image_id] = entries
        except Exception as exc:
            errors.append({"image_id": image_id, "error": str(exc)})

    return {"annotations": result, "errors": errors}


# ---------------------------------------------------------------------------
# Channel splitter — split an RGB(A) image into per-channel grayscale views
# ---------------------------------------------------------------------------

@router.post("/images/{image_id}/split-channels")
async def split_channels(image_id: str):
    """
    Split a multi-channel image into separate single-channel grayscale images.
    Each new image is registered in the store and returned in the response so
    the frontend can open them as new tabs.

    - 2-D grayscale images are rejected (nothing to split).
    - 3-channel images yield {R, G, B}; 4-channel images add {A}.
    - Channel names come from entry["metadata"]["bands"] if available
      (e.g. LIF multi-channel uses physical channel names).
    """
    if not store.contains(image_id):
        raise HTTPException(404, f"Unknown image_id: {image_id}")

    entry = store.get(image_id)
    arr = store.load_array(image_id)

    if arr.ndim != 3 or arr.shape[2] < 2:
        raise HTTPException(400, "Image has only one channel — nothing to split.")

    from ..image_processing import normalize_to_uint8

    nchan = arr.shape[2]
    bands = entry["metadata"].get("bands")
    if not bands or len(bands) < nchan:
        # Default labels for RGB(A)
        default = ["R", "G", "B", "A"]
        bands = default[:nchan]

    created = []
    base_name = entry.get("name") or "image"
    base_stem = base_name.rsplit(".", 1)[0] if "." in base_name else base_name

    for i, ch_label in enumerate(bands[:nchan]):
        ch_arr = arr[..., i].copy()
        ch_display = normalize_to_uint8(ch_arr)

        new_id = uuid.uuid4().hex[:8]
        new_name = f"{base_stem}_{ch_label}"

        metadata = {
            "filename": new_name,
            "width": int(ch_arr.shape[1]),
            "height": int(ch_arr.shape[0]),
            "mode": "L",
            "channels": 1,
            "bands": [ch_label],
            "dtype": str(ch_arr.dtype),
            "bit_depth": int(ch_arr.dtype.itemsize * 8),
            "size_bytes": int(ch_arr.nbytes),
            "min_value": int(ch_arr.min()),
            "max_value": int(ch_arr.max()),
            "mean_value": round(float(ch_arr.mean()), 2),
            "percentile_1": round(float(np.percentile(ch_arr.astype(np.float64), 1)), 2),
            "percentile_99": round(float(np.percentile(ch_arr.astype(np.float64), 99)), 2),
            "source_image_id": image_id,
            "source_channel": ch_label,
        }

        # Copy pixel-size metadata if present
        for k in ("pixel_size_x", "pixel_size_y", "pixel_size_unit"):
            if entry["metadata"].get(k) is not None:
                metadata[k] = entry["metadata"][k]

        store.put(new_id, {
            "path": entry["path"],   # no new file on disk; share original path
            "name": new_name,
            "metadata": metadata,
            "array": ch_arr,
            "display_array": ch_display,
        })
        created.append({"image_id": new_id, "name": new_name, "channel": ch_label, "metadata": metadata})

    return {"source_image_id": image_id, "created": created}
