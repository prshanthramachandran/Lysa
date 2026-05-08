"""
Segmentation routes — classical (skimage) and deep-learning (Cellpose) pipelines.

Pipeline shape (classical):
    grayscale --> threshold --> morphological cleanup --> (optional) watershed
    split --> connected-components labeling --> regionprops measurements.

Pipeline shape (Cellpose):
    image --> Cellpose model.eval --> instance labels --> regionprops measurements.

Both pipelines return the same envelope:
    {
        num_objects, engine, params,
        objects: [ { id, centroid, bbox, contour, ...measurements } ],
        image_id, width, height,
    }

Labels are cached in memory (per image_id) so that follow-up actions
(promote-to-ROI, export, re-render) don't need to re-run the whole pipeline.
"""

import math
import time
import threading
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation as skseg
from skimage.measure import regionprops, find_contours

from .. import store


router = APIRouter(prefix="/api", tags=["segmentation"])


# ---------------------------------------------------------------------------
# Label cache (in-memory, per image)
# ---------------------------------------------------------------------------

_label_cache: dict = {}             # image_id -> { labels, engine, params, intensity, timestamp }
_label_cache_lock = threading.Lock()
_CACHE_MAX_ENTRIES = 16             # keep the last N segmentations in memory


def _store_labels(image_id: str, labels: np.ndarray, engine: str, params: dict, intensity: np.ndarray):
    with _label_cache_lock:
        _label_cache[image_id] = {
            "labels": labels,
            "engine": engine,
            "params": params,
            "intensity": intensity,
            "ts": time.time(),
        }
        # Evict oldest entries if we exceed the cap
        if len(_label_cache) > _CACHE_MAX_ENTRIES:
            oldest = sorted(_label_cache.items(), key=lambda kv: kv[1]["ts"])
            for k, _ in oldest[: len(_label_cache) - _CACHE_MAX_ENTRIES]:
                _label_cache.pop(k, None)


def _get_labels(image_id: str):
    with _label_cache_lock:
        return _label_cache.get(image_id)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ClassicalSegmentParams(BaseModel):
    threshold_method: str = "otsu"          # "otsu" | "triangle" | "manual"
    threshold_value: Optional[float] = None # required if method == "manual", 0..255
    invert: bool = False                    # treat darker pixels as foreground
    gaussian_sigma: float = 0.0             # pre-smoothing sigma (0 = off)
    morph_open: int = 0                     # disk-radius iterations of opening
    morph_close: int = 0                    # disk-radius iterations of closing
    fill_holes: bool = True
    min_size: int = 25                      # drop components smaller than this (px)
    watershed_split: bool = False           # split touching objects via distance transform
    watershed_min_distance: int = 10        # min peak distance for watershed seeds


class CellposeSegmentParams(BaseModel):
    model: str = "cyto3"                    # cyto3 | cyto2 | cyto | nuclei
    diameter: Optional[float] = None        # None -> auto
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    channels: List[int] = [0, 0]            # [cyto, nuclei]; [0,0] = grayscale
    min_size: int = 15


class PromoteToROIParams(BaseModel):
    object_ids: Optional[List[int]] = None  # None = promote all


class SegmentationExportEntry(BaseModel):
    imageId: str
    imageName: str
    engine: str
    params: dict
    objects: List[dict]


class SegmentationExportParams(BaseModel):
    entries: List[SegmentationExportEntry]
    pixelSize: Optional[float] = None
    pixelSizeUnit: Optional[str] = None
    saveDirectory: Optional[str] = None


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def _load_grayscale(image_id: str) -> np.ndarray:
    """Return a uint8 grayscale view of the image (for segmentation input)."""
    if not store.contains(image_id):
        raise HTTPException(404, f"Unknown image_id: {image_id}")
    disp = store.load_display_array(image_id)  # uint8, (H, W) or (H, W, 3/4)
    if disp.ndim == 3:
        # Simple luminance
        r, g, b = disp[..., 0], disp[..., 1], disp[..., 2]
        gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.uint8)
        return gray
    return disp.astype(np.uint8)


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

_SHAPE_PROPS = (
    "label", "area", "perimeter", "bbox", "centroid",
    "equivalent_diameter_area", "eccentricity", "orientation",
    "solidity", "extent",
)

_INTENSITY_PROPS = (
    "intensity_mean", "intensity_min", "intensity_max",
)

_ADVANCED_SHAPE_PROPS = (
    "axis_major_length", "axis_minor_length", "feret_diameter_max",
    "area_convex",
)


def _extract_contour(label_image: np.ndarray, label_value: int) -> list:
    """Return the largest closed contour for a single labeled object as [[x, y], ...] image coords."""
    sub = (label_image == label_value).astype(np.uint8)
    if sub.sum() == 0:
        return []
    contours = find_contours(sub, 0.5)
    if not contours:
        return []
    biggest = max(contours, key=len)
    # find_contours returns (row, col); convert to (x, y)
    # Also subsample if very long to keep payload small
    pts = [[float(c[1]), float(c[0])] for c in biggest]
    if len(pts) > 400:
        step = max(1, len(pts) // 400)
        pts = pts[::step]
    return pts


def _measure_objects(labels: np.ndarray, intensity: np.ndarray) -> list:
    """Run regionprops and build the per-object dict list."""
    if labels.max() == 0:
        return []
    props = regionprops(labels, intensity_image=intensity)
    objs = []
    for p in props:
        try:
            area = int(p.area)
            if area <= 0:
                continue
            # Perimeter is degree-2 in skimage; fall back if it raises.
            try:
                perim = float(p.perimeter)
            except Exception:
                perim = 0.0
            # Circularity = 4πA / P^2
            circ = (4.0 * math.pi * area / (perim * perim)) if perim > 0 else 0.0
            minr, minc, maxr, maxc = [int(v) for v in p.bbox]
            cy, cx = p.centroid  # (row, col)
            obj = {
                "id": int(p.label),
                "area": area,
                "perimeter": perim,
                "circularity": circ,
                "bbox": {"x": minc, "y": minr, "width": maxc - minc, "height": maxr - minr},
                "centroid": [float(cx), float(cy)],
                "equivalent_diameter": float(p.equivalent_diameter_area),
                "eccentricity": float(p.eccentricity),
                "orientation_rad": float(p.orientation),
                "solidity": float(p.solidity),
                "extent": float(p.extent),
                # Advanced shape
                "major_axis": float(p.axis_major_length),
                "minor_axis": float(p.axis_minor_length),
                "aspect_ratio": (float(p.axis_major_length) / float(p.axis_minor_length))
                    if p.axis_minor_length > 0 else 0.0,
                "feret_max": float(p.feret_diameter_max) if hasattr(p, "feret_diameter_max") else 0.0,
                "convex_area": int(p.area_convex),
                # Intensity stats
                "intensity_mean": float(p.intensity_mean) if p.intensity_mean is not None else 0.0,
                "intensity_min": float(p.intensity_min) if p.intensity_min is not None else 0.0,
                "intensity_max": float(p.intensity_max) if p.intensity_max is not None else 0.0,
            }
            # std / sum need manual computation
            mask = labels == p.label
            vals = intensity[mask]
            if vals.size > 0:
                obj["intensity_std"] = float(vals.std())
                obj["intensity_sum"] = float(vals.sum())
                obj["intensity_median"] = float(np.median(vals))
            else:
                obj["intensity_std"] = 0.0
                obj["intensity_sum"] = 0.0
                obj["intensity_median"] = 0.0

            # Contour for overlay and ROI promotion (cropped bbox is faster)
            sub = labels[minr:maxr, minc:maxc]
            contours = find_contours((sub == p.label).astype(np.uint8), 0.5)
            if contours:
                biggest = max(contours, key=len)
                pts = [[float(c[1] + minc), float(c[0] + minr)] for c in biggest]
                if len(pts) > 400:
                    step = max(1, len(pts) // 400)
                    pts = pts[::step]
                obj["contour"] = pts
            else:
                obj["contour"] = []
            objs.append(obj)
        except Exception as exc:
            # Skip degenerate objects rather than failing the whole request
            continue
    return objs


# ---------------------------------------------------------------------------
# Classical pipeline
# ---------------------------------------------------------------------------

def _apply_classical_pipeline(gray: np.ndarray, params: ClassicalSegmentParams) -> np.ndarray:
    """Run the classical pipeline and return a uint32 label image."""
    img = gray.astype(np.float32)
    if params.gaussian_sigma and params.gaussian_sigma > 0:
        img = filters.gaussian(img, sigma=float(params.gaussian_sigma), preserve_range=True)

    # Threshold
    if params.threshold_method == "otsu":
        try:
            t = float(filters.threshold_otsu(img))
        except Exception:
            t = 128.0
    elif params.threshold_method == "triangle":
        try:
            t = float(filters.threshold_triangle(img))
        except Exception:
            t = 128.0
    elif params.threshold_method == "manual":
        if params.threshold_value is None:
            raise HTTPException(400, "Manual threshold requires threshold_value")
        t = float(params.threshold_value)
    else:
        raise HTTPException(400, f"Unknown threshold_method: {params.threshold_method}")

    if params.invert:
        mask = img < t
    else:
        mask = img > t

    # Morphological cleanup
    if params.morph_open and params.morph_open > 0:
        mask = morphology.binary_opening(mask, footprint=morphology.disk(int(params.morph_open)))
    if params.morph_close and params.morph_close > 0:
        mask = morphology.binary_closing(mask, footprint=morphology.disk(int(params.morph_close)))
    if params.fill_holes:
        mask = ndi.binary_fill_holes(mask)
    if params.min_size and params.min_size > 0:
        mask = morphology.remove_small_objects(mask, min_size=int(params.min_size))

    # Connected components (or watershed split)
    if params.watershed_split:
        distance = ndi.distance_transform_edt(mask)
        # Peak detection via h-maxima + local max
        from skimage.feature import peak_local_max
        coords = peak_local_max(
            distance,
            min_distance=max(1, int(params.watershed_min_distance)),
            labels=mask.astype(np.uint8),
        )
        marker_mask = np.zeros(distance.shape, dtype=bool)
        for r, c in coords:
            marker_mask[r, c] = True
        markers, _ = ndi.label(marker_mask)
        labels = skseg.watershed(-distance, markers, mask=mask)
    else:
        labels, _ = ndi.label(mask)

    # Relabel densely 1..N
    labels = measure.label(labels, background=0).astype(np.uint32)
    return labels


@router.post("/images/{image_id}/segment/classical")
def segment_classical(image_id: str, params: ClassicalSegmentParams):
    gray = _load_grayscale(image_id)
    try:
        labels = _apply_classical_pipeline(gray, params)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Classical segmentation failed: {exc}")

    objs = _measure_objects(labels, gray)
    _store_labels(image_id, labels, "classical", params.model_dump(), gray)

    return {
        "engine": "classical",
        "params": params.model_dump(),
        "image_id": image_id,
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
        "num_objects": len(objs),
        "objects": objs,
    }


# ---------------------------------------------------------------------------
# Cellpose pipeline (lazy import)
# ---------------------------------------------------------------------------

@router.post("/images/{image_id}/segment/cellpose")
def segment_cellpose(image_id: str, params: CellposeSegmentParams):
    try:
        from cellpose import models as cp_models
    except ImportError:
        raise HTTPException(
            503,
            "Cellpose is not installed. Install with: "
            "pip install 'cellpose>=3.0' torch"
        )

    gray = _load_grayscale(image_id)
    try:
        model = cp_models.Cellpose(gpu=False, model_type=params.model)
        masks, flows, styles, diams = model.eval(
            gray,
            diameter=params.diameter,
            channels=params.channels,
            flow_threshold=params.flow_threshold,
            cellprob_threshold=params.cellprob_threshold,
        )
    except Exception as exc:
        raise HTTPException(500, f"Cellpose inference failed: {exc}")

    labels = np.asarray(masks, dtype=np.uint32)
    if params.min_size and params.min_size > 0:
        labels = morphology.remove_small_objects(labels, min_size=int(params.min_size))
        labels = measure.label(labels, background=0).astype(np.uint32)

    objs = _measure_objects(labels, gray)
    cached_params = params.model_dump()
    cached_params["auto_diameter_estimate"] = float(diams) if diams is not None else None
    _store_labels(image_id, labels, "cellpose", cached_params, gray)

    return {
        "engine": "cellpose",
        "params": cached_params,
        "image_id": image_id,
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
        "num_objects": len(objs),
        "objects": objs,
    }


# ---------------------------------------------------------------------------
# Clear + Promote-to-ROI
# ---------------------------------------------------------------------------

@router.post("/images/{image_id}/segment/clear")
def clear_segmentation(image_id: str):
    with _label_cache_lock:
        _label_cache.pop(image_id, None)
    return {"ok": True}


@router.post("/images/{image_id}/segment/promote-to-roi")
def promote_to_roi(image_id: str, params: PromoteToROIParams):
    """Return polygon ROIs for selected (or all) segmented objects.

    The frontend consumes these and inserts them into its own ROI store —
    the backend does not mutate the ROI layer itself here.
    """
    entry = _get_labels(image_id)
    if entry is None:
        raise HTTPException(404, "No cached segmentation for this image. Run segmentation first.")
    labels = entry["labels"]
    intensity = entry["intensity"]
    wanted = set(params.object_ids) if params.object_ids is not None else None

    rois = []
    props = regionprops(labels, intensity_image=intensity)
    for p in props:
        if wanted is not None and int(p.label) not in wanted:
            continue
        minr, minc, maxr, maxc = [int(v) for v in p.bbox]
        sub = labels[minr:maxr, minc:maxc]
        contours = find_contours((sub == p.label).astype(np.uint8), 0.5)
        if not contours:
            continue
        biggest = max(contours, key=len)
        pts = [[int(round(c[1] + minc)), int(round(c[0] + minr))] for c in biggest]
        if len(pts) > 200:
            step = max(1, len(pts) // 200)
            pts = pts[::step]
        if len(pts) < 3:
            continue
        rois.append({
            "label": f"obj_{int(p.label)}",
            "shape": "polygon",
            "points": pts,
            "area": int(p.area),
            "centroid": [float(p.centroid[1]), float(p.centroid[0])],
        })

    return {"rois": rois, "count": len(rois)}


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------

_EXPORT_COLUMNS = [
    ("image_name", "Image"),
    ("id", "Object ID"),
    ("area", "Area (px)"),
    ("area_phys", "Area"),
    ("perimeter", "Perimeter (px)"),
    ("perimeter_phys", "Perimeter"),
    ("equivalent_diameter", "Equiv. Diameter (px)"),
    ("equivalent_diameter_phys", "Equiv. Diameter"),
    ("centroid_x", "Centroid X"),
    ("centroid_y", "Centroid Y"),
    ("bbox_x", "BBox X"),
    ("bbox_y", "BBox Y"),
    ("bbox_w", "BBox Width"),
    ("bbox_h", "BBox Height"),
    ("major_axis", "Major Axis (px)"),
    ("minor_axis", "Minor Axis (px)"),
    ("aspect_ratio", "Aspect Ratio"),
    ("feret_max", "Feret Max (px)"),
    ("eccentricity", "Eccentricity"),
    ("solidity", "Solidity"),
    ("extent", "Extent"),
    ("circularity", "Circularity"),
    ("convex_area", "Convex Area (px)"),
    ("orientation_rad", "Orientation (rad)"),
    ("intensity_mean", "Intensity Mean"),
    ("intensity_std", "Intensity Std"),
    ("intensity_min", "Intensity Min"),
    ("intensity_max", "Intensity Max"),
    ("intensity_median", "Intensity Median"),
    ("intensity_sum", "Intensity Sum"),
]


@router.post("/export-segmentation")
def export_segmentation(params: SegmentationExportParams):
    """Export per-object segmentation measurements as a single Excel file.

    Sheet 1 — Metadata (export info, scale, per-image engine + params)
    Sheet 2 — Objects (one row per segmented object, columns per measurement)
    """
    from datetime import datetime
    from pathlib import Path
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment

    if not params.entries:
        raise HTTPException(400, "No segmentation entries to export")

    has_phys = bool(params.pixelSize and params.pixelSize > 0 and params.pixelSizeUnit)
    px_size = params.pixelSize or 1.0
    px_unit = params.pixelSizeUnit or "px"

    wb = Workbook()
    header_font = Font(name="Arial", bold=True, size=11, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="2B5797")
    subheader_font = Font(name="Arial", bold=True, size=10, color="2B5797")
    label_font = Font(name="Arial", bold=True, size=10)
    data_font = Font(name="Arial", size=10)

    # ----- Metadata sheet -----
    ws_meta = wb.active
    ws_meta.title = "Metadata"
    ws_meta.sheet_properties.tabColor = "2B5797"
    for i, txt in enumerate(["Parameter", "Value"], 1):
        c = ws_meta.cell(row=1, column=i, value=txt)
        c.font = header_font
        c.fill = header_fill
        c.alignment = Alignment(horizontal="center")
    r = 2
    ws_meta.cell(row=r, column=1, value="Export Date").font = label_font
    ws_meta.cell(row=r, column=2, value=datetime.now().strftime("%Y-%m-%d %H:%M:%S")).font = data_font
    r += 1
    ws_meta.cell(row=r, column=1, value="Number of Images").font = label_font
    ws_meta.cell(row=r, column=2, value=len(params.entries)).font = data_font
    r += 1
    ws_meta.cell(row=r, column=1, value="Total Objects").font = label_font
    ws_meta.cell(row=r, column=2, value=sum(len(e.objects) for e in params.entries)).font = data_font
    r += 1
    ws_meta.cell(row=r, column=1, value="Scale").font = label_font
    ws_meta.cell(row=r, column=2, value=(f"{px_size} {px_unit}/px" if has_phys else "pixels only")).font = data_font
    r += 2

    for e in params.entries:
        ws_meta.cell(row=r, column=1, value=f"── {e.imageName} ──").font = subheader_font
        r += 1
        ws_meta.cell(row=r, column=1, value="Image ID").font = label_font
        ws_meta.cell(row=r, column=2, value=e.imageId).font = data_font
        r += 1
        ws_meta.cell(row=r, column=1, value="Engine").font = label_font
        ws_meta.cell(row=r, column=2, value=e.engine).font = data_font
        r += 1
        ws_meta.cell(row=r, column=1, value="Objects").font = label_font
        ws_meta.cell(row=r, column=2, value=len(e.objects)).font = data_font
        r += 1
        for pk, pv in (e.params or {}).items():
            ws_meta.cell(row=r, column=1, value=f"  {pk}").font = data_font
            ws_meta.cell(row=r, column=2, value=str(pv)).font = data_font
            r += 1
        r += 1

    ws_meta.column_dimensions["A"].width = 28
    ws_meta.column_dimensions["B"].width = 48

    # ----- Objects sheet -----
    ws_obj = wb.create_sheet("Objects")
    headers = [label for _, label in _EXPORT_COLUMNS]
    for i, h in enumerate(headers, 1):
        c = ws_obj.cell(row=1, column=i, value=h)
        c.font = header_font
        c.fill = header_fill
        c.alignment = Alignment(horizontal="center")

    row = 2
    for e in params.entries:
        for obj in e.objects:
            centroid = obj.get("centroid") or [0, 0]
            bbox = obj.get("bbox") or {}
            area = obj.get("area", 0)
            perim = obj.get("perimeter", 0)
            eqd = obj.get("equivalent_diameter", 0)
            values = {
                "image_name": e.imageName,
                "id": obj.get("id", 0),
                "area": area,
                "area_phys": (area * px_size * px_size) if has_phys else "",
                "perimeter": perim,
                "perimeter_phys": (perim * px_size) if has_phys else "",
                "equivalent_diameter": eqd,
                "equivalent_diameter_phys": (eqd * px_size) if has_phys else "",
                "centroid_x": centroid[0],
                "centroid_y": centroid[1],
                "bbox_x": bbox.get("x", 0),
                "bbox_y": bbox.get("y", 0),
                "bbox_w": bbox.get("width", 0),
                "bbox_h": bbox.get("height", 0),
                "major_axis": obj.get("major_axis", 0),
                "minor_axis": obj.get("minor_axis", 0),
                "aspect_ratio": obj.get("aspect_ratio", 0),
                "feret_max": obj.get("feret_max", 0),
                "eccentricity": obj.get("eccentricity", 0),
                "solidity": obj.get("solidity", 0),
                "extent": obj.get("extent", 0),
                "circularity": obj.get("circularity", 0),
                "convex_area": obj.get("convex_area", 0),
                "orientation_rad": obj.get("orientation_rad", 0),
                "intensity_mean": obj.get("intensity_mean", 0),
                "intensity_std": obj.get("intensity_std", 0),
                "intensity_min": obj.get("intensity_min", 0),
                "intensity_max": obj.get("intensity_max", 0),
                "intensity_median": obj.get("intensity_median", 0),
                "intensity_sum": obj.get("intensity_sum", 0),
            }
            for i, (key, _) in enumerate(_EXPORT_COLUMNS, 1):
                c = ws_obj.cell(row=row, column=i, value=values.get(key))
                c.font = data_font
            row += 1

    if has_phys:
        ws_obj.cell(row=1, column=4, value=f"Area ({px_unit}²)")
        ws_obj.cell(row=1, column=6, value=f"Perimeter ({px_unit})")
        ws_obj.cell(row=1, column=8, value=f"Equiv. Diameter ({px_unit})")

    # Auto-size columns roughly
    for i in range(1, len(headers) + 1):
        col_letter = ws_obj.cell(row=1, column=i).column_letter
        ws_obj.column_dimensions[col_letter].width = max(12, len(headers[i - 1]) + 2)
    ws_obj.freeze_panes = "B2"

    # ----- Write to disk -----
    save_dir = None
    reason = ""
    if params.saveDirectory:
        save_dir = Path(params.saveDirectory).expanduser()
        reason = "user-specified"
    else:
        save_dir = Path.home() / "Downloads"
        reason = "default (~/Downloads)"
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise HTTPException(500, f"Could not create save directory: {exc}")

    base = f"segmentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_path = save_dir / f"{base}.xlsx"
    counter = 1
    while save_path.exists():
        save_path = save_dir / f"{base}_{counter}.xlsx"
        counter += 1
    try:
        wb.save(str(save_path))
    except Exception as exc:
        raise HTTPException(500, f"Failed to write Excel file: {exc}")

    return {
        "saved_path": str(save_path),
        "filename": save_path.name,
        "save_dir": str(save_dir),
        "save_dir_reason": reason,
        "num_images": len(params.entries),
        "num_objects": sum(len(e.objects) for e in params.entries),
        "has_physical_units": has_phys,
    }
