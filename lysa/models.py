"""
Pydantic models for all API request/response schemas.
"""

from typing import Optional, List
from pydantic import BaseModel


# --- Image Adjustments ---

class AdjustmentParams(BaseModel):
    brightness: float = 0.0        # -100 to 100
    contrast: float = 1.0          # 0.1 to 3.0
    gamma: float = 1.0             # 0.1 to 5.0
    invert: bool = False
    channel: Optional[str] = None  # "red", "green", "blue", "gray", or None for all


# --- Region of Interest ---

class ROIParams(BaseModel):
    x: int
    y: int
    width: int
    height: int


class ShapedROIParams(BaseModel):
    """
    Generic ROI request supporting rectangle, ellipse, or polygon shapes.

    - shape == 'rect'    → uses x, y, width, height
    - shape == 'ellipse' → uses x, y, width, height as the bounding box
    - shape == 'polygon' → uses points (list of [x, y])
    """
    shape: str = "rect"            # 'rect', 'ellipse', 'polygon'
    x: Optional[int] = None
    y: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    points: Optional[List[List[float]]] = None
    pixel_size: Optional[float] = None   # µm/px (or whatever unit)
    pixel_unit: Optional[str] = None     # e.g. 'µm'


class PointStatsParams(BaseModel):
    """Multi-point sampling: report intensity + physical coordinates per point."""
    points: List[List[float]]            # [[x, y], ...]
    radius: int = 0                       # 0 = single pixel, >0 = mean over disk
    pixel_size: Optional[float] = None
    pixel_unit: Optional[str] = None


# --- Line Profile ---

class LineProfileParams(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    line_width: int = 1


class PolylineProfileParams(BaseModel):
    """Profile along a multi-segment polyline with adjustable width."""
    points: List[List[int]]       # [[x1,y1], [x2,y2], ...] — at least 2 points
    line_width: int = 1           # 1–100: band width for averaging perpendicular pixels


# --- Thresholding / Segmentation ---

class ThresholdParams(BaseModel):
    method: str = "otsu"           # otsu, manual, adaptive
    value: Optional[float] = None  # for manual threshold
    block_size: int = 35           # for adaptive


# --- Measurements ---

class MeasurementParams(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    pixel_size: float = 1.0        # microns per pixel
    pixel_unit: str = "px"


# --- Folder / File Loading ---

class FolderLoadParams(BaseModel):
    path: str
    pattern: Optional[str] = None  # e.g. "*ch00*" to filter


class LifLoadParams(BaseModel):
    path: str
    image_indices: Optional[List[int]] = None  # which images to load (None = all)
    name_pattern: Optional[str] = None         # fnmatch glob applied to sub-image names
                                               # e.g. "*Merged*" to load only stitched composites


# --- Channel Merge ---

class ChannelSpec(BaseModel):
    image_id: str
    channel: str = "green"          # red, green, blue, cyan, magenta, yellow, white
    weight: float = 1.0             # 0.0–2.0
    # Optional per-channel adjustments — applied to the source channel BEFORE
    # weight + blending. Defaults are pass-through (identity).
    cl_min: float = 0.0             # contrast min, in 0..255 (display-array range)
    cl_max: float = 255.0           # contrast max, in 0..255
    gamma: float = 1.0              # > 0; 1.0 = no gamma


class MergeParams(BaseModel):
    """Legacy 2-image merge (still supported for backward compat)."""
    image_id_1: str
    image_id_2: str
    channel_1: str = "green"
    channel_2: str = "red"
    blend_mode: str = "additive"
    weight_1: float = 1.0
    weight_2: float = 1.0
    name: Optional[str] = None


class MergeNParams(BaseModel):
    """N-channel merge."""
    channels: List[ChannelSpec]
    blend_mode: str = "additive"   # additive, max, average
    name: Optional[str] = None


# --- Image Transformations ---

class RotateParams(BaseModel):
    angle: float = 90.0          # degrees, counter-clockwise
    expand: bool = True          # expand canvas to fit rotated image


class CropParams(BaseModel):
    x: int
    y: int
    width: int
    height: int


class AngledCropParams(BaseModel):
    """Crop using 4 corner points of a rotated rectangle, then straighten."""
    corners: List[List[float]]   # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    save_to_disk: bool = False   # if True, save next to the original file


# --- Profile Export ---

class ProfileEntry(BaseModel):
    """One line-profile measurement."""
    imageId: str
    imageName: str
    profiles: dict                # {channelName: [values...]}
    distances: List[float]        # cumulative pixel distance
    lineWidth: int
    lengthPixels: float
    numPoints: int
    segments: List[float]
    polylinePoints: Optional[List[List[float]]] = None  # [[x,y], ...]


class ProfileExportParams(BaseModel):
    """Full export request for line profiles."""
    entries: List[ProfileEntry]
    pixelSize: Optional[float] = None
    pixelSizeUnit: Optional[str] = None
    # Per-image adjustments snapshot: {imageId: {clMin, clMax, gamma, ...}}
    adjustments: Optional[dict] = None
    # Where to save (next to original image, or fallback)
    saveDirectory: Optional[str] = None


# --- ROI Measurement Export ---

class ROIMeasurementEntry(BaseModel):
    """One completed ROI measurement (rect / ellipse / polygon)."""
    id: str
    imageId: str
    imageName: str
    shape: str                               # 'rect' | 'ellipse' | 'polygon'
    rect: Optional[dict] = None              # {x, y, width, height} for rect/ellipse
    points: Optional[List[List[float]]] = None  # for polygon
    stats: dict                              # server stats response
    label: Optional[str] = None


class ROIMeasurementExportParams(BaseModel):
    """Full export request for ROI measurements."""
    entries: List[ROIMeasurementEntry]
    pixelSize: Optional[float] = None
    pixelSizeUnit: Optional[str] = None
    adjustments: Optional[dict] = None
    saveDirectory: Optional[str] = None


# --- Point Set Export ---

class PointSetEntry(BaseModel):
    """One saved multi-point measurement set."""
    id: str
    imageId: str
    imageName: str
    label: Optional[str] = None
    points: List[List[float]]           # [[x, y], ...]
    stats: Optional[dict] = None        # server point-stats response snapshot


class PointSetExportParams(BaseModel):
    """Full export request for point sets."""
    entries: List[PointSetEntry]
    pixelSize: Optional[float] = None
    pixelSizeUnit: Optional[str] = None
    adjustments: Optional[dict] = None
    saveDirectory: Optional[str] = None


# --- Measurement (angle + freehand curve) Export ---

class MeasurementEntry(BaseModel):
    """One completed angle or freehand curve measurement."""
    id: str
    imageId: str
    imageName: Optional[str] = None
    type: str                                    # 'angle' | 'curve'
    label: Optional[str] = None
    # Angle fields
    vertex: Optional[List[float]] = None         # [x, y]
    arm1: Optional[List[float]] = None           # [x, y]
    arm2: Optional[List[float]] = None           # [x, y]
    angle_degrees: Optional[float] = None
    arm1_length_px: Optional[float] = None
    arm2_length_px: Optional[float] = None
    # Curve fields
    points: Optional[List[List[float]]] = None   # [[x, y], ...]
    length_px: Optional[float] = None
    profile: Optional[dict] = None               # polyline-profile response snapshot


class MeasurementExportParams(BaseModel):
    """Full export request for angle/curve measurements."""
    entries: List[MeasurementEntry]
    pixelSize: Optional[float] = None
    pixelSizeUnit: Optional[str] = None
    saveDirectory: Optional[str] = None


# --- ROI Statistics Export (histograms + scatter charts) ---

class ROIStatisticsExportParams(BaseModel):
    """Full export request for extended ROI statistics (histograms + charts)."""
    entries: List[ROIMeasurementEntry]
    pixelSize: Optional[float] = None
    pixelSizeUnit: Optional[str] = None
    saveDirectory: Optional[str] = None


# --- Annotations (non-measurement overlays) ---

class AnnotationImageGroup(BaseModel):
    """Annotations for a single image, with optional path hint."""
    imageName: Optional[str] = None
    imagePath: Optional[str] = None
    entries: List[dict]                   # free-form annotation dicts


class AnnotationSaveParams(BaseModel):
    """Save annotations as JSON sidecar files."""
    annotations: dict                     # {image_id: AnnotationImageGroup-like dict}
    saveDirectory: Optional[str] = None


class AnnotationLoadParams(BaseModel):
    """Load annotation sidecars for the given image ids."""
    image_ids: List[str]
