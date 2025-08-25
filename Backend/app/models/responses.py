from pydantic import BaseModel
from typing import List, Any, Optional, Dict


class UploadImagesResponse(BaseModel):
    image_ids: List[str]


class UploadMetadataResponse(BaseModel):
    columns: List[str]


class ConfigureMetadataResponse(BaseModel):
    message: str


class FilterSamplingResponse(BaseModel):
    sampled_ids: List[str]


class StratifiedSamplingResponse(BaseModel):
    sampled_ids: List[str]


class HistogramResponse(BaseModel):
    histograms: List[str]  # Base64-encoded PNGs


class KMeansResponse(BaseModel):
    plot: str  # Base64-encoded PNG
    assignments: List[int]


class ShapeFeaturesResponse(BaseModel):
    features: List[Any]
    visualization: Optional[str]


class HaralickResponse(BaseModel):
    labels: List[Any]
    predictions: List[Any]


class CooccurrenceResponse(BaseModel):
    features: List[float]


class StatsAnalysisResponse(BaseModel):
    status: str
    input: Dict[str, Any]


class VisualizationResponse(BaseModel):
    status: str
    data: Dict[str, Any]


# ---- NEW: Contour Extraction ----
class ContourResponse(BaseModel):
    """
    Response model for contour extraction.
    - contours: list of point sets [[(x, y), ...], ...]
    - bounding_boxes: optional list of bounding boxes [x, y, w, h]
    - areas: optional list of contour areas
    - hierarchy: optional contour hierarchy info (OpenCV format)
    - visualization: base64-encoded PNG showing contours drawn on the image
    """
    contours: List[List[List[int]]]
    bounding_boxes: Optional[List[List[int]]] = None
    areas: Optional[List[float]] = None
    hierarchy: Optional[List[Any]] = None
    visualization: Optional[str] = None
