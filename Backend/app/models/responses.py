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


class SimilaritySearchResponse(BaseModel):
    similar_images: List[Dict[str, Any]]  # [{"image_id": "...", "similarity_score": 0.95, "distance": 0.05}, ...]
    query_features: Optional[List[float]] = None  # Optional: return query image features
    computation_time: Optional[float] = None  # Optional: processing time in seconds


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


# ---- Classifier training & prediction ----
class ClassifierMetrics(BaseModel):
    accuracy: Optional[float] = None
    classification_report: Optional[Dict[str, Any]] = None
    confusion_matrix: Optional[List[List[int]]] = None


class PredictionResult(BaseModel):
    image_index: int
    image_id: Optional[str] = None
    prediction: str
    probabilities: Optional[Dict[str, float]] = None
    actual_label: Optional[str] = None


class ClassifierTrainingResponse(BaseModel):
    model_id: str
    feature_type: str
    classifier_type: str
    num_training_samples: int
    feature_vector_length: int
    train_metrics: ClassifierMetrics
    validation_metrics: Optional[ClassifierMetrics] = None
    test_predictions: Optional[List[PredictionResult]] = None
    test_metrics: Optional[ClassifierMetrics] = None


class ClassifierPredictionResponse(BaseModel):
    model_id: str
    predictions: List[PredictionResult]
    metrics: Optional[ClassifierMetrics] = None
