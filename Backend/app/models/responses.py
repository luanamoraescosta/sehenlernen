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
