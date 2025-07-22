from pydantic import BaseModel
from typing import List, Dict, Any

class FilterRequest(BaseModel):
    filters: Dict[str, List[Any]]

class StratifiedRequest(BaseModel):
    target_col: str
    sample_size: int

class HistogramRequest(BaseModel):
    hist_type: str
    image_index: int
    all_images: bool
class ConfigureMetadataRequest(BaseModel):
    image_id_col: str
    col_mapping: Dict[str, str]
class KMeansRequest(BaseModel):
    n_clusters: int
    random_state: int
    selected_images: List[int] = []  
    use_all_images: bool = False

class ShapeRequest(BaseModel):
    method: str
    image_index: int

class StatsRequest(BaseModel):
    data: Dict[str, Any]

class VisualizationRequest(BaseModel):
    data: Dict[str, Any]
