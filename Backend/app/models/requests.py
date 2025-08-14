from pydantic import BaseModel
from typing import List, Dict, Any, Optional

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

class ReplaceImageRequest(BaseModel):
    image_id: str
    image_data_base64: str


# ---- Haralick extraction (new) ----
class HaralickExtractRequest(BaseModel):
    """
    Request body for computing Haralick (GLCM) features on one or more images.
    - image_indices: which uploaded images (by index in the current dataset) to process
    - levels: gray-level quantization (typical: 16/32/64/128/256)
    - distances: list of pixel distances for GLCM
    - angles: list of angles (in radians) for GLCM (e.g., 0, pi/4, pi/2, 3pi/4)
    - resize_width/height: optional resize to normalize & speed up computation
    - average_over_angles: if True, average each property across all (distance, angle)
    - properties: which GLCM properties to compute (skimage set)
    """
    image_indices: List[int]
    levels: int = 256
    distances: List[int] = [1, 2]
    angles: List[float] = [0.0, 0.785398, 1.570796, 2.356194]  # 0, pi/4, pi/2, 3pi/4
    resize_width: Optional[int] = None
    resize_height: Optional[int] = None
    average_over_angles: bool = True
    properties: List[str] = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM",
    ]
