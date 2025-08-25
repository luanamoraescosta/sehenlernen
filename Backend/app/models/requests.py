from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal


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
    selected_images: List[int] = Field(default_factory=list)
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


# ---- Haralick extraction ----
class HaralickExtractRequest(BaseModel):
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


# ---- LBP extraction ----
class LBPRequest(BaseModel):
    image_indices: List[int] = Field(default_factory=list)
    use_all_images: bool = False
    radius: int = 1
    num_neighbors: int = 8
    method: str = "uniform"   # allowed: default | ror | uniform | var
    normalize: bool = True


# ---- NEW: Contour extraction ----
class ContourRequest(BaseModel):
    """
    Request model for contour extraction.
    - image_index: index of the image in the current dataset
    - mode: Contour retrieval mode (cv2.RETR_EXTERNAL, RETR_LIST, RETR_TREE, RETR_CCOMP)
    - method: Approximation method (cv2.CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE)
    """
    image_index: int
    mode: Literal["RETR_EXTERNAL", "RETR_LIST", "RETR_TREE", "RETR_CCOMP"] = "RETR_EXTERNAL"
    method: Literal["CHAIN_APPROX_NONE", "CHAIN_APPROX_SIMPLE"] = "CHAIN_APPROX_SIMPLE"
    min_area: Optional[int] = 10  # filter out tiny contours
    return_bounding_boxes: bool = True
    return_hierarchy: bool = False
