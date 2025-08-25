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
    """
    Generic shape/feature extraction request.

    When method == "HOG", the optional parameters below (orientations, pixels_per_cell,
    cells_per_block, resize_* and visualize) will be used if provided; otherwise
    backend defaults are applied.

    For other methods ("SIFT", "FAST"), these optional fields are ignored.
    """
    method: Literal["HOG", "SIFT", "FAST"]
    image_index: int

    # --- Optional HOG parameters (used only when method == "HOG") ---
    orientations: Optional[int] = Field(
        default=None, ge=1, le=32, description="Number of orientation bins (default: 9)"
    )
    pixels_per_cell: Optional[List[int]] = Field(
        default=None, min_items=2, max_items=2,
        description="Cell size [px_h, px_w] (default: [8, 8])"
    )
    cells_per_block: Optional[List[int]] = Field(
        default=None, min_items=2, max_items=2,
        description="Block size in cells [cells_y, cells_x] (default: [2, 2])"
    )
    resize_width: Optional[int] = Field(
        default=None, ge=8, le=4096, description="Optional pre-extraction resize width"
    )
    resize_height: Optional[int] = Field(
        default=None, ge=8, le=4096, description="Optional pre-extraction resize height"
    )
    visualize: Optional[bool] = Field(
        default=None, description="Return HOG visualization image (default: True)"
    )


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
