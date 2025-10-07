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

    method:
      - "HOG": optional parameters (below) are used if provided; otherwise service defaults.
      - "SIFT": ignores the optional HOG/FAST parameters.
      - "FAST": uses the optional FAST parameters (threshold, nonmax, type).

    Notes:
      * fast_type is only relevant when the backend uses OpenCV's FAST (TYPE_9_16, TYPE_7_12, TYPE_5_8).
        If the backend uses skimage.corner_fast, it's ignored.
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

    # --- Optional FAST parameters (used only when method == "FAST") ---
    fast_threshold: Optional[int] = Field(
        default=None, ge=0, le=255,
        description="FAST corner threshold (default: backend uses 30 if None)"
    )
    fast_nonmax: Optional[bool] = Field(
        default=None, description="Enable non-max suppression (default: True if None)"
    )
    fast_type: Optional[Literal["TYPE_9_16", "TYPE_7_12", "TYPE_5_8"]] = Field(
        default=None,
        description="OpenCV FAST type; ignored by skimage.corner_fast. Default: TYPE_9_16"
    )


class FeatureBaseRequest(BaseModel):
    """
    Common fields for any feature-extraction endpoint that works by image index.
    """
    image_index: Optional[int] = Field(
        None,
        description=(
            "Zero-based index of the image inside the dataset. Ignored if "
            "`all_images=True`."
        ),
    )
    all_images: bool = Field(
        False,
        description="If True, run the extraction on **all** uploaded images.",
    )
    image_indices: Optional[List[int]] = Field(
        None,
        description=(
            "Explicit list of indices. If supplied it overrides `image_index` "
            "and `all_images`."
        ),
    )


class SiftResponse(BaseModel):
    """Returned by the /sift endpoint."""
    features: List[List[float]]               # each SIFT descriptor = 128 floats
    visualization: Optional[str] = None       # base64-encoded PNG with key-points


class EdgeResponse(BaseModel):
    """Returned by the /edges endpoint."""
    edge_image: str                           # base64 PNG of the edge map
    edges_matrix: Optional[List[List[float]]] = None   # optional gradient matrix


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


# ---- Contour extraction ----
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


class SimilaritySearchRequest(BaseModel):
    """Request model for similarity search."""
    query_image_index: Optional[int] = Field(default=None, description="Index of uploaded image to use as query")
    query_image_base64: Optional[str] = Field(default=None, description="Base64 encoded query image (alternative to index)")
    feature_method: Literal["CNN", "HOG", "SIFT", "histogram"] = Field(default="CNN", description="Feature extraction method")
    distance_metric: Literal["cosine", "euclidean", "manhattan"] = Field(default="cosine", description="Distance metric for similarity")
    max_results: int = Field(default=10, ge=1, le=100, description="Maximum number of similar images to return")
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Similarity threshold (0-1, higher = more similar)")
    
    # Optional feature extraction parameters
    resize_dimensions: Optional[List[int]] = Field(default=[224, 224], description="Resize images to [width, height] before feature extraction")
    
    # HOG specific parameters
    hog_orientations: Optional[int] = Field(default=9, description="HOG orientations")
    hog_pixels_per_cell: Optional[List[int]] = Field(default=[8, 8], description="HOG pixels per cell")
    hog_cells_per_block: Optional[List[int]] = Field(default=[2, 2], description="HOG cells per block")
    
    # Histogram parameters
    hist_bins: Optional[int] = Field(default=64, description="Number of bins for color histogram")
    hist_channels: Optional[List[int]] = Field(default=[0, 1, 2], description="Color channels to use for histogram")


# ---- Classifier training & prediction ----
class LabeledSample(BaseModel):
    """Single training sample referencing an uploaded image by index."""
    image_index: int = Field(..., ge=0, description="Zero-based image index")
    label: str = Field(..., description="Target class label")


class EvaluationSample(BaseModel):
    """Sample for evaluation/prediction. Label optional for inference."""
    image_index: int = Field(..., ge=0, description="Zero-based image index")
    label: Optional[str] = Field(
        default=None,
        description="Optional ground-truth label (used for evaluating accuracy)"
    )


class HOGOptions(BaseModel):
    orientations: int = Field(default=9, ge=1, le=24)
    pixels_per_cell: List[int] = Field(default_factory=lambda: [8, 8], min_items=2, max_items=2)
    cells_per_block: List[int] = Field(default_factory=lambda: [2, 2], min_items=2, max_items=2)
    resize_width: int = Field(default=128, ge=8, le=2048)
    resize_height: int = Field(default=128, ge=8, le=2048)
    block_norm: Literal["L2-Hys", "L1", "L1-sqrt", "L2"] = "L2-Hys"


class LBPOptions(BaseModel):
    radius: int = Field(default=1, ge=1, le=16)
    num_neighbors: int = Field(default=8, ge=4, le=32)
    method: Literal["default", "ror", "uniform", "var"] = "uniform"
    normalize: bool = True


class ClassifierTrainingRequest(BaseModel):
    """
    Train an ML classifier using features derived from uploaded images.
    """
    feature_type: Literal["hog", "lbp", "embedding"]
    classifier_type: Literal["svm", "knn", "logistic"]
    training_samples: List[LabeledSample] = Field(..., min_items=1)
    test_samples: Optional[List[EvaluationSample]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    test_size: float = Field(default=0.2, ge=0.0, lt=1.0)
    random_state: Optional[int] = Field(default=42, description="Random seed for train/validation split")
    hog_options: Optional[HOGOptions] = None
    lbp_options: Optional[LBPOptions] = None
    embedding_model: Optional[str] = Field(default="resnet50", description="Pretrained model for embeddings")
    training_features: Optional[List[List[float]]] = Field(
        default=None,
        description="Optional precomputed feature vectors aligned with training_samples",
    )
    test_features: Optional[List[List[float]]] = Field(
        default=None,
        description="Optional precomputed feature vectors aligned with test_samples",
    )
    return_probabilities: bool = True


class ClassifierPredictionRequest(BaseModel):
    """
    Run inference using a previously trained classifier.
    """
    model_id: str = Field(..., description="Identifier returned by the training endpoint")
    samples: List[EvaluationSample] = Field(..., min_items=1)
    feature_vectors: Optional[List[List[float]]] = Field(
        default=None,
        description="Optional feature vectors aligned with samples (bypass image-based extraction)",
    )
    return_probabilities: bool = True
