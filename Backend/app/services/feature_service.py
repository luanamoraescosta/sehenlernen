import io
import base64
import uuid
import numpy as np
import matplotlib
matplotlib.use("Agg")  # <- non-GUI backend to avoid macOS NSWindow crash
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from skimage.transform import resize as sk_resize
import cv2
import pandas as pd
from typing import Any, Optional, List, Dict, Tuple, Set
from fastapi import UploadFile, HTTPException
from io import BytesIO

from app.services.data_service import load_image, get_all_image_ids, metadata_df, image_id_col

# Persistent in-memory storage for trained classifiers (cleared on process restart).
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}
from app.models.requests import (
    ClassifierTrainingRequest,
    ClassifierPredictionRequest,
)

# ---------------------------------------------------------
# Robust GLCM imports across scikit-image versions/builds
# ---------------------------------------------------------
try:
    from skimage.feature import greycomatrix, greycoprops  # type: ignore
except Exception:
    try:
        from skimage.feature import graycomatrix as greycomatrix, graycoprops as greycoprops  # type: ignore
    except Exception:
        try:
            from skimage.feature.texture import greycomatrix, greycoprops  # type: ignore
        except Exception:
            from skimage.feature.texture import graycomatrix as greycomatrix, graycoprops as greycoprops  # type: ignore

from skimage.feature import hog, corner_fast
try:
    from skimage.feature import local_binary_pattern
except Exception:
    from skimage.feature.texture import local_binary_pattern  # type: ignore


# --------------------------
# Histograms
# --------------------------
def generate_histogram_service(hist_type: str, image_index: int, all_images: bool) -> list[str]:
    """
    Generate color or grayscale histograms; returns a list of base64 PNGs.
    """
    img_ids = get_all_image_ids() if all_images else get_all_image_ids()[image_index:image_index+1]
    b64_list = []

    for img_id in img_ids:
        img_bytes = load_image(img_id)
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue

        plt.figure(figsize=(6, 4))
        if hist_type == "Black and White":
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            plt.hist(gray_img.ravel(), 256, [0, 256], color='black', alpha=0.7)
            plt.title("Grayscale Histogram")
        else:
            colors = ('b', 'g', 'r')
            for i, color in enumerate(colors):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                plt.plot(hist, color=color, alpha=0.7)
            plt.xlim([0, 256])
            plt.title("Color Histogram")
            plt.legend(["Blue", "Green", "Red"])

        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        b64_list.append(b64)

    return b64_list


# --------------------------
# K-Means clustering
# --------------------------
def perform_kmeans_service(
    n_clusters: int,
    random_state: int,
    selected_images: list[int],
    use_all_images: bool
) -> tuple[str, list[int]]:
    """
    Cluster images using color-histogram features -> PCA(2) -> KMeans.
    Returns (plot_png_base64, assignments).
    """
    img_ids = get_all_image_ids()
    if not img_ids:
        raise Exception("No images available for clustering")

    if use_all_images or not selected_images:
        selected_ids = img_ids
        selected_indices = list(range(len(img_ids)))
    else:
        selected_indices = [i for i in selected_images if 0 <= i < len(img_ids)]
        selected_ids = [img_ids[i] for i in selected_indices]

    if not selected_ids:
        raise Exception("No valid images selected for clustering")

    # Extract simple color histograms as features
    features = []
    for img_id in selected_ids:
        img_bytes = load_image(img_id)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        r, g, b = img.split()
        hist_r = np.histogram(np.array(r), bins=32)[0]
        hist_g = np.histogram(np.array(g), bins=32)[0]
        hist_b = np.histogram(np.array(b), bins=32)[0]
        features.append(np.concatenate([hist_r, hist_g, hist_b]))

    X = np.vstack(features)
    scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(scaled)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(reduced)

    # Plot
    fig, ax = plt.subplots()
    for i in range(n_clusters):
        ax.scatter(reduced[labels == i, 0], reduced[labels == i, 1], label=f"Cluster {i}", alpha=0.7)
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='*', s=200, c='red')
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    plot_b64 = base64.b64encode(buf.getvalue()).decode()

    return plot_b64, labels.tolist()


# --------------------------
# Shape features (HOG / SIFT / FAST)
# --------------------------
def _as_tuple2(x: Optional[List[int]], default: Tuple[int, int]) -> Tuple[int, int]:
    """
    Convert a 2-length list to a tuple; otherwise return default.
    """
    if isinstance(x, (list, tuple)) and len(x) == 2:
        try:
            return int(x[0]), int(x[1])
        except Exception:
            return default
    return default


def extract_shape_service(
    method: str,
    image_index: int,
    # HOG options (optional; default matches previous behavior)
    orientations: Optional[int] = None,
    pixels_per_cell: Optional[List[int]] = None,
    cells_per_block: Optional[List[int]] = None,
    resize_width: Optional[int] = None,
    resize_height: Optional[int] = None,
    visualize: Optional[bool] = None,
    # FAST options (optional; used when method == "FAST")
    fast_threshold: Optional[int] = None,
    fast_nonmax: Optional[bool] = None,
    fast_type: Optional[str] = None,  # "TYPE_9_16" | "TYPE_7_12" | "TYPE_5_8"
) -> tuple[list[Any], Optional[str]]:
    """
    Extract shape/structure features and optional visualization for a single image.

    Supported methods:
      - "HOG": accepts optional params:
            orientations (int, default 9)
            pixels_per_cell [h,w] (default [8,8])
            cells_per_block [y,x] (default [2,2])
            resize_width / resize_height (pre-resize; defaults to 64x128 legacy if not provided)
            visualize (bool, default True)
      - "SIFT": returns descriptor vectors (or empty if none)
      - "FAST": returns list of [x, y] keypoint coordinates (optionally with OpenCV FAST)

    Returns: (features, visualization_base64 or None)
    """
    img_ids = get_all_image_ids()
    if not img_ids:
        return [], None
    if image_index < 0 or image_index >= len(img_ids):
        raise IndexError("image_index out of range")

    img_bytes = load_image(img_ids[image_index])
    img = Image.open(io.BytesIO(img_bytes))
    features: list[Any] = []
    viz_b64: Optional[str] = None

    if method == "HOG":
        # Defaults (preserve previous behavior if not provided)
        orientations_val = int(orientations) if orientations is not None else 9
        ppc = _as_tuple2(pixels_per_cell, (8, 8))
        cpb = _as_tuple2(cells_per_block, (2, 2))
        visualize_val = True if visualize is None else bool(visualize)

        # Prepare grayscale and resize
        img_gray = img.convert('L')
        if resize_width and resize_height:
            img_resized = sk_resize(np.array(img_gray), (int(resize_height), int(resize_width)))
        else:
            # Legacy default used before: (128, 64)
            img_resized = sk_resize(np.array(img_gray), (128, 64))

        # skimage.hog expects float image typically in [0,1]; sk_resize returns float in [0,1]
        if visualize_val:
            fd, hog_image = hog(
                img_resized,
                orientations=orientations_val,
                pixels_per_cell=ppc,
                cells_per_block=cpb,
                visualize=True
            )
            features = fd.tolist()
            # Encode HOG visualization
            fig, ax = plt.subplots()
            ax.imshow(hog_image, cmap='gray')
            ax.axis('off')
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            viz_b64 = base64.b64encode(buf.getvalue()).decode()
        else:
            fd = hog(
                img_resized,
                orientations=orientations_val,
                pixels_per_cell=ppc,
                cells_per_block=cpb,
                visualize=False
            )
            features = fd.tolist()
            viz_b64 = None

    elif method == "SIFT":
        img_gray = img.convert('L')
        # sk_resize returns a float image in [0,1]; convert to uint8 for OpenCV
        arr = (sk_resize(img_gray, (256, 256)) * 255).astype('uint8')
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(arr, None)
        features = des.tolist() if des is not None else []
        img_kp = cv2.drawKeypoints(arr, kp, None, color=(0, 255, 0))
        pil_kp = Image.fromarray(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        pil_kp.save(buf, format='PNG')
        viz_b64 = base64.b64encode(buf.getvalue()).decode()

    elif method == "FAST":
        # --- Preferred: OpenCV FAST detector (with robust fallbacks) ---
        img_gray = img.convert('L')
        arr = np.array(img_gray)

        # Defaults
        th = int(fast_threshold) if fast_threshold is not None else 30
        nms = True if fast_nonmax is None else bool(fast_nonmax)
        ftype_label = (fast_type or "TYPE_9_16").upper()
        type_map = {
            "TYPE_9_16": getattr(cv2, "FAST_FEATURE_DETECTOR_TYPE_9_16", None),
            "TYPE_7_12": getattr(cv2, "FAST_FEATURE_DETECTOR_TYPE_7_12", None),
            "TYPE_5_8": getattr(cv2, "FAST_FEATURE_DETECTOR_TYPE_5_8", None),
        }
        type_value = type_map.get(ftype_label, None)

        detector = None
        try:
            # Some builds allow passing type in the factory; others only support threshold+nonmax
            if type_value is not None:
                try:
                    detector = cv2.FastFeatureDetector_create(threshold=th, nonmaxSuppression=nms, type=type_value)
                except TypeError:
                    detector = cv2.FastFeatureDetector_create(threshold=th, nonmaxSuppression=nms)
                    if hasattr(detector, "setType"):
                        detector.setType(type_value)
            else:
                detector = cv2.FastFeatureDetector_create(threshold=th, nonmaxSuppression=nms)
        except Exception:
            detector = None

        if detector is not None:
            kps = detector.detect(arr, None)  # list[cv2.KeyPoint]
            # features as [x, y]
            features = [[float(kp.pt[0]), float(kp.pt[1])] for kp in kps]

            # Visualization
            img_kp = cv2.drawKeypoints(
                arr, kps, None,
                color=(0, 255, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            pil_kp = Image.fromarray(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
            buf = io.BytesIO()
            pil_kp.save(buf, format='PNG')
            viz_b64 = base64.b64encode(buf.getvalue()).decode()
        else:
            # Fallback: skimage.corner_fast (coordinates only)
            pts = corner_fast(arr, threshold=th, nonmax_suppression=nms)
            features = [[float(p[1]), float(p[0])] for p in pts]  # convert (row, col) -> (x, y)
            keypoints = [cv2.KeyPoint(float(p[1]), float(p[0]), 1) for p in pts]
            img_kp = cv2.drawKeypoints(arr, keypoints, None, color=(0, 255, 0))
            pil_kp = Image.fromarray(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
            buf = io.BytesIO()
            pil_kp.save(buf, format='PNG')
            viz_b64 = base64.b64encode(buf.getvalue()).decode()

    return features, viz_b64


# ----------------------------------------------------------------------
# SIFT extraction (standalone utility)
# ----------------------------------------------------------------------
def _get_sift_detector():
    """
    Return a callable that creates a SIFT detector.
    Tries the common ways OpenCV exposes SIFT.
    """
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create
    if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "SIFT_create"):
        return cv2.xfeatures2d.SIFT_create
    raise RuntimeError(
        "SIFT is not available in the installed OpenCV package. "
        "Install `opencv-contrib-python` (or a build that includes SIFT)."
    )


def extract_sift_service(
    image_index: Optional[int] = None,
    all_images: bool = False,
    image_indices: Optional[List[int]] = None,
    resize: Optional[int] = 256,
) -> Tuple[List[List[float]], Optional[bytes]]:
    """
    Run OpenCV SIFT on one or more images.
    Returns (features, visualisation_png_bytes_or_None).
    """
    ids = get_all_image_ids()
    if all_images:
        indices = list(range(len(ids)))
    elif image_indices is not None:
        indices = [i for i in image_indices if 0 <= i < len(ids)]
    elif image_index is not None:
        if not (0 <= image_index < len(ids)):
            raise IndexError("image_index out of range")
        indices = [image_index]
    else:
        raise ValueError("Provide image_index, image_indices or all_images=True")

    try:
        sift_ctor = _get_sift_detector()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    sift = sift_ctor()

    all_features: List[List[float]] = []
    viz_png: Optional[bytes] = None

    for idx in indices:
        img_bytes = load_image(ids[idx])
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        if resize:
            img = cv2.resize(img, (resize, resize))

        kp, des = sift.detectAndCompute(img, None)
        if des is not None:
            all_features.extend(des.tolist())

        if len(indices) == 1:
            img_kp = cv2.drawKeypoints(
                img,
                kp,
                None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                color=(0, 255, 0),
            )
            pil_kp = Image.fromarray(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
            buf = io.BytesIO()
            pil_kp.save(buf, format="PNG")
            viz_png = buf.getvalue()

    return all_features, viz_png


# ----------------------------------------------------------------------
# Edge detection (Canny or Sobel)
# ----------------------------------------------------------------------
def extract_edges_service(
    image_index: Optional[int] = None,
    all_images: bool = False,
    image_indices: Optional[List[int]] = None,
    method: str = "canny",
    low_thresh: int = 100,
    high_thresh: int = 200,
    sobel_ksize: int = 3,
) -> Tuple[List[str], List[List[List[float]]]]:
    """
    Apply edge detection (Canny or Sobel) on one or more images.

    Returns
    -------
    edge_images_b64 : list[str]   # base64 PNG for each processed image
    all_matrices    : list[list[list[float]]]  # ALL gradient matrices (one per image)
    """
    ids = get_all_image_ids()
    if all_images:
        indices = list(range(len(ids)))
    elif image_indices is not None:
        indices = [i for i in image_indices if 0 <= i < len(ids)]
    elif image_index is not None:
        if not (0 <= image_index < len(ids)):
            raise IndexError("image_index out of range")
        indices = [image_index]
    else:
        raise ValueError("Provide image_index, image_indices or all_images=True")

    edge_imgs_b64: List[str] = []
    all_matrices: List[List[List[float]]] = []

    for idx in indices:
        img_bytes = load_image(ids[idx])
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        if method.lower() == "canny":
            edges = cv2.Canny(img, low_thresh, high_thresh)
        elif method.lower() == "sobel":
            grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
            grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
            magnitude = cv2.magnitude(grad_x, grad_y)
            edges = np.uint8(np.clip(magnitude, 0, 255))
        else:
            raise ValueError("method must be 'canny' or 'sobel'")

        pil = Image.fromarray(edges)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        edge_imgs_b64.append(base64.b64encode(buf.getvalue()).decode())

        all_matrices.append(edges.astype(float).tolist())

    return edge_imgs_b64, all_matrices


# --------------------------
# Legacy Haralick
# --------------------------
async def extract_haralick_service(train_images: list, train_labels: UploadFile, test_images: list) -> tuple[list, list]:
    X_train = []
    for f in train_images:
        data = await f.read()
        img = Image.open(io.BytesIO(data)).convert('L')
        arr = np.array(img)
        gm = greycomatrix(arr, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        props = greycoprops(gm, 'contrast')[0, 0]
        X_train.append([props])
    labels_df = pd.read_csv(train_labels.file)
    y_train = labels_df.iloc[:, 0].tolist()
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    X_test = []
    for f in test_images:
        data = await f.read()
        img = Image.open(io.BytesIO(data)).convert('L')
        arr = np.array(img)
        gm = greycomatrix(arr, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        props = greycoprops(gm, 'contrast')[0, 0]
        X_test.append([props])
    preds = clf.predict(X_test).tolist()
    return y_train, preds


# --------------------------
# Co-occurrence features
# --------------------------
def extract_cooccurrence_service(image_index: int) -> list[float]:
    img_ids = get_all_image_ids()
    img_bytes = load_image(img_ids[image_index])
    img = Image.open(io.BytesIO(img_bytes)).convert('L')
    arr = np.array(img)
    gm = greycomatrix(arr, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                      levels=256, symmetric=True, normed=True)
    features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        feat = greycoprops(gm, prop).mean()
        features.append(float(feat))
    return features


# --------------------------
# Haralick extraction (table)
# --------------------------
def _to_grayscale_uint(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        gray = (0.2989 * r + 0.5870 * g + 0.1140 * b)
    else:
        raise ValueError("Unsupported image shape for grayscale conversion")
    return np.clip(gray, 0, 255).astype(np.uint8)


def _quantize(gray: np.ndarray, levels: int) -> np.ndarray:
    if levels == 256:
        return gray
    factor = 256.0 / float(levels)
    q = np.floor(gray / factor).astype(np.uint8)
    return q


def extract_haralick_features_service(
    image_indices: List[int],
    levels: int,
    distances: List[int],
    angles: List[float],
    resize_width: Optional[int],
    resize_height: Optional[int],
    average_over_angles: bool,
    properties: List[str],
) -> Dict[str, Any]:
    valid_props = {"contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"}
    props = [p for p in properties if p in valid_props]
    if not props:
        raise ValueError("No valid Haralick properties requested.")

    all_ids = get_all_image_ids()
    rows: List[Dict[str, Any]] = []

    for idx in image_indices:
        if idx < 0 or idx >= len(all_ids):
            raise IndexError(f"image_index {idx} out of range.")
        image_id = all_ids[idx]
        img_bytes = load_image(image_id)
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        arr = np.array(pil)
        if resize_width and resize_height:
            arr = (sk_resize(arr, (resize_height, resize_width), anti_aliasing=True) * 255).astype(np.uint8)
        gray = _to_grayscale_uint(arr)
        gray_q = _quantize(gray, levels)
        glcm = greycomatrix(
            gray_q,
            distances=distances if distances else [1],
            angles=angles if angles else [0.0],
            levels=levels,
            symmetric=True,
            normed=True,
        )
        if average_over_angles:
            feat_values = []
            for p in props:
                val = greycoprops(glcm, p).mean()
                feat_values.append(float(val))
            rows.append({"image_id": image_id, "vector": feat_values})
        else:
            feat_map: Dict[str, float] = {}
            for p in props:
                vals = greycoprops(glcm, p)
                for di, d in enumerate(distances if distances else [1]):
                    for ai, a in enumerate(angles if angles else [0.0]):
                        key = f"{p}_d{d}_a{round(a, 6)}"
                        feat_map[key] = float(vals[di, ai])
            ordered_cols = sorted(feat_map.keys())
            rows.append({"image_id": image_id, "vector": [feat_map[k] for k in ordered_cols], "columns": ordered_cols})

    if average_over_angles:
        columns = ["image_id"] + props
        matrix = [[r["image_id"], *r["vector"]] for r in rows]
    else:
        cols = rows[0]["columns"] if rows else []
        columns = ["image_id"] + cols
        matrix = [[r["image_id"], *r["vector"]] for r in rows]

    return {"columns": columns, "rows": matrix}


# --------------------------
# LBP
# --------------------------
def _lbp_bins_edges(method: str, P: int) -> np.ndarray:
    if method == "uniform":
        n_bins = P + 2
        return np.arange(-0.5, n_bins + 0.5, 1.0)
    else:
        max_label = (1 << P) - 1
        return np.arange(-0.5, max_label + 1.5, 1.0)


def _normalize_hist(h: np.ndarray) -> np.ndarray:
    s = float(h.sum())
    if s <= 0:
        return h.astype(float)
    return (h / s).astype(float)


def compute_lbp_service(
    image_indices: List[int],
    use_all_images: bool,
    radius: int,
    num_neighbors: int,
    method: str,
    normalize: bool,
) -> Dict[str, Any]:
    if radius < 1:
        raise ValueError("radius must be >= 1")
    if num_neighbors < 4:
        raise ValueError("num_neighbors must be >= 4")
    if method not in {"default", "ror", "uniform", "var"}:
        raise ValueError("method must be one of: default, ror, uniform, var")

    all_ids = get_all_image_ids()
    if not all_ids:
        raise ValueError("No images available. Please upload images first.")

    if use_all_images:
        indices = list(range(len(all_ids)))
    else:
        indices = [i for i in image_indices if 0 <= i < len(all_ids)]

    if not indices:
        raise ValueError("No valid images selected.")

    bin_edges = _lbp_bins_edges(method, num_neighbors)
    n_bins = len(bin_edges) - 1

    if len(indices) == 1:
        idx = indices[0]
        image_id = all_ids[idx]
        img_bytes = load_image(image_id)
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        arr = np.array(pil)
        gray = _to_grayscale_uint(arr)

        lbp = local_binary_pattern(gray, P=num_neighbors, R=radius, method=method)
        hist, _ = np.histogram(lbp.ravel(), bins=bin_edges)
        hist = _normalize_hist(hist) if normalize else hist.astype(float)

        lbp_min, lbp_max = float(np.min(lbp)), float(np.max(lbp))
        if lbp_max > lbp_min:
            lbp_vis = ((lbp - lbp_min) / (lbp_max - lbp_min) * 255.0).astype(np.uint8)
        else:
            lbp_vis = np.zeros_like(lbp, dtype=np.uint8)

        lbp_img = Image.fromarray(lbp_vis)
        buf = io.BytesIO()
        lbp_img.save(buf, format="PNG")
        lbp_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "mode": "single",
            "image_id": image_id,
            "bins": list(range(n_bins)),
            "histogram": hist.tolist(),
            "lbp_image_b64": lbp_b64,
        }

    columns = ["image_id"] + [f"bin_{i}" for i in range(n_bins)]
    rows: List[List[Any]] = []

    for idx in indices:
        image_id = all_ids[idx]
        img_bytes = load_image(image_id)
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        arr = np.array(pil)
        gray = _to_grayscale_uint(arr)

        lbp = local_binary_pattern(gray, P=num_neighbors, R=radius, method=method)
        hist, _ = np.histogram(lbp.ravel(), bins=bin_edges)
        hist = _normalize_hist(hist) if normalize else hist.astype(float)

        rows.append([image_id, *hist.tolist()])

    return {"mode": "multi", "columns": columns, "rows": rows}


# --------------------------
# Contour Extraction
# --------------------------
def extract_contours_service(
    image_index: int,
    mode: str,
    method: str,
    min_area: int = 10,
    return_bounding_boxes: bool = True,
    return_hierarchy: bool = False,
) -> Dict[str, Any]:
    """
    Extract contours from a binary/grayscale image using OpenCV findContours.
    Returns contour point sets, areas, optional bounding boxes & hierarchy, and a PNG overlay.
    """
    mode_map = {
        "RETR_EXTERNAL": cv2.RETR_EXTERNAL,
        "RETR_LIST": cv2.RETR_LIST,
        "RETR_TREE": cv2.RETR_TREE,
        "RETR_CCOMP": cv2.RETR_CCOMP,
    }
    method_map = {
        "CHAIN_APPROX_NONE": cv2.CHAIN_APPROX_NONE,
        "CHAIN_APPROX_SIMPLE": cv2.CHAIN_APPROX_SIMPLE,
    }
    if mode not in mode_map or method not in method_map:
        raise ValueError(f"Invalid contour mode/method: {mode}, {method}")

    img_ids = get_all_image_ids()
    if not img_ids:
        raise ValueError("No images available.")
    if image_index < 0 or image_index >= len(img_ids):
        raise IndexError("image_index out of range")

    img_bytes = load_image(img_ids[image_index])
    arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise ValueError("Could not decode image.")

    # Simple fixed threshold to binary; can be enhanced later to Otsu/Adaptive if needed
    _, binary = cv2.threshold(arr, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, mode_map[mode], method_map[method])

    kept_contours = []
    bounding_boxes = []
    areas = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < (min_area or 0):
            continue
        kept_contours.append(c)  # keep cv2 contour for drawing later
        areas.append(area)
        if return_bounding_boxes:
            x, y, w, h = cv2.boundingRect(c)
            bounding_boxes.append([int(x), int(y), int(w), int(h)])

    # Build results as plain lists (x,y pairs)
    results_points = [kc.squeeze().tolist() for kc in kept_contours]

    # Make overlay that only shows kept contours (green)
    overlay = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if kept_contours:
        cv2.drawContours(overlay, kept_contours, -1, (0, 255, 0), 2)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(overlay_rgb).save(buf, format="PNG")
    viz_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "contours": results_points,
        "bounding_boxes": bounding_boxes if return_bounding_boxes else None,
        "areas": areas,
        "hierarchy": hierarchy.tolist() if return_hierarchy and hierarchy is not None else None,
        "visualization": viz_b64,
    }


# --------------------------
# Image Embedding Extraction
# --------------------------
def extract_embedding_service(
    image_indices: Optional[List[int]] = None,
    use_all_images: bool = False,
    model_name: str = "resnet50",
    layer: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract deep learning embeddings from images using pretrained models.
    
    Args:
        image_indices: List of image indices to process
        use_all_images: If True, process all images
        model_name: Name of the model to use ("resnet50", "resnet18", "mobilenet_v2")
        layer: Optional layer name for custom extraction (defaults to last feature layer)
    
    Returns:
        Dictionary containing embeddings and metadata
    """
    try:
        import torch
        import torchvision.models as models
        import torchvision.transforms as transforms
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="PyTorch is not installed. Please install torch and torchvision."
        )
    
    # Determine which images to process
    img_ids = get_all_image_ids()
    if use_all_images:
        indices_to_process = list(range(len(img_ids)))
    elif image_indices:
        indices_to_process = image_indices
    else:
        raise HTTPException(status_code=400, detail="Must specify image_indices or use_all_images=True")
    
    # Load pretrained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove the final classification layer to get embeddings
        model = torch.nn.Sequential(*list(model.children())[:-1])
        embedding_dim = 2048
    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        embedding_dim = 512
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # MobileNetV2 features are in model.features
        model = model.features
        embedding_dim = 1280
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model_name}")
    
    model = model.to(device)
    model.eval()
    
    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Extract embeddings
    embeddings_list = []
    image_ids_list = []
    
    with torch.no_grad():
        for idx in indices_to_process:
            if idx >= len(img_ids):
                continue
                
            img_id = img_ids[idx]
            img_bytes = load_image(img_id)
            
            # Convert bytes to PIL Image
            pil_img = Image.open(io.BytesIO(img_bytes))
            
            # Convert to RGB if needed
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            # Preprocess and get embedding
            img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
            embedding = model(img_tensor)
            
            # Flatten the embedding
            embedding = embedding.view(embedding.size(0), -1)
            embedding_np = embedding.cpu().numpy().flatten()
            
            embeddings_list.append(embedding_np.tolist())
            image_ids_list.append(img_id)
    
    return {
        "embeddings": embeddings_list,
        "image_ids": image_ids_list,
        "model_name": model_name,
        "embedding_dim": embedding_dim,
        "num_images": len(embeddings_list)
    }


# --------------------------
# Classifier Training & Prediction
# --------------------------
def _unique_preserve_order(values: List[int]) -> List[int]:
    seen: Set[int] = set()
    ordered: List[int] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            ordered.append(v)
    return ordered


def _coerce_to_serializable(data: Any) -> Any:
    if isinstance(data, (np.floating,)):
        return float(data)
    if isinstance(data, (np.integer,)):
        return int(data)
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, dict):
        return {k: _coerce_to_serializable(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_coerce_to_serializable(v) for v in data]
    return data


def _compute_hog_feature_vector(
    image_index: int,
    img_ids: List[str],
    orientations: int,
    pixels_per_cell: Tuple[int, int],
    cells_per_block: Tuple[int, int],
    resize_width: int,
    resize_height: int,
    block_norm: str,
) -> np.ndarray:
    if image_index < 0 or image_index >= len(img_ids):
        raise IndexError(f"image_index {image_index} out of range for HOG computation")
    img_bytes = load_image(img_ids[image_index])
    pil = Image.open(io.BytesIO(img_bytes)).convert("L")
    if resize_width and resize_height:
        pil = pil.resize((resize_width, resize_height), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    features = hog(
        arr,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
        transform_sqrt=True,
        feature_vector=True,
    )
    return features.astype(np.float32)


def _compute_lbp_feature_vector(
    image_index: int,
    img_ids: List[str],
    radius: int,
    num_neighbors: int,
    method: str,
    normalize: bool,
) -> np.ndarray:
    if image_index < 0 or image_index >= len(img_ids):
        raise IndexError(f"image_index {image_index} out of range for LBP computation")
    img_bytes = load_image(img_ids[image_index])
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(pil)
    gray = _to_grayscale_uint(arr)
    lbp = local_binary_pattern(gray, P=num_neighbors, R=radius, method=method)
    bin_edges = _lbp_bins_edges(method, num_neighbors)
    hist, _ = np.histogram(lbp.ravel(), bins=bin_edges)
    if normalize:
        hist = _normalize_hist(hist)
    return hist.astype(np.float32)


def _compute_features_for_indices(
    feature_type: str,
    indices: List[int],
    img_ids: List[str],
    hog_options: Dict[str, Any],
    lbp_options: Dict[str, Any],
    embedding_model: Optional[str],
) -> Dict[int, np.ndarray]:
    if not indices:
        return {}

    unique_indices = _unique_preserve_order(indices)
    features: Dict[int, np.ndarray] = {}

    if feature_type == "hog":
        orientations = int(hog_options.get("orientations", 9))
        pixels_per_cell = hog_options.get("pixels_per_cell", [8, 8])
        cells_per_block = hog_options.get("cells_per_block", [2, 2])
        resize_width = int(hog_options.get("resize_width", 128))
        resize_height = int(hog_options.get("resize_height", 128))
        block_norm = hog_options.get("block_norm", "L2-Hys")
        ppc = (int(pixels_per_cell[0]), int(pixels_per_cell[1]))
        cpb = (int(cells_per_block[0]), int(cells_per_block[1]))

        for idx in unique_indices:
            features[idx] = _compute_hog_feature_vector(
                image_index=idx,
                img_ids=img_ids,
                orientations=orientations,
                pixels_per_cell=ppc,
                cells_per_block=cpb,
                resize_width=resize_width,
                resize_height=resize_height,
                block_norm=block_norm,
            )

    elif feature_type == "lbp":
        radius = int(lbp_options.get("radius", 1))
        num_neighbors = int(lbp_options.get("num_neighbors", 8))
        method = lbp_options.get("method", "uniform")
        normalize = bool(lbp_options.get("normalize", True))

        for idx in unique_indices:
            features[idx] = _compute_lbp_feature_vector(
                image_index=idx,
                img_ids=img_ids,
                radius=radius,
                num_neighbors=num_neighbors,
                method=method,
                normalize=normalize,
            )

    elif feature_type == "embedding":
        model_name = embedding_model or "resnet50"
        embedding_result = extract_embedding_service(
            image_indices=unique_indices,
            use_all_images=False,
            model_name=model_name,
        )
        embeddings = embedding_result.get("embeddings", [])
        if len(embeddings) != len(unique_indices):
            raise ValueError("Failed to compute embeddings for all requested images.")
        for idx, vec in zip(unique_indices, embeddings):
            features[idx] = np.asarray(vec, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported feature_type '{feature_type}'.")

    return features


def _assemble_feature_matrix(indices: List[int], feature_lookup: Dict[int, np.ndarray]) -> np.ndarray:
    rows: List[np.ndarray] = []
    for idx in indices:
        vec = feature_lookup.get(idx)
        if vec is None:
            raise ValueError(f"Missing feature vector for image index {idx}")
        rows.append(np.asarray(vec, dtype=np.float32).ravel())

    if not rows:
        raise ValueError("No feature vectors available to assemble matrix.")

    return np.stack(rows, axis=0)


def _build_classifier(classifier_type: str, hyperparameters: Optional[Dict[str, Any]]) -> Any:
    params = dict(hyperparameters or {})
    if classifier_type == "svm":
        base = {"kernel": "rbf", "C": 1.0, "probability": True}
        base.update(params)
        return SVC(**base)
    if classifier_type == "knn":
        base = {"n_neighbors": 5}
        base.update(params)
        return KNeighborsClassifier(**base)
    if classifier_type == "logistic":
        base = {"max_iter": 1000}
        base.update(params)
        return LogisticRegression(**base)
    raise ValueError(f"Unsupported classifier_type '{classifier_type}'.")


def _compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    if not y_true:
        return {"accuracy": None, "classification_report": None, "confusion_matrix": None}

    metrics: Dict[str, Any] = {
        "accuracy": None,
        "classification_report": None,
        "confusion_matrix": None,
    }
    try:
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    except Exception:
        metrics["accuracy"] = None

    try:
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics["classification_report"] = _coerce_to_serializable(report)
    except Exception:
        metrics["classification_report"] = None

    try:
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    except Exception:
        metrics["confusion_matrix"] = None

    return metrics


def _sanitize_metrics(metrics: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not metrics:
        return None
    if all(value is None for value in metrics.values()):
        return None
    return metrics


def train_classifier_service(request: ClassifierTrainingRequest) -> Dict[str, Any]:
    img_ids = get_all_image_ids()
    if not img_ids:
        raise ValueError("No images available. Please upload images before training a classifier.")

    train_indices = [sample.image_index for sample in request.training_samples]
    if any(idx < 0 or idx >= len(img_ids) for idx in train_indices):
        raise IndexError("One or more training image indices are out of range.")

    unique_labels = {sample.label for sample in request.training_samples}
    if len(unique_labels) < 2:
        raise ValueError("At least two distinct labels are required for training.")

    feature_lookup: Dict[int, np.ndarray] = {}

    if request.training_features:
        if len(request.training_features) != len(request.training_samples):
            raise ValueError("training_features must align with training_samples.")
        for sample, vec in zip(request.training_samples, request.training_features):
            feature_lookup[sample.image_index] = np.asarray(vec, dtype=np.float32)

    if request.test_samples and request.test_features:
        if len(request.test_features) != len(request.test_samples):
            raise ValueError("test_features must align with test_samples.")
        for sample, vec in zip(request.test_samples, request.test_features):
            feature_lookup[sample.image_index] = np.asarray(vec, dtype=np.float32)

    indices_needed = list(train_indices)
    if request.test_samples:
        indices_needed.extend(sample.image_index for sample in request.test_samples)
    indices_needed = [idx for idx in indices_needed if idx not in feature_lookup]

    hog_opts = request.hog_options.dict() if request.hog_options else {}
    lbp_opts = request.lbp_options.dict() if request.lbp_options else {}

    computed = _compute_features_for_indices(
        feature_type=request.feature_type,
        indices=indices_needed,
        img_ids=img_ids,
        hog_options=hog_opts,
        lbp_options=lbp_opts,
        embedding_model=request.embedding_model,
    )
    feature_lookup.update(computed)

    X_train_full = _assemble_feature_matrix(train_indices, feature_lookup)
    y_train_full = np.asarray([sample.label for sample in request.training_samples])
    feature_vector_length = int(X_train_full.shape[1])

    X_fit = X_train_full
    y_fit = y_train_full
    X_val: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None

    if not request.test_samples and request.test_size > 0.0 and len(request.training_samples) >= 4:
        stratify = y_train_full if len(np.unique(y_train_full)) > 1 else None
        try:
            X_fit, X_val, y_fit, y_val = train_test_split(
                X_train_full,
                y_train_full,
                test_size=request.test_size,
                random_state=request.random_state,
                stratify=stratify,
            )
        except ValueError:
            X_fit, y_fit = X_train_full, y_train_full
            X_val = None
            y_val = None

    classifier = _build_classifier(request.classifier_type, request.hyperparameters)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])
    pipeline.fit(X_fit, y_fit)

    train_predictions = pipeline.predict(X_train_full)
    train_metrics = _compute_metrics(
        [str(label) for label in y_train_full],
        [str(pred) for pred in train_predictions],
    )

    validation_metrics: Optional[Dict[str, Any]] = None
    if X_val is not None and y_val is not None and len(y_val) > 0:
        val_predictions = pipeline.predict(X_val)
        validation_metrics = _compute_metrics(
            [str(label) for label in y_val],
            [str(pred) for pred in val_predictions],
        )

    test_predictions_output: Optional[List[Dict[str, Any]]] = None
    test_metrics: Optional[Dict[str, Any]] = None
    if request.test_samples:
        test_indices = [sample.image_index for sample in request.test_samples]
        if any(idx < 0 or idx >= len(img_ids) for idx in test_indices):
            raise IndexError("One or more test image indices are out of range.")
        X_test = _assemble_feature_matrix(test_indices, feature_lookup)
        test_predictions = pipeline.predict(X_test)

        prob_matrix: Optional[np.ndarray] = None
        class_labels: List[str] = []
        if request.return_probabilities and hasattr(pipeline, "predict_proba"):
            prob_matrix = pipeline.predict_proba(X_test)
            class_labels = [str(c) for c in pipeline.classes_]

        predictions: List[Dict[str, Any]] = []
        labeled_true: List[str] = []
        labeled_pred: List[str] = []
        for idx, sample in enumerate(request.test_samples):
            probabilities = None
            if prob_matrix is not None:
                probabilities = {
                    class_labels[j]: float(prob_matrix[idx][j])
                    for j in range(len(class_labels))
                }
            actual_label = sample.label
            if actual_label is not None:
                labeled_true.append(str(actual_label))
                labeled_pred.append(str(test_predictions[idx]))

            predictions.append({
                "image_index": sample.image_index,
                "image_id": img_ids[sample.image_index] if sample.image_index < len(img_ids) else None,
                "prediction": str(test_predictions[idx]),
                "probabilities": probabilities,
                "actual_label": str(actual_label) if actual_label is not None else None,
            })

        test_predictions_output = predictions
        test_metrics = _compute_metrics(labeled_true, labeled_pred) if labeled_true else None

    model_id = str(uuid.uuid4())
    MODEL_REGISTRY[model_id] = {
        "pipeline": pipeline,
        "feature_type": request.feature_type,
        "hog_options": hog_opts,
        "lbp_options": lbp_opts,
        "embedding_model": request.embedding_model,
        "feature_vector_length": feature_vector_length,
    }

    return {
        "model_id": model_id,
        "feature_type": request.feature_type,
        "classifier_type": request.classifier_type,
        "num_training_samples": len(request.training_samples),
        "feature_vector_length": feature_vector_length,
        "train_metrics": train_metrics,
        "validation_metrics": _sanitize_metrics(validation_metrics),
        "test_predictions": test_predictions_output,
        "test_metrics": _sanitize_metrics(test_metrics),
    }


def predict_classifier_service(request: ClassifierPredictionRequest) -> Dict[str, Any]:
    model_entry = MODEL_REGISTRY.get(request.model_id)
    if not model_entry:
        raise ValueError(f"Model '{request.model_id}' not found. Please train the model again.")

    pipeline: Pipeline = model_entry["pipeline"]
    feature_type: str = model_entry["feature_type"]
    hog_opts = model_entry.get("hog_options") or {}
    lbp_opts = model_entry.get("lbp_options") or {}
    embedding_model = model_entry.get("embedding_model")

    img_ids = get_all_image_ids()
    if not img_ids:
        raise ValueError("No images available. Please upload images before running predictions.")

    feature_lookup: Dict[int, np.ndarray] = {}
    if request.feature_vectors:
        if len(request.feature_vectors) != len(request.samples):
            raise ValueError("feature_vectors must align with samples.")
        for sample, vec in zip(request.samples, request.feature_vectors):
            feature_lookup[sample.image_index] = np.asarray(vec, dtype=np.float32)

    indices_needed = [
        sample.image_index for sample in request.samples
        if sample.image_index not in feature_lookup
    ]

    computed = _compute_features_for_indices(
        feature_type=feature_type,
        indices=indices_needed,
        img_ids=img_ids,
        hog_options=hog_opts,
        lbp_options=lbp_opts,
        embedding_model=embedding_model,
    )
    feature_lookup.update(computed)

    sample_indices = [sample.image_index for sample in request.samples]
    X = _assemble_feature_matrix(sample_indices, feature_lookup)

    predictions = pipeline.predict(X)

    prob_matrix: Optional[np.ndarray] = None
    class_labels: List[str] = []
    if request.return_probabilities and hasattr(pipeline, "predict_proba"):
        prob_matrix = pipeline.predict_proba(X)
        class_labels = [str(c) for c in pipeline.classes_]

    results: List[Dict[str, Any]] = []
    labeled_true: List[str] = []
    labeled_pred: List[str] = []

    for idx, sample in enumerate(request.samples):
        if sample.image_index < 0 or sample.image_index >= len(img_ids):
            raise IndexError(f"Image index {sample.image_index} is out of range.")

        probabilities = None
        if prob_matrix is not None:
            probabilities = {
                class_labels[j]: float(prob_matrix[idx][j])
                for j in range(len(class_labels))
            }

        actual_label = sample.label
        predicted_label = str(predictions[idx])

        if actual_label is not None:
            labeled_true.append(str(actual_label))
            labeled_pred.append(predicted_label)

        results.append({
            "image_index": sample.image_index,
            "image_id": img_ids[sample.image_index],
            "prediction": predicted_label,
            "probabilities": probabilities,
            "actual_label": str(actual_label) if actual_label is not None else None,
        })

    metrics = _compute_metrics(labeled_true, labeled_pred) if labeled_true else None

    return {
        "model_id": request.model_id,
        "predictions": results,
        "metrics": _sanitize_metrics(metrics),
    }
