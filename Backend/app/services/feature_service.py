import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage.transform import resize as sk_resize
import cv2
import pandas as pd
from typing import Any, Optional, List, Dict, Tuple
from fastapi import UploadFile, HTTPException
from io import BytesIO

from app.services.data_service import load_image, get_all_image_ids, metadata_df, image_id_col

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
    orientations: Optional[int] = None,
    pixels_per_cell: Optional[List[int]] = None,
    cells_per_block: Optional[List[int]] = None,
    resize_width: Optional[int] = None,
    resize_height: Optional[int] = None,
    visualize: Optional[bool] = None,
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
      - "FAST": returns list of keypoint coordinates

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
        # <<< FIXED LINE >>>
        pil_kp = Image.fromarray(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        pil_kp.save(buf, format='PNG')
        viz_b64 = base64.b64encode(buf.getvalue()).decode()

    elif method == "FAST":
        img_gray = img.convert('L')
        arr = np.array(img_gray)
        kp = corner_fast(arr, threshold=30, nonmax_suppression=True)
        features = [list(point) for point in kp]
        keypoints = [cv2.KeyPoint(float(p[1]), float(p[0]), 1) for p in kp]
        img_kp = cv2.drawKeypoints(arr, keypoints, None, color=(0, 255, 0))
        pil_kp = Image.fromarray(img_kp)
        buf = io.BytesIO()
        pil_kp.save(buf, format='PNG')
        viz_b64 = base64.b64encode(buf.getvalue()).decode()

    return features, viz_b64

# ----------------------------------------------------------------------
# SIFT extraction
# ----------------------------------------------------------------------

def _get_sift_detector():
    """
    Return a callable that creates a SIFT detector.
    Tries the three common ways OpenCV exposes SIFT.
    Raises a RuntimeError with a clear message if none are available.
    """
    # 1️⃣ New OpenCV ≥4.4 (SIFT moved to the main module)
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create

    # 2️⃣ OpenCV‑contrib (xfeatures2d) – older builds
    if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "SIFT_create"):
        return cv2.xfeatures2d.SIFT_create

    # 3️⃣ Fallback to ORB (free) – optional, you can just raise
    raise RuntimeError(
        "SIFT is not available in the installed OpenCV package. "
        "Install `opencv-contrib-python` (or a build that includes SIFT) "
        "or use the ORB fallback."
    )
def extract_sift_service(
    image_index: Optional[int] = None,
    all_images: bool = False,
    image_indices: Optional[List[int]] = None,
    resize: Optional[int] = 256,          # size to which the image is resized before detection
) -> Tuple[List[List[float]], Optional[bytes]]:
    """
    Run OpenCV SIFT on one or more images.
    Returns (features, visualisation_png_bytes_or_None).
    """
    # ---------- decide which images to process ----------
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

    # ---------- get a SIFT detector (will raise a clear error if missing) ----------
    try:
        sift_ctor = _get_sift_detector()          # helper defined earlier in the file
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    sift = sift_ctor()

    all_features: List[List[float]] = []
    viz_png: Optional[bytes] = None

    for idx in indices:
        img_bytes = load_image(ids[idx])
        # Decode to grayscale (SIFT works on a single channel)
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        if resize:
            img = cv2.resize(img, (resize, resize))

        kp, des = sift.detectAndCompute(img, None)
        if des is not None:
            all_features.extend(des.tolist())

        # Visualisation only when a *single* image is requested
        if len(indices) == 1:
            # Draw key‑points on the image (still BGR)
            img_kp = cv2.drawKeypoints(
                img,
                kp,
                None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                color=(0, 255, 0),
            )
            # Convert BGR → RGB so Pillow shows the correct colours
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
    all_matrices: List[List[List[float]]] = []  # Will collect ALL matrices

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

        # Encode PNG for the UI
        pil = Image.fromarray(edges)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        edge_imgs_b64.append(base64.b64encode(buf.getvalue()).decode())

        # Collect ALL matrices (not just the first one)
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
# NEW: Contour Extraction
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
        pts = c.squeeze().tolist()
        kept_contours.append(c)  # keep cv2 contour for drawing later
        areas.append(area)
        if return_bounding_boxes:
            x, y, w, h = cv2.boundingRect(c)
            bounding_boxes.append([int(x), int(y), int(w), int(h)])

    # Build results as plain lists
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
