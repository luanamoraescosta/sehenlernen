# Fronted/utils/api_client.py

import os
import requests
import base64
import streamlit as st
from io import BytesIO
from PIL import Image


def _get_base_url():
    return os.getenv("SEHEN_LERNEN_API_URL", "http://localhost:8000")


# -----------------------------
# Upload & Metadata
# -----------------------------
def upload_images(image_files):
    url = f"{_get_base_url()}/upload/images"
    files = [("files", (f.name, f.getvalue(), f.type)) for f in image_files]
    resp = requests.post(url, files=files)
    resp.raise_for_status()
    return resp.json().get("image_ids", [])


def upload_metadata_file(csv_file, delimiter, decimal_sep):
    url = f"{_get_base_url()}/upload/metadata"
    files = {"file": (csv_file.name, csv_file.getvalue(), csv_file.type)}
    data = {"delimiter": delimiter, "decimal_sep": decimal_sep}
    resp = requests.post(url, files=files, data=data)
    resp.raise_for_status()
    return resp.json().get("columns", [])


def configure_metadata(image_id_col, col_mapping):
    url = f"{_get_base_url()}/upload/metadata/configure"
    payload = {"image_id_col": image_id_col, "col_mapping": col_mapping}
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()


# -----------------------------
# Sampling
# -----------------------------
def filter_sampling(filter_values):
    url = f"{_get_base_url()}/sampling/filter"
    resp = requests.post(url, json={"filters": filter_values})
    resp.raise_for_status()
    return resp.json().get("sampled_ids", [])


def stratified_sampling(target_col, sample_size):
    url = f"{_get_base_url()}/sampling/stratified"
    payload = {"target_col": target_col, "sample_size": sample_size}
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    return resp.json().get("sampled_ids", [])


# -----------------------------
# Features
# -----------------------------
def generate_histogram(params):
    url = f"{_get_base_url()}/features/histogram"
    resp = requests.post(url, json=params)
    resp.raise_for_status()
    b64_list = resp.json().get("histograms", [])
    return [base64.b64decode(b) for b in b64_list]


def perform_kmeans(params):
    url = f"{_get_base_url()}/features/kmeans"
    resp = requests.post(url, json=params)
    try:
        resp.raise_for_status()
        data = resp.json()
        plot_bytes = base64.b64decode(data.get("plot", ""))
        assignments = data.get("assignments", [])
        return plot_bytes, assignments
    except requests.exceptions.HTTPError:
        if resp.status_code == 500:
            try:
                error_data = resp.json()
                error_detail = error_data.get("detail", "Internal server error")
            except Exception:
                error_detail = "Internal server error"
            st.error(f"Server error: {error_detail}")
        raise


def extract_shape_features(params):
    """
    Generic shape/feature call (HOG/SIFT/FAST).

    Pass a dict like:
      {
        "method": "HOG" | "SIFT" | "FAST",
        "image_index": int,

        # Optional (HOG only):
        "orientations": int,
        "pixels_per_cell": [h, w],
        "cells_per_block": [y, x],
        "resize_width": int,
        "resize_height": int,
        "visualize": bool,

        # Optional (FAST only; used by backend if provided):
        "fast_threshold": int,
        "fast_nonmax": bool,
        "fast_type": "TYPE_9_16" | "TYPE_7_12" | "TYPE_5_8"
      }
    """
    url = f"{_get_base_url()}/features/shape"
    resp = requests.post(url, json=params)
    resp.raise_for_status()
    data = resp.json()
    result = {"features": data.get("features", [])}
    if data.get("visualization"):
        result["visualization"] = base64.b64decode(data["visualization"])
    return result


def extract_haralick_texture(params):
    url = f"{_get_base_url()}/features/haralick"
    files = []
    for f in params.get("train_images", []):
        files.append(("train_images", (f.name, f.getvalue(), f.type)))
    for f in params.get("test_images", []):
        files.append(("test_images", (f.name, f.getvalue(), f.type)))
    train_labels = params.get("train_labels")
    files.append(("train_labels", (train_labels.name, train_labels.getvalue(), train_labels.type)))
    resp = requests.post(url, files=files)
    resp.raise_for_status()
    data = resp.json()
    return data.get("labels", []), data.get("predictions", [])


def extract_cooccurrence_texture(params):
    url = f"{_get_base_url()}/features/cooccurrence"
    resp = requests.post(url, json=params)
    resp.raise_for_status()
    return resp.json().get("features", [])


# -----------------------------
# Replace image (cropping)
# -----------------------------
def replace_image(image_id: str, pil_image: Image.Image, format_hint: str = "PNG"):
    img_to_save = pil_image
    if format_hint.upper() in ("JPG", "JPEG") and pil_image.mode in ("RGBA", "P"):
        img_to_save = pil_image.convert("RGB")

    buf = BytesIO()
    img_to_save.save(buf, format=format_hint.upper())
    img_bytes = buf.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    url = f"{_get_base_url()}/upload/replace-image"
    payload = {"image_id": image_id, "image_data_base64": img_b64}

    resp = requests.post(url, json=payload)
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError:
        try:
            detail = resp.json().get("detail") or resp.json().get("error")
        except Exception:
            detail = None
        if detail:
            st.error(f"Failed to replace image '{image_id}': {detail}")
        else:
            st.error(f"Failed to replace image '{image_id}'. HTTP {resp.status_code}")
        raise

    data = resp.json()
    if data.get("status") != "success":
        st.warning(f"Image replace returned unexpected response: {data}")
    return data


# -----------------------------
# Extract images from CSV
# -----------------------------
def extract_images_from_csv(csv_file):
    url = f"{_get_base_url()}/upload/extract-from-csv"
    files = {"file": (csv_file.name, csv_file.getvalue(), csv_file.type or "text/csv")}
    resp = requests.post(url, files=files)
    resp.raise_for_status()

    data = resp.json()
    zip_b64 = data.get("zip_b64", "")
    image_ids = data.get("image_ids", [])
    errors = data.get("errors", [])

    try:
        zip_bytes = base64.b64decode(zip_b64) if zip_b64 else b""
    except Exception:
        st.error("Failed to decode ZIP from server.")
        zip_bytes = b""

    return zip_bytes, image_ids, errors


# -----------------------------
# Haralick table extraction
# -----------------------------
def extract_haralick_features(params):
    url = f"{_get_base_url()}/features/haralick/extract"
    resp = requests.post(url, json=params)
    resp.raise_for_status()
    return resp.json()


# -----------------------------
# LBP extraction
# -----------------------------
def extract_lbp_features(params):
    url = f"{_get_base_url()}/features/lbp"
    resp = requests.post(url, json=params)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict) and data.get("mode") == "single":
        lbp_b64 = data.get("lbp_image_b64")
        if lbp_b64:
            try:
                data["lbp_image_bytes"] = base64.b64decode(lbp_b64)
            except Exception:
                data["lbp_image_bytes"] = None

    return data


# -----------------------------
# Contour Extraction
# -----------------------------
def extract_contours(params):
    """
    Call /features/contours to compute contours from a binary/grayscale image.

    Example params:
    {
      "image_index": 0,
      "mode": "RETR_EXTERNAL",
      "method": "CHAIN_APPROX_SIMPLE",
      "min_area": 10,
      "return_bounding_boxes": true,
      "return_hierarchy": false
    }
    """
    url = f"{_get_base_url()}/features/contours"
    resp = requests.post(url, json=params)
    resp.raise_for_status()
    data = resp.json()

    if data.get("visualization"):
        try:
            data["visualization_bytes"] = base64.b64decode(data["visualization"])
        except Exception:
            data["visualization_bytes"] = None

    return data


# -----------------------------
# HOG convenience wrapper
# -----------------------------
def extract_hog_features(
    image_index,
    orientations=None,
    pixels_per_cell=None,   # e.g., [8, 8]
    cells_per_block=None,   # e.g., [2, 2]
    resize_width=None,
    resize_height=None,
    visualize=True,
):
    """
    Convenience wrapper for HOG using /features/shape with method="HOG".
    Returns: {"features": [...], "visualization": bytes|None}
    """
    url = f"{_get_base_url()}/features/shape"
    payload = {
        "method": "HOG",
        "image_index": int(image_index),
        "visualize": bool(visualize),
    }
    if orientations is not None:
        payload["orientations"] = int(orientations)
    if pixels_per_cell is not None and isinstance(pixels_per_cell, (list, tuple)) and len(pixels_per_cell) == 2:
        payload["pixels_per_cell"] = [int(pixels_per_cell[0]), int(pixels_per_cell[1])]
    if cells_per_block is not None and isinstance(cells_per_block, (list, tuple)) and len(cells_per_block) == 2:
        payload["cells_per_block"] = [int(cells_per_block[0]), int(cells_per_block[1])]
    if resize_width is not None and resize_height is not None:
        payload["resize_width"] = int(resize_width)
        payload["resize_height"] = int(resize_height)

    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()

    result = {"features": data.get("features", [])}
    if data.get("visualization"):
        try:
            result["visualization"] = base64.b64decode(data["visualization"])
        except Exception:
            result["visualization"] = None
    else:
        result["visualization"] = None

    return result


# ----------------------------------------------------------------------
# SIFT extraction
# ----------------------------------------------------------------------
def extract_sift_features(params):
    """
    Call the backend endpoint ``POST /features/sift``.
    
    ``params`` must follow the ``FeatureBaseRequest`` model that the backend
    expects – i.e. one (or more) of the following keys:

        * ``image_index`` (int) – index of a single image
        * ``image_indices`` (list[int]) – explicit list of indices
        * ``all_images`` (bool) – process every uploaded image

    Backend returns:
        {
            "features": [[float, …], …],          # list of SIFT descriptors
            "visualization": "iVBORw0KGgo…"      # base-64 PNG (may be None)
        }

    Returns:
        {
            "features": [...],
            "visualization": bytes | None
        }
    """
    url = f"{_get_base_url()}/features/sift"
    resp = requests.post(url, json=params)
    resp.raise_for_status()
    data = resp.json()

    result = {"features": data.get("features", [])}
    if data.get("visualization"):
        try:
            result["visualization"] = base64.b64decode(data["visualization"])
        except Exception:
            result["visualization"] = None
    else:
        result["visualization"] = None

    return result


# ----------------------------------------------------------------------
# Edge-detection extraction (Canny / Sobel)
# ----------------------------------------------------------------------
def extract_edge_features(
    params,
    method: str = "canny",
    low_thresh: int = 100,
    high_thresh: int = 200,
    sobel_ksize: int = 3,
):
    url = f"{_get_base_url()}/features/edges"
    query_params = {
        "method": method,
        "low_thresh": low_thresh,
        "high_thresh": high_thresh,
        "sobel_ksize": sobel_ksize,
    }
    resp = requests.post(url, json=params, params=query_params)
    resp.raise_for_status()
    data = resp.json()

    result = {}
    
    # Handle the LIST of edge images
    edge_images = data.get("edge_images", [])
    if edge_images:
        result["edge_images"] = [base64.b64decode(img_b64) for img_b64 in edge_images]
    else:
        result["edge_images"] = []
    
    # Handle ALL gradient matrices (plural!)
    all_matrices = data.get("edges_matrices", [])
    if all_matrices:
        result["edges_matrices"] = all_matrices
    else:
        result["edges_matrices"] = []
    
    return result


# ----------------------------------------------------------------------
# FAST convenience wrapper
# ----------------------------------------------------------------------
def extract_fast_features(
    image_index: int,
    threshold: int | None = None,
    nonmax: bool | None = None,
    fast_type: str | None = None,  # "TYPE_9_16" | "TYPE_7_12" | "TYPE_5_8"
):
    """
    Convenience wrapper for FAST using /features/shape with method="FAST".
    Optional params are forwarded if provided; otherwise backend defaults apply.
    Returns: {"features": [...], "visualization": bytes|None}
    """
    url = f"{_get_base_url()}/features/shape"
    payload = {
        "method": "FAST",
        "image_index": int(image_index),
    }
    # Use names that match the backend ShapeRequest
    if threshold is not None:
        payload["fast_threshold"] = int(threshold)
    if nonmax is not None:
        payload["fast_nonmax"] = bool(nonmax)
    if fast_type is not None:
        payload["fast_type"] = str(fast_type)

    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()

    result = {"features": data.get("features", [])}
    if data.get("visualization"):
        try:
            result["visualization"] = base64.b64decode(data["visualization"])
        except Exception:
            result["visualization"] = None
    else:
        result["visualization"] = None

    return result
