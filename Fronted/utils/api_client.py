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
# NEW: Contour Extraction
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

    Returns:
    {
      "contours": [[(x,y), ...], ...],
      "bounding_boxes": [[x,y,w,h], ...],
      "areas": [...],
      "hierarchy": [...],
      "visualization": "..."  # base64-encoded PNG
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
