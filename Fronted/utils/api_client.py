# Fronted/utils/api_client.py

import os
import requests
import base64
import streamlit as st
from io import BytesIO
from PIL import Image


def _get_base_url():
    """
    Returns the base URL for the FastAPI backend.
    Defaults to http://localhost:8000 unless overridden by
    the SEHEN_LERNEN_API_URL environment variable.
    """
    return os.getenv("SEHEN_LERNEN_API_URL", "http://localhost:8000")


def upload_images(image_files):
    """
    Upload image files to the backend and return a list of image IDs.
    """
    url = f"{_get_base_url()}/upload/images"
    files = [("files", (f.name, f.getvalue(), f.type)) for f in image_files]
    resp = requests.post(url, files=files)
    resp.raise_for_status()
    return resp.json().get("image_ids", [])


def upload_metadata_file(csv_file, delimiter, decimal_sep):
    """
    Upload metadata CSV/XLSX and retrieve column names.
    """
    url = f"{_get_base_url()}/upload/metadata"
    files = {"file": (csv_file.name, csv_file.getvalue(), csv_file.type)}
    data = {"delimiter": delimiter, "decimal_sep": decimal_sep}
    resp = requests.post(url, files=files, data=data)
    resp.raise_for_status()
    return resp.json().get("columns", [])


def configure_metadata(image_id_col, col_mapping):
    """
    Send column mapping configuration to the backend.
    """
    url = f"{_get_base_url()}/upload/metadata/configure"
    payload = {"image_id_col": image_id_col, "col_mapping": col_mapping}
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()


def filter_sampling(filter_values):
    """
    Apply metadata filters and return sampled image IDs.
    """
    url = f"{_get_base_url()}/sampling/filter"
    resp = requests.post(url, json={"filters": filter_values})
    resp.raise_for_status()
    return resp.json().get("sampled_ids", [])


def stratified_sampling(target_col, sample_size):
    """
    Perform stratified sampling by target column and return sampled image IDs.
    """
    url = f"{_get_base_url()}/sampling/stratified"
    payload = {"target_col": target_col, "sample_size": sample_size}
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    return resp.json().get("sampled_ids", [])


def generate_histogram(params):
    """
    Request histogram generation.
    :param params: dict with keys 'hist_type', 'image_index', 'all_images'
    :return: list of PNG bytes
    """
    url = f"{_get_base_url()}/features/histogram"
    resp = requests.post(url, json=params)
    resp.raise_for_status()
    b64_list = resp.json().get("histograms", [])
    return [base64.b64decode(b) for b in b64_list]


def perform_kmeans(params):
    """
    Requests K-means clustering.
    :param params: dict with keys 'n_clusters', 'random_state', 'selected_images', 'use_all_images'
    :return: (plot_bytes, assignments)
    """
    url = f"{_get_base_url()}/features/kmeans"
    resp = requests.post(url, json=params)

    # Add more detailed error handling
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
    Request shape feature extraction.
    :param params: dict with keys 'method', 'image_index'
    :return: dict {'features': list, 'visualization': bytes (optional)}
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
    """
    Request Haralick texture feature extraction and classification.
    :param params: dict with keys 'train_images', 'train_labels', 'test_images'
    :return: (labels, predictions)
    """
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
    """
    Request co-occurrence texture feature extraction.
    :param params: dict with key 'image_index'
    :return: list of feature values
    """
    url = f"{_get_base_url()}/features/cooccurrence"
    resp = requests.post(url, json=params)
    resp.raise_for_status()
    return resp.json().get("features", [])


# -----------------------------
# Replace image (cropping)
# -----------------------------
def replace_image(image_id: str, pil_image: Image.Image, format_hint: str = "PNG"):
    """
    Persistently replace an image on the backend with a cropped version.

    :param image_id: The backend identifier of the image (filename returned from /upload/images).
    :param pil_image: PIL Image to upload as the replacement.
    :param format_hint: "PNG" or "JPEG" (defaults to PNG). JPEG will be converted with RGB mode.
    :raises: requests.exceptions.RequestException on network/HTTP errors.
    """
    # Normalize mode for JPEG
    img_to_save = pil_image
    if format_hint.upper() in ("JPG", "JPEG") and pil_image.mode in ("RGBA", "P"):
        img_to_save = pil_image.convert("RGB")

    # Serialize to bytes
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
        # Surface server-provided error details if available
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
# NEW: Extract images from CSV of URLs
# -----------------------------
def extract_images_from_csv(csv_file):
    """
    Upload a CSV of image URLs, download them server-side, and get a ZIP back.

    :param csv_file: Streamlit UploadedFile (st.file_uploader) for a .csv
    :return: (zip_bytes: bytes, image_ids: list[str], errors: list[str])
    """
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
# NEW: Haralick table extraction
# -----------------------------
def extract_haralick_features(params):
    """
    Call /features/haralick/extract for GLCM-based Haralick features.

    Example params:
    {
      "image_indices": [0, 1],
      "levels": 64,
      "distances": [1, 2],
      "angles": [0.0, 0.785398, 1.570796, 2.356194],
      "resize_width": 256,
      "resize_height": 256,
      "average_over_angles": true,
      "properties": ["contrast","dissimilarity","homogeneity","energy","correlation","ASM"]
    }

    Returns: {"columns": [...], "rows": [[...], ...]}
    """
    url = f"{_get_base_url()}/features/haralick/extract"
    resp = requests.post(url, json=params)
    resp.raise_for_status()
    return resp.json()
