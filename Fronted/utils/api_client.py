import os
import requests
import base64

# Base URL of the backend FastAPI server
def _get_base_url():
    return os.getenv("SEHEN_LERNEN_API_URL", "http://localhost:8000")

# File upload for images

def upload_images(image_files):
    """
    Upload image files to the backend.
    :param image_files: list of UploadedFile objects
    :return: list of image IDs
    """
    url = f"{_get_base_url()}/upload/metadata/configure"
    files = [("files", (f.name, f.getvalue(), f.type)) for f in image_files]
    resp = requests.post(url, files=files)
    resp.raise_for_status()
    return resp.json().get("image_ids", [])

# Upload metadata file and return column names

def upload_metadata_file(csv_file, delimiter, decimal_sep):
    """
    Upload metadata CSV/XLSX and retrieve column names.
    """
    url = f"{_get_base_url()}/upload-metadata"
    files = {"file": (csv_file.name, csv_file.getvalue(), csv_file.type)}
    data = {"delimiter": delimiter, "decimal_sep": decimal_sep}
    resp = requests.post(url, files=files, data=data)
    resp.raise_for_status()
    return resp.json().get("columns", [])

# Configure metadata mapping

def configure_metadata(image_id_col, col_mapping):
    """
    Send column mapping configuration to the backend.
    """
    url = f"{_get_base_url()}/metadata/configure"
    payload = {"image_id_col": image_id_col, "col_mapping": col_mapping}
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()

# Filtering sampling

def filter_sampling(filter_values):
    """
    Apply metadata filters and return sampled image IDs.
    """
    url = f"{_get_base_url()}/sampling/filter"
    resp = requests.post(url, json={"filters": filter_values})
    resp.raise_for_status()
    return resp.json().get("sampled_ids", [])

# Stratified sampling

def stratified_sampling(target_col, sample_size):
    """
    Perform stratified sampling by target column and return sampled image IDs.
    """
    url = f"{_get_base_url()}/sampling/stratified"
    payload = {"target_col": target_col, "sample_size": sample_size}
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    return resp.json().get("sampled_ids", [])

# Generate histogram images

def generate_histogram(params):
    """
    Request histogram generation.
    :param params: { hist_type: str, image_index: int, all_images: bool }
    :return: list of PNG bytes
    """
    url = f"{_get_base_url()}/features/histogram"
    resp = requests.post(url, json=params)
    resp.raise_for_status()
    b64_list = resp.json().get("histograms", [])
    return [base64.b64decode(b) for b in b64_list]

# Perform k-means clustering

def perform_kmeans(params):
    """
    Request k-means clustering.
    :param params: { n_clusters: int, random_state: int }
    :return: (plot_bytes, assignments)
    """
    url = f"{_get_base_url()}/features/kmeans"
    resp = requests.post(url, json=params)
    resp.raise_for_status()
    data = resp.json()
    plot_bytes = base64.b64decode(data.get("plot", ""))
    assignments = data.get("assignments", [])
    return plot_bytes, assignments

# Extract shape features

def extract_shape_features(params):
    """
    Request shape feature extraction.
    :param params: { method: str, image_index: int }
    :return: { features: list, visualization: bytes (optional) }
    """
    url = f"{_get_base_url()}/features/shape"
    resp = requests.post(url, json=params)
    resp.raise_for_status()
    data = resp.json()
    result = {"features": data.get("features", [])}
    viz_b64 = data.get("visualization")
    if viz_b64:
        result["visualization"] = base64.b64decode(viz_b64)
    return result

# Extract Haralick texture features

def extract_haralick_texture(params):
    """
    Request Haralick texture feature extraction and classification.
    :param params: { train_images, train_labels, test_images }
    :return: (labels, predictions)
    """
    # Files need to be sent via multipart
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

# Extract co-occurrence texture features

def extract_cooccurrence_texture(params):
    """
    Request co-occurrence texture feature extraction.
    :param params: { image_index: int }
    :return: list of feature values
    """
    url = f"{_get_base_url()}/features/cooccurrence"
    resp = requests.post(url, json=params)
    resp.raise_for_status()
    return resp.json().get("features", [])
