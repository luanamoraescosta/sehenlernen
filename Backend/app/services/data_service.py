# backend/app/services/data_service.py

import os
import logging
from pathlib import Path
import shutil
import pandas as pd
from fastapi import UploadFile, HTTPException
from PIL import Image
import io

from app.utils.image_utils import base64_to_bytes
from app.utils.csv_utils import read_metadata_file

# Directory to store images
BASE_DIR = Path(__file__).resolve().parent.parent.parent
IMAGE_DIR = BASE_DIR / "storage" / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# In-memory metadata and configuration
metadata_df = None
image_id_col = None
col_mapping = {}

async def save_uploaded_images(files: list[UploadFile]) -> list[str]:
    """
    Save uploaded image files to disk and return list of image IDs (filenames).
    NOTE: Current behavior clears existing images on each upload batch.
    """
    # Clear existing images
    for f in os.listdir(IMAGE_DIR):
        file_path = IMAGE_DIR / f
        if file_path.is_file():
            file_path.unlink()

    image_ids = []
    for file in files:
        contents = await file.read()
        filename = file.filename
        file_path = IMAGE_DIR / filename
        with open(file_path, "wb") as f:
            f.write(contents)
        image_ids.append(filename)
    return image_ids

async def read_metadata(file: UploadFile, delimiter: str, decimal_sep: str) -> list[str]:
    """
    Read metadata CSV or Excel file into pandas DataFrame.
    Store DataFrame in module-level state and return column names.
    """
    global metadata_df
    metadata_df = read_metadata_file(file, delimiter, decimal_sep)
    return metadata_df.columns.tolist()

async def configure_metadata(id_col: str, mapping: dict) -> None:
    """
    Configure which column represents image ID and other column mappings.
    """
    global image_id_col, col_mapping
    image_id_col = id_col
    col_mapping = mapping

def load_image(image_id: str) -> bytes:
    """
    Load a stored image by its ID (filename) and return raw bytes.
    """
    file_path = IMAGE_DIR / image_id
    if not file_path.exists():
        raise FileNotFoundError(f"Image {image_id} not found")
    return file_path.read_bytes()

def get_all_image_ids() -> list[str]:
    """
    Return list of all saved image IDs (filenames).
    """
    return [p.name for p in IMAGE_DIR.glob("*") if p.is_file()]

def _validate_image_bytes(img_bytes: bytes) -> None:
    """
    Ensure the provided bytes decode into a valid image.
    Raises ValueError if the image cannot be opened.
    """
    try:
        with Image.open(io.BytesIO(img_bytes)) as im:
            im.verify()  # verify does not decode full image but catches truncation/format errors
    except Exception as e:
        raise ValueError(f"Decoded data is not a valid image: {e}") from e

def replace_image(image_id: str, base64_data: str) -> None:
    """
    Replace an existing stored image with the provided base64-encoded image bytes.
    Validates that the bytes form a real image before overwriting.
    """
    file_path = IMAGE_DIR / image_id
    if not file_path.exists():
        logging.error(f"Image file not found: {file_path}")
        raise FileNotFoundError(f"Image {image_id} not found")

    # Decode base64 → bytes
    try:
        img_bytes = base64_to_bytes(base64_data)
    except Exception as e:
        logging.error(f"Base64 decoding error: {e}")
        raise ValueError(f"Invalid base64 image data: {str(e)}")

    # Validate image bytes
    try:
        _validate_image_bytes(img_bytes)
    except ValueError as e:
        logging.error(str(e))
        raise

    # Overwrite file atomically where possible
    try:
        tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        with open(tmp_path, "wb") as f:
            f.write(img_bytes)
        os.replace(tmp_path, file_path)
        logging.info(f"Replaced image on disk: {file_path}")
    except Exception as e:
        logging.error(f"Failed to write image file: {e}")
        raise IOError(f"Failed to write image file: {str(e)}")
