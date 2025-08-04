# backend/app/services/data_service.py

import os
from pathlib import Path
import shutil
import pandas as pd
from fastapi import UploadFile

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
    """
    for f in os.listdir(IMAGE_DIR):
        file_path = IMAGE_DIR / f
        if file_path.is_file():
            file_path.unlink()

    image_ids = []
    for file in files:
        contents = await file.read()
        # Use original filename as ID
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
    contents = await file.read()
    name = file.filename.lower()
    if name.endswith(".csv"):
        metadata_df = pd.read_csv(
            pd.io.common.BytesIO(contents),
            delimiter=delimiter,
            decimal=decimal_sep
        )
    else:
        metadata_df = pd.read_excel(
            pd.io.common.BytesIO(contents)
        )
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