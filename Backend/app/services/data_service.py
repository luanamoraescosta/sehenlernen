# backend/app/services/data_service.py
import os
import logging
from pathlib import Path
import pandas as pd
from fastapi import UploadFile
from PIL import Image
import io
import zipfile
import hashlib
import mimetypes
from urllib.parse import urlparse
import tempfile

import requests

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

# ---- Image upload / metadata config ----

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


async def save_and_extract_zip(zip_file: UploadFile) -> list[str]:
    """
    Save the zip temporarily, extract only image files
    (.png/.jpg/.jpeg/.tif/...) to BASE_DATA_DIR and return
    a list of extracted paths.
    """
    # 1) Save zip to a temporary file
    suffix = "_" + zip_file.filename
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        # Stream copy
        while chunk := await zip_file.read(1024 * 1024):
            tmp.write(chunk)

    # 2) Extract
    extracted = []
    with zipfile.ZipFile(tmp_path) as z:
        for member in z.infolist():
            # Ignore directories
            if member.is_dir():
                continue
            # Filter image extensions
            if not member.filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                continue
            dest_path = IMAGE_DIR / member.filename
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            # Safe extraction: prevent path traversal
            _safe_extract(z, member, dest_path)
            extracted.append(str(dest_path.resolve()))

    # 3) Remove temporary source
    tmp_path.unlink(missing_ok=True)
    # Reset file cursor
    await zip_file.seek(0)
    return extracted

def _safe_extract(zf: zipfile.ZipFile, member: zipfile.ZipInfo, dest_path: Path):
    """
    Prevent path traversal attacks.
    """
    dest_path = dest_path.resolve()
    root_dir = IMAGE_DIR.resolve()
    if not str(dest_path).startswith(str(root_dir)):
        raise RuntimeError("Path traversal detected in ZIP file")
    # Write file
    with zf.open(member) as source, open(dest_path, "wb") as target:
        target.write(source.read())

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

# ---- Cropping replace ----

def _validate_image_bytes(img_bytes: bytes) -> None:
    """
    Ensure the provided bytes decode into a valid image.
    Raises ValueError if the image cannot be opened.
    """
    try:
        with Image.open(io.BytesIO(img_bytes)) as im:
            im.verify()
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

    # Overwrite atomically where possible
    try:
        tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        with open(tmp_path, "wb") as f:
            f.write(img_bytes)
        os.replace(tmp_path, file_path)
        logging.info(f"Replaced image on disk: {file_path}")
    except Exception as e:
        logging.error(f"Failed to write image file: {e}")
        raise OSError(f"Failed to write image file: {str(e)}")

# ---- Extract images from CSV (with UA + 3s timeout) ----

UA = "SehenLernen/1.0 (+contact@example.com) FastAPI-Extractor"  # put a real contact if you have one

def _make_http_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": UA,
        "Accept": "image/*, */*;q=0.8",
    })
    return s

def _infer_ext_from_response(url: str, resp: requests.Response) -> str:
    # Prefer Content-Type
    ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if ctype:
        ext = mimetypes.guess_extension(ctype)
        if ext:
            return ext
    # Fallback to URL path
    path = urlparse(url).path
    _, _, filename = path.rpartition("/")
    if "." in filename:
        ext = "." + filename.rsplit(".", 1)[-1]
        if 1 <= len(ext) <= 5:
            return ext
    # Default
    return ".jpg"

def _safe_filename(url: str, ext: str) -> str:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    if not ext.startswith("."):
        ext = "." + ext
    return f"img_{h}{ext}"

async def extract_images_from_csv(file: UploadFile) -> dict:
    """
    Given a CSV of image URLs, download them all, store to IMAGE_DIR, and return a ZIP (bytes),
    list of saved image IDs (filenames), and a list of error messages.

    Returns:
      {
        "zip_bytes": bytes,
        "image_ids": List[str],
        "errors": List[str]
      }
    """
    # Read CSV into pandas
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        logging.exception("Failed to parse CSV for extractor")
        raise ValueError(f"Failed to parse CSV: {e}")

    if df.shape[1] == 0:
        raise ValueError("CSV has no columns")

    # Pick URL column: try common names, else first column
    url_col = None
    for cand in df.columns:
        lc = str(cand).strip().lower()
        if lc in ("url", "image_url", "image", "link", "img", "img_url", "uri"):
            url_col = cand
            break
    if url_col is None:
        url_col = df.columns[0]

    urls = df[url_col].dropna().astype(str).tolist()
    if not urls:
        raise ValueError("No URLs found in the CSV")

    # Clear existing images (match behavior of save_uploaded_images)
    for f in os.listdir(IMAGE_DIR):
        file_path = IMAGE_DIR / f
        if file_path.is_file():
            file_path.unlink()

    image_ids: list[str] = []
    errors: list[str] = []
    session = _make_http_session()

    for i, url in enumerate(urls, start=1):
        url = url.strip()
        if not url:
            continue
        try:
            resp = session.get(url, timeout=3, allow_redirects=True)
            resp.raise_for_status()

            ctype = (resp.headers.get("Content-Type") or "").lower()
            if not ctype.startswith("image/"):
                raise ValueError(f"URL is not an image (Content-Type: {ctype or 'unknown'})")

            content = resp.content
            _validate_image_bytes(content)

            ext = _infer_ext_from_response(url, resp)
            fname = _safe_filename(url, ext)
            out_path = IMAGE_DIR / fname
            with open(out_path, "wb") as f:
                f.write(content)
            image_ids.append(fname)
        except Exception as e:
            msg = f"Row {i}: {url} -> {e}"
            logging.warning(msg)
            errors.append(msg)

    # Build ZIP in-memory of all successfully downloaded images
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for img_id in image_ids:
            p = IMAGE_DIR / img_id
            if p.exists():
                zf.write(p, arcname=img_id)

    return {
        "zip_bytes": zip_buf.getvalue(),
        "image_ids": image_ids,
        "errors": errors,
    }
