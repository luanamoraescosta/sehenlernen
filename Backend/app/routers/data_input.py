from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
from app.models.requests import ConfigureMetadataRequest
import pandas as pd
from app.services.data_service import save_uploaded_images, read_metadata, configure_metadata

router = APIRouter()

@router.post("/images")
async def upload_images(files: List[UploadFile] = File(...)):
    """
    Upload image files and return their assigned IDs.
    """
    try:
        image_ids = await save_uploaded_images(files)
        return {"image_ids": image_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/metadata")
async def upload_metadata_file(
    file: UploadFile = File(...),
    delimiter: str = Form(","),
    decimal_sep: str = Form(".")
):
    """
    Upload metadata CSV/XLSX and return list of column names.
    """
    try:
        columns = read_metadata(file, delimiter, decimal_sep)
        return {"columns": columns}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/metadata/configure")
async def configure_metadata_endpoint(req: ConfigureMetadataRequest):
    """
    Accepts an image-ID column name plus a mapping of all other
    metadata columns→types, stores them in server state.
    """
    try:
        await configure_metadata(req.image_id_col, req.col_mapping)
        return {"message": "Metadata configuration saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))