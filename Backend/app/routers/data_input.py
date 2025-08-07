# backend/app/routers/data_input.py

from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List
import logging


from app.services import data_service
from app.models.requests import ConfigureMetadataRequest, ReplaceImageRequest

router = APIRouter()

@router.post("/images")
async def upload_images(files: List[UploadFile]):
    try:
        image_ids = await data_service.save_uploaded_images(files)
        return {"image_ids": image_ids}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/metadata")
async def upload_metadata(
    file: UploadFile,
    delimiter: str = Form(","),
    decimal_sep: str = Form(".")
):
    try:
        columns = await data_service.read_metadata(file, delimiter, decimal_sep)
        return {"columns": columns}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/metadata/configure")
async def configure_metadata(req: ConfigureMetadataRequest):
    try:
        await data_service.configure_metadata(req.image_id_col, req.col_mapping)
        return {"status": "success"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/replace-image")
async def replace_image(request: ReplaceImageRequest):
    try:
        logging.info(f"Received request to replace image: {request.image_id}")
        data_service.replace_image(request.image_id, request.image_data_base64)
        logging.info("Image replacement successful")
        return {"status": "success"}
    except Exception as e:
        logging.error(f"Image replacement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))