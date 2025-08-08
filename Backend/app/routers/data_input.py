# backend/app/routers/data_input.py

from fastapi import APIRouter, UploadFile, Form, HTTPException
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
        logging.exception("Failed to upload images")
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
        logging.exception("Failed to upload metadata")
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/metadata/configure")
async def configure_metadata(req: ConfigureMetadataRequest):
    try:
        await data_service.configure_metadata(req.image_id_col, req.col_mapping)
        return {"status": "success"}
    except Exception as e:
        logging.exception("Failed to configure metadata")
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/replace-image")
async def replace_image(request: ReplaceImageRequest):
    """
    Replace an existing image on the backend with a new (cropped) version.
    The replacement is persistent and affects all downstream processing.
    """
    try:
        logging.info(f"Received request to replace image: {request.image_id}")
        data_service.replace_image(request.image_id, request.image_data_base64)
        logging.info("Image replacement successful")
        return {"status": "success"}
    except FileNotFoundError as e:
        logging.error(f"Image not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logging.error(f"Invalid image data: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except IOError as e:
        logging.error(f"Failed to save image: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logging.error(f"Unexpected error replacing image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
