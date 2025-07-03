from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List

from app.services.sampling_service import (
    filter_images_by_metadata,
    stratified_sample_images,
)

router = APIRouter()

class FilterRequest(BaseModel):
    filters: Dict[str, List]

class StratifiedRequest(BaseModel):
    target_col: str
    sample_size: int

@router.post("/filter")
def filter_sampling(request: FilterRequest):
    """
    Apply metadata filters and return list of sampled image IDs.
    """
    try:
        sampled_ids = filter_images_by_metadata(request.filters)
        return {"sampled_ids": sampled_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stratified")
def stratified_sampling(request: StratifiedRequest):
    """
    Perform stratified sampling by target column and return list of sampled image IDs.
    """
    try:
        sampled_ids = stratified_sample_images(
            target_col=request.target_col,
            sample_size=request.sample_size
        )
        return {"sampled_ids": sampled_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
