from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from app.services.feature_service import (
    generate_histogram_service,
    perform_kmeans_service,
    extract_shape_service,
    extract_haralick_service,
    extract_cooccurrence_service,
)

router = APIRouter()

# Request models
class HistogramRequest(BaseModel):
    hist_type: str
    image_index: int
    all_images: bool

class KMeansRequest(BaseModel):
    n_clusters: int
    random_state: int

class ShapeRequest(BaseModel):
    method: str
    image_index: int

# Histogram endpoint\@router.post("/histogram")
def histogram(request: HistogramRequest):
    try:
        b64_list = generate_histogram_service(
            hist_type=request.hist_type,
            image_index=request.image_index,
            all_images=request.all_images
        )
        return {"histograms": b64_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# K-means endpoint\@router.post("/kmeans")
def kmeans(request: KMeansRequest):
    try:
        plot_b64, assignments = perform_kmeans_service(
            n_clusters=request.n_clusters,
            random_state=request.random_state
        )
        return {"plot": plot_b64, "assignments": assignments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Shape features endpoint\@router.post("/shape")
def shape_features(request: ShapeRequest):
    try:
        features, viz_b64 = extract_shape_service(
            method=request.method,
            image_index=request.image_index
        )
        response: Dict[str, Any] = {"features": features}
        if viz_b64:
            response["visualization"] = viz_b64
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Haralick texture endpoint\@router.post("/haralick")
async def haralick(
    train_images: List[UploadFile] = File(...),
    train_labels: UploadFile = File(...),
    test_images: List[UploadFile] = File(...)
):
    try:
        labels, predictions = await extract_haralick_service(
            train_images=train_images,
            train_labels=train_labels,
            test_images=test_images
        )
        return {"labels": labels, "predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Co-occurrence texture endpoint\@router.post("/cooccurrence")
def cooccurrence(request: ShapeRequest):  # reuse index model
    try:
        features = extract_cooccurrence_service(
            image_index=request.image_index
        )
        return {"features": features}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
