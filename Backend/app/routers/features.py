# backend/app/routers/features.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from app.services.feature_service import (
    generate_histogram_service,
    perform_kmeans_service,
    extract_shape_service,
    extract_haralick_service,
    extract_cooccurrence_service,
)

router = APIRouter()

# ---- Request Models ----
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

# ---- Endpoints ----

@router.post("/histogram")
def histogram(request: HistogramRequest):
    """
    Generate one or more histograms.
    """
    try:
        b64_list = generate_histogram_service(
            hist_type=request.hist_type,
            image_index=request.image_index,
            all_images=request.all_images
        )
        return {"histograms": b64_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/kmeans")
def kmeans(request: KMeansRequest):
    """
    Perform k-means clustering and return plot + assignments.
    """
    try:
        plot_b64, assignments = perform_kmeans_service(
            n_clusters=request.n_clusters,
            random_state=request.random_state
        )
        return {"plot": plot_b64, "assignments": assignments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/shape")
def shape_features(request: ShapeRequest):
    """
    Extract shape features (HOG, SIFT, FAST) for a given image index.
    """
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

@router.post("/haralick")
async def haralick(
    train_images: List[UploadFile] = File(...),
    train_labels: UploadFile = File(...),
    test_images: List[UploadFile] = File(...)
):
    """
    Extract Haralick texture features, train random forest, predict test labels.
    """
    try:
        labels, predictions = await extract_haralick_service(
            train_images=train_images,
            train_labels=train_labels,
            test_images=test_images
        )
        return {"labels": labels, "predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cooccurrence")
def cooccurrence(request: ShapeRequest):
    """
    Extract co-occurrence texture features for a given image index.
    """
    try:
        features = extract_cooccurrence_service(
            image_index=request.image_index
        )
        return {"features": features}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
