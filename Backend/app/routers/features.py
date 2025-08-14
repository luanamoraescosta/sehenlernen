# backend/app/routers/features.py

import logging
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from app.services.feature_service import (
    generate_histogram_service,
    perform_kmeans_service,
    extract_shape_service,
    extract_haralick_service,          # legacy train/predict demo (multipart)
    extract_cooccurrence_service,
    extract_haralick_features_service,  # NEW: table-style Haralick extraction
    compute_lbp_service,                # NEW: LBP service
)

from app.models.requests import HaralickExtractRequest  # NEW request model

router = APIRouter()

# ---- Request Models ----

class HistogramRequest(BaseModel):
    hist_type: str
    image_index: int
    all_images: bool

class KMeansRequest(BaseModel):
    n_clusters: int
    random_state: int
    selected_images: List[int] = []
    use_all_images: bool = False

class ShapeRequest(BaseModel):
    method: str
    image_index: int

# LBP request model (supports one, multiple, or all images)
class LBPRequest(BaseModel):
    image_indices: List[int] = []         # ignored if use_all_images = True
    use_all_images: bool = False
    radius: int = 1
    num_neighbors: int = 8
    method: str = "uniform"               # {"default","ror","uniform","var"}
    normalize: bool = True                # normalize histogram to sum=1


# ---- Endpoints ----

@router.post("/histogram")
async def histogram(request: HistogramRequest):
    """
    Generate one or more histograms for the uploaded images.
    """
    try:
        b64_list = generate_histogram_service(
            hist_type=request.hist_type,
            image_index=request.image_index,
            all_images=request.all_images
        )
        return {"histograms": b64_list}
    except Exception:
        logging.exception("Failed to generate histogram")
        # Return a controlled JSON error response
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error generating histogram"}
        )

@router.post("/kmeans")
def kmeans(request: KMeansRequest):
    """
    Do the clustering K-means on selected images.
    """
    try:
        plot_b64, assignments = perform_kmeans_service(
            n_clusters=request.n_clusters,
            random_state=request.random_state,
            selected_images=request.selected_images,
            use_all_images=request.use_all_images
        )
        return {"plot": plot_b64, "assignments": assignments}
    except Exception as e:
        logging.exception("Failed to do the clustering K-means")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/shape")
def shape_features(request: ShapeRequest):
    """
    Extract shape features (HOG, SIFT, FAST) from a single image.
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
    except Exception:
        logging.exception("Failed to extract shape features")
        raise HTTPException(status_code=500, detail="Internal server error extracting shape features")

@router.post("/haralick")
async def haralick(
    train_images: List[UploadFile] = File(...),
    train_labels: UploadFile = File(...),
    test_images: List[UploadFile] = File(...)
):
    """
    Legacy demo: Extract a simple Haralick feature from training images, train a classifier,
    and predict labels for test images. (multipart upload)
    """
    try:
        labels, predictions = await extract_haralick_service(
            train_images=train_images,
            train_labels=train_labels,
            test_images=test_images
        )
        return {"labels": labels, "predictions": predictions}
    except Exception:
        logging.exception("Failed to extract Haralick texture features (legacy)")
        raise HTTPException(status_code=500, detail="Internal server error extracting Haralick features")

@router.post("/cooccurrence")
def cooccurrence(request: ShapeRequest):
    """
    Extract gray-level co-occurrence features from a single image.
    """
    try:
        features = extract_cooccurrence_service(image_index=request.image_index)
        return {"features": features}
    except Exception:
        logging.exception("Failed to extract co-occurrence features")
        raise HTTPException(status_code=500, detail="Internal server error extracting co-occurrence features")

# ---- NEW: Haralick extraction as table (JSON) ----
@router.post("/haralick/extract")
def haralick_extract(req: HaralickExtractRequest):
    """
    Compute Haralick (GLCM) features for selected images.

    Request (JSON):
    {
      "image_indices": [0, 1],
      "levels": 64,
      "distances": [1, 2],
      "angles": [0.0, 0.785398, 1.570796, 2.356194],
      "resize_width": 256,
      "resize_height": 256,
      "average_over_angles": true,
      "properties": ["contrast","dissimilarity","homogeneity","energy","correlation","ASM"]
    }

    Response:
    {
      "columns": ["image_id", ...],
      "rows": [
        ["img001.jpg", feat1, feat2, ...],
        ...
      ]
    }
    """
    try:
        result = extract_haralick_features_service(
            image_indices=req.image_indices,
            levels=req.levels,
            distances=req.distances,
            angles=req.angles,
            resize_width=req.resize_width,
            resize_height=req.resize_height,
            average_over_angles=req.average_over_angles,
            properties=req.properties
        )
        return result
    except Exception as e:
        logging.exception("Failed to extract Haralick table features")
        raise HTTPException(status_code=400, detail=str(e))

# ---- NEW: LBP extraction (single or multi-image) ----
@router.post("/lbp")
def lbp_extract(req: LBPRequest):
    """
    Compute Local Binary Pattern (LBP) histograms.

    Supports:
      - Single image (returns histogram + an LBP visualization PNG in base64)
      - Multiple images (returns a table with columns ["image_id","bin_0",...])

    Request (JSON):
    {
      "image_indices": [0, 2],        # ignored if use_all_images = true
      "use_all_images": false,
      "radius": 1,
      "num_neighbors": 8,
      "method": "uniform",            # {"default","ror","uniform","var"}
      "normalize": true               # histogram normalized to sum=1
    }
    """
    try:
        result = compute_lbp_service(
            image_indices=req.image_indices,
            use_all_images=req.use_all_images,
            radius=req.radius,
            num_neighbors=req.num_neighbors,
            method=req.method,
            normalize=req.normalize,
        )
        return result
    except Exception as e:
        logging.exception("Failed to compute LBP features")
        raise HTTPException(status_code=400, detail=str(e))
