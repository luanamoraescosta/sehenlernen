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
    extract_contours_service,           # NEW: Contour extraction
)

from app.models.requests import (
    HaralickExtractRequest,
    LBPRequest,
    ContourRequest,                     # NEW request model
)

router = APIRouter()

# ---- Request Models (legacy inline ones kept for backwards compat) ----

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


# ---- Endpoints ----

@router.post("/histogram")
async def histogram(request: HistogramRequest):
    try:
        b64_list = generate_histogram_service(
            hist_type=request.hist_type,
            image_index=request.image_index,
            all_images=request.all_images
        )
        return {"histograms": b64_list}
    except Exception:
        logging.exception("Failed to generate histogram")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error generating histogram"}
        )


@router.post("/kmeans")
def kmeans(request: KMeansRequest):
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
    try:
        features = extract_cooccurrence_service(image_index=request.image_index)
        return {"features": features}
    except Exception:
        logging.exception("Failed to extract co-occurrence features")
        raise HTTPException(status_code=500, detail="Internal server error extracting co-occurrence features")


# ---- Haralick extraction (table) ----
@router.post("/haralick/extract")
def haralick_extract(req: HaralickExtractRequest):
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


# ---- LBP extraction ----
@router.post("/lbp")
def lbp_extract(req: LBPRequest):
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


# ---- NEW: Contour extraction ----
@router.post("/contours")
def contour_extract(req: ContourRequest):
    """
    Extract contours from a binary/grayscale version of the selected image.
    Uses OpenCV findContours to detect closed shapes and outlines.
    Returns contour points, areas, bounding boxes, and visualization.
    """
    try:
        result = extract_contours_service(
            image_index=req.image_index,
            mode=req.mode,
            method=req.method,
            min_area=req.min_area or 10,
            return_bounding_boxes=req.return_bounding_boxes,
            return_hierarchy=req.return_hierarchy,
        )
        return result
    except Exception as e:
        logging.exception("Failed to extract contours")
        raise HTTPException(status_code=400, detail=str(e))
