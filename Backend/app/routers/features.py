# backend/app/routers/features.py

import logging
import base64
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.services.feature_service import (
    generate_histogram_service,
    perform_kmeans_service,
    extract_shape_service,
    extract_haralick_service,          # legacy train/predict demo (multipart)
    extract_cooccurrence_service,
    extract_haralick_features_service,  # table-style Haralick extraction
    compute_lbp_service,                # LBP service
    extract_contours_service,           # Contour extraction
    extract_sift_service,               # SIFT endpoint (multi/single)
    extract_edges_service,              # Edge detection endpoint
)

from app.models.requests import (
    HaralickExtractRequest,
    LBPRequest,
    ContourRequest,                     # contour request model
    ShapeRequest,                       # enhanced ShapeRequest (HOG + FAST options)
    FeatureBaseRequest,                 # base selector for single/multi/all image ops
    SiftResponse,                       # (imported for consistency; response returned as dict)
    EdgeResponse,                       # (imported for consistency; response returned as dict)
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


# -------------------------------
# Endpoints
# -------------------------------
@router.post("/histogram")
async def histogram(request: HistogramRequest):
    """
    Generate one or more histograms for the uploaded images (color or grayscale).
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
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error generating histogram"}
        )


@router.post("/kmeans")
def kmeans(request: KMeansRequest):
    """
    Perform K-means clustering on selected images (color-histogram features + PCA2 for plotting).
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
    Extract shape/structure features from a single image.

    Supported methods:
      - "HOG": optional parameters:
          orientations, pixels_per_cell [h,w], cells_per_block [y,x],
          resize_width, resize_height, visualize
      - "SIFT": ignores HOG/FAST extras
      - "FAST": optional parameters:
          fast_threshold, fast_nonmax, fast_type ("TYPE_9_16" | "TYPE_7_12" | "TYPE_5_8")
    """
    try:
        features, viz_b64 = extract_shape_service(
            method=request.method,
            image_index=request.image_index,
            # HOG params (service ignores when method != "HOG")
            orientations=getattr(request, "orientations", None),
            pixels_per_cell=getattr(request, "pixels_per_cell", None),
            cells_per_block=getattr(request, "cells_per_block", None),
            resize_width=getattr(request, "resize_width", None),
            resize_height=getattr(request, "resize_height", None),
            visualize=getattr(request, "visualize", None),
            # FAST params (service ignores when method != "FAST")
            fast_threshold=getattr(request, "fast_threshold", None),
            fast_nonmax=getattr(request, "fast_nonmax", None),
            fast_type=getattr(request, "fast_type", None),
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
    Extract gray-level co-occurrence features from a single image (fixed params).
    """
    try:
        features = extract_cooccurrence_service(image_index=request.image_index)
        return {"features": features}
    except Exception:
        logging.exception("Failed to extract co-occurrence features")
        raise HTTPException(status_code=500, detail="Internal server error extracting co-occurrence features")


# ---- Haralick extraction (table) ----
@router.post("/haralick/extract")
def haralick_extract(req: HaralickExtractRequest):
    """
    Compute GLCM (Haralick) properties for a set of uploaded images; returns table-like JSON.
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


# ---- LBP extraction ----
@router.post("/lbp")
def lbp_extract(req: LBPRequest):
    """
    Compute LBP histograms for one, multiple, or all images.
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


# ---- Contour extraction ----
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


# ----------------------------------------------------------------------
# SIFT endpoint
# ----------------------------------------------------------------------
@router.post("/sift")
def sift(request: FeatureBaseRequest):
    """
    Run OpenCV SIFT on one or more images (single-image returns a visualization).
    Response:
        {
            "features": [[float, ...], ...],
            "visualization": base64_str | null
        }
    """
    try:
        feats, viz_bytes = extract_sift_service(
            image_index=request.image_index,
            all_images=request.all_images,
            image_indices=request.image_indices,
        )
        viz_b64 = base64.b64encode(viz_bytes).decode() if viz_bytes else None
        return {"features": feats, "visualization": viz_b64}
    except Exception as e:
        logging.exception("SIFT extraction failed")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------------
# Edge-detection endpoint (Canny by default)
# ----------------------------------------------------------------------
@router.post("/edges")
def edges(
    request: FeatureBaseRequest,
    method: str = "canny",
    low_thresh: int = 100,
    high_thresh: int = 200,
    sobel_ksize: int = 3,
):
    """
    Apply edge detection (Canny or Sobel) to one/multiple/all images selected via FeatureBaseRequest.
    Query params can adjust the method and thresholds.

    Response:
        {
            "edge_images": [base64_str, ...],   # one per processed image
            "edges_matrices": [                 # ALL gradient matrices
                [[float, ...], ...],            # for image 0
                [[float, ...], ...],            # for image 1
                ...
            ]
        }
    """
    try:
        edge_imgs_b64, all_matrices = extract_edges_service(
            image_index=request.image_index,
            all_images=request.all_images,
            image_indices=request.image_indices,
            method=method,
            low_thresh=low_thresh,
            high_thresh=high_thresh,
            sobel_ksize=sobel_ksize,
        )
        return {"edge_images": edge_imgs_b64, "edges_matrices": all_matrices}
    except Exception as e:
        logging.exception("Edge detection failed")
        raise HTTPException(status_code=500, detail=str(e))
