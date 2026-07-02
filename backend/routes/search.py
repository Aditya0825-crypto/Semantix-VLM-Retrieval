"""
Search API Routes
==================
POST /api/search/text-to-image — semantic text→image retrieval
POST /api/search/image-to-text — semantic image→text retrieval
"""

import io

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image

from ..services.retrieval_service import retrieval_service
from ..schemas.models import TextSearchRequest, VizMethod
from ..utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("/text-to-image")
async def text_to_image(request: TextSearchRequest):
    """
    Search images by text query.

    Uses CLIP to encode the query, then searches the FAISS image index
    for the most semantically similar images.
    """
    try:
        logger.info(f"Text→Image search: '{request.query}' (top_k={request.top_k})")
        result = retrieval_service.search_images_by_text(
            query=request.query,
            top_k=request.top_k,
            viz_method=request.viz_method,
        )
        logger.info(
            f"  → {len(result['results'])} results in {result['metadata']['retrieval_ms']}ms"
        )
        return result
    except Exception as e:
        logger.error(f"Text→Image search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/image-to-text")
async def image_to_text(
    file: UploadFile = File(..., description="Image file to search with"),
    top_k: int = Form(default=8, ge=1, le=50),
    viz_method: VizMethod = Form(default="pca"),
):
    """
    Search captions by uploading an image.

    Uses CLIP to encode the image, then searches the FAISS text index
    for the most semantically similar captions.
    """
    try:
        # Validate file type
        content_type = file.content_type or ""
        if not content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"Expected image file, got {content_type}",
            )

        # Read and decode image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        logger.info(
            f"Image→Text search: {file.filename} ({image.size}) (top_k={top_k})"
        )
        result = retrieval_service.search_captions_by_image(
            image=image,
            top_k=top_k,
            viz_method=viz_method,
        )
        logger.info(
            f"  → {len(result['results'])} results in {result['metadata']['retrieval_ms']}ms"
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image→Text search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
