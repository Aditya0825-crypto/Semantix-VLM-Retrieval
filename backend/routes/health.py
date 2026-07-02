"""
Health Check Route
===================
GET /api/health — system health and readiness status
"""

from fastapi import APIRouter

from ..services.clip_service import clip_service
from ..services.faiss_service import faiss_service
from ..schemas.models import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Return backend health status.

    Includes model loading state, FAISS index stats,
    dataset counts, and device information.
    """
    meta = faiss_service.metadata or {}

    return HealthResponse(
        status="ok" if (clip_service.is_loaded and faiss_service.is_loaded) else "degraded",
        model_loaded=clip_service.is_loaded,
        faiss_loaded=faiss_service.is_loaded,
        device=clip_service.device_name,
        total_images=meta.get("total_images", 0),
        total_captions=meta.get("total_captions", 0),
        embedding_dim=meta.get("embedding_dim", 0),
        image_index_size=faiss_service.image_index.ntotal if faiss_service.image_index else 0,
        text_index_size=faiss_service.text_index.ntotal if faiss_service.text_index else 0,
    )
