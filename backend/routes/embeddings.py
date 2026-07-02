"""
Embedding Visualization Routes
================================
GET  /api/embeddings/visualize — 2D projection of sampled embedding space
POST /api/embeddings/query     — visualization for a specific text query
"""

import sys
import numpy as np
from pathlib import Path
from fastapi import APIRouter, Query, HTTPException

from ..schemas.models import VizMethod
from ..services.clip_service import clip_service
from ..services.faiss_service import faiss_service
from ..services.retrieval_enhancements import (
    candidate_pool_size,
    rank_text_to_image_candidates,
)
from ..utils.logger import get_logger

# Import projector from ml_pipeline
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml_pipeline.visualization.projector import (
    build_visualization_points,
    project_pca,
    project_tsne,
    normalize_coordinates,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/api/embeddings", tags=["embeddings"])


@router.get("/visualize")
async def visualize_embeddings(
    n_samples: int = Query(default=200, ge=10, le=1000),
    method: VizMethod = Query(default="pca"),
):
    """
    Return a 2D projection of a random sample from the embedding space.

    Useful for rendering a background visualization of the latent space.
    """
    try:
        sample = faiss_service.get_sample_embeddings(n=n_samples, emb_type="image")

        if method == "tsne":
            coords = project_tsne(sample)
        else:
            coords = project_pca(sample)

        coords = normalize_coordinates(coords)

        # Pairwise mean similarity to centroid (normalized vectors → cosine via dot product)
        centroid = sample.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        sims = sample @ centroid

        points = []
        for i in range(len(coords)):
            points.append({
                "id": f"s{i}",
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "label": f"vec_{i:04x}",
                "kind": "result",
                "similarity": float(sims[i]),
            })

        return {
            "points": points,
            "method": method,
            "n_samples": len(points),
        }

    except Exception as e:
        logger.error(f"Visualization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
async def query_embedding_visualization(
    query: str = Query(..., min_length=1),
    top_k: int = Query(default=8, ge=1, le=50),
    method: VizMethod = Query(default="pca"),
):
    """
    Generate an embedding visualization centered on a text query.

    Returns the query point, its K nearest image neighbors,
    and background context vectors — all projected to 2D.
    """
    try:
        # Encode query with prompt ensembling (matches dashboard retrieval)
        query_emb = clip_service.encode_text_for_retrieval(query)

        img_pool, _ = candidate_pool_size(top_k, faiss_service.image_index.ntotal)
        txt_pool = min(img_pool * 2, faiss_service.text_index.ntotal)
        img_scores, img_indices = faiss_service.search_images(query_emb, img_pool)
        txt_scores, txt_indices = faiss_service.search_texts(query_emb, txt_pool)

        ranked = rank_text_to_image_candidates(
            query_emb,
            img_scores,
            img_indices,
            txt_scores,
            txt_indices,
            faiss_service.metadata,
            faiss_service.text_embeddings,
            top_k,
        )

        result_embeddings = []
        result_metadata = []

        for img_idx, fused_score, _ in ranked:
            if img_idx < 0 or img_idx >= len(faiss_service.metadata["image_filenames"]):
                continue
            result_embeddings.append(faiss_service.image_embeddings[img_idx])
            result_metadata.append({
                "image_filename": faiss_service.metadata["image_filenames"][img_idx],
                "similarity": float(fused_score),
            })

        result_emb = (
            np.array(result_embeddings)
            if result_embeddings
            else np.zeros((0, query_emb.shape[0]))
        )
        background = faiss_service.get_sample_embeddings(n=40, emb_type="image")

        points = build_visualization_points(
            query_emb,
            result_emb,
            result_metadata,
            background_embeddings=background,
            method=method,
        )

        return {
            "points": points,
            "query": query,
            "method": method,
            "top_k": top_k,
        }

    except Exception as e:
        logger.error(f"Query visualization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
