"""
Retrieval Service
==================
Orchestrates the complete retrieval pipeline:
  1. Encode query via CLIP
  2. Search FAISS index
  3. Build result objects
  4. Generate 2D visualization coordinates

Used by the API route handlers.
"""

import sys
import time
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any, List

from .clip_service import clip_service
from .faiss_service import faiss_service
from .retrieval_enhancements import (
    candidate_pool_size,
    rank_text_to_image_candidates,
)
from ..utils.logger import get_logger

# Add project root so we can import ml_pipeline
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml_pipeline.visualization.projector import build_visualization_points

logger = get_logger(__name__)


class RetrievalService:
    """Orchestrates text-to-image and image-to-text retrieval."""

    def search_images_by_text(
        self,
        query: str,
        top_k: int = 8,
        viz_method: str = "pca",
    ) -> Dict[str, Any]:
        """
        Text → Image retrieval.

        Flow:
        1. Encode text query via CLIP
        2. Search FAISS image index
        3. Build ranked results with metadata
        4. Generate 2D visualization points

        Args:
            query: natural language search text
            top_k: number of results
            viz_method: projection method for visualization

        Returns:
            Dict with 'results', 'points', and 'metadata'
        """
        t0 = time.time()

        # 1. Encode query with CLIP prompt ensembling (better than raw text)
        query_embedding = clip_service.encode_text_for_retrieval(query)

        # 2. Hybrid retrieval: image index + text-index bridge + caption rerank
        img_pool, txt_pool = candidate_pool_size(
            top_k, faiss_service.image_index.ntotal
        )
        img_scores, img_indices = faiss_service.search_images(query_embedding, img_pool)
        txt_scores, txt_indices = faiss_service.search_texts(
            query_embedding, min(txt_pool, faiss_service.text_index.ntotal)
        )

        ranked = rank_text_to_image_candidates(
            query_embedding,
            img_scores,
            img_indices,
            txt_scores,
            txt_indices,
            faiss_service.metadata,
            faiss_service.text_embeddings,
            top_k,
        )

        # 3. Build results
        results: List[Dict[str, Any]] = []
        result_embeddings: List[np.ndarray] = []

        for rank, (idx, fused_score, breakdown) in enumerate(ranked):
            if idx < 0 or idx >= len(faiss_service.metadata["image_filenames"]):
                continue

            image_filename = faiss_service.metadata["image_filenames"][idx]

            caption_indices = faiss_service.metadata[
                "image_to_caption_indices"
            ].get(str(idx), [])
            captions = [
                faiss_service.metadata["captions"][ci]
                for ci in caption_indices
                if ci < len(faiss_service.metadata["captions"])
            ]

            results.append({
                "rank": rank + 1,
                "id": f"img_{idx}",
                "image_filename": image_filename,
                "url": f"/static/images/{image_filename}",
                "caption": captions[0] if captions else "",
                "tags": self._extract_tags(captions[0] if captions else ""),
                "similarity": float(fused_score),
            })
            result_embeddings.append(faiss_service.image_embeddings[idx])

        # 4. Build visualization
        result_emb = (
            np.array(result_embeddings)
            if result_embeddings
            else np.zeros((0, query_embedding.shape[0]))
        )
        background = faiss_service.get_sample_embeddings(n=40, emb_type="image")

        points = build_visualization_points(
            query_embedding,
            result_emb,
            results,
            background_embeddings=background,
            method=viz_method,
        )

        elapsed_ms = int((time.time() - t0) * 1000)

        return {
            "results": results,
            "points": points,
            "metadata": {
                "query": query,
                "mode": "text2image",
                "top_k": top_k,
                "retrieval_ms": elapsed_ms,
                "total_indexed": faiss_service.image_index.ntotal,
                "similarity": results[0]["similarity"] if results else 0.0,
                "distance": 1.0 - (results[0]["similarity"] if results else 0.0),
                "confidence": results[0]["similarity"] if results else 0.0,
                "model": "CLIP-ViT-B/32",
                "device": clip_service.device_name,
                "retrieval_mode": "enhanced",
            },
        }

    def search_captions_by_image(
        self,
        image: Image.Image,
        top_k: int = 8,
        viz_method: str = "pca",
    ) -> Dict[str, Any]:
        """
        Image → Text retrieval.

        Flow:
        1. Encode uploaded image via CLIP
        2. Search FAISS text index
        3. Build ranked captions (deduplicated)
        4. Generate 2D visualization points

        Args:
            image: PIL Image in RGB mode
            top_k: number of unique captions to return
            viz_method: projection method for visualization

        Returns:
            Dict with 'results', 'points', and 'metadata'
        """
        t0 = time.time()

        # 1. Encode image
        query_embedding = clip_service.encode_image(image)

        # 2. Search text index (fetch extra to handle dedup)
        raw_k = min(top_k * 3, faiss_service.text_index.ntotal)
        scores, indices = faiss_service.search_texts(query_embedding, raw_k)

        # 3. Build results with deduplication
        results: List[Dict[str, Any]] = []
        result_embeddings: List[np.ndarray] = []
        seen_captions: set = set()

        for score, idx in zip(scores, indices):
            idx = int(idx)
            if idx < 0 or idx >= len(faiss_service.metadata["captions"]):
                continue

            caption = faiss_service.metadata["captions"][idx]
            if caption in seen_captions:
                continue
            seen_captions.add(caption)

            image_idx = faiss_service.metadata["caption_to_image_idx"][idx]
            image_filename = faiss_service.metadata["image_filenames"][image_idx]

            results.append({
                "rank": len(results) + 1,
                "caption": caption,
                "image_filename": image_filename,
                "similarity": float(score),
            })
            result_embeddings.append(faiss_service.text_embeddings[idx])

            if len(results) >= top_k:
                break

        # 4. Build visualization
        result_emb = (
            np.array(result_embeddings)
            if result_embeddings
            else np.zeros((0, query_embedding.shape[0]))
        )
        background = faiss_service.get_sample_embeddings(n=40, emb_type="text")

        points = build_visualization_points(
            query_embedding,
            result_emb,
            results,
            background_embeddings=background,
            method=viz_method,
        )

        elapsed_ms = int((time.time() - t0) * 1000)

        return {
            "results": results,
            "points": points,
            "metadata": {
                "query": "uploaded_image",
                "mode": "image2text",
                "top_k": top_k,
                "retrieval_ms": elapsed_ms,
                "total_indexed": faiss_service.text_index.ntotal,
                "similarity": results[0]["similarity"] if results else 0.0,
                "distance": 1.0 - (results[0]["similarity"] if results else 0.0),
                "confidence": results[0]["similarity"] if results else 0.0,
                "model": "CLIP-ViT-B/32",
                "device": clip_service.device_name,
            },
        }

    @staticmethod
    def _extract_tags(caption: str) -> List[str]:
        """Extract simple keyword tags from a caption string."""
        stop_words = {
            "a", "an", "the", "is", "are", "in", "on", "at", "to", "of",
            "and", "with", "for", "it", "its", "has", "have", "was", "were",
            "be", "been", "being", "that", "this", "from", "they", "their",
            "two", "three", "some", "each", "into",
        }
        words = caption.lower().replace(".", "").replace(",", "").split()
        tags = [w for w in words if len(w) > 2 and w not in stop_words]
        return tags[:5]


# Singleton instance
retrieval_service = RetrievalService()
