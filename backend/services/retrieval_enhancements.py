"""
Retrieval Enhancements
======================
Improves text→image accuracy beyond raw FAISS + cosine similarity.

Cosine similarity is already optimal when vectors are L2-normalized (IndexFlatIP).
These techniques address the real bottlenecks:
  1. CLIP prompt ensembling — queries match training distribution better
  2. Text-index bridge — search captions, map to images (text↔text alignment)
  3. Caption reranking — re-score candidates using all captions per image
  4. Score fusion — combine signals before final ranking
  5. Minimum similarity filter — drop low-confidence matches
"""

from __future__ import annotations

import numpy as np
from typing import Any

# CLIP zero-shot prompt templates (standard practice for ViT-B/32)
QUERY_PROMPT_TEMPLATES = (
    "a photo of {}",
    "a picture of {}",
    "an image of {}",
    "a photograph of {}",
    "{}",
)

# Fusion weights (image FAISS, text-index bridge, caption rerank)
WEIGHT_IMAGE = 0.50
WEIGHT_TEXT_BRIDGE = 0.30
WEIGHT_CAPTION = 0.20

# Drop results below this cosine similarity (typical CLIP noise floor ~0.12–0.18)
MIN_SIMILARITY = 0.15

# Over-fetch factor for candidate pool before fusion/rerank
CANDIDATE_MULTIPLIER = 5
MIN_CANDIDATES = 50


def fuse_retrieval_score(
    image_sim: float,
    text_bridge_sim: float,
    caption_sim: float,
) -> float:
    """Weighted fusion of three cosine-similarity signals."""
    return (
        WEIGHT_IMAGE * image_sim
        + WEIGHT_TEXT_BRIDGE * text_bridge_sim
        + WEIGHT_CAPTION * caption_sim
    )


def max_caption_similarity(
    query_embedding: np.ndarray,
    image_idx: int,
    metadata: dict,
    text_embeddings: np.ndarray,
) -> float:
    """Max cosine similarity between query and any caption for this image."""
    caption_indices = metadata["image_to_caption_indices"].get(str(image_idx), [])
    if not caption_indices:
        return 0.0
    valid = [ci for ci in caption_indices if ci < len(text_embeddings)]
    if not valid:
        return 0.0
    cap_embs = text_embeddings[valid]
    # Vectors are L2-normalized → dot product = cosine similarity
    return float(np.max(cap_embs @ query_embedding))


def build_text_bridge_scores(
    query_embedding: np.ndarray,
    text_scores: np.ndarray,
    text_indices: np.ndarray,
    metadata: dict,
) -> dict[int, float]:
    """
    Map text-index hits to parent images, keeping the best score per image.
    """
    bridge: dict[int, float] = {}
    for score, cap_idx in zip(text_scores, text_indices):
        cap_idx = int(cap_idx)
        if cap_idx < 0 or cap_idx >= len(metadata["caption_to_image_idx"]):
            continue
        img_idx = int(metadata["caption_to_image_idx"][cap_idx])
        sim = float(score)
        if sim > bridge.get(img_idx, 0.0):
            bridge[img_idx] = sim
    return bridge


def rank_text_to_image_candidates(
    query_embedding: np.ndarray,
    image_scores: np.ndarray,
    image_indices: np.ndarray,
    text_scores: np.ndarray,
    text_indices: np.ndarray,
    metadata: dict,
    text_embeddings: np.ndarray,
    top_k: int,
    min_similarity: float = MIN_SIMILARITY,
) -> list[tuple[int, float, dict[str, float]]]:
    """
    Fuse image-index, text-bridge, and caption signals; return ranked candidates.

    Returns:
        List of (image_idx, fused_score, score_breakdown) sorted by fused_score desc.
    """
    text_bridge = build_text_bridge_scores(
        query_embedding, text_scores, text_indices, metadata
    )

    candidates: dict[int, dict[str, float]] = {}

    for score, idx in zip(image_scores, image_indices):
        idx = int(idx)
        if idx < 0:
            continue
        candidates.setdefault(idx, {"image_sim": 0.0, "text_bridge_sim": 0.0, "caption_sim": 0.0})
        candidates[idx]["image_sim"] = max(candidates[idx]["image_sim"], float(score))

    for img_idx, sim in text_bridge.items():
        candidates.setdefault(img_idx, {"image_sim": 0.0, "text_bridge_sim": 0.0, "caption_sim": 0.0})
        candidates[img_idx]["text_bridge_sim"] = max(candidates[img_idx]["text_bridge_sim"], sim)

    for img_idx in candidates:
        candidates[img_idx]["caption_sim"] = max_caption_similarity(
            query_embedding, img_idx, metadata, text_embeddings
        )

    ranked: list[tuple[int, float, dict[str, float]]] = []
    for img_idx, parts in candidates.items():
        fused = fuse_retrieval_score(
            parts["image_sim"], parts["text_bridge_sim"], parts["caption_sim"]
        )
        if fused >= min_similarity:
            ranked.append((img_idx, fused, parts))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def candidate_pool_size(top_k: int, index_size: int) -> tuple[int, int]:
    """Return (image_candidates, text_candidates) fetch sizes."""
    pool = min(max(top_k * CANDIDATE_MULTIPLIER, MIN_CANDIDATES), index_size)
    text_pool = min(pool * 2, index_size)
    return pool, text_pool
