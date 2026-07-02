"""
Retrieval Engine
================
Core semantic similarity search using FAISS.
Provides functions for image and caption retrieval.
"""

import numpy as np
import faiss
from typing import List, Dict, Any, Tuple


def search_index(
    query_embedding: np.ndarray,
    index: faiss.IndexFlatIP,
    top_k: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search a FAISS index for nearest neighbors.

    Uses inner product (= cosine similarity for L2-normalized vectors).

    Args:
        query_embedding: (D,) normalized query vector
        index: FAISS IndexFlatIP
        top_k: number of results to retrieve

    Returns:
        (scores, indices) — both shape (top_k,)
    """
    query = query_embedding.reshape(1, -1).astype(np.float32)
    scores, indices = index.search(query, top_k)
    return scores[0], indices[0]


def retrieve_images(
    query_embedding: np.ndarray,
    image_index: faiss.IndexFlatIP,
    metadata: dict,
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """
    Text → Image retrieval.

    Given a text query embedding, find the top-K most similar images.

    Args:
        query_embedding: (D,) normalized text embedding
        image_index: FAISS index over image embeddings
        metadata: dataset metadata dict
        top_k: number of results

    Returns:
        List of result dicts with rank, image info, similarity, etc.
    """
    scores, indices = search_index(query_embedding, image_index, top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores, indices)):
        idx = int(idx)
        if idx < 0 or idx >= len(metadata["image_filenames"]):
            continue

        image_filename = metadata["image_filenames"][idx]

        # Get captions associated with this image
        caption_indices = metadata["image_to_caption_indices"].get(str(idx), [])
        captions = [
            metadata["captions"][ci]
            for ci in caption_indices
            if ci < len(metadata["captions"])
        ]

        results.append({
            "rank": rank + 1,
            "image_id": f"img_{idx}",
            "image_filename": image_filename,
            "caption": captions[0] if captions else "",
            "all_captions": captions,
            "similarity": float(score),
            "index": idx,
        })

    return results


def retrieve_captions(
    query_embedding: np.ndarray,
    text_index: faiss.IndexFlatIP,
    metadata: dict,
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """
    Image → Text retrieval.

    Given an image query embedding, find the top-K most similar captions.
    Deduplicates identical caption strings.

    Args:
        query_embedding: (D,) normalized image embedding
        text_index: FAISS index over text embeddings
        metadata: dataset metadata dict
        top_k: number of unique results

    Returns:
        List of result dicts with rank, caption text, similarity, etc.
    """
    # Retrieve more than top_k to account for duplicates
    raw_k = min(top_k * 3, text_index.ntotal)
    scores, indices = search_index(query_embedding, text_index, raw_k)

    results = []
    seen_captions: set = set()

    for score, idx in zip(scores, indices):
        idx = int(idx)
        if idx < 0 or idx >= len(metadata["captions"]):
            continue

        caption = metadata["captions"][idx]
        if caption in seen_captions:
            continue
        seen_captions.add(caption)

        # Find the image this caption belongs to
        image_idx = metadata["caption_to_image_idx"][idx]
        image_filename = metadata["image_filenames"][image_idx]

        results.append({
            "rank": len(results) + 1,
            "caption": caption,
            "image_filename": image_filename,
            "image_id": f"img_{image_idx}",
            "similarity": float(score),
            "index": idx,
        })

        if len(results) >= top_k:
            break

    return results
