"""
Retrieval Evaluation Metrics
=============================
Standard information retrieval metrics for evaluating
the semantic search quality of the CLIP + FAISS pipeline.
"""

import numpy as np
from typing import List, Set


def recall_at_k(
    retrieved_indices: List[int],
    relevant_indices: Set[int],
    k: int,
) -> float:
    """
    Recall@K — fraction of relevant items found in the top-K results.

    Args:
        retrieved_indices: ordered list of retrieved item indices
        relevant_indices: set of ground-truth relevant indices
        k: cutoff rank

    Returns:
        recall score in [0, 1]
    """
    if not relevant_indices:
        return 0.0
    retrieved_top_k = set(retrieved_indices[:k])
    hits = len(retrieved_top_k & relevant_indices)
    return hits / len(relevant_indices)


def precision_at_k(
    retrieved_indices: List[int],
    relevant_indices: Set[int],
    k: int,
) -> float:
    """
    Precision@K — fraction of top-K results that are relevant.

    Args:
        retrieved_indices: ordered list of retrieved item indices
        relevant_indices: set of ground-truth relevant indices
        k: cutoff rank

    Returns:
        precision score in [0, 1]
    """
    if k == 0:
        return 0.0
    retrieved_top_k = retrieved_indices[:k]
    hits = sum(1 for idx in retrieved_top_k if idx in relevant_indices)
    return hits / k


def mean_reciprocal_rank(
    retrieved_indices: List[int],
    relevant_indices: Set[int],
) -> float:
    """
    Mean Reciprocal Rank (MRR) — inverse of the rank of the first relevant result.

    Args:
        retrieved_indices: ordered list of retrieved item indices
        relevant_indices: set of ground-truth relevant indices

    Returns:
        reciprocal rank in [0, 1], or 0 if no relevant item found
    """
    for rank, idx in enumerate(retrieved_indices, 1):
        if idx in relevant_indices:
            return 1.0 / rank
    return 0.0


def average_precision(
    retrieved_indices: List[int],
    relevant_indices: Set[int],
) -> float:
    """
    Average Precision (AP) — area under the precision-recall curve for a single query.

    Args:
        retrieved_indices: ordered list of retrieved item indices
        relevant_indices: set of ground-truth relevant indices

    Returns:
        average precision in [0, 1]
    """
    if not relevant_indices:
        return 0.0

    hits = 0
    sum_precisions = 0.0

    for rank, idx in enumerate(retrieved_indices, 1):
        if idx in relevant_indices:
            hits += 1
            sum_precisions += hits / rank

    return sum_precisions / len(relevant_indices)


def ndcg_at_k(
    retrieved_indices: List[int],
    relevant_indices: Set[int],
    k: int,
) -> float:
    """
    Normalized Discounted Cumulative Gain (nDCG@K).

    Binary relevance: relevant items get gain=1, others get gain=0.

    Args:
        retrieved_indices: ordered list of retrieved item indices
        relevant_indices: set of ground-truth relevant indices
        k: cutoff rank

    Returns:
        nDCG score in [0, 1]
    """
    # DCG
    dcg = 0.0
    for i, idx in enumerate(retrieved_indices[:k]):
        rel = 1.0 if idx in relevant_indices else 0.0
        dcg += rel / np.log2(i + 2)  # i+2 because rank starts at 1, log2(1)=0

    # Ideal DCG
    n_relevant = min(len(relevant_indices), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))

    if idcg == 0:
        return 0.0
    return dcg / idcg
