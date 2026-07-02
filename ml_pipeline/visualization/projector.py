"""
Embedding Visualization Projector
==================================
Projects high-dimensional CLIP embeddings (512-d) to 2D coordinates
for frontend visualization using PCA, t-SNE, or UMAP.

Output format matches the frontend EmbeddingPoint type:
    { id, x, y, label, kind, similarity? }
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Dict, Any, Optional


def project_pca(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Project embeddings to 2D using PCA (fast, deterministic)."""
    pca = PCA(n_components=n_components)
    return pca.fit_transform(embeddings.astype(np.float64))


def project_tsne(
    embeddings: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
) -> np.ndarray:
    """Project embeddings to 2D using t-SNE (slower, reveals clusters)."""
    n_samples = embeddings.shape[0]
    # t-SNE requires perplexity < n_samples
    perp = min(perplexity, max(2.0, n_samples - 1.0))
    tsne = TSNE(
        n_components=n_components,
        perplexity=perp,
        random_state=42,
        max_iter=500,
    )
    return tsne.fit_transform(embeddings.astype(np.float64))


def project_umap(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Project embeddings to 2D using UMAP (optional dependency)."""
    try:
        import umap
    except ImportError:
        raise ImportError(
            "umap-learn is not installed. Install with: pip install umap-learn"
        )
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    return reducer.fit_transform(embeddings.astype(np.float64))


def normalize_coordinates(coords: np.ndarray) -> np.ndarray:
    """
    Normalize 2D coordinates to [-0.85, 0.85] range.
    The frontend EmbeddingCanvas expects x/y in roughly [-1, 1].
    """
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    ranges = maxs - mins
    # Avoid division by zero
    ranges[ranges == 0] = 1.0
    # Map to [-1, 1] then scale down slightly to avoid edge clipping
    normalized = 2.0 * (coords - mins) / ranges - 1.0
    return normalized * 0.85


def build_visualization_points(
    query_embedding: np.ndarray,
    result_embeddings: np.ndarray,
    result_metadata: List[Dict[str, Any]],
    background_embeddings: Optional[np.ndarray] = None,
    method: str = "pca",
) -> List[Dict[str, Any]]:
    """
    Build frontend-ready visualization points by projecting embeddings to 2D.

    Args:
        query_embedding: (D,) — the query vector
        result_embeddings: (K, D) — top-K result vectors
        result_metadata: list of dicts with at least 'similarity' and a label field
        background_embeddings: optional (N, D) — background vectors for context
        method: projection method — "pca", "tsne", or "umap"

    Returns:
        List of dicts matching frontend EmbeddingPoint:
        [{ id, x, y, label, kind, similarity? }, ...]
    """
    # Assemble all vectors for joint projection
    parts = [query_embedding.reshape(1, -1)]

    n_results = result_embeddings.shape[0] if len(result_embeddings.shape) == 2 else 0
    if n_results > 0:
        parts.append(result_embeddings)

    n_background = 0
    if background_embeddings is not None and len(background_embeddings) > 0:
        parts.append(background_embeddings)
        n_background = background_embeddings.shape[0]

    combined = np.concatenate(parts, axis=0).astype(np.float32)

    # Need at least 3 points for meaningful projection
    if combined.shape[0] < 3:
        # Pad with slight noise copies
        pad_count = 3 - combined.shape[0]
        noise = np.random.randn(pad_count, combined.shape[1]).astype(np.float32) * 0.01
        combined = np.concatenate([combined, noise], axis=0)

    # Project
    if method == "tsne":
        coords_2d = project_tsne(combined)
    elif method == "umap":
        coords_2d = project_umap(combined)
    else:
        coords_2d = project_pca(combined)

    # Normalize to [-0.85, 0.85]
    coords_2d = normalize_coordinates(coords_2d)

    points: List[Dict[str, Any]] = []

    # Query point (index 0)
    points.append({
        "id": "query",
        "x": float(coords_2d[0, 0]),
        "y": float(coords_2d[0, 1]),
        "label": "query",
        "kind": "query",
    })

    # Result points (indices 1 .. n_results)
    for i in range(n_results):
        meta = result_metadata[i] if i < len(result_metadata) else {}
        label = meta.get("caption", meta.get("image_filename", f"match_{i + 1}"))
        points.append({
            "id": f"r{i}",
            "x": float(coords_2d[1 + i, 0]),
            "y": float(coords_2d[1 + i, 1]),
            "label": str(label)[:32],
            "kind": "result",
            "similarity": float(meta.get("similarity", 0.0)),
        })

    # Background points — cosine similarity to query (vectors are L2-normalized)
    for i in range(n_background):
        idx = 1 + n_results + i
        if idx >= coords_2d.shape[0]:
            break
        bg_vec = background_embeddings[i]
        sim = float(np.dot(query_embedding, bg_vec))
        points.append({
            "id": f"n{i}",
            "x": float(coords_2d[idx, 0]),
            "y": float(coords_2d[idx, 1]),
            "label": f"vec_{i:04x}",
            "kind": "result",
            "similarity": sim,
        })

    return points
