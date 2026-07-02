"""
Pydantic Schemas
=================
Request/response models for all Semantix API endpoints.
Shapes are designed to match what the frontend expects.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

VizMethod = Literal["pca", "tsne", "umap"]


# ── Request Models ────────────────────────────────────────────


class TextSearchRequest(BaseModel):
    """Request body for POST /api/search/text-to-image"""

    query: str = Field(..., min_length=1, description="Natural language search query")
    top_k: int = Field(default=8, ge=1, le=50, description="Number of results to return")
    viz_method: VizMethod = Field(
        default="pca",
        description="Visualization projection method: pca, tsne, or umap",
    )


# ── Response Components ──────────────────────────────────────


class ImageResult(BaseModel):
    """A single text→image retrieval result."""

    rank: int
    id: str
    image_filename: str
    url: str
    caption: str
    tags: List[str]
    similarity: float


class CaptionResult(BaseModel):
    """A single image→text retrieval result."""

    rank: int
    caption: str
    image_filename: str
    similarity: float


class EmbeddingPoint(BaseModel):
    """
    A 2D-projected embedding point for frontend visualization.

    Matches the frontend EmbeddingCanvas component's EmbeddingPoint type:
    { id, x, y, label, kind, similarity? }
    """

    id: str
    x: float = Field(description="X coordinate in [-1, 1] range")
    y: float = Field(description="Y coordinate in [-1, 1] range")
    label: str
    kind: str = Field(description="Point type: query, result, or cluster")
    similarity: Optional[float] = Field(
        default=None, description="Cosine similarity to query (0–1)"
    )


class RetrievalMetadata(BaseModel):
    """Metadata about a retrieval operation."""

    query: str
    mode: str  # "text2image" or "image2text"
    top_k: int
    retrieval_ms: int
    total_indexed: int
    similarity: float
    distance: float
    confidence: float
    model: str
    device: str


# ── Top-level Response Models ────────────────────────────────


class TextSearchResponse(BaseModel):
    """Response for POST /api/search/text-to-image"""

    results: List[ImageResult]
    points: List[EmbeddingPoint]
    metadata: RetrievalMetadata


class ImageSearchResponse(BaseModel):
    """Response for POST /api/search/image-to-text"""

    results: List[CaptionResult]
    points: List[EmbeddingPoint]
    metadata: RetrievalMetadata


class HealthResponse(BaseModel):
    """Response for GET /api/health"""

    status: str
    model_loaded: bool
    faiss_loaded: bool
    device: str
    total_images: int
    total_captions: int
    embedding_dim: int
    image_index_size: int
    text_index_size: int
