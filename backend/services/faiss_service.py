"""
FAISS Service
==============
Manages FAISS vector indexes and dataset metadata.
Loads persisted indexes and embeddings from disk at startup.
"""

import json
import numpy as np
import faiss
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)


class FAISSService:
    """Manages FAISS indexes, embeddings, and metadata."""

    def __init__(self):
        self.image_index: faiss.IndexFlatIP | None = None
        self.text_index: faiss.IndexFlatIP | None = None
        self.metadata: dict | None = None
        self.image_embeddings: np.ndarray | None = None
        self.text_embeddings: np.ndarray | None = None
        self._loaded = False

    def load(self, project_root: Path):
        """
        Load FAISS indexes, embeddings, and metadata from disk.

        Expects:
            project_root/faiss_index/image_faiss.index
            project_root/faiss_index/text_faiss.index
            project_root/embeddings/image_embeddings.npy
            project_root/embeddings/text_embeddings.npy
            project_root/embeddings/metadata.json

        Raises:
            FileNotFoundError if indexes or embeddings are missing.
        """
        faiss_dir = project_root / "faiss_index"
        embeddings_dir = project_root / "embeddings"

        # Check that pipeline has been run
        image_idx_path = faiss_dir / "image_faiss.index"
        text_idx_path = faiss_dir / "text_faiss.index"
        img_emb_path = embeddings_dir / "image_embeddings.npy"
        txt_emb_path = embeddings_dir / "text_embeddings.npy"
        meta_path = embeddings_dir / "metadata.json"

        missing = []
        for p in [image_idx_path, text_idx_path, img_emb_path, txt_emb_path, meta_path]:
            if not p.exists():
                missing.append(str(p))

        if missing:
            raise FileNotFoundError(
                "Required files not found — run the ML pipeline first:\n"
                "  python -m ml_pipeline.embedding_generator.generate_embeddings\n"
                "  python -m ml_pipeline.faiss_manager.build_index\n\n"
                f"Missing: {', '.join(missing)}"
            )

        # Load FAISS indexes
        logger.info("Loading FAISS indexes...")
        self.image_index = faiss.read_index(str(image_idx_path))
        self.text_index = faiss.read_index(str(text_idx_path))
        logger.info(f"  Image index: {self.image_index.ntotal} vectors")
        logger.info(f"  Text index:  {self.text_index.ntotal} vectors")

        # Load embeddings (needed for visualization)
        logger.info("Loading embeddings for visualization...")
        self.image_embeddings = np.load(img_emb_path)
        self.text_embeddings = np.load(txt_emb_path)

        # Load metadata
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        logger.info(
            f"  Metadata: {self.metadata['total_images']} images, "
            f"{self.metadata['total_captions']} captions"
        )

        self._loaded = True
        logger.info("✓ FAISS service ready")

    def search_images(
        self, query_embedding: np.ndarray, top_k: int = 8
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search the image index.

        Args:
            query_embedding: (D,) normalized query vector
            top_k: number of results

        Returns:
            (scores, indices) — both shape (top_k,)
        """
        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.image_index.search(query, top_k)
        return scores[0], indices[0]

    def search_texts(
        self, query_embedding: np.ndarray, top_k: int = 8
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search the text index.

        Args:
            query_embedding: (D,) normalized query vector
            top_k: number of results

        Returns:
            (scores, indices) — both shape (top_k,)
        """
        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.text_index.search(query, top_k)
        return scores[0], indices[0]

    def get_sample_embeddings(
        self, n: int = 200, emb_type: str = "image"
    ) -> np.ndarray:
        """
        Get a random sample of embeddings for visualization context.

        Args:
            n: number of samples
            emb_type: "image" or "text"

        Returns:
            np.ndarray of shape (n, D)
        """
        embeddings = (
            self.image_embeddings if emb_type == "image" else self.text_embeddings
        )
        if embeddings is None or len(embeddings) == 0:
            return np.zeros((0, 512), dtype=np.float32)
        if len(embeddings) <= n:
            return embeddings
        rng = np.random.RandomState(42)
        indices = rng.choice(len(embeddings), n, replace=False)
        return embeddings[indices]

    @property
    def is_loaded(self) -> bool:
        """Whether indexes and metadata are loaded."""
        return self._loaded


# Singleton instance
faiss_service = FAISSService()
