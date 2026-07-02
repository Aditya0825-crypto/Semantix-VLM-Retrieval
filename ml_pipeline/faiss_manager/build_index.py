"""
FAISS Index Builder
====================
Builds and persists FAISS IndexFlatIP indexes from saved embeddings.

For L2-normalized vectors, Inner Product = Cosine Similarity.

Usage:
    cd project_root
    python -m ml_pipeline.faiss_manager.build_index
"""

import sys
import time
import numpy as np
import faiss
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS IndexFlatIP (Inner Product search).

    Since embeddings are L2-normalized, inner product equals cosine similarity.

    Args:
        embeddings: (N, D) float32 array of normalized vectors

    Returns:
        FAISS index with N vectors
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def main():
    print("=" * 60)
    print("  SEMANTIX — FAISS Index Builder")
    print("=" * 60)

    embeddings_dir = PROJECT_ROOT / "embeddings"
    output_dir = PROJECT_ROOT / "faiss_index"
    output_dir.mkdir(exist_ok=True)

    # Verify embeddings exist
    img_path = embeddings_dir / "image_embeddings.npy"
    txt_path = embeddings_dir / "text_embeddings.npy"

    if not img_path.exists() or not txt_path.exists():
        print("\n  ✗ Embeddings not found!")
        print("  Run this first: python -m ml_pipeline.embedding_generator.generate_embeddings")
        sys.exit(1)

    # --- Load embeddings ---
    print("\n[1/4] Loading embeddings...")
    image_embeddings = np.load(img_path)
    text_embeddings = np.load(txt_path)
    print(f"  Image embeddings: {image_embeddings.shape}")
    print(f"  Text embeddings:  {text_embeddings.shape}")

    # --- Build image index ---
    print("\n[2/4] Building image FAISS index...")
    t0 = time.time()
    image_index = build_faiss_index(image_embeddings)
    img_time = time.time() - t0
    print(f"  ✓ {image_index.ntotal} vectors indexed in {img_time:.2f}s")

    # --- Build text index ---
    print("\n[3/4] Building text FAISS index...")
    t0 = time.time()
    text_index = build_faiss_index(text_embeddings)
    txt_time = time.time() - t0
    print(f"  ✓ {text_index.ntotal} vectors indexed in {txt_time:.2f}s")

    # --- Save indexes ---
    print("\n[4/4] Saving indexes to disk...")
    image_index_path = output_dir / "image_faiss.index"
    text_index_path = output_dir / "text_faiss.index"

    faiss.write_index(image_index, str(image_index_path))
    faiss.write_index(text_index, str(text_index_path))

    img_size = image_index_path.stat().st_size / 1e6
    txt_size = text_index_path.stat().st_size / 1e6

    print(f"\n{'=' * 60}")
    print("  ✓ FAISS indexes built successfully!")
    print(f"{'=' * 60}")
    print(f"  image_faiss.index : {image_index.ntotal} vectors, dim={image_embeddings.shape[1]}  ({img_size:.1f} MB)")
    print(f"  text_faiss.index  : {text_index.ntotal} vectors, dim={text_embeddings.shape[1]}  ({txt_size:.1f} MB)")
    print(f"  Output directory  : {output_dir}")
    print(f"{'=' * 60}")
    print("\nNext step: uvicorn backend.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
