"""
Retrieval Evaluation Runner
=============================
Evaluates text→image retrieval quality on the Flickr30k dataset.

For each image, uses its first caption as a query and checks whether
the correct image is retrieved in the top-K results.

Usage:
    cd project_root
    python -m ml_pipeline.evaluation.run_evaluation
"""

import sys
import json
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml_pipeline.evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    mean_reciprocal_rank,
    average_precision,
)


def main():
    print("=" * 60)
    print("  SEMANTIX — Retrieval Evaluation")
    print("=" * 60)

    embeddings_dir = PROJECT_ROOT / "embeddings"
    faiss_dir = PROJECT_ROOT / "faiss_index"

    # --- Load data ---
    print("\n[1/3] Loading data...")
    image_embeddings = np.load(embeddings_dir / "image_embeddings.npy")
    text_embeddings = np.load(embeddings_dir / "text_embeddings.npy")

    with open(embeddings_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    image_index = faiss.read_index(str(faiss_dir / "image_faiss.index"))

    print(f"  Image embeddings: {image_embeddings.shape}")
    print(f"  Text embeddings:  {text_embeddings.shape}")
    print(f"  Image index:      {image_index.ntotal} vectors")

    # --- Evaluate Text → Image retrieval ---
    print("\n[2/3] Evaluating Text → Image retrieval...")

    # Sample images for evaluation (full eval takes a while)
    n_total = metadata["total_images"]
    n_eval = min(1000, n_total)
    rng = np.random.RandomState(42)
    eval_indices = rng.choice(n_total, n_eval, replace=False)

    k_values = [1, 5, 10]
    max_k = max(k_values)

    recalls = {k: [] for k in k_values}
    precisions = {k: [] for k in k_values}
    mrrs = []
    aps = []

    for img_idx in tqdm(eval_indices, desc="  Evaluating"):
        caption_indices = metadata["image_to_caption_indices"].get(str(img_idx), [])
        if not caption_indices:
            continue

        # Use the first caption as the query
        caption_idx = caption_indices[0]
        query_emb = text_embeddings[caption_idx].reshape(1, -1).astype(np.float32)

        # Search the image index
        scores, retrieved = image_index.search(query_emb, max_k)
        retrieved = retrieved[0].tolist()

        # Ground truth: the image this caption belongs to
        relevant = {int(img_idx)}

        for k in k_values:
            recalls[k].append(recall_at_k(retrieved, relevant, k))
            precisions[k].append(precision_at_k(retrieved, relevant, k))

        mrrs.append(mean_reciprocal_rank(retrieved, relevant))
        aps.append(average_precision(retrieved, relevant))

    # --- Print results ---
    print(f"\n[3/3] Results")
    print(f"\n{'=' * 50}")
    print(f"  Text → Image Retrieval  (n = {n_eval} queries)")
    print(f"{'=' * 50}")
    for k in k_values:
        avg_recall = np.mean(recalls[k]) * 100
        avg_precision = np.mean(precisions[k]) * 100
        print(f"  Recall@{k:<3}   : {avg_recall:6.2f}%")
        print(f"  Precision@{k:<3}: {avg_precision:6.2f}%")
    print(f"  {'─' * 30}")
    print(f"  MRR          : {np.mean(mrrs):.4f}")
    print(f"  MAP          : {np.mean(aps):.4f}")
    print(f"{'=' * 50}")

    # Interpretation
    r1 = np.mean(recalls[1]) * 100
    r5 = np.mean(recalls[5]) * 100
    r10 = np.mean(recalls[10]) * 100
    print(f"\n  Summary: R@1={r1:.1f}%  R@5={r5:.1f}%  R@10={r10:.1f}%")
    print(f"  (CLIP ViT-B/32 baseline on Flickr30k: ~58% R@1, ~83% R@5, ~90% R@10)")


if __name__ == "__main__":
    main()
