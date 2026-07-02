"""
Embedding Generator — OpenCLIP ViT-B-32
=========================================
Generates normalized image and text embeddings for the entire Flickr30k dataset.
Saves embeddings and metadata to the embeddings/ directory.

Usage:
    cd project_root
    python -m ml_pipeline.embedding_generator.generate_embeddings

This is a ONE-TIME operation. Results are persisted to disk.
"""

import os
import sys
import json
import time
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml_pipeline.preprocessing.dataset_loader import load_dataset


def get_device() -> torch.device:
    """Auto-detect CUDA/CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  GPU detected: {gpu_name}")
        return device
    print("  No GPU detected — using CPU (this will be slower)")
    return torch.device("cpu")


def generate_image_embeddings(
    model,
    preprocess,
    image_paths: list,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Generate L2-normalized image embeddings in batches.

    Args:
        model: OpenCLIP model
        preprocess: image preprocessing transform
        image_paths: list of absolute paths to images
        device: torch device
        batch_size: images per batch

    Returns:
        np.ndarray of shape (N, D) with normalized embeddings
    """
    all_embeddings = []
    failed_indices = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="  Encoding images"):
        batch_paths = image_paths[i : i + batch_size]
        images = []

        for j, path in enumerate(batch_paths):
            try:
                img = Image.open(path).convert("RGB")
                img_tensor = preprocess(img)
                images.append(img_tensor)
            except Exception as e:
                # Use a zero tensor as placeholder for corrupted images
                print(f"\n  ⚠ Error loading {os.path.basename(path)}: {e}")
                images.append(torch.zeros(3, 224, 224))
                failed_indices.append(i + j)

        batch = torch.stack(images).to(device)

        with torch.no_grad(), torch.amp.autocast(device_type=str(device)):
            embeddings = model.encode_image(batch)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        all_embeddings.append(embeddings.cpu().float().numpy())

    if failed_indices:
        print(f"  ⚠ {len(failed_indices)} images failed to load (zero-filled)")

    return np.concatenate(all_embeddings, axis=0)


def generate_text_embeddings(
    model,
    tokenizer,
    captions: list,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Generate L2-normalized text embeddings in batches.

    Args:
        model: OpenCLIP model
        tokenizer: OpenCLIP tokenizer
        captions: list of caption strings
        device: torch device
        batch_size: captions per batch

    Returns:
        np.ndarray of shape (N, D) with normalized embeddings
    """
    all_embeddings = []

    for i in tqdm(range(0, len(captions), batch_size), desc="  Encoding captions"):
        batch_captions = captions[i : i + batch_size]
        tokens = tokenizer(batch_captions).to(device)

        with torch.no_grad(), torch.amp.autocast(device_type=str(device)):
            embeddings = model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        all_embeddings.append(embeddings.cpu().float().numpy())

    return np.concatenate(all_embeddings, axis=0)


def main():
    print("=" * 60)
    print("  SEMANTIX — Embedding Generator")
    print("  OpenCLIP ViT-B-32 · laion2b_s34b_b79k")
    print("=" * 60)

    # Paths
    dataset_dir = PROJECT_ROOT / "dataset"
    output_dir = PROJECT_ROOT / "embeddings"
    output_dir.mkdir(exist_ok=True)

    # --- Step 1: Load dataset ---
    print("\n[1/5] Loading Flickr30k dataset...")
    dataset = load_dataset(str(dataset_dir))
    print(f"  Total images:   {dataset['total_images']}")
    print(f"  Total captions: {dataset['total_captions']}")
    if dataset["missing_images"]:
        print(f"  Missing images: {len(dataset['missing_images'])}")

    # --- Step 2: Load model ---
    device = get_device()
    print(f"\n[2/5] Loading OpenCLIP ViT-B-32 on {device}...")

    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    print("  ✓ Model loaded successfully")

    # --- Step 3: Generate image embeddings ---
    print(f"\n[3/5] Generating image embeddings ({dataset['total_images']} images)...")
    t0 = time.time()
    image_embeddings = generate_image_embeddings(
        model, preprocess, dataset["image_paths"], device, batch_size=64
    )
    img_time = time.time() - t0
    print(f"  ✓ Done in {img_time:.1f}s — shape: {image_embeddings.shape}")

    # --- Step 4: Generate text embeddings ---
    print(f"\n[4/5] Generating text embeddings ({dataset['total_captions']} captions)...")
    t0 = time.time()
    text_embeddings = generate_text_embeddings(
        model, tokenizer, dataset["captions"], device, batch_size=256
    )
    txt_time = time.time() - t0
    print(f"  ✓ Done in {txt_time:.1f}s — shape: {text_embeddings.shape}")

    # --- Step 5: Save to disk ---
    print("\n[5/5] Saving embeddings and metadata...")
    np.save(output_dir / "image_embeddings.npy", image_embeddings)
    np.save(output_dir / "text_embeddings.npy", text_embeddings)

    metadata = {
        "model": "ViT-B-32",
        "pretrained": "laion2b_s34b_b79k",
        "embedding_dim": int(image_embeddings.shape[1]),
        "total_images": dataset["total_images"],
        "total_captions": dataset["total_captions"],
        "image_filenames": dataset["image_filenames"],
        "captions": dataset["captions"],
        "caption_to_image_idx": dataset["caption_to_image_idx"],
        "image_to_caption_indices": {
            str(k): v for k, v in dataset["image_to_caption_indices"].items()
        },
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    # Summary
    img_size_mb = os.path.getsize(output_dir / "image_embeddings.npy") / 1e6
    txt_size_mb = os.path.getsize(output_dir / "text_embeddings.npy") / 1e6
    meta_size_mb = os.path.getsize(output_dir / "metadata.json") / 1e6

    print(f"\n{'=' * 60}")
    print("  ✓ Embedding generation complete!")
    print(f"{'=' * 60}")
    print(f"  image_embeddings.npy : {image_embeddings.shape}  ({img_size_mb:.1f} MB)")
    print(f"  text_embeddings.npy  : {text_embeddings.shape}  ({txt_size_mb:.1f} MB)")
    print(f"  metadata.json        : {meta_size_mb:.1f} MB")
    print(f"  Total time           : {img_time + txt_time:.1f}s")
    print(f"  Output directory     : {output_dir}")
    print(f"{'=' * 60}")
    print("\nNext step: python -m ml_pipeline.faiss_manager.build_index")


if __name__ == "__main__":
    main()
