"""
Flickr30k Dataset Loader
========================
Parses captions.txt and builds structured image-caption mappings.
Validates image files and reports missing data.
"""

import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple


def parse_captions(captions_path: str) -> Dict[str, List[str]]:
    """
    Parse the Flickr30k captions.txt file.

    Expected format (CSV):
        image,caption
        1000092795.jpg, Two young guys with shaggy hair ...
        1000092795.jpg," Two young , White males are outside ..."

    Returns:
        dict mapping image_filename → list of caption strings
    """
    image_captions: Dict[str, List[str]] = {}

    with open(captions_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        # Skip header row
        header = next(reader, None)
        if header and header[0].strip().lower() == "image":
            pass  # Header consumed
        else:
            # No header — reprocess first row
            if header and len(header) >= 2:
                fname = header[0].strip()
                cap = header[1].strip()
                image_captions.setdefault(fname, []).append(cap)

        for row in reader:
            if len(row) < 2:
                continue
            image_filename = row[0].strip()
            caption = row[1].strip()
            if not image_filename or not caption:
                continue
            image_captions.setdefault(image_filename, []).append(caption)

    return image_captions


def validate_dataset(
    images_dir: str, image_captions: Dict[str, List[str]]
) -> Tuple[List[str], List[str]]:
    """
    Check which images referenced in captions actually exist on disk.

    Returns:
        (valid_images, missing_images) — both lists of filenames
    """
    valid_images: List[str] = []
    missing_images: List[str] = []

    for image_filename in image_captions:
        image_path = os.path.join(images_dir, image_filename)
        if os.path.isfile(image_path):
            valid_images.append(image_filename)
        else:
            missing_images.append(image_filename)

    return valid_images, missing_images


def load_dataset(dataset_dir: str) -> dict:
    """
    Load and validate the complete Flickr30k dataset.

    Args:
        dataset_dir: path to the dataset/ folder containing Images/ and captions.txt

    Returns:
        Dictionary with:
        - image_filenames: list of valid image filenames
        - image_paths: list of absolute paths to images
        - captions: flat list of all captions
        - caption_to_image_idx: for each caption index → image index
        - image_to_caption_indices: for each image index → list of caption indices
        - missing_images: filenames with no corresponding image file
        - total_images / total_captions: counts
    """
    captions_path = os.path.join(dataset_dir, "captions.txt")
    images_dir = os.path.join(dataset_dir, "Images")

    if not os.path.isfile(captions_path):
        raise FileNotFoundError(f"captions.txt not found at {captions_path}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found at {images_dir}")

    # 1. Parse captions
    image_captions = parse_captions(captions_path)
    print(f"  Parsed {len(image_captions)} unique images from captions.txt")

    # 2. Validate which images exist
    valid_images, missing_images = validate_dataset(images_dir, image_captions)
    print(f"  Valid images on disk: {len(valid_images)}")
    if missing_images:
        print(f"  Missing images: {len(missing_images)}")

    # 3. Build ordered structures
    image_paths = [os.path.join(images_dir, img) for img in valid_images]

    # Flat caption list with bidirectional mappings
    all_captions: List[str] = []
    caption_to_image_idx: List[int] = []
    image_to_caption_indices: Dict[int, List[int]] = {}

    for img_idx, image_filename in enumerate(valid_images):
        captions = image_captions[image_filename]
        start = len(all_captions)
        caption_indices = []
        for caption in captions:
            caption_indices.append(len(all_captions))
            all_captions.append(caption)
            caption_to_image_idx.append(img_idx)
        image_to_caption_indices[img_idx] = caption_indices

    return {
        "image_filenames": valid_images,
        "image_paths": image_paths,
        "captions": all_captions,
        "caption_to_image_idx": caption_to_image_idx,
        "image_to_caption_indices": image_to_caption_indices,
        "missing_images": missing_images,
        "total_images": len(valid_images),
        "total_captions": len(all_captions),
    }


if __name__ == "__main__":
    # Quick test
    project_root = Path(__file__).resolve().parent.parent.parent
    dataset_dir = project_root / "dataset"
    data = load_dataset(str(dataset_dir))
    print(f"\nDataset loaded successfully:")
    print(f"  Images:   {data['total_images']}")
    print(f"  Captions: {data['total_captions']}")
    print(f"  Missing:  {len(data['missing_images'])}")
    print(f"  Sample caption: {data['captions'][0][:80]}...")
