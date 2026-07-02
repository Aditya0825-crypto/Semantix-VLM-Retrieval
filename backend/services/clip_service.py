"""
CLIP Service
=============
Singleton wrapper around the OpenCLIP ViT-B-32 model.
Handles model loading, text encoding, and image encoding.
"""

import numpy as np
import torch
from PIL import Image

from ..utils.logger import get_logger

logger = get_logger(__name__)


class CLIPService:
    """Manages the OpenCLIP model lifecycle and inference."""

    def __init__(self):
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.device: torch.device | None = None
        self._loaded = False

    def load(self):
        """
        Load OpenCLIP ViT-B-32 model and preprocessing transforms.
        Auto-detects CUDA; falls back to CPU.
        """
        if self._loaded:
            return

        import open_clip

        # Device selection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logger.info("No GPU detected — using CPU")

        logger.info(f"Loading OpenCLIP ViT-B-32 on {self.device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

        self._loaded = True
        logger.info("✓ OpenCLIP model loaded successfully")

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a text string to a normalized 512-d embedding.

        Args:
            text: natural language query string

        Returns:
            np.ndarray of shape (512,), L2-normalized
        """
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type):
            embedding = self.model.encode_text(tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().float().numpy().squeeze()

    def encode_text_for_retrieval(self, text: str) -> np.ndarray:
        """
        Encode a query using CLIP prompt ensembling for better retrieval accuracy.

        Averages embeddings from multiple prompt templates, then L2-normalizes.
        This aligns short user queries with how CLIP was trained/evaluated.
        """
        from .retrieval_enhancements import QUERY_PROMPT_TEMPLATES

        embeddings = [
            self.encode_text(template.format(text.strip()))
            for template in QUERY_PROMPT_TEMPLATES
        ]
        avg = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg)
        if norm < 1e-8:
            return embeddings[0]
        return (avg / norm).astype(np.float32)

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Encode a PIL Image to a normalized 512-d embedding.

        Args:
            image: PIL Image in RGB mode

        Returns:
            np.ndarray of shape (512,), L2-normalized
        """
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type):
            embedding = self.model.encode_image(img_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().float().numpy().squeeze()

    @property
    def is_loaded(self) -> bool:
        """Whether the model is loaded and ready."""
        return self._loaded

    @property
    def device_name(self) -> str:
        """Human-readable device name."""
        if self.device is None:
            return "not loaded"
        if self.device.type == "cuda":
            return f"cuda ({torch.cuda.get_device_name(0)})"
        return "cpu"


# Singleton instance — shared across the backend
clip_service = CLIPService()
