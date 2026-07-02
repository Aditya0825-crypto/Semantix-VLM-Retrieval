"""
Semantix Backend — FastAPI Application
========================================
Multimodal semantic retrieval API powered by CLIP + FAISS.

Startup sequence:
  1. Load OpenCLIP ViT-B-32 model
  2. Load FAISS indexes from disk
  3. Load metadata and embeddings
  4. Mount static files for serving dataset images
  5. Register API route handlers

Run:
    cd project_root
    uvicorn backend.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .services.clip_service import clip_service
from .services.faiss_service import faiss_service
from .routes import search, embeddings, health
from .utils.logger import get_logger

logger = get_logger(__name__)

# Project root = parent of backend/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and indexes on server startup."""
    logger.info("=" * 60)
    logger.info("  SEMANTIX — Backend starting")
    logger.info("=" * 60)

    # 1. Load CLIP model
    clip_service.load()

    # 2. Load FAISS indexes + metadata
    faiss_service.load(PROJECT_ROOT)

    logger.info("=" * 60)
    logger.info("  ✓ Backend ready — accepting requests")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down...")


# ── App ──────────────────────────────────────────────────────

app = FastAPI(
    title="Semantix API",
    description=(
        "Multimodal semantic retrieval API. "
        "Powered by OpenCLIP ViT-B-32 and FAISS vector search."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ─────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Frontend dev server on any port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static Files ─────────────────────────────────────────────

images_dir = PROJECT_ROOT / "dataset" / "Images"
if images_dir.is_dir():
    app.mount(
        "/static/images",
        StaticFiles(directory=str(images_dir)),
        name="dataset-images",
    )
    logger.info(f"Serving images from: {images_dir}")
else:
    logger.warning(f"Images directory not found: {images_dir}")

# ── Routes ───────────────────────────────────────────────────

app.include_router(search.router)
app.include_router(embeddings.router)
app.include_router(health.router)


@app.get("/", tags=["root"])
async def root():
    """API root — redirects to docs."""
    return {
        "name": "Semantix API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }
