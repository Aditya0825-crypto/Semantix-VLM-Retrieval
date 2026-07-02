/**
 * Semantix API Client
 * ====================
 * Communicates with the FastAPI backend for real CLIP+FAISS retrieval.
 * Falls back gracefully if the backend is unreachable.
 */

const API_BASE =
  (typeof import.meta.env.VITE_API_BASE === "string" && import.meta.env.VITE_API_BASE) ||
  "http://localhost:8000";

// ── Types ────────────────────────────────────────────────────

export type ImageResult = {
  rank: number;
  id: string;
  image_filename: string;
  url: string;
  caption: string;
  tags: string[];
  similarity: number;
};

export type CaptionResult = {
  rank: number;
  caption: string;
  image_filename: string;
  similarity: number;
};

export type EmbeddingPointAPI = {
  id: string;
  x: number;
  y: number;
  label: string;
  kind: "query" | "result" | "cluster";
  similarity?: number;
};

export type RetrievalMetadata = {
  query: string;
  mode: string;
  top_k: number;
  retrieval_ms: number;
  total_indexed: number;
  similarity: number;
  distance: number;
  confidence: number;
  model: string;
  device: string;
};

export type TextSearchResponse = {
  results: ImageResult[];
  points: EmbeddingPointAPI[];
  metadata: RetrievalMetadata;
};

export type ImageSearchResponse = {
  results: CaptionResult[];
  points: EmbeddingPointAPI[];
  metadata: RetrievalMetadata;
};

export type HealthResponse = {
  status: string;
  model_loaded: boolean;
  faiss_loaded: boolean;
  device: string;
  total_images: number;
  total_captions: number;
  embedding_dim: number;
  image_index_size: number;
  text_index_size: number;
};

// ── API Functions ────────────────────────────────────────────

/**
 * Text → Image retrieval.
 * Sends a natural language query, returns ranked images.
 */
export async function searchTextToImage(
  query: string,
  topK: number = 8,
  vizMethod: string = "pca",
): Promise<TextSearchResponse> {
  const res = await fetch(`${API_BASE}/api/search/text-to-image`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK, viz_method: vizMethod }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Text search failed (${res.status}): ${detail}`);
  }
  return res.json();
}

/**
 * Image → Text retrieval.
 * Uploads an image, returns ranked captions.
 */
export async function searchImageToText(
  file: File,
  topK: number = 8,
  vizMethod: string = "pca",
): Promise<ImageSearchResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("top_k", String(topK));
  formData.append("viz_method", vizMethod);

  const res = await fetch(`${API_BASE}/api/search/image-to-text`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Image search failed (${res.status}): ${detail}`);
  }
  return res.json();
}

/**
 * Fetch embedding visualization for a text query.
 * Returns 2D-projected points: query + results + background.
 */
export async function fetchEmbeddingVisualization(
  query: string,
  topK: number = 8,
  method: string = "pca",
): Promise<{ points: EmbeddingPointAPI[]; query: string }> {
  const params = new URLSearchParams({
    query,
    top_k: String(topK),
    method,
  });
  const res = await fetch(`${API_BASE}/api/embeddings/query?${params}`, {
    method: "POST",
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Embedding visualization failed (${res.status}): ${detail}`);
  }
  return res.json();
}

/**
 * Health check — verify backend is running and ready.
 */
export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

/**
 * Resolve an image filename to a full URL served by the backend.
 */
export function imageUrl(filename: string): string {
  return `${API_BASE}/static/images/${filename}`;
}

/**
 * Check if the backend is reachable.
 */
export async function isBackendAvailable(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(3000) });
    return res.ok;
  } catch {
    return false;
  }
}
