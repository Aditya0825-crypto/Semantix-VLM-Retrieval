// Tiny cross-route store so the embeddings page can read the last retrieval.
import { useEffect, useState } from "react";
import type { EmbeddingPoint } from "@/components/EmbeddingCanvas";

type RetrievalSnapshot = {
  mode: "text2image" | "image2text";
  query: string;
  points: EmbeddingPoint[];
  topK: number;
  similarity: number;
  distance: number;
  retrievalMs: number;
  total: number;
  confidence: number;
  ts: number;
};

let state: RetrievalSnapshot | null = null;
const listeners = new Set<() => void>();

export function setRetrieval(s: RetrievalSnapshot) {
  state = s;
  listeners.forEach((l) => l());
}

export function useRetrieval() {
  const [, force] = useState(0);
  useEffect(() => {
    const l = () => force((n) => n + 1);
    listeners.add(l);
    return () => { listeners.delete(l); };
  }, []);
  return state;
}

// Helper: build a fresh batch of embedding points seeded by a query string.
export function buildEmbeddingPoints(query: string, resultCount = 8): EmbeddingPoint[] {
  // Hash the query for deterministic placement
  let h = 0;
  for (let i = 0; i < query.length; i++) h = (h * 31 + query.charCodeAt(i)) | 0;
  const rand = (n: number) => {
    const x = Math.sin(h + n) * 10000;
    return x - Math.floor(x);
  };

  const points: EmbeddingPoint[] = [];

  // 3 background clusters
  for (let c = 0; c < 4; c++) {
    points.push({
      id: `c${c}`,
      kind: "cluster",
      x: (rand(c * 7) - 0.5) * 1.6,
      y: (rand(c * 7 + 1) - 0.5) * 1.6,
      label: `cluster_${c}`,
    });
  }

  // Background noise points (other vectors in the index)
  for (let i = 0; i < 40; i++) {
    points.push({
      id: `n${i}`,
      kind: "result",
      x: (rand(100 + i) - 0.5) * 1.8,
      y: (rand(200 + i) - 0.5) * 1.8,
      label: `vec_${i.toString(16).padStart(4, "0")}`,
      similarity: 0.2 + rand(300 + i) * 0.25,
    });
  }

  // Query at center-ish
  const qx = (rand(999) - 0.5) * 0.4;
  const qy = (rand(998) - 0.5) * 0.4;
  points.push({ id: "query", kind: "query", x: qx, y: qy, label: query.slice(0, 32) || "query" });

  // Top-K results close to query
  for (let i = 0; i < resultCount; i++) {
    const r = 0.15 + i * 0.07;
    const a = rand(500 + i) * Math.PI * 2;
    points.push({
      id: `r${i}`,
      kind: "result",
      x: qx + Math.cos(a) * r,
      y: qy + Math.sin(a) * r,
      label: `match_${i + 1}`,
      similarity: 0.95 - i * 0.06,
    });
  }

  return points;
}
