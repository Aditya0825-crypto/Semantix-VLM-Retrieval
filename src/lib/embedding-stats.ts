import type { EmbeddingPoint } from "@/components/EmbeddingCanvas";

/** 2D stats for retrieved neighbors relative to the query point. */
export function computeNeighborhoodStats(points: EmbeddingPoint[]) {
  const query = points.find((p) => p.kind === "query");
  const results = points.filter((p) => p.kind === "result" && p.similarity !== undefined);

  if (!query || results.length === 0) {
    return { spread: 0, density: 0, avgSimilarity: 0, resultCount: results.length };
  }

  const distances = results.map((p) => Math.hypot(p.x - query.x, p.y - query.y));
  const spread = distances.length
    ? distances.reduce((a, d) => a + d, 0) / distances.length
    : 0;

  const xs = results.map((p) => p.x);
  const ys = results.map((p) => p.y);
  const area =
    (Math.max(...xs) - Math.min(...xs) + 0.05) *
    (Math.max(...ys) - Math.min(...ys) + 0.05);
  const density = results.length / area;

  const avgSimilarity =
    results.reduce((a, p) => a + (p.similarity ?? 0), 0) / results.length;

  return { spread, density, avgSimilarity, resultCount: results.length };
}
