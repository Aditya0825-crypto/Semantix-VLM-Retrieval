/** Allowed Top-K values for FAISS nearest-neighbor retrieval. */
export const TOP_K_OPTIONS = [1, 3, 5, 7, 8, 10, 12] as const;

export type TopK = (typeof TOP_K_OPTIONS)[number];

export const DEFAULT_TOP_K: TopK = 8;

export function isTopK(value: number): value is TopK {
  return (TOP_K_OPTIONS as readonly number[]).includes(value);
}

/** Coerce an arbitrary number to a valid TopK (falls back to default). */
export function coerceTopK(value: number): TopK {
  return isTopK(value) ? value : DEFAULT_TOP_K;
}
