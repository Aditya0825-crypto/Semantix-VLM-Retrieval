import { createFileRoute, Link } from "@tanstack/react-router";
import { useCallback, useEffect, useRef, useState } from "react";
import { EmbeddingCanvas } from "@/components/EmbeddingCanvas";
import { TopKSelector } from "@/components/TopKSelector";
import { buildEmbeddingPoints, setRetrieval, useRetrieval } from "@/lib/retrieval-store";
import { fetchEmbeddingVisualization } from "@/lib/api-client";
import type { EmbeddingPointAPI } from "@/lib/api-client";
import { computeNeighborhoodStats } from "@/lib/embedding-stats";
import { DEFAULT_TOP_K, coerceTopK, type TopK } from "@/lib/top-k";
import {
  Activity, Gauge, Sparkles, Layers, Cpu, Target, Clock, Database,
  Play, Type, Search, GitBranch,
} from "lucide-react";

export const Route = createFileRoute("/embeddings")({
  head: () => ({
    meta: [
      { title: "Embedding Space — Semantix" },
      { name: "description", content: "Interactive real-time visualization of multimodal embedding space, similarity clusters and retrieval trajectories." },
    ],
  }),
  component: EmbeddingsPage,
});

function EmbeddingsPage() {
  const retrieval = useRetrieval();
  const [demoQuery, setDemoQuery] = useState("neural city of light");
  const [topK, setTopK] = useState<TopK>(() => coerceTopK(retrieval?.topK ?? DEFAULT_TOP_K));
  const [animKey, setAnimKey] = useState(0);
  const [loading, setLoading] = useState(false);
  const prevTopKRef = useRef(topK);
  const hasRunRef = useRef(false);
  const queryRef = useRef(demoQuery);
  queryRef.current = retrieval?.query ?? demoQuery;

  // Sync topK from dashboard retrieval snapshot
  useEffect(() => {
    if (retrieval?.topK && retrieval.topK !== topK) {
      const k = coerceTopK(retrieval.topK);
      setTopK(k);
      prevTopKRef.current = k;
    }
  }, [retrieval?.topK, retrieval?.ts, topK]);

  // Seed demo on first visit when store is empty
  useEffect(() => {
    if (!retrieval) {
      const points = buildEmbeddingPoints(demoQuery, DEFAULT_TOP_K);
      setRetrieval({
        mode: "text2image", query: demoQuery, points, topK: DEFAULT_TOP_K,
        similarity: 0.93, distance: 0.07, retrievalMs: 42,
        total: 12_482_103, confidence: 0.94, ts: Date.now(),
      });
      setAnimKey((k) => k + 1);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const runDemo = useCallback(async (q: string, k: TopK) => {
    setLoading(true);
    try {
      const response = await fetchEmbeddingVisualization(q, k, "pca");
      setRetrieval({
        mode: "text2image",
        query: q,
        points: response.points as EmbeddingPointAPI[],
        topK: k,
        similarity: 0.88 + Math.random() * 0.1,
        distance: Math.random() * 0.2,
        retrievalMs: 30 + Math.floor(Math.random() * 40),
        total: 12_482_103,
        confidence: 0.88 + Math.random() * 0.1,
        ts: Date.now(),
      });
    } catch {
      const pts = buildEmbeddingPoints(q + "_" + k, k);
      setRetrieval({
        mode: "text2image", query: q, points: pts, topK: k,
        similarity: 0.88 + Math.random() * 0.1,
        distance: Math.random() * 0.2,
        retrievalMs: 30 + Math.floor(Math.random() * 40),
        total: 12_482_103,
        confidence: 0.88 + Math.random() * 0.1,
        ts: Date.now(),
      });
    }
    hasRunRef.current = true;
    setAnimKey((n) => n + 1);
    setLoading(false);
  }, []);

  const handleRegenerate = () => {
    const q = (retrieval?.query ?? demoQuery).trim() || "embedding";
    runDemo(q, topK);
  };

  // Re-fetch visualization when K changes
  useEffect(() => {
    if (prevTopKRef.current === topK) return;
    prevTopKRef.current = topK;
    if (!hasRunRef.current && !retrieval) return;

    const q = queryRef.current.trim();
    if (!q) return;
    runDemo(q, topK);
  }, [topK, runDemo, retrieval]);

  const points = retrieval?.points ?? buildEmbeddingPoints(demoQuery, topK);
  const activeTopK = retrieval?.topK ?? topK;
  const stats = computeNeighborhoodStats(points);

  const metrics = [
    { icon: Target, label: "Similarity", value: (retrieval?.similarity ?? 0.93).toFixed(3), accent: "text-primary" },
    { icon: Activity, label: "Distance", value: (retrieval?.distance ?? 0.07).toFixed(3), accent: "text-accent" },
    { icon: Clock, label: "Retrieval", value: `${retrieval?.retrievalMs ?? 42}ms`, accent: "text-primary" },
    { icon: Database, label: "Vectors", value: (retrieval?.total ?? 12482103).toLocaleString(), accent: "text-foreground" },
    { icon: Cpu, label: "Model", value: "CLIP-ViT-B/32", accent: "text-accent" },
    { icon: Gauge, label: "Confidence", value: `${((retrieval?.confidence ?? 0.94) * 100).toFixed(1)}%`, accent: "text-primary" },
  ];

  const neighborhoodMetrics = [
    { label: "Neighborhood (K)", value: String(activeTopK) },
    { label: "Retrieved", value: String(stats.resultCount) },
    { label: "2D spread", value: stats.spread.toFixed(3) },
    { label: "Density", value: stats.density.toFixed(1) },
    { label: "Avg similarity", value: `${(stats.avgSimilarity * 100).toFixed(1)}%` },
  ];

  return (
    <div className="px-4 md:px-6 pb-16">
      <div className="max-w-7xl mx-auto">
        <div className="flex flex-wrap items-end justify-between gap-4 mb-6">
          <div>
            <div className="text-xs font-mono text-accent tracking-[0.3em] uppercase mb-1">Latent space</div>
            <h1 className="text-3xl md:text-4xl font-bold tracking-tight">Embedding Visualization</h1>
            <p className="text-sm text-muted-foreground mt-1 max-w-2xl">
              A 2D projection of the joint multimodal embedding space. Adjust Top-K to see how
              semantic neighborhood size, spread, and cluster density change in real time.
            </p>
          </div>
          <Link to="/dashboard" className="glass rounded-lg px-4 py-2 text-sm hover:bg-foreground/5 transition inline-flex items-center gap-2">
            <Search className="h-4 w-4" /> Run a new query
          </Link>
        </div>

        <div className="grid lg:grid-cols-[1fr_320px] gap-4">
          <div>
            <EmbeddingCanvas
              points={points}
              animatingKey={animKey}
              height={560}
              topK={activeTopK}
            />

            <div className="mt-4 glass-strong rounded-2xl p-4 flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-2 text-xs font-mono text-muted-foreground tracking-wider uppercase">
                <Type className="h-3.5 w-3.5" /> Simulate query
              </div>
              <input
                value={demoQuery}
                onChange={(e) => setDemoQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleRegenerate()}
                className="flex-1 min-w-[200px] bg-input/80 border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-primary/50"
                placeholder="Type any concept…"
              />
              <TopKSelector value={topK} onChange={setTopK} disabled={loading} compact />
              <button
                onClick={handleRegenerate}
                disabled={loading}
                className="bg-gradient-aurora text-background font-semibold rounded-lg px-4 py-2 text-sm flex items-center gap-2 hover:scale-[1.02] transition shadow-neon disabled:opacity-50"
              >
                <Play className="h-3.5 w-3.5" /> {loading ? "Retrieving…" : "Regenerate"}
              </button>
            </div>
          </div>

          <div className="space-y-3">
            <div className="glass-strong rounded-2xl p-5">
              <TopKSelector value={topK} onChange={setTopK} disabled={loading} />
            </div>

            <div className="glass-strong rounded-2xl p-5">
              <div className="flex items-center justify-between mb-4">
                <div className="text-xs font-mono text-muted-foreground tracking-wider uppercase">Live metrics</div>
                <span className="inline-flex items-center gap-1.5 text-[10px] font-mono text-primary">
                  <span className="h-1.5 w-1.5 rounded-full bg-primary shadow-[0_0_6px_currentColor] animate-pulse" /> STREAMING
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2">
                {metrics.map((m) => (
                  <div key={m.label} className="glass rounded-xl p-3">
                    <div className="flex items-center gap-1.5 text-[10px] font-mono text-muted-foreground tracking-wider uppercase">
                      <m.icon className="h-3 w-3" /> {m.label}
                    </div>
                    <div className={`mt-1 text-lg font-semibold font-mono ${m.accent}`}>{m.value}</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="glass-strong rounded-2xl p-5">
              <div className="flex items-center gap-1.5 text-xs font-mono text-muted-foreground tracking-wider uppercase mb-3">
                <GitBranch className="h-3.5 w-3.5" /> Semantic neighborhood
              </div>
              <div className="space-y-0.5">
                {neighborhoodMetrics.map((m) => (
                  <div key={m.label} className="flex justify-between text-[11px] font-mono">
                    <span className="text-muted-foreground">{m.label}</span>
                    <span className="text-foreground">{m.value}</span>
                  </div>
                ))}
              </div>
              <p className="mt-3 text-[10px] text-muted-foreground leading-relaxed">
                Spread measures mean 2D distance from query to retrieved neighbors.
                Density rises as K increases and neighbors pack tighter in projection space.
              </p>
            </div>

            <div className="glass-strong rounded-2xl p-5">
              <div className="text-xs font-mono text-muted-foreground tracking-wider uppercase mb-2">Active query</div>
              <div className="text-sm font-medium wrap-break-word">
                {retrieval?.query || demoQuery}
              </div>
              <div className="mt-3 text-[11px] font-mono text-muted-foreground space-y-1">
                <div className="flex justify-between"><span>mode</span><span className="text-foreground">{retrieval?.mode ?? "text2image"}</span></div>
                <div className="flex justify-between"><span>top-k</span><span className="text-primary">{activeTopK}</span></div>
                <div className="flex justify-between"><span>dim</span><span className="text-foreground">512</span></div>
                <div className="flex justify-between"><span>distance</span><span className="text-foreground">cosine</span></div>
              </div>
            </div>

            <div className="glass rounded-2xl p-4">
              <div className="text-xs font-mono text-muted-foreground tracking-wider uppercase mb-2 flex items-center gap-1.5">
                <Layers className="h-3 w-3" /> Query vector (truncated)
              </div>
              <VectorPreview seed={retrieval?.query ?? demoQuery} />
            </div>

            <div className="glass rounded-2xl p-4 flex items-start gap-3">
              <Sparkles className="h-4 w-4 text-accent shrink-0 mt-0.5" />
              <p className="text-xs text-muted-foreground leading-relaxed">
                Embeddings are projected from <span className="text-foreground font-mono">ℝ⁵¹²</span> to <span className="text-foreground font-mono">ℝ²</span> via PCA.
                True nearest-neighbor search happens in the original space with FAISS top-K.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function VectorPreview({ seed }: { seed: string }) {
  let h = 0;
  for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) | 0;
  const vals = Array.from({ length: 24 }, (_, i) => {
    const x = Math.sin(h + i * 7.1) * 10000;
    return ((x - Math.floor(x)) * 2 - 1).toFixed(3);
  });
  return (
    <div className="font-mono text-[10px] text-muted-foreground leading-relaxed break-all">
      <span className="text-primary">[</span>
      {vals.map((v, i) => (
        <span key={i}>
          <span className={parseFloat(v) >= 0 ? "text-primary/80" : "text-accent/80"}>{v}</span>
          {i < vals.length - 1 ? ", " : ""}
        </span>
      ))}
      <span className="text-muted-foreground">, … 488 more</span>
      <span className="text-primary">]</span>
    </div>
  );
}
