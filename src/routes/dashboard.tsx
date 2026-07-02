import { createFileRoute, Link } from "@tanstack/react-router";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  Search, Type, Image as ImageIcon, Upload, Sparkles, X, ArrowRight,
  Loader2, Zap, Network,
} from "lucide-react";
import { EmbeddingCanvas } from "@/components/EmbeddingCanvas";
import { TopKSelector } from "@/components/TopKSelector";
import { searchImagesByText, captionsForImage, type ImageItem } from "@/lib/mock-data";
import { buildEmbeddingPoints, setRetrieval, useRetrieval } from "@/lib/retrieval-store";
import { searchTextToImage, searchImageToText, imageUrl, fetchHealth, type HealthResponse } from "@/lib/api-client";
import type { EmbeddingPointAPI } from "@/lib/api-client";
import { DEFAULT_TOP_K, type TopK } from "@/lib/top-k";

export const Route = createFileRoute("/dashboard")({
  head: () => ({
    meta: [
      { title: "Retrieval Dashboard — Semantix" },
      { name: "description", content: "Run text-to-image and image-to-text semantic retrieval with live similarity ranking." },
    ],
  }),
  component: Dashboard,
});

type Mode = "text2image" | "image2text";

function Dashboard() {
  const [mode, setMode] = useState<Mode>("text2image");
  const [query, setQuery] = useState("");
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [textResults, setTextResults] = useState<{ item: ImageItem; sim: number }[]>([]);
  const [imageResults, setImageResults] = useState<{ caption: string; sim: number }[]>([]);
  const [animKey, setAnimKey] = useState(0);
  const [modalImg, setModalImg] = useState<ImageItem | null>(null);
  const [embeddingKey, setEmbeddingKey] = useState(0);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [backendOnline, setBackendOnline] = useState(false);
  const [topK, setTopK] = useState<TopK>(DEFAULT_TOP_K);
  const [hasSearched, setHasSearched] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const prevTopKRef = useRef(topK);
  const searchContextRef = useRef({ mode, query, imageFile, hasSearched: false });

  useEffect(() => {
    fetchHealth()
      .then((h) => {
        setHealth(h);
        setBackendOnline(h.status === "ok" && h.model_loaded && h.faiss_loaded);
      })
      .catch(() => {
        setHealth(null);
        setBackendOnline(false);
      });
  }, []);

  const runTextRetrieval = useCallback(async (q: string, k: TopK) => {
    setLoading(true);
    try {
      const response = await searchTextToImage(q, k);
      const mapped = response.results.map((r) => ({
        item: {
          id: r.id,
          url: imageUrl(r.image_filename),
          caption: r.caption,
          tags: r.tags,
        } as ImageItem,
        sim: r.similarity,
      }));
      setTextResults(mapped);
      setImageResults([]);
      setRetrieval({
        mode: "text2image",
        query: q,
        points: response.points as EmbeddingPointAPI[],
        topK: response.metadata.top_k,
        similarity: response.metadata.similarity,
        distance: response.metadata.distance,
        retrievalMs: response.metadata.retrieval_ms,
        total: response.metadata.total_indexed,
        confidence: response.metadata.confidence,
        ts: Date.now(),
      });
    } catch (err) {
      console.warn("Backend unreachable, falling back to mock:", err);
      const res = searchImagesByText(q, k);
      setTextResults(res);
      setImageResults([]);
      const points = buildEmbeddingPoints(q, res.length);
      setRetrieval({
        mode: "text2image", query: q, points, topK: k,
        similarity: res[0]?.sim ?? 0, distance: 1 - (res[0]?.sim ?? 0),
        retrievalMs: 0, total: 12_482_103, confidence: 0.94, ts: Date.now(),
      });
    }
    setHasSearched(true);
    setAnimKey((n) => n + 1);
    setEmbeddingKey((n) => n + 1);
    setLoading(false);
  }, []);

  const runImageRetrieval = useCallback(async (file: File, k: TopK) => {
    setLoading(true);
    try {
      const response = await searchImageToText(file, k);
      const mapped = response.results.map((r) => ({
        caption: r.caption,
        sim: r.similarity,
      }));
      setImageResults(mapped);
      setTextResults([]);
      setRetrieval({
        mode: "image2text",
        query: file.name.slice(0, 30),
        points: response.points as EmbeddingPointAPI[],
        topK: response.metadata.top_k,
        similarity: response.metadata.similarity,
        distance: response.metadata.distance,
        retrievalMs: response.metadata.retrieval_ms,
        total: response.metadata.total_indexed,
        confidence: response.metadata.confidence,
        ts: Date.now(),
      });
    } catch (err) {
      console.warn("Backend unreachable, falling back to mock:", err);
      const res = captionsForImage(file.name, k);
      setImageResults(res);
      setTextResults([]);
      const points = buildEmbeddingPoints(file.name, res.length);
      setRetrieval({
        mode: "image2text", query: file.name.slice(0, 30), points, topK: k,
        similarity: res[0]?.sim ?? 0, distance: 1 - (res[0]?.sim ?? 0),
        retrievalMs: 0, total: 12_482_103, confidence: 0.91, ts: Date.now(),
      });
    }
    setHasSearched(true);
    setAnimKey((n) => n + 1);
    setEmbeddingKey((n) => n + 1);
    setLoading(false);
  }, []);

  searchContextRef.current = { mode, query, imageFile, hasSearched };

  // Re-run FAISS retrieval when Top-K changes after an initial search
  useEffect(() => {
    if (prevTopKRef.current === topK) return;
    prevTopKRef.current = topK;

    const ctx = searchContextRef.current;
    if (!ctx.hasSearched) return;

    if (ctx.mode === "text2image" && ctx.query.trim()) {
      runTextRetrieval(ctx.query.trim(), topK);
    } else if (ctx.mode === "image2text" && ctx.imageFile) {
      runImageRetrieval(ctx.imageFile, topK);
    }
  }, [topK, runTextRetrieval, runImageRetrieval]);

  const handleSearchText = () => {
    if (!query.trim()) return;
    runTextRetrieval(query.trim(), topK);
  };

  const handleFile = (file: File) => {
    const url = URL.createObjectURL(file);
    setImagePreview(url);
    setImageFile(file);
  };

  const handleAnalyze = () => {
    if (!imageFile) return;
    runImageRetrieval(imageFile, topK);
  };

  const livePoints = useRetrievalPoints();
  const activeTopK = useRetrieval()?.topK ?? topK;

  return (
    <div className="px-4 md:px-6 pb-16">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex flex-wrap items-end justify-between gap-4 mb-6">
          <div>
            <div className="text-xs font-mono text-primary tracking-[0.3em] uppercase mb-1">Console</div>
            <h1 className="text-3xl md:text-4xl font-bold tracking-tight">Retrieval Dashboard</h1>
            <p className="text-sm text-muted-foreground mt-1">
              Run cross-modal queries across the multimodal index. Latency, ranking, and embeddings shown live.
            </p>
          </div>
          <Link
            to="/embeddings"
            className="inline-flex items-center gap-2 glass rounded-lg px-4 py-2 text-sm hover:bg-foreground/5 transition"
          >
            <Network className="h-4 w-4 text-accent" /> Open full embedding map
            <ArrowRight className="h-3.5 w-3.5" />
          </Link>
        </div>

        <div className="grid lg:grid-cols-[380px_1fr] gap-4">
          {/* LEFT PANEL */}
          <div className="space-y-4">
            <div className="glass-strong rounded-2xl p-1.5">
              <div className="grid grid-cols-2 gap-1">
                {(["text2image", "image2text"] as Mode[]).map((m) => (
                  <button
                    key={m}
                    onClick={() => setMode(m)}
                    className={`relative px-3 py-2.5 rounded-xl text-sm font-medium transition flex items-center justify-center gap-2 ${
                      mode === m ? "bg-gradient-aurora text-background shadow-neon" : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {m === "text2image" ? <Type className="h-4 w-4" /> : <ImageIcon className="h-4 w-4" />}
                    {m === "text2image" ? "Text → Image" : "Image → Text"}
                  </button>
                ))}
              </div>
            </div>

            {mode === "text2image" ? (
              <div className="glass-strong rounded-2xl p-5 space-y-4">
                <label className="text-xs font-mono text-muted-foreground tracking-wider uppercase">Natural language query</label>
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <input
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleSearchText()}
                    placeholder="e.g. a neon hologram of a city"
                    className="w-full bg-input/80 border border-border rounded-xl pl-9 pr-3 py-3 text-sm placeholder:text-muted-foreground/60 focus:outline-none focus:border-primary/50 focus:bg-input transition"
                  />
                </div>
                <button
                  onClick={handleSearchText}
                  disabled={loading || !query.trim()}
                  className="w-full bg-gradient-aurora text-background font-semibold rounded-xl py-3 text-sm flex items-center justify-center gap-2 disabled:opacity-50 hover:scale-[1.01] transition shadow-neon"
                >
                  {loading ? <><Loader2 className="h-4 w-4 animate-spin" /> Encoding query…</> : <><Zap className="h-4 w-4" /> Retrieve</>}
                </button>
              </div>
            ) : (
              <div className="glass-strong rounded-2xl p-5 space-y-4">
                <label className="text-xs font-mono text-muted-foreground tracking-wider uppercase">Upload image</label>
                <div
                  onDrop={(e) => {
                    e.preventDefault();
                    const f = e.dataTransfer.files?.[0];
                    if (f) handleFile(f);
                  }}
                  onDragOver={(e) => e.preventDefault()}
                  onClick={() => fileRef.current?.click()}
                  className="relative cursor-pointer border-2 border-dashed border-border hover:border-primary/50 rounded-xl p-6 text-center transition group bg-muted/30"
                >
                  {imagePreview ? (
                    <div className="relative">
                      <img src={imagePreview} alt="upload" className="w-full h-48 object-cover rounded-lg" />
                      <button
                        onClick={(e) => { e.stopPropagation(); setImagePreview(null); setImageFile(null); }}
                        className="absolute top-2 right-2 glass-strong rounded-full p-1.5 hover:bg-destructive/80"
                      >
                        <X className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  ) : (
                    <div className="py-6">
                      <div className="mx-auto h-12 w-12 rounded-2xl bg-gradient-aurora/20 border border-border flex items-center justify-center mb-3 group-hover:scale-110 transition">
                        <Upload className="h-5 w-5 text-primary" />
                      </div>
                      <div className="text-sm font-medium">Drop image here</div>
                      <div className="text-xs text-muted-foreground mt-1">or click to browse · PNG, JPG, WEBP</div>
                    </div>
                  )}
                  <input
                    ref={fileRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
                  />
                </div>
                <button
                  onClick={handleAnalyze}
                  disabled={loading || !imageFile}
                  className="w-full bg-gradient-aurora text-background font-semibold rounded-xl py-3 text-sm flex items-center justify-center gap-2 disabled:opacity-50 hover:scale-[1.01] transition shadow-neon"
                >
                  {loading ? <><Loader2 className="h-4 w-4 animate-spin" /> Analyzing pixels…</> : <><Sparkles className="h-4 w-4" /> Analyze</>}
                </button>
              </div>
            )}

            <div className="glass-strong rounded-2xl p-5">
              <TopKSelector
                value={topK}
                onChange={setTopK}
                disabled={loading}
              />
            </div>

            {/* Status panel */}
            <div className="glass rounded-2xl p-4 font-mono text-[11px] space-y-2">
              <div className="flex items-center gap-2">
                <span className={`h-1.5 w-1.5 rounded-full shadow-[0_0_6px_currentColor] ${backendOnline ? "bg-primary animate-pulse" : "bg-amber-500"}`} />
                <span className="text-muted-foreground tracking-wider uppercase">Backend</span>
                <span className={`ml-auto ${backendOnline ? "text-primary" : "text-amber-500"}`}>
                  {backendOnline ? "ONLINE" : "OFFLINE"}
                </span>
              </div>
              <div className="flex justify-between text-muted-foreground"><span>Encoder</span><span className="text-foreground">CLIP ViT-B/32</span></div>
              <div className="flex justify-between text-muted-foreground"><span>Index</span><span className="text-foreground">FAISS · cosine</span></div>
              <div className="flex justify-between text-muted-foreground"><span>Dim</span><span className="text-foreground">{health?.embedding_dim ?? 512}-d</span></div>
              <div className="flex justify-between text-muted-foreground"><span>Images</span><span className="text-foreground">{(health?.total_images ?? 0).toLocaleString()}</span></div>
              <div className="flex justify-between text-muted-foreground"><span>Captions</span><span className="text-foreground">{(health?.total_captions ?? 0).toLocaleString()}</span></div>
              <div className="flex justify-between text-muted-foreground"><span>Top-K</span><span className="text-primary">{topK}</span></div>
              {!backendOnline && (
                <p className="text-[10px] text-amber-500/90 pt-1 leading-relaxed">
                  Start API: uvicorn backend.main:app --port 8000
                </p>
              )}
            </div>
          </div>

          {/* RIGHT PANEL */}
          <div className="space-y-4">
            {/* Mini embedding preview */}
            <div>
              <div className="flex items-center justify-between mb-2 px-1">
                <div className="text-xs font-mono text-muted-foreground tracking-wider uppercase">Live embedding space</div>
                <Link to="/embeddings" className="text-xs text-primary hover:underline font-mono">
                  expand ↗
                </Link>
              </div>
              <EmbeddingCanvas
                points={livePoints}
                animatingKey={embeddingKey}
                height={280}
                topK={activeTopK}
              />
            </div>

            {/* Results */}
            <div className="glass-strong rounded-2xl p-5 min-h-[400px]">
              <ResultsArea
                loading={loading}
                mode={mode}
                textResults={textResults}
                imageResults={imageResults}
                animKey={animKey}
                onOpen={setModalImg}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Modal */}
      {modalImg && (
        <div
          className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 animate-fade-in"
          onClick={() => setModalImg(null)}
        >
          <div className="max-w-3xl glass-strong rounded-2xl p-4 relative" onClick={(e) => e.stopPropagation()}>
            <button onClick={() => setModalImg(null)} className="absolute top-3 right-3 glass-strong rounded-full p-2 hover:bg-destructive/80">
              <X className="h-4 w-4" />
            </button>
            <img src={modalImg.url} alt={modalImg.caption} className="w-full max-h-[70vh] object-contain rounded-xl" />
            <div className="mt-3 text-sm">{modalImg.caption}</div>
            <div className="mt-1 flex gap-1.5">
              {modalImg.tags.map((t) => (
                <span key={t} className="text-[10px] font-mono glass rounded-full px-2 py-0.5 text-muted-foreground">#{t}</span>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ResultsArea({
  loading, mode, textResults, imageResults, animKey, onOpen,
}: {
  loading: boolean;
  mode: Mode;
  textResults: { item: ImageItem; sim: number }[];
  imageResults: { caption: string; sim: number }[];
  animKey: number;
  onOpen: (i: ImageItem) => void;
}) {
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-16 gap-4">
        <div className="relative">
          <div className="h-20 w-20 rounded-full border-2 border-primary/30 border-t-primary animate-spin" />
          <Sparkles className="h-7 w-7 text-primary absolute inset-0 m-auto animate-pulse" />
        </div>
        <div className="font-mono text-xs text-muted-foreground tracking-wider">
          <TypingLine lines={[
            "› Tokenizing query…",
            "› Encoding via CLIP ViT-B/32…",
            "› Projecting to 512-d latent…",
            "› FAISS search · cosine similarity…",
            "› Ranking top-K matches…",
          ]} />
        </div>
      </div>
    );
  }

  if (textResults.length === 0 && imageResults.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <div className="h-16 w-16 rounded-2xl bg-gradient-aurora/20 border border-border flex items-center justify-center mb-4">
          <Search className="h-7 w-7 text-primary" />
        </div>
        <div className="text-lg font-semibold">Ready to retrieve</div>
        <div className="text-sm text-muted-foreground mt-1">
          Enter a query or upload an image to see ranked semantic matches.
        </div>
      </div>
    );
  }

  if (mode === "text2image" && textResults.length > 0) {
    return (
      <div key={animKey} className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
        {textResults.map((r, i) => (
          <button
            key={r.item.id + animKey}
            onClick={() => onOpen(r.item)}
            className="group relative overflow-hidden rounded-xl bg-muted/40 aspect-square animate-slide-up text-left"
            style={{ animationDelay: `${i * 60}ms` }}
          >
            <img
              src={r.item.url}
              alt={r.item.caption}
              loading="lazy"
              className="absolute inset-0 w-full h-full object-cover group-hover:scale-110 transition duration-500"
            />
            <div className="absolute inset-0 bg-linear-to-t from-black/90 via-black/30 to-transparent" />
            <div className="absolute top-2 left-2 glass-strong rounded-md px-2 py-0.5 text-[10px] font-mono">
              <span className="text-primary">{(r.sim * 100).toFixed(1)}%</span>
            </div>
            <div className="absolute top-2 right-2 glass-strong rounded-md px-1.5 py-0.5 text-[10px] font-mono text-muted-foreground">
              #{i + 1}
            </div>
            <div className="absolute bottom-0 inset-x-0 p-2.5">
              <div className="text-[11px] text-foreground/90 line-clamp-2">{r.item.caption}</div>
              <div className="mt-1.5 h-1 rounded-full bg-white/10 overflow-hidden">
                <div className="h-full bg-gradient-aurora" style={{ width: `${r.sim * 100}%` }} />
              </div>
            </div>
          </button>
        ))}
      </div>
    );
  }

  return (
    <div key={animKey} className="space-y-2.5">
      {imageResults.map((r, i) => (
        <div
          key={i + "-" + animKey}
          className="glass rounded-xl p-4 animate-slide-up hover:bg-foreground/5 transition"
          style={{ animationDelay: `${i * 70}ms` }}
        >
          <div className="flex items-start justify-between gap-3">
            <div className="flex items-start gap-3 flex-1">
              <div className="font-mono text-xs text-muted-foreground mt-0.5">#{i + 1}</div>
              <div className="flex-1">
                <div className="text-sm leading-relaxed">{r.caption}</div>
                <div className="mt-2 h-1 rounded-full bg-white/10 overflow-hidden">
                  <div className="h-full bg-gradient-aurora" style={{ width: `${r.sim * 100}%` }} />
                </div>
              </div>
            </div>
            <div className="text-right shrink-0">
              <div className="text-lg font-bold text-gradient">{(r.sim * 100).toFixed(1)}<span className="text-xs">%</span></div>
              <div className="text-[10px] font-mono text-muted-foreground">confidence</div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function TypingLine({ lines }: { lines: string[] }) {
  const [i, setI] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setI((n) => (n + 1) % lines.length), 600);
    return () => clearInterval(id);
  }, [lines.length]);
  return <div>{lines[i]}</div>;
}

function useRetrievalPoints() {
  const r = useRetrieval();
  if (r) return r.points;
  return buildEmbeddingPoints("idle", 6);
}
