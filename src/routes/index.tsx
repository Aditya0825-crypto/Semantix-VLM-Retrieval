import { createFileRoute, Link } from "@tanstack/react-router";
import { ParticleField } from "@/components/ParticleField";
import {
  ArrowRight, Image as ImageIcon, Type, Network, Sparkles, Layers, Gauge,
  Github, FileText, Mail, BookOpen,
} from "lucide-react";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Semantix — Multimodal AI Semantic Retrieval" },
      { name: "description", content: "Search across images and text using Vision Language Models. Real-time embedding visualization, semantic similarity, instant retrieval." },
      { property: "og:title", content: "Semantix — Multimodal AI Semantic Retrieval" },
      { property: "og:description", content: "VLM-powered image–text retrieval with live embedding space analysis." },
    ],
  }),
  component: Landing,
});

const features = [
  { icon: Type, title: "Text → Image Retrieval", desc: "Describe what you want in natural language and surface the most semantically aligned images." },
  { icon: ImageIcon, title: "Image → Text Retrieval", desc: "Upload an image and receive ranked captions and descriptions that match its content." },
  { icon: Network, title: "Real-time Embedding Map", desc: "Watch query and result vectors materialize in shared embedding space as you search." },
  { icon: Sparkles, title: "Semantic Similarity", desc: "Cosine-distance ranking across a shared multimodal latent space, not keyword matching." },
  { icon: Layers, title: "Vision Language Model", desc: "CLIP-class encoder produces 512-d joint embeddings for any modality." },
  { icon: Gauge, title: "Sub-second Retrieval", desc: "Approximate nearest-neighbor index returns top-K matches in under 50ms." },
];

const steps = [
  { n: "01", title: "Input", desc: "Image or natural-language query" },
  { n: "02", title: "Encode", desc: "VLM → 512-d embedding vector" },
  { n: "03", title: "Match", desc: "ANN search · cosine similarity" },
  { n: "04", title: "Retrieve", desc: "Ranked results · visualized" },
];

function Landing() {
  return (
    <div>
      {/* HERO */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 grid-bg opacity-40" />
        <div className="absolute inset-0">
          <ParticleField density={70} />
        </div>
        <div className="absolute -top-40 left-1/2 -translate-x-1/2 h-[500px] w-[800px] rounded-full bg-gradient-aurora opacity-30 blur-3xl animate-pulse-glow" />

        <div className="relative max-w-6xl mx-auto px-6 pt-20 pb-32 text-center">
          <div className="inline-flex items-center gap-2 glass rounded-full px-4 py-1.5 mb-8 animate-fade-in">
            <span className="h-1.5 w-1.5 rounded-full bg-primary shadow-[0_0_8px_currentColor]" />
            <span className="text-xs font-mono text-muted-foreground tracking-wider">
              VLM · v1.0 · MULTIMODAL INDEX ONLINE
            </span>
          </div>

          <h1 className="text-5xl md:text-7xl lg:text-8xl font-bold tracking-tight leading-[0.95] animate-slide-up">
            Search across <br />
            <span className="text-gradient">pixels and words.</span>
          </h1>
          <p className="mt-6 max-w-2xl mx-auto text-base md:text-lg text-muted-foreground animate-slide-up">
            Semantix is a multimodal retrieval engine powered by Vision Language Models.
            Find images from text, find text from images, and watch every query
            unfold in real-time embedding space.
          </p>

          <div className="mt-10 flex flex-wrap items-center justify-center gap-3 animate-slide-up">
            <Link
              to="/dashboard"
              className="group relative inline-flex items-center gap-2 rounded-xl bg-gradient-aurora px-6 py-3 text-sm font-semibold text-background shadow-neon hover:scale-[1.02] transition"
            >
              Try Retrieval <ArrowRight className="h-4 w-4 group-hover:translate-x-0.5 transition" />
            </Link>
            <Link
              to="/embeddings"
              className="inline-flex items-center gap-2 rounded-xl glass px-6 py-3 text-sm font-semibold hover:bg-foreground/5 transition"
            >
              <Network className="h-4 w-4" /> Explore Embeddings
            </Link>
          </div>

          {/* Stat strip */}
          <div className="mt-20 grid grid-cols-2 md:grid-cols-4 gap-4 max-w-3xl mx-auto">
            {[
              ["512", "Latent dims"],
              ["<50ms", "Top-K latency"],
              ["12M+", "Indexed vectors"],
              ["99.4%", "MRR@10"],
            ].map(([v, k]) => (
              <div key={k} className="glass rounded-xl px-4 py-4">
                <div className="text-2xl font-bold text-gradient">{v}</div>
                <div className="text-[11px] font-mono text-muted-foreground tracking-wider uppercase mt-1">{k}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* FEATURES */}
      <section className="relative py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <div className="text-xs font-mono text-primary tracking-[0.3em] uppercase mb-3">Capabilities</div>
            <h2 className="text-4xl md:text-5xl font-bold tracking-tight">
              Engineered for <span className="text-gradient">semantic understanding</span>
            </h2>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {features.map((f, i) => (
              <div
                key={f.title}
                className="group relative glass rounded-2xl p-6 hover:bg-foreground/5 transition-all hover:-translate-y-1"
                style={{ animationDelay: `${i * 50}ms` }}
              >
                <div className="absolute -inset-px rounded-2xl bg-gradient-aurora opacity-0 group-hover:opacity-20 blur transition" />
                <div className="relative">
                  <div className="h-11 w-11 rounded-xl bg-gradient-aurora/20 border border-border flex items-center justify-center mb-4">
                    <f.icon className="h-5 w-5 text-primary" />
                  </div>
                  <h3 className="text-lg font-semibold mb-1.5">{f.title}</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">{f.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* WORKFLOW */}
      <section className="relative py-24 px-6 overflow-hidden">
        <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 h-px bg-linear-to-r from-transparent via-primary/40 to-transparent" />
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <div className="text-xs font-mono text-accent tracking-[0.3em] uppercase mb-3">Pipeline</div>
            <h2 className="text-4xl md:text-5xl font-bold tracking-tight">
              From query to <span className="text-gradient">ranked retrieval</span>
            </h2>
          </div>

          <div className="grid md:grid-cols-4 gap-4 relative">
            {steps.map((s, i) => (
              <div key={s.n} className="relative glass rounded-2xl p-6 group">
                <div className="font-mono text-xs text-primary tracking-widest">{s.n}</div>
                <div className="mt-2 text-xl font-semibold">{s.title}</div>
                <div className="mt-1.5 text-sm text-muted-foreground">{s.desc}</div>
                <div className="mt-6 h-1.5 rounded-full bg-white/5 overflow-hidden">
                  <div
                    className="h-full bg-gradient-aurora"
                    style={{
                      width: `${(i + 1) * 25}%`,
                      animation: `shimmer 2.5s linear infinite`,
                      backgroundSize: "200% 100%",
                    }}
                  />
                </div>
                {i < steps.length - 1 && (
                  <ArrowRight className="hidden md:block absolute -right-3 top-1/2 -translate-y-1/2 h-5 w-5 text-primary/60 z-10" />
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="relative py-24 px-6">
        <div className="max-w-4xl mx-auto glass-strong border-gradient rounded-3xl p-12 text-center relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-radial opacity-60" />
          <div className="relative">
            <h2 className="text-4xl md:text-5xl font-bold tracking-tight">
              Ready to <span className="text-gradient">see semantics</span>?
            </h2>
            <p className="mt-4 text-muted-foreground max-w-xl mx-auto">
              Open the dashboard and run your first cross-modal retrieval in seconds.
            </p>
            <Link
              to="/dashboard"
              className="mt-8 inline-flex items-center gap-2 rounded-xl bg-gradient-aurora px-6 py-3 text-sm font-semibold text-background shadow-neon"
            >
              Launch dashboard <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="border-t border-border py-10 px-6 mt-12">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-2 text-sm text-muted-foreground font-mono">
            <Sparkles className="h-4 w-4 text-primary" />
            <span className="text-gradient font-semibold">Semantix</span>
            <span>· © 2026</span>
          </div>
          <div className="flex items-center gap-6 text-xs font-mono text-muted-foreground">
            <a className="hover:text-foreground transition flex items-center gap-1.5" href="#"><BookOpen className="h-3.5 w-3.5" /> Docs</a>
            <a className="hover:text-foreground transition flex items-center gap-1.5" href="#"><FileText className="h-3.5 w-3.5" /> Research</a>
            <a className="hover:text-foreground transition flex items-center gap-1.5" href="#"><Mail className="h-3.5 w-3.5" /> Contact</a>
            <a className="hover:text-foreground transition flex items-center gap-1.5" href="#"><Github className="h-3.5 w-3.5" /> GitHub</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
