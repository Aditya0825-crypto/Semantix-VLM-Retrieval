import { useEffect, useRef, useState } from "react";
import { useTheme } from "@/components/ThemeProvider";

export type EmbeddingPoint = {
  id: string;
  x: number; // -1..1
  y: number; // -1..1
  label: string;
  kind: "query" | "result" | "cluster";
  similarity?: number; // 0..1
};

interface Props {
  points: EmbeddingPoint[];
  animatingKey?: number | string;
  height?: number;
  /** Active Top-K — drives neighborhood halo radius and overlay badge. */
  topK?: number;
}

type CanvasPalette = {
  grid: string;
  glowCenter: string;
  glowEdge: string;
  query: string;
  result: string;
  cluster: string;
  lineFrom: (a: number) => string;
  lineTo: (a: number) => string;
  haloInner: string;
  haloMid: string;
  haloOuter: string;
  haloStroke: (n: number) => string;
  queryPulse: (pulse: number) => string;
  canvasBg: string;
  legendQuery: string;
  legendResult: string;
  legendCluster: string;
};

const PALETTES: Record<"light" | "dark", CanvasPalette> = {
  dark: {
    grid: "rgba(255,255,255,0.04)",
    glowCenter: "rgba(120, 80, 220, 0.18)",
    glowEdge: "rgba(0,0,0,0)",
    query: "rgba(120, 230, 255, 1)",
    result: "rgba(240, 130, 220, 1)",
    cluster: "rgba(200, 120, 255, 0.25)",
    lineFrom: (a) => `rgba(120, 220, 255, ${a})`,
    lineTo: (a) => `rgba(200, 120, 255, ${a * 0.6})`,
    haloInner: "rgba(120, 220, 255, 0.12)",
    haloMid: "rgba(200, 120, 255, 0.06)",
    haloOuter: "rgba(0, 0, 0, 0)",
    haloStroke: (n) => `rgba(120, 230, 255, ${0.15 + n * 0.02})`,
    queryPulse: (pulse) => `rgba(120, 230, 255, ${0.5 - pulse * 0.3})`,
    canvasBg: "rgba(18, 16, 28, 0.85)",
    legendQuery: "rgb(120,230,255)",
    legendResult: "rgb(240,130,220)",
    legendCluster: "rgb(200,120,255)",
  },
  light: {
    grid: "rgba(15, 23, 42, 0.07)",
    glowCenter: "rgba(80, 120, 220, 0.12)",
    glowEdge: "rgba(255,255,255,0)",
    query: "rgba(0, 140, 190, 1)",
    result: "rgba(170, 40, 150, 1)",
    cluster: "rgba(120, 60, 200, 0.2)",
    lineFrom: (a) => `rgba(0, 160, 210, ${a})`,
    lineTo: (a) => `rgba(140, 60, 200, ${a * 0.65})`,
    haloInner: "rgba(0, 160, 210, 0.14)",
    haloMid: "rgba(120, 60, 200, 0.08)",
    haloOuter: "rgba(255, 255, 255, 0)",
    haloStroke: (n) => `rgba(0, 140, 190, ${0.2 + n * 0.025})`,
    queryPulse: (pulse) => `rgba(0, 140, 190, ${0.45 - pulse * 0.25})`,
    canvasBg: "rgba(248, 250, 255, 0.92)",
    legendQuery: "rgb(0,140,190)",
    legendResult: "rgb(170,40,150)",
    legendCluster: "rgb(120,60,200)",
  },
};

export function EmbeddingCanvas({ points, animatingKey, height = 480, topK }: Props) {
  const { theme } = useTheme();
  const palette = PALETTES[theme];
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<EmbeddingPoint | null>(null);
  const [mouse, setMouse] = useState({ x: 0, y: 0 });
  const stateRef = useRef({ zoom: 1, ox: 0, oy: 0, dragging: false, lx: 0, ly: 0 });
  const progressRef = useRef(0);

  useEffect(() => {
    progressRef.current = 0;
  }, [animatingKey]);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    let raf = 0;

    const resize = () => {
      canvas.width = canvas.offsetWidth * devicePixelRatio;
      canvas.height = canvas.offsetHeight * devicePixelRatio;
    };
    resize();
    window.addEventListener("resize", resize);

    const draw = () => {
      const w = canvas.width;
      const h = canvas.height;
      const s = stateRef.current;
      ctx.clearRect(0, 0, w, h);

      // themed canvas fill
      ctx.fillStyle = palette.canvasBg;
      ctx.fillRect(0, 0, w, h);

      // grid
      ctx.strokeStyle = palette.grid;
      ctx.lineWidth = devicePixelRatio;
      const gridSize = 40 * devicePixelRatio * s.zoom;
      for (let x = (s.ox * devicePixelRatio) % gridSize; x < w; x += gridSize) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      }
      for (let y = (s.oy * devicePixelRatio) % gridSize; y < h; y += gridSize) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }

      // center radial glow
      const cx = w / 2 + s.ox * devicePixelRatio;
      const cy = h / 2 + s.oy * devicePixelRatio;
      const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, 300 * devicePixelRatio);
      grad.addColorStop(0, palette.glowCenter);
      grad.addColorStop(1, palette.glowEdge);
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);

      const toScreen = (p: EmbeddingPoint) => {
        const px = cx + p.x * 220 * devicePixelRatio * s.zoom;
        const py = cy + p.y * 220 * devicePixelRatio * s.zoom;
        return { px, py };
      };

      progressRef.current = Math.min(1, progressRef.current + 0.02);
      const prog = progressRef.current;

      const query = points.find((p) => p.kind === "query");
      const retrieved = points.filter((p) => p.kind === "result" && p.similarity !== undefined);
      if (query) {
        const q = toScreen(query);

        if (retrieved.length > 0) {
          const maxDist = Math.max(
            ...retrieved.map((p) => Math.hypot(p.x - query.x, p.y - query.y)),
            0.12,
          );
          const kScale = topK ? 0.85 + Math.sqrt(topK) * 0.12 : 1;
          const haloR = (maxDist * 1.35 * kScale) * 220 * devicePixelRatio * s.zoom * prog;
          const halo = ctx.createRadialGradient(q.px, q.py, 0, q.px, q.py, haloR);
          halo.addColorStop(0, palette.haloInner);
          halo.addColorStop(0.6, palette.haloMid);
          halo.addColorStop(1, palette.haloOuter);
          ctx.fillStyle = halo;
          ctx.beginPath();
          ctx.arc(q.px, q.py, haloR, 0, Math.PI * 2);
          ctx.fill();

          ctx.strokeStyle = palette.haloStroke(retrieved.length);
          ctx.lineWidth = devicePixelRatio;
          ctx.setLineDash([4 * devicePixelRatio, 4 * devicePixelRatio]);
          ctx.beginPath();
          ctx.arc(q.px, q.py, haloR, 0, Math.PI * 2);
          ctx.stroke();
          ctx.setLineDash([]);
        }

        for (const p of points) {
          if (p.kind !== "result") continue;
          const t = toScreen(p);
          const sim = p.similarity ?? 0.5;
          const a = sim * 0.7 * prog;
          const lg = ctx.createLinearGradient(q.px, q.py, t.px, t.py);
          lg.addColorStop(0, palette.lineFrom(a));
          lg.addColorStop(1, palette.lineTo(a));
          ctx.strokeStyle = lg;
          ctx.lineWidth = (1 + sim * 1.8) * devicePixelRatio;
          ctx.beginPath();
          ctx.moveTo(q.px, q.py);
          const mx = (q.px + t.px) / 2 + (Math.sin(Date.now() / 1000 + p.x * 4) * 20 * devicePixelRatio);
          const my = (q.py + t.py) / 2 + (Math.cos(Date.now() / 1000 + p.y * 4) * 20 * devicePixelRatio);
          ctx.quadraticCurveTo(mx, my, t.px, t.py);
          ctx.stroke();
        }
      }

      for (const p of points) {
        const { px, py } = toScreen(p);
        const baseR =
          p.kind === "query" ? 9 : p.kind === "cluster" ? 14 : 5 + (p.similarity ?? 0.5) * 4;
        const r = baseR * devicePixelRatio * prog;
        const color =
          p.kind === "query" ? palette.query
          : p.kind === "cluster" ? palette.cluster
          : palette.result;

        if (p.kind === "cluster") {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(px, py, r * 4, 0, Math.PI * 2);
          ctx.fill();
        } else {
          ctx.shadowColor = color;
          ctx.shadowBlur = theme === "dark" ? 18 * devicePixelRatio : 10 * devicePixelRatio;
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(px, py, r, 0, Math.PI * 2);
          ctx.fill();
          ctx.shadowBlur = 0;

          if (p.kind === "query") {
            const pulse = (Math.sin(Date.now() / 400) + 1) / 2;
            ctx.strokeStyle = palette.queryPulse(pulse);
            ctx.lineWidth = devicePixelRatio;
            ctx.beginPath();
            ctx.arc(px, py, r + 8 * devicePixelRatio + pulse * 16 * devicePixelRatio, 0, Math.PI * 2);
            ctx.stroke();
          }
        }
      }

      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, [points, topK, palette, theme]);

  const handleMove = (e: React.MouseEvent) => {
    const c = canvasRef.current!;
    const rect = c.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    setMouse({ x: mx, y: my });
    const s = stateRef.current;
    if (s.dragging) {
      s.ox += (e.clientX - s.lx);
      s.oy += (e.clientY - s.ly);
      s.lx = e.clientX; s.ly = e.clientY;
      return;
    }
    const cx = c.offsetWidth / 2 + s.ox;
    const cy = c.offsetHeight / 2 + s.oy;
    let nearest: EmbeddingPoint | null = null;
    let nd = 18;
    for (const p of points) {
      if (p.kind === "cluster") continue;
      const px = cx + p.x * 220 * s.zoom;
      const py = cy + p.y * 220 * s.zoom;
      const d = Math.hypot(px - mx, py - my);
      if (d < nd) { nd = d; nearest = p; }
    }
    setHover(nearest);
  };

  return (
    <div className="relative w-full rounded-2xl overflow-hidden glass" style={{ height }}>
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full cursor-crosshair"
        onMouseMove={handleMove}
        onMouseDown={(e) => {
          const s = stateRef.current; s.dragging = true; s.lx = e.clientX; s.ly = e.clientY;
        }}
        onMouseUp={() => { stateRef.current.dragging = false; }}
        onMouseLeave={() => { stateRef.current.dragging = false; setHover(null); }}
        onWheel={(e) => {
          const s = stateRef.current;
          s.zoom = Math.max(0.5, Math.min(3, s.zoom + (e.deltaY < 0 ? 0.1 : -0.1)));
        }}
      />
      {hover && (
        <div
          className="absolute pointer-events-none glass-strong rounded-lg px-3 py-2 text-xs font-mono z-10"
          style={{ left: mouse.x + 14, top: mouse.y + 14 }}
        >
          <div className="text-foreground font-semibold">{hover.label}</div>
          <div className="text-muted-foreground">
            kind: <span className="text-primary">{hover.kind}</span>
          </div>
          {hover.similarity !== undefined && (
            <div className="text-muted-foreground">
              sim: <span className="text-accent">{(hover.similarity * 100).toFixed(1)}%</span>
            </div>
          )}
          <div className="text-muted-foreground">
            vec: [{hover.x.toFixed(2)}, {hover.y.toFixed(2)}, …]
          </div>
        </div>
      )}
      <div className="absolute bottom-3 left-3 flex flex-wrap gap-2 text-[10px] font-mono text-muted-foreground">
        <span className="glass rounded-md px-2 py-1">
          scroll: zoom · drag: pan
        </span>
        {topK !== undefined && (
          <span className="glass rounded-md px-2 py-1 text-primary">
            K={topK} · {retrievedCount(points)} neighbors
          </span>
        )}
      </div>
      <div className="absolute top-3 right-3 flex flex-wrap justify-end gap-2 text-[10px] font-mono">
        <Legend color={palette.legendQuery} label="QUERY" />
        <Legend color={palette.legendResult} label="RETRIEVED" />
        <Legend color={palette.legendCluster} label="CLUSTER" muted />
      </div>
    </div>
  );
}

function retrievedCount(points: EmbeddingPoint[]) {
  return points.filter((p) => p.kind === "result" && p.similarity !== undefined).length;
}

function Legend({ color, label, muted }: { color: string; label: string; muted?: boolean }) {
  return (
    <div className="flex items-center gap-1.5 glass rounded-md px-2 py-1">
      <span
        className="h-2 w-2 rounded-full"
        style={{ background: color, boxShadow: `0 0 8px ${color}`, opacity: muted ? 0.5 : 1 }}
      />
      <span className="text-muted-foreground tracking-wider">{label}</span>
    </div>
  );
}
