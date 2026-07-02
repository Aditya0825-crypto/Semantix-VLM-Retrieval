import { useEffect, useRef } from "react";
import { useTheme } from "@/components/ThemeProvider";

export function ParticleField({ density = 60 }: { density?: number }) {
  const { isDark } = useTheme();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    let raf = 0;

    const resize = () => {
      canvas.width = canvas.offsetWidth * devicePixelRatio;
      canvas.height = canvas.offsetHeight * devicePixelRatio;
    };
    resize();
    window.addEventListener("resize", resize);

    const lineAlpha = isDark ? 0.15 : 0.1;
    const dotAlpha = isDark ? 0.8 : 0.55;
    const glowBlur = isDark ? 8 : 4;

    type P = { x: number; y: number; vx: number; vy: number; r: number; hue: number };
    const points: P[] = Array.from({ length: density }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      r: Math.random() * 1.8 + 0.6,
      hue: Math.random() < 0.5 ? 200 : 295,
    }));

    const tick = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      for (const p of points) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
      }
      for (let i = 0; i < points.length; i++) {
        for (let j = i + 1; j < points.length; j++) {
          const dx = points[i].x - points[j].x;
          const dy = points[i].y - points[j].y;
          const d = Math.hypot(dx, dy);
          if (d < 140 * devicePixelRatio) {
            const a = 1 - d / (140 * devicePixelRatio);
            ctx.strokeStyle = `hsla(${points[i].hue}, ${isDark ? 90 : 70}%, ${isDark ? 70 : 45}%, ${a * lineAlpha})`;
            ctx.lineWidth = devicePixelRatio * 0.6;
            ctx.beginPath();
            ctx.moveTo(points[i].x, points[i].y);
            ctx.lineTo(points[j].x, points[j].y);
            ctx.stroke();
          }
        }
      }
      for (const p of points) {
        ctx.fillStyle = `hsla(${p.hue}, ${isDark ? 95 : 75}%, ${isDark ? 70 : 42}%, ${dotAlpha})`;
        ctx.shadowColor = `hsla(${p.hue}, 95%, 65%, 0.9)`;
        ctx.shadowBlur = glowBlur * devicePixelRatio;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r * devicePixelRatio, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.shadowBlur = 0;
      raf = requestAnimationFrame(tick);
    };
    tick();

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, [density, isDark]);

  return (
    <canvas
      ref={canvasRef}
      className={`absolute inset-0 w-full h-full pointer-events-none ${isDark ? "opacity-70" : "opacity-45"}`}
    />
  );
}
