import { Link, useRouterState } from "@tanstack/react-router";
import { Sparkles, LayoutDashboard, Network, Home, Github } from "lucide-react";
import { ThemeToggle } from "@/components/ThemeToggle";

const links = [
  { to: "/", label: "Home", icon: Home },
  { to: "/dashboard", label: "Retrieval", icon: LayoutDashboard },
  { to: "/embeddings", label: "Embeddings", icon: Network },
];

export function Navbar() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  return (
    <header className="fixed top-0 inset-x-0 z-50 px-4 pt-4">
      <nav className="glass-strong mx-auto max-w-6xl rounded-2xl px-4 py-2.5 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2 group">
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-aurora blur-md opacity-70 group-hover:opacity-100 transition" />
            <div className="relative h-8 w-8 rounded-lg bg-gradient-aurora flex items-center justify-center">
              <Sparkles className="h-4 w-4 text-background" />
            </div>
          </div>
          <span className="font-display font-semibold tracking-tight">
            <span className="text-gradient">Semantix</span>
            <span className="text-muted-foreground text-xs ml-1.5 font-mono">/ai</span>
          </span>
        </Link>

        <div className="hidden md:flex items-center gap-1">
          {links.map((l) => {
            const active = pathname === l.to;
            return (
              <Link
                key={l.to}
                to={l.to}
                className={`relative px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all flex items-center gap-2 ${
                  active
                    ? "text-foreground bg-foreground/5"
                    : "text-muted-foreground hover:text-foreground hover:bg-foreground/5"
                }`}
              >
                <l.icon className="h-3.5 w-3.5" />
                {l.label}
                {active && (
                  <span className="absolute inset-x-3 -bottom-px h-px bg-gradient-aurora" />
                )}
              </Link>
            );
          })}
        </div>

        <div className="flex items-center gap-2">
          <ThemeToggle compact />
          <a
            href="#"
            className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-mono text-muted-foreground hover:text-foreground transition border border-border hover:border-primary/30"
          >
            <Github className="h-3.5 w-3.5" /> v1.0
          </a>
        </div>
      </nav>
    </header>
  );
}
