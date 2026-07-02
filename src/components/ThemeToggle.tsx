import { Moon, Sun } from "lucide-react";
import { useTheme } from "@/components/ThemeProvider";

export function ThemeToggle({ compact }: { compact?: boolean }) {
  const { theme, toggleTheme, isDark } = useTheme();

  return (
    <button
      type="button"
      onClick={toggleTheme}
      aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
      title={isDark ? "Light mode" : "Dark mode"}
      className={
        compact
          ? "relative flex h-9 w-9 items-center justify-center rounded-lg border border-border bg-muted/40 text-muted-foreground transition hover:bg-muted hover:text-foreground"
          : "relative flex items-center gap-2 rounded-lg border border-border bg-muted/40 px-3 py-1.5 text-xs font-mono text-muted-foreground transition hover:bg-muted hover:text-foreground"
      }
    >
      <span className="relative h-4 w-4">
        <Sun
          className={`absolute inset-0 h-4 w-4 transition-all duration-300 ${
            isDark ? "rotate-90 scale-0 opacity-0" : "rotate-0 scale-100 opacity-100 text-amber-500"
          }`}
        />
        <Moon
          className={`absolute inset-0 h-4 w-4 transition-all duration-300 ${
            isDark ? "rotate-0 scale-100 opacity-100 text-primary" : "-rotate-90 scale-0 opacity-0"
          }`}
        />
      </span>
      {!compact && <span className="hidden sm:inline">{theme === "dark" ? "Dark" : "Light"}</span>}
    </button>
  );
}
