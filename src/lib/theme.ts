export type Theme = "light" | "dark";

export const THEME_STORAGE_KEY = "semantix-theme";

export function getStoredTheme(): Theme | null {
  if (typeof window === "undefined") return null;
  const stored = localStorage.getItem(THEME_STORAGE_KEY);
  if (stored === "light" || stored === "dark") return stored;
  return null;
}

export function getSystemTheme(): Theme {
  if (typeof window === "undefined") return "dark";
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function resolveTheme(stored: Theme | null): Theme {
  return stored ?? getSystemTheme();
}

export function applyTheme(theme: Theme) {
  const root = document.documentElement;
  root.classList.remove("light", "dark");
  root.classList.add(theme);
  root.style.colorScheme = theme;
}

export function persistTheme(theme: Theme) {
  localStorage.setItem(THEME_STORAGE_KEY, theme);
}

/** Inline script to prevent theme flash before React hydrates. */
export const THEME_INIT_SCRIPT = `(function(){try{var k="${THEME_STORAGE_KEY}";var s=localStorage.getItem(k);var d=window.matchMedia("(prefers-color-scheme: dark)").matches;var t=s==="light"||s==="dark"?s:(d?"dark":"light");document.documentElement.classList.add(t);document.documentElement.style.colorScheme=t;}catch(e){document.documentElement.classList.add("dark");}})();`;
