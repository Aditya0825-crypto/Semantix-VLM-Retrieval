// @lovable.dev/vite-tanstack-config already includes the following — do NOT add them manually
// or the app will break with duplicate plugins:
//   - tanstackStart, viteReact, tailwindcss, tsConfigPaths, cloudflare (build-only),
//     componentTagger (dev-only), VITE_* env injection, @ path alias, React/TanStack dedupe,
//     error logger plugins, and sandbox detection (port/host/strictPort).
// You can pass additional config via defineConfig({ vite: { ... } }) if needed.
import type { Plugin, RunnableDevEnvironment, ViteDevServer } from "vite";
import { createServerModuleRunner, isRunnableDevEnvironment } from "vite";
import { defineConfig } from "@lovable.dev/vite-tanstack-config";

const SSR_ENV = "ssr";
const SERVER_ENTRY = "virtual:tanstack-start-server-entry";
const MODULE_TIMEOUT_MS = 300_000;

type TransportWithTimeout = { timeout?: number };

function getSsrEnv(server: ViteDevServer): RunnableDevEnvironment | undefined {
  const env = server.environments[SSR_ENV];
  return env && isRunnableDevEnvironment(env) ? env : undefined;
}

function setTransportTimeout(runner: object): void {
  (runner as { transport: TransportWithTimeout }).transport.timeout =
    MODULE_TIMEOUT_MS;
}

function patchRunnerTimeout(server: ViteDevServer): boolean {
  const env = getSsrEnv(server);
  if (!env) return false;
  try {
    setTransportTimeout(env.runner);
    return true;
  } catch {
    return false;
  }
}

/**
 * TanStack Start SSR uses Vite's module runner (default 60s RPC timeout).
 * On Windows the first compile of virtual:tanstack-start-server-entry often exceeds that.
 */
function tanstackSsrDevFix(): Plugin {
  return {
    name: "tanstack-ssr-dev-fix",
    apply: "serve",
    enforce: "pre",
    configureServer(server) {
      const ssrEnv = getSsrEnv(server);
      if (ssrEnv) {
        const internal = ssrEnv as unknown as {
          _runnerFactory?: typeof createServerModuleRunner;
        };
        const originalFactory = internal._runnerFactory ?? createServerModuleRunner;
        if (!internal._runnerFactory) {
          internal._runnerFactory = (
            environment: Parameters<typeof createServerModuleRunner>[0],
            options?: Parameters<typeof createServerModuleRunner>[1],
          ) => {
            const runner = originalFactory(environment, options);
            setTransportTimeout(runner);
            return runner;
          };
        }
      }

      server.httpServer?.once("listening", () => {
        void (async () => {
          patchRunnerTimeout(server);
          const warmed = getSsrEnv(server);
          if (!warmed) return;
          try {
            console.log("[tanstack-ssr-dev-fix] Pre-compiling SSR entry (first run may take 1–2 min)…");
            await warmed.runner.import(SERVER_ENTRY);
            console.log("[tanstack-ssr-dev-fix] SSR entry ready.");

            const port = server.config.server.port ?? 8080;
            const host = server.config.server.host === true ? "127.0.0.1" : (server.config.server.host ?? "localhost");
            const origin = `http://${String(host).replace("::", "127.0.0.1")}:${port}`;
            console.log("[tanstack-ssr-dev-fix] Warming routes…");
            await fetch(`${origin}/`);
            await fetch(`${origin}/dashboard`);
            console.log("[tanstack-ssr-dev-fix] Dev server fully warmed — open", origin);
          } catch (err) {
            console.warn("[tanstack-ssr-dev-fix] SSR warmup failed:", err);
          }
        })();
      });

      return () => {
        server.middlewares.use((_req, _res, next) => {
          patchRunnerTimeout(server);
          next();
        });
      };
    },
  };
}

// Production / Cloudflare uses src/server.ts (SSR error wrapper).
export default defineConfig(({ command }) => ({
  tanstackStart: {
    // Lighter dev SSR (shell); full SSR only on production build.
    ...(command === "serve" ? { spa: { enabled: true } } : {}),
    ...(command === "build" ? { server: { entry: "server" } } : {}),
  },
  plugins: [tanstackSsrDevFix()],
  vite: {
    server: {
      hmr: { timeout: MODULE_TIMEOUT_MS },
      warmup: {
        clientFiles: [
          "./src/routes/dashboard.tsx",
          "./src/routes/index.tsx",
          "./src/routes/embeddings.tsx",
          "./src/components/EmbeddingCanvas.tsx",
        ],
      },
    },
    optimizeDeps: {
      include: [
        "react",
        "react-dom",
        "react/jsx-runtime",
        "@tanstack/react-router",
        "@tanstack/react-query",
        "lucide-react",
      ],
    },
  },
}));
