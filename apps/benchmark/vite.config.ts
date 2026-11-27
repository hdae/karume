import { fileURLToPath } from "node:url";
import { readCheckout } from "@karume/sweep-app-support/node";
import { defineConfig } from "vite-plus";

const REPOSITORY_ROOT = fileURLToPath(new URL("../../", import.meta.url));

/** Cross-origin isolation on the dev and preview servers, as the workbench's (a static host sets its own headers). */
const ISOLATION = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig(async () => ({
  // The checkout every record of the page carries, read when the config loads: when the dev server
  // starts, or when the static site is built.
  define: { __BENCHMARK_CHECKOUT__: JSON.stringify(await readCheckout(REPOSITORY_ROOT)) },
  // Relative asset paths, so the built site runs from any directory of any static host.
  base: "./",
  // Loopback only, as the workbench (8791) and the playground (8790).
  server: { host: "127.0.0.1", port: 8792, strictPort: true, headers: ISOLATION },
  preview: { host: "127.0.0.1", port: 8792, strictPort: true, headers: ISOLATION },
  test: { include: ["tests/**/*_test.ts"] },
}));
