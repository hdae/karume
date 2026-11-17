import { MOUNTS } from "@karume/local-distributions/mounts";
import { distributionsPlugin } from "@karume/local-distributions/plugin";
import { defineConfig } from "vite-plus";

import { resultsPlugin } from "./vite-plugin-results.ts";

export default defineConfig(({ command }) => ({
  plugins: [
    // `/models/` and `/hf/` from the mount table the playground also serves, `/inputs/` (the
    // files a local entry reads by ranges — the playground does not serve it), and COOP / COEP on
    // every dev and preview response (first, so the results plugin's answers carry them too).
    distributionsPlugin({ distributions: MOUNTS, inputs: "inputs" }),
    // The write-back and the checkout endpoint (dev server only), the baked-in checkout
    // (production build).
    resultsPlugin(),
  ],
  // Loopback only: another machine reaches the page through an SSH tunnel (README). The playground
  // listens on 8790.
  server: { host: "127.0.0.1", port: 8791, strictPort: true },
  preview: { host: "127.0.0.1", port: 8791, strictPort: true },
  // A production build serves no distribution: the registry's module of local entries resolves to
  // an empty one, so those entries are outside the build's module graph.
  resolve: {
    alias:
      command === "build"
        ? [{ find: "./local-entries.ts", replacement: "./no-local-entries.ts" }]
        : [],
  },
  test: {
    include: ["tests/**/*_test.ts"],
    // The registry lists entries with `import.meta.glob`, whose expansion depends on which files
    // exist; a cached transform of it would outlive an added or removed entry file.
    fsModuleCache: false,
  },
}));
