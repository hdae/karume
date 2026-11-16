import { MOUNTS } from "@karume/local-distributions/mounts";
import { distributionsPlugin } from "@karume/local-distributions/plugin";
import { defineConfig } from "vite-plus";

import { resultsPlugin } from "./vite-plugin-results.ts";

export default defineConfig({
  plugins: [
    // `/models/` and `/hf/` from the mount table the playground also serves, and COOP / COEP on
    // every dev and preview response (first, so the results plugin's answers carry them too).
    distributionsPlugin({ distributions: MOUNTS }),
    // The write-back and the checkout endpoint (dev server only), the baked-in checkout
    // (production build).
    resultsPlugin(),
  ],
  // Loopback only: another machine reaches the page through an SSH tunnel (README). The playground
  // listens on 8790.
  server: { host: "127.0.0.1", port: 8791, strictPort: true },
  preview: { host: "127.0.0.1", port: 8791, strictPort: true },
  test: { include: ["tests/**/*_test.ts"] },
});
