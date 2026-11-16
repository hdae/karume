import { defineConfig } from "vite-plus";

import { resultsPlugin } from "./vite-plugin-results.ts";

export default defineConfig({
  // The write-back and the checkout endpoint (dev server only), cross-origin isolation (dev and
  // preview), the baked-in checkout (production build).
  plugins: [resultsPlugin()],
  // Loopback only: another machine reaches the page through an SSH tunnel (README). The playground
  // listens on 8790.
  server: { host: "127.0.0.1", port: 8791, strictPort: true },
  preview: { host: "127.0.0.1", port: 8791, strictPort: true },
  test: { include: ["tests/**/*_test.ts"] },
});
