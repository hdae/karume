import { fileURLToPath } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import { WAN_SOURCES } from "@karume/models/wan";
import { tanstackRouter } from "@tanstack/router-plugin/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite-plus";

import { distributionsPlugin } from "./vite-plugin-distributions.ts";

export default defineConfig({
  plugins: [
    // Emit the generated route tree in the repository's format (double quotes, semicolons), so
    // `vp check` accepts it as generated.
    tanstackRouter({
      target: "react",
      autoCodeSplitting: true,
      quoteStyle: "double",
      semicolons: true,
    }),
    react(),
    tailwindcss(),
    // Paths are relative to the repository root (`models/` is the local asset root — docs/assets-layout.md).
    // Wan2.2 is also served on the HF-compatible route under its public repo name, so the page can
    // read it through the hub's cache; umT5 is served there as the cross-repo reference of its manifest.
    distributionsPlugin({
      distributions: {
        wan22: { directory: "models/karume-wan2.2", hfRepo: WAN_SOURCES["wan2.2"].repo },
        umt5: "models/karume-umt5-xxl",
      },
    }),
  ],
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  // Loopback only: another machine reaches the page through an SSH tunnel (README).
  server: { host: "127.0.0.1", port: 8790, strictPort: true },
  preview: { host: "127.0.0.1", port: 8790, strictPort: true },
  test: { include: ["tests/**/*_test.ts"] },
});
