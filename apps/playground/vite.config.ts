import { fileURLToPath } from "node:url";
import tailwindcss from "@tailwindcss/vite";
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
    distributionsPlugin({
      distributions: { wan22: "models/karume-wan2.2", umt5: "models/karume-umt5-xxl" },
    }),
  ],
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  // Loopback only: another machine reaches the page through an SSH tunnel (README).
  server: { host: "127.0.0.1", port: 8790, strictPort: true },
  preview: { host: "127.0.0.1", port: 8790, strictPort: true },
  test: { include: ["tests/**/*_test.ts"] },
});
