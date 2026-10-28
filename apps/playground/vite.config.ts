import { fileURLToPath } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import { tanstackRouter } from "@tanstack/router-plugin/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite-plus";

import { MOUNTS } from "./src/mounts.ts";
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
    // The mount table (src/mounts.ts): `/models/<name>/` → a directory under `models/`, and the public
    // repo name a mount is also served under on the HF-compatible route.
    distributionsPlugin({ distributions: MOUNTS }),
  ],
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  // Loopback only: another machine reaches the page through an SSH tunnel (README).
  server: { host: "127.0.0.1", port: 8790, strictPort: true },
  preview: { host: "127.0.0.1", port: 8790, strictPort: true },
  test: { include: ["tests/**/*_test.ts"] },
});
