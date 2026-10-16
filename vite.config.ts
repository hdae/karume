import { defineConfig } from "vite-plus";

export default defineConfig({
  lint: {
    options: { typeAware: true, typeCheck: true },
  },
  test: {
    include: ["packages/*/tests/**/*_test.ts"],
    setupFiles: ["./packages/test-support/src/setup.ts"],
    // GPU tests share one physical device and the sha256 reference rows assume one run at a time:
    // every file runs in a single forked worker, one file after another.
    pool: "forks",
    fileParallelism: false,
    isolate: true,
    // Some tests measure memory and call `globalThis.gc()` (Deno ran with --v8-flags=--expose-gc).
    execArgv: ["--expose-gc"],
    // Deno had no per-test time limit; model lanes legitimately run for minutes.
    testTimeout: 0,
    hookTimeout: 0,
    passWithNoTests: false,
  },
});
