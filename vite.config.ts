import { defineConfig } from "vite-plus";
import { laneProjects } from "./packages/test-support/src/lanes.ts";

export default defineConfig({
  fmt: {
    ignorePatterns: [
      // Documents stay hand-formatted: the formatter would add trailing commas to the JSONC
      // manifest examples and reflow the WGSL / TS snippets that documents quote verbatim.
      "**/*.md",
      // The Python packages keep their own tooling (uv, ruff).
      "tools/exporter/**",
      "tools/export-recipes/**",
    ],
  },
  lint: {
    // Repository rule (ADR 0002): TypedArray types in packages/*/src must name their buffer type.
    jsPlugins: [{ name: "karume", specifier: "./tools/lint/oxlint-plugin.ts" }],
    rules: {
      "karume/typed-array-buffer": "error",
      // Off (maintainer ruling 2026-10-17, phase 2 step 2-3):
      // - the conditional-property idiom `...(cond ? {} : { x })` is deliberate under
      //   exactOptionalPropertyTypes (an absent key is not an undefined value);
      "unicorn/no-useless-spread": "off",
      // - `new Array<number>(n)` pre-sizes the reference implementations on purpose
      //   (Array.from({ length }) is slower and no clearer);
      "unicorn/no-new-array": "off",
      // - error messages interpolate op names and unions; the real risk, an object printed as
      //   [object Object], stays covered by no-base-to-string.
      "typescript/restrict-template-expressions": "off",
    },
    options: { typeAware: true, typeCheck: true },
  },
  test: {
    // No root `include`: the lanes (projects) define what runs; `extends: true` would otherwise
    // concatenate a root include into every lane.
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
    // Lanes (see packages/test-support/src/lanes.ts): `vp test --project <lane>`.
    projects: laneProjects(),
  },
});
