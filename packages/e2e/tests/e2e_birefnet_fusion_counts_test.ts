/**
 * Fusion hit counts of the BiRefNet matting graphs — BiRefNet_HR and its derivative Lucida (no GPU
 * — `planFusions` on part 0 only; the shared planning side and the rationale live in
 * `./helpers/fusion-counts.ts`).
 *
 * MUST: the assets are the distribution mirrors `models/karume-birefnet-hr/` and
 * `models/karume-lucida/` (untracked local assets); where one is missing its case SKIPs explicitly
 * with a reason (ADR 0005).
 */

import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { parseManifest } from "@karume/hub";
import type { ContainerManifest } from "../../runtime/tests/helpers/container-graph.ts";
import { mirrorPresent, mirrorRoot } from "../../models/tests/helpers/mirror.ts";
import { boundFusionCounts, NONE, readDistributionGraph } from "./helpers/fusion-counts.ts";

/** Both distributions are self-contained: a cross-repo reference fails loudly. */
const NO_MIRRORS: ReadonlyMap<string, URL> = new Map();

/**
 * Every model (the 1024² and 2048² graphs) and preset of a mirror: the same graph at two input
 * sizes, so the same hits.
 *
 * - **identityExpand 192**: 192 of the 208 `expand` nodes keep their input's shape and are aliases
 *   (around the batched products of the Swin attention).
 * - Nothing else. The Swin backbone normalises with `layer_norm` straight into a `linear` or a
 *   reshape (adaLN wants the modulation chain after the norm) and has no `rms_norm`; the graph's 23
 *   `sigmoid → mul` pairs multiply another tensor by the sigmoid, not the sigmoid's own input, so
 *   they are not SiLU.
 */
const EXPECTED = { ...NONE, identityExpand: 192 };

for (const name of ["karume-birefnet-hr", "karume-lucida"] as const) {
  const mirror = mirrorRoot(name);
  const available = mirrorPresent(mirror);
  if (!available) {
    console.warn(
      `[karume] ${mirror.pathname} has no karume.json, so its BiRefNet fusion hit counts SKIP` +
        " (build it with dist.py in tools/export-recipes)",
    );
  }
  test.skipIf(!available)(
    `the real ${name} matting graphs keep 192 identity expands and grab no fusion on every model and preset`,
    async () => {
      const text = await readTextFile(new URL("karume.json", mirror));
      const manifest: ContainerManifest = JSON.parse(text);
      const parsed = parseManifest(text);
      assertEquals(Object.keys(parsed.models), ["1024", "2048"], "the models planned");
      for (const [model, entry] of Object.entries(parsed.models)) {
        assertEquals(Object.keys(entry.weights), ["matte"], `${model}: the graphs planned`);
        for (const [preset, quant] of Object.entries(entry.quants)) {
          assertEquals(quant.session, {}, `${model} ${preset}: the preset sets no Session option`);
          const storage = quant.weights.matte;
          const graph = await readDistributionGraph(
            mirror,
            manifest,
            model,
            "matte",
            storage,
            NO_MIRRORS,
          );
          assertEquals(
            boundFusionCounts(graph, {}),
            EXPECTED,
            `${model} ${preset} matte/${storage}`,
          );
        }
      }
    },
  );
}
