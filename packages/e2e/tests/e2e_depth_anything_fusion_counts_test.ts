/**
 * Fusion hit counts of the Depth Anything V2 graph (no GPU — `planFusions` on part 0 only; the
 * shared planning side and the rationale live in `./helpers/fusion-counts.ts`).
 *
 * MUST: the asset is the distribution mirror `models/karume-depth-anything-v2/` (untracked local
 * asset); where it is missing the case SKIPs explicitly with a reason (ADR 0005).
 */

import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { parseManifest } from "@karume/hub";
import type { ContainerManifest } from "../../runtime/tests/helpers/container-graph.ts";
import { mirrorPresent, mirrorRoot } from "../../models/tests/helpers/mirror.ts";
import { boundFusionCounts, NONE, readDistributionGraph } from "./helpers/fusion-counts.ts";

const MIRROR = mirrorRoot("karume-depth-anything-v2");
const AVAILABLE = mirrorPresent(MIRROR);
if (!AVAILABLE) {
  console.warn(
    `[karume] ${MIRROR.pathname} has no karume.json, so the Depth Anything fusion hit counts SKIP` +
      " (build it with dist.py in tools/export-recipes)",
  );
}

/** The distribution is self-contained: a cross-repo reference fails loudly. */
const NO_MIRRORS: ReadonlyMap<string, URL> = new Map();

/**
 * The small model's graph (a 12-layer ViT encoder and the DPT head, at the static 518² input):
 *
 * - **identityExpand 49**: every one of its 49 `expand` nodes (four around each of the 12
 *   attentions, and the class token's) keeps its input's shape and is an alias.
 * - Nothing else. The encoder normalises with `layer_norm` straight into a `linear` (adaLN wants
 *   the modulation chain after the norm); its 12 `linear → gelu → linear` pairs would fold only
 *   into an i8a8 linear (the linear prologue, which needs `linearCompute: "a8"` and i8 storage):
 *   the distribution stores f32 only, so even `"a8"` folds none, and no preset sets it;
 *   the head resizes with `upsample_bilinear2d`, not the VAE's nearest ×2 chain that upsample2x
 *   takes.
 */
test.skipIf(!AVAILABLE)(
  "the real Depth Anything V2 graph keeps its 49 identity expands and grabs no fusion",
  async () => {
    const text = await readTextFile(new URL("karume.json", MIRROR));
    const manifest: ContainerManifest = JSON.parse(text);
    const parsed = parseManifest(text);
    assertEquals(Object.keys(parsed.models), ["small"], "the models planned");
    for (const [preset, quant] of Object.entries(parsed.models.small.quants)) {
      assertEquals(quant.session, {}, `${preset}: the preset sets no Session option`);
      const storage = quant.weights.depth;
      const graph = await readDistributionGraph(
        MIRROR,
        manifest,
        "small",
        "depth",
        storage,
        NO_MIRRORS,
      );
      assertEquals(
        boundFusionCounts(graph, {}),
        { ...NONE, identityExpand: 49 },
        `${preset} depth/${storage}`,
      );
    }
  },
);
