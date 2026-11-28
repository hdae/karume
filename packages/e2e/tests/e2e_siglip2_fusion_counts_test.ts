/**
 * Fusion hit counts of the SigLIP 2 vision tower graphs (no GPU — `planFusions` on part 0 only; the
 * shared planning side and the rationale live in `./helpers/fusion-counts.ts`). The family is not
 * run by default: `KARUME_ALL_FAMILIES=1 vp test --project models:siglip2`.
 *
 * MUST: the asset is the distribution mirror `models/karume-siglip2/` (untracked local asset);
 * where it is missing the case SKIPs explicitly with a reason (ADR 0005).
 */

import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { parseManifest } from "@karume/hub";
import type { ContainerManifest } from "../../runtime/tests/helpers/container-graph.ts";
import { mirrorPresent, mirrorRoot } from "../../models/tests/helpers/mirror.ts";
import { boundFusionCounts, NONE, readDistributionGraph } from "./helpers/fusion-counts.ts";

const MIRROR = mirrorRoot("karume-siglip2");
const AVAILABLE = mirrorPresent(MIRROR);
if (!AVAILABLE) {
  console.warn(
    `[karume] ${MIRROR.pathname} has no karume.json, so the SigLIP 2 fusion hit counts SKIP` +
      " (build it with dist.py --pipeline siglip2 in tools/export-recipes)",
  );
}

/** The distribution is self-contained: a cross-repo reference fails loudly. */
const NO_MIRRORS: ReadonlyMap<string, URL> = new Map();

/**
 * Per model (static input — 224² for base, 384² for so400m): every `expand` of the vision tower
 * keeps its input's shape and is an alias — four around each attention (the layers' and the
 * pooling head's) and the pooling head's query: base **53** (12 layers: 4 × 13 + 1), so400m
 * **113** (27 layers: 4 × 28 + 1). Nothing else: the tower normalises with `layer_norm` straight into a `linear` (adaLN wants the modulation chain
 * after the norm), and its `linear → gelu_tanh → linear` pairs would fold only into an i8a8 linear
 * (the linear prologue under `linearCompute: "a8"`), which no preset sets.
 */
const EXPECTED: Readonly<Record<string, number>> = { base: 53, so400m: 113 };

test.skipIf(!AVAILABLE)(
  "the real SigLIP 2 vision towers keep their identity expands (base 53, so400m 113) and grab no fusion",
  async () => {
    const text = await readTextFile(new URL("karume.json", MIRROR));
    const manifest: ContainerManifest = JSON.parse(text);
    const parsed = parseManifest(text);
    assertEquals(Object.keys(parsed.models), Object.keys(EXPECTED), "the models planned");
    for (const [model, entry] of Object.entries(parsed.models)) {
      for (const [preset, quant] of Object.entries(entry.quants)) {
        assertEquals(quant.session, {}, `${model} ${preset}: the preset sets no Session option`);
        const storage = quant.weights.vision;
        const graph = await readDistributionGraph(
          MIRROR,
          manifest,
          model,
          "vision",
          storage,
          NO_MIRRORS,
        );
        assertEquals(
          boundFusionCounts(graph, {}),
          { ...NONE, identityExpand: EXPECTED[model] },
          `${model} ${preset} vision/${storage}`,
        );
      }
    }
  },
);
