/**
 * Fusion hit counts of the umT5-XXL text encoder graph (no GPU — `planFusions` on part 0 only; the
 * shared planning side and the rationale live in `./helpers/fusion-counts.ts`).
 *
 * MUST: the asset is the distribution mirror `models/karume-umt5-xxl/` (untracked local asset);
 * where it is missing the case SKIPs explicitly with a reason (ADR 0005).
 */

import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { parseManifest } from "@karume/hub";
import type { ContainerManifest } from "../../runtime/tests/helpers/container-graph.ts";
import { mirrorPresent, mirrorRoot } from "../../models/tests/helpers/mirror.ts";
import { boundFusionCounts, NONE, readDistributionGraph } from "./helpers/fusion-counts.ts";

const MIRROR = mirrorRoot("karume-umt5-xxl");
const AVAILABLE = mirrorPresent(MIRROR);
if (!AVAILABLE) {
  console.warn(
    `[karume] ${MIRROR.pathname} has no karume.json, so the umT5 fusion hit counts SKIP` +
      " (build it with dist.py --pipeline umt5 in tools/export-recipes)",
  );
}

/** The umT5 distribution is self-contained: a cross-repo reference fails loudly. */
const NO_MIRRORS: ReadonlyMap<string, URL> = new Map();

/**
 * The encoder's 24 layers, at two token lengths (the counts do not depend on `L`):
 *
 * - **identityExpand 96**: every one of its 96 `expand` nodes (4 per layer, around the attention's
 *   batched products) keeps its input's shape and is an alias.
 * - Nothing else. Its norm (`rms_norm`) comes after the residual `add` (`add → rms_norm → linear`),
 *   while RMS→add takes a norm followed by the add (and only at the measured widths — the model's
 *   4,096 is not one); the gated GELU (`gelu_tanh → mul`) feeds no static quantize, so the
 *   gelu_tanh → mul → SRQ fold does not apply.
 */
test.skipIf(!AVAILABLE)(
  "the real umT5-XXL text encoder keeps its 96 identity expands and grabs no fusion",
  async () => {
    const text = await readTextFile(new URL("karume.json", MIRROR));
    const manifest: ContainerManifest = JSON.parse(text);
    const parsed = parseManifest(text);
    assertEquals(Object.keys(parsed.models), ["xxl"], "the models planned");
    const quants = parsed.models.xxl.quants;
    assertEquals(Object.keys(quants), ["i8"], "the presets planned");
    assertEquals(quants.i8.session, {}, "the preset sets no Session option");
    const graph = await readDistributionGraph(
      MIRROR,
      manifest,
      "xxl",
      "text_encoder",
      quants.i8.weights.text_encoder,
      NO_MIRRORS,
    );
    for (const length of [64, 512]) {
      assertEquals(
        boundFusionCounts(graph, { L: length }),
        { ...NONE, identityExpand: 96 },
        `L=${length}`,
      );
    }
  },
);
