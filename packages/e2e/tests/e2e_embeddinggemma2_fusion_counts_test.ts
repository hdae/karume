/**
 * Fusion hit counts of the EmbeddingGemma 2 graphs — the text model and the vision and audio towers
 * (no GPU — `planFusions` on part 0 only; the shared planning side and the rationale live in
 * `./helpers/fusion-counts.ts`).
 *
 * MUST: the asset is the distribution mirror `models/karume-embeddinggemma-2/` (untracked local
 * asset); where it is missing the case SKIPs explicitly with a reason (ADR 0005).
 */

import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { parseManifest } from "@karume/hub";
import type { FusionCounts } from "../../runtime/src/runtime/fusion.ts";
import type { ContainerManifest } from "../../runtime/tests/helpers/container-graph.ts";
import { mirrorPresent, mirrorRoot } from "../../models/tests/helpers/mirror.ts";
import {
  boundFusionCounts,
  NONE,
  readDistributionGraph,
  type SymbolBinding,
} from "./helpers/fusion-counts.ts";

const MIRROR = mirrorRoot("karume-embeddinggemma-2");
const AVAILABLE = mirrorPresent(MIRROR);
if (!AVAILABLE) {
  console.warn(
    `[karume] ${MIRROR.pathname} has no karume.json, so the EmbeddingGemma 2 fusion hit counts SKIP` +
      " (build it with dist.py --pipeline embeddinggemma-2 in tools/export-recipes)",
  );
}

/** The distribution is self-contained: a cross-repo reference fails loudly. */
const NO_MIRRORS: ReadonlyMap<string, URL> = new Map();

/**
 * Per graph: two bindings of its symbol (the counts do not depend on it) and its hits, the same on
 * both presets (`bf16`, `i8` — storage only; the three Sessions get the same, empty options).
 *
 * - **model** (the text model, 24 layers): **rope 48** — q and k of every layer.
 * - **vision** (16 layers, `9n` patches): **rope 64** — four per layer (q and k, each on the two
 *   position axes).
 * - **audio** (12 conformer layers, `4n` frames): **silu 36** — three per layer. Its other 12
 *   sigmoids multiply another tensor (the convolution module's gated linear unit), so they are not
 *   SiLU.
 * - RMS→add 0 on all three: they have `rms_norm → add` pairs (72 / 32 / 12), but at the widths 512,
 *   768 and 1,024, which are not among the rule's measured widths (256, 1,536, 2,560, 2,816,
 *   3,840).
 */
const GRAPHS: readonly (readonly [string, readonly SymbolBinding[], Partial<FusionCounts>])[] = [
  ["model", [{ T: 128 }, { T: 2048 }], { rope: 48 }],
  ["vision", [{ n: 16 }, { n: 64 }], { rope: 64 }],
  ["audio", [{ n: 16 }, { n: 64 }], { silu: 36 }],
];

test.skipIf(!AVAILABLE)(
  "the real EmbeddingGemma 2 graphs grab rope 48 (text), rope 64 (vision) and silu 36 (audio) on both presets",
  async () => {
    const text = await readTextFile(new URL("karume.json", MIRROR));
    const manifest: ContainerManifest = JSON.parse(text);
    const parsed = parseManifest(text);
    assertEquals(Object.keys(parsed.models), ["text"], "the models planned");
    const entry = parsed.models.text;
    assertEquals(
      Object.keys(entry.weights),
      GRAPHS.map(([component]) => component),
      "the graphs planned",
    );
    for (const [preset, quant] of Object.entries(entry.quants)) {
      assertEquals(quant.session, {}, `${preset}: the preset sets no Session option`);
      for (const [component, bindings, expected] of GRAPHS) {
        const storage = quant.weights[component];
        const graph = await readDistributionGraph(
          MIRROR,
          manifest,
          "text",
          component,
          storage,
          NO_MIRRORS,
        );
        for (const binding of bindings) {
          assertEquals(
            boundFusionCounts(graph, binding),
            { ...NONE, ...expected },
            `${preset} ${component}/${storage} ${JSON.stringify(binding)}`,
          );
        }
      }
    }
  },
);
