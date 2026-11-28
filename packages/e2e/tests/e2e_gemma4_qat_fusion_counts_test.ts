/**
 * Fusion hit counts of the q4_0 QAT Gemma 4 distribution graphs — E2B, E4B and 12B with their
 * drafters, 12B's image embedder and 26B A4B (no GPU — `planFusions` on part 0 only; the shared
 * planning side and the rationale live in `./helpers/fusion-counts.ts`). The regular Gemma 4 graphs
 * have their own gate in `e2e_gemma4_fusion_counts_test.ts`, the mobile QAT line its own in
 * `e2e_gemma4_qat_mobile_fusion_counts_test.ts`.
 *
 * MUST: the asset is the distribution mirror `models/karume-gemma4-qat/` (untracked local asset);
 * where it is missing the case SKIPs explicitly with a reason (ADR 0005).
 */

import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { parseManifest } from "@karume/hub";
import { resolveGemmaSessionOptions } from "../../models/src/gemma/session-options.ts";
import type { FusionCounts } from "../../runtime/src/runtime/fusion.ts";
import type { ContainerManifest } from "../../runtime/tests/helpers/container-graph.ts";
import { GEMMA4_QAT_MIRROR } from "../../models/tests/helpers/gemma-mirror.ts";
import { mirrorPresent } from "../../models/tests/helpers/mirror.ts";
import {
  boundFusionCounts,
  decodeFusionCounts,
  NONE,
  readDistributionGraph,
} from "./helpers/fusion-counts.ts";

const AVAILABLE = mirrorPresent(GEMMA4_QAT_MIRROR);
if (!AVAILABLE) {
  console.warn(
    `[karume] ${GEMMA4_QAT_MIRROR.pathname} has no karume.json, so the q4_0 QAT fusion hit counts SKIP` +
      " (get the q4_0 QAT distribution mirror)",
  );
}

/** The q4_0 QAT distribution is self-contained: a cross-repo reference fails loudly. */
const NO_MIRRORS: ReadonlyMap<string, URL> = new Map();

/**
 * Per (model, component, storage): the hits of the graph, the same at every M and on both presets
 * (`i4` — the defaults; `i4-fast` — the level `reorder`, which takes the parallel GEMV).
 *
 * - **RoPE**: q of every layer and k of every layer that owns its cache — E2B 35 + 15, E4B 42 + 24,
 *   12B 48 + 48, 26B 30 + 30; each drafter graph 12.
 * - **RMS→add**: the rule accepts only the measured last dimensions (256, 1,536, 2,560, 2,816 and
 *   3,840). 12B's drafter has 24 `rms_norm → add` pairs, at the width 1,024 of its blocks, so it
 *   takes none (E2B's and E4B's drafters have theirs at 256: 24).
 * - The q4_0 QAT graphs carry no static-quantize node, so the linear→SRQ fusion and the gelu_tanh →
 *   mul → SRQ fold find nothing, the parallel GEMV of `i4-fast` included.
 * - 12B's image embedder (10 nodes: three `layer_norm`, two `linear`, two `embedding`, two `add`
 *   and a final `rms_norm`) holds no chain a rule takes. Its Session gets none of the preset's
 *   options.
 */
const DECODE_FUSIONS = [
  ["e2b", "model", "i4", { rope: 50, rmsNormAdd: 106 }],
  ["e2b", "drafter", "i4", { rope: 12, rmsNormAdd: 24 }],
  ["e4b", "model", "i4", { rope: 66, rmsNormAdd: 127 }],
  ["e4b", "drafter", "i4", { rope: 12, rmsNormAdd: 24 }],
  ["12b", "model", "i4", { rope: 96, rmsNormAdd: 96 }],
  ["12b", "drafter", "i4", { rope: 12 }],
  ["26b", "model", "i4", { rope: 60, rmsNormAdd: 90 }],
] as const satisfies readonly (readonly [string, string, string, Partial<FusionCounts>])[];

test.skipIf(!AVAILABLE)(
  "the real q4_0 QAT graphs plan the same fusions at i4 and i4-fast (E2B, E4B and 12B with their drafters, 26B; nothing in 12B's image embedder)",
  async () => {
    const text = await readTextFile(new URL("karume.json", GEMMA4_QAT_MIRROR));
    const manifest: ContainerManifest = JSON.parse(text);
    const parsed = parseManifest(text);
    // A model or component the table does not plan fails here instead of going unchecked.
    assertEquals(
      Object.entries(parsed.models).map(([model, entry]) => [model, Object.keys(entry.weights)]),
      [
        ["e2b", ["model", "drafter"]],
        ["e4b", ["model", "drafter"]],
        ["12b", ["model", "drafter", "vision"]],
        ["26b", ["model"]],
      ],
      "the graphs planned",
    );
    const read = (model: string, component: string, storage: string) =>
      readDistributionGraph(GEMMA4_QAT_MIRROR, manifest, model, component, storage, NO_MIRRORS);
    for (const [model, component, storage, expected] of DECODE_FUSIONS) {
      const graph = await read(model, component, storage);
      const quants = parsed.models[model].quants;
      assertEquals(Object.keys(quants), ["i4", "i4-fast"], `${model}: the presets planned`);
      for (const [preset, quant] of Object.entries(quants)) {
        assertEquals(quant.weights[component], storage, `${model} ${preset}: ${component}`);
        const options = resolveGemmaSessionOptions(quant.session, {}, `${model} ${preset}`);
        for (const rows of [1, 4, 8, 32]) {
          assertEquals(
            decodeFusionCounts(graph, rows, options),
            { ...NONE, ...expected },
            `${model} ${component} ${preset} M=${rows}`,
          );
        }
      }
    }
    const vision = await read("12b", "vision", "bf16");
    for (const patches of [256, 1024]) {
      assertEquals(boundFusionCounts(vision, { N: patches }), NONE, `12b vision N=${patches}`);
    }
  },
);
