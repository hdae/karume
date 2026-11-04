/**
 * Fusion hit counts of the Gemma 4 E2B QAT distribution graph (no GPU — `planFusions` on part 0
 * only; the shared planning side and the rationale live in `runtime/tests/helpers/fusion-counts.ts`).
 * The non-QAT Gemma 4 graphs have their own gate in `e2e_gemma4_fusion_counts_test.ts`.
 *
 * MUST: the asset is `models/karume-gemma4-qat-mobile/` (untracked local asset); where it is missing the
 * case SKIPs explicitly with a reason (ADR 0005).
 */

import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import {
  type ContainerManifest,
  containerPart0,
  readContainerGraph,
} from "../../runtime/tests/helpers/container-graph.ts";
import { decodeFusionCounts, exists, NONE } from "./helpers/fusion-counts.ts";

// The distribution also takes its first part (part 0) from the manifest. Both the old series and
// QAT keep the emission order.
const family = "gemma4-qat-mobile";
const root = new URL(`../../../models/karume-${family}/`, import.meta.url);
const manifestUrl = new URL("karume.json", root);
const available = await exists(manifestUrl);
if (!available) {
  console.warn(
    `[karume] ${manifestUrl.pathname} is missing, so the distribution's RoPE check is SKIPPED`,
  );
}
test.skipIf(!available)(
  `the real ${family} E2B distribution matches RoPE 50 and the opt-in RMS→add 106`,
  async () => {
    const manifest: ContainerManifest = JSON.parse(await readTextFile(manifestUrl));
    // MUST: the quant is named (the SRQ expectations below depend on the i2 / i4 storage and the
    // group length — taking the first entry of the manifest would split the expectations by order
    // the day a quant is added).
    const head = containerPart0(manifest, "e2b", "model", "i4");
    if (head.repo !== undefined) {
      throw new Error("the Gemma real-asset fusion test requires a self-contained distribution");
    }
    const graph = await readContainerGraph(new URL(head.path, root), "model");
    for (const rows of [1, 4, 8, 32, 40, 64]) {
      assertEquals(decodeFusionCounts(graph, rows), { ...NONE, rope: 50 }, `${family} M=${rows}`);
      assertEquals(
        decodeFusionCounts(graph, rows, true),
        { ...NONE, rope: 50, rmsNormAdd: 106 },
        `${family} fused M=${rows}`,
      );
      assertEquals(
        decodeFusionCounts(graph, rows, true, true),
        {
          ...NONE,
          rope: 50,
          rmsNormAdd: 106,
          linearStaticQuantize: rows <= 8 ? 275 : 0,
        },
        `${family} linear SRQ M=${rows}`,
      );
      // Packed activations (ADR 0105) match only those **SRQs left standalone** whose consumers
      // all land on a measured shape with `packedActivations: true` (ADR 0105 addendum 1).
      // A QAT decode plan (M ≤ 8) has 70 — the breakdown is exactly the per-key measured counts:
      // 20 for i2 1536×12288 / 43 for i4 g2048 1536×{2048,6144} / 7 for i4 g4096 1536×4096.
      // Before addendum 1 it was 210; the 140 difference are SRQs that land on shapes where the
      // measurement showed no gain (l4 / l32 of i4 g512, i8, l2 of i2 12288×1536). They do not
      // overlap the 275 the linear→SRQ fusion matches (the consumers on that output side include
      // ops other than linear). A prefill plan (M ≥ 32) does not take the parallel GEMV, so it is
      // 0. Whether the fusion is on does not move this count.
      for (const fuseSrq of [false, true]) {
        assertEquals(
          decodeFusionCounts(graph, rows, true, fuseSrq, true),
          {
            ...NONE,
            rope: 50,
            rmsNormAdd: 106,
            linearStaticQuantize: fuseSrq && rows <= 8 ? 275 : 0,
            packedStaticQuantize: rows <= 8 ? 70 : 0,
          },
          `${family} packed SRQ M=${rows} fuse=${fuseSrq}`,
        );
      }
      // The i4-fast set plus the gelu_tanh → mul → SRQ fold (perf-ledger K-48 step 1b): one
      // per-layer-input gate per layer (35) in a decode or verification plan (M ≤ 8, the measured
      // form); a prefill plan keeps the plain chain. No other counter moves: the folded SRQs feed
      // per_layer_projection (i8 1536×256), which never takes packed activations.
      assertEquals(
        decodeFusionCounts(graph, rows, true, true, true, true),
        {
          ...NONE,
          rope: 50,
          rmsNormAdd: 106,
          linearStaticQuantize: rows <= 8 ? 275 : 0,
          packedStaticQuantize: rows <= 8 ? 70 : 0,
          geluMulStaticQuantize: rows <= 8 ? 35 : 0,
        },
        `${family} gelu mul SRQ M=${rows}`,
      );
    }
  },
);
