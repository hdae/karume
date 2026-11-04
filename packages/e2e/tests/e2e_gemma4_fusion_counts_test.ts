/**
 * Fusion hit counts of the Gemma 4 E2B decode graphs (no GPU — `planFusions` on part 0 only; the
 * shared planning side and the rationale live in `runtime/tests/helpers/fusion-counts.ts`). The QAT
 * distribution has its own gate in `e2e_gemma4_qat_mobile_fusion_counts_test.ts` (the gemma4-qat-mobile lane).
 *
 * MUST: the assets are `outputs/series/gemma4-e2b-decode/` and `models/karume-gemma4/`
 * (untracked local assets); where they are missing the cases SKIP explicitly with a reason
 * (ADR 0005). The decode series graph is not shipped, so its case is an implementation-time series
 * check (ADR 0139 decision 6 — KARUME_SERIES=1); the distribution case runs in the default lane.
 */

import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import {
  type ContainerManifest,
  containerPart0,
  readContainerGraph,
} from "../../runtime/tests/helpers/container-graph.ts";
import { SERIES_CHECKS, SERIES_CHECKS_OPT_IN } from "../../models/tests/helpers/series-checks.ts";
import { decodeFusionCounts, exists, NONE, seriesSource } from "./helpers/fusion-counts.ts";

// A thunk: the series directory is not listed at all without KARUME_SERIES=1.
const gemma4DecodeModel = () => seriesSource("gemma4-e2b-decode");

const GEMMA4_DECODE_AVAILABLE = SERIES_CHECKS && (await exists(gemma4DecodeModel().part0));
if (SERIES_CHECKS && !GEMMA4_DECODE_AVAILABLE) {
  console.warn(
    `[karume] ${gemma4DecodeModel().part0.pathname} is missing, so the Gemma 4 decode fusion hit counts are SKIPPED`,
  );
}

/**
 * Gemma 4 E2B has 50 chains of BSHD / table B S 1 D (q = 35, owned k = 15).
 * The old BHSD matcher matched only the k with M = 1 and H = 1, giving decode = 15 / prefill = 0.
 * Dedicated BSHD indexing was added, so all 50 chains are pinned in both (ADR 0040 addendum
 * 2026-09-12).
 */
test.skipIf(!GEMMA4_DECODE_AVAILABLE)(
  `the real Gemma 4 E2B asset matches rope 50 in both decode and prefill${SERIES_CHECKS_OPT_IN}`,
  async () => {
    const source = gemma4DecodeModel();
    const graph = await readContainerGraph(source.part0, source.graph);
    assertEquals(
      decodeFusionCounts(graph, 1),
      { ...NONE, rope: 50 },
      "logits opt-in decode（M=1）",
    );
    assertEquals(
      decodeFusionCounts(graph, 32),
      { ...NONE, rope: 50 },
      "logits opt-in prefill shape (M=32)",
    );
  },
);

// The distribution also takes its first part (part 0) from the manifest. Both the old series and
// QAT keep the emission order.
const family = "gemma4";
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
    // MUST: the storage is named — the expectations below are one container's graph (the SRQ ones
    // depend on the storage), and taking the manifest's first container would split them by order
    // the day a storage is added.
    const head = containerPart0(manifest, "e2b", "model", "bf16");
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
      // The non-QAT distribution carries no static-quantize nodes, so neither the linear→SRQ
      // fusion nor packed activations (ADR 0105) find anything at any M.
      assertEquals(
        decodeFusionCounts(graph, rows, true, true),
        { ...NONE, rope: 50, rmsNormAdd: 106, linearStaticQuantize: 0 },
        `${family} linear SRQ M=${rows}`,
      );
      for (const fuseSrq of [false, true]) {
        assertEquals(
          decodeFusionCounts(graph, rows, true, fuseSrq, true),
          { ...NONE, rope: 50, rmsNormAdd: 106, linearStaticQuantize: 0, packedStaticQuantize: 0 },
          `${family} packed SRQ M=${rows} fuse=${fuseSrq}`,
        );
      }
    }
  },
);
