/**
 * Fusion hit counts of the Gemma 4 E2B QAT distribution graph (no GPU — `planFusions` on part 0
 * only; the shared planning side and the rationale live in `runtime/tests/helpers/fusion-counts.ts`).
 * The non-QAT Gemma 4 graphs have their own gate in `e2e_gemma4_fusion_counts_test.ts`.
 *
 * MUST: the asset is `models/karume-gemma4-qat/` (untracked local asset); where it is missing the
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

// 配布形も先頭part（part 0）をmanifestから引く。旧seriesとQATの両方で発行順を守る。
const family = "gemma4-qat";
const root = new URL(`../../../models/karume-${family}/`, import.meta.url);
const manifestUrl = new URL("karume.json", root);
const available = await exists(manifestUrl);
if (!available) {
  console.warn(`[karume] ${manifestUrl.pathname} が無いため配布形のRoPE検査をSKIPする`);
}
test.skipIf(!available)(
  `実配布 ${family} E2B は RoPE 50 と任意指定のRMS→add 106を掴む`,
  async () => {
    const manifest: ContainerManifest = JSON.parse(await readTextFile(manifestUrl));
    // MUST: quant は名指しする（下の SRQ の期待値は i2 / i4 の格納と group 長に依存する —
    // manifest の並びの先頭を採ると、quant を足した日に並び順で期待値が割れる）。
    const head = containerPart0(manifest, "e2b", "model", "i4");
    if (head.repo !== undefined) {
      throw new Error("Gemma実資産の融合テストは自己完結配布を要求する");
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
      // packed 活性（ADR 0105）は**素のまま残った SRQ** のうち、消費先が全て
      // `packedActivations: true` の実測形（ADR 0105 追記 1）に落ちるものだけを掴む。
      // QAT decode 計画（M ≤ 8）では 70 本 — 内訳は per-key 実測の本数そのもので、
      // i2 1536×12288 が 20 / i4 g2048 1536×{2048,6144} が 43 / i4 g4096 1536×4096 が 7。
      // 追記 1 の前は 210 本で、差の 140 本は実測で効かなかった形（i4 g512 の l4 / l32・
      // i8・i2 12288×1536 の l2）へ落ちる SRQ。linear→SRQ 融合が掴む 275 本とは重ならない
      // （あちらの出力側の消費先には linear 以外が混ざる）。prefill 計画（M ≥ 32）は
      // 並列 GEMV に落ちないので 0 になる。融合の有無でこの数は動かない。
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
