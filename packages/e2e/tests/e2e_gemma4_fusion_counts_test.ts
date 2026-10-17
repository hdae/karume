/**
 * Fusion hit counts of the Gemma 4 E2B decode graphs (no GPU — `planFusions` on part 0 only; the
 * shared planning side and the rationale live in `runtime/tests/helpers/fusion-counts.ts`). The QAT
 * distribution has its own gate in `e2e_gemma4_qat_fusion_counts_test.ts` (the gemma4-qat lane).
 *
 * MUST: the assets are `outputs/series/gemma4-e2b-decode{,-token}/` and `models/karume-gemma4/`
 * (untracked local assets); where they are missing the cases SKIP explicitly with a reason
 * (ADR 0005).
 */

import { readTextFile } from "@karume/test-support/fs";
import { test } from "@karume/test-support/test";
import { assertEquals } from "@std/assert";
import {
  type ContainerManifest,
  containerPart0,
  readContainerGraph,
} from "../../runtime/tests/helpers/container-graph.ts";
import { decodeFusionCounts, exists, NONE, seriesSource } from "./helpers/fusion-counts.ts";

const GEMMA4_DECODE_MODEL = seriesSource("gemma4-e2b-decode");
const GEMMA4_TOKEN_MODEL = seriesSource("gemma4-e2b-decode-token");

const GEMMA4_DECODE_AVAILABLE =
  (await exists(GEMMA4_DECODE_MODEL.part0)) && (await exists(GEMMA4_TOKEN_MODEL.part0));
if (!GEMMA4_DECODE_AVAILABLE) {
  console.warn(
    `[karume] ${GEMMA4_DECODE_MODEL.part0.pathname} と ${GEMMA4_TOKEN_MODEL.part0.pathname} が揃っていない` +
      "ため Gemma 4 decode の融合ヒット数を SKIP する",
  );
}

/**
 * Gemma 4 E2B は BSHD / table B S 1 D の50鎖（q=35、所有k=15）。
 * 旧BHSD matcherではM=1・H=1のkだけが一致し、decode=15 / prefill=0だった。
 * BSHDの専用添字を追加したため、両方で全50鎖を固定する（ADR 0040追記2026-09-12）。
 */
test({
  name: "実資産の Gemma 4 E2B は decode / prefill とも rope 50 を掴む（token-only 形も同一）",
  ignore: !GEMMA4_DECODE_AVAILABLE,
  fn: async () => {
    for (const [name, source] of [
      ["logits opt-in", GEMMA4_DECODE_MODEL],
      ["token-only", GEMMA4_TOKEN_MODEL],
    ] as const) {
      const graph = await readContainerGraph(source.part0, source.graph);
      assertEquals(decodeFusionCounts(graph, 1), { ...NONE, rope: 50 }, `${name} decode（M=1）`);
      assertEquals(
        decodeFusionCounts(graph, 32),
        { ...NONE, rope: 50 },
        `${name} prefill 形（M=32）`,
      );
    }
  },
});

// 配布形も先頭part（part 0）をmanifestから引く。旧seriesとQATの両方で発行順を守る。
const family = "gemma4";
const root = new URL(`../../../models/karume-${family}/`, import.meta.url);
const manifestUrl = new URL("karume.json", root);
const available = await exists(manifestUrl);
if (!available) {
  console.warn(`[karume] ${manifestUrl.pathname} が無いため配布形のRoPE検査をSKIPする`);
}
test({
  name: `実配布 ${family} E2B は RoPE 50 と任意指定のRMS→add 106を掴む`,
  ignore: !available,
  fn: async () => {
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
});
