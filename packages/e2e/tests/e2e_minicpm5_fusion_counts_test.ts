/**
 * Fusion hit counts of the MiniCPM5-1B decode series graph (no GPU — `planFusions` on part 0 only;
 * the shared planning side and the rationale live in `runtime/tests/helpers/fusion-counts.ts`).
 *
 * MUST: the asset is `outputs/series/minicpm5-1b-decode/` (untracked local asset); where it is
 * missing the case SKIPs explicitly with a reason (ADR 0005).
 */

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import type { FusionCounts } from "../../runtime/src/runtime/fusion.ts";
import { readContainerGraph } from "../../runtime/tests/helpers/container-graph.ts";
import { decodeFusionCounts, exists, NONE, seriesSource } from "./helpers/fusion-counts.ts";

const MINICPM5_DECODE_MODEL = seriesSource("minicpm5-1b-decode");

const MINICPM5_DECODE_AVAILABLE = await exists(MINICPM5_DECODE_MODEL.part0);
if (!MINICPM5_DECODE_AVAILABLE) {
  console.warn(
    `[karume] ${MINICPM5_DECODE_MODEL.part0.pathname} が無いため MiniCPM5 decode の融合ヒット数を` +
      " SKIP する",
  );
}

/**
 * MiniCPM5-1B decode（24 層 × q/k = 48 鎖・KV 共有なし）。gemma4 と違い**全鎖が M 非依存で
 * 適合する** — 同じ表引き RoPE でもここまで割れる、という gemma4 側の対照実証でもある
 * （片方だけ動いたら、どちらの発行形が変わったのかがこの対で割れる）。silu 24 は 24 層の
 * gate MLP。
 */
test.skipIf(!MINICPM5_DECODE_AVAILABLE)(
  "実資産の MiniCPM5 decode は rope 48 / silu 24 を掴む（M 非依存）",
  async () => {
    const graph = await readContainerGraph(
      MINICPM5_DECODE_MODEL.part0,
      MINICPM5_DECODE_MODEL.graph,
    );
    const expected: FusionCounts = { ...NONE, rope: 48, silu: 24 };
    for (const rows of [1, 32]) {
      assertEquals(decodeFusionCounts(graph, rows), expected, `M=${rows}`);
    }
  },
);
