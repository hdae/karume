/**
 * Fusion hit counts of the MiniCPM5-1B decode series graph (no GPU — `planFusions` on part 0 only;
 * the shared planning side and the rationale live in `runtime/tests/helpers/fusion-counts.ts`).
 *
 * MUST: the asset is `outputs/series/minicpm5-1b-decode/` (untracked local asset); where it is
 * missing the case SKIPs explicitly with a reason (ADR 0005). MiniCPM5 has no distribution, so this
 * is an implementation-time series check (ADR 0139 decision 6 — KARUME_SERIES=1).
 */

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import type { FusionCounts } from "../../runtime/src/runtime/fusion.ts";
import { readContainerGraph } from "../../runtime/tests/helpers/container-graph.ts";
import { SERIES_CHECKS, SERIES_CHECKS_OPT_IN } from "../../models/tests/helpers/series-checks.ts";
import { decodeFusionCounts, exists, NONE, seriesSource } from "./helpers/fusion-counts.ts";

// A thunk: the series directory is not listed at all without KARUME_SERIES=1.
const minicpm5DecodeModel = () => seriesSource("minicpm5-1b-decode");

const MINICPM5_DECODE_AVAILABLE = SERIES_CHECKS && (await exists(minicpm5DecodeModel().part0));
if (SERIES_CHECKS && !MINICPM5_DECODE_AVAILABLE) {
  console.warn(
    `[karume] ${minicpm5DecodeModel().part0.pathname} が無いため MiniCPM5 decode の融合ヒット数を` +
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
  `実資産の MiniCPM5 decode は rope 48 / silu 24 を掴む（M 非依存）${SERIES_CHECKS_OPT_IN}`,
  async () => {
    const source = minicpm5DecodeModel();
    const graph = await readContainerGraph(source.part0, source.graph);
    const expected: FusionCounts = { ...NONE, rope: 48, silu: 24 };
    for (const rows of [1, 32]) {
      assertEquals(decodeFusionCounts(graph, rows), expected, `M=${rows}`);
    }
  },
);
