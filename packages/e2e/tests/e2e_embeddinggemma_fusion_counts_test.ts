/**
 * Fusion hit counts of the EmbeddingGemma-300m series graph (no GPU — `planFusions` on part 0 only;
 * the shared planning side and the rationale live in `runtime/tests/helpers/fusion-counts.ts`).
 *
 * MUST: the asset is `outputs/series/embeddinggemma-300m/` (untracked local asset); where it is
 * missing the case SKIPs explicitly with a reason (ADR 0005). EmbeddingGemma has no published
 * distribution yet, so this is an implementation-time series check (ADR 0139 decision 6 —
 * KARUME_SERIES=1).
 *
 * NOTE: EmbeddingGemma gets a published distribution (ADR 0139 Rulings D2, staging step 7); this
 * case then reads part 0 from `models/<repo>/` through the manifest and returns to the default lane.
 */

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import type { FusionCounts } from "../../runtime/src/runtime/fusion.ts";
import { readContainerGraph } from "../../runtime/tests/helpers/container-graph.ts";
import { SERIES_CHECKS, SERIES_CHECKS_OPT_IN } from "../../models/tests/helpers/series-checks.ts";
import { exists, fusionCounts, NONE, seriesSource } from "./helpers/fusion-counts.ts";

// A thunk: the series directory is not listed at all without KARUME_SERIES=1.
const gemmaModel = () => seriesSource("embeddinggemma-300m");

const GEMMA_AVAILABLE = SERIES_CHECKS && (await exists(gemmaModel().part0));
if (SERIES_CHECKS && !GEMMA_AVAILABLE) {
  console.warn(
    `[karume] ${gemmaModel().part0.pathname} が無いため EmbeddingGemma の融合ヒット数を SKIP する`,
  );
}

/**
 * EmbeddingGemma-300m（24 層 × q / k の 2 本 = 48 鎖）。Anima と違い head 幅は 256 で、
 * cos / sin 表は θ 系統（local 1e4 / global 1e6）ごとに `sym_prefix_slice` で T へ縮められる。
 * その初出 2 箇所は鎖の隙間に落ちるので、窓内 passthrough を跨げないと 48 のうち 2 本が
 * 黙って外れる — **ヒット数 48 はそこまで含めた門**。
 *
 * identityExpand は **0**。SDPA を保存した資産（ADR 0023 改訂 — mask 付き attention）では
 * 分解由来の恒等 expand が IR ごと消える（決定 6 と同じ機序。分解資産の頃は 96 だった）。
 */
test.skipIf(!GEMMA_AVAILABLE)(
  `実資産の EmbeddingGemma は rope 48（head 幅 256・窓内 passthrough 込み）を掴む${SERIES_CHECKS_OPT_IN}`,
  async () => {
    const source = gemmaModel();
    const graph = await readContainerGraph(source.part0, source.graph);
    const expected: FusionCounts = { ...NONE, rope: 48 };
    // ヒット数は T に依存しない（Tmax = 512 の内側で 2 点）。
    for (const sequence of [12, 318]) {
      assertEquals(
        fusionCounts(graph, { input_ids: [1, sequence], pool_mask: [1, sequence] }),
        expected,
        `T=${sequence}`,
      );
    }
  },
);
