/**
 * Fusion hit counts of the Wan2.1 distribution graphs (no GPU — `planFusions` on part 0 only; the
 * shared planning side and the rationale live in `runtime/tests/helpers/fusion-counts.ts`).
 *
 * MUST: the asset is `models/karume-wan2.1/` (untracked local asset); where it is missing the case
 * SKIPs explicitly with a reason (ADR 0005).
 */

import { readTextFile } from "@karume/test-support/fs";
import { test } from "@karume/test-support/test";
import { assertEquals } from "@std/assert";
import type { IrGraph } from "../../runtime/src/format/ir.ts";
import type { ContainerManifest } from "../../runtime/tests/helpers/container-graph.ts";
import { exists, fusionCounts, NONE, readDistributionGraph } from "./helpers/fusion-counts.ts";

/**
 * Wan2.1 T2V 1.3B の配布形（`models/karume-wan2.1/` — ADR 0118 段 7・HF へは未公開）。DiT は S 形 1 本、
 * VAE は chunk グラフ 2 本（タイル 32 の静的形）。
 */
const WAN_DIR = new URL("../../../models/karume-wan2.1/", import.meta.url);
const WAN_AVAILABLE = await exists(new URL("karume.json", WAN_DIR));
if (!WAN_AVAILABLE) {
  console.warn(`[karume] ${WAN_DIR.pathname} が無いため Wan2.1 の融合ヒット数を SKIP する`);
}

/**
 * The Wan2.1 distribution is self-contained, so no cross-repo mirror is known: a cross-repo
 * reference in its manifest fails loudly instead of being read from somewhere else.
 */
const NO_MIRRORS: ReadonlyMap<string, URL> = new Map();

/** 既定モデル（`t2v-1.3b`）の部品のグラフ（f16 席だけの配布形）。 */
const readWanGraph = async (component: string): Promise<IrGraph> => {
  const manifest: ContainerManifest = JSON.parse(
    await readTextFile(new URL("karume.json", WAN_DIR)),
  );
  return await readDistributionGraph(
    WAN_DIR,
    manifest,
    manifest.defaultModel,
    component,
    "f16",
    NO_MIRRORS,
  );
};

/** 入力の記号次元（DiT の `S`）を `sequence` で解いた shape（VAE は全部静的なので素通し）。 */
const wanShapes = (graph: IrGraph, sequence: number): Readonly<Record<string, readonly number[]>> =>
  Object.fromEntries(
    graph.inputs.map((spec) => [
      spec.name,
      spec.shape.map((dim) => (typeof dim === "number" ? dim : sequence)),
    ]),
  );

/**
 * Wan2.1 の 3 部品。実測（2026-10-03）の値を凍結する:
 *
 * - **transformer: silu 2 だけ**。`sigmoid` 2 本は時刻埋め込みの MLP（`time_embedder` の SiLU と
 *   `time_proj` の前の SiLU）で、両方 SiLU として掴む。
 *   - rope 0: RoPE は複素形を「隣接対の入れ替え + cos / sin の要素積」に実数化した**対の形**
 *     （`recipe wan/dit_patch.py`）で、ROPE_RULE が受理する `[1,H,S,D]` の半割り形ではない。
 *   - adaln 0: 変調ベクトルを layer_norm の**前**に切り出す並びで、ADALN_RULE が要求する layer_norm
 *     直後の reshape 2〜3 本が無い（perf-ledger K-73）。
 *   - rowBlockAttention / identityExpand 0: attention は融合 op（`attention` 60 本）で、分解の綴りを
 *     持たない。
 * - **VAE の chunk グラフ（first / next）: silu 29 + upsample2x 3**。`sigmoid` 29 本は全て SiLU
 *   （resnet の活性と head の前）で、upsample2x は空間 ×2 の 3 段（nearest ×2 + conv2d）。時間方向の
 *   upsample（next の `time_conv`）は upsample2x の形ではない。
 *
 * ヒット数は S に依存しない（DiT は S = 192 と実寸 14,040 の 2 点）。0 が非 0 に変わったら、まず
 * ROPE_RULE / ADALN_RULE の受理集合が意図せず広がっていないかを見る（Irodori DiT の門と同じ読み方）。
 */
test({
  name: "実資産の Wan2.1 は run 1 回で DiT silu 2・VAE の chunk グラフ silu 29 / upsample2x 3 を掴む（rope / adaln は綴りが違って 0）",
  ignore: !WAN_AVAILABLE,
  fn: async () => {
    const transformer = await readWanGraph("transformer");
    for (const sequence of [192, 14040]) {
      assertEquals(
        fusionCounts(transformer, wanShapes(transformer, sequence)),
        { ...NONE, silu: 2 },
        `transformer S=${sequence}`,
      );
    }
    for (const name of ["vae_decoder_first", "vae_decoder_next"]) {
      const graph = await readWanGraph(name);
      assertEquals(
        fusionCounts(graph, wanShapes(graph, 0)),
        { ...NONE, silu: 29, upsample2x: 3 },
        name,
      );
    }
  },
});
