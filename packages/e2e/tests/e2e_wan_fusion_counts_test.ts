/**
 * Fusion hit counts of the Wan2.1 distribution graphs (no GPU — `planFusions` on part 0 only; the
 * shared planning side and the rationale live in `runtime/tests/helpers/fusion-counts.ts`).
 *
 * MUST: the asset is `models/karume-wan2.1/` (untracked local asset); where it is missing the case
 * SKIPs explicitly with a reason (ADR 0005).
 */

import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
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
 * The three Wan2.1 components. The counts are frozen from measurements (2026-10-03; rope from
 * 2026-10-24):
 *
 * - **transformer: silu 2 and rope 60**. The two `sigmoid`s are the time-embedding MLP's (the SiLU
 *   inside `time_embedder` and the one before `time_proj`); both are grabbed as SiLU.
 *   - rope 60: the RoPE of q and k in the 30 self-attention blocks, written as the complex
 *     rotation in real numbers (the **pair form** — swap of adjacent pairs + elementwise products
 *     with cos / sin; `recipe wan/dit_patch.py`). ROPE_RULE accepts it exactly as its second form
 *     (the pair rotation of `kernels/rope.ts`).
 *   - adaln 0: the modulation vectors are sliced **before** the layer_norm, so the 2–3 reshapes
 *     right after the layer_norm that ADALN_RULE's broadcast form requires are absent, and there is
 *     no `where` of the per-token form (perf-ledger K-73).
 *   - gatedResidual 0: the gate is a broadcast vector (no per-token `where`).
 *   - rowBlockAttention / identityExpand 0: attention is the fused op (`attention`, 60 of them),
 *     with no decomposed spelling.
 * - **The VAE chunk graphs (first / next): silu 29 + upsample2x 3**. All 29 `sigmoid`s are SiLU
 *   (the resnet activations and before the head), and upsample2x is the three spatial ×2 stages
 *   (nearest ×2 + conv2d). The temporal upsample (`time_conv` of next) is not the upsample2x form.
 *
 * The counts do not depend on S (the DiT is checked at S = 192 and the full size 14,040). If a
 * count moves, first check whether a rule's accepted set widened or narrowed unintentionally (the
 * same reading as the Irodori DiT gate).
 */
test.skipIf(!WAN_AVAILABLE)(
  "the real Wan2.1 grabs DiT silu 2 / rope 60 and the VAE chunk graphs' silu 29 / upsample2x 3 per run (adaln is a different spelling: 0)",
  async () => {
    const transformer = await readWanGraph("transformer");
    for (const sequence of [192, 14040]) {
      assertEquals(
        fusionCounts(transformer, wanShapes(transformer, sequence)),
        { ...NONE, silu: 2, rope: 60 },
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
);
