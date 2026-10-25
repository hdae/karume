/**
 * Fusion hit counts of the Irodori distribution graphs (no GPU — `planFusions` on part 0 only; the
 * shared planning side and the rationale live in `runtime/tests/helpers/fusion-counts.ts`).
 *
 * MUST: the asset is `models/karume-irodori-v4-small/` (untracked local asset); where it is missing
 * the cases SKIP explicitly with a reason (ADR 0005).
 */

import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import type { IrGraph } from "../../runtime/src/format/ir.ts";
import type { FusionCounts } from "../../runtime/src/runtime/fusion.ts";
import {
  type ContainerManifest,
  containerPart0,
  readContainerGraph,
} from "../../runtime/tests/helpers/container-graph.ts";
import { exists, fusionCounts, NONE } from "./helpers/fusion-counts.ts";

const IRODORI_DIR = new URL("../../../models/karume-irodori-v4-small/", import.meta.url);

/**
 * 融合ヒット数は**ノード列だけ**で決まる（格納 dtype はどの initializer をどう読むかしか
 * 変えない）ので、系列が増えても f32 の 1 本を見れば足りる — 見ているのはエクスポータの
 * ノード発行順で、それは系列を跨いで同じ 1 回の変換から出る。
 */
const readIrodoriGraph = async (name: string): Promise<IrGraph> => {
  const manifest: ContainerManifest = JSON.parse(
    await readTextFile(new URL("karume.json", IRODORI_DIR)),
  );
  const head = containerPart0(manifest, "v4-small", name, "f32");
  return await readContainerGraph(new URL(head.path, IRODORI_DIR), name);
};

// 見るのは**ファイル**（ディレクトリではない）— 配布形の綴りが動いたときに、空でない
// ディレクトリだけを見て「資産あり」と判断すると、読めない path で FAIL する形になる。
const IRODORI_AVAILABLE = await exists(new URL("karume.json", IRODORI_DIR));
if (!IRODORI_AVAILABLE) {
  console.warn(`[karume] ${IRODORI_DIR.pathname} が無いため Irodori の融合ヒット数を SKIP する`);
}

/** DiT の条件側 K/V 射影 `dit_context` の入力 shape（条件 state 3 本 — Tmax 右 pad 済み・静的）。 */
const IRODORI_CONTEXT_SHAPES: Readonly<Record<string, readonly number[]>> = {
  text_state: [1, 256, 512],
  speaker_state: [1, 751, 768],
  caption_state: [1, 512, 512],
};

/**
 * DiT の入力 shape（S = latent フレーム数。mask は `S+1519` の派生次元）。条件側は `dit_context` の
 * 出力 24 本（12 ブロック × K / V・各 `[1,1519,20,64]` — ADR 0114）を同名の入力で受ける。
 */
const irodoriDitShapes = (sequence: number): Readonly<Record<string, readonly number[]>> => ({
  x_t: [1, sequence, 32],
  t_embed: [1, 512],
  mask: [1, 1, 1, sequence + 1519],
  ...Object.fromEntries(
    Array.from({ length: 12 }, (_, block) => block).flatMap((block) => [
      [`context_k_${block}`, [1, 1519, 20, 64]],
      [`context_v_${block}`, [1, 1519, 20, 64]],
    ]),
  ),
});

/**
 * The Irodori DiT (12 blocks). The record of what the matchers' accepted sets cover in this
 * spelling:
 *
 * - **rope 23**: the DiT's RoPE is the **pair form** (view as `[1,S,H·D/2,2]`, slice / neg / cat
 *   on the last axis, reshape back — on the first 10 of the 20 heads), the same spelling as Wan's
 *   DiT, which ROPE_RULE accepts exactly as its second form (2026-10-24). 12 blocks × q / k = 24
 *   sites; the first use of the shared cos / sin tables puts their two `sym_prefix_slice`s inside
 *   the window, which the pair form does not accept, so that one site stays plain (23 = 24 − 1).
 *   The fused kernel performs the same f32 operations in the same order (the WAV sha256 rows of
 *   `e2e_irodori_wav_test.ts` held on the RTX 3080 Ti with it — 2026-10-24).
 * - adaln 0: ADALN_RULE's leading op is `layer_norm`, but the DiT normalises with `rms_norm` (87)
 *   and has no `layer_norm` at all.
 * - silu 17: the timestep MLP's 5 + one per block (12). The remaining 12 sigmoids are the gates of
 *   `mul(v, sigmoid(u))` (not applied to themselves, so not SiLU).
 * - **rowBlockAttention 12**: every decomposed attention of the 12 blocks (`bmm → reshape →
 *   add(mask) → safe_softmax → expand → reshape → expand → reshape → bmm`). Among the real assets
 *   only the DiT emits this spelling; the other 7 parts carry the `attention` op (ADR 0023).
 * - identityExpand 24: of the 12 blocks × 4, the two inside the window (P side and V side) are
 *   swallowed by the fused step and no longer count as aliases. **48 → 24 is that move, not a
 *   regression**; if `rowBlockAttention` went back to 0 it would be 48 again (only one of the two
 *   moving means an accident in an accepted set).
 *
 * So this gate guards both "the grabbed kinds do not drop out" and "**it moves when an accepted
 * set widens**". If a count changes, first check whether ROPE_RULE / ADALN_RULE accepted something
 * unintentionally (the moment a set widens past an exact form, the fallback's correctness
 * guarantee is gone).
 */
test.skipIf(!IRODORI_AVAILABLE)(
  "the real Irodori DiT grabs rope 23 / rowBlockAttention 12 / silu 17 / identityExpand 24 per run (adaln is a different spelling: 0)",
  async () => {
    const graph = await readIrodoriGraph("dit");
    const expected: FusionCounts = {
      ...NONE,
      silu: 17,
      rope: 23,
      rowBlockAttention: 12,
      identityExpand: 24,
    };
    // ヒット数は S に依存しない（`ditSymMax` 750 の内側で 2 点）。**行ブロック枚数は
    // S で変わる**（128MiB 上限では S=125 が 1 枚・S=750 が 2 枚）が、ヒット数は動かない。
    for (const sequence of [125, 750]) {
      assertEquals(fusionCounts(graph, irodoriDitShapes(sequence)), expected, `S=${sequence}`);
    }
    // 条件側 K/V 射影（ADR 0114 で `dit` から割り出した 72 linear + rms_norm + cat）は融合を 1 つも
    // 掴まない — `dit` の 3 種がここへ移っていないことの対の観測点。
    assertEquals(
      fusionCounts(await readIrodoriGraph("dit_context"), IRODORI_CONTEXT_SHAPES),
      NONE,
      "dit_context",
    );
  },
);

test.skipIf(!IRODORI_AVAILABLE)("実資産の Irodori 条件経路と codec の融合ヒット数", async () => {
  // The backbone (ModernBERT-ja, 25 layers) is the only half-split RoPE: 25 layers × q / k = 50.
  // It is the counterpart of Irodori's own blocks (dit / speaker — the pair form): **if only one
  // side drops out, this tells which spelling changed**.
  assertEquals(
    fusionCounts(await readIrodoriGraph("backbone"), { input_ids: [1, 256] }),
    { ...NONE, rope: 50 },
    "backbone",
  );
  assertEquals(
    fusionCounts(await readIrodoriGraph("text_proj"), { hidden: [1, 256, 768] }),
    { ...NONE, silu: 1 },
    "text_proj",
  );
  assertEquals(
    fusionCounts(await readIrodoriGraph("caption_proj"), { hidden: [1, 512, 768] }),
    { ...NONE, silu: 1 },
    "caption_proj",
  );
  // speaker has 8 blocks (silu 8 + 8 gates). Its RoPE is the DiT's pair form on all 12 heads:
  // 8 blocks × q / k = 16 sites, of which the first use of the tables stays plain (its two
  // sym_prefix_slices sit inside the window) — 15.
  assertEquals(
    fusionCounts(await readIrodoriGraph("speaker"), { latent: [1, 750, 128] }),
    { ...NONE, silu: 8, rope: 15 },
    "speaker",
  );
  assertEquals(
    fusionCounts(await readIrodoriGraph("duration"), {
      text_state: [1, 256, 512],
      speaker_vec: [1, 768],
      has_speaker: [1, 1],
      caption_vec: [1, 512],
      has_caption: [1, 1],
    }),
    { ...NONE, silu: 5 },
    "duration",
  );
  // codec は Snake 活性（`sin` 29 本 — sigmoid ではない）と `conv_transpose1d` 4 本で、
  // silu / upsample2x のどちらの形も持たない。全 0 は「掴む形が無い」の記録で、非 0 へ
  // 変わったら受理集合が広がったということ。
  for (const [name, shapes] of [
    ["codec_decoder", { latent: [1, 750, 32] }],
    ["codec_encoder", { wav: [1, 750, 1920] }],
  ] as const) {
    assertEquals(fusionCounts(await readIrodoriGraph(name), shapes), NONE, name);
  }
});
