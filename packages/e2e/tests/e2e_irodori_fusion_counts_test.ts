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
 * Irodori DiT（12 ブロック）。**rope も adaln も 0** で、これは退行ではなく matcher の受理集合が
 * この綴りを含まないことの記録:
 *
 * - rope 0: DiT の RoPE は**偶奇形**（`[1,S,H·D/2,2]` へ view して最終軸を slice / neg / cat し
 *   reshape で戻す）。`runtime/fusion.ts` の ROPE_RULE は `[1,H,S,D]` の半割り形だけを受理し、
 *   偶奇形は「式が似ている」で広げない MUST を掲げている（掴めなければ素の列で値は正しい）。
 * - adaln 0: ADALN_RULE の先頭 op は `layer_norm` だが、DiT の正規化は `rms_norm` 87 本で
 *   `layer_norm` は 1 本も無い。
 * - silu 17: 前段の timestep MLP 5 本 + 12 ブロック × 1 本。残る sigmoid 12 本は
 *   `mul(v, sigmoid(u))` のゲート（自分自身に掛からないので SiLU ではない）。
 * - **rowBlockAttention 12**: 12 ブロックの分解 attention（`bmm → reshape → add(mask) →
 *   safe_softmax → expand → reshape → expand → reshape → bmm`）を全て掴む。この綴りを出すのは
 *   実資産では DiT だけで、他の 7 パートは `attention` op（ADR 0023）を持つ。
 * - identityExpand 24: 12 ブロック × 4 本のうち、窓の内側の 2 本（P 側と V 側）は融合ステップに
 *   飲まれるので別名化として数えられなくなる。**48 → 24 は退行ではなくこの移動**で、
 *   `rowBlockAttention` が 0 に戻れば 48 に戻る（片方だけ動いたら受理集合の事故）。
 *
 * したがってこの門が守るのは「掴めている 3 種が外れないこと」と「**受理集合を広げたとき
 * ここが動く**こと」の両方。0 が非 0 に変わったら、まず ROPE_RULE / ADALN_RULE の受理集合が
 * 意図せず広がっていないかを見る（広げた瞬間 fallback の正しさ保証が消える）。
 */
test.skipIf(!IRODORI_AVAILABLE)(
  "実資産の Irodori DiT は run 1 回で rowBlockAttention 12 / silu 17 / identityExpand 24 を掴む（rope / adaln は綴りが違って 0）",
  async () => {
    const graph = await readIrodoriGraph("dit");
    const expected: FusionCounts = {
      ...NONE,
      silu: 17,
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
  // backbone（ModernBERT-ja 25 層）だけは半割り形の RoPE なので、25 層 × q/k = 50 を掴む。
  // Irodori 自前のブロック（dit / speaker）が 0 なのと対になる観測点で、**片方だけ外れたら
  // どちらの綴りが変わったのか**がここで割れる。
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
  // speaker は 8 ブロック（silu 8 + ゲート 8）。RoPE は DiT と同じ偶奇形なので 0。
  assertEquals(
    fusionCounts(await readIrodoriGraph("speaker"), { latent: [1, 750, 128] }),
    { ...NONE, silu: 8 },
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
