/**
 * **実配布グラフに対する融合ヒット数の門**（GPU 不要）。
 *
 * 融合はエクスポータのノード発行順が 1 つ変わるだけで黙って外れる。値は正しいまま
 * （掴めなければ素の列に落ちるので）性能だけが戻り、例外も警告も出ない — ADR 0040 §3 が
 * `lastRunFusions` を常設した理由がここで、**その数字に突合相手を与える**のが本ファイル。
 *
 * `planFusions` は純関数なので、実資産から IR を読んで計画するだけで判定できる（1 dispatch も
 * 出さない）。安全のため safetensors の**ヘッダだけ**を読む — 実体は合計 7GB 級で、
 * IR は `__metadata__.karume_ir` に載っている。
 *
 * MUST: 資産は `models/karume-anima-turbo/`（untracked・ローカル資産）。無い環境は理由を出して
 * **明示 SKIP** する（テストを消して無音で緑にしない — ADR 0005）。
 *
 * NOTE: ここで固定するのは **run 1 回あたり**の値（`lastRunFusions` と同じ寿命）。ADR 0040 の
 * 実測欄が載せている predict 1 回ぶんの合計は、これらをパイプラインの run 回数で畳んだもの:
 * rope 56×8step + 55 = **503** / silu 2×8 + 28 + 29×9tile = **305** /
 * upsample2x 3×9tile = **27** / identityExpand 112 + 48 = **160** / adaln 85×8step = **680**
 * （ADR の「adaln 85」は静的な鎖の本数 = 本ファイルの DiT 1 run ぶん）。
 * VAE の 9 タイルは 1024px（latent 128・タイル 64・最小重なり 8）で 1 軸 3 枚 × 2 軸。
 */

import { assertEquals } from "@std/assert";
import { type IrGraph, parseIrGraph } from "../src/format/ir.ts";
import { type FusionCounts, planFusions } from "../src/runtime/fusion.ts";
import { bindSymbols, countUses, planGraph } from "../src/runtime/plan.ts";

const ASSETS_DIR = new URL("../../../models/karume-anima-turbo/anima-turbo/", import.meta.url);

/** safetensors のヘッダ JSON だけを読む（実体は読まない）。 */
const readIrGraph = async (relative: string): Promise<IrGraph> => {
  const file = await Deno.open(new URL(relative, ASSETS_DIR), { read: true });
  try {
    const lengthBytes = new Uint8Array(8);
    await file.read(lengthBytes);
    const length = Number(new DataView(lengthBytes.buffer).getBigUint64(0, true));
    const headerBytes = new Uint8Array(length);
    for (let read = 0; read < length;) {
      const chunk = await file.read(headerBytes.subarray(read));
      if (chunk === null) throw new Error(`${relative}: ヘッダ ${length} バイトを読み切れない`);
      read += chunk;
    }
    const header = JSON.parse(new TextDecoder().decode(headerBytes));
    return parseIrGraph(header.__metadata__.karume_ir);
  } finally {
    file.close();
  }
};

const ASSETS_AVAILABLE = await Deno.stat(new URL("transformer/", ASSETS_DIR))
  .then(() => true)
  .catch(() => false);
if (!ASSETS_AVAILABLE) {
  console.warn(
    `[karume] ${ASSETS_DIR.pathname} が無いため実資産の融合ヒット数を SKIP する` +
      "（エクスポータのノード発行順の退行は実資産でしか検出できない）",
  );
}

const fusionCounts = (
  graph: IrGraph,
  inputShapes: Readonly<Record<string, readonly number[]>>,
): FusionCounts =>
  planFusions(planGraph(graph, bindSymbols(graph, inputShapes)).nodes, {
    useCounts: countUses(graph),
    outputNames: new Set(graph.outputs),
  }).counts;

/**
 * DiT の入力 shape（S = パッチトークン数。1024px = 64×64 = 4096）。ヒット数は S に依存しない
 * ことを下のテストが 2 点で確かめる。
 */
const ditShapes = (sequence: number): Readonly<Record<string, readonly number[]>> => ({
  tokens: [1, sequence, 68],
  timesteps_proj: [1, 2048],
  encoder_hidden_states: [1, 512, 1024],
  rope_cos: [1, 1, sequence, 128],
  rope_sin: [1, 1, sequence, 128],
});

const NONE: FusionCounts = { silu: 0, upsample2x: 0, rope: 0, adaln: 0, identityExpand: 0 };

Deno.test({
  name: "実資産の DiT は run 1 回で adaln 85 / rope 56 / silu 2 を掴む（w8a8 と f16 で同一）",
  ignore: !ASSETS_AVAILABLE,
  fn: async () => {
    const expected: FusionCounts = { ...NONE, silu: 2, rope: 56, adaln: 85 };
    for (const quant of ["i8", "f16"] as const) {
      const graph = await readIrGraph(`transformer/model.${quant}.safetensors`);
      // 融合は f32 の計算経路だけを見るので、重みの格納形が変わってもヒット数は動かない。
      assertEquals(fusionCounts(graph, ditShapes(4096)), expected, `${quant}: 1024px（S=4096）`);
      assertEquals(fusionCounts(graph, ditShapes(1024)), expected, `${quant}: 512px（S=1024）`);
    }
  },
});

Deno.test({
  name: "実資産の text encoder / conditioner / VAE decoder の融合ヒット数",
  ignore: !ASSETS_AVAILABLE,
  fn: async () => {
    const textEncoder = await readIrGraph("text_encoder/model.safetensors");
    assertEquals(
      fusionCounts(textEncoder, { input_ids: [1, 64] }),
      { ...NONE, silu: 28, rope: 55, identityExpand: 112 },
      "text encoder",
    );
    const conditioner = await readIrGraph("text_conditioner/model.safetensors");
    assertEquals(
      fusionCounts(conditioner, {
        source_hidden_states: [1, 64, 1024],
        target_input_ids: [1, 512],
      }),
      { ...NONE, identityExpand: 48 },
      "conditioner",
    );
    const vae = await readIrGraph("vae_decoder/model.safetensors");
    assertEquals(
      fusionCounts(vae, { latents: [1, 16, 64, 64] }),
      { ...NONE, silu: 29, upsample2x: 3 },
      "VAE decoder（タイル 1 枚ぶん）",
    );
  },
});
