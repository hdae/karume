/**
 * latent と条件テンソルの整形（IR の外側）。
 */

import type { Tensor } from "@karume/runtime";

const f32 = Math.fround;

/**
 * VAE の per-channel latent 平均 / 標準偏差（`AutoencoderKLQwenImage.config.latents_mean` /
 * `latents_std`）。
 *
 * **なぜパイプライン側に定数として置くか**: この 2 本は VAE の config にしかなく、IR にも
 * 配布資産にも入っていない（VAE decoder のグラフは**逆正規化済み**の latent を受ける）。
 * manifest の `pipelineConfig` にも置かない — アーキ定数はパイプライン実装が持つ、が
 * ADR 0038 §2 の「導出元を一意に保つ」規律。
 *
 * MUST: 手写しの数を検証なしで置かない。参照フィクスチャの `latents_mean` / `latents_std` との
 * ビット一致をテストが固定する。値を差し替えるモデルを使うならその 1 本が落ちる。
 *
 * MUST: モジュールスコープに TypedArray を持たない（`Float32Array.from` を import 時に実行し
 * ない）— 横断不変条件「全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止）」。
 * ここは `readonly number[]` の配列リテラルに留め、`Float32Array` は {@link animaLatents} が
 * 呼び出しごとに組む（16 要素 × 2 の生成費用しか増えない）。消費側（`pipeline.ts` の逆正規化・
 * barrel の利用者）はそのアクセサだけを通る。
 *
 * 各値は VAE config の 4 桁小数（例 `-0.7571`）を f32 へ丸めた値の最短 10 進表記で、参照
 * フィクスチャ（torch の f32 テンソル由来）と同じ表記になる。f64 の小数のまま置くと、この
 * 配列自体が逆正規化で実際に使う f32 と食い違う。
 */
export const ANIMA_LATENTS_MEAN: readonly number[] = [
  -0.757099986076355,
  -0.708899974822998,
  -0.911300003528595,
  0.10750000178813934,
  -0.1745000034570694,
  0.9653000235557556,
  -0.1517000049352646,
  1.5507999658584595,
  0.41339999437332153,
  -0.07150000333786011,
  0.5516999959945679,
  -0.36320000886917114,
  -0.19220000505447388,
  -0.9496999979019165,
  0.25029999017715454,
  -0.2921000123023987,
];

/** {@link ANIMA_LATENTS_MEAN} と対の標準偏差。 */
export const ANIMA_LATENTS_STD: readonly number[] = [
  2.8183999061584473,
  1.4541000127792358,
  2.327500104904175,
  2.6558001041412354,
  1.219599962234497,
  1.770799994468689,
  2.6052000522613525,
  2.0743000507354736,
  3.268699884414673,
  2.152600049972534,
  2.8652000427246094,
  1.5578999519348145,
  1.638200044631958,
  1.1253000497817993,
  2.8250999450683594,
  1.9160000085830688,
];

/**
 * 逆正規化の定数対を**呼び出しごとに独立した写しで**返す（公開面が出すのはこちら）。
 *
 * MUST: 呼び出しごとに新しい `Float32Array` を組む（使い回さない）。返り値を消費側が
 * 書き換えても、`generate` の逆正規化にも次の呼び出しの返り値にも波及しない。
 *
 * NOTE: 「凍らせて出す」は成立しない — 要素を持つ TypedArray は `Object.freeze` が
 * TypeError を投げる。写しを返すアクセサが唯一の形。
 */
export const animaLatents = (): {
  mean: Float32Array<ArrayBuffer>;
  std: Float32Array<ArrayBuffer>;
} => ({
  mean: Float32Array.from(ANIMA_LATENTS_MEAN),
  std: Float32Array.from(ANIMA_LATENTS_STD),
});

/**
 * latent の per-channel 逆正規化。
 * MUST: `latents · std` に直さない — 参照実装は **std の逆数を作って割る**ので、
 * 掛け算に変えると最終桁が変わる（ビット一致が崩れる）。
 * MUST: **B=1 前提**を入口で検査する。チャネルごとの区間を `length / channels` で
 * 割り出しているので、B>1 の `[B,C,H,W]` を渡すと 1 区間が B 枚ぶんに伸び、B 枚に別々の
 * mean/std が当たった**沈黙誤値**になる（長さは割り切れるので黙って通る）。
 */
export const denormalizeLatents = (
  latents: Float32Array,
  shape: readonly number[],
  mean: Float32Array,
  std: Float32Array,
): Float32Array<ArrayBuffer> => {
  if (shape[0] !== 1) {
    throw new Error(`逆正規化は batch=1 前提（B=${shape[0]} はチャネル区間が崩れる）`);
  }
  if (shape[1] !== mean.length || shape[1] !== std.length) {
    throw new Error(
      `逆正規化: latent のチャネル数 ${shape[1]} が mean ${mean.length} / std ${std.length} と違う`,
    );
  }
  const elements = shape.reduce((a, b) => a * b, 1);
  if (latents.length !== elements) {
    throw new Error(`逆正規化: 要素数 ${latents.length} が shape [${shape}] と違う`);
  }
  const perChannel = latents.length / mean.length;
  const out = new Float32Array(latents.length);
  for (let channel = 0; channel < mean.length; channel += 1) {
    const inverseStd = f32(1 / std[channel]);
    for (let index = 0; index < perChannel; index += 1) {
      const at = channel * perChannel + index;
      out[at] = f32(f32(latents[at] / inverseStd) + mean[channel]);
    }
  }
  return out;
};

/**
 * conditioner 出力を `min_sequence_length` 行までゼロ詰めする（IR の外側）。
 *
 * MUST: **B=1 前提**と行数上限を入口で検査する。`[B,T,W]` を平坦にコピーしているので、
 * B>1 だと 2 枚目以降が 1 枚目の余白へ流れ込み、行数さえ足りていれば黙って通る。
 * 行数超過（T > rows）は `set` が RangeError を投げるが、その診断は「どの段の話か」を
 * 何も言わないのでここで名前付きで落とす。
 * MUST: dtype も見る。i32 / bool のテンソルを渡すと `TypedArray.prototype.set` が**黙って
 * 数値変換**し、shape も長さも合ったまま値だけが別物になる（沈黙誤値）。
 */
export const padSequence = (hidden: Tensor, rows: number): Float32Array<ArrayBuffer> => {
  if (hidden.dtype !== "f32") {
    throw new Error(`512 パディングは f32 前提（dtype ${hidden.dtype} は黙って数値変換される）`);
  }
  if (hidden.shape[0] !== 1) {
    throw new Error(`512 パディングは batch=1 前提（B=${hidden.shape[0]} は余白へ流れ込む）`);
  }
  if (hidden.shape[1] > rows) {
    throw new Error(`512 パディング: 行数 ${hidden.shape[1]} が上限 ${rows} を超えている`);
  }
  const width = hidden.shape[2];
  const padded = new Float32Array(rows * width);
  padded.set(hidden.data);
  return padded;
};
