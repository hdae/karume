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
 * MUST: この 2 本を**公開面（barrel）へは出さない** — TypedArray は可変なので、実体を配ると
 * 消費側の書き換えが `generate` の逆正規化へ黙って波及する（グローバル可変状態の禁止 —
 * CLAUDE.md 横断不変条件）。`export` はモジュール間の内部利用（`pipeline.ts`）のためで、
 * barrel が出すのは写しを返す {@link animaLatents}。
 */
export const ANIMA_LATENTS_MEAN: Float32Array<ArrayBuffer> = Float32Array.from([
  -0.7571,
  -0.7089,
  -0.9113,
  0.1075,
  -0.1745,
  0.9653,
  -0.1517,
  1.5508,
  0.4134,
  -0.0715,
  0.5517,
  -0.3632,
  -0.1922,
  -0.9497,
  0.2503,
  -0.2921,
]);

/** {@link ANIMA_LATENTS_MEAN} と対の標準偏差。 */
export const ANIMA_LATENTS_STD: Float32Array<ArrayBuffer> = Float32Array.from([
  2.8184,
  1.4541,
  2.3275,
  2.6558,
  1.2196,
  1.7708,
  2.6052,
  2.0743,
  3.2687,
  2.1526,
  2.8652,
  1.5579,
  1.6382,
  1.1253,
  2.8251,
  1.916,
]);

/**
 * 逆正規化の定数対を**呼び出しごとに独立した写しで**返す（公開面が出すのはこちら）。
 *
 * MUST: 実体（{@link ANIMA_LATENTS_MEAN} / {@link ANIMA_LATENTS_STD}）を返さない。返り値を
 * 消費側が書き換えても、`generate` の逆正規化にも次の呼び出しの返り値にも波及しない。
 *
 * NOTE: 「凍らせて出す」は成立しない — 要素を持つ TypedArray は `Object.freeze` が
 * TypeError を投げる。写しを返すアクセサが唯一の形。
 */
export const animaLatents = (): {
  mean: Float32Array<ArrayBuffer>;
  std: Float32Array<ArrayBuffer>;
} => ({
  mean: ANIMA_LATENTS_MEAN.slice(),
  std: ANIMA_LATENTS_STD.slice(),
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
