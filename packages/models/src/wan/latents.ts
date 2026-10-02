/**
 * Wan2.1 の潜在とテキスト文脈の整形（グラフの外側 — ADR 0118 決定 2 / 4）。
 */

const f32 = Math.fround;

/**
 * VAE の per-channel の潜在の平均 / 標準偏差（上流 `AutoencoderKLWan.config.latents_mean` /
 * `latents_std` — pin した revision の `vae/config.json`）。
 *
 * パイプライン側に定数として置く理由は anima（`anima/latents.ts`）と同じ: この 2 本は VAE の config
 * にしかなく、IR にも配布資産にも入っていない（VAE の chunk グラフは**逆正規化済み**の潜在を受ける）。
 * 各値は config の 4 桁小数を f32 へ丸めた値の最短 10 進表記（逆正規化で実際に使う f32 そのもの）。
 *
 * NOTE: 値は anima（QwenImage の VAE — Wan の VAE からの fine-tune）の定数と同じだが、家族をまたいで
 * import しない（家族ごとに資産の出所が違う — 片方の差し替えがもう片方を動かさないように）。
 *
 * MUST: モジュールスコープに TypedArray を持たない（横断不変条件「全モジュール副作用ゼロ」）。
 */
export const WAN_LATENTS_MEAN: readonly number[] = [
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

/** {@link WAN_LATENTS_MEAN} と対の標準偏差。 */
export const WAN_LATENTS_STD: readonly number[] = [
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
 * 潜在 `[C, F, H, W]` の per-channel の逆正規化（VAE の前 — 上流 `pipeline_wan.py` の
 * `latents / (1 / std) + mean` の逐語）。
 *
 * MUST: `latents · std` に直さない — 上流は std の逆数を f32 で作ってから割るので、掛け算に変えると
 * 最終桁が変わる。
 */
export const denormalizeWanLatents = (latents: Float32Array): Float32Array<ArrayBuffer> => {
  const channels = WAN_LATENTS_MEAN.length;
  const perChannel = latents.length / channels;
  if (!Number.isInteger(perChannel) || perChannel < 1) {
    throw new Error(`逆正規化: 要素数 ${latents.length} が ${channels} チャネルで割り切れない`);
  }
  const out = new Float32Array(latents.length);
  for (let channel = 0; channel < channels; channel += 1) {
    const inverseStd = f32(1 / f32(WAN_LATENTS_STD[channel]));
    const mean = f32(WAN_LATENTS_MEAN[channel]);
    const from = channel * perChannel;
    for (let index = from; index < from + perChannel; index += 1) {
      out[index] = f32(f32(latents[index] / inverseStd) + mean);
    }
  }
  return out;
};
