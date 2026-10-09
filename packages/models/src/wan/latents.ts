/**
 * Wan の潜在とテキスト文脈の整形（グラフの外側 — ADR 0118 決定 2 / 4）。
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
 * Wan2.2（TI2V 5B）の VAE の per-channel の潜在の平均（48 チャネル — 上流
 * `AutoencoderKLWan.config.latents_mean`・pin した revision の `vae/config.json`）。置く理由と値の規約
 * （config の小数〈4 桁以下〉を f32 へ丸めた値の最短 10 進表記）は {@link WAN_LATENTS_MEAN} と同じ。
 *
 * MUST: 値を変えるときは `tests/fixtures/wan-latents/wan22-ti2v.json` も同じ値にする — fixture を挟んだ
 * 両側のテスト（TS のホストテストと recipe の `wan/tests/test_ti2v_generation_stats.py` が pin の config と
 * 照合）が片側だけの更新を赤にする。
 */
export const WAN22_LATENTS_MEAN: readonly number[] = [
  -0.2289000004529953,
  -0.005200000014156103,
  -0.13230000436306,
  -0.23389999568462372,
  -0.2799000144004822,
  0.017400000244379044,
  0.18379999697208405,
  0.15569999814033508,
  -0.13819999992847443,
  0.05420000106096268,
  0.28130000829696655,
  0.08910000324249268,
  0.15700000524520874,
  -0.009800000116229057,
  0.03750000149011612,
  -0.18250000476837158,
  -0.22460000216960907,
  -0.12070000171661377,
  -0.0697999969124794,
  0.5109000205993652,
  0.26649999618530273,
  -0.21080000698566437,
  -0.21580000221729279,
  0.2502000033855438,
  -0.20550000667572021,
  -0.03220000118017197,
  0.11089999973773956,
  0.156700000166893,
  -0.07289999723434448,
  0.08990000188350677,
  -0.2799000144004822,
  -0.12300000339746475,
  -0.031300000846385956,
  -0.164900004863739,
  0.011699999682605267,
  0.0723000019788742,
  -0.2838999927043915,
  -0.20829999446868896,
  -0.052000001072883606,
  0.3747999966144562,
  0.015200000256299973,
  0.195700004696846,
  0.14329999685287476,
  -0.29440000653266907,
  0.3573000133037567,
  -0.05480000004172325,
  -0.1680999994277954,
  -0.06669999659061432,
];

/** {@link WAN22_LATENTS_MEAN} と対の標準偏差。 */
export const WAN22_LATENTS_STD: readonly number[] = [
  0.476500004529953,
  1.0363999605178833,
  0.4514000117778778,
  1.1677000522613525,
  0.5313000082969666,
  0.49900001287460327,
  0.48179998993873596,
  0.5012999773025513,
  0.8158000111579895,
  1.0343999862670898,
  0.5893999934196472,
  1.0901000499725342,
  0.6884999871253967,
  0.6165000200271606,
  0.8453999757766724,
  0.49779999256134033,
  0.5759000182151794,
  0.3522999882698059,
  0.7135000228881836,
  0.680400013923645,
  0.583299994468689,
  1.4146000146865845,
  0.8985999822616577,
  0.5659000277519226,
  0.7069000005722046,
  0.5338000059127808,
  0.48890000581741333,
  0.4916999936103821,
  0.40689998865127563,
  0.4999000132083893,
  0.6866000294685364,
  0.4092999994754791,
  0.570900022983551,
  0.6065000295639038,
  0.6414999961853027,
  0.4943999946117401,
  0.5726000070571899,
  1.204200029373169,
  0.545799970626831,
  1.6886999607086182,
  0.3971000015735626,
  1.059999942779541,
  0.39430001378059387,
  0.5536999702453613,
  0.5443999767303467,
  0.4088999927043915,
  0.7468000054359436,
  0.774399995803833,
];

/**
 * 潜在の正規化・逆正規化の per-channel の統計（世代ごとの表 — 世代の記述子 `descriptor.ts` が持つ参照）。
 * チャネル数は表の本数（`mean` と `std` は同じ本数）。
 */
export type WanLatentStats = {
  readonly mean: readonly number[];
  readonly std: readonly number[];
};

/**
 * std の逆数（上流の `1.0 / torch.tensor(latents_std)` = f32 の std から f32 の割り算）。正規化と逆正規化が共有する
 * — 上流は 2 つの向きとも同じ式で逆数を作る（`pipeline_wan.py` / `pipeline_wan_i2v.py`）。
 *
 * MUST: config の 10 進値から f64 で作った逆数を f32 へ丸める形にしない — 2.2 の 48 本のうち 14 本が違う値になる
 * （段 9 の調査の実測）。
 */
const inverseLatentStd = (std: number): number => f32(1 / f32(std));

/** 統計の本数と潜在の要素数の整合を見て、1 チャネルあたりの要素数を返す（`what` は文言の頭）。 */
const elementsPerChannel = (
  latents: Float32Array,
  stats: WanLatentStats,
  what: string,
): number => {
  if (stats.mean.length !== stats.std.length) {
    throw new Error(
      `${what}: mean ${stats.mean.length} 本と std ${stats.std.length} 本の数が違う`,
    );
  }
  const channels = stats.mean.length;
  const perChannel = latents.length / channels;
  if (!Number.isInteger(perChannel) || perChannel < 1) {
    throw new Error(`${what}: 要素数 ${latents.length} が ${channels} チャネルで割り切れない`);
  }
  return perChannel;
};

/**
 * 潜在 `[C, F, H, W]` の per-channel の逆正規化（VAE の前 — 上流 `pipeline_wan.py` の
 * `latents / (1 / std) + mean` の逐語）。`C` は `stats` の本数。
 *
 * MUST: `latents · std` に直さない — 上流は std の逆数を f32 で作ってから割るので、掛け算に変えると
 * 最終桁が変わる。
 */
export const denormalizeWanLatents = (
  latents: Float32Array,
  stats: WanLatentStats,
): Float32Array<ArrayBuffer> => {
  const perChannel = elementsPerChannel(latents, stats, "逆正規化");
  const channels = stats.mean.length;
  const out = new Float32Array(latents.length);
  for (let channel = 0; channel < channels; channel += 1) {
    const inverseStd = inverseLatentStd(stats.std[channel]);
    const mean = f32(stats.mean[channel]);
    const from = channel * perChannel;
    for (let index = from; index < from + perChannel; index += 1) {
      out[index] = f32(f32(latents[index] / inverseStd) + mean);
    }
  }
  return out;
};

/**
 * 潜在 `[C, F, H, W]` の per-channel の正規化（VAE encoder の出口の mu → DiT の条件の潜在 — 上流
 * `pipeline_wan_i2v.py` の `(latent_condition − latents_mean) * latents_std` の逐語。`latents_std` は std の逆数）。
 * {@link denormalizeWanLatents} の逆向きで、`C` は `stats` の本数。
 *
 * MUST: `(x − mean) / std` に直さない — 上流は std の逆数を f32 で作ってから掛けるので、割り算に変えると
 * 最終桁が変わる。
 */
export const normalizeWanLatents = (
  latents: Float32Array,
  stats: WanLatentStats,
): Float32Array<ArrayBuffer> => {
  const perChannel = elementsPerChannel(latents, stats, "正規化");
  const channels = stats.mean.length;
  const out = new Float32Array(latents.length);
  for (let channel = 0; channel < channels; channel += 1) {
    const inverseStd = inverseLatentStd(stats.std[channel]);
    const mean = f32(stats.mean[channel]);
    const from = channel * perChannel;
    for (let index = from; index < from + perChannel; index += 1) {
      out[index] = f32(f32(latents[index] - mean) * inverseStd);
    }
  }
  return out;
};
