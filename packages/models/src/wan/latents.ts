/**
 * Shaping of Wan's latents and text context (outside the graph — ADR 0118 decisions 2 / 4).
 */

const f32 = Math.fround;

/**
 * The VAE's per-channel latent mean / standard deviation (upstream
 * `AutoencoderKLWan.config.latents_mean` / `latents_std` — `vae/config.json` of the pinned
 * revision).
 *
 * They are constants on the pipeline side for the same reason as in anima (`anima/latents.ts`): the
 * two lists exist only in the VAE config and are in neither the IR nor the distribution assets (the
 * VAE chunk graphs take **already denormalized** latents). Each value is the shortest decimal form of
 * the config's 4-digit decimal rounded to f32 (exactly the f32 that denormalization uses).
 *
 * NOTE: the values equal anima's constants (QwenImage's VAE — a fine-tune of Wan's VAE), but the
 * families do not import each other (each family's assets have their own provenance — so that
 * replacing one does not move the other).
 *
 * MUST: no TypedArray at module scope (cross-cutting invariant "every module is side-effect free").
 */
export const WAN_LATENTS_MEAN: readonly number[] = [
  -0.757099986076355, -0.708899974822998, -0.911300003528595, 0.10750000178813934,
  -0.1745000034570694, 0.9653000235557556, -0.1517000049352646, 1.5507999658584595,
  0.41339999437332153, -0.07150000333786011, 0.5516999959945679, -0.36320000886917114,
  -0.19220000505447388, -0.9496999979019165, 0.25029999017715454, -0.2921000123023987,
];

/** The standard deviation paired with {@link WAN_LATENTS_MEAN}. */
export const WAN_LATENTS_STD: readonly number[] = [
  2.8183999061584473, 1.4541000127792358, 2.327500104904175, 2.6558001041412354, 1.219599962234497,
  1.770799994468689, 2.6052000522613525, 2.0743000507354736, 3.268699884414673, 2.152600049972534,
  2.8652000427246094, 1.5578999519348145, 1.638200044631958, 1.1253000497817993, 2.8250999450683594,
  1.9160000085830688,
];

/**
 * The per-channel latent mean of the Wan2.2 (TI2V 5B) VAE (48 channels — upstream
 * `AutoencoderKLWan.config.latents_mean`, `vae/config.json` of the pinned revision). The reason for
 * keeping it here and the value convention (shortest decimal form of the config's decimal (at most 4
 * digits) rounded to f32) are the same as {@link WAN_LATENTS_MEAN}.
 *
 * MUST: when changing a value, change `tests/fixtures/wan-latents/wan22-ti2v.json` to the same value —
 * the tests on both sides of the fixture (the TS host test, and the recipe's
 * `wan/tests/test_ti2v_generation_stats.py` which matches it against the pinned config) turn a
 * one-sided update red.
 */
export const WAN22_LATENTS_MEAN: readonly number[] = [
  -0.2289000004529953, -0.005200000014156103, -0.13230000436306, -0.23389999568462372,
  -0.2799000144004822, 0.017400000244379044, 0.18379999697208405, 0.15569999814033508,
  -0.13819999992847443, 0.05420000106096268, 0.28130000829696655, 0.08910000324249268,
  0.15700000524520874, -0.009800000116229057, 0.03750000149011612, -0.18250000476837158,
  -0.22460000216960907, -0.12070000171661377, -0.0697999969124794, 0.5109000205993652,
  0.26649999618530273, -0.21080000698566437, -0.21580000221729279, 0.2502000033855438,
  -0.20550000667572021, -0.03220000118017197, 0.11089999973773956, 0.156700000166893,
  -0.07289999723434448, 0.08990000188350677, -0.2799000144004822, -0.12300000339746475,
  -0.031300000846385956, -0.164900004863739, 0.011699999682605267, 0.0723000019788742,
  -0.2838999927043915, -0.20829999446868896, -0.052000001072883606, 0.3747999966144562,
  0.015200000256299973, 0.195700004696846, 0.14329999685287476, -0.29440000653266907,
  0.3573000133037567, -0.05480000004172325, -0.1680999994277954, -0.06669999659061432,
];

/** The standard deviation paired with {@link WAN22_LATENTS_MEAN}. */
export const WAN22_LATENTS_STD: readonly number[] = [
  0.476500004529953, 1.0363999605178833, 0.4514000117778778, 1.1677000522613525, 0.5313000082969666,
  0.49900001287460327, 0.48179998993873596, 0.5012999773025513, 0.8158000111579895,
  1.0343999862670898, 0.5893999934196472, 1.0901000499725342, 0.6884999871253967,
  0.6165000200271606, 0.8453999757766724, 0.49779999256134033, 0.5759000182151794,
  0.3522999882698059, 0.7135000228881836, 0.680400013923645, 0.583299994468689, 1.4146000146865845,
  0.8985999822616577, 0.5659000277519226, 0.7069000005722046, 0.5338000059127808,
  0.48890000581741333, 0.4916999936103821, 0.40689998865127563, 0.4999000132083893,
  0.6866000294685364, 0.4092999994754791, 0.570900022983551, 0.6065000295639038, 0.6414999961853027,
  0.4943999946117401, 0.5726000070571899, 1.204200029373169, 0.545799970626831, 1.6886999607086182,
  0.3971000015735626, 1.059999942779541, 0.39430001378059387, 0.5536999702453613,
  0.5443999767303467, 0.4088999927043915, 0.7468000054359436, 0.774399995803833,
];

/**
 * Per-channel statistics for normalizing and denormalizing latents (one table per generation — the
 * generation descriptor `descriptor.ts` holds the reference). The channel count is the length of the
 * lists (`mean` and `std` have the same length).
 */
export type WanLatentStats = {
  readonly mean: readonly number[];
  readonly std: readonly number[];
};

/**
 * The reciprocal of std (upstream's `1.0 / torch.tensor(latents_std)` = an f32 division of the f32
 * std). Normalization and denormalization share it — upstream builds the reciprocal with the same
 * expression in both directions (`pipeline_wan.py` / `pipeline_wan_i2v.py`).
 *
 * MUST: do not compute the reciprocal in f64 from the config's decimal value and round it to f32 —
 * 14 of 2.2's 48 values come out different (measured in the stage 9 investigation).
 */
const inverseLatentStd = (std: number): number => f32(1 / f32(std));

/**
 * Checks the statistics' length against the latents' element count and returns the elements per
 * channel (`what` heads the message).
 */
const elementsPerChannel = (
  latents: Float32Array<ArrayBuffer>,
  stats: WanLatentStats,
  what: string,
): number => {
  if (stats.mean.length !== stats.std.length) {
    throw new Error(`${what}: mean ${stats.mean.length} 本と std ${stats.std.length} 本の数が違う`);
  }
  const channels = stats.mean.length;
  const perChannel = latents.length / channels;
  if (!Number.isInteger(perChannel) || perChannel < 1) {
    throw new Error(`${what}: 要素数 ${latents.length} が ${channels} チャネルで割り切れない`);
  }
  return perChannel;
};

/**
 * Per-channel denormalization of latents `[C, F, H, W]` (before the VAE — verbatim upstream
 * `pipeline_wan.py` `latents / (1 / std) + mean`). `C` is the length of `stats`.
 *
 * MUST: do not rewrite this as `latents · std` — upstream builds the reciprocal of std in f32 and
 * then divides, so turning it into a multiplication changes the last digit.
 */
export const denormalizeWanLatents = (
  latents: Float32Array<ArrayBuffer>,
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
 * Per-channel normalization of latents `[C, F, H, W]` (mu at the VAE encoder exit → the DiT's
 * condition latent — verbatim upstream `pipeline_wan_i2v.py`
 * `(latent_condition − latents_mean) * latents_std`, where `latents_std` is the reciprocal of std).
 * The inverse of {@link denormalizeWanLatents}; `C` is the length of `stats`.
 *
 * MUST: do not rewrite this as `(x − mean) / std` — upstream builds the reciprocal of std in f32 and
 * then multiplies, so turning it into a division changes the last digit.
 */
export const normalizeWanLatents = (
  latents: Float32Array<ArrayBuffer>,
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
