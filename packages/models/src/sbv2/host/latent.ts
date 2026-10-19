/**
 * Construction of z_p (decode front's m_p / logs_p from phonemes to frames and add noise).
 *
 * The reference implementation `SynthesizerTrnJPExtra.infer`'s
 * `m_p = attn · m_pᵀ` → `z_p = m_p + randn_like(m_p)·exp(logs_p)·noise_scale`.
 * attn is the 0/1 matrix of a monotonic path, so the matrix product is equivalent to a gather by
 * `expandIdx` (see the doc of `duration.ts`).
 */

/**
 * Builds `z_p[c][f] = m_p[c][idx[f]] + noise[c][f]·exp(logs_p[c][idx[f]])·noiseScale`.
 *
 * @param mP front's output 2 (row-major data of `[1, C, P]`).
 * @param logsP front's output 3 (same shape).
 * @param expandIdx frame → phoneme index (length Ty).
 * @param channels C (taken from the shape of front's graph output — not hard-coded here).
 * @param noise standard normal sequence (length `C·Ty`). **Not a surface the user supplies** — it
 * always comes from the internal `random.normals`. It is an argument because
 * `examples/sbv2/dump.ts` (removed in 0.15) wrote the same sequence out as `zp_noise`, and the
 * reference side (`demo.py`) reads it to compare.
 *
 * NOTE: the expression runs in f64 and is rounded once on assignment to `Float32Array` (the same
 * house style as `random.ts` in the same directory). The reference is sequential f32, so 40% of the
 * elements differ by 1 ulp, but that is 3 orders of magnitude below the measured maxAbs 5.16e-5 of
 * the waveform comparison that measures this path, and no discretisation like ceil sits in between.
 */
export const buildZp = (
  mP: Float32Array<ArrayBuffer>,
  logsP: Float32Array<ArrayBuffer>,
  expandIdx: Int32Array<ArrayBuffer>,
  channels: number,
  noise: Float32Array<ArrayBuffer>,
  noiseScale: number,
): Float32Array<ArrayBuffer> => {
  if (!Number.isInteger(channels) || channels <= 0) {
    throw new Error(`Channel count ${channels} is not an integer >= 1`);
  }
  if (logsP.length !== mP.length) {
    throw new Error(`m_p(${mP.length}) and logs_p(${logsP.length}) differ in element count`);
  }
  const phonemes = mP.length / channels;
  if (!Number.isInteger(phonemes)) {
    throw new Error(`m_p element count ${mP.length} is not divisible by C=${channels}`);
  }
  const frames = expandIdx.length;
  if (noise.length !== channels * frames) {
    throw new Error(`Noise length ${noise.length} differs from C·Ty(${channels * frames})`);
  }
  const zP = new Float32Array(channels * frames);
  for (let c = 0; c < channels; c += 1) {
    const rowSource = c * phonemes;
    const rowTarget = c * frames;
    for (let f = 0; f < frames; f += 1) {
      const index = expandIdx[f];
      if (index < 0 || index >= phonemes) {
        throw new Error(`Expansion index ${index} (frame ${f}) is outside P=${phonemes}`);
      }
      const source = rowSource + index;
      zP[rowTarget + f] = mP[source] + noise[rowTarget + f] * Math.exp(logsP[source]) * noiseScale;
    }
  }
  return zP;
};
