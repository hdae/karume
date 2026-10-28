/**
 * Host preprocessing from the reference audio (wav) up to the input of the codec encoder
 * (upstream's `_normalize_loudness` / `encode_waveform` of `codec.py`, and `DACVAE._pad`).
 *
 * There are 3 steps, and the order matters: **truncation → normalization → reflect pad**.
 * Truncating after normalization would mix the loudness of the discarded part into the LUFS
 * (upstream takes `wav[:, :int(max_ref_seconds·sr)]` first). Padding before normalization would mix
 * the mirrored part into the peak test.
 *
 * The numeric source of truth is the golden `outputs/series/dacvae-32dim/host/` (5 cases written by
 * `tools/export-recipes/irodori/dacvae/host.py`, which **calls** upstream), and the comparison is
 * `packages/models/tests/irodori_reference_test.ts`.
 */

import { ModelInputError } from "../../errors.ts";
import { integratedLoudness } from "./loudness.ts";

/**
 * Target loudness of the reference audio (LUFS).
 *
 * The default of upstream's `inference_runtime.SamplingRequest.ref_normalize_db`, not in
 * `pipelineConfig` (a distribution has no way to declare another value). Treated the same as the
 * tail-trim thresholds (`trim.ts`): **not held as a runtime knob**.
 */
const TARGET_DB = -16;

/**
 * dB → linear conversion factor (`ln(10)/20`). The same number as audiotools'
 * `AudioSignal.GAIN_FACTOR`; `gainFactor` in the golden's `meta.json` holds the measured value.
 */
const GAIN_FACTOR = Math.LN10 / 20;

/**
 * Result of {@link normalizeReference}. The scalars are returned as observation points for the
 * golden comparison.
 */
type NormalizedReference = {
  readonly data: Float32Array<ArrayBuffer>;
  /** The measured integrated loudness (LUFS). */
  readonly refDb: number;
  /** LUFS gain (`10^((−16 − refDb)/20)`). */
  readonly loudnessGain: number;
  /** Peak after the LUFS gain (only when it exceeds 1 does the following peak gain drop below 1). */
  readonly peakBeforeScale: number;
  /** Gain of the peak limit (`1/peak` only when `peak > 1`, otherwise exactly 1). */
  readonly peakGain: number;
};

/**
 * LUFS normalization + peak limit (upstream `_normalize_loudness` → `ensure_max_of_audio`).
 *
 * Only **2 gains** are applied, and their product is the whole normalization
 * (`_decomposition_evidence` in `irodori/dacvae/host.py` measures bit identity with upstream's
 * output every time).
 */
export const normalizeReference = (
  samples: Float32Array<ArrayBuffer>,
  sampleRate: number,
): NormalizedReference => {
  // MUST: an empty waveform fails at this entry. `integratedLoudness` further on would throw a
  // `RangeError` with 0 windows, but that is a check of the premises of derived values
  // (`sampleRate` and the window length), so as a diagnostic about the length of the caller's
  // waveform its origin is unreadable (the "audio dimensions" of ADR 0107 decision 5).
  if (samples.length === 0) {
    throw new ModelInputError("irodori: the reference audio has 0 samples (the waveform is empty)");
  }
  const refDb = integratedLoudness(samples, sampleRate);
  const loudnessGain = Math.exp((TARGET_DB - refDb) * GAIN_FACTOR);
  const scaled = new Float32Array(samples.length) as Float32Array<ArrayBuffer>;
  let peakBeforeScale = 0;
  for (let i = 0; i < samples.length; i += 1) {
    // MUST: non-finite samples fail here. NaN slips past the peak test because
    // `magnitude > peakBeforeScale` is always false, and +Inf **crushes every sample to 0** through
    // `peakGain = 1/∞ = 0` (only a single NaN remains) — neither raises an exception, and both pass
    // as "almost silent reference audio".
    if (!Number.isFinite(samples[i])) {
      // The sample values are the content of the wav the caller passed, so input-caused (ADR 0107
      // decision 2).
      throw new ModelInputError(
        `irodori: sample ${i} of the reference audio is non-finite (${samples[i]})`,
      );
    }
    // The assignment rounds to f32. The peak is taken from **the rounded value** (upstream also
    // takes the max of an f32 tensor).
    scaled[i] = samples[i] * loudnessGain;
    const magnitude = Math.abs(scaled[i]);
    if (magnitude > peakBeforeScale) peakBeforeScale = magnitude;
  }
  const peakGain = peakBeforeScale > 1 ? 1 / peakBeforeScale : 1;
  if (peakGain !== 1) {
    for (let i = 0; i < scaled.length; i += 1) scaled[i] *= peakGain;
  }
  return { data: scaled, refDb, loudnessGain, peakBeforeScale, peakGain };
};

/**
 * Fills the tail with a mirror image to round the length up to a multiple of `hopLength`
 * (upstream's reflect pad in `DACVAE._pad`).
 *
 * The mirror **does not include the edge sample** (adding 2 to `[a,b,c,d]` gives `[a,b,c,d,c,b]`)
 * — the spelling of torch's `F.pad(mode="reflect")`. When the length is already a multiple, the
 * input is returned as is (no copy is made either).
 *
 * MUST: input shorter than `hopLength` fails. The padding is at most `hopLength − 1`, so with
 * shorter input the mirror would jump past its own start (torch throws under the same condition).
 */
export const reflectPadToHop = (
  samples: Float32Array<ArrayBuffer>,
  hopLength: number,
): Float32Array<ArrayBuffer> => {
  if (!Number.isInteger(hopLength) || hopLength <= 0) {
    throw new RangeError(`reflectPadToHop: hopLength ${hopLength} is not a positive integer`);
  }
  if (samples.length < hopLength) {
    // The length comes from the reference audio the caller passed (the wav of
    // `SpeakerInput`), so the 400 side (the "audio dimensions" of ADR 0107 decision 5). The
    // `hopLength` just above is a number the distribution declares, which the caller cannot fix,
    // so it stays a `RangeError`.
    throw new ModelInputError(
      `reflectPadToHop: the reference audio is only ${samples.length} samples (hopLength ${hopLength} is required)`,
    );
  }
  const pad = (hopLength - (samples.length % hopLength)) % hopLength;
  if (pad === 0) return samples;
  const padded = new Float32Array(samples.length + pad) as Float32Array<ArrayBuffer>;
  padded.set(samples);
  for (let i = 0; i < pad; i += 1) padded[samples.length + i] = samples[samples.length - 2 - i];
  return padded;
};
