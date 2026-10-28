/**
 * Deciding the latent length S (the 2 paths of upstream's `inference_runtime._synthesize`).
 *
 * MUST: rounding is Python's `round` (**banker's rounding** = exactly 0.5 goes to the even side).
 * `Math.round` always rounds up, so only when the predicted frame count is exactly .5 it is off by
 * one frame — only the output length changes, no exception is raised, and it stays silent until
 * the golden comparison.
 *
 * MUST: `expm1` is taken in f32. The graph output (log frames) is f32, so feeding the exponential
 * in f64 can change the rounding target in boundary cases.
 */

import { ModelInputError } from "../../errors.ts";

const f32 = Math.fround;

/**
 * The clamp range of S and the seconds ↔ frames conversion (the inner surface only `clampRange`
 * uses).
 */
type FrameBounds = {
  readonly frameRate: number;
  readonly minSeconds: number;
  readonly maxSeconds: number;
};

/**
 * The numbers needed to decide S (source of truth: the manifest's `pipelineConfig`). It holds both
 * seconds → frames and seconds → samples → frames because upstream's formula differs per path.
 * Their consistency (`sampleRate == frameRate × hopLength`) is checked by `config.ts` at parse time.
 */
export type SampleBounds = FrameBounds & {
  readonly sampleRate: number;
  readonly hopLength: number;
};

/**
 * Python's `round(x)` (half-to-even).
 *
 * NOTE: exactly 0.5 is observable only when the value is exactly a half-integer in binary;
 * otherwise it is the same as plain round-to-nearest. The test is written with `x - floor(x)` so
 * that such exact half-integers are not missed.
 */
export const bankerRound = (value: number): number => {
  if (!Number.isFinite(value))
    throw new RangeError(`Value to round ${value} is not a finite number`);
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction > 0.5) return lower + 1;
  if (fraction < 0.5) return lower;
  return lower % 2 === 0 ? lower : lower + 1;
};

/** Lower / upper clamp frames (the same `ceil` / `floor` asymmetry as upstream). */
const clampRange = (bounds: FrameBounds): { readonly min: number; readonly max: number } => ({
  min: Math.max(1, Math.ceil(bounds.minSeconds * bounds.frameRate)),
  max: Math.max(1, Math.floor(bounds.maxSeconds * bounds.frameRate)),
});

/**
 * The decided latent length S, and the length (samples) the waveform is cut to from it.
 *
 * `targetSamples` is returned together with S because the formula differs between the 2 paths (the
 * duration path gives exactly `S × hopLength`; the manual-seconds path is up to `hopLength − 1`
 * samples shorter). Rebuilt on the caller's side, one of the paths would silently emit the full
 * length.
 */
export type SequencePlan = {
  readonly frames: number;
  readonly targetSamples: number;
};

/**
 * Decides S from the output of the duration graph (log frames).
 *
 * `expm1` → banker's rounding → clamp to `[minFrames, maxFrames]` (upstream `_synthesize`).
 * `duration_scale` defaults to 1.0 and is not held as a runtime knob (it is not among the
 * distribution's defaults).
 * On this path `targetSamples` is exactly `S × hopLength` (the cut is effectively a no-op).
 */
export const sequenceLengthFromLogFrames = (
  logFrames: number,
  bounds: SampleBounds,
): SequencePlan => {
  const frames = f32(Math.expm1(f32(logFrames)));
  if (!Number.isFinite(frames)) {
    throw new Error(`The frame count derived from the duration output ${logFrames} is not finite`);
  }
  const { min, max } = clampRange(bounds);
  const bounded = Math.max(min, Math.min(max, bankerRound(frames)));
  return { frames: bounded, targetSamples: bounded * bounds.hopLength };
};

/**
 * Decides S and the cut length from directly given seconds (upstream's `manual_seconds` path — the
 * duration graph does not run).
 *
 * Exactly upstream's (`_synthesize`) spelling: the seconds clamped to `[minSeconds, maxSeconds]`
 * are **first turned into a sample count** (`max(1, trunc(clamped × sampleRate))`) and then into
 * frames (`ceil(targetSamples / hopLength)`).
 *
 * MUST: do not substitute `ceil(clamped × frameRate)` through `frameRate`. The two differ by one
 * frame when "seconds × frameRate is just above an integer (by less than one sample)" (e.g.
 * 1.000005 s → upstream 25 / through frameRate 26). It shows up only in the **length**, not in the
 * values, so without the comparison check it is a difference nobody would notice.
 *
 * `targetSamples` is up to `hopLength − 1` samples shorter than S × hopLength — the caller cuts the
 * waveform to this length (otherwise the audio comes out longer than the requested seconds).
 */
export const sequenceLengthFromSeconds = (seconds: number, bounds: SampleBounds): SequencePlan => {
  // `seconds` is the caller's request knob (`GenerateRequest.durationSeconds`), so the 400
  // side (ADR 0107). There is also a path where the distribution's `maxSeconds` enters through
  // `maxSequenceLength`, but it comes after `config.ts` confirmed finiteness at parse time, so it
  // never reaches here.
  if (!Number.isFinite(seconds)) {
    throw new ModelInputError(`durationSeconds ${seconds} is not a finite number`);
  }
  // Upstream does not clamp again on the frame side (the seconds were clamped first, so the range
  // fits automatically).
  const clamped = Math.min(bounds.maxSeconds, Math.max(bounds.minSeconds, seconds));
  const targetSamples = Math.max(1, Math.trunc(clamped * bounds.sampleRate));
  return { frames: Math.ceil(targetSamples / bounds.hopLength), targetSamples };
};

/**
 * **The maximum S reachable** from the declared `maxSeconds` (the larger of the 2 paths).
 *
 * The rounding from seconds to S differs per path (the duration path has the clamp maximum
 * `floor(maxSeconds·frameRate)`; the manual-seconds path has
 * `ceil(trunc(maxSeconds·sampleRate) / hopLength)`), so stating the maximum with only one of the
 * formulas misses one frame (measured: with frameRate=25 / hopLength=1920, maxSeconds=30.5 gives
 * duration 762 / manual seconds 763). This is the surface `config.ts` uses to check consistency
 * with `ditSymMax` at parse time, and **the formulas stay canonical in this module** (no copy on
 * the config side).
 */
export const maxSequenceLength = (bounds: SampleBounds): number => {
  const { min, max } = clampRange(bounds);
  // The clamp gives the lower bound priority (with a `min > max` declaration the lower side is
  // returned), so both are checked.
  return Math.max(min, max, sequenceLengthFromSeconds(bounds.maxSeconds, bounds).frames);
};
