/**
 * Durations (front's logw → frames per phoneme → frame expansion index).
 *
 * A port of the reference implementation `SynthesizerTrnJPExtra.infer`'s
 * `logw = sdp(...)·r + dp(...)·(1−r)` → `w = exp(logw)·x_mask·length_scale` →
 * `w_ceil = ceil(w)` → `attn = generate_path(w_ceil, ...)`. **The attn matrix (Ty×P) is not built**
 * — the path is monotonic, so `expandIdx` (a phoneme index sequence of length Ty) holds the same
 * information. The following expansion of m_p / logs_p becomes a gather instead of a matrix product
 * (the host's responsibility per ADR 0013).
 *
 * ## Known difference: `w_ceil` is not bit-identical to torch (a note in the same form as ADR 0048)
 *
 * Here the whole expression is evaluated in JS f64 before `Math.ceil`, whereas the reference
 * applies `torch.ceil` while still in f32 tensor arithmetic. When the f32 half ulp (relative 6e-8)
 * between `f32(exp_f32(x))` and `exp_f64(x)` crosses the threshold of ceil, it **shifts by 1
 * frame** (measured: 5.5e-7 per phoneme, P ≈ 1.3e-4 for one utterance of 229 phonemes).
 *
 * MUST: aligning this to sequential `Math.fround` **does not make the torch match deterministic** —
 * the GPU/CPU difference of 1e-5 in the front output itself causes the same threshold crossing at a
 * rate more than 2 orders of magnitude higher, and the dominant term remains upstream
 * (`tools/export-recipes/sbv2/README.md` explicitly allows it by design and puts the positions that
 * split into `w_ceil_diffs`). karume's own reproducibility (the WAV sha256 check) is intact because
 * the f64 path is deterministic.
 */

import { Sbv2InputError } from "../errors.ts";

type DurationPlan = {
  /** Frames per phoneme (after ceil and masking). */
  readonly wCeil: Int32Array<ArrayBuffer>;
  /** Total frame count Ty. */
  readonly totalFrames: number;
  /** Frame → phoneme expansion index (length Ty). */
  readonly expandIdx: Int32Array<ArrayBuffer>;
};

/**
 * Builds per-phoneme durations and the frame expansion index from logw (sdp / dp).
 *
 * @param logwSdp front's output 0 (raw data of `[1,1,P]`).
 * @param logwDp front's output 1 (same).
 * @param xMask front's input x_mask (`[1,1,P]`). Drops padding columns to 0 frames.
 * @param maxFrames upper limit of the total frame count (the distribution's
 * `pipelineConfig.maxFrames`).
 *
 * MUST: place the limit check **before** allocating `expandIdx`. The total frame count is a
 * request-dependent number set by the product of logw and lengthScale, and can be so large that not
 * even the expansion sequence of length T, let alone the downstream `(T, T)` tables (8·T² bytes
 * class), can be allocated — checking after allocating turns it into an OOM far from its cause.
 */
export const durationsToFrames = (
  logwSdp: Float32Array<ArrayBuffer>,
  logwDp: Float32Array<ArrayBuffer>,
  xMask: Float32Array<ArrayBuffer>,
  sdpRatio: number,
  lengthScale: number,
  maxFrames: number,
): DurationPlan => {
  const phonemes = logwSdp.length;
  if (logwDp.length !== phonemes || xMask.length !== phonemes) {
    throw new Error(
      `長さ不一致 logw_sdp=${phonemes} logw_dp=${logwDp.length} x_mask=${xMask.length}`,
    );
  }
  const wCeil = new Int32Array(phonemes);
  let total = 0;
  for (let i = 0; i < phonemes; i += 1) {
    const logw = logwSdp[i] * sdpRatio + logwDp[i] * (1 - sdpRatio);
    const frames = Math.ceil(Math.exp(logw) * xMask[i] * lengthScale);
    if (!Number.isFinite(frames) || frames < 0) {
      // Passing a diverged logw (broken front values) downstream as "a huge Ty" makes only the
      // allocation size abnormal and turns it into an OOM far from its cause. Fail here.
      throw new Error(`音素 ${i} の継続長 ${frames} が有限の非負値でない`);
    }
    wCeil[i] = frames;
    total += frames;
  }
  if (total < 1) {
    throw new Error("総フレーム数が 0（発話にならない — x_mask か front 出力を疑う）");
  }
  if (total > maxFrames) {
    // MUST: only this one is a `Sbv2InputError`. The other throws in this file mean the front
    // output is broken = an internal fault (500), but exceeding the limit means "the given text /
    // lengthScale cannot be accepted" = input-caused (400), which the caller fixes by fixing the
    // request (the classification axis of `errors.ts`).
    throw new Sbv2InputError(
      `総フレーム数 ${total} が配布形の上限 maxFrames=${maxFrames} を超えている` +
        "（text を短く分けるか lengthScale を下げる）",
    );
  }
  const expandIdx = new Int32Array(total);
  let position = 0;
  for (let i = 0; i < phonemes; i += 1) {
    expandIdx.fill(i, position, position + wCeil[i]);
    position += wCeil[i];
  }
  return { wCeil, totalFrames: total, expandIdx };
};
