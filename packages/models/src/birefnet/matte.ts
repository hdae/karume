/**
 * Alpha matte post-processing for the BiRefNet family (the graph's logit map → an 8-bit α of the
 * same size as the input image).
 *
 * Step 4 of the module doc of `pipeline.ts`. It is a pure function that touches neither the GPU nor
 * the graph, so it lives outside the pipeline and is pinned directly by a CPU-only test
 * (`tests/birefnet_matte_test.ts` — checked against the goldens made with the upstream functions,
 * `tests/fixtures/birefnet-matte/`).
 *
 * ## The actual upstream procedures (they differ per checkpoint)
 *
 * - **BiRefNet_HR** (bundled `handler.py:131-137`; the model card's usage example has the same
 *   order): logits from the half-precision model → half-precision `sigmoid` →
 *   `transforms.ToPILImage()` (torchvision 0.28 does `(x * 255).astype(np.uint8)` = truncation) →
 *   Pillow's `pred_pil.resize(image.size)` (resample omitted = BICUBIC, 8-bit one band).
 * - **Lucida** (model card `README.md:86-89`): f32 logits → `sigmoid` →
 *   `transforms.functional.resize` (tensor branch, bilinear, antialias by default) →
 *   `(alpha.numpy() * 255).astype("uint8")` = truncation.
 *
 * ## Procedures ({@link MatteProcedure} — ADR 0125)
 *
 * Upstream parity is reached at the level of computed values (or of their intent); upstream errors
 * and choices made for speed are not copied (ADR 0125 decision 1). What is copied per checkpoint is
 * the interpolation kernel and the order of stages; what is not copied is the truncation (every
 * procedure rounds = `Math.round(255·α)`, 0.5 rounds up) and HR's half precision.
 *
 * - `float-bilinear` (Lucida's upstream intent = the pipeline's path itself before procedures were
 *   chosen by declaration): σ computed in f64 and stored in f32 → `resizePlaneF32` (antialiased
 *   bilinear, f64, no rounding between passes) → `Math.round(255·α)`.
 * - `u8-pil-bicubic` (HR's upstream intent): σ computed in f64 → `Math.round(255σ)` into an 8-bit
 *   plane → Pillow 12.3.0's 8-bit one-band BICUBIC (`src/image/pillow-resample.ts` — 22-bit fixed
 *   point, uint8 intermediate, a pass is skipped on an axis whose size is unchanged).
 *
 * Which procedure applies is declared by the distribution's `pipelineConfig.postprocess`
 * (`./config.ts` — the build step decides it from the checkpoint: HR → `u8-pil-bicubic`, Lucida →
 * `float-bilinear`). The blind A/B of HR's two procedures answered "indistinguishable" for all 8
 * pairs, and the upstream procedure was adopted as the rule set in advance required (ADR 0125,
 * "Results (addendum 2026-10-11) — A/B answers, rulings and the distribution declaration"). The
 * procedure argument receives the declaration; it is not a public option an application chooses
 * (not exposed on `mod.ts` / the subpath surface — ADR 0008).
 */

import { resizePlaneF32 } from "../image/preprocess.ts";
import { resizeU8PlaneBicubic } from "../image/pillow-resample.ts";

/** Maximum value of the 8-bit α. */
const ALPHA_MAX = 255;

/** An alpha matte of the same size as the input (one byte per pixel, row-major). */
export type AlphaMatte = {
  readonly data: Uint8Array<ArrayBuffer>;
  readonly width: number;
  readonly height: number;
};

/** Names of the post-processing procedures (module doc "Procedures" — ADR 0125 owns the names). */
export const MATTE_PROCEDURES = ["float-bilinear", "u8-pil-bicubic"] as const;

/** One of {@link MATTE_PROCEDURES}. */
export type MatteProcedure = (typeof MATTE_PROCEDURES)[number];

/** The body of one procedure (logit map → 8-bit α pixel row of the same size as the input). */
type ProcedureBody = (
  logits: Float32Array<ArrayBuffer>,
  sourceWidth: number,
  sourceHeight: number,
  width: number,
  height: number,
) => Uint8Array<ArrayBuffer>;

/**
 * σ of one logit (f64).
 *
 * MUST: never return a non-finite value silently. Quantizing α (assigning into a `Uint8Array`)
 * turns NaN into 0 = "a fully transparent pixel", **a matte that looks normal**. Resizing is a
 * weighted sum that merely propagates NaN, so the check sits here, closest to the source (the same
 * check point as `anima/image.ts`).
 */
const sigmoidAt = (
  logits: Float32Array<ArrayBuffer>,
  index: number,
  sourceWidth: number,
): number => {
  const raw = logits[index];
  if (!Number.isFinite(raw)) {
    const x = index % sourceWidth;
    const y = Math.floor(index / sourceWidth);
    throw new Error(`birefnet: マットの logit (x=${x}, y=${y}) が非有限値`);
  }
  return 1 / (1 + Math.exp(-raw));
};

/** `float-bilinear` (module doc — Lucida's upstream intent). */
const floatBilinear: ProcedureBody = (logits, sourceWidth, sourceHeight, width, height) => {
  const alpha = new Float32Array(logits.length);
  for (let index = 0; index < logits.length; index += 1) {
    alpha[index] = sigmoidAt(logits, index, sourceWidth);
  }
  const scaled = resizePlaneF32(alpha, sourceWidth, sourceHeight, width, height);
  const data = new Uint8Array(scaled.length) as Uint8Array<ArrayBuffer>;
  for (let index = 0; index < scaled.length; index += 1) {
    data[index] = Math.round(scaled[index] * ALPHA_MAX);
  }
  return data;
};

/**
 * `u8-pil-bicubic` (module doc — HR's upstream intent).
 *
 * MUST: do not store σ in f32 before rounding (unlike `float-bilinear`, the f64 σ goes straight
 * into `Math.round` — the same formula as the golden generator
 * (`tools/export-recipes/birefnet/matte_fixture.py`). Storing in f32 can flip the rounding
 * direction of samples near `k + 0.5`).
 */
const u8PilBicubic: ProcedureBody = (logits, sourceWidth, sourceHeight, width, height) => {
  const quantized = new Uint8Array(logits.length);
  for (let index = 0; index < logits.length; index += 1) {
    quantized[index] = Math.round(sigmoidAt(logits, index, sourceWidth) * ALPHA_MAX);
  }
  return resizeU8PlaneBicubic(quantized, sourceWidth, sourceHeight, width, height);
};

const PROCEDURES: Readonly<Record<MatteProcedure, ProcedureBody>> = {
  "float-bilinear": floatBilinear,
  "u8-pil-bicubic": u8PilBicubic,
};

/**
 * The graph's logit map → an 8-bit α of the same size as the input (step 4 of the module doc of
 * `pipeline.ts`).
 *
 * `procedure` takes the distribution's declaration (`pipelineConfig.postprocess`) as is. It has no
 * default so that a caller who forgot to pass the declaration does not silently run with one
 * checkpoint's procedure. Dropping to 8 bit is **a decision about this distribution's output
 * format**, and every procedure quantizes by rounding (ADR 0125 — upstream truncates for both
 * checkpoints).
 *
 * NOTE: the `export` serves the pipeline, the tests and the A/B tool (`tools/birefnet-ab/`, removed
 * in 0.15) (not exposed on `mod.ts` / the subpath surface — ADR 0008).
 */
export const matteFromLogits = (
  logits: Float32Array<ArrayBuffer>,
  sourceWidth: number,
  sourceHeight: number,
  width: number,
  height: number,
  procedure: MatteProcedure,
): AlphaMatte => {
  // MUST: look up with `Object.hasOwn` — a JS caller passing `"toString"` etc. would otherwise get
  // an Object.prototype member accepted as a procedure. A misspelled name, so a plain `Error`
  // (ADR 0107 decision 3).
  if (!Object.hasOwn(PROCEDURES, procedure)) {
    throw new Error(
      `birefnet: 後処理の手順 ${JSON.stringify(procedure)} は未対応（期待 ${MATTE_PROCEDURES.map(
        (name) => `'${name}'`,
      ).join(" / ")}）`,
    );
  }
  return {
    data: PROCEDURES[procedure](logits, sourceWidth, sourceHeight, width, height),
    width,
    height,
  };
};
