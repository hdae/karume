/**
 * `@karume/models/birefnet` — the subpath surface of the BiRefNet family (image → alpha matte /
 * background removal).
 *
 * ADR 0008: this is an **explicitly designed thin surface**; internal modules are not re-exported
 * wholesale. The surface follows the user's stories — build ({@link Pipeline.fromPretrained}
 * / {@link Pipeline.fromAssets}) / cut out (`segment`) / release (`dispose`).
 *
 * Decoding the pixels is the user's responsibility; resize / rescale / normalize are applied inside
 * `segment` with the distribution's constants. The result is one **8-bit alpha with the same size as
 * the input**; compositing into RGBA is up to the user (the policy depends on the use — the module
 * doc of `src/birefnet/pipeline.ts`). {@link Rgb8Image} and {@link AlphaMatte} are the argument and
 * return types of `segment`, so they are exported here as well to be nameable from this subpath.
 *
 * MUST: every module is free of side effects (no execution at import, no global mutable state —
 * CLAUDE.md). Tree-shaking through the barrel (`mod.ts`) works only on top of this invariant.
 */

export { Pipeline } from "./src/birefnet/pipeline.ts";
export { SOURCES } from "./src/birefnet/config.ts";
export type { Assets, FromPretrainedOptions, PipelineOptions } from "./src/birefnet/pipeline.ts";
export type { AlphaMatte } from "./src/birefnet/matte.ts";
export type { Rgb8Image } from "./src/image/preprocess.ts";

/**
 * A failure caused by the input (the request itself cannot be accepted = fixing the input makes it
 * pass — the size of the input image and the length of its pixel array). **One type across all
 * families**, so a host serving several families can tell 400 from 500 with this alone. Broken
 * internal invariants, asset mismatches and GPU capacity are thrown as plain `Error`s.
 */
export { ModelInputError } from "./src/errors.ts";
