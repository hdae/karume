/**
 * `@karume/models/vowel-detector` — the subpath surface of the vowel detection family (Japanese
 * speech → a `.lab` for lip sync).
 *
 * ADR 0008: this is an **explicitly designed thin surface**; internal modules are not re-exported
 * wholesale. The surface follows the user's stories — build
 * ({@link Pipeline.fromPretrained} / {@link Pipeline.fromAssets}) / detect
 * (`detect`) / release (`dispose`).
 *
 * Decoding and resampling the WAV are the user's responsibility (the input is a 16 kHz mono
 * `Float32Array` — the module doc of `src/vowel-detector/pipeline.ts`). `decodeWav` is exported
 * from the barrel.
 *
 * MUST: every module is free of side effects (no execution at import, no global mutable state —
 * CLAUDE.md). Tree-shaking through the barrel (`mod.ts`) works only on top of this invariant.
 */

export { Pipeline } from "./src/vowel-detector/pipeline.ts";
export type {
  Assets,
  FromPretrainedOptions,
  PipelineOptions,
  Result,
} from "./src/vowel-detector/pipeline.ts";
export type { LabSegment } from "./src/vowel-detector/postprocess.ts";

/**
 * A failure caused by the input (the request itself cannot be accepted = fixing the input makes it
 * pass — for this family, the length of the waveform). **One type across all families**, so a host
 * serving several families can tell 400 from 500 with this alone. Broken internal invariants, asset
 * mismatches and GPU capacity are thrown as plain `Error`s.
 */
export { ModelInputError } from "./src/errors.ts";
