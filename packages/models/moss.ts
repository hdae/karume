/**
 * `@karume/models/moss` — the subpath surface of the MOSS-Image (text → image) family (ADR 0138
 * and its addendum of 2026-11-01).
 *
 * ADR 0008: this is an **explicitly designed thin surface**; internal modules are not re-exported
 * wholesale. The surface follows the user's stories — build ({@link Pipeline.fromPretrained}) /
 * generate (`generate` — a prompt string; every knob left out takes the manifest's default) /
 * turn the image into pixels ({@link imageToRgba}) / release (`dispose`).
 *
 * The distribution `karume-moss-image` is **not published** on Hugging Face, so this subpath has
 * no source table (`SOURCES`): pass a local distribution as a source handle (`nodeDirectory` of
 * `@karume/hub/node`, `denoDirectory` of `@karume/hub/deno`). It holds the Qwen3.5 text encoder
 * (about 7.4 GiB of int8 weights), the int8 transformer and the f32 VAE — about 12.3 GiB in all.
 *
 * MUST: every module is free of side effects (no execution at import, no global mutable state —
 * CLAUDE.md). Tree-shaking through the barrel (`mod.ts`) works only on top of this invariant.
 */

export { Pipeline } from "./src/moss/pipeline.ts";
export type {
  FromPretrainedOptions,
  GeneratedImage,
  GenerateRequest,
  PipelineOptions,
} from "./src/moss/pipeline.ts";

/**
 * A generated image (`[3, height, width]` f32 in `[-1, 1]`) → interleaved 8-bit RGBA (can be passed
 * as is to `encodePng`). The package is the source of truth for the uint8 rule
 * (`clamp(x/2 + 0.5, 0, 1)` → `round(·255)`), so consumers do not rewrite it.
 */
export { imageToRgba } from "./src/moss/image.ts";

/**
 * A failure caused by the input (the request itself cannot be accepted = fixing the input makes it
 * pass — a size that is not a multiple of 16 or over the device's area limit, a prompt over the
 * token bound, a knob out of range). **One type across all families**, so a host serving several
 * families can tell 400 from 500 with this alone. Broken internal invariants, asset mismatches and
 * GPU capacity are thrown as plain `Error`s.
 */
export { ModelInputError } from "./src/errors.ts";
