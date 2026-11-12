/**
 * `@karume/models/moss` — the subpath surface of the MOSS-Image (text → image) family (ADR 0138
 * and its addendum of 2026-11-01).
 *
 * ADR 0008: this is an **explicitly designed thin surface**; internal modules are not re-exported
 * wholesale. The surface follows the user's stories — build ({@link Pipeline.fromPretrained}) /
 * choose a source ({@link SOURCES}) / generate (`generate` — a prompt string; every knob left out
 * takes the manifest's default) / turn the image into pixels ({@link imageToRgba}) / release
 * (`dispose`) / build Sessions of the family's graphs with the family's fusion rule
 * ({@link EW_CHAIN_RULE}, experimental).
 *
 * The distribution `karume-moss-image` is published on Hugging Face (`"moss-image"` of
 * {@link SOURCES}); a local distribution can be passed as a source handle instead (`nodeDirectory`
 * of `@karume/hub/node`, `denoDirectory` of `@karume/hub/deno`). It holds the Qwen3.5 text encoder
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
 * **The table of sources verified with this package version** (one table per family; keys are the
 * Hugging Face repository name without `karume-`, values are pinned commit SHAs — ADR 0073 / 0092):
 *
 * - `SOURCES["moss-image"]` = `hdae/karume-moss-image` (read by {@link Pipeline.fromPretrained};
 *   self-contained — no cross-repository reference)
 *
 * **An opt-in for callers who want to follow the package version automatically** — pass the entry
 * as is as the first argument of {@link Pipeline.fromPretrained}. To pin reproducibility yourself,
 * write your own `{ repo, revision }` instead of using this table (`fromPretrained` has no
 * default).
 */
export { SOURCES } from "./src/moss/config.ts";

/**
 * A generated image (`[3, height, width]` f32 in `[-1, 1]`) → interleaved 8-bit RGBA (can be passed
 * as is to `encodePng`). The package is the source of truth for the uint8 rule
 * (`clamp(x/2 + 0.5, 0, 1)` → `round(·255)`), so consumers do not rewrite it.
 */
export { imageToRgba } from "./src/moss/image.ts";

/**
 * **Experimental**: the fusion rule `models/ew_chain` ({@link EW_CHAIN_RULE_NAME} — a run of
 * consecutive f32 elementwise nodes in one dispatch, u32-identical to the unfused plan), which
 * {@link Pipeline} passes by default to the Sessions of its image graphs (not the text encoder's).
 * Exported for code that builds its own Sessions from the family's components — to pass it as
 * `SessionOptions.fusionRules`, or to compare a Session with and without it; its name is its
 * counter in `lastRunFusions`. Built with `defineRule` of `@karume/runtime/kernel` (ADR 0148), so
 * it may change in a minor version while the packages are 0.x. Not for
 * `PipelineOptions.fusionRules`: the pipeline already passes it, and a rule given twice is refused.
 */
export { EW_CHAIN_RULE, EW_CHAIN_RULE_NAME } from "./src/moss/ew-chain.ts";

/**
 * A failure caused by the input (the request itself cannot be accepted = fixing the input makes it
 * pass — a size that is not a multiple of 16 or over the device's area limit, a prompt over the
 * token bound, a knob out of range). **One type across all families**, so a host serving several
 * families can tell 400 from 500 with this alone. Broken internal invariants, asset mismatches and
 * GPU capacity are thrown as plain `Error`s.
 */
export { ModelInputError } from "./src/errors.ts";
