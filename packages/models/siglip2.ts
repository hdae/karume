/**
 * `@karume/models/siglip2` — the subpath surface of the SigLIP2 (image → embedding) family.
 *
 * ADR 0008: this is an **explicitly designed thin surface**; internal modules are not re-exported
 * wholesale. The surface follows the user's stories — build ({@link Pipeline.fromPretrained}
 * / {@link Pipeline.fromAssets}) / choose a source ({@link SOURCES}) / embed
 * (`embed`) / release (`dispose`).
 *
 * Decoding the pixels is the user's responsibility; resize / rescale / normalize are applied inside
 * `embed` with the distribution's constants (the module doc of `src/siglip2/pipeline.ts`). Only
 * {@link Rgb8Image} is re-exported, because it is the argument type of `embed` and must be nameable
 * from this subpath.
 *
 * MUST: every module is free of side effects (no execution at import, no global mutable state —
 * CLAUDE.md). Tree-shaking through the barrel (`mod.ts`) works only on top of this invariant.
 */

export { Pipeline } from "./src/siglip2/pipeline.ts";
export type { Assets, FromPretrainedOptions, PipelineOptions } from "./src/siglip2/pipeline.ts";
/**
 * **The table of sources verified with this package version** (one table per family; keys are the
 * Hugging Face repository name without `karume-`, values are pinned commit SHAs — ADR 0073 / 0092).
 * One family = one repository, so one entry, holding two models that differ only in size:
 *
 * - `SOURCES["siglip2"]` = `hdae/karume-siglip2` (base 224 / so400m 384; default = base)
 *
 * For the non-default model write `fromPretrained(SOURCES["siglip2"], { model: "so400m" })`
 * ({@link PipelineOptions}).
 *
 * **An opt-in for callers who want to follow the package version automatically** — pass an entry as
 * is as the first argument of {@link Pipeline.fromPretrained}. To pin reproducibility
 * yourself, write your own `{ repo, revision }` instead of using this table (`fromPretrained` has no
 * default).
 */
export { SOURCES } from "./src/siglip2/config.ts";
export type { Rgb8Image } from "./src/image/preprocess.ts";

/**
 * A failure caused by the input (the request itself cannot be accepted = fixing the input makes it
 * pass — the size of the input image and the length of its pixel array). **One type across all
 * families**, so a host serving several families can tell 400 from 500 with this alone. Broken
 * internal invariants, asset mismatches and GPU capacity are thrown as plain `Error`s.
 */
export { ModelInputError } from "./src/errors.ts";
