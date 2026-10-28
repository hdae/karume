/**
 * `@karume/models/depth-anything` — the subpath surface of the Depth Anything V2 family (image →
 * relative depth map).
 *
 * ADR 0008: this is an **explicitly designed thin surface**; internal modules are not re-exported
 * wholesale. The surface follows the user's stories — build
 * ({@link Pipeline.fromPretrained} / {@link Pipeline.fromAssets}) / choose
 * a source ({@link SOURCES}) / estimate depth (`estimate`) / release (`dispose`).
 *
 * Decoding the pixels is the user's responsibility; resize / rescale / normalize are applied inside
 * `estimate` with the distribution's constants. The result is **raw f32 relative depth with the same
 * size as the input**; normalizing to `[0, 1]` and colouring are up to the user (they are
 * visualization decisions, not part of inference — the module doc of
 * `src/depth-anything/pipeline.ts`). {@link Rgb8Image} and {@link DepthMap} are the argument and
 * return types of `estimate`, so they are exported here as well to be nameable from this subpath.
 *
 * MUST: every module is free of side effects (no execution at import, no global mutable state —
 * CLAUDE.md). Tree-shaking through the barrel (`mod.ts`) works only on top of this invariant.
 */

export { Pipeline } from "./src/depth-anything/pipeline.ts";
export type {
  Assets,
  FromPretrainedOptions,
  PipelineOptions,
  DepthMap,
} from "./src/depth-anything/pipeline.ts";
/**
 * **The table of sources verified with this package version** (one table per family; keys are the
 * Hugging Face repository name without `karume-`, values are pinned commit SHAs — ADR 0073 / 0092).
 * Generations and versions are separate repositories = separate entries (the generation is part of
 * the repository name), so one entry for now:
 *
 * - `SOURCES["depth-anything-v2"]` = `hdae/karume-depth-anything-v2`
 *   (default = Small)
 *
 * **An opt-in for callers who want to follow the package version automatically** — pass an entry as
 * is as the first argument of {@link Pipeline.fromPretrained}. To pin reproducibility
 * yourself, write your own `{ repo, revision }` instead of using this table (`fromPretrained` has no
 * default).
 */
export { SOURCES } from "./src/depth-anything/config.ts";
export type { Rgb8Image } from "./src/image/preprocess.ts";

/**
 * A failure caused by the input (the request itself cannot be accepted = fixing the input makes it
 * pass — the size of the input image and the length of its pixel array). **One type across all
 * families**, so a host serving several families can tell 400 from 500 with this alone. Broken
 * internal invariants, asset mismatches and GPU capacity are thrown as plain `Error`s.
 */
export { ModelInputError } from "./src/errors.ts";
