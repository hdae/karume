/**
 * `@karume/models/sbv2` — the subpath surface of the SBV2 (text → speech) family.
 *
 * ADR 0008: this is an **explicitly designed thin surface**; internal modules are not re-exported
 * wholesale. The surface follows the user's stories — build ({@link Pipeline.fromPretrained} /
 * {@link Pipeline.fromAssets}) / turn into an utterance ({@link toUtterance}) / synthesize
 * (`generate`) / release (`dispose`).
 *
 * **The 0.6.0 re-ruling (breaking)**: text analysis (dictionary, correction dictionary) became the
 * caller's responsibility, and `generate` takes an analysed {@link Utterance} as its first
 * argument. The injection slots of ADR 0072 (`text` / `overlay` / `prosody` / `givenTone` /
 * `analyzeProsody`) and their vocabulary were removed without a shim — since the package has no
 * analyser, the entry point checks only that "the given structure is self-consistent".
 *
 * The accepted sets of style names and speaker names are held by **the distribution's
 * `pipelineConfig`** (ADR 0038 §1), so this family has no "source of truth for spellings" like
 * Anima's `parseResolution`.
 *
 * MUST: every module is free of side effects (no execution at import, no global mutable state —
 * CLAUDE.md). Tree-shaking through the barrel (`mod.ts`) works only on top of this invariant.
 */

export { Pipeline } from "./src/sbv2/pipeline.ts";
export type {
  GeneratedAudio,
  Assets,
  FromPretrainedOptions,
  GenerateOptions,
  PipelineOptions,
  RunComponent,
} from "./src/sbv2/pipeline.ts";

/**
 * Conversion from a phrase-structure analysis to an utterance (a pure function, no GPU needed). It
 * expands the accent nucleus into a binary tone per mora and moves phrase punctuation onto the last
 * mora. **The output of an analyser that returns readings as accent phrases with nuclei**
 * structurally satisfies {@link Phrases}, so it can be passed as is.
 */
export { toUtterance } from "./src/sbv2/text/utterance.ts";

/**
 * The vocabulary of an utterance. `moras` is the editable surface (changing `tone` moves the
 * accent), and `words` is read-only. It has **only the fields SBV2 actually reads** (so that the
 * public surface has no field that can be passed but has no effect).
 */
export type { Mora, Phrases, Utterance, Word } from "./src/sbv2/text/utterance.ts";

/**
 * A failure caused by the input (the request the caller passed cannot be accepted). Broken internal
 * invariants are thrown as plain `Error`s, so an HTTP server can tell 400 from 500 with
 * `instanceof`.
 */
export { InputError } from "./src/sbv2/errors.ts";

/**
 * **The table of sources verified with this package version** (one table per family; keys are the
 * Hugging Face repository name without `karume-`, values are pinned commit SHAs — ADR 0073 / 0092).
 * One entry for now:
 *
 * - `SOURCES["sbv2-jvnv"]` = `hdae/karume-sbv2-jvnv`
 *
 * **An opt-in for callers who want to follow the package version automatically** — pass an entry as
 * is as the first argument of {@link Pipeline.fromPretrained}. To pin reproducibility yourself,
 * write your own `{ repo, revision }` instead of using this table (`fromPretrained` has no default).
 */
export { SOURCES } from "./src/sbv2/config.ts";

/**
 * **The cross-family parent** of input-caused failures ({@link InputError} is a subclass of it).
 * A host serving several families can tell 400 from 500 with this one type, without knowing each
 * family's type.
 */
export { ModelInputError } from "./src/errors.ts";
