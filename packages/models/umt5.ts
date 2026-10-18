/**
 * `@karume/models/umt5` — the subpath surface of umT5-XXL (text → the encoder's last hidden state)
 * (distribution `karume-umt5-xxl` — pipeline `umt5-encoder/1`).
 *
 * ADR 0008: this is an **explicitly designed thin surface**; internal modules are not re-exported
 * wholesale. The surface follows the user's stories — build ({@link Umt5Encoder.fromPretrained} /
 * {@link Umt5Encoder.fromAssets}) / choose a source ({@link UMT5_SOURCES}) / run only the tokenizer
 * (`tokenize`) / encode (`encode` — {@link Umt5Encoding}) / abort (the `signal` of `encode`) /
 * release (`dispose`).
 *
 * The source of truth for the input rules (no preprocessing; rejects out-of-vocabulary characters,
 * special tokens in the text, `▁` right after whitespace, whitespace other than U+0020 and trailing
 * whitespace, more than 512 tokens, and the empty string) is the module doc of
 * `src/umt5/tokenizer.ts`. Wan's `WanPipeline` / `WanTi2vPipeline` run the upstream preprocessing
 * (`prompt_clean`) before the same tokenizer.
 *
 * A local distribution is passed to `fromPretrained` with `nodeDirectory` of `@karume/hub/node` or
 * `denoDirectory` of `@karume/hub/deno`.
 *
 * MUST: every module is free of side effects (no execution at import, no global mutable state —
 * CLAUDE.md). Tree-shaking through the barrel (`mod.ts`) works only on top of this invariant.
 */

export { Umt5Encoder } from "./src/umt5/encoder.ts";
export type {
  Umt5EncodeOptions,
  Umt5EncoderAssets,
  Umt5EncoderFromPretrainedOptions,
  Umt5EncoderOptions,
  Umt5Encoding,
} from "./src/umt5/encoder.ts";
/**
 * **The table of sources verified with this package version** (one table per family; keys are the
 * Hugging Face repository name without `karume-`, values are pinned commit SHAs — ADR 0073 / 0092).
 * One family = one repository, so one entry:
 *
 * - `UMT5_SOURCES["umt5-xxl"]` = `hdae/karume-umt5-xxl` (the encoder of model `"xxl"` and the
 *   tokenizer assets)
 *
 * **An opt-in for callers who want to follow the package version automatically** — pass an entry as
 * is as the first argument of {@link Umt5Encoder.fromPretrained}. To pin reproducibility yourself,
 * write your own `{ repo, revision }` instead of using this table (`fromPretrained` has no default).
 */
export { UMT5_SOURCES } from "./src/umt5/config.ts";

/**
 * A failure caused by the input (the request itself cannot be accepted = fixing the input makes it
 * pass — a string the tokenizer rejects, with a message that says how to fix it). **One type across
 * all families**, so a host serving several families can tell 400 from 500 with this alone. Broken
 * internal invariants, asset mismatches and GPU capacity are thrown as plain `Error`s.
 */
export { ModelInputError } from "./src/errors.ts";
