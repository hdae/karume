/**
 * `@karume/models/irodori` — the subpath surface of the Irodori (text → speech) family.
 *
 * ADR 0008: this is an **explicitly designed thin surface**; internal modules are not re-exported
 * wholesale. The surface follows the user's stories — build ({@link Pipeline.fromPretrained}
 * / {@link Pipeline.fromAssets}) / generate (`generate` / `generateLatent`) / subscribe to
 * generation progress (`onEvent` — {@link GenerateEvent}) / release (`dispose`).
 *
 * Waveforms go in and out as f32 (`encodeWav`, which writes WAV, and `decodeWav`, which reads
 * reference audio, are in the barrel — both are common processing independent of any family).
 *
 * The same {@link Pipeline} runs the rectified-flow models (v4 / v4.1 Small and v4-Large — Euler
 * steps with classifier-free guidance) and the MeanFlow student of v4.1 Small (v4.1 Small MF — 4
 * steps, one forward each, no guidance at run time); the distribution's `flowParameterization`
 * picks the sampler (ADR 0150). v4-Large differs in its dimensions, which the distribution declares,
 * and in its text tokenizer (Gemma BPE instead of Unigram), which its tokenizer asset declares.
 *
 * MUST: every module is free of side effects (no execution at import, no global mutable state —
 * CLAUDE.md). Tree-shaking through the barrel (`mod.ts`) works only on top of this invariant.
 */

export { Pipeline } from "./src/irodori/pipeline.ts";
export type {
  Assets,
  FromPretrainedOptions,
  GeneratedAudio,
  GeneratedLatent,
  GenerateEvent,
  GenerateRequest,
  LatentSnapshot,
  PipelineOptions,
  RunComponent,
  SpeakerInput,
} from "./src/irodori/pipeline.ts";

/**
 * **The table of sources verified with this package version** (one table per family; keys are the
 * Hugging Face repository name without `karume-`, values are pinned commit SHAs — ADR 0073 / 0092).
 * Upstream v4 / v4.1 / v4.1 MF / v4-Large are separate repositories, so four entries:
 *
 * - `SOURCES["irodori-v4-small"]` = `hdae/karume-irodori-v4-small`
 * - `SOURCES["irodori-v4.1-small"]` = `hdae/karume-irodori-v4.1-small` (the follow-up
 *   version in which only the duration predictor was retrained)
 * - `SOURCES["irodori-v4.1-small-mf"]` = `hdae/karume-irodori-v4.1-small-mf` (upstream's MeanFlow
 *   student of v4.1 — 4 steps without CFG; ADR 0150)
 * - `SOURCES["irodori-v4-large"]` = `hdae/karume-irodori-v4-large` (upstream's v4-Large — the
 *   T5Gemma 2 text encoder and the Gemma tokenizer asset, under the Gemma Terms of Use; ADR 0151)
 *
 * The pin of the older version is kept too (**there is no default entry** — the caller spells out
 * which one to use).
 *
 * **An opt-in for callers who want to follow the package version automatically** — pass an entry as
 * is as the first argument of {@link Pipeline.fromPretrained}. To pin reproducibility
 * yourself, write your own `{ repo, revision }` instead of using this table (`fromPretrained` has no
 * default).
 */
export { SOURCES } from "./src/irodori/config.ts";

/**
 * A failure caused by the input (the request itself cannot be accepted = fixing the input makes it
 * pass — the range of `seed`, the speaker specification, the length specification). **One type
 * across all families**, so a host serving several families can tell 400 from 500 with this alone.
 * Broken internal invariants, asset mismatches and GPU capacity are thrown as plain `Error`s.
 */
export { ModelInputError } from "./src/errors.ts";
