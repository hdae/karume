/**
 * `@karume/models/embeddinggemma` — the subpath surface of the EmbeddingGemma family (text →
 * sentence embedding; today the text model of EmbeddingGemma 2 — distribution
 * `karume-embeddinggemma-2`, pipeline `embeddinggemma/1`).
 *
 * ADR 0008: this is an **explicitly designed thin surface**; internal modules are not re-exported
 * wholesale. The surface follows the user's stories — build ({@link Pipeline.fromPretrained} /
 * {@link Pipeline.fromAssets}) / embed (`embed` — a string or {@link EmbedContentPart}s, with
 * {@link EmbedOptions}: a prompt, a width, an abort signal) / count (`tokenize` — the token ids
 * `embed` would run, with {@link TokenizeOptions}) / release (`dispose`).
 *
 * ```ts
 * const pipeline = await Pipeline.fromPretrained(ref, { gpu });
 * const vector = await pipeline.embed("how do I reset my password", { prompt: "query", dimension: 256 });
 * ```
 *
 * The result is a unit vector, so the dot product of two results is their cosine similarity. The
 * prompt names, the widths and the token limit are the distribution's (`pipelineConfig`); the
 * source of truth for the input rules is the module doc of `src/embeddinggemma/input.ts`.
 *
 * The distribution is not published yet, so this subpath has no `SOURCES` table: a local
 * distribution is passed to `fromPretrained` with `nodeDirectory` of `@karume/hub/node` or
 * `denoDirectory` of `@karume/hub/deno`.
 *
 * MUST: every module is free of side effects (no execution at import, no global mutable state —
 * CLAUDE.md). Tree-shaking through the barrel (`mod.ts`) works only on top of this invariant.
 */

export { Pipeline } from "./src/embeddinggemma/pipeline.ts";
export type {
  Assets,
  FromPretrainedOptions,
  PipelineOptions,
} from "./src/embeddinggemma/pipeline.ts";
export type {
  EmbedContentPart,
  EmbedOptions,
  TokenizeOptions,
} from "./src/embeddinggemma/input.ts";

/**
 * A failure caused by the input (the request itself cannot be accepted = fixing the input makes it
 * pass — an empty or unknown content part, an unknown prompt, an undeclared width, a placeholder
 * token in the text, a text over the token limit). **One type across all families**, so a host
 * serving several families can tell 400 from 500 with this alone. Broken internal invariants, asset
 * mismatches and GPU capacity are thrown as plain `Error`s.
 */
export { ModelInputError } from "./src/errors.ts";
