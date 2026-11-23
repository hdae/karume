/**
 * `@karume/models/embeddinggemma` — the subpath surface of the EmbeddingGemma family (text,
 * images and audio → sentence embedding; today EmbeddingGemma 2 — distribution
 * `karume-embeddinggemma-2`, pipeline `embeddinggemma/1`, its vision tower built with
 * `vision: true`, its audio tower with `audio: true`).
 *
 * ADR 0008: this is an **explicitly designed thin surface**; internal modules are not re-exported
 * wholesale. The surface follows the user's stories — build ({@link Pipeline.fromPretrained} /
 * {@link Pipeline.fromAssets}, `vision` for image input, `audio` for audio input) / embed (`embed`
 * — a string or {@link EmbedContentPart}s — text, decoded {@link Rgb8Image}s and 16 kHz
 * {@link MonoAudio}s —, with {@link EmbedOptions}: a prompt, a width, an abort signal) / count
 * (`tokenize` — the token ids `embed` would run, with {@link TokenizeOptions}) / release
 * (`dispose`).
 *
 * ```ts
 * const pipeline = await Pipeline.fromPretrained(ref, { gpu, vision: true, audio: true });
 * const vector = await pipeline.embed("how do I reset my password", { prompt: "query", dimension: 256 });
 * const picture = await pipeline.embed([{ type: "image", image }]); // image: { data, width, height } RGB8
 * const clip = await pipeline.embed([{ type: "audio", audio }]); // audio: { data, sampleRate: 16000 }
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
/** Decoded RGB8 pixels (row-major, 3 bytes per pixel) and their size — what an image part holds. */
export type { Rgb8Image } from "./src/image/preprocess.ts";
/**
 * Mono f32 samples and their sample rate (16,000 Hz, at most 30 s) — what an audio part holds; the
 * shape of `decodeWav`'s result.
 */
export type { MonoAudio } from "./src/embeddinggemma/audio.ts";

/**
 * A failure caused by the input (the request itself cannot be accepted = fixing the input makes it
 * pass — an empty or unknown content part, an image part without the vision tower, an audio part
 * without the audio tower, an image that is not RGB8, an audio that is not 16 kHz, longer than 30 s
 * or too short for one frame, an unknown prompt, an undeclared width, a placeholder token in the
 * text, an input over the token limit). **One type across all families**, so a host
 * serving several families can tell 400 from 500 with this alone. Broken internal invariants, asset
 * mismatches and GPU capacity are thrown as plain `Error`s.
 */
export { ModelInputError } from "./src/errors.ts";
