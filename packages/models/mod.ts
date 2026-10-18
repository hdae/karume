/**
 * `@karume/models` — the barrel of the pipelines. Offered alongside the per-family subpaths
 * (`./anima` / `./birefnet` / `./depth-anything` / `./gemma` / `./gemma4-qat` / `./irodori` /
 * `./sbv2` / `./siglip2` / `./umt5` / `./vowel-detector` / `./wan`) (ADR 0037).
 *
 * A **thin surface** in the manner of ADR 0008 — it lists only the user's stories, "build a
 * pipeline and generate" and "write out the resulting image"; internal modules are not re-exported
 * wholesale.
 *
 * MUST: every module is free of side effects (no execution at import, no global mutable state, no
 * top-level registration — CLAUDE.md). Importing one export from the barrel drops the other
 * families only while this invariant holds.
 *
 * NOTE: the `format` of the distribution manifest that `fromPretrained` reads is fixed per version,
 * and **older versions are not read**. The "package version ⇔ `format`" table is in the module doc of
 * `@karume/hub` (a table for planning ahead when a distribution has to be rebuilt).
 */

/**
 * The fetch-layer knobs **shared** by the `fromPretrained` option types of every family (`headers` /
 * `onProgress` / `onCacheError` / `onRetry` / `fetch` / `caches`). Each family's type is the
 * intersection of this with its construction options, so there is no point exporting one per
 * family; it is exported once, here — a caller that loads several families with the same fetch
 * settings can give those settings a name (otherwise it would copy the fields from every family's
 * type).
 */
export type { FromPretrainedHubOptions } from "./src/hub/load-options.ts";

/**
 * The **component replacement option** shared by the `fromPretrained` option types of every family
 * (ADR 0108 decision 19 / ADR 0109 decision 10) — role (a weights name in the manifest) → the same
 * role in another `karume/5` repository. Only components whose **graph descriptor has the same
 * sha256** can be replaced, and the identity is checked from the declarations of the two manifests
 * alone, before a single weight byte is fetched.
 */
export type { FromPretrainedComponentOptions } from "./src/hub/load-options.ts";
export type { ComponentSource } from "./src/hub/components.ts";

/**
 * A failure caused by the input (the request itself cannot be accepted = fixing the input makes it
 * pass). **One type across all families**, so a host serving every family can tell 400 from 500
 * with this one type. Broken internal invariants, asset mismatches and GPU capacity are thrown as
 * plain `Error`s (see each subpath).
 */
export { ModelInputError } from "./src/errors.ts";

export { BirefnetPipeline } from "./src/birefnet/pipeline.ts";
export { BIREFNET_SOURCES } from "./src/birefnet/config.ts";
export type {
  BirefnetAssets,
  BirefnetFromPretrainedOptions,
  BirefnetPipelineOptions,
} from "./src/birefnet/pipeline.ts";
export type { AlphaMatte } from "./src/birefnet/matte.ts";

export { DepthAnythingPipeline } from "./src/depth-anything/pipeline.ts";
export type {
  DepthAnythingAssets,
  DepthAnythingFromPretrainedOptions,
  DepthAnythingPipelineOptions,
  DepthMap,
} from "./src/depth-anything/pipeline.ts";
/**
 * The table of sources verified with this package version (see `./depth-anything` — an opt-in for
 * callers who want to follow it). Generations and versions live in separate repositories, so it has
 * one entry for now (ADR 0092).
 */
export { DEPTH_ANYTHING_SOURCES } from "./src/depth-anything/config.ts";

export { AnimaPipeline } from "./src/anima/pipeline.ts";
export type {
  AnimaAssets,
  AnimaFromPretrainedOptions,
  AnimaGenerateEvent,
  AnimaGenerateRequest,
  AnimaLatentSnapshot,
  AnimaPipelineOptions,
  AnimaRunComponent,
  GeneratedImage,
} from "./src/anima/pipeline.ts";
/**
 * The vocabulary of `AnimaGenerateRequest.sampler` (the type and the list of values — see
 * `./anima`).
 */
export type { AnimaSamplerType } from "./src/anima/config.ts";
export { ANIMA_SAMPLER_TYPES } from "./src/anima/config.ts";
/**
 * The vocabulary of `AnimaPipelineOptions.residency` / `AnimaGenerateRequest.residency` and the
 * `residency` event (see `./anima`).
 */
export type {
  AnimaResidency,
  AnimaResidencyAction,
  AnimaResidencyReason,
} from "./src/anima/residency.ts";
/**
 * The spelling of resolutions (`1344x768` / the square shorthand `512`) and the accepted set. **This
 * package is the source of truth for the accepted set** (ADR 0038 §2), so exporting only
 * {@link ImageSize} without these functions would make CLIs / UIs rewrite the `WxH` spelling contract
 * (see `./anima`).
 */
export { formatResolution, parseResolution } from "./src/anima/resolution.ts";
export type { ImageSize } from "./src/anima/resolution.ts";
/**
 * A linear approximation from an intermediate latent to RGB. Pass the two fields returned by
 * `copyLatents()` of `denoise-step` ({@link AnimaLatentSnapshot}) as they are. The VAE can be loaded
 * only **after** the DiT is released (a VRAM MUST), so this is the only way to show per-step
 * progress as a picture — exporting the snapshot type without the approximation would leave
 * subscribers unable to use the latents they receive (see `./anima`).
 */
export { approximatePreview } from "./src/anima/preview.ts";
/**
 * The constants for denormalizing an intermediate latent (`const { mean, std } = animaLatents()` →
 * `denormalizeLatents`). **Not needed for previews** — a surface for consumers that want to handle
 * latents on the same footing as the VAE decode; without it, consumers would keep their own copy of
 * the two constants (see `./anima`).
 */
export { animaLatents, denormalizeLatents } from "./src/anima/latents.ts";
/**
 * The table of sources verified with this package version (see `./anima` — an opt-in for callers
 * who want to follow it). Two entries: the repository of the official variants and the repository
 * of third-party fine-tunes (ADR 0087 / 0092).
 */
export { ANIMA_SOURCES } from "./src/anima/config.ts";

export { IrodoriPipeline } from "./src/irodori/pipeline.ts";
export type {
  GeneratedLatent,
  IrodoriAssets,
  IrodoriFromPretrainedOptions,
  IrodoriGeneratedAudio,
  IrodoriGenerateEvent,
  IrodoriGenerateRequest,
  IrodoriLatentSnapshot,
  IrodoriPipelineOptions,
  IrodoriRunComponent,
  IrodoriSpeakerInput,
} from "./src/irodori/pipeline.ts";
/**
 * The table of sources verified with this package version (see `./irodori` — an opt-in for callers
 * who want to follow it). Upstream v4 / v4.1 are separate repositories, so two entries (the pin of
 * the older version is kept — ADR 0092).
 */
export { IRODORI_SOURCES } from "./src/irodori/config.ts";

export { Sbv2Pipeline } from "./src/sbv2/pipeline.ts";
export type {
  GeneratedAudio,
  Sbv2Assets,
  Sbv2FromPretrainedOptions,
  Sbv2GenerateOptions,
  Sbv2PipelineOptions,
  Sbv2RunComponent,
} from "./src/sbv2/pipeline.ts";
/**
 * Conversion from a phrase-structure analysis to an utterance (see `./sbv2`). Since 0.6.0 text
 * analysis is the caller's responsibility, and `generate` takes the utterance as its first argument
 * (the injection slots of ADR 0072 were removed without a shim).
 */
export { toSbv2Utterance } from "./src/sbv2/text/utterance.ts";
/**
 * The vocabulary of an utterance (mora sequence + word alignment) — only the fields SBV2 reads (see
 * `./sbv2`).
 */
export type { Sbv2Mora, Sbv2Phrases, Sbv2Utterance, Sbv2Word } from "./src/sbv2/text/utterance.ts";
/**
 * SBV2's input-caused failure (a subclass of {@link ModelInputError} — a host serving only SBV2 can
 * branch on this one. Broken internal invariants stay plain `Error`s — see `./sbv2`).
 */
export { Sbv2InputError } from "./src/sbv2/errors.ts";
/**
 * The table of sources verified with this package version (see `./sbv2` — an opt-in for callers who
 * want to follow it).
 */
export { SBV2_SOURCES } from "./src/sbv2/config.ts";

export { Siglip2Pipeline } from "./src/siglip2/pipeline.ts";
export type {
  Siglip2Assets,
  Siglip2FromPretrainedOptions,
  Siglip2PipelineOptions,
} from "./src/siglip2/pipeline.ts";
/**
 * The table of sources verified with this package version (see `./siglip2` — an opt-in for callers
 * who want to follow it). base and so400m share one repository, so one entry (ADR 0092).
 */
export { SIGLIP2_SOURCES } from "./src/siglip2/config.ts";

export { Gemma4Pipeline } from "./src/gemma/pipeline.ts";
/**
 * The default prefill buckets of gemma4 (the value of `Gemma4PipelineOptions.chunkBuckets` when
 * omitted — the ladder of physical chunk row counts allowed in addition to `chunkLength`). The list
 * that runs short utterances without padding, exported as a starting point for callers who add or
 * remove entries. The list was fixed by measurement (see `./gemma`).
 */
export { GEMMA4_CHUNK_BUCKETS } from "./src/gemma/pipeline.ts";
export type {
  Gemma4Assets,
  Gemma4ChatOptions,
  /**
   * The generation shape `estimateSessionMemory` estimates (capacity / chunk length — see
   * `./gemma`).
   */
  Gemma4EstimateOptions,
  Gemma4FromPretrainedOptions,
  Gemma4PipelineOptions,
  /** The specification of `sequence()` (the capacity this conversation allocates). */
  Gemma4SequenceOptions,
} from "./src/gemma/pipeline.ts";
export type {
  /**
   * The stop reason of `chat` (the sequence layer's reasons + `stop-string`, which only this layer
   * can decide).
   */
  Gemma4ChatStop,
  Gemma4ChatStream,
  /** One prefill progress report (`chunk / chunks` — received by `onPrefill`). */
  Gemma4PrefillProgress,
  /**
   * Which run one diagnostics report belongs to (a prefill chunk / a decode step — received by
   * `onRunDiagnostics`).
   */
  Gemma4RunPhase,
} from "./src/gemma/chat-turn.ts";
/**
 * The knobs of the self-tuning gate (`gate` of `Gemma4PipelineOptions.speculative` — static knobs
 * for measurement and acceptance; the defaults are enough. See `./gemma`).
 */
export type { SpeculationGateOptions } from "./src/generation/speculation-gate.ts";
/**
 * Where PLE lives (the values of `Gemma4PipelineOptions.pleResidency` — default `"host"`. `"gpu"` is
 * the opt-in value that keeps the quantized bytes resident on the GPU and also gathers on the GPU.
 * See `./gemma`).
 */
export type { Gemma4PleResidency } from "./src/gemma/ple-gpu.ts";
/**
 * The middle layer that carries a multi-turn conversation (between `chat` and `sequence` — it holds
 * the conversation history, carries the KV over, and truncates a turn that does not fit the
 * capacity before sending it. The default truncation is `dropOldestTurns` = drop the oldest
 * user / assistant pair. See `./gemma`).
 */
export { dropOldestTurns, Gemma4ChatSession } from "./src/gemma/chat-session.ts";
export type {
  Gemma4ChatOverflow,
  Gemma4ChatOverflowPolicy,
  Gemma4ChatSessionHost,
  Gemma4ChatSessionOptions,
  Gemma4ChatTurnOptions,
} from "./src/gemma/chat-session.ts";
export type { Gemma4DefaultSampler, Gemma4PipelineConfig } from "./src/gemma/config.ts";
/**
 * The table of sources verified with this package version (see `./gemma` — an opt-in for callers
 * who want to follow it).
 */
export { GEMMA4_SOURCES } from "./src/gemma/config.ts";
/**
 * A raw declaration (`unknown`) → a validated `Gemma4PipelineConfig`. **An entry point for consumers
 * that use `fromAssets`**; `fromPretrained` runs it internally, so there is no need to call it
 * there (see `./gemma`).
 */
export { parseGemma4PipelineConfig } from "./src/gemma/config.ts";
/**
 * The RoPE parameters declared by the distribution (`pipelineConfig.rope` — the distribution has no
 * table; the host computes cos / sin for one chunk at a time. See `./gemma`).
 */
export type { Gemma4RopeLayerSpec, Gemma4RopeLayerType, Gemma4RopeSpec } from "./src/gemma/rope.ts";
/**
 * Positions → the four RoPE graph inputs (f32 of `[1, rows, headDim]`). `Gemma4Pipeline` runs this
 * internally, so it is needed only by callers that run the production graph themselves (see
 * `./gemma`).
 */
export { gemma4RopeInputs } from "./src/gemma/rope.ts";
/**
 * Conversation → token ids (including `<bos>` — accepts plain conversations only; tools / thinking
 * fail loudly. See `./gemma`). `chat` runs this internally, so it is needed only when driving the
 * low-level surface (`sequence`) yourself.
 */
export { gemma4ChatPrompt } from "./src/gemma/text/chat.ts";
/**
 * One new message → the **delta** tokens appended as the continuation of the conversation (the entry
 * point for the second and later turns when running multiple turns with `sequence`. It does not
 * include the `<turn|>` that closes the previous turn — see `./gemma`).
 */
export { gemma4ChatTurn } from "./src/gemma/text/chat.ts";
export type { Gemma4ChatMessage, Gemma4ChatRole } from "./src/gemma/text/chat.ts";
/**
 * The entry point that turns token ids received on the low-level surface back into strings (see
 * `./gemma`).
 */
export type { GemmaDecodeOptions, GemmaTokenizer } from "./src/gemma/text/tokenizer.ts";
export type { StreamingDetokenizer } from "./src/text/detokenizer.ts";
/**
 * The generation vocabulary (ADR 0083). The types needed to drive `Gemma4Pipeline.sequence()`
 * yourself, and the exception that tells "the conversation does not fit" apart from other failures
 * (decision 10 — see `./gemma`).
 */
export { GenerationCapacityError } from "./src/generation/sequence.ts";
export type {
  GenerationCapacityConstraint,
  GenerationCapacityDetail,
  GenerationEvent,
  /**
   * The gate state carried by one diagnostics report (`GenerationRunPhase.gate` — see `./gemma`).
   */
  GenerationGateTrace,
  GenerationRequest,
  /** One diagnostics report per run (what `Gemma4RunPhase` is — see `./gemma`). */
  GenerationRunPhase,
  GenerationSequence,
  /** The speculation tally (`GenerationStop.speculation` — see `./gemma`). */
  GenerationSpeculation,
  GenerationStop,
  GenerationStream,
} from "./src/generation/sequence.ts";
export type { GenerationProgram } from "./src/generation/program.ts";
export type { SamplerSpec } from "./src/generation/sampler.ts";

export { VowelDetectorPipeline } from "./src/vowel-detector/pipeline.ts";
export type {
  VowelDetectorAssets,
  VowelDetectorFromPretrainedOptions,
  VowelDetectorPipelineOptions,
  VowelDetectorResult,
} from "./src/vowel-detector/pipeline.ts";
export type { LabSegment } from "./src/vowel-detector/postprocess.ts";

export { WanPipeline } from "./src/wan/pipeline.ts";
/**
 * Wan2.2 TI2V 5B: text (+ a conditioning image) → video (see `./wan` — only the request type is
 * specific to 2.2; the other public types are shared with `WanPipeline`).
 */
export { WanTi2vPipeline } from "./src/wan/ti2v-pipeline.ts";
export type { WanTi2vGenerateRequest } from "./src/wan/ti2v-pipeline.ts";
/** The vocabulary of how the I2V conditioning image is fitted to the size (see `./wan`). */
export type { WanI2vFit } from "./src/wan/i2v-preprocess.ts";
export type {
  GeneratedVideo,
  WanAssets,
  WanFromPretrainedOptions,
  WanGenerateEvent,
  WanGenerateRequest,
  WanLatentSnapshot,
  WanPipelineOptions,
  WanRunComponent,
} from "./src/wan/pipeline.ts";
/**
 * The vocabulary of one accepted prompt line and its role (see `./wan` — the `"precomputed"` text
 * encoder accepts only the fixed prompts of the asset; the default `"gpu"` path accepts any prompt).
 */
export type { WanPrompt, WanPromptRole } from "./src/wan/text-embeds.ts";
/**
 * One video frame → 8-bit RGBA (see `./wan` — the source of truth for the uint8 conversion rule).
 */
export { wanFrameToRgba } from "./src/wan/frames.ts";
/**
 * The table of sources verified with this package version (see `./wan` — an opt-in for callers who
 * want to follow it). Only Wan2.2 is published; there is no Wan2.1 key (ADR 0118 addendum
 * (2026-10-15)).
 */
export { WAN_SOURCES } from "./src/wan/config.ts";

/**
 * umT5-XXL: text → the encoder's last hidden state (see `./umt5` — distribution `karume-umt5-xxl`;
 * the Session stays resident from construction until `dispose`. The input rules are the same single
 * implementation as Wan's tokenizer, without Wan's preprocessing).
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
 * The table of sources verified with this package version (see `./umt5` — an opt-in for callers who
 * want to follow it).
 */
export { UMT5_SOURCES } from "./src/umt5/config.ts";

/**
 * The source tables of every family merged into one (**the same keys and the same values** as each
 * family's `*_SOURCES` — ADR 0092). A surface **only in the barrel**, for callers that "walk every
 * published distribution repository" (connectivity smoke tests, prefetching, listings). A consumer
 * that uses only one family should use that family's table from its subpath (using the barrel pulls
 * in the configs of nine families).
 */
export { KARUME_SOURCES } from "./src/sources.ts";

/**
 * RGBA → PNG / f32 waveform ↔ WAV / RGB8 → model input. **Common processing independent of any
 * pipeline** (image generation models generally end with PNG, audio generation models with WAV,
 * voice references come as WAV, and models that take images go through resize → rescale →
 * normalize), so it sits at the top of the barrel rather than in a family subpath.
 */
export { encodePng } from "./src/image/png.ts";
export { normalizeToNchw, resizeRgb8 } from "./src/image/preprocess.ts";
export type { Rgb8Image } from "./src/image/preprocess.ts";
export { decodeWav, encodeWav } from "./src/audio/wav.ts";
export type { DecodedWav } from "./src/audio/wav.ts";

/**
 * NOTE: the fixed-length greedy generation loop (`generateGreedy` / `GreedySpec`) was **removed**
 * from here (ADR 0083 decision 9 — a breaking change; the consumer-facing doc is limitations). It
 * remains in `src/generation/greedy.ts` as an internal helper for parity acceptance; the public
 * generation surface is the `sequence` of the generation-API phase.
 */

// The fixed mobile QAT is a separate family. The shared conversation and configuration types use
// Gemma's public surface.
export { Gemma4QatPipeline } from "./src/gemma/pipeline.ts";
export type {
  Gemma4QatFromPretrainedOptions,
  Gemma4QatPipelineOptions,
} from "./src/gemma/pipeline.ts";
export { gemma4QatRopeInputs } from "./src/gemma/rope.ts";
