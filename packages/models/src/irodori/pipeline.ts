/**
 * `IrodoriPipeline` — one surface from text (+ caption + reference speaker) to a **waveform**.
 *
 * It has two exits: {@link IrodoriPipeline.generate} (waveform) and
 * {@link IrodoriPipeline.generateLatent} (patched latent `[S,32]` — the codec does not run).
 * The latter is the basis of the latent check (`e2e_irodori_latent_test.ts`) and the entry point
 * for uses that need only the embedding.
 *
 * The reference speaker can be given in three ways (`IrodoriSpeakerInput`) — the reference audio
 * itself (`audio` — stage 0 below) / its DACVAE latent (`latent`) / a ready-made speaker state
 * (`stateOverride`).
 *
 * Pipeline (every NN stage runs on Karume — no torch):
 *
 * 0. Host: truncate the reference audio to 120 s → LUFS −16 normalization → reflect pad →
 *    `codec_encoder` to a latent (only when `audio` is given — `host/reference.ts`)
 * 1. Host: `normalize_text` → prepend BOS → a **packed** token sequence (static scheme — not called
 *    with padding)
 * 2. `backbone` (ModernBERT) twice, for text and caption (**in the same single session**)
 * 3. `text-proj` → text condition / `caption-proj` (2 outputs) → caption condition + `caption_vec`
 * 4. Host: patch the reference latent → `speaker` → **prepend the mean token** (outside the graph)
 * 5. `duration` → log frames → the host decides S (expm1 → banker's rounding → clamp)
 * 6. Host: right-pad the condition states to Tmax + build the segment mask
 * 6'. Run `dit_context` once per generation (context K/V projection — copies each block's context
 *    K / V into resident tensors, and the Session is closed before `dit` opens. DECIDED: ADR 0114)
 * 7. `dit` runs 40 to 100 forwards in one session (it takes the K / V of 6' as resident inputs. The
 *    CFG combination and Euler are two small GPU-resident graphs — the whole loop is one batch, and
 *    the only trip to the host is the final latent, once) → latent
 * 8. Host: decide the tail-trim position **on z** (`host/trim.ts`)
 * 9. `codec_decoder` runs over the tiles in one session (`codec.ts`) → full-length waveform
 * 10. Host: cut at sample granularity to the shorter of the requested seconds / the tail trim
 *
 * ## MUST: graphs are opened and closed per stage (only `dit` and `codec_decoder` run several times)
 *
 * {@link IrodoriPipeline.fromAssets} **opens no Session at all** — it goes as far as opening the
 * containers (`openContainer` = parsing the 2 documents only), and GPU residency is set up and torn
 * down stage by stage inside {@link IrodoriPipeline.generate}. `backbone` alone is 1.26GB, so the
 * condition encoders and the DiT are never alive at the same time. For the same reason the codec is
 * opened after the DiT is closed. `dit_context` is also closed **before** `dit` is opened (the
 * weights and output slots of the condition-side linears do not stay during the DiT loop — only the
 * K / V copied into resident tensors remain). Only the DiT step opens, besides `dit`, two small
 * graphs built on the host (`runDitLoopResident` in {@link "./dit-loop.ts"}) at the same time, but
 * they are 5 nodes without weights, so they do not matter for VRAM.
 *
 * MUST: this sequencing is kept **on the public API side too** — `generate` / `generateLatent` go
 * on a serialization chain (concurrent calls wait and run in order), and `dispose` waits for its
 * completion before destroying the GPU. Without the chain, the graphs of two concurrent calls are
 * resident at the same time, and a dispose during generation breaks flush-before-destroy.
 *
 * ## NOTE: `codec_encoder` is not tiled (asymmetric with the decoder)
 *
 * The reference audio goes through in one encode. The encoder's intermediate tensors reach
 * 1.47GB × 2 at 120 s (T = 3000), so **on a machine whose `maxStorageBufferBindingSize` is only
 * the default 128MiB, a long reference fails to allocate** (there is no counterpart of the
 * decoder's {@link DEFAULT_CODEC_TILE_FRAMES}). A reference audio is usually a few seconds to a
 * dozen or so, and a single pass works there. If tiling turns out to be needed, add one more of
 * the same form as the decoder (translation equivariance with a halo).
 *
 * ## MUST: the low-precision knobs apply only to `dit` and `dit_context`
 *
 * The quant's `session` is passed only to the Sessions of `dit` and `dit_context`. The five
 * condition encoders run only once each (the dominant term of synthesis time is the DiT's 40 to
 * 100 forwards), which keeps the comparison axis of the execution knobs on the DiT. `dit_context`
 * is the condition side split out of the DiT, and before the split it ran inside `dit` with the
 * same options — running it with other options moves the numbers.
 *
 * ## MUST: uncond is only "cond's state + an all-False mask over the segment" (ADR 0047 decision 1)
 *
 * A distribution whose `speakerUncondMode` / `cfgGuidanceMode` is unsupported is **rejected by
 * `config.ts` at parse time**, so there is no branch here.
 *
 * ## MUST: numerical correctness is not guaranteed here
 *
 * Correctness is guaranteed by the per-graph golden E2E (`packages/e2e/tests/e2e_irodori_test.ts`)
 * and the full-loop latent golden (`pipeline/case.*.safetensors`, written by
 * `tools/export-recipes/irodori/pipeline_ref.py`).
 */

import {
  acquireGpu,
  type GpuContext,
  type SessionDiagnostics,
  type SessionOptions,
  type Tensor,
} from "@karume/runtime";
import {
  type DistributionSource,
  type HubRepoRef,
  loadManifest,
  type Manifest,
  resolveSelection,
} from "@karume/hub";

import {
  admitIrodori,
  assetJson,
  assetOpener,
  BACKBONE,
  CAPTION_PROJ,
  CODEC_DECODER,
  CODEC_ENCODER,
  COMPONENT_KEYS,
  DIT,
  DIT_CONTEXT,
  type DitContextOutput,
  DURATION,
  type IrodoriAdmission,
  SPEAKER,
  TEXT_PROJ,
  TOKENIZER,
} from "./admission.ts";
import {
  assertSpeakerReference,
  type ConditionState,
  emptyCondition,
  encodeSpeaker,
  rightPad,
} from "./conditioning.ts";
import {
  type DitLoop,
  type LatentStage,
  runDitLoopOnHost,
  runDitLoopResident,
  type UncondVariant,
} from "./dit-loop.ts";
import {
  asF32,
  bool,
  type EmitEvent,
  emitter,
  f32,
  i32,
  outputAt,
  withStageSession,
} from "./stage.ts";
import type { IrodoriPipelineConfig } from "./config.ts";
import { IrodoriTokenizer, parseIrodoriTokenizerAsset } from "./text/tokenizer.ts";
import {
  assertCodecTileFrames,
  type CodecTile,
  decodeTiles,
  DEFAULT_CODEC_TILE_FRAMES,
  planCodecTiles,
} from "./codec.ts";
import { packCaptionIds, packIds } from "./host/pack.ts";
import {
  buildDitMask,
  type IrodoriSegment,
  SEGMENT_ORDER,
  type SegmentLengths,
} from "./host/mask.ts";
import { rowMean } from "./host/pooling.ts";
import { Randn } from "./host/random.ts";
import {
  type SampleBounds,
  sequenceLengthFromLogFrames,
  sequenceLengthFromSeconds,
  type SequencePlan,
} from "./host/round.ts";
import { tSchedule } from "./host/sampler.ts";
import { timestepFrequencies } from "./host/t-embed.ts";
import { findFlatteningPoint, trimmedSampleCount } from "./host/trim.ts";
import { settleAbort } from "../concurrency/abort.ts";
import { createOperationChain } from "../concurrency/serial.ts";
import {
  assertGpuFeaturesGranted,
  assertRequiredLimitsBeforeDownload,
  assertRequiredLimitsSatisfied,
  toAcquireGpuOptions,
} from "../session/gpu-features.ts";
import { toManifestSource } from "../hub/repo-ref.ts";
import {
  type FromPretrainedComponentOptions,
  type FromPretrainedHubOptions,
  hubLoadOptions,
} from "../hub/load-options.ts";
import {
  type ComponentOpener,
  loadContainerComponents,
  type ModelComponent,
} from "../hub/components.ts";
import { ModelInputError } from "../errors.ts";
import { assertAcceptableSeed } from "../request-gates.ts";

/** Generation result. `data` is the patched latent `[frames × latentDim]` (row-major). */
export type GeneratedLatent = {
  readonly data: Float32Array<ArrayBuffer>;
  /** Number of latent frames S. */
  readonly frames: number;
  readonly latentDim: number;
  /**
   * The random seed actually used. `undefined` for a generation given `initialNoise`, which never
   * draws a random number.
   */
  readonly seed?: number;
  /** How many times `dit` ran (cond + the CFG uncond passes). */
  readonly forwards: number;
};

/**
 * Generation result (waveform). `data` is **already cut** mono f32 (can be passed to `encodeWav`
 * as is).
 *
 * The name carries a prefix because the barrel (`mod.ts`) already exports SBV2's `GeneratedAudio`
 * (that one has the 2 fields `{ sampleRate, data }`; this one also returns observations from the
 * latent side).
 */
export type IrodoriGeneratedAudio = {
  readonly data: Float32Array<ArrayBuffer>;
  readonly sampleRate: number;
  /** Number of generated latent frames S. **The tail trim can make the waveform shorter**. */
  readonly frames: number;
  /** The random seed actually used. `undefined` for a generation given `initialNoise`. */
  readonly seed?: number;
  /** How many times `dit` ran (cond + the CFG uncond passes). */
  readonly forwards: number;
};

/**
 * How the reference speaker is given.
 *
 * - `audio` — the reference audio itself (mono f32 + sampling rate). The return value of
 *   `decodeWav` can be passed as is. It is normalized on the host → goes through `codec_encoder`
 *   and then joins the same path as `latent`. **The rate must match the distribution's
 *   `sampleRate`** (there is no resampling — a mismatch fails loudly). The minimum length is
 *   `(2 × speakerPatchSize − 1) × hopLength + 1` samples (the speaker minimum of 2 tokens after
 *   patching — 13,441 samples for the v4 / v4.1 small distributions); a shorter one is a
 *   `ModelInputError` at the entry.
 * - `latent` — the DACVAE latent `[frames × latentDim]` of the reference audio. It goes through the
 *   **regular path** of patch → `speaker` → prepending the mean token. The minimum frame count is
 *   `2 × speakerPatchSize` (8 frames for the same distributions); a shorter one is a
 *   `ModelInputError` at the entry.
 * - `stateOverride` — a speaker state `[rows × speakerDim]` built beforehand. It is used as the
 *   condition as is, **bypassing** the `speaker` graph, `speaker_norm` and the mean-token prepend
 *   (upstream's `speaker_state_override` — the entry point for distributing embeddings).
 */
export type IrodoriSpeakerInput =
  | { readonly audio: { readonly data: Float32Array<ArrayBuffer>; readonly sampleRate: number } }
  | { readonly latent: Float32Array<ArrayBuffer> }
  | { readonly stateOverride: Float32Array<ArrayBuffer> };

/** One generation request. */
export type IrodoriGenerateRequest = {
  readonly text: string;
  /**
   * Voice instruction text (Voice Design). Empty / unset gives a zero caption condition + CFG off.
   */
  readonly caption?: string;
  readonly speaker?: IrodoriSpeakerInput;
  /** Random seed (default 0 — the same seed gives the same latent). */
  readonly seed?: number;
  /**
   * Direct injection of the initial noise `[frames × latentDim]`. **An entry point for
   * reproduction and verification**, not used in normal generation (it exists so the integration
   * check can feed torch's random sequence as is). Its length must match the decided S, so it is
   * normally used together with `durationSeconds`.
   */
  readonly initialNoise?: Float32Array<ArrayBuffer>;
  /**
   * Direct utterance length (seconds). When given, the `duration` graph **does not run**
   * (upstream's `manual_seconds`).
   */
  readonly durationSeconds?: number;
  /**
   * How many latent frames one codec decode takes (default {@link DEFAULT_CODEC_TILE_FRAMES}).
   * **A performance and memory knob that does not change the output** — the kept spans, with the
   * halo discarded, are bit-identical to a full-length decode (module doc of `codec.ts`).
   * When S is at most this value it collapses to one tile (= a single decode).
   */
  readonly codecTileFrames?: number;
  /**
   * Hook for generation events ({@link IrodoriGenerateEvent}) — called at the start / end of each
   * step and at the completion of each DiT step.
   *
   * **It is awaited** (the firing order becomes deterministic, and the consumer can thin out /
   * throttle). **Exceptions are not swallowed** (the same convention as `onRunDiagnostics` = fail
   * loudly) — as a by-product, **a throw is the way to abort at stage granularity** (generation
   * rejects, and the Session is released in the `finally` of `withSession`).
   *
   * MUST: when given, the DiT loop runs on the **host path** (the same selection mechanism as
   * `gpuTiming`). The resident path (ADR 0054) bundles all stages into 1 batch + a single fence, so
   * observing midway is structurally impossible, and firing at `enqueue` time would be a lie as
   * "progress". **The cost is wall time only** — it gives up the 1.76× that ADR 0054 brought to
   * the DiT loop (whole generation measured at 7.2 → 8.6 s / S 170, reference environment
   * 2026-08-16). The outputs of the 2 paths are bit-identical (the MUST of `runDitLoopResident`),
   * so **the waveform does not change by a single bit** (the onEvent step of
   * `e2e_irodori_wav_test.ts` checks it with the same sha256).
   *
   * MUST: inside `onEvent`, do not await `generate` / `generateLatent` / `dispose` of the same
   * pipeline (a self-deadlock on the serialization chain — abort by throwing).
   */
  readonly onEvent?: (event: IrodoriGenerateEvent) => void | Promise<void>;
};

/**
 * The component names that {@link IrodoriPipelineOptions.onRunDiagnostics} receives.
 * The stage names of the `stage` event ({@link IrodoriGenerateEvent}) are the same 9 names. The
 * spelling is the weights role name with `_` turned into `-` (`dit_context` → `dit-context`).
 */
export type IrodoriRunComponent =
  | "backbone"
  | "text-proj"
  | "caption-proj"
  | "speaker"
  | "duration"
  | "dit-context"
  | "dit"
  | "codec-encoder"
  | "codec-decoder";

/** Copy of an intermediate latent returned by `copyLatents()` of `denoise-step`. */
export type IrodoriLatentSnapshot = {
  readonly data: Float32Array<ArrayBuffer>;
  /** Shape of the latent `[frames, latentDim]`. */
  readonly shape: readonly number[];
};

/**
 * Generation events received by {@link IrodoriGenerateRequest.onEvent}.
 *
 * NOTE: same shape as anima's `AnimaGenerateEvent`, but there is no counterpart of `vae-tile` —
 * the codec decode tiles are **a performance and memory knob** (`codecTileFrames`), and as a step
 * they are covered by the `stage` of `codec-decoder`.
 */
export type IrodoriGenerateEvent =
  /**
   * **Before** the step's Session is built (`start`) and **after** it is released (`end`) — shows
   * the progress of GB-class loads.
   */
  | {
      readonly kind: "stage";
      readonly component: IrodoriRunComponent;
      readonly at: "start" | "end";
    }
  | {
      readonly kind: "denoise-step";
      /** Number of completed steps (1-based). **Not the number of forwards inside CFG**. */
      readonly step: number;
      readonly steps: number;
      /** Time `t` consumed by that step (flow-matching schedule — anima's sigma counterpart). */
      readonly t: number;
      /**
       * Copies and returns the intermediate latent only when called (`latentSnapshot` in
       * {@link "./dit-loop.ts"}).
       */
      readonly copyLatents: () => IrodoriLatentSnapshot;
    };

/**
 * Construction options (shared by {@link IrodoriPipeline.fromAssets} /
 * {@link IrodoriPipeline.fromPretrained}).
 */
export type IrodoriPipelineOptions = {
  /**
   * Shares an existing GPU. **The caller keeps ownership**, so {@link IrodoriPipeline.dispose} does
   * not destroy it. When omitted, the pipeline calls `acquireGpu` internally and destroys it in
   * `dispose()`.
   */
  readonly gpu?: GpuContext;
  /** Model (a key of the manifest's models). `defaultModel` when omitted. */
  readonly model?: string;
  /** Execution configuration (a key of that model's quants). `defaultQuant` when omitted. */
  readonly quant?: string;
  /**
   * Hook that receives the diagnostics of every run (1 generation = 5 to 7 condition-encoder runs +
   * 1 `dit-context` run + 40 to 100 `dit` runs).
   * For per-op GPU time (`lastRunTiming`), pass `acquireGpu({ gpuTiming: true })` as `gpu`
   * (ADR 0021 — the default does not measure).
   *
   * NOTE: on a device with timing enabled the DiT loop runs on the **host path** (a readback per
   * forward) — the batch the resident path uses is incompatible with timing (`beginBatch` rejects
   * it). The output is the same but the wall time nearly doubles, so do not enable timing except to
   * take a breakdown. The DiT diagnostics arrive per `enqueue`, and on the resident path
   * `lastRun` / `lastRunTiming` are `undefined` (neither an arena nor a timing window is created).
   *
   * Exceptions from the callback are not swallowed (fail loudly — the whole generation fails).
   */
  readonly onRunDiagnostics?: (
    component: IrodoriRunComponent,
    diagnostics: SessionDiagnostics,
  ) => void;
  /**
   * Aborts construction. {@link IrodoriPipeline.fromAssets} checks it at stage boundaries (entry /
   * between components / tokenizer parsing / before and after acquiring the GPU). At every boundary
   * except the entry it **yields to the event loop once before** checking, so an abort that arrives
   * during synchronous parsing takes effect at the next boundary (the same holds on a path with no
   * await at all, when `options.gpu` is given).
   * {@link IrodoriPipeline.fromPretrained} passes the same signal to the fetch layer, so the same
   * knob aborts **both during download and during assembly** (no abort button that goes dead only
   * after the download finishes).
   *
   * The abort exception throws `signal.reason` **as is** (not wrapped — the consumer can identify
   * its own abort with `error === controller.signal.reason`).
   */
  readonly signal?: AbortSignal;
  /**
   * Sets the execution form of the linears of `dit` / `dit_context`, taking precedence over the
   * quant's declaration (when omitted: the quant's `session` declaration → the runtime default —
   * ADR 0058 addendum 2026-09-26). `"f16"` needs the device's shader-f16; a GPU acquired by the
   * pipeline gets the requirement added. If a shared GPU (`gpu`) lacks it, it fails before the
   * weights are fetched. An invalid value is a `ModelInputError` (ADR 0107). The attention knobs
   * are not accepted (the `dit` graph has no place where they would apply —
   * `IRODORI_SESSION_POLICY` in `src/irodori/admission.ts`).
   */
  readonly linearCompute?: SessionOptions["linearCompute"];
};

/**
 * Additional fetch-layer options that {@link IrodoriPipeline.fromPretrained} accepts (passed
 * through to the hub). `signal` is shared with construction, so {@link IrodoriPipelineOptions}
 * holds it.
 *
 * NOTE: the per-field descriptions, including that `headers` / `fetch` / `caches` / `onRetry` are
 * **for HTTP sources only**, are kept in one place in {@link FromPretrainedHubOptions}.
 */
export type IrodoriFromPretrainedOptions = IrodoriPipelineOptions &
  FromPretrainedHubOptions &
  FromPretrainedComponentOptions;

/**
 * Input for building directly from fetched assets (pass the return value of the hub's
 * `fetchAssets` as is).
 */
export type IrodoriAssets = {
  readonly manifest: Manifest;
  readonly assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>;
};

/** Internal state of {@link IrodoriPipeline} (not exposed on the public surface). */
export type IrodoriState = {
  readonly gpu: GpuContext;
  readonly ownsGpu: boolean;
  readonly config: IrodoriPipelineConfig;
  readonly tokenizer: IrodoriTokenizer;
  readonly backbone: ModelComponent;
  readonly textProj: ModelComponent;
  readonly captionProj: ModelComponent;
  readonly speaker: ModelComponent;
  readonly duration: ModelComponent;
  readonly dit: ModelComponent;
  readonly ditContext: ModelComponent;
  readonly codecEncoder: ModelComponent;
  readonly codecDecoder: ModelComponent;
  /**
   * Name of the symbolic dimension S of `dit` (derived once only, in `admitIrodori` of
   * {@link "./admission.ts"}).
   */
  readonly ditSymbol: string;
  /**
   * Output list of `dit_context` (derived once only, in `admitIrodori` of
   * {@link "./admission.ts"}).
   */
  readonly ditContextOutputs: readonly DitContextOutput[];
  /**
   * Low-precision knobs. Passed **only to the Sessions of `dit` and `dit_context`** (the MUST of
   * the module doc).
   */
  readonly ditSessionOptions: SessionOptions;
  readonly onRunDiagnostics?: (
    component: IrodoriRunComponent,
    diagnostics: SessionDiagnostics,
  ) => void;
};

/**
 * Builds the execution state from the material that passed admission + the assets.
 *
 * MUST: asset parsing fails **before the GPU is acquired**. If the order slips, on an environment
 * without a GPU the failure turns into a different exception and the reader never learns "what
 * was wrong". The only checks allowed after acquiring the GPU are of GPU capabilities — features
 * (shader-f16 — ADR 0028) and the limits of a device acquired here (ADR 0089 decision 5 — for a
 * shared GPU `admitIrodori` checks them).
 * MUST: no Session is opened (the VRAM MUST — module doc).
 */
const buildIrodoriState = async (
  admitted: IrodoriAdmission,
  assets: IrodoriAssets["assets"],
  open: ComponentOpener,
  options: IrodoriPipelineOptions = {},
): Promise<IrodoriState> => {
  const { config, quantName, quant, ditSymbol, ditContextOutputs, ditSessionOptions, gpuFeatures } =
    admitted;
  // The supplier is **the same one** admission saw (`open` only looks up the opened components).
  const backbone = open(BACKBONE);
  const textProj = open(TEXT_PROJ);
  const captionProj = open(CAPTION_PROJ);
  const speaker = open(SPEAKER);
  const duration = open(DURATION);
  const dit = open(DIT);
  const ditContext = open(DIT_CONTEXT);
  const codecDecoder = open(CODEC_DECODER);
  const codecEncoder = open(CODEC_ENCODER);

  await settleAbort(options.signal);
  const tokenizer = new IrodoriTokenizer(
    parseIrodoriTokenizerAsset(assetJson(assets, TOKENIZER), TOKENIZER),
  );

  await settleAbort(options.signal);

  // MUST: declared features can be requested only when the device is created (ADR 0028). They
  // cannot be requested for a shared GPU passed in, so a missing capability fails by name (a shared
  // GPU has already been checked by the same function in `admitIrodori` of
  // {@link "./admission.ts"}, but for a device acquired here this is the only check).
  const gpu = options.gpu ?? (await acquireGpu(toAcquireGpuOptions(gpuFeatures)));
  const ownsGpu = options.gpu === undefined;
  try {
    // MUST: the abort check **after** acquiring the GPU stays inside the try — outside it, the
    // internally acquired device would be left with nobody able to release it (it rides on the
    // same cleanup as the feature check). Yield to a macrotask here too: the await of `acquireGpu`
    // resolves as a microtask continuation, so an abort task queued by a click during the wait
    // has not run yet.
    await settleAbort(options.signal);
    assertGpuFeaturesGranted(gpuFeatures, gpu, `IrodoriPipeline: quant '${quantName}'`);
    // The limits of a device acquired here are also checked here — `fromAssets` has no place for a
    // pre-acquisition check (`fromPretrained` checks against the adapter values), and without this
    // a shortfall in the declaration would surface only when the Session is built, and when the
    // declaration is stricter than the real demand the comparison would never run at all. A shared
    // GPU is checked by `admitIrodori`.
    if (ownsGpu) {
      assertRequiredLimitsSatisfied(
        quant.requiredLimits,
        gpu.limits,
        `IrodoriPipeline: quant '${quantName}'`,
      );
    }
    return {
      gpu,
      ownsGpu,
      config,
      tokenizer,
      backbone,
      textProj,
      captionProj,
      speaker,
      duration,
      dit,
      ditContext,
      codecEncoder,
      codecDecoder,
      ditSymbol,
      ditContextOutputs,
      ditSessionOptions,
      ...(options.onRunDiagnostics === undefined
        ? {}
        : { onRunDiagnostics: options.onRunDiagnostics }),
    };
  } catch (error) {
    // An internally acquired GPU can no longer be released by anyone if construction fails, so
    // return it here.
    if (ownsGpu) gpu.destroy();
    throw error;
  }
};

/**
 * Reads the numbers needed to decide S from `pipelineConfig` (the entry check and stage 5 read the
 * same one).
 */
const sampleBounds = (config: IrodoriPipelineConfig): SampleBounds => ({
  frameRate: config.frameRate,
  minSeconds: config.minSeconds,
  maxSeconds: config.maxSeconds,
  sampleRate: config.sampleRate,
  hopLength: config.hopLength,
});

/**
 * Checks the acceptance sets of the request knobs **before generation starts** (the entry of
 * `generate` / `generateLatent`).
 *
 * MUST: the acceptance sets themselves are not copied here — this only calls the canonical owner of
 * each knob (`assertCodecTileFrames` / `assertAcceptableSeed` / `sequenceLengthFromSeconds` /
 * `assertSpeakerReference`). Copying the conditions means a loosening of only one side goes
 * unnoticed.
 *
 * Why at the entry: these checks originally existed only **after** latent generation
 * (`planCodecTiles` right before decode, `Randn` at stage 7, the seconds request at stage 5, the
 * reference length at stage 4), so a typo such as `codecTileFrames: 10` or `seed: 1.5` failed only
 * after loading 1.26GB of weights and running every DiT step. The caller's typo is not paid for in
 * compute time.
 *
 * NOTE (3 changes in public behaviour): (1) `codecTileFrames` is checked even by
 * {@link IrodoriPipeline.generateLatent}, which does not run the codec (an unused value still
 * fails if it is a typo) (2) `seed` is checked even for a generation given `initialNoise`
 * (previously it never went through `Randn`, so an invalid value was silently ignored). (1) and (2)
 * are deliberate changes toward "do not silently accept a knob that has no effect". (3) The checks
 * of the `speaker` reference (`audio` / `latent`) (broken shape, too many rows, rate, length) were
 * moved forward from stage 4 to the entry — both `generate` and `generateLatent` fail before the
 * text / caption stages and the codec encoder run.
 *
 * NOTE: `export` is for the tests that pin the acceptance sets without a GPU (not exposed on
 * `mod.ts` / the subpath surface — ADR 0008, same convention as `latentSnapshot`).
 */
export const assertIrodoriRequest = (
  request: IrodoriGenerateRequest,
  config: IrodoriPipelineConfig,
): void => {
  // The default is checked too — the caller is not obliged to pass `codecTileFrames`, so a
  // distribution whose relation between the default tile and `codecHaloFrames` is broken must not
  // be allowed to "pass as long as nothing is given".
  assertCodecTileFrames(
    request.codecTileFrames ?? DEFAULT_CODEC_TILE_FRAMES,
    config.codecHaloFrames,
  );
  if (request.seed !== undefined) assertAcceptableSeed(request.seed);
  // The return value is discarded (stage 5 decides S again with the same function — this only
  // borrows the acceptance set).
  if (request.durationSeconds !== undefined) {
    sequenceLengthFromSeconds(request.durationSeconds, sampleBounds(config));
  }
  // Whether the reference (audio / latent) reaches the speaker minimum — waiting until stage 4 would
  // fail only after the text / caption stages and the codec encoder ran on the GPU.
  assertSpeakerReference(request.speaker, config);
};

/** Builds a latent from one text (+ caption / reference speaker). */
const generateLatent = async (
  state: IrodoriState,
  emit: EmitEvent,
  request: IrodoriGenerateRequest,
): Promise<LatentStage> => {
  const { config } = state;
  const textIds = packIds(state.tokenizer, request.text, config.maxTextLen, "text");
  // Whether a caption exists is decided on the **raw string** (upstream
  // `str(req.caption).strip() != ""`).
  const captionText = request.caption ?? "";
  const captionIds =
    captionText.trim().length === 0
      ? undefined
      : packCaptionIds(state.tokenizer, captionText, config.maxCaptionLen);

  // --- (1) backbone (text / caption run in the same single session) ------
  // One session with 2 runs, so the 1.26GB of weights is not uploaded twice. Diagnostics arrive
  // per run as "backbone".
  const hidden = await withStageSession(
    state,
    emit,
    "backbone",
    state.backbone,
    {},
    async (run) => {
      const text = outputAt(
        state.backbone,
        await run({ input_ids: i32(textIds, [1, textIds.length]) }),
        0,
      );
      if (captionIds === undefined) return { text, caption: undefined };
      const captionHidden = outputAt(
        state.backbone,
        await run({ input_ids: i32(captionIds, [1, captionIds.length]) }),
        0,
      );
      return { text, caption: captionHidden };
    },
  );

  // --- (2) text-proj -----------------------------------------------------
  const textState = await withStageSession(
    state,
    emit,
    "text-proj",
    state.textProj,
    {},
    async (run) => {
      const output = outputAt(state.textProj, await run({ hidden: hidden.text }), 0);
      return { data: asF32(output, "text-proj の出力"), rows: output.shape[1] };
    },
  );

  // --- (3) caption-proj (2 outputs — the 2nd output is the `caption_norm`-ed sequence) ---
  let captionState: ConditionState = emptyCondition();
  let captionVec: Float32Array<ArrayBuffer> | undefined;
  if (hidden.caption !== undefined) {
    const captionHidden = hidden.caption;
    const encoded = await withStageSession(
      state,
      emit,
      "caption-proj",
      state.captionProj,
      {},
      async (run) => {
        const outputs = await run({ hidden: captionHidden });
        const raw = outputAt(state.captionProj, outputs, 0);
        const normed = outputAt(state.captionProj, outputs, 1);
        return {
          state: { data: asF32(raw, "caption-proj の第 1 出力"), rows: raw.shape[1] },
          normed: asF32(normed, "caption-proj の第 2 出力"),
        };
      },
    );
    captionState = encoded.state;
    // `caption_vec` is the row mean of the **`caption_norm`-ed** sequence (taking it from the 1st
    // output gives a different vector).
    captionVec = rowMean(encoded.normed, captionState.rows, config.captionDim);
  }

  // --- (4) speaker -------------------------------------------------------
  const speakerState = await encodeSpeaker(state, emit, request.speaker);
  const hasSpeaker = speakerState.rows > 0;

  // --- (5) deciding S ----------------------------------------------------
  const bounds = sampleBounds(config);
  let plan: SequencePlan;
  if (request.durationSeconds !== undefined) {
    // A manual length does not run the duration graph (upstream's `manual_seconds` path).
    plan = sequenceLengthFromSeconds(request.durationSeconds, bounds);
  } else {
    const logFrames = await withStageSession(
      state,
      emit,
      "duration",
      state.duration,
      {},
      async (run) => {
        const outputs = await run({
          text_state: f32(textState.data, [1, textState.rows, config.textDim]),
          // `speaker_vec` is the **first row** of the speaker state (= the mean token).
          speaker_vec: f32(
            hasSpeaker
              ? speakerState.data.slice(0, config.speakerDim)
              : new Float32Array(config.speakerDim),
            [1, config.speakerDim],
          ),
          has_speaker: bool(hasSpeaker),
          caption_vec: f32(captionVec ?? new Float32Array(config.captionDim), [
            1,
            config.captionDim,
          ]),
          has_caption: bool(captionVec !== undefined),
        });
        return asF32(outputAt(state.duration, outputs, 0), "duration の出力")[0];
      },
    );
    plan = sequenceLengthFromLogFrames(logFrames, bounds);
  }
  const { frames } = plan;
  if (frames > config.ditSymMax) {
    throw new Error(
      `IrodoriPipeline: 決まった latent 長 ${frames} が dit の宣言上限 ${config.ditSymMax} を超えている`,
    );
  }

  // --- (6) right-padding of the conditions and the segment mask (ADR 0047 decisions 1 / 4) ---
  const caps: SegmentLengths = {
    text: config.maxTextLen,
    speaker: config.speakerRows,
    caption: config.maxCaptionLen,
  };
  const used: SegmentLengths = {
    text: textState.rows,
    speaker: speakerState.rows,
    caption: captionState.rows,
  };
  const conditionValues = {
    text_state: rightPad(textState, caps.text, config.textDim, "text 条件"),
    speaker_state: rightPad(speakerState, caps.speaker, config.speakerDim, "speaker 条件"),
    caption_state: rightPad(captionState, caps.caption, config.captionDim, "caption 条件"),
  };
  // Inputs of `dit_context` (used once in (6') — projected to K / V and then passed to `dit`).
  const conditions = {
    text_state: f32(conditionValues.text_state, [1, caps.text, config.textDim]),
    speaker_state: f32(conditionValues.speaker_state, [1, caps.speaker, config.speakerDim]),
    caption_state: f32(conditionValues.caption_state, [1, caps.caption, config.captionDim]),
  };
  const maskShape = [1, 1, 1, frames + caps.text + caps.speaker + caps.caption];
  const condMask: Tensor = {
    dtype: "bool",
    shape: maskShape,
    data: buildDitMask(frames, used, caps),
  };
  // Conditions with CFG enabled (upstream `has_*_cfg`): the strength is positive and the condition
  // is actually present (text always has at least 1 token — packIds rejects an empty string). The
  // uncond masks are not rebuilt per step; they are built once here.
  const uncondVariants: readonly UncondVariant[] = SEGMENT_ORDER.filter(
    (segment: IrodoriSegment) => config.cfgScales[segment] > 0 && used[segment] > 0,
  ).map((segment) => ({
    segment,
    // MUST: the strength is held as f32 — the host path's `combineCfg` feeds it into the multiply
    // as f64, while the resident path rounds it to f32 before passing it to the GPU, so a value
    // that is not exact in f32 splits the 2 paths by 1 to 2 ulp. `parseCfgScales` accepts only
    // f32-exact values, so on real input this is the identity, but rounding here makes "the host
    // side sees f32" readable locally (the numeric source of truth is `host/sampler.ts`).
    scale: Math.fround(config.cfgScales[segment]),
    mask: {
      dtype: "bool",
      shape: maskShape,
      data: buildDitMask(frames, used, caps, segment),
    } satisfies Tensor,
  }));

  // --- (6') context K/V projection + (7) Euler + CFG independent (both paths in dit-loop.ts) ---
  const noiseLength = frames * config.latentDim;
  const seed = request.seed ?? 0;
  let initial: Float32Array<ArrayBuffer>;
  if (request.initialNoise === undefined) {
    initial = new Randn(seed).normals(noiseLength);
  } else {
    if (request.initialNoise.length !== noiseLength) {
      // The length of an array the caller passed, so the 400 side (ADR 0107). S is not determined
      // uniquely by the request (there is the duration-graph path), so the entry cannot check it
      // and stage 7 stays its owner.
      throw new ModelInputError(
        `IrodoriPipeline: initialNoise の長さ ${request.initialNoise.length} が` +
          ` ${frames}×${config.latentDim} と違う（決まった latent 長は ${frames}）`,
      );
    }
    initial = request.initialNoise;
  }
  const loop: DitLoop = {
    frames,
    initial,
    schedule: tSchedule(config.steps, config.initScale),
    frequencies: timestepFrequencies(config.timestepEmbedDim),
    conditions,
    condMask,
    uncondVariants,
  };
  // MUST: a device with timing enabled cannot open a batch (ADR 0021 — unresolved timestamps pile
  // up for the whole span). To keep the per-op GPU time hook (`onRunDiagnostics` + `gpuTiming`),
  // only that device falls back to the original host loop. The operations queued are of the same
  // form on both, and the outputs must be the same.
  // Subscribing to generation events (`onEvent`) picks the host path for the same reason — on the
  // resident path, 1 batch + a single fence, the completion of a step cannot be observed from the
  // host at all (the MUST in the doc of `onEvent`).
  const { x, forwards } =
    state.gpu.gpuTimingEnabled || request.onEvent !== undefined
      ? await runDitLoopOnHost(state, emit, loop)
      : await runDitLoopResident(state, loop);

  return {
    latent: {
      data: x,
      frames,
      latentDim: config.latentDim,
      ...(request.initialNoise === undefined ? { seed } : {}),
      forwards,
    },
    plan,
  };
};

/**
 * Turns the latent into a waveform (runs the tiles in order in one session).
 *
 * MUST: builds the waveform for the **full length**. The tail trim and the cut to the requested
 * seconds happen **after** decode, at waveform sample granularity (cutting the latent before
 * decode changes the boundary padding, and the result is not bit-identical to the leading part of
 * a full-length decode — module doc of `codec.ts`).
 */
const decodeWaveform = async (
  state: IrodoriState,
  emit: EmitEvent,
  latent: GeneratedLatent,
  tiles: readonly CodecTile[],
): Promise<Float32Array<ArrayBuffer>> => {
  const { latentDim, hopLength } = state.config;
  return await withStageSession(
    state,
    emit,
    "codec-decoder",
    state.codecDecoder,
    {},
    async (run) =>
      await decodeTiles(latent.data, { latentDim, hopLength, tiles }, async (slice, frames) => {
        const outputs = await run({ latent: f32(slice, [1, frames, latentDim]) });
        return asF32(outputAt(state.codecDecoder, outputs, 0), "codec decoder の出力");
      }),
  );
};

/** Builds a waveform from one text (latent → tail trim → decode → cut). */
const generateAudio = async (
  state: IrodoriState,
  emit: EmitEvent,
  request: IrodoriGenerateRequest,
): Promise<IrodoriGeneratedAudio> => {
  const { config } = state;
  const { latent, plan } = await generateLatent(state, emit, request);
  // The tail trim is decided on z (before decode) — the same order as upstream's `_synthesize`.
  const flattening = findFlatteningPoint(latent.data, latent.frames, config.latentDim);
  const samples = trimmedSampleCount(plan.targetSamples, flattening, config.hopLength);
  const tiles = planCodecTiles(latent.frames, {
    tileFrames: request.codecTileFrames ?? DEFAULT_CODEC_TILE_FRAMES,
    haloFrames: config.codecHaloFrames,
  });
  const waveform = await decodeWaveform(state, emit, latent, tiles);
  const data =
    samples === waveform.length
      ? waveform
      : (waveform.slice(0, samples) as Float32Array<ArrayBuffer>);
  // MUST: do not silently return non-finite values. WAV encoding rounds NaN to 0 and pins ±Inf to
  // full scale, so a silent wrong value would come out as "audio that is silent in places / roaring
  // in places".
  for (const sample of data) {
    if (!Number.isFinite(sample)) throw new Error("irodori: 波形に非有限値が含まれる");
  }
  return {
    data,
    sampleRate: config.sampleRate,
    frames: latent.frames,
    ...(latent.seed === undefined ? {} : { seed: latent.seed }),
    forwards: latent.forwards,
  };
};

/**
 * The text → speech pipeline of Irodori-TTS v4.
 *
 * Construction has only two entry points, {@link IrodoriPipeline.fromPretrained} (fetch from HF)
 * and {@link IrodoriPipeline.fromAssets} (already fetched bytes) — the constructor is private so
 * that a half-built state that bypasses the manifest checks and the asset matching cannot be
 * created (ADR 0008).
 */
export class IrodoriPipeline {
  readonly #state: IrodoriState;
  /**
   * The serialization chain of generate / generateLatent and dispose (keeps "one at a time" on the
   * public API side).
   */
  readonly #chain = createOperationChain();
  /**
   * The single dispose. **Being not undefined means "disposed"** (holding a separate boolean would
   * be derived state updated independently).
   */
  #disposal: Promise<void> | undefined;

  private constructor(state: IrodoriState) {
    this.#state = state;
  }

  /**
   * Fetches from a distribution and builds (`loadManifest` → `resolveSelection` → fetch **only the
   * descriptor (part 0) of each component** and run admission → warm the weight parts →
   * `fetchAssets` for the remaining assets → build). Blocks are read in part order at the moment
   * the Session is built (ADR 0109 — `src/hub/components.ts`). To replace a component with the same
   * role from another repository, use {@link FromPretrainedComponentOptions.components} (the roles
   * are `backbone` / `text_proj` / `caption_proj` / `speaker` / `duration` / `dit` /
   * `dit_context` / `codec_decoder` / `codec_encoder`).
   * A string `ref` reads as `{ repo }` (= follows `main`). **`ref` is required** (sources have no
   * default — the MUST of `src/hub/repo-ref.ts`).
   *
   * A local distribution is passed as a **source handle** (`localDirectory` / `denoDirectory` of
   * `@karume/hub/deno`). It skips the spelling check of HF's `owner/name`, and goes through neither
   * the network nor CacheStorage (the HTTP-only knobs of {@link IrodoriFromPretrainedOptions} have
   * no effect).
   */
  static async fromPretrained(
    ref: string | HubRepoRef | DistributionSource,
    options: IrodoriFromPretrainedOptions = {},
  ): Promise<IrodoriPipeline> {
    const source = toManifestSource(
      ref,
      "IrodoriPipeline.fromPretrained",
      'IRODORI_SOURCES["irodori-v4.1-small"]（@karume/models/irodori）',
    );
    const hubOptions = hubLoadOptions(options);
    const loaded = await loadManifest(source, hubOptions);
    const choice = {
      ...(options.model === undefined ? {} : { model: options.model }),
      ...(options.quant === undefined ? {} : { quant: options.quant }),
    };
    const selection = resolveSelection(loaded.manifest, choice);
    // The signal goes to **both** the fetch layer and construction (no window where the abort stops
    // working the moment the download ends).
    const buildOptions: IrodoriPipelineOptions = {
      ...(options.gpu === undefined ? {} : { gpu: options.gpu }),
      ...choice,
      ...(options.onRunDiagnostics === undefined
        ? {}
        : { onRunDiagnostics: options.onRunDiagnostics }),
      ...(options.signal === undefined ? {} : { signal: options.signal }),
      ...(options.linearCompute === undefined ? {} : { linearCompute: options.linearCompute }),
    };
    // The family checks run in the admission hook (before the weight parts are fetched —
    // `hub/components.ts`).
    const { admitted, assets, open } = await loadContainerComponents(
      "IrodoriPipeline.fromPretrained",
      loaded,
      selection,
      COMPONENT_KEYS,
      async (open) => {
        const admitted = await admitIrodori(loaded.manifest, open, buildOptions);
        // The `requiredLimits` declared by the distribution are checked here, **before the weight
        // parts are fetched** (ADR 0089 decision 5 — the limits of a shared GPU, or the measured
        // adapter values on the path that acquires its own).
        await assertRequiredLimitsBeforeDownload(
          admitted.quant.requiredLimits,
          buildOptions.gpu,
          `IrodoriPipeline: quant '${admitted.quantName}'`,
        );
        return admitted;
      },
      {
        ...hubOptions,
        ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
        ...(options.components === undefined ? {} : { components: options.components }),
      },
    );
    return new IrodoriPipeline(await buildIrodoriState(admitted, assets, open, buildOptions));
  }

  /**
   * Builds from an already fetched manifest + assets. The contract checks, the interpretation of
   * the assets and `openContainer` all happen here, and **no Session is opened**. Passing
   * {@link IrodoriPipelineOptions.signal} allows aborting at step boundaries (the NOTE of
   * `admitIrodori`).
   *
   * Component keys **accept both forms** — the bytes of a single-form `krm` as one entry (`dit`),
   * and the part list of the split form (`dit[0]` / `dit[1]` / … — in index order from part 0). The
   * split form is opened as a part list without concatenating the bytes. A missing index or mixing
   * with a single-form key fails loudly (the receiving implementation is the one in
   * `src/hub/components.ts`).
   */
  static async fromAssets(
    input: IrodoriAssets,
    options: IrodoriPipelineOptions = {},
  ): Promise<IrodoriPipeline> {
    // MUST: the entry abort check comes **before** the containers are opened (the same one as the
    // entry of `admitIrodori`, moved forward). The receiver **opens every component first** in
    // order to return a synchronous supplier, so without this an already aborted call would open
    // 9 containers before failing.
    options.signal?.throwIfAborted();
    const open = await assetOpener(input.assets);
    const admitted = await admitIrodori(input.manifest, open, options);
    return new IrodoriPipeline(await buildIrodoriState(admitted, input.assets, open, options));
  }

  /**
   * Generates one waveform from text (f32 mono that can be passed to `encodeWav` as is).
   *
   * The same seed and the same request give the same waveform (both the random numbers and the
   * host glue are deterministic — `host/random.ts`).
   *
   * Concurrent calls **wait and run in order** (graphs are never resident at the same time —
   * module doc).
   */
  async generate(request: IrodoriGenerateRequest): Promise<IrodoriGeneratedAudio> {
    // The disposed check happens at call time (checking inside the chain would also fail
    // generations accepted before the dispose).
    if (this.#disposal !== undefined) {
      throw new Error("IrodoriPipeline: dispose 済みでは生成できない");
    }
    // The request-knob checks also happen at call time (outside the chain) — inside the chain, a
    // typo would fail only after waiting for the preceding generation to settle
    // (`createOperationChain` delays by 1 microtask even on an empty chain).
    assertIrodoriRequest(request, this.#state.config);
    return await this.#chain(() => generateAudio(this.#state, emitter(request.onEvent), request));
  }

  /**
   * Generates one latent from text (does not run the codec — the surface for the latent check and
   * for embedding uses).
   *
   * The same seed and the same request give the same latent. It rides on **the same chain** as
   * {@link IrodoriPipeline.generate}, so even mixed concurrent calls run in order.
   */
  async generateLatent(request: IrodoriGenerateRequest): Promise<GeneratedLatent> {
    if (this.#disposal !== undefined) {
      throw new Error("IrodoriPipeline: dispose 済みでは生成できない");
    }
    assertIrodoriRequest(request, this.#state.config);
    return await this.#chain(
      async () => (await generateLatent(this.#state, emitter(request.onEvent), request)).latent,
    );
  }

  /**
   * Releases. Destroys **only an internally acquired GPU** (a GpuContext passed in `options.gpu`
   * belongs to the caller and is not touched).
   *
   * MUST: destroys only after in-flight generations complete (flush-before-destroy) — putting the
   * destruction on the chain too decides the order of waiting and destroying in one place. The
   * second and later calls return the same completion (returning earlier would let the caller treat
   * it as "destroyed" and move on).
   */
  dispose(): Promise<void> {
    this.#disposal ??= this.#chain(() => {
      if (this.#state.ownsGpu) this.#state.gpu.destroy();
    });
    return this.#disposal;
  }

  /** Supports `await using` (Explicit Resource Management) — an alias of {@link dispose}. */
  [Symbol.asyncDispose](): Promise<void> {
    return this.dispose();
  }
}
