/**
 * `Pipeline` — the single surface for text → image (Anima).
 *
 * Pipeline (every stage in Karume, no torch):
 *
 * 1. Turn the prompt into id sequences with two tokenizers (Qwen2 BPE / T5 Unigram)
 * 2. `text_encoder` (Qwen3) → `text_conditioner` → zero-pad to 512
 * 3. Run the `transformer` (S-form DiT) for N steps, with CFG mixing + the update on the host (the
 *    update rule is the request's `sampler`, defaulting to the manifest's
 *    `pipelineConfig.scheduler.type` — Euler / DPM++ 2M)
 * 4. Denormalise the latent per channel → run `vae_decoder` **always tiled** (its Session plans with
 *    the VAE decoder's fusion rule `models/vae_rms_silu` — `../vae/fusion-rules.ts`)
 * 5. Convert to RGBA and return (PNG encoding is `encodePng` — shared, pipeline-independent)
 *
 * ## MUST: by default graphs are opened and closed one at a time (ADR 0016 / 0112)
 *
 * With the default `residency: "per-stage"` the **position** of each release matters. The text path
 * (weights 1,396MiB) is released **before** the DiT load, and the DiT **before** the VAE load. The
 * DiT weights differ by a factor of two between quant presets (default preset
 * `f16+dit8-a8-attn8-s16` = 1,875MiB, `f16` preset = 3,733MiB — the VRAM table of the 2026-08-05
 * final-perf-bench). Since the VAE is always tiled, **the chain peak is the DiT stage**, and the
 * default VRAM assumption is "the largest single stage". So {@link Pipeline.fromAssets}
 * **opens no Session at all** — it only opens containers (`openContainer` = parsing the two
 * documents), and GPU residency is opened and closed per stage inside
 * {@link Pipeline.generate}.
 *
 * The opt-in `residency: "transformer"` keeps only the DiT Session across generate calls (the state
 * machine is `residency.ts`). The text / VAE stages stack **on top of** the resident DiT, so the
 * chain peak always rises (+2,646MiB for the resident share at 1024² with the default preset —
 * ADR 0112). So while a resident DiT exists, before opening a stage (two points: before the text
 * stages, and after the DiT stage and before the VAE stage), the amount the next stage needs is
 * measured with a headroom probe (the runtime's `fitsHeadroom`), and if it does not fit, the
 * resident DiT is evicted **first** (before hitting an OOM — an OOM in a weight upload can lose the
 * device: "The two lines of eviction" in `residency.ts`). When the estimate is off and a stage
 * throws `GpuOutOfMemoryError`, the resident DiT is evicted and that stage re-run once (the second
 * line). After either eviction this pipeline no longer keeps the DiT resident (downgrade). The
 * resident DiT, too, is created in the DiT stage of the first generate (nothing is opened at
 * construction).
 *
 * MUST: **the public API also** keeps this sequencing — `generate` goes on the serialisation chain
 * (concurrent calls wait and run in order), and `dispose` waits for it to finish before destroying
 * the GPU. Otherwise the graphs of two concurrent calls are resident at once and the VRAM
 * assumption breaks, and a dispose during generation breaks flush-before-destroy.
 *
 * ## MUST: the DiT is S-form, the VAE is always tiled (ADR 0038 §4)
 *
 * The distributed transformer is only the S form, which holds no resolution, and the VAE decoder is
 * only one fixed tile of latent 64×64. So **the assets are independent of the resolution**, and
 * there is no non-tiled path and no static-shape branch (512px degenerates to one tile,
 * bit-identical to the non-tiled decode — the check of ADR 0033). The source of truth for the
 * accepted set is the constants in `resolution.ts`.
 *
 * ## MUST: the "correctness" of the output image is not guaranteed here
 *
 * Numerical correctness is guaranteed by the E2E against the reference fixtures (real GPU). The
 * seeded random numbers are **a different sequence from torch's `randn`**, so a freely generated
 * picture cannot be compared with torch (the doc of `random.ts`).
 */

import {
  acquireGpu,
  fitsHeadroom,
  type GpuContext,
  type Session,
  type SessionDiagnostics,
  type SessionOptions,
  type Tensor,
} from "@karume/runtime";
import {
  type DistributionSource,
  type GpuFeaturesSpec,
  type HubRepoRef,
  loadManifest,
  type Manifest,
  type ModelEntry,
  type Quant,
  resolveSelection,
} from "@karume/hub";

import {
  ANIMA_PIPELINE_MAJOR,
  ANIMA_PIPELINE_NAME,
  type AnimaPipelineConfig,
  type SamplerType,
  assertAnimaSamplerType,
  parseAnimaPipelineConfig,
} from "./config.ts";
import { assertAcceptableSteps, cfgEulerStep, sigmaSchedule, timestepsProj } from "./sampler.ts";
import {
  type DpmSolverMultistepInput,
  dpmSolverMultistepStep,
  needsUncond,
} from "../generation/dpm-solver-multistep.ts";
import { latentStats, denormalizeLatents, padSequence } from "./latents.ts";
import { imageToRgba } from "./image.ts";
import { assertAcceptableResolution, formatResolution, type ImageSize } from "./resolution.ts";
import { blendExtentAt, decodeTiled, planVaeTiling, tileCount } from "./tiling.ts";
import {
  ANIMA_SPATIAL_COMPRESSION,
  type DitPatchGeometry,
  ditPatchGeometry,
  patchifyLatents,
  ropeTables,
  tokenCount,
  unpatchifyTokens,
} from "./dit-tokens.ts";
import { parseRopeBase, type RopeBase, ropeWidth } from "./rope-base.ts";
import { type AnimaTokenizers, createTokenizers } from "./text/tokenizer.ts";
import { Randn } from "./random.ts";
import { ModelInputError } from "../errors.ts";
import { assertAcceptableSeed } from "../request-gates.ts";
import { settleAbort } from "../concurrency/abort.ts";
import { createOperationChain } from "../concurrency/serial.ts";
import {
  assertAcquiredGpuCapabilities,
  assertGpuFeaturesGranted,
  assertRequiredLimitsBeforeDownload,
  assertRequiredLimitsSatisfied,
  sessionGpuFeatures,
  toAcquireGpuOptions,
} from "../session/gpu-features.ts";
import {
  fusionRulesEstimateOptions,
  type PipelineFusionRules,
  withFusionRules,
} from "../session/fusion-rules.ts";
import { type FamilySessionPolicy, resolveSessionOptions } from "../session/options.ts";
import { vaeDecoderFusionRules } from "../vae/fusion-rules.ts";
import { disposeSteps } from "../session/dispose-steps.ts";
import { settleReleasedMemory } from "../session/settle-released-memory.ts";
import { toManifestSource } from "../hub/repo-ref.ts";
import {
  type FromPretrainedComponentOptions,
  type FromPretrainedHubOptions,
  hubLoadOptions,
} from "../hub/load-options.ts";
import {
  assetComponentOpener,
  type ComponentOpener,
  type GraphOwner,
  loadContainerComponents,
  type ModelComponent,
} from "../hub/components.ts";
import { readAssetBuffer, readWholeAsset } from "../hub/asset-readers.ts";
import {
  type Residency,
  type ResidencyAction,
  type ResidencyReason,
  assertAnimaResidency,
  DEFAULT_ANIMA_RESIDENCY,
  generateMemoryNeed,
  type ResidencyNotice,
  stageMemoryNeed,
  TransformerResidency,
} from "./residency.ts";

/**
 * Names that appear in the manifest's weights / assets tables (the conventional names of ADR 0041
 * §3).
 */
const TEXT_ENCODER = "text_encoder";
const TEXT_CONDITIONER = "text_conditioner";
const TRANSFORMER = "transformer";
const VAE_DECODER = "vae_decoder";
const TOKENIZER = "tokenizer";
const TOKENIZER_2 = "tokenizer_2";

/**
 * Asset name of the rope base table declared by the `transformer` container (role `rope-base` —
 * ADR 0109 decision 4).
 */
const ROPE_BASE = "rope_base";

/** This family's weight components (also the role names component replacement accepts). */
const COMPONENT_KEYS = [TEXT_ENCODER, TEXT_CONDITIONER, TRANSFORMER, VAE_DECODER] as const;

/** Generation result. `data` is 8-bit RGBA (4 bytes per pixel, row-major). */
export type GeneratedImage = {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8Array<ArrayBuffer>;
};

/**
 * One generation request. Fields left unspecified are filled from the manifest's
 * `pipelineConfig.defaults`.
 */
export type GenerateRequest = {
  readonly prompt: string;
  /**
   * Negative prompt. With `guidanceScale === 1` the uncond side is never computed, so **specifying
   * it fails loudly** (a knob with no effect is not silently accepted — {@link needsUncond}).
   */
  readonly negativePrompt?: string;
  /**
   * Number of denoise steps (an integer from 2 to 1000; default `pipelineConfig.defaults.steps`).
   * Out of range gives `ModelInputError` before touching the GPU.
   */
  readonly steps?: number;
  readonly guidanceScale?: number;
  readonly resolution?: ImageSize;
  /** Seed of the initial noise (default 0 — the same seed gives the same image). */
  readonly seed?: number;
  /**
   * Update rule of denoising ({@link SamplerType}). Defaults to the manifest's
   * `pipelineConfig.scheduler.type` (`"euler"` in the official distribution).
   *
   * DPM++ 2M is **one of the choices**, not the distribution default (re-ruling of 2026-08-25). The
   * update rule is a host-side formula independent of the assets, so **both values are always valid
   * whatever the distribution** — with the same assets and seed, a different value gives different
   * pixels (specifying `euler` is bit-identical to a distribution whose manifest default is euler).
   */
  readonly sampler?: SamplerType;
  /**
   * Whether the DiT Session is kept **after this generate** ({@link Residency}). Defaults to
   * {@link PipelineOptions.residency}.
   *
   * If a resident DiT already exists at the start, it is used whatever the value (no reload). In a
   * series of generations, using `"transformer"` in between and **`"per-stage"` only for the last
   * one** releases the resident DiT after that generate (`released` / `request` of the `residency`
   * event). A pipeline downgraded by an eviction (OOM / not enough headroom) does not keep it even
   * when `"transformer"` is requested, and reports `released` / `downgraded`. An unknown spelling
   * gives `ModelInputError` before touching the GPU.
   */
  readonly residency?: Residency;
  /**
   * Observation hook for generation events ({@link GenerateEvent}) — called at each stage
   * start / end, each completed denoise step, each completed VAE tile and each change of DiT
   * residency.
   *
   * **It is awaited** (the firing order becomes deterministic, and the consumer can thin out /
   * throttle). **Exceptions are not swallowed** (the same style as `onRunDiagnostics` = fail
   * loudly) — as a by-product, **a throw is the means of aborting at step granularity** (the
   * generation rejects and the stage's Session is released). Only a throw inside the denoise loop
   * (`denoise-step`) discards a carried-over resident DiT, and then `evicted` / `failure` of
   * `residency` is reported. A throw from a `stage` / `vae-tile` / `residency` notification leaves
   * the resident DiT healthy, so it is not discarded (but with effective `"per-stage"` it is
   * released and `released` / `request` is reported — the instruction "do not keep it afterwards"
   * takes effect even in a failed generate). Even if these notifications rethrow the same
   * exception, the generation throws only the original one.
   *
   * NOTE: no per-step VAE preview is provided. In the default per-stage operation the VAE can be
   * loaded only **after** the DiT is released (the VRAM MUST — module doc), so the only
   * intermediate result that can be handed out is the raw latent (`copyLatents`).
   * `approximatePreview` (`@karume/models/anima`) approximates a preview from this latent.
   *
   * MUST NOT await `generate` / `dispose` of the same pipeline inside `onEvent` (a self-deadlock of
   * the serialisation chain — abort with a throw instead).
   */
  readonly onEvent?: (event: GenerateEvent) => void | Promise<void>;
};

/**
 * Component names {@link PipelineOptions.onRunDiagnostics} receives (one Session = one name).
 * The stage names of the `stage` event ({@link GenerateEvent}) are the same four.
 */
export type RunComponent = "text_encoder" | "text_conditioner" | "transformer" | "vae_decoder";

/** Copy of the intermediate latent returned by `copyLatents()` of `denoise-step`. */
export type LatentSnapshot = {
  readonly data: Float32Array<ArrayBuffer>;
  /** Latent shape `[1,C,H,W]` (the value the DiT plan decided). */
  readonly shape: readonly number[];
};

/** Generation events {@link GenerateRequest.onEvent} receives. */
export type GenerateEvent =
  /**
   * Stage start (`start`) and end (`end`). In per-stage operation they fall **before** Session
   * construction and **after** release, so the progress of GB-scale loads is visible. A
   * `transformer` stage that uses a carried-over resident DiT with effective `"transformer"` has no
   * load or release between start and end (only the stage body). With effective `"per-stage"` the
   * release is inside start → end, and after an OOM eviction the release and the per-stage load are
   * inside. If the stage fails midway, `end` is not emitted.
   */
  | { readonly kind: "stage"; readonly component: RunComponent; readonly at: "start" | "end" }
  | {
      readonly kind: "denoise-step";
      /** Number of completed steps (1-based). */
      readonly step: number;
      readonly steps: number;
      /** The sigma consumed in that step. */
      readonly sigma: number;
      /** Copies and returns the intermediate latent only when called ({@link latentSnapshot}). */
      readonly copyLatents: () => LatentSnapshot;
    }
  /** Decode of one VAE tile completed (`tile` is 1-based). */
  | { readonly kind: "vae-tile"; readonly tile: number; readonly tiles: number }
  /**
   * Change of DiT residency (ADR 0112). Not emitted by a generate in the default per-stage
   * operation unrelated to residency.
   *
   * - `retained` / `request`: the DiT is kept after this generate too (before the end of the
   *   `transformer` stage).
   * - `released` / `request`: the carried-over resident DiT was released as effective `"per-stage"`
   *   says (before the end of the DiT stage — when the generate fails before the DiT stage, after
   *   that failure).
   * - `released` / `downgraded`: residency was requested, but the pipeline is downgraded by an
   *   eviction (`out-of-memory` / `headroom`), so the DiT is not kept.
   * - `evicted` / `headroom`: the next stage's need does not fit on top of the resident DiT (the
   *   headroom probe hit OOM), so the resident DiT was discarded **before the stage was opened**.
   *   It is emitted in one of two places — before the `stage` start of `text_encoder` (measured
   *   with the needs of text_encoder / text_conditioner — a generate with a carried-over DiT), or
   *   between the `stage` end of `transformer` and the `stage` start of `vae_decoder` (measured
   *   with the vae_decoder need — when a DiT is in the slot at that point). Afterwards this
   *   pipeline does not keep the DiT resident (downgrade — not undone, like an OOM eviction). With
   *   an eviction before the text stages and effective `"transformer"`, `released` / `downgraded`
   *   follows before the end of this generate's DiT stage (with effective `"per-stage"` the DiT
   *   stage runs per-stage and reports nothing). With an eviction before the VAE stage, the DiT
   *   stage has already reported `retained` / `request`.
   * - `evicted` / `out-of-memory`: a stage threw `GpuOutOfMemoryError` while a resident DiT
   *   existed, so the resident DiT was discarded and **that stage is re-run once from the start**
   *   (a DiT stage in per-stage operation — `denoise-step` / `vae-tile` restart from 1 afterwards).
   *   Afterwards this pipeline does not keep the DiT resident. In the DiT stage only a DiT carried
   *   over from an earlier generate is evicted (an OOM of a DiT created in this generate has the
   *   same VRAM layout as per-stage operation, so it is thrown as is without eviction — ADR 0112
   *   decision 3).
   * - `evicted` / `failure`: the DiT stage using a carried-over resident DiT failed (including a
   *   throw from `onEvent` in the denoise loop), so the resident DiT was discarded.
   */
  | {
      readonly kind: "residency";
      readonly component: "transformer";
      readonly action: ResidencyAction;
      readonly reason: ResidencyReason;
    };

/**
 * Builds the function that returns the intermediate latent (**lazy copy** — copies only when
 * called).
 *
 * A consumer that subscribes only to progress pays no copy cost, and since the internal array is
 * never handed out, the accident "the subscriber holds the next step's input" cannot happen
 * structurally.
 *
 * MUST: copy `shape` as well as `data`. The actual argument is **the raw mutable array** of
 * `plan.latentShape`; passing the reference lets a subscriber's modification break the patchify /
 * rope correspondence of the next step (the element count does not change, so the final "output
 * image size == requested resolution" check passes too, and a different picture comes out
 * silently).
 *
 * MUST: copy the array **as of creation**, not as of the call (bind it through the argument). The
 * denoise loop's `current` is **replaced with a new array** each step, so this binding is exactly
 * "that step's latent". Capturing the loop variable would return another step's latent to a
 * subscriber that calls later.
 *
 * NOTE: `export` is for the tests that pin the independence without a GPU (not exposed on `mod.ts`
 * / the subpath surface — ADR 0008).
 */
export const latentSnapshot =
  (latents: Float32Array<ArrayBuffer>, shape: readonly number[]): (() => LatentSnapshot) =>
  (): LatentSnapshot => ({ data: new Float32Array(latents), shape: [...shape] });

/**
 * Execution knobs Anima accepts (both from the manifest's quant declaration and explicit options).
 * Precedence, value ranges, combinations and the classification of the thrown type are held by the
 * cross-family `resolveSessionOptions`; this only decides the accepted set.
 *
 * It accepts the three fields for the execution form of linear / fused attention and not the four
 * for parallel GEMV and fusion — only the former are declared by the distribution and confirmed by
 * reference values; the latter (knobs added for Gemma's decode) are combinations whose effect and
 * numerics have not been confirmed on Anima's graphs.
 *
 * NOTE: `export` is so that the equivalence test runs every quant of the mirror through the same
 * table (not exposed on `mod.ts` / the subpath surface — ADR 0008).
 */
export const ANIMA_SESSION_POLICY: FamilySessionPolicy = {
  linearCompute: true,
  convCompute: false,
  attentionCompute: true,
  attentionScoreStorage: true,
  linearGemvReduce: false,
  stateAttentionReduce: false,
  fuseRmsNormAdd: false,
  fuseLinearStaticQuantize: false,
  fuseGeluMulStaticQuantize: false,
  packedStaticQuantize: false,
  integerGemv: false,
  linearGemvPlanes: false,
};

/**
 * Construction options (shared by {@link Pipeline.fromAssets} /
 * {@link Pipeline.fromPretrained}).
 */
export type PipelineOptions = {
  /**
   * Shares an existing GPU. **The caller keeps ownership**, so {@link Pipeline.dispose} does
   * not destroy it. When omitted, the pipeline calls `acquireGpu` internally and destroys it in
   * `dispose()`.
   */
  readonly gpu?: GpuContext;
  /** Model (a key of the manifest's models). Defaults to `defaultModel`. */
  readonly model?: string;
  /** Execution configuration (a key of that model's quants). Defaults to `defaultQuant`. */
  readonly quant?: string;
  /**
   * Observation hook that receives the diagnostics of each `Session.run` (DiT: one step = one run,
   * VAE: one tile = one run). For per-op GPU time (`lastRunTiming`), pass
   * `acquireGpu({ gpuTiming: true })` as `gpu` (ADR 0021 — not measured by default). Exceptions
   * from the callback are not swallowed (fail loudly — the whole generation fails).
   */
  readonly onRunDiagnostics?: (component: RunComponent, diagnostics: SessionDiagnostics) => void;
  /**
   * Abort of construction. {@link Pipeline.fromAssets} checks it at the boundaries between
   * steps (entry / tokenizer parsing / between each `openContainer` / before and after acquiring
   * the GPU). At every boundary except the entry it **yields to the event loop once** before
   * checking, so an abort that arrives during synchronous parsing takes effect at the next boundary
   * (the same on a path with no await at all because `options.gpu` was passed).
   * {@link Pipeline.fromPretrained} passes the same signal to the fetch layer, so the same
   * knob aborts **during either download or assembly** (no abort button that stops responding only
   * after the download finished).
   *
   * The abort exception is `signal.reason` thrown **as is** (not wrapped — the consumer can
   * recognise its own abort with `error === controller.signal.reason`).
   */
  readonly signal?: AbortSignal;
  /**
   * Execution form of linear, taking precedence over the quant's declaration (when omitted: the
   * quant's `session` declaration → the runtime default — ADR 0058 addendum 2026-09-26). It applies
   * **only to the DiT (transformer) Session**; the text stages and the VAE run as the quant
   * declares (f32) (the same scope as the quant preset's `session`). `"f16"` needs the device's
   * shader-f16; a GPU acquired internally gets the request added. If a shared GPU (`gpu`) lacks it,
   * this fails before the weights are fetched. Invalid values / combinations give `ModelInputError`
   * (ADR 0107).
   */
  readonly linearCompute?: SessionOptions["linearCompute"];
  /**
   * Execution form of fused attention, taking precedence over the quant's declaration (precedence
   * and shader-f16 handling as in {@link PipelineOptions.linearCompute}). Combined with
   * `attentionScoreStorage: "f16"` it is a combination the runtime does not accept, so
   * `ModelInputError`.
   */
  readonly attentionCompute?: SessionOptions["attentionCompute"];
  /**
   * Storage form of fused attention's S, taking precedence over the quant's declaration (precedence
   * as in {@link PipelineOptions.linearCompute}; shader-f16 is not needed).
   */
  readonly attentionScoreStorage?: SessionOptions["attentionScoreStorage"];
  /**
   * Default for whether the DiT Session is kept after a generate ({@link Residency}; when
   * omitted `"per-stage"` = the old behaviour of opening and closing per stage). Each generate can
   * override it with {@link GenerateRequest.residency}.
   *
   * `"transformer"` removes the DiT reload, plan derivation and intermediate-buffer rebuild from
   * the second generate on, at the cost of the text / VAE stages stacking on the resident DiT
   * (+2,646MiB at 1024² with the default preset; not usable on 4 GB-class GPUs —
   * docs/limitations.md). On an OOM for lack of VRAM the resident DiT is evicted, the stage re-run,
   * and residency is not used afterwards (reported with the `residency` event — ADR 0112). No
   * Session is opened at construction (the resident DiT is created in the DiT stage of the first
   * generate). An unknown spelling gives `ModelInputError` before the weights are fetched.
   */
  readonly residency?: Residency;
  /**
   * **Experimental**: fusion rules written outside the runtime (`defineRule` of
   * `@karume/runtime/kernel`), passed as `SessionOptions.fusionRules` to the Sessions of the image
   * graphs: the DiT (`transformer`, resident or per stage — the caller's array unchanged) and the
   * VAE decoder. The text stages (`text_encoder`, `text_conditioner`) do not get them. The VAE
   * decoder's Session always plans with the rule `models/vae_rms_silu` (`VAE_RMS_SILU_RULE` —
   * u32-identical to the unfused plan, on by default with no option to turn it off); the rules given
   * here follow it, in their order. Giving `models/vae_rms_silu` here again is refused as a rule
   * given twice by the VAE decoder's Session. Absent = no rule on the DiT, the family's rule alone on
   * the VAE decoder.
   *
   * It exists to measure a candidate rule on a whole generation (with and without the rule); it may
   * change or go away in a minor version. Passing a rule is the opt-in, and the rule's author owns
   * the evidence for its numbers (ADR 0148): the runtime does not check them. The VAE stage's
   * memory estimate (the pre-emptive eviction before the VAE stage, with a resident DiT) plans the
   * VAE with the same rule list (with the device's `maxComputeWorkgroupsPerDimension`, which a
   * rule's matcher may read). A value the runtime refuses fails when the first such Session is
   * built.
   *
   * DECIDED: docs/decisions/0148-external-fusion-rules.md
   */
  readonly fusionRules?: PipelineFusionRules;
};

/**
 * Extra fetch-layer options {@link Pipeline.fromPretrained} accepts (passed through to the
 * hub). `signal` is shared with construction, so {@link PipelineOptions} holds it.
 *
 * NOTE: the per-field documentation, including that `headers` / `fetch` / `caches` / `onRetry` are
 * **for HTTP sources only**, lives in one place: {@link FromPretrainedHubOptions}.
 */
export type FromPretrainedOptions = PipelineOptions &
  FromPretrainedHubOptions &
  FromPretrainedComponentOptions;

/**
 * Input for building directly from fetched assets (pass the return value of the hub's `fetchAssets`
 * as is).
 */
export type Assets = {
  readonly manifest: Manifest;
  readonly assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>;
};

/**
 * Turns fetched bytes into an ArrayBuffer that can be passed to `openContainer` (the check itself
 * is {@link readAssetBuffer}).
 *
 * MUST: do not copy with `slice` — the DiT alone is 3.7GiB, and the host RAM peak would double.
 */
const assetBuffer = (
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  key: string,
): ArrayBuffer => readAssetBuffer("anima", "weights / assets", assets, key);

/**
 * Component opener of the whole-asset surface (`fromAssets`) (the implementation is shared by every
 * family — {@link assetComponentOpener}). A component key is either the single-form `krm`
 * (`transformer`) or the parts of the split form (`transformer[0]` / `transformer[1]` / …).
 */
const assetOpener = (assets: Assets["assets"]): Promise<ComponentOpener> =>
  assetComponentOpener("anima", assets, (key) => assetBuffer(assets, key), COMPONENT_KEYS);

const assetBytes = (
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  key: string,
): Uint8Array<ArrayBuffer> => {
  assetBuffer(assets, key);
  return assets[key];
};

/**
 * Reads the **static** dimensions of a graph input.
 * MUST: do not put literals on the pipeline side — the container is the source of truth, and the
 * values follow when the model is replaced.
 */
const graphInputShape = (model: GraphOwner, inputName: string): readonly (number | string)[] => {
  const spec = model.graph.inputs.find((input) => input.name === inputName);
  if (spec === undefined) throw new Error(`Graph input '${inputName}' is missing`);
  return spec.shape;
};

const staticInputShape = (model: GraphOwner, inputName: string): readonly number[] =>
  graphInputShape(model, inputName).map((dim, axis) => {
    if (typeof dim !== "number") {
      throw new Error(
        `Graph input '${inputName}': axis ${axis} is not a static dimension (${String(dim)})`,
      );
    }
    return dim;
  });

const graphOutputShape = (model: GraphOwner): readonly (number | string)[] =>
  model.graph.values[model.graph.outputs[0]].shape;

const staticOutputShape = (model: GraphOwner): readonly number[] => {
  const name = model.graph.outputs[0];
  return graphOutputShape(model).map((dim, axis) => {
    if (typeof dim !== "number") {
      throw new Error(
        `Graph output '${name}': axis ${axis} is not a static dimension (${String(dim)})`,
      );
    }
    return dim;
  });
};

/**
 * Reads the **last dimension** (= feature width) of a shape that may have symbolic dimensions. Axis
 * 1 of the S-form `tokens [1,S,68]` is symbolic, so {@link staticInputShape} cannot read it, but
 * deriving the patch geometry needs only the last dimension.
 */
const featureWidth = (dims: readonly (number | string)[], where: string): number => {
  const last = dims.at(-1);
  if (typeof last !== "number") throw new Error(`${where}: the last dimension is not static`);
  return last;
};

const asF32 = (tensor: Tensor, where: string): Float32Array<ArrayBuffer> => {
  if (tensor.dtype !== "f32") throw new Error(`${where}: not f32 (${tensor.dtype})`);
  return tensor.data;
};

const idsTensor = (values: Int32Array<ArrayBuffer>): Tensor => ({
  dtype: "i32",
  shape: [1, values.length],
  data: values,
});

/**
 * Bytes one stage needs on top of the resident DiT ({@link stageMemoryNeed} in `residency.ts`).
 *
 * The two device limits are passed to every stage — the VAE's intermediates get their row blocks
 * from the limits, and the estimate fails without them (passed the same way to the text stages too,
 * so the estimate matches the stage's actual arguments).
 */
const stageNeed = (
  state: AnimaState,
  model: ModelComponent,
  bindings: Record<string, number>,
  fusionRules?: PipelineFusionRules,
): number =>
  stageMemoryNeed(
    model.estimate({
      maxStorageBufferBindingSize: state.gpu.limits.maxStorageBufferBindingSize,
      maxBufferSize: state.gpu.limits.maxBufferSize,
      bindings,
      // MUST: the rules the stage's Session gets (the estimate plans the fusion it runs).
      ...fusionRulesEstimateOptions(fusionRules, state.gpu.limits),
    }),
    model.maxPartBytes,
  );

/**
 * Bytes text_encoder / text_conditioner need on top of the resident DiT, used by the measurement
 * before the text stages (pre-emptive eviction — "The two lines of eviction" in `residency.ts`)
 * (the per-stage maximum — {@link generateMemoryNeed}).
 *
 * The bindings come from the shapes this generate actually passes to the stages: text_encoder's `T`
 * = the length of the qwen id sequence, text_conditioner's `Tsrc` = the length of that output (=
 * the length of the qwen id sequence) and `Ttgt` = the length of the T5 id sequence. The two CFG
 * prompts (positive / negative) run in turn on the same Session, so the longer one is measured.
 */
const textStagesNeed = (
  state: AnimaState,
  prompts: readonly ReturnType<AnimaTokenizers["encode"]>[],
): number => {
  const qwen = Math.max(...prompts.map((ids) => ids.qwenIds.length));
  const t5 = Math.max(...prompts.map((ids) => ids.t5Ids.length));
  return generateMemoryNeed([
    stageNeed(state, state.textEncoder, { T: qwen }),
    stageNeed(state, state.textConditioner, { Tsrc: qwen, Ttgt: t5 }),
  ]);
};

/**
 * Bytes vae_decoder needs on top of the resident DiT, used by the measurement before the VAE stage
 * (after the DiT stage) (static shape — no bindings), planned with the VAE decoder's rule list (the
 * list its Session gets — {@link animaStageSessionOptions}).
 */
const vaeStageNeed = (state: AnimaState): number =>
  stageNeed(state, state.vaeDecoder, {}, vaeDecoderFusionRules(state.fusionRules));

/**
 * Decides the negative prompt actually passed to the uncond side (the entry of
 * {@link Pipeline.generate}; a pure check before touching the GPU).
 *
 * MUST: do not silently accept a knob that has no effect. guidance=1 skips the uncond branch
 * entirely, so not a single character of the negative prompt is used (looking as if it could be
 * specified is the worst case). Conversely, with a setting that computes uncond and no spelling
 * given, this fails here before entering the GPU.
 *
 * Whether uncond is needed is a function of `guidance` alone, so it is derived here rather than
 * taken as an argument (leaving no room to disagree with the caller's `wantsUncond`).
 */
export const resolveNegativePrompt = (
  requested: string | undefined,
  fallback: string | undefined,
  guidance: number,
): string | undefined => {
  const wantsUncond = needsUncond(guidance);
  // Both are violations of the **combination** of `negativePrompt` and `guidanceScale`, and the fix
  // is on the request side (range, type and combination per ADR 0107 decision 3). The type matches
  // the non-finite `guidanceScale` check at the same entry.
  if (!wantsUncond && requested !== undefined) {
    throw new ModelInputError(
      `guidanceScale ${guidance} does not compute the uncond side, so negativePrompt has no effect` +
        " (to make it take effect, set guidanceScale to a value other than 1)",
    );
  }
  const negativePrompt = requested ?? fallback;
  if (wantsUncond && negativePrompt === undefined) {
    throw new ModelInputError(
      `guidanceScale ${guidance} computes the uncond side, so negativePrompt is required` +
        " (pass it in the manifest's pipelineConfig.defaults.negativePrompt or in the request)",
    );
  }
  return negativePrompt;
};

/**
 * Update result of one step. `previousX0` is state **carried as is into the next step's input**,
 * the only history DPM++ 2M's second-order term reads (Euler has no history, so it is always
 * `undefined`).
 */
export type AnimaDenoiseUpdate = {
  readonly sample: Float32Array<ArrayBuffer>;
  readonly previousX0: Float32Array<ArrayBuffer> | undefined;
};

/**
 * The point where the sampler type selects the update rule (**the only branch** the denoise loop
 * calls per step).
 *
 * MUST: `"euler"` does not change a single bit of `cfgEulerStep` and the spelling of Δσ — existing
 * distribution manifests without `scheduler.type` fall back to the default `"euler"` (`config.ts`),
 * so **the output images of already distributed repositories have to stay bit-identical** here.
 *
 * Both update rules do the CFG mixing inside (the same formula). Whether the uncond forward is
 * needed is decided by {@link needsUncond} from `guidance` alone, so it **does not depend on the
 * type** — the caller's branches for "how many steps" and "whether to compute uncond" stay shared.
 *
 * MUST: check the finiteness of the updated latent here (an O(N) host scan). Even a CFG scale
 * finite in f32 can overflow the mix in combination with the DiT output (scale 1e38 × difference
 * 10). Passing a non-finite latent silently to the next step and the VAE stage either fails late at
 * the final RGBA conversion in a form whose real cause cannot be read, or the clamp inside the VAE
 * turns it into ordinary pixels (the same check as Wan's denoise).
 *
 * NOTE: the input shape borrows {@link DpmSolverMultistepInput} as is (Euler reads `sigmas` /
 * `index` / `guidance` / the two outputs, a subset of DPM++ 2M's input).
 * NOTE: `export` is for the tests that pin the selection without a GPU (not exposed on `mod.ts` /
 * the subpath surface — ADR 0008).
 */
export const denoiseStep = (
  type: SamplerType,
  input: DpmSolverMultistepInput,
): AnimaDenoiseUpdate => {
  const update = applyUpdateRule(type, input);
  const broken = update.sample.findIndex((value) => !Number.isFinite(value));
  if (broken !== -1) {
    const steps = input.sigmas.length - 1;
    const value = update.sample[broken];
    throw new Error(
      `anima.Pipeline: step ${
        input.index + 1
      }/${steps}: element ${broken} of the updated latent is non-finite` +
        ` (${value}) — the DiT output, the CFG mix or the update rule overflowed (not passed to the next step or the VAE stage)`,
    );
  }
  return update;
};

/** Applies one update rule (the body of the branch in {@link denoiseStep}). */
const applyUpdateRule = (type: SamplerType, input: DpmSolverMultistepInput): AnimaDenoiseUpdate => {
  switch (type) {
    case "euler": {
      const { sample, cond, uncond, guidance, sigmas, index } = input;
      return {
        sample: cfgEulerStep(
          sample,
          cond,
          uncond,
          Math.fround(sigmas[index + 1] - sigmas[index]),
          guidance,
        ),
        previousX0: undefined,
      };
    }
    case "dpmpp-2m": {
      const update = dpmSolverMultistepStep(input);
      return { sample: update.sample, previousX0: update.x0 };
    }
  }
};

/**
 * Material of the S-form DiT that does not change between steps (the rope tables and patch geometry
 * are functions of the resolution only).
 */
type DynDitPlan = {
  readonly geometry: DitPatchGeometry;
  readonly latentShape: readonly number[];
  readonly tokenShape: readonly number[];
  readonly ropeShape: readonly number[];
  readonly cos: Float32Array<ArrayBuffer>;
  readonly sin: Float32Array<ArrayBuffer>;
};

/**
 * Builds the S-form DiT material from the resolution. Called once outside the denoise loop
 * (rebuilding it every step would add a waste of 4MB×2 per step at 1024px).
 */
const planDynDit = (model: GraphOwner, ropeBase: RopeBase, resolution: ImageSize): DynDitPlan => {
  const tokenWidth = featureWidth(graphInputShape(model, "tokens"), "graph input 'tokens'");
  const geometry = ditPatchGeometry(
    tokenWidth,
    featureWidth(graphOutputShape(model), "graph output"),
  );
  // MUST: derive the latent size **from the resolution** (the S-form graph holds no resolution). A
  // mix-up is caught by the "output image size == requested resolution" check at the end of
  // generate.
  // MUST: the axis order is `[1,C,H,W]` (the reverse of the spelling WxH). Swapping them here for a
  // non-square size transposes the latent while the element count still matches, and the picture
  // silently becomes a different one (a mix-up undetectable with square sizes).
  const latentHeight = resolution.height / ANIMA_SPATIAL_COMPRESSION;
  const latentWidth = resolution.width / ANIMA_SPATIAL_COMPRESSION;
  if (!Number.isInteger(latentHeight) || !Number.isInteger(latentWidth)) {
    throw new Error(
      `Resolution ${resolution.width}×${resolution.height}: the spatial compression ratio ${ANIMA_SPATIAL_COMPRESSION}` +
        " does not divide it",
    );
  }
  const latentShape = [1, geometry.channels, latentHeight, latentWidth];
  const { cos, sin } = ropeTables(ropeBase, latentShape, geometry);
  return {
    geometry,
    latentShape,
    tokenShape: [1, tokenCount(latentShape, geometry), tokenWidth],
    ropeShape: [1, 1, tokenCount(latentShape, geometry), ropeWidth(ropeBase)],
    cos,
    sin,
  };
};

/**
 * The run a stage body receives (returns the one output — including the notification to the
 * observation hook).
 */
type StageRun = (inputs: Record<string, Tensor>) => Promise<Tensor>;

/**
 * Wraps one Session's run for a stage body (extracts the one output and passes the diagnostics to
 * the observation hook).
 */
const stageRun = (
  session: Session,
  model: ModelComponent,
  observe: ((diagnostics: SessionDiagnostics) => void) | undefined,
): StageRun => {
  const outputName = model.graph.outputs[0];
  return async (inputs) => {
    const outputs = await session.run(inputs);
    if (observe !== undefined) observe(session.diagnostics());
    return outputs[outputName];
  };
};

/**
 * Body of the pipeline's `dispose`: fold the resident DiT, then destroy the GPU (`destroyGpu` is
 * passed only for an internally acquired GPU — a shared GPU belongs to the caller).
 *
 * MUST: the resident DiT first (flush-before-destroy). Even if folding it fails, the GPU is
 * returned (the doc of `disposeSteps` — failures are thrown after every step has run).
 *
 * NOTE: `export` is for the tests that pin the order and failure handling without a GPU (not
 * exposed on `mod.ts` / the subpath surface — ADR 0008). The real-GPU e2e has no way to observe
 * "the resident DiT was folded" (GpuContext has no diagnostics of live bytes).
 */
export const disposeResidencyThenGpu = (
  residency: { readonly dispose: () => Promise<void> },
  destroyGpu: (() => void) | undefined,
): Promise<void> =>
  disposeSteps([() => residency.dispose(), ...(destroyGpu === undefined ? [] : [destroyGpu])]);

/**
 * Opens a Session for one graph and always releases it when done.
 * MUST: dispose in `finally` — if VRAM stays after a failure midway, a later stage fails to
 * allocate and fails "somewhere other than the first failure".
 */
const withSession = async <T>(
  gpu: GpuContext,
  model: ModelComponent,
  sessionOptions: SessionOptions,
  observe: ((diagnostics: SessionDiagnostics) => void) | undefined,
  body: (run: StageRun) => Promise<T>,
): Promise<T> => {
  const session = await model.createSession(gpu, sessionOptions);
  let failure: { readonly error: unknown } | undefined;
  try {
    return await body(stageRun(session, model, observe));
  } catch (error) {
    failure = { error };
    throw error;
  } finally {
    // MUST: a dispose failure does not overwrite the body's failure (the doc of `disposeSteps`).
    await disposeSteps([
      () => {
        if (failure !== undefined) throw failure.error;
      },
      () => session.dispose(),
    ]);
  }
};

/**
 * The Session options of a stage other than the DiT: none of the quant's knobs (the text stages /
 * VAE run as the quant declares — f32); the VAE alone plans with the VAE decoder's rule list
 * (`models/vae_rms_silu`, then the experimental `fusionRules` — {@link PipelineOptions.fusionRules}),
 * the text stages with none.
 *
 * NOTE: `export` is for the host test (not exposed on `mod.ts` / the subpath surface — ADR 0008).
 */
export const animaStageSessionOptions = (
  component: Exclude<RunComponent, "transformer">,
  fusionRules: PipelineFusionRules,
): SessionOptions =>
  component === "vae_decoder" ? withFusionRules({}, vaeDecoderFusionRules(fusionRules)) : {};

/**
 * Binds the component name for the observation hook
 * ({@link PipelineOptions.onRunDiagnostics}).
 */
const observer = (
  state: AnimaState,
  component: RunComponent,
): ((diagnostics: SessionDiagnostics) => void) | undefined => {
  const listener = state.onRunDiagnostics;
  return listener === undefined ? undefined : (diagnostics) => listener(component, diagnostics);
};

/**
 * Internal state of {@link Pipeline} (the constructor is private, so the type is not public).
 */
type AnimaState = {
  readonly gpu: GpuContext;
  readonly ownsGpu: boolean;
  readonly config: AnimaPipelineConfig;
  readonly sessionOptions: SessionOptions;
  readonly tokenizers: AnimaTokenizers;
  readonly textEncoder: ModelComponent;
  readonly textConditioner: ModelComponent;
  readonly transformer: ModelComponent;
  readonly ropeBase: RopeBase;
  readonly vaeDecoder: ModelComponent;
  /** Effective residency when a generate does not override it in the request (checked). */
  readonly residency: Residency;
  /**
   * Experimental ({@link PipelineOptions.fusionRules}): the DiT's Session gets them; the VAE's
   * Session and estimate get them after `models/vae_rms_silu`; the text stages' get none. Absent =
   * no rule on the DiT, the family's rule alone on the VAE.
   */
  readonly fusionRules?: PipelineFusionRules;
  readonly onRunDiagnostics?: (component: RunComponent, diagnostics: SessionDiagnostics) => void;
};

/** Material the family admission (`Pipeline.#admit`) settles. */
type AnimaAdmission = {
  readonly config: AnimaPipelineConfig;
  readonly quantName: string;
  readonly quant: Quant;
  /**
   * Effective settings combining explicit options and the quant declaration
   * ({@link ANIMA_SESSION_POLICY}).
   */
  readonly sessionOptions: SessionOptions;
  /**
   * The quant declaration plus the features the effective settings need (both the request and the
   * check read it).
   */
  readonly gpuFeatures: GpuFeaturesSpec | undefined;
  /** Residency from the construction options (checked; the default when omitted). */
  readonly residency: Residency;
};

/**
 * Anima's text → image pipeline.
 *
 * Construction has only two entries: {@link Pipeline.fromPretrained} (fetch from HF) or
 * {@link Pipeline.fromAssets} (already fetched bytes) — the constructor is private so that no
 * half-built state can bypass the manifest checks and the asset comparison (the same style as
 * `createSession` / `acquireGpu` — ADR 0008).
 */
export class Pipeline {
  readonly #state: AnimaState;
  /**
   * Serialisation chain of generate and dispose (keeps the module doc's "one at a time" on the
   * public API side).
   */
  readonly #chain = createOperationChain();
  /**
   * The one dispose. **Not undefined means "disposed"** (a separate boolean would be derived state
   * updated independently).
   */
  #disposal: Promise<void> | undefined;
  /**
   * The DiT resident slot (ADR 0112). Owned by the pipeline — folded in `dispose` even when
   * `options.gpu` is shared. Touched only from inside the chain ({@link Pipeline.#chain}).
   */
  readonly #residency: TransformerResidency<Session>;

  private constructor(state: AnimaState) {
    this.#state = state;
    this.#residency = new TransformerResidency<Session>({
      dispose: (session) => session.dispose(),
      // After an eviction (OOM / not enough headroom), wait for the release to reach the device
      // (the reason is in the doc of settle-released-memory.ts). The empty submit is safe here
      // because an eviction runs only when usage is at or below the allocation line (97 % of the
      // budget) (the pre-emptive probe returns after waiting for the release; a reactive eviction
      // comes right after an allocation was refused with OOM) or when the device is already dead —
      // so it never hits wgpu's after-submit 99 % line check (exceeding it loses the device).
      // NOTE: on the B570 the pipeline's eviction → re-run (evict-probe, 6 GiB dummy) hit device
      // loss because the stage's OOM was on the staging side of `queue.writeBuffer`, which had
      // already invalidated the device (docs/research/2026-09-27-h35-oom-device-lost.md) — waiting
      // for the release does not fix it. The main line is pre-emptive eviction ("The two lines of
      // eviction" in `residency.ts`).
      settleRelease: () => settleReleasedMemory(state.gpu),
    });
  }

  /**
   * Fetches from a distribution and builds (`loadManifest` → `resolveSelection` → fetch **only each
   * component's descriptor (part 0)** for admission → warm the weight parts → `fetchAssets` for the
   * remaining assets → build). Blocks are read in part order at the moment each Session is built
   * (ADR 0109 — `src/hub/components.ts`). A string `ref` is read as `{ repo }` (= following
   * `main`). **`ref` is required** (there is no default source — the MUST in
   * `src/hub/repo-ref.ts`).
   *
   * To replace a component with the same role from another repository, use
   * {@link FromPretrainedComponentOptions.components} (`text_encoder` / `text_conditioner` /
   * `transformer` / `vae_decoder`).
   *
   * A local distribution is passed as **a source handle** (`localDirectory` / `nodeDirectory` of
   * `@karume/hub/node` / `denoDirectory` of `@karume/hub/deno`). It skips the check on HF's
   * `owner/name` spelling and goes through neither the network nor CacheStorage (the HTTP-only
   * knobs of {@link FromPretrainedOptions} have no effect).
   */
  static async fromPretrained(
    ref: string | HubRepoRef | DistributionSource,
    options: FromPretrainedOptions = {},
  ): Promise<Pipeline> {
    const source = toManifestSource(
      ref,
      "anima.Pipeline.fromPretrained",
      'SOURCES["anima"] (@karume/models/anima)',
    );
    const hubOptions = hubLoadOptions(options);
    const loaded = await loadManifest(source, hubOptions);
    const choice = {
      ...(options.model === undefined ? {} : { model: options.model }),
      ...(options.quant === undefined ? {} : { quant: options.quant }),
    };
    const selection = resolveSelection(loaded.manifest, choice);
    // The signal goes to **both** the fetch layer and the build (no window where aborting stops
    // working the moment the download ends).
    const buildOptions: PipelineOptions = {
      ...(options.gpu === undefined ? {} : { gpu: options.gpu }),
      ...choice,
      ...(options.onRunDiagnostics === undefined
        ? {}
        : { onRunDiagnostics: options.onRunDiagnostics }),
      ...(options.signal === undefined ? {} : { signal: options.signal }),
      ...(options.linearCompute === undefined ? {} : { linearCompute: options.linearCompute }),
      ...(options.attentionCompute === undefined
        ? {}
        : { attentionCompute: options.attentionCompute }),
      ...(options.attentionScoreStorage === undefined
        ? {}
        : { attentionScoreStorage: options.attentionScoreStorage }),
      ...(options.residency === undefined ? {} : { residency: options.residency }),
      ...(options.fusionRules === undefined ? {} : { fusionRules: options.fusionRules }),
    };
    // The family checks run in the admission hook (before the weight parts are fetched —
    // `src/hub/components.ts`).
    const { admitted, assets, open } = await loadContainerComponents(
      "anima.Pipeline.fromPretrained",
      loaded,
      selection,
      COMPONENT_KEYS,
      async () => {
        const admitted = Pipeline.#admit(loaded.manifest, buildOptions);
        // The `requiredLimits` the distribution declares are checked here **before the weight parts
        // are fetched** (ADR 0089 decision 5 — the limits of a shared GPU, or the adapter's
        // measured values on the path that acquires its own).
        await assertRequiredLimitsBeforeDownload(
          admitted.quant.requiredLimits,
          buildOptions.gpu,
          `anima.Pipeline: quant '${admitted.quantName}'`,
        );
        return admitted;
      },
      {
        ...hubOptions,
        ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
        ...(options.components === undefined ? {} : { components: options.components }),
      },
    );
    return await Pipeline.#build(admitted, assets, open, buildOptions);
  }

  /**
   * Checks whether this manifest can run as anima (the family admission hook of
   * `src/hub/components.ts` — on the fetch surface it is called **before a single byte of the
   * weight parts is fetched**).
   *
   * MUST: gather the family checks in this one place. Scattered into later steps, they would fail
   * on the fetch surface only **after** GB-scale weights were downloaded (the implementation would
   * be narrower than the wording of ADR 0070 decision 5).
   * MUST: manifest contract violations fail **before acquiring the GPU** (the same order as the
   * other families).
   *
   * NOTE: it takes no graph because this family has no "graph declaration × pipelineConfig"
   * comparison at construction (`planDynDit` derives the patch geometry and rope per resolution at
   * generation). Parsing the assets (two tokenizers / the rope base table) cannot live in this hook
   * either — at admission only the **declarations** of the assets (name → role, logical length) are
   * at hand, not the bytes (waiting for them would lose the position before the weight prefetch),
   * so it stays in {@link Pipeline.#build}.
   */
  static #admit(manifest: Manifest, options: PipelineOptions): AnimaAdmission {
    // Abort checks sit at **the boundaries between steps** (each step is indivisible — there is no
    // way to fold a 3.7GiB component midway). The entry is the first one: when called already
    // aborted, it returns without touching a byte of the assets.
    options.signal?.throwIfAborted();
    const modelName = options.model ?? manifest.defaultModel;
    if (!Object.hasOwn(manifest.models, modelName)) {
      throw new Error(
        `anima.Pipeline: model '${modelName}' is not in the manifest` +
          ` (available: ${manifest.available.models.join(" / ")})`,
      );
    }
    const entry: ModelEntry = manifest.models[modelName];
    const { name, major } = entry.pipeline;
    if (name !== ANIMA_PIPELINE_NAME) {
      throw new Error(
        `anima.Pipeline: the manifest's pipeline is '${name}/${major}'` +
          ` ('${ANIMA_PIPELINE_NAME}/${ANIMA_PIPELINE_MAJOR}' is required)`,
      );
    }
    if (major !== ANIMA_PIPELINE_MAJOR) {
      // The only check that stops the silent degradation of "old implementation × new repository"
      // (ADR 0038 §6).
      throw new Error(
        `anima.Pipeline: the major of pipeline '${name}/${major}' is not supported` +
          ` (this implementation reads ${ANIMA_PIPELINE_NAME}/${ANIMA_PIPELINE_MAJOR})`,
      );
    }
    const config = parseAnimaPipelineConfig(entry.pipelineConfig);

    const quantName = options.quant ?? entry.defaultQuant;
    if (!Object.hasOwn(entry.quants, quantName)) {
      throw new Error(
        `anima.Pipeline: quant '${quantName}' is not in the manifest` +
          ` (available: ${entry.available.quants.join(" / ")})`,
      );
    }
    const quant = entry.quants[quantName];
    // Explicit > quant declaration > runtime default (one implementation shared by every family).
    // Unsupported declarations and wrong explicit options both fail here (before the weight parts
    // are fetched).
    const sessionOptions = resolveSessionOptions(
      ANIMA_SESSION_POLICY,
      quant.session,
      options,
      `anima.Pipeline: quant '${quantName}'`,
    );
    const gpuFeatures = sessionGpuFeatures(quant.gpuFeatures, sessionOptions);
    // residency is checked in the same hook (an unknown spelling is not carried past the weight
    // fetch).
    const residency =
      options.residency === undefined
        ? DEFAULT_ANIMA_RESIDENCY
        : assertAnimaResidency(options.residency, "anima.Pipeline: residency");

    // MUST: a shared GPU's missing capabilities (feature / device limit) fail in this hook — unlike
    // acquiring a GPU internally there is no reason to wait for `acquireGpu`, and this is the only
    // family check that knows before the weights are downloaded (the mapping between request and
    // check is the single one in `session/gpu-features.ts`, and the later checks call the same
    // function).
    if (options.gpu !== undefined) {
      assertGpuFeaturesGranted(gpuFeatures, options.gpu, `anima.Pipeline: quant '${quantName}'`);
      assertRequiredLimitsSatisfied(
        quant.requiredLimits,
        options.gpu.limits,
        `anima.Pipeline: quant '${quantName}'`,
      );
    }

    return { config, quantName, quant, sessionOptions, gpuFeatures, residency };
  }

  /**
   * Builds from the admitted material + the assets (the one implementation shared by
   * {@link Pipeline.fromAssets} and `fromPretrained`). The two surfaces differ only in `open`
   * (the component opener); they pass the same checks in the same order.
   *
   * MUST: the assets are parsed **before acquiring the GPU** (the ordering MUST of the docstring).
   */
  static async #build(
    admitted: AnimaAdmission,
    assets: Assets["assets"],
    open: ComponentOpener,
    options: PipelineOptions,
  ): Promise<Pipeline> {
    const { config, quantName, quant, sessionOptions, gpuFeatures, residency } = admitted;

    // Asset parsing comes before the GPU (the ordering MUST of the docstring). Opening the 3.7GiB
    // DiT is heavier than creating the device, but not hiding the real cause of a broken
    // distribution wins — without a GPU, acquireGpu itself fails, so placed after it, "the asset is
    // missing" would never be visible.
    await settleAbort(options.signal);
    const tokenizers = createTokenizers(
      assetBytes(assets, TOKENIZER),
      assetBytes(assets, TOKENIZER_2),
    );
    await settleAbort(options.signal);
    const textEncoder = open(TEXT_ENCODER);
    await settleAbort(options.signal);
    const textConditioner = open(TEXT_CONDITIONER);
    await settleAbort(options.signal);
    const transformer = open(TRANSFORMER);
    // The rope base table is **a container asset** of `transformer` (role `rope-base` — ADR 0109
    // decision 4). It is 66 KB, so it is read whole (range reads are needed only for row-wise
    // tables like PLE).
    const ropeBase = parseRopeBase(await readWholeAsset(transformer.asset(ROPE_BASE)));
    await settleAbort(options.signal);
    const vaeDecoder = open(VAE_DECODER);

    await settleAbort(options.signal);

    // MUST: declared features can only be requested at device creation (ADR 0028). A shared GPU
    // cannot be asked, so the missing capability fails here by name — passing would fail later at
    // Session construction (or silently fall back to another path). {@link Pipeline.#admit}
    // already checks a shared GPU with the same function, but for an internally acquired device
    // this is the only check.
    const gpu = options.gpu ?? (await acquireGpu(toAcquireGpuOptions(gpuFeatures)));
    const ownsGpu = options.gpu === undefined;
    try {
      // MUST: the abort check **after** acquiring the GPU sits inside the try — outside it, the
      // internally acquired device would be left with nobody to release it (it rides on the same
      // clean-up as the feature check). Yield to a macrotask here too: the await of `acquireGpu`
      // resolves as a microtask continuation, so an abort task queued by a click during the wait
      // has not run yet.
      await settleAbort(options.signal);
      // The features and limits of the GPU in hand — one helper for every family (the reasons are in
      // its doc).
      assertAcquiredGpuCapabilities(
        gpuFeatures,
        quant.requiredLimits,
        gpu,
        `anima.Pipeline: quant '${quantName}'`,
      );
      return new Pipeline({
        gpu,
        ownsGpu,
        config,
        sessionOptions,
        tokenizers,
        textEncoder,
        textConditioner,
        transformer,
        ropeBase,
        vaeDecoder,
        residency,
        // Experimental (a measurement opt-in): the DiT and VAE Sessions get the rules.
        ...(options.fusionRules === undefined ? {} : { fusionRules: options.fusionRules }),
        ...(options.onRunDiagnostics === undefined
          ? {}
          : { onRunDiagnostics: options.onRunDiagnostics }),
      });
    } catch (error) {
      // An internally acquired GPU would have nobody to release it if construction fails, so it is
      // returned here.
      if (ownsGpu) gpu.destroy();
      throw error;
    }
  }

  /**
   * Builds from an already fetched manifest + assets. **This is the core**, and all of the
   * following is done at construction:
   *
   * - selecting `model` (an unknown model name fails loudly with the list of available ones —
   *   ADR 0041 §8)
   * - checking `pipeline`'s contract name and major (**an unknown major fails loudly** — the check
   *   is the models side's responsibility; ADR 0038 §1)
   * - hand-written schema validation of `pipelineConfig` (unknown keys fail loudly too)
   * - the **explicit mapping** of the quant's `session` → runtime `SessionOptions` and interpreting
   *   `gpuFeatures`
   * - `openContainer` for every weight / the rope base table / interpreting the two tokenizers
   *
   * Weight component keys are **accepted in both forms** — the single-form `krm` bytes
   * (`transformer`) and the parts of the split form (`transformer[0]` / `transformer[1]` / … — in
   * index order from part 0). The split form is opened as parts without concatenating the bytes. A
   * missing index or mixing with the single-form key fails loudly (the implementation is
   * `src/hub/components.ts`). The rope base table is a container asset of `transformer`, so it is
   * not listed in `assets` on this surface either.
   *
   * MUST: manifest contract violations and **asset parsing** fail **before acquiring the GPU** (the
   * same order as the other families). Out of order, in an environment without a GPU they turn into
   * another exception and the reader never learns "what was wrong". The only checks allowed after
   * acquiring the GPU are its capabilities — features (shader-f16 — ADR 0028) and the limits of an
   * internally acquired device (ADR 0089 decision 5 — `#admit` checks a shared GPU).
   * MUST: open no Session at all (the VRAM MUST — module doc).
   *
   * NOTE: each step is indivisible (there is no way to fold a 3.7GiB component midway), so the
   * checks of {@link PipelineOptions.signal} sit only at **the boundaries between steps**,
   * each yielding to the event loop once before checking ({@link settleAbort}) — an abort that
   * arrives during synchronous parsing takes effect at the next boundary (the same when
   * `options.gpu` is supplied).
   */
  static async fromAssets(input: Assets, options: PipelineOptions = {}): Promise<Pipeline> {
    const admitted = Pipeline.#admit(input.manifest, options);
    return await Pipeline.#build(admitted, input.assets, await assetOpener(input.assets), options);
  }

  /**
   * Generates one image from a prompt.
   *
   * The same seed and knobs give the same pixels (random numbers and rounding are deterministic on
   * the host — `random.ts` / `sampler.ts`).
   *
   * Concurrent calls **wait and run in order** (graphs are never resident at the same time — module
   * doc).
   */
  async generate(request: GenerateRequest): Promise<GeneratedImage> {
    // The disposed check happens at call time (checked inside the chain, generations accepted
    // before the dispose would fail along with it).
    if (this.#disposal !== undefined) {
      throw new Error("anima.Pipeline: cannot generate after dispose");
    }
    return await this.#chain(() => this.#generate(request));
  }

  async #generate(request: GenerateRequest): Promise<GeneratedImage> {
    const state = this.#state;
    const defaults = state.config.defaults;
    const steps = request.steps ?? defaults.steps;
    const guidance = request.guidanceScale ?? defaults.guidanceScale;
    const resolution = request.resolution ?? defaults.resolution;
    const seed = request.seed ?? 0;
    // The request's update rule takes precedence; only when omitted does it fall back to the
    // distributor's declaration. An unknown spelling fails before touching the GPU, showing
    // expected and actual (for JS calls that bypass the TS types — no silent fallback to the
    // default).
    const sampler =
      request.sampler === undefined
        ? state.config.scheduler.type
        : assertAnimaSamplerType(request.sampler, "sampler");
    assertAcceptableResolution(resolution);
    // The seed check sits at the entry too. The generator is created in the DiT stage
    // (`new Randn(seed)`), after the text encoder / conditioner have run and the DiT weights are
    // uploaded, so with the check only in `Randn` an invalid seed would fail after a GB-scale load.
    assertAcceptableSeed(seed);
    // steps is checked at the entry too (alongside the other knobs, so input-caused errors come
    // back before the prompt layer — the single owner of the accepted set is
    // `assertAcceptableSteps` on the ladder side. The ladder is built below, before touching the
    // GPU).
    assertAcceptableSteps(steps);
    // Checked in f32: CFG multiplies by the scale in f32 (`cfgEulerStep` /
    // `dpmSolverMultistepStep`), so a value finite in f64 but ±Infinity in f32 (such as
    // `Number.MAX_VALUE`) breaks the mix. A value finite in f32 that overflows in combination with
    // the DiT output fails, with the step named, in the post-update finiteness check
    // (`denoiseStep`).
    if (!Number.isFinite(Math.fround(guidance))) {
      throw new ModelInputError(
        `guidanceScale ${guidance} is not a finite number that fits in f32`,
      );
    }
    // Effective residency (request ?? construction option). It means "keep the DiT **after** this
    // generate" — a resident DiT present at the start is used whatever the value
    // (`transformerSource` in `residency.ts`).
    const residency =
      request.residency === undefined
        ? state.residency
        : assertAnimaResidency(request.residency, "residency");

    const wantsUncond = needsUncond(guidance);
    const negativePrompt = resolveNegativePrompt(
      request.negativePrompt,
      defaults.negativePrompt,
      guidance,
    );

    // The emitter of generation events. Without a subscriber it collapses into a no-op, so the
    // firing points have no branch.
    const { onEvent } = request;
    const emit: (event: GenerateEvent) => Promise<void> =
      onEvent === undefined
        ? () => Promise.resolve()
        : async (event) => {
            await onEvent(event);
          };
    const notifyResidency = (notice: ResidencyNotice): Promise<void> =>
      emit({ kind: "residency", component: "transformer", ...notice });
    /**
     * Headroom probe of pre-emptive eviction (the `probe` of
     * `TransformerResidency.ensureHeadroom`).
     */
    const probeHeadroom = (bytes: number): Promise<boolean> => fitsHeadroom(state.gpu, bytes);
    /**
     * Runs one stage other than the DiT (`stage` events wrap Session construction and release — if
     * the stage fails midway, `end` is not emitted). An OOM while a resident DiT exists evicts it
     * and re-runs the stage once (`TransformerResidency.runStage`). The text stages / VAE do not
     * take the quant's session; the VAE alone takes its rule list ({@link animaStageSessionOptions}).
     */
    const withStage = async <T>(
      component: Exclude<RunComponent, "transformer">,
      model: ModelComponent,
      body: (run: StageRun) => Promise<T>,
    ): Promise<T> => {
      await emit({ kind: "stage", component, at: "start" });
      const result = await this.#residency.runStage(
        () =>
          withSession(
            state.gpu,
            model,
            animaStageSessionOptions(component, state.fusionRules),
            observer(state, component),
            body,
          ),
        notifyResidency,
      );
      await emit({ kind: "stage", component, at: "end" });
      return result;
    };

    // --- ① Prompt layer (no GPU, deterministic) ------------------------------------
    const positive = state.tokenizers.encode(request.prompt, "prompt");
    const negative = wantsUncond
      ? state.tokenizers.encode(negativePrompt as string, "negative prompt")
      : undefined;
    const sigmas = sigmaSchedule(steps, state.config.scheduler.shift);

    // When the generate fails in a stage that touches the GPU (the text stages onwards), apply the
    // effective per-stage instruction "do not keep it after this generate" (a failure before the
    // DiT stage — a text stage, a throw from the `stage` onEvent, etc. — also releases the
    // carried-over DiT). A failure inside the DiT stage is settled by `runTransformer`. A request
    // rejected by the input checks (above) touches neither the GPU nor the resident slot.
    try {
      // --- ①' Pre-emptive eviction (before the text stages; carried-over resident DiT only) ---
      // If the two text stages do not fit on top of the resident DiT, evict before hitting an OOM
      // ("The two lines of eviction" in `residency.ts`). A throw from the probe (validation /
      // device loss) flows as is to the failure path below. `holding` is checked here too so that
      // per-stage operation does not pay for the estimate (the decision whether to measure belongs
      // to `ensureHeadroom`).
      if (this.#residency.holding) {
        await this.#residency.ensureHeadroom(
          textStagesNeed(state, negative === undefined ? [positive] : [positive, negative]),
          probeHeadroom,
          notifyResidency,
        );
      }

      // --- ② Text path (released before the DiT load) ---------------------------
      const hidden = await withStage("text_encoder", state.textEncoder, (run) =>
        Promise.all([
          run({ input_ids: idsTensor(positive.qwenIds) }),
          ...(negative === undefined ? [] : [run({ input_ids: idsTensor(negative.qwenIds) })]),
        ]),
      );
      const embeds = await withStage("text_conditioner", state.textConditioner, (run) =>
        Promise.all([
          run({ source_hidden_states: hidden[0], target_input_ids: idsTensor(positive.t5Ids) }),
          ...(negative === undefined
            ? []
            : [
                run({
                  source_hidden_states: hidden[1],
                  target_input_ids: idsTensor(negative.t5Ids),
                }),
              ]),
        ]),
      );

      // --- ③ denoise (DiT for N steps; released before the VAE load in per-stage operation) ---
      // The low-precision knobs (the quant's session) apply **only to the DiT Session** — the
      // text stages / VAE are excluded (keeping the axis of comparison to the DiT alone). The
      // Session's source (resident / newly resident / per stage) and what happens after the stage
      // are decided by `TransformerResidency` (ADR 0112). The body is called again **from the
      // start** on a re-run after an OOM eviction, so the denoise state (random numbers, latent,
      // DPM history) is created inside the body.
      const observeTransformer = observer(state, "transformer");
      await emit({ kind: "stage", component: "transformer", at: "start" });
      const { latents, latentShape } = await this.#residency.runTransformer({
        effective: residency,
        open: () =>
          state.transformer.createSession(
            state.gpu,
            withFusionRules(state.sessionOptions, state.fusionRules),
          ),
        notify: notifyResidency,
        body: async (session) => {
          const model = state.transformer;
          const run = stageRun(session, model, observeTransformer);
          const plan = planDynDit(model, state.ropeBase, resolution);
          const [, projWidth] = staticInputShape(model, "timesteps_proj");
          const embedShape = staticInputShape(model, "encoder_hidden_states");
          const [, rows, width] = embedShape;
          if (embeds[0].shape[2] !== width) {
            throw new Error(
              `Conditioner output width ${embeds[0].shape[2]} does not match the DiT's ${width}`,
            );
          }
          /**
           * Runs one latent through the DiT. It only wraps patchify at the entry and unpatchify at
           * the exit, so the caller (CFG / Euler) sees only latents.
           */
          const predict = async (
            current: Float32Array<ArrayBuffer>,
            proj: Tensor,
            embed: Float32Array<ArrayBuffer>,
          ): Promise<Float32Array<ArrayBuffer>> => {
            const output = await run({
              tokens: {
                dtype: "f32",
                shape: plan.tokenShape,
                data: patchifyLatents(current, plan.latentShape, plan.geometry),
              },
              timesteps_proj: proj,
              encoder_hidden_states: { dtype: "f32", shape: embedShape, data: embed },
              rope_cos: { dtype: "f32", shape: plan.ropeShape, data: plan.cos },
              rope_sin: { dtype: "f32", shape: plan.ropeShape, data: plan.sin },
            });
            return unpatchifyTokens(
              asF32(output, "DiT output (S form)"),
              plan.latentShape,
              plan.geometry,
            );
          };
          const padded = embeds.map((embed) => padSequence(embed, rows));
          const elements = plan.latentShape.reduce((a, b) => a * b, 1);
          let current = new Randn(seed).normals(elements);
          // The only history DPM++ 2M reads (absent at step 0). Always undefined on the Euler path.
          let previousX0: Float32Array<ArrayBuffer> | undefined;
          for (let index = 0; index < steps; index += 1) {
            const proj: Tensor = {
              dtype: "f32",
              shape: [1, projWidth],
              data: timestepsProj(
                sigmas[index],
                projWidth,
                state.config.scheduler.numTrainTimesteps,
              ),
            };
            const predictions: Float32Array<ArrayBuffer>[] = [];
            for (const embed of padded) predictions.push(await predict(current, proj, embed));
            const update = denoiseStep(sampler, {
              sample: current,
              cond: predictions[0],
              uncond: predictions[1],
              guidance,
              sigmas,
              index,
              previousX0,
            });
            current = update.sample;
            previousX0 = update.previousX0;
            await emit({
              kind: "denoise-step",
              step: index + 1,
              steps,
              sigma: sigmas[index],
              copyLatents: latentSnapshot(current, plan.latentShape),
            });
          }
          return { latents: current, latentShape: plan.latentShape };
        },
      });
      await emit({ kind: "stage", component: "transformer", at: "end" });

      // --- ③' Pre-emptive eviction (before the VAE stage; resident DiT at this point only) ---
      // Measures here the two cases the measurement before the text stages cannot cover — the
      // DiT was created in this generate and put in the slot (the first generate), and a
      // carried-over DiT grew its backing at a new resolution ("The two lines of eviction" in
      // `residency.ts`). The DiT stage has already benefited from residency, so evicting as late as
      // possible is cheaper.
      if (this.#residency.holding) {
        await this.#residency.ensureHeadroom(vaeStageNeed(state), probeHeadroom, notifyResidency);
      }

      // --- ④ Denormalise → VAE decode (always tiled — ADR 0038 §4) -------------------
      const { mean: latentsMean, std: latentsStd } = latentStats();
      const denormalized = denormalizeLatents(latents, latentShape, latentsMean, latentsStd);
      const decoded = await withStage("vae_decoder", state.vaeDecoder, async (run) => {
        const tileShape = staticInputShape(state.vaeDecoder, "latents");
        const sampleShape = staticOutputShape(state.vaeDecoder);
        const geometry = planVaeTiling(latentShape, tileShape, sampleShape);
        const tiles = tileCount(geometry);
        const blendsOf = (axis: typeof geometry.rows): number[] =>
          axis.starts.slice(1).map((_, index) => blendExtentAt(axis, geometry.scale, index + 1));
        let decodedTiles = 0;
        const pixels = await decodeTiled(denormalized, geometry, async (tile, row, col) => {
          const output = await run({ latents: { dtype: "f32", shape: tileShape, data: tile } });
          const sample = asF32(output, `VAE output (tile ${row},${col})`);
          decodedTiles += 1;
          await emit({ kind: "vae-tile", tile: decodedTiles, tiles });
          return sample;
        });
        // The **geometry** is the source of truth for the size (latent length × scale). Do not
        // derive it back from the pixel count — for non-square sizes `3·H·W` does not factor
        // uniquely, and a derivation could never detect a height / width mix-up.
        return {
          pixels,
          width: geometry.cols.extent * geometry.scale,
          height: geometry.rows.extent * geometry.scale,
          tiles,
          // Blend widths are **per pair** (with rounded equal spacing they vary by up to 1 latent).
          // The diagnostics carry the geometry itself; folding it to a representative value would
          // erase the clue to a mismatch.
          blend: [blendsOf(geometry.rows), blendsOf(geometry.cols)] as const,
        };
      });

      // MUST: the size of the output image is decided by **the assets**, not the knobs. A mismatch
      // here means the opened export differs from what was expected, so do not return silently.
      // Consistency with the element count is checked by `imageToRgba`.
      if (decoded.width !== resolution.width || decoded.height !== resolution.height) {
        throw new Error(
          `VAE output ${decoded.width}×${decoded.height} does not match the requested resolution ${formatResolution(
            resolution,
          )} (${decoded.tiles} tiles / blend vertical [${decoded.blend[0]}] horizontal [${
            decoded.blend[1]
          }]px)`,
        );
      }

      const rgba = imageToRgba(decoded.pixels, decoded.width, decoded.height);
      return {
        width: decoded.width,
        height: decoded.height,
        data: new Uint8Array(rgba.buffer),
      };
    } catch (error) {
      throw await this.#residency.releaseIfRequested(error, residency, notifyResidency);
    }
  }

  /**
   * Releases. The resident DiT ({@link PipelineOptions.residency}) is always folded, and
   * **only an internally acquired GPU** is destroyed (a GpuContext passed as `options.gpu` belongs
   * to the caller and is not touched).
   *
   * dispose is required even with a shared `options.gpu` — the resident DiT (about 2.6 GiB at 1024²
   * with the default preset) stays on the shared device. If the device is lost, disposing the
   * resident DiT (the Session's flush) fails and dispose may reject (an internally acquired GPU is
   * destroyed anyway).
   *
   * MUST: wait for in-flight generation to finish before destroying (flush-before-destroy) —
   * putting the destruction on the chain too decides the order of waiting and destroying in one
   * place. Later calls return the same completion (returning earlier would let the caller treat it
   * as "destroyed" and move on).
   */
  dispose(): Promise<void> {
    // MUST: fold the resident DiT first, then destroy the GPU (flush-before-destroy). The resident
    // DiT belongs to the pipeline even with a shared GPU (`options.gpu`), so it is always folded.
    // Even if folding fails, an internally acquired GPU is returned (the doc of `disposeSteps`).
    this.#disposal ??= this.#chain(() =>
      disposeResidencyThenGpu(
        this.#residency,
        this.#state.ownsGpu ? () => this.#state.gpu.destroy() : undefined,
      ),
    );
    return this.#disposal;
  }

  /** `await using` support (Explicit Resource Management) — an alias of {@link dispose}. */
  [Symbol.asyncDispose](): Promise<void> {
    return this.dispose();
  }
}
