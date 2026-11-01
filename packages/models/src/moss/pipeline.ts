/**
 * `Pipeline` — MOSS-Image text → image over its distribution `karume-moss-image` (ADR 0138 and its
 * addendum of 2026-11-01; the text encoder is ADR 0137's).
 *
 * The stages are the family's generate path (`./generate.ts` — ADR 0138 decision 5): the chat
 * template and the tokenizer on the host, then the Qwen3.5 text encoder, `dit_text`, the DiT pair
 * (`dit` and `dit_skip22`, which borrows `dit`'s weights) and the three VAE decoder graphs on the
 * GPU, each stage's Sessions closed before the next stage opens, with a headroom check before each.
 * This module adds what a distribution brings:
 *
 * - **The preset** is the manifest's: `quant` when given, else the model's `defaultQuant`
 *   (`f32+dit8-a8-attn8-online` in `karume-moss-image` — ADR 0128 decision 1); its knobs are the
 *   family table's (`./preset.ts` — `mossPresetOf` checks that the quant's `session` declares
 *   exactly them).
 * - **The request defaults** are the manifest's `pipelineConfig` (upstream's — `./config.ts`):
 *   width, height, steps, guidance, shift, and the negative prompt, which fills `negativePrompt`
 *   only while guidance ≠ 1 (at guidance 1 a negative prompt has no effect and is refused when
 *   given — ADR 0138 decision 9). The seed defaults to 0 (exclusive with `latents`), as in every
 *   family.
 * - **The text is a prompt string.** The family's other form — precomputed encoder outputs, which
 *   the parity checks feed through the internal entry — is not on this surface.
 *
 * ## Construction ({@link Pipeline.fromPretrained})
 *
 * `loadManifest` → `resolveSelection` → each component's descriptor (part 0) → the admission
 * ({@link admitMoss} — every check that needs no weight byte: the pipeline name and major,
 * `pipelineConfig`, the preset, the shared GPU's features and limits, the three contracts and the
 * container assets they read) → the weight parts → the tokenizer asset → the tokenizer's bound →
 * the GPU (acquired unless one is passed). No Session is built: every stage builds and closes its
 * own per generation (the text encoder alone holds about 7.4 GiB of weights on the device).
 *
 * MUST: `generate` runs on one serialisation chain (concurrent calls wait in turn — the stages'
 * headroom checks assume nothing else of this pipeline is open), and `dispose` waits for it before
 * destroying an internally acquired GPU.
 *
 * The published distribution is `SOURCES["moss-image"]` (`./config.ts`); a local distribution is
 * passed as a source handle (`nodeDirectory` of `@karume/hub/node`, `denoDirectory` of
 * `@karume/hub/deno`, or a served directory in the browser).
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state).
 */

import { acquireGpu, type GpuContext } from "@karume/runtime";
import {
  type DistributionSource,
  type GpuFeaturesSpec,
  type HubRepoRef,
  loadManifest,
  type Manifest,
  type Quant,
  resolveSelection,
} from "@karume/hub";

import { settleAbort } from "../concurrency/abort.ts";
import { createOperationChain } from "../concurrency/serial.ts";
import { readAssetJson } from "../hub/asset-readers.ts";
import { type ComponentOpener, loadContainerComponents } from "../hub/components.ts";
import {
  type FromPretrainedComponentOptions,
  type FromPretrainedHubOptions,
  hubLoadOptions,
} from "../hub/load-options.ts";
import { toManifestSource } from "../hub/repo-ref.ts";
import { parseQwen3_5TokenizerAsset, Qwen3_5Tokenizer } from "../qwen3_5/tokenizer.ts";
import {
  assertGpuFeaturesGranted,
  assertRequiredLimitsBeforeDownload,
  assertRequiredLimitsSatisfied,
  sessionGpuFeatures,
  toAcquireGpuOptions,
} from "../session/gpu-features.ts";
import {
  MOSS_PIPELINE_MAJOR,
  MOSS_PIPELINE_NAME,
  MOSS_TOKENIZER_ASSET,
  type MossPipelineConfig,
  parseMossPipelineConfig,
} from "./config.ts";
import {
  MOSS_DIT,
  MOSS_DIT_SKIP22,
  MOSS_DIT_TEXT,
  type MossDitContract,
  mossDitContract,
  mossDitGraphContract,
  readMossDitAssets,
} from "./dit-contract.ts";
import { generateMoss, type MossGenerateRequest, type MossGenerateState } from "./generate.ts";
import { type MossPreset, mossPresetOf } from "./preset.ts";
import { mossSchedule } from "./sampler.ts";
import {
  MOSS_TEXT_ENCODER,
  mossTextEncoderContract,
  mossTextEncoderGraphContract,
} from "./text-stage.ts";
import {
  MOSS_VAE_DECODER_ATTN,
  MOSS_VAE_DECODER_POST,
  MOSS_VAE_DECODER_PRE,
  type MossVaeContract,
  mossVaeContract,
  mossVaeGraphContract,
  readMossLatentBatchNorm,
} from "./vae-contract.ts";

/** The public name in messages (ADR 0136 decision 5). */
const OWNER = "moss.Pipeline";

/** The family name the shared stages and contracts put in their messages. */
const FAMILY = "MOSS-Image";

/** The components a generation reads (the manifest's weights names), in the stages' order. */
const COMPONENT_KEYS: readonly string[] = [
  MOSS_TEXT_ENCODER,
  MOSS_DIT_TEXT,
  MOSS_DIT,
  MOSS_DIT_SKIP22,
  MOSS_VAE_DECODER_PRE,
  MOSS_VAE_DECODER_ATTN,
  MOSS_VAE_DECODER_POST,
];

/** A generated image: `data` is `[3, height, width]` f32 in `[-1, 1]` (upstream's decode, clamped). */
export type GeneratedImage = {
  readonly width: number;
  readonly height: number;
  readonly data: Float32Array<ArrayBuffer>;
};

/**
 * One generation request. A field left out takes the manifest's `pipelineConfig` default
 * (upstream's: 1024 × 1024, 50 steps, guidance 3, shift 5, the empty negative prompt).
 */
export type GenerateRequest = {
  /** The prompt (any string the chat template and the tokenizer accept, up to 4,096 tokens). */
  readonly prompt: string;
  /**
   * The negative prompt (default `pipelineConfig.defaults.negativePrompt` while guidance ≠ 1).
   * Giving one at guidance 1 is a `ModelInputError` (no negative branch runs, so it would have no
   * effect).
   */
  readonly negativePrompt?: string;
  /**
   * What to do with a prompt over 4,096 tokens: `"reject"` (default — a `ModelInputError`) or
   * `"truncate"`, which cuts it as upstream does (the maintainer's ruling 50).
   */
  readonly overflow?: "reject" | "truncate";
  /** Pixels, a positive multiple of 16 (default `pipelineConfig.defaults.width`). */
  readonly width?: number;
  /** Pixels, a positive multiple of 16 (default `pipelineConfig.defaults.height`). */
  readonly height?: number;
  /**
   * The requested step count (default `pipelineConfig.defaults.steps`). Upstream snaps the schedule
   * to its 1,000-point training grid and de-duplicates it, so it may run fewer.
   */
  readonly steps?: number;
  /** The CFG scale, finite and ≥ 0 (default `pipelineConfig.defaults.guidance`; 1 runs no negative branch). */
  readonly guidance?: number;
  /** The flow-matching shift, finite and > 0 (default `pipelineConfig.scheduler.shift`). */
  readonly shift?: number;
  /**
   * The seed of the initial noise (default 0 — exclusive with {@link GenerateRequest.latents}).
   * Karume's own generator, not torch's: the same seed does not give upstream's image.
   */
  readonly seed?: number;
  /** The initial latents `[128, height/16, width/16]` f32 (exclusive with {@link GenerateRequest.seed}). */
  readonly latents?: Float32Array<ArrayBuffer>;
};

/** Construction options. */
export type PipelineOptions = {
  /** The model (a key of the manifest's models). Defaults to `defaultModel`. */
  readonly model?: string;
  /** The preset (a key of that model's quants). Defaults to `defaultQuant`. */
  readonly quant?: string;
  /**
   * Shares an existing GPU (the caller keeps ownership — `dispose` does not destroy it). When
   * omitted the pipeline calls `acquireGpu` (subgroups requested when available, for the online
   * attention preset) and destroys it in `dispose`.
   */
  readonly gpu?: GpuContext;
  /**
   * Cancels the construction (the fetch layer and the assembly steps); `signal.reason` is thrown
   * as is. Not carried over to generation.
   */
  readonly signal?: AbortSignal;
};

/** The options of {@link Pipeline.fromPretrained} (the fetch-layer knobs pass through to hub). */
export type FromPretrainedOptions = PipelineOptions &
  FromPretrainedHubOptions &
  FromPretrainedComponentOptions;

/** What the manifest alone settles ({@link mossSelectionOf}). */
export type MossSelection = {
  readonly config: MossPipelineConfig;
  readonly quantName: string;
  readonly quant: Quant;
  readonly preset: MossPreset;
  readonly gpuFeatures: GpuFeaturesSpec | undefined;
};

/** What the admission ({@link admitMoss}) settles before the weights are fetched. */
type MossAdmission = MossSelection & {
  readonly dit: MossDitContract;
  readonly vae: MossVaeContract;
};

/** The internal state of a pipeline. */
type MossPipelineState = {
  readonly generate: MossGenerateState;
  readonly config: MossPipelineConfig;
  readonly ownsGpu: boolean;
};

/**
 * The internal entry's request of a public request (module doc "The request defaults"). Only the
 * public fields are carried over (the embeddings form stays internal).
 *
 * NOTE: `export` is for the host tests (not on a public surface).
 */
export const mossRequestOf = (
  request: GenerateRequest,
  config: MossPipelineConfig,
): MossGenerateRequest => {
  const guidance = request.guidance ?? config.defaults.guidance;
  // The default negative applies only where a negative branch runs; at guidance 1 an explicit one
  // is passed through so that the acceptance check refuses it.
  const negativePrompt =
    request.negativePrompt ?? (guidance === 1 ? undefined : config.defaults.negativePrompt);
  const { seed, latents, overflow } = request;
  return {
    prompt: request.prompt,
    ...(negativePrompt === undefined ? {} : { negativePrompt }),
    ...(overflow === undefined ? {} : { overflow }),
    width: request.width ?? config.defaults.width,
    height: request.height ?? config.defaults.height,
    steps: request.steps ?? config.defaults.steps,
    guidance,
    shift: request.shift ?? config.scheduler.shift,
    // Both or neither are passed through as given, so that the acceptance check refuses both; the
    // default seed fills only the request that gives neither.
    ...(seed === undefined ? {} : { seed }),
    ...(latents === undefined ? {} : { latents }),
    ...(seed === undefined && latents === undefined ? { seed: 0 } : {}),
  };
};

/**
 * The default steps × shift pair gives a σ schedule on the `dit` container's training grid —
 * otherwise a request that sets neither would fail as an input error for the distribution's value.
 */
const assertDefaultSchedule = (config: MossPipelineConfig, dit: MossDitContract): void => {
  const { steps } = config.defaults;
  const { shift } = config.scheduler;
  try {
    mossSchedule(dit.trainingGrid, steps, shift);
  } catch (error) {
    throw new Error(
      `${OWNER}: pipelineConfig's defaults.steps ${steps} with scheduler.shift ${shift} give no σ` +
        ` schedule (${error instanceof Error ? error.message : String(error)})`,
      { cause: error },
    );
  }
};

/**
 * The part of the admission the manifest alone decides (module doc "The preset" and "The request
 * defaults"): the model, its pipeline contract, `pipelineConfig`, the preset (`quant` or
 * `defaultQuant`, its knobs checked against the family table) and the GPU features it needs, and
 * the tokenizer asset's declaration. Every failure is a plain `Error` naming what there is.
 *
 * NOTE: `export` is for the host tests and the mirror's e2e (not on a public surface).
 */
export const mossSelectionOf = (
  manifest: Manifest,
  options: Pick<PipelineOptions, "model" | "quant">,
): MossSelection => {
  const modelName = options.model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, modelName)) {
    throw new Error(
      `${OWNER}: model '${modelName}' is not in the manifest` +
        ` (available: ${manifest.available.models.join(" / ")})`,
    );
  }
  const entry = manifest.models[modelName];
  const { name, major } = entry.pipeline;
  if (name !== MOSS_PIPELINE_NAME || major !== MOSS_PIPELINE_MAJOR) {
    throw new Error(
      `${OWNER}: the manifest's pipeline is '${name}/${major}' (this implementation reads` +
        ` '${MOSS_PIPELINE_NAME}/${MOSS_PIPELINE_MAJOR}')`,
    );
  }
  const config = parseMossPipelineConfig(entry.pipelineConfig);
  const quantName = options.quant ?? entry.defaultQuant;
  if (!Object.hasOwn(entry.quants, quantName)) {
    throw new Error(
      `${OWNER}: quant '${quantName}' is not in the manifest` +
        ` (available: ${entry.available.quants.join(" / ")})`,
    );
  }
  const quant = entry.quants[quantName];
  const where = `${OWNER}: quant '${quantName}'`;
  const preset = mossPresetOf(quantName, quant.session, where);
  const gpuFeatures = sessionGpuFeatures(quant.gpuFeatures, preset.dit);
  // The tokenizer arrives after the weights; its declaration is checked before them.
  if (!Object.hasOwn(entry.assets, MOSS_TOKENIZER_ASSET)) {
    throw new Error(
      `${OWNER}: the manifest's assets have no '${MOSS_TOKENIZER_ASSET}' (the Qwen3.5 tokenizer;` +
        ` assets: ${Object.keys(entry.assets).join(" / ")})`,
    );
  }
  return { config, quantName, quant, preset, gpuFeatures };
};

/**
 * The admission (module doc "Construction") — every check that needs no weight byte, run before
 * the weight parts are fetched. MUST: the family's checks sit here, not in a later step, where
 * they would fail only after about 12 GiB were downloaded (ADR 0070 decision 5).
 */
const admitMoss = async (
  manifest: Manifest,
  open: ComponentOpener,
  options: Pick<PipelineOptions, "model" | "quant" | "gpu">,
): Promise<MossAdmission> => {
  const selection = mossSelectionOf(manifest, options);
  const { config, quant, gpuFeatures, quantName } = selection;
  if (options.gpu !== undefined) {
    // A shared GPU's missing capability is known now (features are requested only at device
    // creation — ADR 0028).
    const where = `${OWNER}: quant '${quantName}'`;
    assertGpuFeaturesGranted(gpuFeatures, options.gpu, where);
    assertRequiredLimitsSatisfied(quant.requiredLimits, options.gpu.limits, where);
  }
  mossTextEncoderGraphContract(open(MOSS_TEXT_ENCODER));
  const ditComponents = {
    dit: open(MOSS_DIT),
    ditSkip22: open(MOSS_DIT_SKIP22),
    ditText: open(MOSS_DIT_TEXT),
  };
  const ditGraph = mossDitGraphContract(ditComponents, FAMILY);
  const { ropeBase, trainingGrid } = await readMossDitAssets(ditComponents.dit);
  const dit = mossDitContract(ditGraph, ropeBase, trainingGrid, FAMILY);
  assertDefaultSchedule(config, dit);
  const vaeComponents = {
    pre: open(MOSS_VAE_DECODER_PRE),
    attn: open(MOSS_VAE_DECODER_ATTN),
    post: open(MOSS_VAE_DECODER_POST),
  };
  const vae = mossVaeContract(
    mossVaeGraphContract(vaeComponents, FAMILY),
    await readMossLatentBatchNorm(vaeComponents.pre),
    FAMILY,
  );
  return { ...selection, dit, vae };
};

/** Fetches and builds (the body of {@link Pipeline.fromPretrained} — module doc "Construction"). */
const loadMoss = async (
  ref: string | HubRepoRef | DistributionSource,
  options: FromPretrainedOptions,
): Promise<MossPipelineState> => {
  const source = toManifestSource(
    ref,
    `${OWNER}.fromPretrained`,
    'SOURCES["moss-image"] (@karume/models/moss)',
  );
  const hubOptions = hubLoadOptions(options);
  const loaded = await loadManifest(source, hubOptions);
  const selection = resolveSelection(loaded.manifest, {
    ...(options.model === undefined ? {} : { model: options.model }),
    ...(options.quant === undefined ? {} : { quant: options.quant }),
  });
  const { admitted, assets, open } = await loadContainerComponents(
    `${OWNER}.fromPretrained`,
    loaded,
    selection,
    COMPONENT_KEYS,
    async (opener) => {
      await settleAbort(options.signal);
      const admission = await admitMoss(loaded.manifest, opener, options);
      // The limits the distribution declares, before the weight parts are fetched (ADR 0089
      // decision 5 — a shared GPU's limits, or the adapter's on the path that acquires its own).
      await assertRequiredLimitsBeforeDownload(
        admission.quant.requiredLimits,
        options.gpu,
        `${OWNER}: quant '${admission.quantName}'`,
      );
      return admission;
    },
    {
      ...hubOptions,
      ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
      ...(options.components === undefined ? {} : { components: options.components }),
    },
  );
  await settleAbort(options.signal);
  const tokenizer = new Qwen3_5Tokenizer(
    parseQwen3_5TokenizerAsset(
      readAssetJson(OWNER, "weights / assets", assets, MOSS_TOKENIZER_ASSET),
      MOSS_TOKENIZER_ASSET,
    ),
  );
  const textEncoder = open(MOSS_TEXT_ENCODER);
  const textContract = mossTextEncoderContract(textEncoder, tokenizer);
  const { preset, gpuFeatures, quantName } = admitted;
  const gpu = options.gpu ?? (await acquireGpu(toAcquireGpuOptions(gpuFeatures, preset.dit)));
  const ownsGpu = options.gpu === undefined;
  try {
    // Inside the try: an internally acquired device must not be left without an owner.
    await settleAbort(options.signal);
    assertGpuFeaturesGranted(gpuFeatures, gpu, `${OWNER}: quant '${quantName}'`);
    if (ownsGpu) {
      assertRequiredLimitsSatisfied(
        admitted.quant.requiredLimits,
        gpu.limits,
        `${OWNER}: quant '${quantName}'`,
      );
    }
    return {
      generate: {
        gpu,
        preset,
        dit: {
          // The same opener the admission saw (it only looks up the opened components).
          components: {
            dit: open(MOSS_DIT),
            ditSkip22: open(MOSS_DIT_SKIP22),
            ditText: open(MOSS_DIT_TEXT),
          },
          contract: admitted.dit,
        },
        vae: {
          components: {
            pre: open(MOSS_VAE_DECODER_PRE),
            attn: open(MOSS_VAE_DECODER_ATTN),
            post: open(MOSS_VAE_DECODER_POST),
          },
          contract: admitted.vae,
        },
        text: { component: textEncoder, tokenizer, contract: textContract },
      },
      config: admitted.config,
      ownsGpu,
    };
  } catch (error) {
    if (ownsGpu) gpu.destroy();
    throw error;
  }
};

/**
 * The MOSS-Image text → image pipeline (module doc).
 *
 * The only construction entry is {@link Pipeline.fromPretrained} (the constructor is private — no
 * half-built state that bypasses the manifest checks — ADR 0008).
 */
export class Pipeline {
  readonly #state: MossPipelineState;
  /** The serialisation chain of generate and dispose. */
  readonly #chain = createOperationChain();
  /** The one dispose (defined means "disposed"). */
  #disposal: Promise<void> | undefined;

  private constructor(state: MossPipelineState) {
    this.#state = state;
  }

  /**
   * Fetches from the distribution and builds (module doc "Construction"). **`ref` is required**
   * (no default source — `src/hub/repo-ref.ts`): the published distribution
   * `SOURCES["moss-image"]`, any `{ repo, revision }`, or a local one as a source handle, e.g.
   * `nodeDirectory("models/karume-moss-image")` of `@karume/hub/node`. A component can be replaced
   * by the same role of another repository with an identical graph descriptor ({@link FromPretrainedComponentOptions.components}).
   */
  static async fromPretrained(
    ref: string | HubRepoRef | DistributionSource,
    options: FromPretrainedOptions = {},
  ): Promise<Pipeline> {
    return new Pipeline(await loadMoss(ref, options));
  }

  /**
   * Generates one image. The same request (seed or latents, knobs) on the same device gives the
   * same values. Concurrent calls wait and run in turn.
   */
  async generate(request: GenerateRequest): Promise<GeneratedImage> {
    if (this.#disposal !== undefined) throw new Error(`${OWNER}: cannot generate after dispose`);
    return await this.#chain(async () => {
      const { image, width, height } = await generateMoss(
        this.#state.generate,
        mossRequestOf(request, this.#state.config),
      );
      return { width, height, data: image };
    });
  }

  /**
   * Releases resources. Destroys **only a GPU acquired internally** (`options.gpu` belongs to the
   * caller). MUST: waits for the generation in flight first (flush before destroy — on the chain).
   */
  dispose(): Promise<void> {
    this.#disposal ??= this.#chain(() => {
      if (this.#state.ownsGpu) this.#state.generate.gpu.destroy();
    });
    return this.#disposal;
  }

  /** `await using` support — an alias of {@link dispose}. */
  [Symbol.asyncDispose](): Promise<void> {
    return this.dispose();
  }
}
