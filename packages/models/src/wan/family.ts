/**
 * The body of Wan's family admission, construction and generation (ADR 0121 decision 10 — one body
 * shared by the Wan2.1 / 2.2 classes).
 *
 * The public classes (`Pipeline` in `./pipeline.ts`, `Ti2vPipeline` in `./ti2v-pipeline.ts`) are
 * thin shells (private constructor, serialization chain, `dispose`);
 * the manifest checks, component matching, GPU acquisition and the stage order ((encoder →) text → DiT
 * → VAE) live here. The stage order, Session lifetimes and the cancellation MUSTs are defined by the
 * doc at the top of `./pipeline.ts`. Every value that differs by generation is carried by
 * {@link WanFamilySpec} (message prefix, pipeline name and major, generation descriptor). Whether the
 * I2V encoder components exist is also decided by the generation descriptor (a DiT input form of
 * `"ti2v"`).
 *
 * NOTE: internal only (not on the `mod.ts` / subpath surface — ADR 0008). The public types are taken
 * from `./pipeline.ts` and `./ti2v-pipeline.ts` with `import type` only (no value imports — both use
 * values from here, so no value import cycle is created).
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state —
 * CLAUDE.md).
 */

import {
  acquireGpu,
  type GpuContext,
  type SessionDiagnostics,
  type SessionOptions,
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

import { settleAbort } from "../concurrency/abort.ts";
import type { PipelineFusionRules } from "../session/fusion-rules.ts";
import { type FamilySessionPolicy, resolveSessionOptions } from "../session/options.ts";
import {
  assertAcquiredGpuCapabilities,
  assertGpuFeaturesGranted,
  assertRequiredLimitsBeforeDownload,
  assertRequiredLimitsSatisfied,
  sessionGpuFeatures,
  toAcquireGpuOptions,
} from "../session/gpu-features.ts";
import { readAssetBuffer, readWholeAsset } from "../hub/asset-readers.ts";
import {
  assetComponentOpener,
  type ComponentOpener,
  loadContainerComponents,
  type ModelComponent,
} from "../hub/components.ts";
import { toManifestSource } from "../hub/repo-ref.ts";
import { hubLoadOptions } from "../hub/load-options.ts";
import {
  parseWanPipelineConfig,
  WAN_PIPELINE_MAJOR,
  WAN_PIPELINE_NAME,
  WAN_TI2V_PIPELINE_MAJOR,
  WAN_TI2V_PIPELINE_NAME,
  type WanPipelineConfig,
} from "./config.ts";
import { parseWanRopeBase, type WanRopeBase } from "./dit-rope.ts";
import {
  WAN21_GENERATION,
  WAN22_TI2V_GENERATION,
  type WanGenerationDescriptor,
} from "./descriptor.ts";
import {
  type DitContract,
  ditContract,
  ROPE_BASE,
  runWanDenoise,
  TRANSFORMER,
  type WanContexts,
  wanDitPatch,
} from "./dit-loop.ts";
import {
  type PlanLayout,
  planWanRequest,
  type WanGenerationKnobs,
  type WanGenerationPlan,
  type WanVaeTile32Plan,
} from "./plan.ts";
import type {
  GeneratedVideo,
  Assets,
  FromPretrainedOptions,
  GenerateEvent,
  GenerateRequest,
  PipelineOptions,
  RunComponent,
} from "./pipeline.ts";
import type { Ti2vGenerateRequest } from "./ti2v-pipeline.ts";
import {
  admitWanText,
  assertTextEncoderDeclared,
  assertTokenizerDeclared,
  encodeWanPrompts,
  gpuPromptGate,
  loadWanTextStage,
  precomputedContexts,
  precomputedPromptGate,
  TEXT_ENCODER,
  textEncoderRouteOf,
  type WanTextAdmission,
  type WanTextEncoderRoute,
  type WanTextStage,
} from "./text-stage.ts";
import type { WanTextEmbedding, WanTextEmbeds } from "./text-embeds.ts";
import type { WanPromptEncoder } from "./text/tokenizer.ts";
import {
  assertWanVaeMatchesGeneration,
  assertWanVaeTilesCover,
  decodeWanVaeStage,
  VAE_DECODER_FIRST,
  VAE_DECODER_FIRST_32,
  VAE_DECODER_NEXT,
  VAE_DECODER_NEXT_32,
  type WanVaeTile32,
  type WanVaeTile32Contract,
  wanVaeTile32Contract,
} from "./tile-decode.ts";
import { type WanVaeChunkLayout, wanVaeChunkLayout } from "./vae-chunks.ts";
import {
  encodeWanImageStage,
  VAE_ENCODER_ATTN,
  VAE_ENCODER_POST,
  VAE_ENCODER_PRE,
  WAN_VAE_ENCODER_KEYS,
  type WanVaeEncoder,
  type WanVaeEncoderContract,
  wanVaeEncoderContract,
} from "./vae-encoder.ts";

/**
 * The components fetched per generation and path (`componentKeys` of the fetch surface, the
 * components the full-asset surface opens). A generation whose DiT input form is `"ti2v"` (Wan2.2)
 * fetches the three I2V VAE encoder components even for use without an image (no extra construction
 * entry point — the distribution always carries them).
 *
 * MUST: `"precomputed"` does not open the umT5 component — opening it would add the umT5 fetch (about
 * 5.3 GiB in i8) to the light use as well (decision 7's "light use without umT5" would be lost).
 * MUST: for a generation of input form `"t2v"` (Wan2.1) the component order stays as before the
 * encoder was added (2.1's fetch does not change).
 * The 32-latent decoder pair (ADR 0134) is fetched only when the preset's weights carry it
 * (`vaeTile32` — {@link presetCarriesVaeTile32}); a preset without it fetches what it did before.
 */
const wanComponentKeys = (
  generation: WanGenerationDescriptor,
  route: WanTextEncoderRoute,
  vaeTile32: boolean,
): readonly string[] => [
  TRANSFORMER,
  VAE_DECODER_FIRST,
  VAE_DECODER_NEXT,
  ...(vaeTile32 ? [VAE_DECODER_FIRST_32, VAE_DECODER_NEXT_32] : []),
  ...(generation.ditInputForm === "ti2v" ? WAN_VAE_ENCODER_KEYS : []),
  ...(route === "gpu" ? [TEXT_ENCODER] : []),
];

/**
 * Whether a preset's weights carry the 32-latent decoder pair (ADR 0134 decision 5 — both roles or
 * neither; one alone fails loudly). `weights` is the preset's weights map (`quants.<name>.weights`).
 */
const presetCarriesVaeTile32 = (
  weights: Readonly<Record<string, string>>,
  where: string,
): boolean => {
  const first = Object.hasOwn(weights, VAE_DECODER_FIRST_32);
  const next = Object.hasOwn(weights, VAE_DECODER_NEXT_32);
  if (first !== next) {
    const [present, missing] = first
      ? [VAE_DECODER_FIRST_32, VAE_DECODER_NEXT_32]
      : [VAE_DECODER_NEXT_32, VAE_DECODER_FIRST_32];
    throw new Error(
      `${where}: the weights carry '${present}' without '${missing}'` +
        " (the tile-32 VAE decoder is a pair — both roles or neither)",
    );
  }
  return first;
};

/**
 * Whether the preset `options` selects in `manifest` carries the 32-latent decoder pair. Read before
 * the components are listed, so the fetch and whole-bytes surfaces list the components admission
 * checks with the same function. A model or quant not in the manifest gives false (admission rejects
 * it with its own message).
 */
const selectionCarriesVaeTile32 = (
  manifest: Manifest,
  options: Pick<PipelineOptions, "model" | "quant">,
  owner: string,
): boolean => {
  const modelName = options.model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, modelName)) return false;
  const entry = manifest.models[modelName];
  const quantName = options.quant ?? entry.defaultQuant;
  if (!Object.hasOwn(entry.quants, quantName)) return false;
  return presetCarriesVaeTile32(entry.quants[quantName].weights, `${owner}: quant '${quantName}'`);
};

/** The values that differ per family (a class passes its own to {@link loadWanFromPretrained} etc.). */
export type WanFamilySpec = {
  /**
   * The message prefix (the class name). Also passed to the shared stages (`text-stage.ts` /
   * `dit-loop.ts` / `tile-decode.ts` / `graph-io.ts`).
   */
  readonly owner: string;
  /** The manifest `pipeline` name and major (the family admission check — ADR 0038 §6). */
  readonly pipeline: { readonly name: string; readonly major: number };
  /** The generation descriptor (accepted set, defaults, fps, statistics, VAE match, DiT input form). */
  readonly generation: WanGenerationDescriptor;
  /**
   * The spelling of the table that the diagnostic points to when `ref` is omitted (the third argument
   * of `toManifestSource` — `src/hub/repo-ref.ts`). Omitted for a generation without a public
   * distribution repository (Wan2.1 — `SOURCES` has no key for it) (no pointing at a key that does
   * not exist).
   */
  readonly sourcesExample?: string;
};

/** Wan2.1 T2V 1.3B (`Pipeline` — pipeline `wan/1`, {@link WAN21_GENERATION}). */
export const WAN21_FAMILY: WanFamilySpec = {
  owner: "wan.Pipeline",
  pipeline: { name: WAN_PIPELINE_NAME, major: WAN_PIPELINE_MAJOR },
  generation: WAN21_GENERATION,
};

/** Wan2.2 TI2V 5B (`Ti2vPipeline` — pipeline `wan-ti2v/1`, {@link WAN22_TI2V_GENERATION}). */
export const WAN22_TI2V_FAMILY: WanFamilySpec = {
  owner: "wan.Ti2vPipeline",
  pipeline: { name: WAN_TI2V_PIPELINE_NAME, major: WAN_TI2V_PIPELINE_MAJOR },
  generation: WAN22_TI2V_GENERATION,
  sourcesExample: 'SOURCES["wan2.2"] (@karume/models/wan)',
};

/**
 * Checks a generation request and turns it into a plan (the entry of `generate` — a pure check before
 * touching the GPU; the body of the checks and its MUSTs are in {@link planWanRequest}). The entry of
 * the `"precomputed"` path: the prompt is looked up in the text embedding asset's set
 * ({@link precomputedPromptGate}). The accepted set and defaults of size and frame count come from
 * `generation` ({@link WanFamilySpec.generation}); the prefix of rejection messages is `owner`.
 *
 * NOTE: `export` is for tests that pin the checks without a GPU (not on the `mod.ts` / subpath
 * surface — ADR 0008).
 */
export const planWanGeneration = (
  request: Ti2vGenerateRequest,
  embeds: WanTextEmbeds,
  layout: PlanLayout,
  config: WanPipelineConfig,
  generation: WanGenerationDescriptor,
  owner: string,
  tile32?: WanVaeTile32Plan,
): WanGenerationPlan<WanTextEmbedding> =>
  planWanRequest(request, precomputedPromptGate(embeds, owner), layout, config, generation, tile32);

/**
 * The entry of the `"gpu"` path (the same check order and the same knob checks as
 * {@link planWanGeneration}). The prompt becomes an id sequence through umT5's prompt layer (the
 * `prompt_clean` mirror → tokenizer — {@link gpuPromptGate}). A rejection is a `ModelInputError` (a
 * preprocessing rejection is the derived `PromptCleanError`), and the message says how to fix it.
 * When `negativePrompt` is omitted, the default negative (`WAN_DEFAULT_NEGATIVE_PROMPT` in
 * `text-stage.ts`) is encoded through the same check.
 *
 * MUST: encoding is finished here — no failing on out-of-vocabulary input after the umT5 weights
 * (about 5.3 GiB in i8) are uploaded.
 *
 * NOTE: `export` is for tests that pin the checks without a GPU (not on the `mod.ts` / subpath
 * surface — ADR 0008).
 */
export const planWanGpuGeneration = (
  request: Ti2vGenerateRequest,
  encoder: WanPromptEncoder,
  layout: PlanLayout,
  config: WanPipelineConfig,
  generation: WanGenerationDescriptor,
  tile32?: WanVaeTile32Plan,
): WanGenerationPlan<Int32Array<ArrayBuffer>> =>
  planWanRequest(request, gpuPromptGate(encoder), layout, config, generation, tile32);

/**
 * The keys this family accepts from the manifest's `session` and explicit overrides (the acceptance
 * table — `session/options.ts`).
 *
 * The three execution-form fields of linear / fused attention are accepted: the distribution's
 * practical preset `f16+dit8-a8-attn8-s16` declares these three (ADR 0120 decisions 1 / 7 — the same
 * declaration as anima's preset of the same name). The four fields of the parallel GEMV and fusion are
 * not accepted — for Wan's DiT graph neither the presets where they would apply nor their numerics
 * have been checked. A declaration of an unaccepted key fails loudly before the weights are fetched
 * (not silently run with the default).
 *
 * Preset selection is only the existing `quant`; no explicit override for the caller is added to the
 * public surface (ADR 0120 decision 7 — the third argument of `resolveSessionOptions` stays empty. The
 * acceptance table is the same single one, so an override can be added later if requested).
 *
 * NOTE: `export` is for the test that pins coverage of every family's acceptance table (not on the
 * `mod.ts` / subpath surface — ADR 0008).
 */
export const WAN_SESSION_POLICY: FamilySessionPolicy = {
  linearCompute: true,
  convCompute: true,
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

/** The material the family admission ({@link admitWan}) settles. */
type WanAdmission = {
  readonly config: WanPipelineConfig;
  readonly quantName: string;
  readonly quant: Quant;
  /**
   * The effective settings of the quant declaration after the acceptance table
   * ({@link WAN_SESSION_POLICY} — passed to the DiT Session).
   */
  readonly sessionOptions: SessionOptions;
  readonly gpuFeatures: GpuFeaturesSpec | undefined;
  readonly layout: WanVaeChunkLayout;
  /** The 32-latent decoder pair's contract (undefined when the preset's weights carry no `_32` roles). */
  readonly vaeTile32: WanVaeTile32Contract | undefined;
  readonly ropeBase: WanRopeBase;
  readonly dit: DitContract;
  /** The path (`"gpu"` comes with the umT5 graph contract). */
  readonly textEncoder: WanTextAdmission;
  /**
   * The contract of the three I2V VAE encoder graphs (only for a generation whose DiT input form is
   * `"ti2v"` — otherwise undefined).
   */
  readonly vaeEncoder: WanVaeEncoderContract | undefined;
};

/** Internal state of a Wan class (built by {@link loadWanFromPretrained} / {@link loadWanFromAssets}). */
export type WanState = {
  readonly gpu: GpuContext;
  readonly ownsGpu: boolean;
  readonly config: WanPipelineConfig;
  /** The preset's name (named by the rejection of `vaeTile` 32 without the 32-latent pair). */
  readonly quantName: string;
  readonly sessionOptions: SessionOptions;
  readonly transformer: ModelComponent;
  readonly vaeFirst: ModelComponent;
  readonly vaeNext: ModelComponent;
  /** The I2V VAE encoder (only for a generation whose DiT input form is `"ti2v"`; else undefined). */
  readonly vaeEncoder: WanVaeEncoder | undefined;
  readonly layout: WanVaeChunkLayout;
  /** The 32-latent decoder pair (ADR 0134 — undefined when the preset's weights carry no `_32` roles). */
  readonly vaeTile32: WanVaeTile32 | undefined;
  readonly ropeBase: WanRopeBase;
  readonly dit: DitContract;
  /** The embedding asset (the accepted set of `"precomputed"`, `prompts` on both paths). */
  readonly textEmbeds: WanTextEmbeds;
  readonly text: WanTextStage;
  /**
   * Experimental ({@link PipelineOptions.fusionRules}): the DiT, VAE decoder and VAE encoder
   * Sessions get them; umT5's does not. Absent = no rule anywhere.
   */
  readonly fusionRules?: PipelineFusionRules;
  readonly onRunDiagnostics?: (component: RunComponent, diagnostics: SessionDiagnostics) => void;
};

/**
 * Fetches from the distribution and builds (the body of the class's `fromPretrained` — the procedure
 * and MUSTs are in the doc of `Pipeline.fromPretrained`).
 */
export const loadWanFromPretrained = async (
  spec: WanFamilySpec,
  ref: string | HubRepoRef | DistributionSource,
  options: FromPretrainedOptions,
): Promise<WanState> => {
  // The path is decided before fetching (the components to fetch change) — a misspelling fails
  // without fetching a byte.
  const route = textEncoderRouteOf(options, spec.owner);
  const source = toManifestSource(ref, `${spec.owner}.fromPretrained`, spec.sourcesExample);
  // The signal goes to **both** the fetch layer and construction (no window where cancellation stops
  // working the moment the download ends — copied by `hubLoadOptions`, the same form as anima /
  // irodori).
  const hubOptions = hubLoadOptions(options);
  const loaded = await loadManifest(source, hubOptions);
  assertTextEncoderDeclared(loaded.manifest, options, route, spec.owner);
  const choice = {
    ...(options.model === undefined ? {} : { model: options.model }),
    ...(options.quant === undefined ? {} : { quant: options.quant }),
  };
  const selection = resolveSelection(loaded.manifest, choice);
  const vaeTile32 = selectionCarriesVaeTile32(loaded.manifest, choice, spec.owner);
  const buildOptions: PipelineOptions = {
    ...(options.gpu === undefined ? {} : { gpu: options.gpu }),
    ...choice,
    textEncoder: route,
    ...(options.onRunDiagnostics === undefined
      ? {}
      : { onRunDiagnostics: options.onRunDiagnostics }),
    ...(options.signal === undefined ? {} : { signal: options.signal }),
    ...(options.fusionRules === undefined ? {} : { fusionRules: options.fusionRules }),
  };
  // The family checks run in the admission callback (before the weight parts are fetched —
  // `src/hub/components.ts`).
  const { admitted, assets, open } = await loadContainerComponents(
    `${spec.owner}.fromPretrained`,
    loaded,
    selection,
    wanComponentKeys(spec.generation, route, vaeTile32),
    async (open) => {
      const admitted = await admitWan(spec, loaded.manifest, open, buildOptions);
      // The `requiredLimits` the distribution declares are checked here, **before the weight parts are
      // fetched** (ADR 0089 decision 5 — a shared GPU's limits, or the adapter's measured values on the
      // path that acquires its own).
      await assertRequiredLimitsBeforeDownload(
        admitted.quant.requiredLimits,
        buildOptions.gpu,
        `${spec.owner}: quant '${admitted.quantName}'`,
      );
      return admitted;
    },
    {
      ...hubOptions,
      ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
      ...(options.components === undefined ? {} : { components: options.components }),
    },
  );
  return await buildWan(spec, admitted, assets, open, buildOptions);
};

/**
 * Builds from an already-fetched manifest + assets ({@link Assets}). Opens the containers and runs
 * the same family admission ({@link admitWan}) and assembly ({@link buildWan}) as the fetch surface —
 * the two surfaces differ only in how components are supplied.
 */
export const loadWanFromAssets = async (
  spec: WanFamilySpec,
  input: Assets,
  options: PipelineOptions,
): Promise<WanState> => {
  const route = textEncoderRouteOf(options, spec.owner);
  // The entry check (if already cancelled, no container is opened — opening includes validating
  // GB-scale part sequences).
  await settleAbort(options.signal);
  assertTextEncoderDeclared(input.manifest, options, route, spec.owner);
  const buffer = (key: string): ArrayBuffer =>
    readAssetBuffer(spec.owner, "weights / assets", input.assets, key);
  const open = await assetComponentOpener(
    spec.owner,
    input.assets,
    buffer,
    wanComponentKeys(
      spec.generation,
      route,
      selectionCarriesVaeTile32(input.manifest, options, spec.owner),
    ),
  );
  const admitted = await admitWan(spec, input.manifest, open, options);
  return await buildWan(spec, admitted, input.assets, open, options);
};

/**
 * Checks whether this manifest and the components' graph declarations can run as the generation of
 * `spec` (Wan2.1 for {@link WAN21_FAMILY}) (the family admission — on the fetch surface it is called
 * **before a single byte of the weight parts is fetched**).
 *
 * MUST: the family checks are gathered in this one function (pipeline name / major,
 * `pipelineConfig`, the quant's `session`, the shared GPU's capabilities, the timing device, graph
 * declarations × host contracts). Scattered into later stages, on the fetch surface they would fail
 * only **after** GB-scale weights were downloaded (ADR 0070 decision 5).
 * MUST: manifest contract violations fail **before the GPU is acquired** (the same order as the other
 * families).
 *
 * NOTE: the RoPE base table (an asset of the `transformer` container) is read here — an asset that
 * shares no part with weight blocks can be read during admission too (`src/hub/components.ts`). The
 * embedding asset and the tokenizer asset (the manifest's `assets`) have not arrived yet, so their
 * contents are matched in {@link buildWan} (here only their declaration is checked).
 */
const admitWan = async (
  spec: WanFamilySpec,
  manifest: Manifest,
  open: ComponentOpener,
  options: PipelineOptions,
): Promise<WanAdmission> => {
  // A step boundary (after the containers are opened, before the weight parts are fetched).
  await settleAbort(options.signal);
  const route = textEncoderRouteOf(options, spec.owner);
  const modelName = options.model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, modelName)) {
    throw new Error(
      `${spec.owner}: model '${modelName}' is not in the manifest` +
        ` (available: ${manifest.available.models.join(" / ")})`,
    );
  }
  const entry: ModelEntry = manifest.models[modelName];
  const { name, major } = entry.pipeline;
  if (name !== spec.pipeline.name) {
    throw new Error(
      `${spec.owner}: the manifest's pipeline is '${name}/${major}'` +
        ` ('${spec.pipeline.name}/${spec.pipeline.major}' is required)`,
    );
  }
  if (major !== spec.pipeline.major) {
    // The only check that stops the silent degradation of "old implementation × new repository"
    // (ADR 0038 §6).
    throw new Error(
      `${spec.owner}: pipeline '${name}/${major}' has an unsupported major` +
        ` (this implementation reads ${spec.pipeline.name}/${spec.pipeline.major})`,
    );
  }
  const config = parseWanPipelineConfig(entry.pipelineConfig);

  const quantName = options.quant ?? entry.defaultQuant;
  if (!Object.hasOwn(entry.quants, quantName)) {
    throw new Error(
      `${spec.owner}: quant '${quantName}' is not in the manifest` +
        ` (available: ${entry.available.quants.join(" / ")})`,
    );
  }
  const quant = entry.quants[quantName];
  // The `"gpu"` path reads the tokenizer asset — without a declaration, fail before fetching umT5
  // (about 5.3 GiB) (its contents are checked by {@link buildWan} once they arrive).
  assertTokenizerDeclared(entry, route, spec.owner);
  // Unsupported declarations fail before the weight parts are fetched (one function shared by every
  // family).
  const sessionOptions = resolveSessionOptions(
    WAN_SESSION_POLICY,
    quant.session,
    {},
    `${spec.owner}: quant '${quantName}'`,
  );
  const gpuFeatures = sessionGpuFeatures(quant.gpuFeatures, sessionOptions);

  // MUST: a shared GPU's missing capabilities (feature / device limit) fail here — the only family
  // check that knows before the weights are downloaded (the later checks call the same functions).
  if (options.gpu !== undefined) {
    assertGpuFeaturesGranted(gpuFeatures, options.gpu, `${spec.owner}: quant '${quantName}'`);
    assertRequiredLimitsSatisfied(
      quant.requiredLimits,
      options.gpu.limits,
      `${spec.owner}: quant '${quantName}'`,
    );
  }

  const transformer = open(TRANSFORMER);
  const layout = wanVaeChunkLayout(open(VAE_DECODER_FIRST), open(VAE_DECODER_NEXT));
  // The latent channel count is read from the VAE declaration and matched against the descriptor (the
  // number of statistics, the unpatchify factor). Matching against the width of the DiT's `tokens`
  // happens in ditContract, which checks the patch built with the same channel count.
  assertWanVaeMatchesGeneration(layout, spec.generation, spec.owner);
  assertWanVaeTilesCover(layout, spec.generation, spec.owner);
  // The 32-latent pair (ADR 0134) is checked against its lenders here, before the weight parts are
  // fetched. The same function decided whether its components were listed.
  const vaeTile32 = presetCarriesVaeTile32(quant.weights, `${spec.owner}: quant '${quantName}'`)
    ? wanVaeTile32Contract(
        { first: open(VAE_DECODER_FIRST_32), next: open(VAE_DECODER_NEXT_32) },
        { first: open(VAE_DECODER_FIRST), next: open(VAE_DECODER_NEXT) },
        layout,
        spec.generation,
        spec.owner,
      )
    : undefined;
  const ropeBase = parseWanRopeBase(await readWholeAsset(transformer.asset(ROPE_BASE)));
  const dit = ditContract(
    transformer,
    ropeBase,
    wanDitPatch(layout.latentChannels),
    spec.generation.ditInputForm,
    spec.owner,
  );
  const textEncoder = admitWanText(route, open, dit, spec.owner);
  // The I2V encoder is checked against the decoder's latent and spatial compression (including
  // coverage of the accepted sizes — `wanVaeEncoderContract`).
  // An unknown DiT input form has already failed in ditContract.
  const vaeEncoder =
    spec.generation.ditInputForm === "ti2v"
      ? wanVaeEncoderContract(
          {
            pre: open(VAE_ENCODER_PRE),
            attn: open(VAE_ENCODER_ATTN),
            post: open(VAE_ENCODER_POST),
          },
          layout,
          spec.generation,
          spec.owner,
        )
      : undefined;
  return {
    config,
    quantName,
    quant,
    sessionOptions,
    gpuFeatures,
    layout,
    vaeTile32,
    ropeBase,
    dit,
    textEncoder,
    vaeEncoder,
  };
};

/**
 * Builds from the material that passed admission + the assets (one function shared by
 * {@link loadWanFromAssets} and {@link loadWanFromPretrained}). Parses the embedding asset, matches it
 * against the DiT's context input, and acquires the GPU (not for a shared GPU).
 *
 * MUST: parsing and matching the assets come **before the GPU is acquired** (the true cause of a broken
 * asset is not masked by another exception on a machine without a GPU — the same order as the other
 * families). No Session is created (the VRAM MUST — the doc at the top of `./pipeline.ts`).
 */
const buildWan = async (
  spec: WanFamilySpec,
  admitted: WanAdmission,
  assets: Assets["assets"],
  open: ComponentOpener,
  options: PipelineOptions,
): Promise<WanState> => {
  const { config, quantName, sessionOptions, gpuFeatures, layout, ropeBase, dit } = admitted;
  // A step boundary (after the assets arrive, before parsing).
  await settleAbort(options.signal);
  const { textEmbeds, text } = await loadWanTextStage(
    admitted.textEncoder,
    assets,
    open,
    dit,
    options.signal,
    spec.owner,
  );

  await settleAbort(options.signal);
  // The online attention preset (`attentionScoreStorage: "none"`) asks for subgroups when the adapter
  // has them (ADR 0132 decision 5); a shared GPU runs the kernel's portable form or, on a 32 KiB
  // device without subgroups, the three dispatches.
  const gpu = options.gpu ?? (await acquireGpu(toAcquireGpuOptions(gpuFeatures, sessionOptions)));
  const ownsGpu = options.gpu === undefined;
  try {
    // MUST: the cancellation check **after** acquiring the GPU stays inside the try — outside it, the
    // internally acquired device would be left with nobody to release it. Resuming from the
    // `acquireGpu` await is a microtask, so yield to a macrotask here too (a cancellation task queued
    // during the wait has not run yet — the same as anima).
    await settleAbort(options.signal);
    // Declared features can be requested only when the device is created (ADR 0028) — for a device
    // acquired here this is the only check (a shared GPU is checked by {@link admitWan} with the same
    // function).
    // The features and limits of the GPU in hand — one helper for every family (the reasons are in
    // its doc).
    assertAcquiredGpuCapabilities(
      gpuFeatures,
      admitted.quant.requiredLimits,
      gpu,
      `${spec.owner}: quant '${quantName}'`,
    );
    return {
      gpu,
      ownsGpu,
      config,
      quantName,
      sessionOptions,
      // The supplier is **the same one** admission saw (`open` only looks up the opened components).
      transformer: open(TRANSFORMER),
      vaeFirst: open(VAE_DECODER_FIRST),
      vaeNext: open(VAE_DECODER_NEXT),
      vaeEncoder:
        admitted.vaeEncoder === undefined
          ? undefined
          : {
              pre: open(VAE_ENCODER_PRE),
              attn: open(VAE_ENCODER_ATTN),
              post: open(VAE_ENCODER_POST),
              contract: admitted.vaeEncoder,
            },
      layout,
      vaeTile32:
        admitted.vaeTile32 === undefined
          ? undefined
          : {
              ...admitted.vaeTile32,
              first: open(VAE_DECODER_FIRST_32),
              next: open(VAE_DECODER_NEXT_32),
            },
      ropeBase,
      dit,
      textEmbeds,
      text,
      // Experimental (a measurement opt-in): the DiT and VAE stages' Sessions get the rules.
      ...(options.fusionRules === undefined ? {} : { fusionRules: options.fusionRules }),
      ...(options.onRunDiagnostics === undefined
        ? {}
        : { onRunDiagnostics: options.onRunDiagnostics }),
    };
  } catch (error) {
    // A GPU acquired internally can no longer be released by anyone if construction fails, so it is
    // returned here.
    if (ownsGpu) gpu.destroy();
    throw error;
  }
};

/**
 * Takes ownership of the caller's initial noise at the entry of the class's `generate`, before the
 * request waits its turn on the serialization chain (the one place both classes call).
 *
 * MUST: copy here, not inside the chain. A generation waits for the ones queued before it, and the
 * stages after the entry check await the event callback and `settleAbort`, so a copy taken any later
 * would let values the caller rewrote in between (including NaN that never passed the check) reach
 * the DiT stage — the call would not generate from the values it was given. The check itself
 * (`planWanRequest`) runs on this copy, so what it verified is what runs.
 *
 * Only a `Float32Array` is copied: anything else is passed through as is so that the check rejects
 * it by name (`Float32Array.from` would silently turn a `number[]` into a valid-looking array).
 */
export const ownWanRequestLatents = <Request extends GenerateRequest>(request: Request): Request =>
  request.latents instanceof Float32Array
    ? { ...request, latents: Float32Array.from(request.latents) }
    : request;

/**
 * Generates one video from a prompt (and the I2V condition image) (the body the class's `generate` puts
 * on the serialization chain — the stage order, Session lifetimes and cancellation are in the doc at
 * the top of `./pipeline.ts`).
 *
 * MUST (caller's precondition): `request` is the return value of {@link ownWanRequestLatents} — this
 * function uses `request.latents` without copying (the DiT stage's precondition in `dit-loop.ts`).
 */
export const generateWanVideo = async (
  spec: WanFamilySpec,
  state: WanState,
  request: Ti2vGenerateRequest,
): Promise<GeneratedVideo> => {
  const { dit } = state;
  const { onEvent, signal } = request;
  const emit =
    onEvent === undefined
      ? () => Promise.resolve()
      : async (event: GenerateEvent) => {
          await onEvent(event);
        };
  // The entry check is per path (both are pure checks before touching the GPU). The prompt contexts
  // are built per path, and the encoder, DiT and VAE stages go through the same single body. The text
  // stage runs after the encoder stage, so only how to build it is decided here.
  let knobs: WanGenerationKnobs;
  let encodeContexts: () => Promise<WanContexts>;
  const tile32: WanVaeTile32Plan = {
    ...(state.vaeTile32 === undefined ? {} : { layout: state.vaeTile32.layout }),
    quant: state.quantName,
  };
  if (state.text.kind === "gpu") {
    const text = state.text;
    const plan = planWanGpuGeneration(
      request,
      text.encoder,
      state.layout,
      state.config,
      spec.generation,
      tile32,
    );
    knobs = plan;
    encodeContexts = async () => {
      // A stage boundary (entry / encoder → text). A cancellation that arrived while waiting on the
      // serialization chain also takes effect here.
      await settleAbort(signal);
      return await encodeWanPrompts(state, text, plan, emit, signal, spec.owner);
    };
  } else {
    const plan = planWanGeneration(
      request,
      state.textEmbeds,
      state.layout,
      state.config,
      spec.generation,
      spec.owner,
      tile32,
    );
    knobs = plan;
    encodeContexts = () => Promise.resolve(precomputedContexts(plan, dit));
  }
  // NOTE: `request.latents` is already the copy {@link ownWanRequestLatents} took at the entry of the
  // class's `generate` (outside the chain), so the check above verified the values the DiT stage
  // uses, and that stage uses them without copying again.
  // The I2V encoder stage comes first (before the text stage — ADR 0121 decision 11). A request
  // without an image has no such stage.
  let condition: Float32Array<ArrayBuffer> | undefined;
  if (knobs.conditionImage !== undefined) {
    const { vaeEncoder } = state;
    if (vaeEncoder === undefined) {
      // The check (plan.ts) lets an image through only for a generation of input form "ti2v", and that
      // generation builds the encoder at construction, so reaching here means broken wiring.
      throw new Error(
        `${spec.owner}: received a condition image but the VAE encoder component is not built`,
      );
    }
    // A stage boundary (entry → encoder).
    await settleAbort(signal);
    condition = await encodeWanImageStage(
      { ...state, vaeEncoder },
      knobs.conditionImage,
      spec.generation,
      emit,
      signal,
      spec.owner,
    );
  }
  const contexts = await encodeContexts();
  // A stage boundary (text → DiT).
  await settleAbort(signal);
  const latents = await runWanDenoise(state, knobs, contexts, condition, emit, signal, spec.owner);
  // A stage boundary (DiT → VAE).
  await settleAbort(signal);
  const data = await decodeWanVaeStage(
    state,
    knobs,
    latents,
    spec.generation,
    emit,
    signal,
    spec.owner,
  );
  return {
    frames: knobs.frames,
    width: knobs.width,
    height: knobs.height,
    fps: spec.generation.fps,
    data,
  };
};
