/**
 * `Siglip2Pipeline` — the single surface for image → embedding vector (SigLIP2's vision tower).
 *
 * Pipeline (every step in Karume, no torch):
 *
 * 1. Host: resize the RGB8 pixel row to the size in `pipelineConfig` (antialiased bilinear)
 * 2. Host: apply rescale + normalize folded together into `[1, 3, H, W]` f32
 * 3. Run `vision` once → `pooler_output` (`[1, hidden]` through the MAP head)
 *
 * The implementation and the reference implementation of the two preprocessing steps live in
 * `src/image/preprocess.ts` (`resizeRgb8` / `normalizeToNchw`); this only wires them with the
 * `pipelineConfig` constants. **Decoding is not karume's job** — the entry is "RGB8 pixel row +
 * width + height", and there is no PNG / JPEG decoder (in a browser `createImageBitmap` + canvas,
 * on Node / Deno any decoder).
 *
 * ## MUST: one Session is created at construction and kept until `dispose` (unlike other families)
 *
 * Anima / Irodori create and tear down per step because several huge graphs cannot be resident at
 * once (ADR 0016). Here there is one graph (base 372MB / so400m 1.71GB) with nothing to make room
 * for; the opposite, "create and tear down per image", **would re-upload the whole weights for
 * every image**. Embedding normally runs many images, so residency is the default shape.
 *
 * MUST: it still goes on the serialization chain — concurrent `embed` calls would share one Session
 * between two `run`s, and `dispose` would destroy the GPU without waiting for in-flight work (a
 * flush-before-destroy violation).
 *
 * ## The output is `pooler_output` **itself** (no L2 normalization, no cosine)
 *
 * SigLIP2's pooler_output is a raw vector that upstream does not normalize either (measured L2 norm
 * 12.7–13.1), and
 *
 * - whether to normalize **depends on the downstream use** (normalize for cosine search, keep raw
 *   as linear-probe input). Folding it in here would remove the way to get the raw vector.
 * - SigLIP's actual similarity is the value through the text tower's `logit_scale` / `logit_bias`,
 *   and this distribution has only the vision tower. **The "right similarity" cannot come from
 *   here**, so a plausible-looking `similarity()` would promise a wrong meaning.
 * - The caller's implementation is the two lines of `v.map(x => x / norm)`, not worth an
 *   abstraction.
 *
 * {@link Siglip2Pipeline.embed} therefore returns `pooler_output` bare (ADR 0008's "thin surface" —
 * the user story ends at "embed an image").
 *
 * ## MUST: numerical correctness is not guaranteed here
 *
 * Correctness is guaranteed by the per-graph golden E2E (`packages/e2e/tests/e2e_siglip2_test.ts` —
 * real weights of 2 series × 4 cases) and the preprocessing parity check
 * (`packages/models/tests/image_preprocess_test.ts`).
 */

import {
  acquireGpu,
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
  parseSiglip2PipelineConfig,
  SIGLIP2_PIPELINE_MAJOR,
  SIGLIP2_PIPELINE_NAME,
  type Siglip2PipelineConfig,
} from "./config.ts";
import { normalizeToNchw, resizeRgb8, type Rgb8Image } from "../image/preprocess.ts";
import { createOperationChain } from "../concurrency/serial.ts";
import {
  assertGpuFeaturesGranted,
  assertRequiredLimitsBeforeDownload,
  assertRequiredLimitsSatisfied,
  sessionGpuFeatures,
  toAcquireGpuOptions,
} from "../session/gpu-features.ts";
import { disposeSteps } from "../session/dispose-steps.ts";
import { type FamilySessionPolicy, resolveSessionOptions } from "../session/options.ts";
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
import { readAssetBuffer } from "../hub/asset-readers.ts";
import { assertGraphInputDim, assertGraphInputDtype } from "../hub/graph-gates.ts";

/** Component name in the manifest's weights table (the conventional name of ADR 0041 §3). */
const VISION = "vision";

/** This family's weights components (also the role names component replacement accepts). */
const COMPONENT_KEYS = [VISION] as const;

/** Name of the graph input (`INPUT_NAME` in `tools/export-recipes/siglip2/export.py`). */
const PIXEL_VALUES = "pixel_values";

/** Input channel count (RGB). The graph is baked with static 1 / 3 for it and the batch. */
const CHANNELS = 3;

/**
 * Construction options (shared by {@link Siglip2Pipeline.fromAssets} /
 * {@link Siglip2Pipeline.fromPretrained}).
 */
export type Siglip2PipelineOptions = {
  /**
   * Shares an existing GPU. **The caller keeps ownership**, so {@link Siglip2Pipeline.dispose} does
   * not destroy it. When omitted, the pipeline calls `acquireGpu` internally and destroys it in
   * `dispose()`.
   */
  readonly gpu?: GpuContext;
  /** Model (a key of the manifest's models). Defaults to `defaultModel`. */
  readonly model?: string;
  /** Execution configuration (a key of that model's quants). Defaults to `defaultQuant`. */
  readonly quant?: string;
  /**
   * Observation hook that receives the diagnostics of each run (one embedding = one `vision` run).
   * For per-op GPU time (`lastRunTiming`) pass `acquireGpu({ gpuTiming: true })` as `gpu` (ADR 0021
   * — no timing by default).
   *
   * NOTE: unlike other families, no component name is passed — there is one graph, so the name
   * would always be the same single value (no argument the receiver cannot branch on).
   *
   * Exceptions from the callback are not caught (fail loudly — the embedding fails with it).
   */
  readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics) => void;
};

/**
 * Fetch-layer options used only by {@link Siglip2Pipeline.fromPretrained} (passed through to the
 * hub).
 *
 * NOTE: the description of the fetch-layer knobs, including that `headers` / `fetch` / `caches` /
 * `onRetry` are **for HTTP sources only**, lives in one place, {@link FromPretrainedHubOptions}.
 */
export type Siglip2FromPretrainedOptions = Siglip2PipelineOptions &
  FromPretrainedHubOptions &
  FromPretrainedComponentOptions & {
    readonly signal?: AbortSignal;
  };

/** Input for building directly from fetched assets (pass the hub's `fetchAssets` result as is). */
export type Siglip2Assets = {
  readonly manifest: Manifest;
  readonly assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>;
};

/**
 * Turns the fetched bytes into an ArrayBuffer that can be passed to `openContainer` (the check
 * itself is {@link readAssetBuffer}).
 *
 * MUST: do not copy with `slice` — so400m is 1.71GB in one file, and the host RAM peak would
 * double.
 */
const assetBuffer = (
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  key: string,
): ArrayBuffer => readAssetBuffer("siglip2", "weights", assets, key);

/**
 * The component supplier of the full-asset surface (`fromAssets`) (the receiving implementation is
 * shared by every family — {@link assetComponentOpener}). A component key is either one single-form
 * `krm` (`vision`) or the part sequence of the split form (`vision[0]` / `vision[1]` / …).
 */
const assetOpener = (assets: Siglip2Assets["assets"]): Promise<ComponentOpener> =>
  assetComponentOpener("siglip2", assets, (key) => assetBuffer(assets, key), COMPONENT_KEYS);

/**
 * Checks that the **static** dimension of one graph-input axis matches the `pipelineConfig`
 * declaration.
 *
 * MUST: do not drop this check. Preprocessing resizes to the declared size, so even when the graph
 * was baked at another resolution **the host side passes all the way through** (the failure comes
 * from the Session's shape check, by which time the reader cannot tell "which number is right").
 * This is the only detector of mixed-up base and so400m assets.
 *
 * NOTE: the `export` is for tests that pin the rejection paths without a GPU (not exposed on
 * `mod.ts` / the subpath surface — ADR 0008).
 */
export const assertStaticDim = (
  model: GraphOwner,
  inputName: string,
  axis: number,
  expected: number,
  where: string,
): void => assertGraphInputDim("siglip2", model, inputName, axis, expected, where);

/**
 * Checks that one axis of the graph **output** has the declared **static** dimension.
 *
 * MUST: do not drop this check. `hiddenDim` is used neither by preprocessing nor by execution, so
 * if it is off the embedding still comes out without complaint (the reader would hand out 1152-dim
 * vectors believing they are 768-dim).
 *
 * NOTE: the `export` is for tests that pin the rejection paths without a GPU (not exposed on
 * `mod.ts` / the subpath surface — ADR 0008).
 */
export const assertOutputDim = (
  model: GraphOwner,
  axis: number,
  expected: number,
  where: string,
): void => {
  const name = model.graph.outputs[0];
  const value = model.graph.values[name];
  if (value === undefined) {
    throw new Error(`siglip2: グラフ出力 '${name}' の宣言が無い（${where}）`);
  }
  const dim = value.shape[axis];
  if (dim !== expected) {
    throw new Error(
      `siglip2: ${where} — グラフ出力 '${name}' の軸 ${axis} が ${String(dim)}、` +
        `pipelineConfig は ${expected}`,
    );
  }
};

/**
 * Checks that the graph **output** is f32 `[1, hiddenDim]` (rank 2, batch axis 1) (the width is
 * checked by {@link assertOutputDim}).
 *
 * MUST: do not drop this check. The later step only checks that the output has `hiddenDim` elements
 * **after** the run, so another graph baked with batch 2 (`[2, hiddenDim]`) or a non-f32 output
 * would pass construction, create the GPU and the Session, run once, and only then fail with a
 * message that does not point at the mixed-up asset.
 *
 * NOTE: the `export` is for tests that pin the rejection paths without a GPU (not exposed on
 * `mod.ts` / the subpath surface — ADR 0008).
 */
export const assertPooledOutput = (model: GraphOwner, where: string): void => {
  const name = model.graph.outputs[0];
  const value = model.graph.values[name];
  if (value === undefined) {
    throw new Error(`siglip2: グラフ出力 '${name}' の宣言が無い（${where}）`);
  }
  if (value.dtype !== "f32" || value.shape.length !== 2 || value.shape[0] !== 1) {
    throw new Error(
      `siglip2: ${where} — グラフ出力 '${name}' が ${value.dtype} ` +
        `[${value.shape.map(String).join(", ")}]、期待は f32 の [1, hiddenDim]`,
    );
  }
};

const asF32 = (tensor: Tensor, where: string): Float32Array<ArrayBuffer> => {
  if (tensor.dtype !== "f32") throw new Error(`${where}: f32 でない（${tensor.dtype}）`);
  return tensor.data;
};

/**
 * RGB8 pixel row → the `pixel_values` passed to the graph (`[3, H, W]` flattened).
 *
 * The preprocessing itself is owned by `src/image/preprocess.ts`; all this does is wire it with the
 * `pipelineConfig` constants. **`resizeRgb8` takes `(width, height)` in that order**, so swapping
 * the two declared fields would silently transpose a non-square distribution.
 *
 * MUST: the interpolation filter also comes from `config.interpolation` (do not rely on the default
 * argument). With the declaration and the effective value as one, and the accepted set a literal
 * type, **the type checker guarantees they are identical** — the path by which a new family forgets
 * to pass it disappears structurally (the implementation side of `preprocess.ts`'s MUST "it is part
 * of the model card, so the caller states it explicitly").
 *
 * NOTE: the `export` is for tests that call the wiring directly (reaching here through `fromAssets`
 * needs a real IR container). Not exposed on `mod.ts` / the subpath surface (ADR 0008).
 */
export const preprocessPixelValues = (
  config: Siglip2PipelineConfig,
  image: Rgb8Image,
): Float32Array<ArrayBuffer> =>
  normalizeToNchw(
    resizeRgb8(image, config.imageWidth, config.imageHeight, config.interpolation),
    config.imageMean,
    config.imageStd,
  );

/** Internal state of {@link Siglip2Pipeline} (not on the public surface). */
type Siglip2State = {
  readonly gpu: GpuContext;
  readonly ownsGpu: boolean;
  readonly config: Siglip2PipelineConfig;
  readonly vision: ModelComponent;
  /** The one Session created at construction and kept until `dispose` (the module doc's MUST). */
  readonly session: Session;
  readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics) => void;
};

/**
 * The execution knobs siglip2 accepts (the manifest's quant declaration). The merge rule
 * (explicit > declared > runtime default), value ranges, combinations and the classification of
 * the thrown types are owned by `resolveSessionOptions`, shared by every family; this only decides
 * the accepted set. There is no explicit override (the construction options have no execution-knob
 * field).
 *
 * No key is accepted because the distribution has only the f32 default path and no quant that
 * declares execution knobs (when a declaration appears, check it against the reference values
 * before accepting it — do not silently pass it to the runtime).
 *
 * NOTE: the `export` is for the equivalence test that runs every quant of the mirror through the
 * same table (not exposed on `mod.ts` / the subpath surface — ADR 0008).
 */
export const SIGLIP2_SESSION_POLICY: FamilySessionPolicy = {
  linearCompute: false,
  attentionCompute: false,
  attentionScoreStorage: false,
  linearGemvReduce: false,
  stateAttentionReduce: false,
  fuseRmsNormAdd: false,
  fuseLinearStaticQuantize: false,
  packedStaticQuantize: false,
};

/** What the family admission ({@link admitSiglip2}) settles. */
type Siglip2Admission = {
  readonly config: Siglip2PipelineConfig;
  readonly quantName: string;
  readonly quant: Quant;
  /**
   * The effective settings after passing the quant declaration through the acceptance table
   * ({@link SIGLIP2_SESSION_POLICY}).
   */
  readonly sessionOptions: SessionOptions;
  /**
   * The quant declaration plus the features the effective settings need (both the request and the
   * check read this).
   */
  readonly gpuFeatures: GpuFeaturesSpec | undefined;
};

/**
 * Checks whether this manifest and this graph can run as siglip2 (the {@link FamilyAdmission} hook
 * of `hub/components.ts` — on the fetch surface it is called **before a single byte of the weight
 * parts is fetched**).
 *
 * MUST: gather the family's checks in this one function. Scattered into later steps, on the fetch
 * surface they would fail only **after** GB-class weights are downloaded (the implementation would
 * be narrower than the wording of ADR 0070 decision 5).
 * MUST: manifest contract violations and **the match against the graph** fail **before the GPU is
 * acquired**. In the other order, an environment without a GPU turns them into a different
 * exception and the reader cannot tell "what was wrong".
 */
const admitSiglip2 = (
  manifest: Manifest,
  open: ComponentOpener,
  options: Siglip2PipelineOptions,
): Siglip2Admission => {
  const modelName = options.model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, modelName)) {
    throw new Error(
      `Siglip2Pipeline: model '${modelName}' は manifest に無い` +
        `（利用可能: ${manifest.available.models.join(" / ")}）`,
    );
  }
  const entry: ModelEntry = manifest.models[modelName];
  const { name, major } = entry.pipeline;
  if (name !== SIGLIP2_PIPELINE_NAME) {
    throw new Error(
      `Siglip2Pipeline: manifest の pipeline が '${name}/${major}'` +
        `（'${SIGLIP2_PIPELINE_NAME}/${SIGLIP2_PIPELINE_MAJOR}' が必要）`,
    );
  }
  if (major !== SIGLIP2_PIPELINE_MAJOR) {
    // The only check that stops silent degradation from "old implementation × new repository"
    // (ADR 0038 §6).
    throw new Error(
      `Siglip2Pipeline: pipeline '${name}/${major}' の major に未対応` +
        `（この実装が読めるのは ${SIGLIP2_PIPELINE_NAME}/${SIGLIP2_PIPELINE_MAJOR}）`,
    );
  }
  const config = parseSiglip2PipelineConfig(entry.pipelineConfig);

  const quantName = options.quant ?? entry.defaultQuant;
  if (!Object.hasOwn(entry.quants, quantName)) {
    throw new Error(
      `Siglip2Pipeline: quant '${quantName}' は manifest に無い` +
        `（利用可能: ${entry.available.quants.join(" / ")}）`,
    );
  }
  const quant = entry.quants[quantName];
  // Unsupported declarations fail before components are opened (= before weight parts are fetched;
  // one path shared by every family).
  const sessionOptions = resolveSessionOptions(
    SIGLIP2_SESSION_POLICY,
    quant.session,
    {},
    `Siglip2Pipeline: quant '${quantName}'`,
  );
  const gpuFeatures = sessionGpuFeatures(quant.gpuFeatures, sessionOptions);

  const vision = open(VISION);
  // Match the graph declaration against pipelineConfig. Checking that there is exactly one input
  // and one output matters because, if another graph including the text tower or a multi-output
  // version made for goldens got mixed in, the later steps that pick by position would silently
  // read other values.
  if (vision.graph.inputs.length !== 1 || vision.graph.outputs.length !== 1) {
    throw new Error(
      `Siglip2Pipeline: vision グラフの入出力が ${vision.graph.inputs.length} /` +
        ` ${vision.graph.outputs.length} 本（どちらも 1 本の vision tower が必要）`,
    );
  }
  assertGraphInputDtype("siglip2", vision, PIXEL_VALUES, "f32", "前処理の pixel_values");
  assertStaticDim(vision, PIXEL_VALUES, 0, 1, "batch は静的 1");
  assertStaticDim(vision, PIXEL_VALUES, 1, CHANNELS, "RGB の 3 チャネル");
  assertStaticDim(vision, PIXEL_VALUES, 2, config.imageHeight, "imageHeight");
  assertStaticDim(vision, PIXEL_VALUES, 3, config.imageWidth, "imageWidth");
  assertPooledOutput(vision, "pooler_output");
  assertOutputDim(vision, 1, config.hiddenDim, "hiddenDim");

  // MUST: a shared GPU's missing capability (feature / device limit) fails here — unlike acquiring
  // our own, there is no reason to wait for `acquireGpu`, and this is the only family check that
  // knows before the weights are downloaded (the mapping between request and check is the one
  // function in `session/gpu-features.ts`, and the later check calls the same function).
  if (options.gpu !== undefined) {
    assertGpuFeaturesGranted(gpuFeatures, options.gpu, `Siglip2Pipeline: quant '${quantName}'`);
    assertRequiredLimitsSatisfied(
      quant.requiredLimits,
      options.gpu.limits,
      `Siglip2Pipeline: quant '${quantName}'`,
    );
  }

  return { config, quantName, quant, sessionOptions, gpuFeatures };
};

/**
 * Builds the execution state from the materials that passed admission (GPU acquisition and Session
 * construction only).
 *
 * MUST: the only checks allowed after GPU acquisition are GPU capabilities (the shader-f16 feature
 * — ADR 0028 / device limits — ADR 0089 decision 5). After acquisition this function checks only
 * features; limits are checked by {@link admitSiglip2} for a shared GPU, and for a device acquired
 * here by `assertRequiredLimits` right after acquisition.
 */
const openSiglip2State = async (
  admitted: Siglip2Admission,
  open: ComponentOpener,
  options: Siglip2PipelineOptions = {},
): Promise<Siglip2State> => {
  const { config, quantName, quant, sessionOptions, gpuFeatures } = admitted;
  // The supplier is **the same one** admission saw (`open` only looks up components already
  // opened).
  const vision = open(VISION);
  // MUST: declared features can only be requested at device creation (ADR 0028). A shared GPU
  // cannot be asked for them, so the missing capability is named and fails (a shared GPU was
  // already checked by {@link admitSiglip2} through the same function, but for a device acquired
  // here this is the only check).
  const gpu = options.gpu ?? (await acquireGpu(toAcquireGpuOptions(gpuFeatures)));
  const ownsGpu = options.gpu === undefined;
  try {
    assertGpuFeaturesGranted(gpuFeatures, gpu, `Siglip2Pipeline: quant '${quantName}'`);
    // Also check the limits of a device acquired here — `fromAssets` has no pre-fetch check
    // (`fromPretrained` checks against adapter values); without this a missing declaration surfaces
    // only at Session construction, and a declaration stricter than the real demand is never
    // matched at all. A shared GPU is checked by `admitSiglip2`.
    if (ownsGpu) {
      assertRequiredLimitsSatisfied(
        quant.requiredLimits,
        gpu.limits,
        `Siglip2Pipeline: quant '${quantName}'`,
      );
    }
    return {
      gpu,
      ownsGpu,
      config,
      vision,
      session: await vision.createSession(gpu, sessionOptions),
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

/** Runs one image (preprocessing → one `vision` run → `pooler_output`). */
const embedImage = async (
  state: Siglip2State,
  image: Rgb8Image,
): Promise<Float32Array<ArrayBuffer>> => {
  const { config } = state;
  const pixelValues: Tensor = {
    dtype: "f32",
    shape: [1, CHANNELS, config.imageHeight, config.imageWidth],
    data: preprocessPixelValues(config, image),
  };
  const outputs = await state.session.run({ [PIXEL_VALUES]: pixelValues });
  if (state.onRunDiagnostics !== undefined) state.onRunDiagnostics(state.session.diagnostics());
  const name = state.vision.graph.outputs[0];
  const tensor = outputs[name];
  if (tensor === undefined) throw new Error(`siglip2: グラフ出力 '${name}' が実行結果に無い`);
  const data = asF32(tensor, "pooler_output");
  if (data.length !== config.hiddenDim) {
    throw new Error(
      `siglip2: pooler_output の要素数 ${data.length} が hiddenDim ${config.hiddenDim} と違う`,
    );
  }
  return data;
};

/**
 * SigLIP2's image → embedding pipeline (one vision tower).
 *
 * Construction enters only through {@link Siglip2Pipeline.fromPretrained} (fetch from HF) or
 * {@link Siglip2Pipeline.fromAssets} (already-fetched bytes) — the constructor is private so that
 * no half-built state bypassing the manifest checks and the asset matching can be created
 * (ADR 0008).
 */
export class Siglip2Pipeline {
  readonly #state: Siglip2State;
  /**
   * Serialization chain for embed and dispose ("one at a time" kept on the public API side — module
   * doc).
   */
  readonly #chain = createOperationChain();
  /**
   * The one dispose. **Not being undefined means "disposed"** (a separate boolean would be derived
   * state updated independently).
   */
  #disposal: Promise<void> | undefined;

  private constructor(state: Siglip2State) {
    this.#state = state;
  }

  /**
   * Fetches a distribution and builds (`loadManifest` → `resolveSelection` → fetch **only each
   * component's descriptor (part 0)** for admission → warm the weight parts → `fetchAssets` for the
   * remaining assets → construction). Blocks are read in part order at the moment the Session is
   * built (ADR 0109 — `src/hub/components.ts`). To replace a component with the same role from
   * another repository use {@link FromPretrainedComponentOptions.components} (one role, `vision`).
   *
   * **`ref` is required** (sources have no default — the MUST of `src/hub/repo-ref.ts`). The source
   * this package version verified is `"siglip2"` of {@link SIGLIP2_SOURCES} (`./config.ts`) — to
   * pin reproducibility yourself, the caller states `{ repo, revision }` explicitly. A string `ref`
   * reads as `{ repo }` (= follows `main`).
   *
   * A local distribution is passed as a **source handle** (`localDirectory` / `denoDirectory` of
   * `@karume/hub/deno`). It skips the check of HF's `owner/name` spelling and touches neither the
   * network nor CacheStorage (the HTTP-only knobs of {@link Siglip2FromPretrainedOptions} have no
   * effect).
   */
  static async fromPretrained(
    ref: string | HubRepoRef | DistributionSource,
    options: Siglip2FromPretrainedOptions = {},
  ): Promise<Siglip2Pipeline> {
    const source = toManifestSource(
      ref,
      "Siglip2Pipeline.fromPretrained",
      'SIGLIP2_SOURCES["siglip2"]（@karume/models/siglip2）',
    );
    const hubOptions = hubLoadOptions(options);
    const loaded = await loadManifest(source, hubOptions);
    const choice = {
      ...(options.model === undefined ? {} : { model: options.model }),
      ...(options.quant === undefined ? {} : { quant: options.quant }),
    };
    const selection = resolveSelection(loaded.manifest, choice);
    const buildOptions: Siglip2PipelineOptions = {
      ...(options.gpu === undefined ? {} : { gpu: options.gpu }),
      ...choice,
      ...(options.onRunDiagnostics === undefined
        ? {}
        : { onRunDiagnostics: options.onRunDiagnostics }),
    };
    // The family checks run in the admission hook (before weight parts are fetched —
    // `hub/components.ts`).
    const { admitted, open } = await loadContainerComponents(
      "Siglip2Pipeline.fromPretrained",
      loaded,
      selection,
      COMPONENT_KEYS,
      async (open) => {
        const admitted = admitSiglip2(loaded.manifest, open, buildOptions);
        // The `requiredLimits` the distribution declared are checked here, **before weight parts
        // are fetched** (ADR 0089 decision 5 — a shared GPU's limits, or the adapter's measured
        // values on the path that acquires its own).
        await assertRequiredLimitsBeforeDownload(
          admitted.quant.requiredLimits,
          buildOptions.gpu,
          `Siglip2Pipeline: quant '${admitted.quantName}'`,
        );
        return admitted;
      },
      {
        ...hubOptions,
        ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
        ...(options.components === undefined ? {} : { components: options.components }),
      },
    );
    return new Siglip2Pipeline(await openSiglip2State(admitted, open, buildOptions));
  }

  /**
   * Builds from an already-fetched manifest + assets. Contract checks, asset interpretation,
   * `openContainer` and the match against the graph all happen here, and it returns **with the one
   * `vision` Session created** (the module doc's MUST).
   *
   * Component keys are accepted **in both forms** — one single-form `krm` byte array (`vision`) and
   * the part sequence of the split form (`vision[0]` / `vision[1]` / … — in index order from
   * part 0). The split form is opened as a part sequence without concatenating the bytes. A missing
   * index or mixing with the single-form key fails loudly (the receiving implementation is the one
   * in `src/hub/components.ts`).
   */
  static async fromAssets(
    input: Siglip2Assets,
    options: Siglip2PipelineOptions = {},
  ): Promise<Siglip2Pipeline> {
    const open = await assetOpener(input.assets);
    const admitted = admitSiglip2(input.manifest, open, options);
    return new Siglip2Pipeline(await openSiglip2State(admitted, open, options));
  }

  /**
   * Embeds one image. It returns `pooler_output` **itself** (f32 of length `hiddenDim`, not L2
   * normalized — the module doc's judgement).
   *
   * The input is an RGB8 pixel row (row-major, 3 bytes per pixel); resize / rescale / normalize are
   * applied inside with the distribution's constants. The aspect ratio is not kept (SigLIP2's
   * preprocessing stretches straight to the height/width pair, with no crop and no pad).
   *
   * Concurrent calls **wait and run in order** (one Session is not shared between two runs).
   */
  async embed(image: Rgb8Image): Promise<Float32Array<ArrayBuffer>> {
    // The disposed check happens at call time (checking inside the chain would also fail the
    // embeddings accepted before dispose).
    if (this.#disposal !== undefined) {
      throw new Error("Siglip2Pipeline: dispose 済みでは埋め込めない");
    }
    return await this.#chain(() => embedImage(this.#state, image));
  }

  /**
   * Releases. The Session is always torn down, and **only a GPU acquired internally** is destroyed
   * (a GpuContext passed in `options.gpu` belongs to the caller and is not touched).
   *
   * MUST: destroy only after in-flight embeddings have finished (flush-before-destroy) — putting
   * the destruction on the chain too fixes the order of waiting and destroying in one place. Later
   * calls return the same completion (returning earlier would let the caller treat it as
   * "destroyed" and move on).
   *
   * MUST: `gpu.destroy()` always runs even if `Session.dispose()` fails ({@link disposeSteps}) —
   * with a serial `await`, a reject in the earlier step never reaches the later one, and "a GPU
   * acquired internally is always returned", which the catch of `openXState` promises, would break
   * on the dispose path only. Worse, the rejected promise would stay in `#disposal`, so a second
   * `dispose()` would not retry.
   */
  dispose(): Promise<void> {
    this.#disposal ??= this.#chain(() =>
      disposeSteps([
        () => this.#state.session.dispose(),
        () => {
          if (this.#state.ownsGpu) this.#state.gpu.destroy();
        },
      ]),
    );
    return this.#disposal;
  }

  /** `await using` support (Explicit Resource Management) — an alias of {@link dispose}. */
  [Symbol.asyncDispose](): Promise<void> {
    return this.dispose();
  }
}
