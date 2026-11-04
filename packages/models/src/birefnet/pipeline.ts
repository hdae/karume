/**
 * `Pipeline` — the single surface for image → alpha matte (background removal / salient
 * object segmentation).
 *
 * Pipeline (every step in Karume, no torch):
 *
 * 1. Host: resize the RGB8 pixel row to the size in `pipelineConfig` (antialiased bilinear)
 * 2. Host: apply rescale + normalize folded together into `[1, 3, S, S]` f32
 * 3. Run `matte` once → **the logits before sigmoid** `[1, 1, S, S]`
 * 4. Host: `sigmoid` → **back to the original image resolution** → 8-bit α (`./matte.ts`)
 *
 * Step 4 applies sigmoid before the resize because both checkpoints' upstreams (the bundled
 * `handler.py` and the usage example on Lucida's model card) do `preds.sigmoid()` before going back
 * to the original resolution (interpolating logits extrapolates the saturated region and changes α
 * at the boundary). Step 4's interpolation and order of stages follow each checkpoint's upstream,
 * and which one applies is decided by the distribution's `pipelineConfig.postprocess` declaration
 * (HR = `u8-pil-bicubic`, Lucida = `float-bilinear`. Both quantize by rounding; upstream truncation
 * is not copied — ADR 0125. Both checkpoints' actual upstream procedures are in the module doc of
 * `./matte.ts`).
 *
 * ## MUST: one Session is created at construction and kept until `dispose` (unlike Anima / Irodori)
 *
 * Those create and tear down per step because several huge graphs cannot be resident at once
 * (ADR 0016). Here there is one graph (964MB at 1024²) with nothing to make room for; the opposite,
 * "create and tear down per image", **would re-upload the whole weights for every image**.
 * Background removal normally runs many images, so residency is the default shape (the same reason
 * and the same shape as SigLIP2).
 *
 * MUST: it still goes on the serialization chain — concurrent `segment` calls would share one
 * Session between two `run`s, and `dispose` would destroy the GPU without waiting for in-flight
 * work (a flush-before-destroy violation).
 *
 * ## It returns **one alpha matte only** (no RGBA compositing, no PNG)
 *
 * {@link Pipeline.segment} returns "an α of the same size as the input, one byte per
 * pixel", not an RGBA cut-out of the foreground.
 *
 * - **The compositing policy depends on the use** (pass straight α to a canvas / composite onto
 *   white or a solid colour and make a JPEG / insert colour decontamination). Choosing one here
 *   closes the door on the others.
 * - Compositing itself is one line, `rgba[i * 4 + 3] = alpha[i]`, not worth an abstraction (see
 *   "Usage" below).
 * - `encodePng` is truecolor RGB only and **does not accept pixels whose α is not 255**
 *   (`src/image/png.ts`). Adding an entry that returns RGBA has no use outside the browser (Deno /
 *   Node) unless an entry that writes it is added at the same time — one entry "for convenience"
 *   becomes two features.
 *
 * Usage (compositing onto white and writing a PNG):
 *
 * ```ts
 * const matte = await pipeline.segment(image);
 * const rgba = new Uint8Array(matte.width * matte.height * 4);
 * for (let i = 0; i < matte.data.length; i += 1) {
 *   const a = matte.data[i] / 255;
 *   for (let c = 0; c < 3; c += 1) {
 *     rgba[i * 4 + c] = Math.round(image.data[i * 3 + c] * a + 255 * (1 - a));
 *   }
 *   rgba[i * 4 + 3] = 255;
 * }
 * await encodePng(rgba, matte.width, matte.height);
 * ```
 *
 * ## MUST: numerical correctness is not guaranteed here
 *
 * Correctness is guaranteed by the per-graph golden E2E (`packages/e2e/tests/e2e_birefnet_test.ts`
 * — real weights of 2 series × 4 synthetic + 4 real-image cases) and the preprocessing parity check
 * (`packages/models/tests/image_preprocess_test.ts` / real images in
 * `packages/e2e/tests/e2e_birefnet_real_test.ts`).
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
  BIREFNET_PIPELINE_MAJOR,
  BIREFNET_PIPELINE_NAME,
  type BirefnetPipelineConfig,
  parseBirefnetPipelineConfig,
} from "./config.ts";
import { normalizeToNchw, resizeRgb8, type Rgb8Image } from "../image/preprocess.ts";
import { type AlphaMatte, matteFromLogits } from "./matte.ts";
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
const MATTE = "matte";

/** This family's weights components (also the role names component replacement accepts). */
const COMPONENT_KEYS = [MATTE] as const;

/** Name of the graph input (`INPUT_NAME` in `tools/export-recipes/birefnet/export.py`). */
const PIXEL_VALUES = "pixel_values";

/** Input channel count (RGB). The graph is baked with static 1 / 3 for it and the batch. */
const CHANNELS = 3;

/** Output channel count (one matte). */
const MATTE_CHANNELS = 1;

/**
 * Construction options (shared by {@link Pipeline.fromAssets} /
 * {@link Pipeline.fromPretrained}).
 */
export type PipelineOptions = {
  /**
   * Shares an existing GPU. **The caller keeps ownership**, so {@link Pipeline.dispose}
   * does not destroy it. When omitted, the pipeline calls `acquireGpu` internally and destroys it
   * in `dispose()`.
   */
  readonly gpu?: GpuContext;
  /** Model (a key of the manifest's models). Defaults to `defaultModel`. */
  readonly model?: string;
  /** Execution configuration (a key of that model's quants). Defaults to `defaultQuant`. */
  readonly quant?: string;
  /**
   * Observation hook that receives the diagnostics of each run (one matte = one `matte` run). For
   * per-op GPU time (`lastRunTiming`) pass `acquireGpu({ gpuTiming: true })` as `gpu` (ADR 0021 —
   * no timing by default).
   *
   * NOTE: like SigLIP2, no component name is passed — there is one graph, so the name would always
   * be the same single value (no argument the receiver cannot branch on).
   *
   * Exceptions from the callback are not caught (fail loudly — the segmentation fails with it).
   */
  readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics) => void;
};

/**
 * Fetch-layer options used only by {@link Pipeline.fromPretrained} (passed through to the
 * hub).
 *
 * NOTE: the description of the fetch-layer knobs, including that `headers` / `fetch` / `caches` /
 * `onRetry` are **for HTTP sources only**, lives in one place, {@link FromPretrainedHubOptions}.
 */
export type FromPretrainedOptions = PipelineOptions &
  FromPretrainedHubOptions &
  FromPretrainedComponentOptions & {
    readonly signal?: AbortSignal;
  };

/** Input for building directly from fetched assets (pass the hub's `fetchAssets` result as is). */
export type Assets = {
  readonly manifest: Manifest;
  readonly assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>;
};

/**
 * Turns the fetched bytes into an ArrayBuffer that can be passed to `openContainer` (the check
 * itself is {@link readAssetBuffer}).
 *
 * MUST: do not copy with `slice` — a 1024² distribution is 964MB in one file, and the host RAM peak
 * would double.
 */
const assetBuffer = (
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  key: string,
): ArrayBuffer => readAssetBuffer("birefnet", "weights", assets, key);

/**
 * The component supplier of the full-asset surface (`fromAssets`) (the receiving implementation is
 * shared by every family — {@link assetComponentOpener}). A component key is either one single-form
 * `krm` (`matte`) or the part sequence of the split form (`matte[0]` / `matte[1]` / …).
 */
const assetOpener = (assets: Assets["assets"]): Promise<ComponentOpener> =>
  assetComponentOpener("birefnet", assets, (key) => assetBuffer(assets, key), COMPONENT_KEYS);

/**
 * Checks that the **static** dimension of one graph-input axis matches the `pipelineConfig`
 * declaration.
 *
 * MUST: do not drop this check. Preprocessing resizes to the declared size, so even when the graph
 * was baked at another resolution **the host side passes all the way through** (the failure comes
 * from the Session's shape check, by which time the reader cannot tell "which number is right").
 * This is the only detector of mixed-up assets between resolution series (1024² / 2048²).
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
): void => assertGraphInputDim("birefnet", model, inputName, axis, expected, where);

/**
 * Checks that the graph **output** has shape `[1, 1, imageHeight, imageWidth]`.
 *
 * MUST: do not drop this check. The later steps could check only "element count is H×W", but then a
 * graph baked with the intermediate predictions of multi-scale supervision (`[1, 3, S, S]` or
 * similar) would pass through **with different values read as α**.
 *
 * NOTE: the `export` is for tests that pin the rejection paths without a GPU (not exposed on
 * `mod.ts` / the subpath surface — ADR 0008).
 */
export const assertMatteShape = (
  model: GraphOwner,
  config: BirefnetPipelineConfig,
  where: string,
): void => {
  const name = model.graph.outputs[0];
  const value = model.graph.values[name];
  if (value === undefined) {
    throw new Error(`birefnet: graph output '${name}' has no declaration (${where})`);
  }
  const expected = [1, MATTE_CHANNELS, config.imageHeight, config.imageWidth];
  if (
    value.shape.length !== expected.length ||
    value.shape.some((dim, axis) => dim !== expected[axis])
  ) {
    throw new Error(
      `birefnet: ${where} — graph output '${name}' has shape ` +
        `[${value.shape.map(String).join(", ")}], expected [${expected.join(", ")}]`,
    );
  }
};

const asF32 = (tensor: Tensor, where: string): Float32Array<ArrayBuffer> => {
  if (tensor.dtype !== "f32") throw new Error(`${where}: not f32 (${tensor.dtype})`);
  return tensor.data;
};

/**
 * RGB8 pixel row → the `pixel_values` passed to the graph (`[3, S, S]` flattened).
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
  config: BirefnetPipelineConfig,
  image: Rgb8Image,
): Float32Array<ArrayBuffer> =>
  normalizeToNchw(
    resizeRgb8(image, config.imageWidth, config.imageHeight, config.interpolation),
    config.imageMean,
    config.imageStd,
  );

/**
 * Internal state of {@link Pipeline} (not on the public surface).
 *
 * NOTE: the `export` exists because the A/B tool (`tools/birefnet-ab/`, removed in 0.15) used
 * {@link loadBirefnetState} / {@link matteLogits} (not exposed on `mod.ts` / the subpath surface —
 * ADR 0008).
 */
export type BirefnetState = {
  readonly gpu: GpuContext;
  readonly ownsGpu: boolean;
  readonly config: BirefnetPipelineConfig;
  readonly matte: ModelComponent;
  /** The one Session created at construction and kept until `dispose` (the module doc's MUST). */
  readonly session: Session;
  readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics) => void;
};

/**
 * The execution knobs birefnet accepts (the manifest's quant declaration). The merge rule
 * (explicit > declared > runtime default), value ranges, combinations and the classification of
 * the thrown types are owned by `resolveSessionOptions`, shared by every family; this only decides
 * the accepted set. There is no explicit override (the construction options have no execution-knob
 * field).
 *
 * No key is accepted because the distribution has no quant that declares execution knobs (`session`
 * is empty for both the f32 and the f16 preset — only the weight storage differs). When a
 * declaration appears, check it against the reference values before accepting it — do not silently
 * pass it to the runtime.
 *
 * NOTE: the `export` is for the equivalence test that runs every quant of the mirror through the
 * same table (not exposed on `mod.ts` / the subpath surface — ADR 0008).
 */
export const BIREFNET_SESSION_POLICY: FamilySessionPolicy = {
  linearCompute: false,
  convCompute: false,
  attentionCompute: false,
  attentionScoreStorage: false,
  linearGemvReduce: false,
  stateAttentionReduce: false,
  fuseRmsNormAdd: false,
  fuseLinearStaticQuantize: false,
  fuseGeluMulStaticQuantize: false,
  packedStaticQuantize: false,
  integerGemv: false,
  linearGemvPlanes: false,
};

/** What the family admission ({@link admitBirefnet}) settles. */
type BirefnetAdmission = {
  readonly config: BirefnetPipelineConfig;
  readonly quantName: string;
  readonly quant: Quant;
  /**
   * The effective settings after passing the quant declaration through the acceptance table
   * ({@link BIREFNET_SESSION_POLICY}).
   */
  readonly sessionOptions: SessionOptions;
  /**
   * The quant declaration plus the features the effective settings need (both the request and the
   * check read this).
   */
  readonly gpuFeatures: GpuFeaturesSpec | undefined;
};

/**
 * Checks whether this manifest and this graph can run as birefnet (the family admission hook of
 * `hub/components.ts` — on the fetch surface it is called **before a single byte of the weight
 * parts is fetched**).
 *
 * MUST: gather the family's checks in this one function. Scattered into later steps, on the fetch
 * surface they would fail only **after** GB-class weights are downloaded (the implementation would
 * be narrower than the wording of ADR 0070 decision 5).
 * MUST: manifest contract violations and **the match against the graph** fail **before the GPU is
 * acquired**. In the other order, an environment without a GPU turns them into a different
 * exception and the reader cannot tell "what was wrong".
 */
const admitBirefnet = (
  manifest: Manifest,
  open: ComponentOpener,
  options: PipelineOptions,
): BirefnetAdmission => {
  const modelName = options.model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, modelName)) {
    throw new Error(
      `birefnet.Pipeline: model '${modelName}' is not in the manifest` +
        ` (available: ${manifest.available.models.join(" / ")})`,
    );
  }
  const entry: ModelEntry = manifest.models[modelName];
  const { name, major } = entry.pipeline;
  if (name !== BIREFNET_PIPELINE_NAME) {
    throw new Error(
      `birefnet.Pipeline: the manifest's pipeline is '${name}/${major}'` +
        ` ('${BIREFNET_PIPELINE_NAME}/${BIREFNET_PIPELINE_MAJOR}' is required)`,
    );
  }
  if (major !== BIREFNET_PIPELINE_MAJOR) {
    // The only check that stops silent degradation from "old implementation × new repository"
    // (ADR 0038 §6).
    throw new Error(
      `birefnet.Pipeline: the major of pipeline '${name}/${major}' is not supported` +
        ` (this implementation reads ${BIREFNET_PIPELINE_NAME}/${BIREFNET_PIPELINE_MAJOR})`,
    );
  }
  const config = parseBirefnetPipelineConfig(entry.pipelineConfig);

  const quantName = options.quant ?? entry.defaultQuant;
  if (!Object.hasOwn(entry.quants, quantName)) {
    throw new Error(
      `birefnet.Pipeline: quant '${quantName}' is not in the manifest` +
        ` (available: ${entry.available.quants.join(" / ")})`,
    );
  }
  const quant = entry.quants[quantName];
  // Unsupported declarations fail before components are opened (= before weight parts are fetched;
  // one path shared by every family).
  const sessionOptions = resolveSessionOptions(
    BIREFNET_SESSION_POLICY,
    quant.session,
    {},
    `birefnet.Pipeline: quant '${quantName}'`,
  );
  const gpuFeatures = sessionGpuFeatures(quant.gpuFeatures, sessionOptions);

  const matte = open(MATTE);
  // Match the graph declaration against pipelineConfig. Checking that there is exactly one input
  // and one output matters because, if a graph baked with the intermediate predictions of
  // multi-scale supervision got mixed in, the later steps that pick by position would silently read
  // other values as α.
  if (matte.graph.inputs.length !== 1 || matte.graph.outputs.length !== 1) {
    throw new Error(
      `birefnet.Pipeline: the matte graph has ${matte.graph.inputs.length} /` +
        ` ${matte.graph.outputs.length} inputs / outputs (one matte requires exactly 1 of each)`,
    );
  }
  assertGraphInputDtype("birefnet", matte, PIXEL_VALUES, "f32", "preprocessing's pixel_values");
  assertStaticDim(matte, PIXEL_VALUES, 0, 1, "batch is static 1");
  assertStaticDim(matte, PIXEL_VALUES, 1, CHANNELS, "3 RGB channels");
  assertStaticDim(matte, PIXEL_VALUES, 2, config.imageHeight, "imageHeight");
  assertStaticDim(matte, PIXEL_VALUES, 3, config.imageWidth, "imageWidth");
  assertMatteShape(matte, config, "matte shape");

  // MUST: a shared GPU's missing capability (feature / device limit) fails here — unlike acquiring
  // our own, there is no reason to wait for `acquireGpu`, and this is the only family check that
  // knows before the weights are downloaded (the mapping between request and check is the one
  // function in `session/gpu-features.ts`, and the later check calls the same function).
  if (options.gpu !== undefined) {
    assertGpuFeaturesGranted(gpuFeatures, options.gpu, `birefnet.Pipeline: quant '${quantName}'`);
    assertRequiredLimitsSatisfied(
      quant.requiredLimits,
      options.gpu.limits,
      `birefnet.Pipeline: quant '${quantName}'`,
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
 * features; limits are checked by {@link admitBirefnet} for a shared GPU, and for a device acquired
 * here by `assertRequiredLimits` right after acquisition.
 */
const openBirefnetState = async (
  admitted: BirefnetAdmission,
  open: ComponentOpener,
  options: PipelineOptions = {},
): Promise<BirefnetState> => {
  const { config, quantName, quant, sessionOptions, gpuFeatures } = admitted;
  // The supplier is **the same one** admission saw (`open` only looks up components already
  // opened).
  const matte = open(MATTE);
  // MUST: declared features can only be requested at device creation (ADR 0028). A shared GPU
  // cannot be asked for them, so the missing capability is named and fails (a shared GPU was
  // already checked by {@link admitBirefnet} through the same function, but for a device acquired
  // here this is the only check).
  const gpu = options.gpu ?? (await acquireGpu(toAcquireGpuOptions(gpuFeatures)));
  const ownsGpu = options.gpu === undefined;
  try {
    assertGpuFeaturesGranted(gpuFeatures, gpu, `birefnet.Pipeline: quant '${quantName}'`);
    // Also check the limits of a device acquired here — `fromAssets` has no pre-fetch check
    // (`fromPretrained` checks against adapter values); without this a missing declaration surfaces
    // only at Session construction, and a declaration stricter than the real demand is never
    // matched at all. A shared GPU is checked by `admitBirefnet`.
    if (ownsGpu) {
      assertRequiredLimitsSatisfied(
        quant.requiredLimits,
        gpu.limits,
        `birefnet.Pipeline: quant '${quantName}'`,
      );
    }
    return {
      gpu,
      ownsGpu,
      config,
      matte,
      session: await matte.createSession(gpu, sessionOptions),
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
 * The logit map of one image (preprocessing → one `matte` run — `imageWidth × imageHeight` f32,
 * flattened row-major).
 *
 * NOTE: the `export` exists for the A/B tool (`tools/birefnet-ab/`, removed in 0.15 — it applied
 * every post-processing procedure to the logits of one run instead of re-running the graph per
 * procedure) (not exposed on `mod.ts` / the subpath surface — ADR 0008).
 *
 * MUST (caller): called from outside the pipeline there is no serialization chain — with the same
 * `state`, wait for the previous call to finish before the next one, and own the release of the
 * Session (and of the GPU you let it acquire) when done (the MUST of {@link loadBirefnetState}).
 */
export const matteLogits = async (
  state: BirefnetState,
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
  const name = state.matte.graph.outputs[0];
  const tensor = outputs[name];
  if (tensor === undefined)
    throw new Error(`birefnet: graph output '${name}' is not in the run result`);
  const logits = asF32(tensor, "matte");
  const expected = config.imageWidth * config.imageHeight;
  if (logits.length !== expected) {
    throw new Error(
      `birefnet: matte element count ${logits.length} does not match ` +
        `${config.imageWidth}×${config.imageHeight}`,
    );
  }
  return logits;
};

/**
 * Runs one image (preprocessing → one `matte` run → sigmoid → 8-bit α at the original resolution).
 *
 * MUST: the post-processing procedure comes from the distribution's declaration
 * (`config.postprocess`) — like the preprocessing interpolation, the declaration and the effective
 * value are one (the same reason as the MUST of `preprocessPixelValues`).
 */
const segmentImage = async (state: BirefnetState, image: Rgb8Image): Promise<AlphaMatte> =>
  matteFromLogits(
    await matteLogits(state, image),
    state.config.imageWidth,
    state.config.imageHeight,
    image.width,
    image.height,
    state.config.postprocess,
  );

/**
 * Fetches a distribution and builds the execution state (the body of
 * {@link Pipeline.fromPretrained} — the sequence is described in its doc).
 *
 * NOTE: the `export` exists so that the A/B tool (`tools/birefnet-ab/`, removed in 0.15) could take
 * logits through the same checks and the same graph as the pipeline (not exposed on `mod.ts` / the
 * subpath surface — ADR 0008).
 *
 * MUST (caller): the returned state comes without {@link Pipeline}'s serialization chain
 * and `dispose`, so the caller takes over both roles — call {@link matteLogits} only after the
 * previous call has finished (do not share one Session between two `run`s). When done, call
 * `state.session.dispose()` yourself (after in-flight runs have finished — flush-before-destroy).
 * If `options.gpu` was omitted and this function acquired the GPU (`state.ownsGpu`), the
 * `state.gpu.destroy()` after tearing down the Session is also the caller's (the shape that always
 * runs both is {@link disposeSteps}, the same as {@link Pipeline.dispose}).
 */
export const loadBirefnetState = async (
  ref: string | HubRepoRef | DistributionSource,
  options: FromPretrainedOptions = {},
): Promise<BirefnetState> => {
  const source = toManifestSource(
    ref,
    "birefnet.Pipeline.fromPretrained",
    'SOURCES["birefnet-hr"] (@karume/models/birefnet)',
  );
  const hubOptions = hubLoadOptions(options);
  const loaded = await loadManifest(source, hubOptions);
  const choice = {
    ...(options.model === undefined ? {} : { model: options.model }),
    ...(options.quant === undefined ? {} : { quant: options.quant }),
  };
  const selection = resolveSelection(loaded.manifest, choice);
  const buildOptions: PipelineOptions = {
    ...(options.gpu === undefined ? {} : { gpu: options.gpu }),
    ...choice,
    ...(options.onRunDiagnostics === undefined
      ? {}
      : { onRunDiagnostics: options.onRunDiagnostics }),
  };
  // The family checks run in the admission hook (before weight parts are fetched —
  // `hub/components.ts`).
  const { admitted, open } = await loadContainerComponents(
    "birefnet.Pipeline.fromPretrained",
    loaded,
    selection,
    COMPONENT_KEYS,
    async (open) => {
      const admitted = admitBirefnet(loaded.manifest, open, buildOptions);
      // The `requiredLimits` the distribution declared are checked here, **before weight parts are
      // fetched** (ADR 0089 decision 5 — a shared GPU's limits, or the adapter's measured values on
      // the path that acquires its own).
      await assertRequiredLimitsBeforeDownload(
        admitted.quant.requiredLimits,
        buildOptions.gpu,
        `birefnet.Pipeline: quant '${admitted.quantName}'`,
      );
      return admitted;
    },
    {
      ...hubOptions,
      ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
      ...(options.components === undefined ? {} : { components: options.components }),
    },
  );
  return await openBirefnetState(admitted, open, buildOptions);
};

/**
 * The BiRefNet family's image → alpha matte pipeline (one graph).
 *
 * Construction enters only through {@link Pipeline.fromPretrained} (fetch from HF) or
 * {@link Pipeline.fromAssets} (already-fetched bytes) — the constructor is private so that
 * no half-built state bypassing the manifest checks and the asset matching can be created
 * (ADR 0008).
 */
export class Pipeline {
  readonly #state: BirefnetState;
  /**
   * Serialization chain for segment and dispose ("one at a time" kept on the public API side —
   * module doc).
   */
  readonly #chain = createOperationChain();
  /**
   * The one dispose. **Not being undefined means "disposed"** (a separate boolean would be derived
   * state updated independently).
   */
  #disposal: Promise<void> | undefined;

  private constructor(state: BirefnetState) {
    this.#state = state;
  }

  /**
   * Fetches a distribution and builds (`loadManifest` → `resolveSelection` → fetch **only each
   * component's descriptor (part 0)** for admission → warm the weight parts → `fetchAssets` for the
   * remaining assets → construction). Blocks are read in part order at the moment the Session is
   * built (ADR 0109 — `src/hub/components.ts`). To replace a component with the same role from
   * another repository use {@link FromPretrainedComponentOptions.components} (one role, `matte`). A
   * string `ref` reads as `{ repo }` (= follows `main`). **`ref` is required** (sources have no
   * default — the MUST of `src/hub/repo-ref.ts`). The verified pins are `"birefnet-hr"` /
   * `"lucida"` of {@link SOURCES} (`./config.ts`) — one repository holds the two
   * models 1024² / 2048², and 2048² is chosen with `{ model: "2048" }`.
   *
   * A local distribution is passed as a **source handle** (`localDirectory` / `denoDirectory` of
   * `@karume/hub/deno`). It skips the check of HF's `owner/name` spelling and touches neither the
   * network nor CacheStorage (the HTTP-only knobs of {@link FromPretrainedOptions} have no
   * effect).
   */
  static async fromPretrained(
    ref: string | HubRepoRef | DistributionSource,
    options: FromPretrainedOptions = {},
  ): Promise<Pipeline> {
    return new Pipeline(await loadBirefnetState(ref, options));
  }

  /**
   * Builds from an already-fetched manifest + assets. Contract checks, asset interpretation,
   * `openContainer` and the match against the graph all happen here, and it returns **with the one
   * `matte` Session created** (the module doc's MUST).
   *
   * Component keys are accepted **in both forms** — one single-form `krm` byte array (`matte`) and
   * the part sequence of the split form (`matte[0]` / `matte[1]` / … — in index order from part 0).
   * The split form is opened as a part sequence without concatenating the bytes. A missing index or
   * mixing with the single-form key fails loudly (the receiving implementation is the one in
   * `src/hub/components.ts`).
   */
  static async fromAssets(input: Assets, options: PipelineOptions = {}): Promise<Pipeline> {
    const open = await assetOpener(input.assets);
    const admitted = admitBirefnet(input.manifest, open, options);
    return new Pipeline(await openBirefnetState(admitted, open, options));
  }

  /**
   * Produces the alpha matte of one image. It returns an 8-bit α **of the same size as the input**
   * (the module doc's "It returns one alpha matte only").
   *
   * The input is an RGB8 pixel row (row-major, 3 bytes per pixel); resize / rescale / normalize are
   * applied inside with the distribution's constants. The aspect ratio is not kept (the upstream
   * preprocessing stretches straight to the height/width pair, with no crop and no pad).
   *
   * Concurrent calls **wait and run in order** (one Session is not shared between two runs).
   */
  async segment(image: Rgb8Image): Promise<AlphaMatte> {
    // The disposed check happens at call time (checking inside the chain would also fail the
    // segmentations accepted before dispose).
    if (this.#disposal !== undefined) {
      throw new Error("birefnet.Pipeline: cannot segment after dispose");
    }
    return await this.#chain(() => segmentImage(this.#state, image));
  }

  /**
   * Releases. The Session is always torn down, and **only a GPU acquired internally** is destroyed
   * (a GpuContext passed in `options.gpu` belongs to the caller and is not touched).
   *
   * MUST: destroy only after in-flight runs have finished (flush-before-destroy) — putting the
   * destruction on the chain too fixes the order of waiting and destroying in one place. Later
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
