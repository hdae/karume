/**
 * `Pipeline` — the single surface for image → relative depth map (single-image depth
 * estimation).
 *
 * Pipeline (every step in Karume, no torch):
 *
 * 1. Host: resize the RGB8 pixel row to the size in `pipelineConfig` (antialiased **bicubic**)
 * 2. Host: apply rescale + normalize folded together into `[1, 3, S, S]` f32
 * 3. Run `depth` once → **relative depth** `[1, S, S]` (after the head's final ReLU, non-negative)
 * 4. Host: back to the original image resolution
 *
 * The implementation and the reference implementation of the two preprocessing steps live in
 * `src/image/preprocess.ts` (`resizeRgb8` / `normalizeToNchw`); this only wires them with the
 * `pipelineConfig` constants. **Decoding is not karume's job** — the entry is "RGB8 pixel row +
 * width + height", and there is no PNG / JPEG decoder (in a browser `createImageBitmap` + canvas,
 * on Node / Deno any decoder).
 *
 * MUST: the interpolation is **bicubic** (`"resample": 3` of the upstream
 * `preprocessor_config.json`). It is a different filter from the bilinear of SigLIP2 / BiRefNet,
 * and the declaration's accepted set is narrowed to one value in `src/depth-anything/config.ts` (a
 * mix-up shifts `pixel_values` by up to 0.59 = 34 times one uint8 LSB).
 *
 * ## MUST: one Session is created at construction and kept until `dispose` (unlike Anima / Irodori)
 *
 * Those create and tear down per step because several huge graphs cannot be resident at once
 * (ADR 0016). Here there is one graph (99MB for Small) with nothing to make room for; the opposite,
 * "create and tear down per image", **would re-upload the whole weights for every image**. Depth
 * estimation normally runs many images, so residency is the default shape (the same reason and the
 * same shape as SigLIP2 / BiRefNet).
 *
 * MUST: it still goes on the serialization chain — concurrent `estimate` calls would share one
 * Session between two `run`s, and `dispose` would destroy the GPU without waiting for in-flight
 * work (a flush-before-destroy violation).
 *
 * ## It returns **raw relative depth in f32** (not folded into `[0, 1]`, no PNG)
 *
 * {@link Pipeline.estimate} returns "one f32 per pixel, at the size of the input",
 * with no normalization and no colouring.
 *
 * - **Normalization is a visualization decision, not part of inference.** Folding by min-max makes
 *   the original scale unrecoverable and removes the way to put two depth maps on the same scale or
 *   to use a threshold across them (the other direction takes three lines — "Usage" below). The
 *   same line as BiRefNet returning an alpha matte without RGBA compositing.
 * - Nor does it return `min` / `max` alongside — they are **derivable** from the returned array,
 *   and no independently updated field is created (CLAUDE.md's "do not hold derivable state
 *   twice").
 * - **Relative depth has neither a unit nor an origin** (larger = nearer). An entry that looks like
 *   `[0, 1]` would read as a promise of absolute distance — these weights are not metric depth
 *   (`depth_estimation_type` is `relative`; `tools/export-recipes/depth_anything/patch.py` checks
 *   the configuration).
 *
 * Usage (folding into `[0, 1]` for visualization):
 *
 * ```ts
 * const depth = await pipeline.estimate(image);
 * let min = Infinity;
 * let max = -Infinity;
 * for (const value of depth.data) {
 *   if (value < min) min = value;
 *   if (value > max) max = value;
 * }
 * const span = max - min;
 * const normalized = depth.data.map((value) => (span > 0 ? (value - min) / span : 0));
 * ```
 *
 * ## NOTE: step 4 is bilinear (unlike upstream's `post_process_depth_estimation`)
 *
 * Step 4 uses `resizePlaneF32` (`src/image/preprocess.ts` — **antialiased bilinear** with the same
 * support as Pillow / torchvision). Upstream transformers'
 * `DPTImageProcessor.post_process_depth_estimation` is
 * `F.interpolate(mode="bicubic", align_corners=False)` (a = −0.75, no antialias), so **both the
 * kernel and the support stretch on downscaling differ** (when going back to an image smaller than
 * the baked 518², this side applies area averaging). They are not matched because
 *
 * - the model's answer is the `predicted_depth` at the baked resolution (518²) itself, and
 *   upscaling from there adds no information — it is **a stretch to the display resolution**.
 *   Numerical correctness is checked on that 518² map by
 *   `packages/e2e/tests/e2e_depth_anything_real_test.ts`.
 * - `post_process_depth_estimation` is an optional convenience function on the transformers side
 *   and does not appear in the checkpoint's definition (`preprocessor_config.json`). Unlike the
 *   preprocessing `resample`, **it is not a fact the distribution holds as a declaration**.
 *
 * Conversely, **no** claim is made of "bit-for-bit agreement with the upstream post-processing". If
 * that becomes necessary, the design would add an `F.interpolate`-compatible cubic branch (a
 * = −0.75, no antialias) to `resizePlaneF32` (today bilinear is fixed).
 *
 * ## MUST: numerical correctness is not guaranteed here
 *
 * Correctness is guaranteed by `packages/e2e/tests/e2e_depth_anything_real_test.ts` (the model
 * output pinned per environment by its sha256 rows — ADR 0106 — and the real-image discrimination
 * test; the torch golden-parity cases were dropped in the 0.15 test triage) and the preprocessing
 * parity check (`packages/models/tests/image_preprocess_test.ts` / real images in the same
 * `packages/e2e/tests/e2e_depth_anything_real_test.ts`).
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
  DEPTH_ANYTHING_PIPELINE_MAJOR,
  DEPTH_ANYTHING_PIPELINE_NAME,
  type DepthAnythingPipelineConfig,
  parseDepthAnythingPipelineConfig,
} from "./config.ts";
import {
  normalizeToNchw,
  resizePlaneF32,
  resizeRgb8,
  type Rgb8Image,
} from "../image/preprocess.ts";
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
const DEPTH = "depth";

/** This family's weights components (also the role names component replacement accepts). */
const COMPONENT_KEYS = [DEPTH] as const;

/** Name of the graph input (`INPUT_NAME` in `tools/export-recipes/depth_anything/export.py`). */
const PIXEL_VALUES = "pixel_values";

/** Input channel count (RGB). The graph is baked with static 1 / 3 for it and the batch. */
const CHANNELS = 3;

/**
 * A relative depth map of the same size as the input (one f32 per pixel, row-major, non-negative,
 * larger = nearer).
 */
export type DepthMap = {
  readonly data: Float32Array<ArrayBuffer>;
  readonly width: number;
  readonly height: number;
};

/**
 * Construction options (shared by {@link Pipeline.fromAssets} /
 * {@link Pipeline.fromPretrained}).
 */
export type PipelineOptions = {
  /**
   * Shares an existing GPU. **The caller keeps ownership**, so
   * {@link Pipeline.dispose} does not destroy it. When omitted, the pipeline calls
   * `acquireGpu` internally and destroys it in `dispose()`.
   */
  readonly gpu?: GpuContext;
  /** Model (a key of the manifest's models). Defaults to `defaultModel`. */
  readonly model?: string;
  /** Execution configuration (a key of that model's quants). Defaults to `defaultQuant`. */
  readonly quant?: string;
  /**
   * Observation hook that receives the diagnostics of each run (one depth map = one `depth` run).
   * For per-op GPU time (`lastRunTiming`) pass `acquireGpu({ gpuTiming: true })` as `gpu` (ADR 0021
   * — no timing by default).
   *
   * NOTE: like SigLIP2 / BiRefNet, no component name is passed — there is one graph, so the name
   * would always be the same single value (no argument the receiver cannot branch on).
   *
   * Exceptions from the callback are not caught (fail loudly — the estimation fails with it).
   */
  readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics) => void;
};

/**
 * Fetch-layer options used only by {@link Pipeline.fromPretrained} (passed through to
 * the hub).
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
 * MUST: do not copy with `slice` — the distribution is 99MB in one file, and the host RAM peak
 * would double.
 */
const assetBuffer = (
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  key: string,
): ArrayBuffer => readAssetBuffer("depth-anything", "weights", assets, key);

/**
 * The component supplier of the full-asset surface (`fromAssets`) (the receiving implementation is
 * shared by every family — {@link assetComponentOpener}). A component key is either one single-form
 * `krm` (`depth`) or the part sequence of the split form (`depth[0]` / `depth[1]` / …).
 */
const assetOpener = (assets: Assets["assets"]): Promise<ComponentOpener> =>
  assetComponentOpener("depth-anything", assets, (key) => assetBuffer(assets, key), COMPONENT_KEYS);

/**
 * Checks that the **static** dimension of one graph-input axis matches the `pipelineConfig`
 * declaration.
 *
 * MUST: do not drop this check. Preprocessing resizes to the declared size, so even when the graph
 * was baked at another resolution **the host side passes all the way through** (the failure comes
 * from the Session's shape check, by which time the reader cannot tell "which number is right").
 * DINOv2's position embeddings are tied to the patch count, so a graph at anything but the
 * pretraining resolution cannot be baked in the first place (② of
 * `tools/export-recipes/depth_anything/patch.py`) — so a failure here is always **a mixed-up
 * asset**.
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
): void => assertGraphInputDim("depth-anything", model, inputName, axis, expected, where);

/**
 * Checks that the graph **output** has shape `[1, imageHeight, imageWidth]`.
 *
 * MUST: do not drop this check. The later steps could check only "element count is H×W", but then a
 * graph baked with the channel axis kept (`[1, 1, S, S]`) or another export that also emits
 * intermediate stages would pass through **with different values read as depth**. The rank is
 * checked too because `[1, S, S]` and `[1, 1, S, S]` cannot be told apart by element count.
 *
 * NOTE: the `export` is for tests that pin the rejection paths without a GPU (not exposed on
 * `mod.ts` / the subpath surface — ADR 0008).
 */
export const assertDepthShape = (
  model: GraphOwner,
  config: DepthAnythingPipelineConfig,
  where: string,
): void => {
  const name = model.graph.outputs[0];
  const value = model.graph.values[name];
  if (value === undefined) {
    throw new Error(`depth-anything: graph output '${name}' has no declaration (${where})`);
  }
  const expected = [1, config.imageHeight, config.imageWidth];
  if (
    value.shape.length !== expected.length ||
    value.shape.some((dim, axis) => dim !== expected[axis])
  ) {
    throw new Error(
      `depth-anything: ${where} — graph output '${name}' has shape ` +
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
 * The aspect ratio is not kept (the graph accepts one square size only, so stretching is the only
 * way — the NOTE in `config.ts`).
 *
 * NOTE: the `export` is for tests that call the wiring directly (reaching here through `fromAssets`
 * needs a real IR container). Not exposed on `mod.ts` / the subpath surface (ADR 0008).
 */
export const preprocessPixelValues = (
  config: DepthAnythingPipelineConfig,
  image: Rgb8Image,
): Float32Array<ArrayBuffer> =>
  normalizeToNchw(
    resizeRgb8(image, config.imageWidth, config.imageHeight, config.interpolation),
    config.imageMean,
    config.imageStd,
  );

/**
 * The graph's depth map → the original image resolution (step 4 of the module doc).
 *
 * **The values are not touched** — no normalization, no clamp (it returns raw relative depth — the
 * module doc's "It returns raw relative depth in f32"). The weights of `resizePlaneF32` are
 * non-negative and sum to 1, so the output's non-negativity (the head's final ReLU) survives
 * upscaling.
 *
 * At the identity size (`sourceWidth === width && sourceHeight === height`) the values pass through
 * unchanged (the support of `resizePlaneF32` shrinks to the nearest single point — pinned by the
 * preprocessing layer's tests), so a call with "original resolution = baked resolution" incurs no
 * rounding loss.
 *
 * MUST: non-finite values fail with their pixel coordinates **before** the resize (the position
 * closest to the source). `resizePlaneF32` is a weighted sum that spreads NaN to the neighbourhood
 * before returning, and the min / max scan of the visualization recipe this doc hands out passes
 * NaN through because NaN comparisons are always false (`span` becomes non-positive and **every
 * pixel 0 = a uniformly black depth map** comes out looking like a normal value). The same check
 * point as birefnet's `matteFromLogits`, sbv2's waveform check and `anima/image.ts`.
 *
 * NOTE: the `export` is for tests (the same reason as {@link preprocessPixelValues}).
 */
export const resampleDepth = (
  depth: Float32Array<ArrayBuffer>,
  sourceWidth: number,
  sourceHeight: number,
  width: number,
  height: number,
): DepthMap => {
  for (let index = 0; index < depth.length; index += 1) {
    if (!Number.isFinite(depth[index])) {
      const x = index % sourceWidth;
      const y = Math.floor(index / sourceWidth);
      throw new Error(`depth-anything: depth value (x=${x}, y=${y}) is non-finite`);
    }
  }
  return {
    data: resizePlaneF32(depth, sourceWidth, sourceHeight, width, height),
    width,
    height,
  };
};

/** Internal state of {@link Pipeline} (not on the public surface). */
type DepthAnythingState = {
  readonly gpu: GpuContext;
  readonly ownsGpu: boolean;
  readonly config: DepthAnythingPipelineConfig;
  readonly depth: ModelComponent;
  /** The one Session created at construction and kept until `dispose` (the module doc's MUST). */
  readonly session: Session;
  readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics) => void;
};

/**
 * The execution knobs depth-anything accepts (the manifest's quant declaration). The merge rule
 * (explicit > declared > runtime default), value ranges, combinations and the classification of the
 * thrown types are owned by `resolveSessionOptions`, shared by every family; this only decides the
 * accepted set. There is no explicit override (the construction options have no execution-knob
 * field).
 *
 * No key is accepted because the distribution has only the f32 default path and no quant that
 * declares execution knobs (when a declaration appears, check it against the reference values
 * before accepting it — do not silently pass it to the runtime).
 *
 * NOTE: the `export` is for the equivalence test that runs every quant of the mirror through the
 * same table (not exposed on `mod.ts` / the subpath surface — ADR 0008).
 */
export const DEPTH_ANYTHING_SESSION_POLICY: FamilySessionPolicy = {
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
};

/** What the family admission ({@link admitDepthAnything}) settles. */
type DepthAnythingAdmission = {
  readonly config: DepthAnythingPipelineConfig;
  readonly quantName: string;
  readonly quant: Quant;
  /**
   * The effective settings after passing the quant declaration through the acceptance table
   * ({@link DEPTH_ANYTHING_SESSION_POLICY}).
   */
  readonly sessionOptions: SessionOptions;
  /**
   * The quant declaration plus the features the effective settings need (both the request and the
   * check read this).
   */
  readonly gpuFeatures: GpuFeaturesSpec | undefined;
};

/**
 * Checks whether this manifest and this graph can run as depth-anything (the family admission hook
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
const admitDepthAnything = (
  manifest: Manifest,
  open: ComponentOpener,
  options: PipelineOptions,
): DepthAnythingAdmission => {
  const modelName = options.model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, modelName)) {
    throw new Error(
      `depthAnything.Pipeline: model '${modelName}' is not in the manifest` +
        ` (available: ${manifest.available.models.join(" / ")})`,
    );
  }
  const entry: ModelEntry = manifest.models[modelName];
  const { name, major } = entry.pipeline;
  if (name !== DEPTH_ANYTHING_PIPELINE_NAME) {
    throw new Error(
      `depthAnything.Pipeline: the manifest's pipeline is '${name}/${major}'` +
        ` ('${DEPTH_ANYTHING_PIPELINE_NAME}/${DEPTH_ANYTHING_PIPELINE_MAJOR}' is required)`,
    );
  }
  if (major !== DEPTH_ANYTHING_PIPELINE_MAJOR) {
    // The only check that stops silent degradation from "old implementation × new repository"
    // (ADR 0038 §6).
    throw new Error(
      `depthAnything.Pipeline: the major of pipeline '${name}/${major}' is not supported` +
        ` (this implementation reads ${DEPTH_ANYTHING_PIPELINE_NAME}/` +
        `${DEPTH_ANYTHING_PIPELINE_MAJOR})`,
    );
  }
  const config = parseDepthAnythingPipelineConfig(entry.pipelineConfig);

  const quantName = options.quant ?? entry.defaultQuant;
  if (!Object.hasOwn(entry.quants, quantName)) {
    throw new Error(
      `depthAnything.Pipeline: quant '${quantName}' is not in the manifest` +
        ` (available: ${entry.available.quants.join(" / ")})`,
    );
  }
  const quant = entry.quants[quantName];
  // Unsupported declarations fail before components are opened (= before weight parts are fetched;
  // one path shared by every family).
  const sessionOptions = resolveSessionOptions(
    DEPTH_ANYTHING_SESSION_POLICY,
    quant.session,
    {},
    `depthAnything.Pipeline: quant '${quantName}'`,
  );
  const gpuFeatures = sessionGpuFeatures(quant.gpuFeatures, sessionOptions);

  const depth = open(DEPTH);
  // Match the graph declaration against pipelineConfig. Checking that there is exactly one input
  // and one output matters because, if another export that also emits intermediate-stage depth got
  // mixed in, the later steps that pick by position would silently read other values as depth.
  if (depth.graph.inputs.length !== 1 || depth.graph.outputs.length !== 1) {
    throw new Error(
      `depthAnything.Pipeline: the depth graph has ${depth.graph.inputs.length} /` +
        ` ${depth.graph.outputs.length} inputs / outputs (one depth map requires exactly 1 of each)`,
    );
  }
  assertGraphInputDtype(
    "depth-anything",
    depth,
    PIXEL_VALUES,
    "f32",
    "preprocessing's pixel_values",
  );
  assertStaticDim(depth, PIXEL_VALUES, 0, 1, "batch is static 1");
  assertStaticDim(depth, PIXEL_VALUES, 1, CHANNELS, "3 RGB channels");
  assertStaticDim(depth, PIXEL_VALUES, 2, config.imageHeight, "imageHeight");
  assertStaticDim(depth, PIXEL_VALUES, 3, config.imageWidth, "imageWidth");
  assertDepthShape(depth, config, "depth map shape");

  // MUST: a shared GPU's missing capability (feature / device limit) fails here — unlike acquiring
  // our own, there is no reason to wait for `acquireGpu`, and this is the only family check that
  // knows before the weights are downloaded (the mapping between request and check is the one
  // function in `session/gpu-features.ts`, and the later check calls the same function).
  if (options.gpu !== undefined) {
    assertGpuFeaturesGranted(
      gpuFeatures,
      options.gpu,
      `depthAnything.Pipeline: quant '${quantName}'`,
    );
    assertRequiredLimitsSatisfied(
      quant.requiredLimits,
      options.gpu.limits,
      `depthAnything.Pipeline: quant '${quantName}'`,
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
 * features; limits are checked by {@link admitDepthAnything} for a shared GPU, and for a device
 * acquired here by `assertRequiredLimits` right after acquisition.
 */
const openDepthAnythingState = async (
  admitted: DepthAnythingAdmission,
  open: ComponentOpener,
  options: PipelineOptions = {},
): Promise<DepthAnythingState> => {
  const { config, quantName, quant, sessionOptions, gpuFeatures } = admitted;
  // The supplier is **the same one** admission saw (`open` only looks up components already
  // opened).
  const depth = open(DEPTH);
  // MUST: declared features can only be requested at device creation (ADR 0028). A shared GPU
  // cannot be asked for them, so the missing capability is named and fails (a shared GPU was
  // already checked by {@link admitDepthAnything} through the same function, but for a device
  // acquired here this is the only check).
  const gpu = options.gpu ?? (await acquireGpu(toAcquireGpuOptions(gpuFeatures)));
  const ownsGpu = options.gpu === undefined;
  try {
    assertGpuFeaturesGranted(gpuFeatures, gpu, `depthAnything.Pipeline: quant '${quantName}'`);
    // Also check the limits of a device acquired here — `fromAssets` has no pre-fetch check
    // (`fromPretrained` checks against adapter values); without this a missing declaration surfaces
    // only at Session construction, and a declaration stricter than the real demand is never
    // matched at all. A shared GPU is checked by `admitDepthAnything`.
    if (ownsGpu) {
      assertRequiredLimitsSatisfied(
        quant.requiredLimits,
        gpu.limits,
        `depthAnything.Pipeline: quant '${quantName}'`,
      );
    }
    return {
      gpu,
      ownsGpu,
      config,
      depth,
      session: await depth.createSession(gpu, sessionOptions),
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

/** Runs one image (preprocessing → one `depth` run → back to the original resolution). */
const estimateImage = async (state: DepthAnythingState, image: Rgb8Image): Promise<DepthMap> => {
  const { config } = state;
  const pixelValues: Tensor = {
    dtype: "f32",
    shape: [1, CHANNELS, config.imageHeight, config.imageWidth],
    data: preprocessPixelValues(config, image),
  };
  const outputs = await state.session.run({ [PIXEL_VALUES]: pixelValues });
  if (state.onRunDiagnostics !== undefined) state.onRunDiagnostics(state.session.diagnostics());
  const name = state.depth.graph.outputs[0];
  const tensor = outputs[name];
  if (tensor === undefined) {
    throw new Error(`depth-anything: graph output '${name}' is not in the run result`);
  }
  const depth = asF32(tensor, "depth");
  const expected = config.imageWidth * config.imageHeight;
  if (depth.length !== expected) {
    throw new Error(
      `depth-anything: depth map element count ${depth.length} does not match ` +
        `${config.imageWidth}×${config.imageHeight}`,
    );
  }
  return resampleDepth(depth, config.imageWidth, config.imageHeight, image.width, image.height);
};

/**
 * Depth Anything V2's image → relative depth map pipeline (one graph).
 *
 * Construction enters only through {@link Pipeline.fromPretrained} (fetch from HF) or
 * {@link Pipeline.fromAssets} (already-fetched bytes) — the constructor is private so
 * that no half-built state bypassing the manifest checks and the asset matching can be created
 * (ADR 0008).
 */
export class Pipeline {
  readonly #state: DepthAnythingState;
  /**
   * Serialization chain for estimate and dispose ("one at a time" kept on the public API side —
   * module doc).
   */
  readonly #chain = createOperationChain();
  /**
   * The one dispose. **Not being undefined means "disposed"** (a separate boolean would be derived
   * state updated independently).
   */
  #disposal: Promise<void> | undefined;

  private constructor(state: DepthAnythingState) {
    this.#state = state;
  }

  /**
   * Fetches a distribution and builds (`loadManifest` → `resolveSelection` → fetch **only each
   * component's descriptor (part 0)** for admission → warm the weight parts → `fetchAssets` for the
   * remaining assets → construction). Blocks are read in part order at the moment the Session is
   * built (ADR 0109 — `src/hub/components.ts`). To replace a component with the same role from
   * another repository use {@link FromPretrainedComponentOptions.components} (one role, `depth`).
   *
   * **`ref` is required** (sources have no default — the MUST of `src/hub/repo-ref.ts`). The source
   * this package version verified is `"depth-anything-v2"` of {@link SOURCES}
   * (`./config.ts`) — to pin reproducibility yourself, the caller states `{ repo, revision }`
   * explicitly. A string `ref` reads as `{ repo }` (= follows `main`).
   *
   * A local distribution is passed as a **source handle** (`localDirectory` / `denoDirectory` of
   * `@karume/hub/deno`). It skips the check of HF's `owner/name` spelling and touches neither the
   * network nor CacheStorage (the HTTP-only knobs of {@link FromPretrainedOptions}
   * have no effect).
   */
  static async fromPretrained(
    ref: string | HubRepoRef | DistributionSource,
    options: FromPretrainedOptions = {},
  ): Promise<Pipeline> {
    const source = toManifestSource(
      ref,
      "depthAnything.Pipeline.fromPretrained",
      'SOURCES["depth-anything-v2"] (@karume/models/depth-anything)',
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
      "depthAnything.Pipeline.fromPretrained",
      loaded,
      selection,
      COMPONENT_KEYS,
      async (open) => {
        const admitted = admitDepthAnything(loaded.manifest, open, buildOptions);
        // The `requiredLimits` the distribution declared are checked here, **before weight parts
        // are fetched** (ADR 0089 decision 5 — a shared GPU's limits, or the adapter's measured
        // values on the path that acquires its own).
        await assertRequiredLimitsBeforeDownload(
          admitted.quant.requiredLimits,
          buildOptions.gpu,
          `depthAnything.Pipeline: quant '${admitted.quantName}'`,
        );
        return admitted;
      },
      {
        ...hubOptions,
        ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
        ...(options.components === undefined ? {} : { components: options.components }),
      },
    );
    return new Pipeline(await openDepthAnythingState(admitted, open, buildOptions));
  }

  /**
   * Builds from an already-fetched manifest + assets. Contract checks, asset interpretation,
   * `openContainer` and the match against the graph all happen here, and it returns **with the one
   * `depth` Session created** (the module doc's MUST).
   *
   * Component keys are accepted **in both forms** — one single-form `krm` byte array (`depth`) and
   * the part sequence of the split form (`depth[0]` / `depth[1]` / … — in index order from part 0).
   * The split form is opened as a part sequence without concatenating the bytes. A missing index or
   * mixing with the single-form key fails loudly (the receiving implementation is the one in
   * `src/hub/components.ts`).
   */
  static async fromAssets(input: Assets, options: PipelineOptions = {}): Promise<Pipeline> {
    const open = await assetOpener(input.assets);
    const admitted = admitDepthAnything(input.manifest, open, options);
    return new Pipeline(await openDepthAnythingState(admitted, open, options));
  }

  /**
   * Estimates the relative depth of one image. It returns raw f32 **of the same size as the input**
   * (the module doc's "It returns raw relative depth in f32") — larger = nearer, with neither a
   * unit nor an origin.
   *
   * The input is an RGB8 pixel row (row-major, 3 bytes per pixel); resize / rescale / normalize are
   * applied inside with the distribution's constants. The aspect ratio is not kept (the graph
   * accepts one square size only — the NOTE in `config.ts`).
   *
   * Concurrent calls **wait and run in order** (one Session is not shared between two runs).
   */
  async estimate(image: Rgb8Image): Promise<DepthMap> {
    // The disposed check happens at call time (checking inside the chain would also fail the
    // estimations accepted before dispose).
    if (this.#disposal !== undefined) {
      throw new Error("depthAnything.Pipeline: cannot estimate after dispose");
    }
    return await this.#chain(() => estimateImage(this.#state, image));
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
