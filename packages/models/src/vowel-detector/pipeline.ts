/**
 * `Pipeline` — the one surface for 16kHz mono audio → a vowel sequence for lip sync
 * (`.lab`).
 *
 * Pipeline (every step is Karume, no torch):
 *
 * 1. host: waveform → 80-dim log-mel + 3 DSP dims = `[T10, 83]` (`features.ts`; the mel basis is
 *    received from the distribution's `assets` field)
 * 2. host: drop the one leftover frame of an odd frame count (the output is a 20ms grid)
 * 3. run `crnn` once → logits `[1, T10/2, 8]` (20ms grid)
 * 4. host: log_softmax → Viterbi → short-segment merge → cons absorption → `.lab`
 *    (`postprocess.ts`)
 *
 * **Resampling is the caller's responsibility** (drawing the line exactly where upstream
 * `@hdae/vowel-detector` draws it). The entry is "a 16kHz mono `Float32Array`", with neither WAV
 * decoding nor rate conversion (`decodeWav` is exported separately from the barrel). Whether the
 * given array really is 16kHz **cannot be observed** — passing a different rate makes `detect` run
 * to the end and emit a different vowel sequence, so that is an invariant on the caller's side.
 *
 * ## The length is a symbolic axis (one graph, no pad, no per-length variants)
 *
 * The graph input is `features [1, 2T, 83]` and the output `[1, T, 8]`, and **only the time axis is
 * symbolic**, `T` (20ms grid). `detect` runs once at the audio's actual length and passes `T`
 * bound. Passing the symbolic length by replacing `nn.GRU` with `gru_scan` is ADR 0056, and
 * binding a symbol from a derived dimension like `2T` is ADR 0057 (the history of the earlier
 * per-length graphs + right zero pad is in the same ADR).
 *
 * MUST: **the 10ms frame count is even** (the graph declares `2T`). The one leftover frame of an
 * odd frame count is **truncated** — a half frame would become an output that carries only one
 * input frame, and post-processing's "1 frame = 20ms" would break only at the end. Forgetting the
 * truncation makes `bindSymbols` fail because the measured length (e.g. 285) does not have the
 * form of the declared `2T` (no silent rounding).
 *
 * MUST: **below `minFrames` and above `maxFrames` both fail loudly** (no silent truncation, no
 * silent pad). Truncation silently discards the end of the audio = the `.lab` only becomes shorter
 * than the input, which looks like a normal result to the caller. Letting through the lower side
 * would run with the symbol `T` bound outside the traced range (the graph is baked with
 * `Dim("T", min=…, max=…)`). Both sides are numbers held by the distribution's `pipelineConfig`,
 * in units of 10ms frames (the IR carries no symbol ranges, so only here can they be stopped —
 * `config.ts`).
 *
 * ### Why there is no right zero pad (measured — the debt the form before ADR 0057 carried)
 *
 * The backward GRU carries state back from the pad side, so the pad moves the logits by O(1). **It
 * is neither monotonic nor proportional in the pad amount, and saturates already at 2 frames
 * (40ms)** (measured on 4 real recordings × 10 pad sizes: the max abs diff over the body segment
 * only is 2.2–3.2 at pad 2 and 5.0–6.0 at pad 1024). The amplitude concentrates at the end, but
 * `|diff| > 1e-2` reaches every frame, and since Viterbi is globally optimal, `.lab` differences
 * appeared anywhere in the utterance (the measured differences were of 3 types: (1) a 20ms
 * boundary shift, (2) a 40ms `pau` added or removed at the end, (3) a 40ms `pau` inserted
 * mid-utterance). Now that the length is symbolic, these 3 types **are gone**: the `.lab` through
 * the distribution matches exactly the `.lab` of the actual-length path pinned by the real-weight
 * E2E (`packages/e2e/tests/e2e_vowel_detector_chain_test.ts`).
 *
 * ## The Session is created and torn down per `detect`
 *
 * Now that there is one graph, the same residency as SigLIP2 / BiRefNet is possible, but this phase
 * keeps **create and tear down** (a form that confines VRAM to the duration of detection — the
 * same as Irodori creating and tearing down its huge graphs one at a time; `withSession` in
 * `src/session/with-session.ts`). The weights are 2.66MB and the measured re-upload is 30–144ms, so
 * making it resident is **a performance decision**, not a correctness decision.
 *
 * MUST: it still rides the serialization chain — concurrent `detect` calls would create the form
 * where `dispose` destroys the GPU without waiting for in-flight completion (a flush-before-destroy
 * violation).
 *
 * ## MUST: numerical correctness is not guaranteed here
 *
 * Correctness is guaranteed by the real-weight E2E (logit comparison of 4 synthetic golden cases ×
 * 4 series = `e2e_vowel_detector_test.ts` (removed in the 0.15 test triage — the chain's sha
 * reference row takes it over); exact `.lab` match of the whole chain on 4 real recordings =
 * `packages/e2e/tests/e2e_vowel_detector_chain_test.ts`) and by the host layer's parity check
 * (`packages/models/tests/vowel_detector_host_test.ts` — comparison with the canonical Python).
 */

import {
  acquireGpu,
  type GpuContext,
  parseSafetensors,
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
  parseVowelDetectorPipelineConfig,
  VOWEL_DETECTOR_PIPELINE_MAJOR,
  VOWEL_DETECTOR_PIPELINE_NAME,
  type VowelDetectorPipelineConfig,
} from "./config.ts";
import {
  extractFeatures,
  featureFrameCount,
  HOP,
  MEL_BINS,
  N_MELS,
  SAMPLE_RATE,
} from "./features.ts";
import { type LabSegment, logitsToSegments, toLab } from "./postprocess.ts";
import { ModelInputError } from "../errors.ts";
import { createOperationChain } from "../concurrency/serial.ts";
import {
  assertGpuFeaturesGranted,
  assertRequiredLimitsBeforeDownload,
  assertRequiredLimitsSatisfied,
  sessionGpuFeatures,
  toAcquireGpuOptions,
} from "../session/gpu-features.ts";
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
import { withSession } from "../session/with-session.ts";

/**
 * The fetch key that appears in the manifest's assets table, and the tensor name in that
 * safetensors (paired with `tools/export-recipes/vowel_detector/distribution.py`).
 */
const MEL_BASIS = "mel_basis";

/** Name of the graph input (`INPUT_NAME` in `tools/export-recipes/vowel_detector/export.py`). */
const INPUT_NAME = "features";

/** Step of the output time axis (2 input frames = 1 output frame — the conv's stride 2). */
const TIME_STRIDE = 2;

/**
 * Input frames per second (hop 160 / 16kHz = 10ms grid) — used only for messages that speak in
 * seconds.
 */
const FRAMES_PER_SECOND = SAMPLE_RATE / HOP;

/**
 * The fetch key that appears in the manifest's weights table (one CRNN —
 * `VOWEL_DETECTOR_GRAPH_ROLE` in `tools/export-recipes/vowel_detector/distribution.py`).
 */
const GRAPH_ROLE = "crnn";

/** This family's weights components (also the role names the component replacement accepts). */
const COMPONENT_KEYS = [GRAPH_ROLE] as const;

/**
 * The result of {@link Pipeline.detect} (upstream `@hdae/vowel-detector`'s
 * `DetectResult`).
 */
export type Result = {
  /**
   * Segment sequence in seconds (a/i/u/e/o/N/pau — cons is already absorbed by post-processing).
   */
  readonly segments: readonly LabSegment[];
  /**
   * The body in `.lab` format (lines of `start end label`, seconds to 7 digits, with a trailing
   * newline).
   */
  readonly lab: string;
};

/** Construction options (shared by {@link Pipeline.fromAssets} / `fromPretrained`). */
export type PipelineOptions = {
  /**
   * Shares an existing GPU. **The passing side owns it**, so
   * {@link Pipeline.dispose} does not destroy it. When omitted, the pipeline calls
   * `acquireGpu` internally and destroys it in `dispose()`.
   */
  readonly gpu?: GpuContext;
  /** Model (a key of the manifest's models). `defaultModel` when omitted. */
  readonly model?: string;
  /** Execution configuration (a key of that model's quants). `defaultQuant` when omitted. */
  readonly quant?: string;
  /**
   * Observer hook that receives the diagnostics of each run (1 detect = 1 CRNN run). When per-op
   * GPU time (`lastRunTiming`) is needed, pass `acquireGpu({ gpuTiming: true })` as `gpu` (ADR
   * 0021 — no timing by default).
   *
   * NOTE: like SigLIP2 / BiRefNet, no component name is passed — the graph that runs is always the
   * one CRNN, and only the length differs (not an axis the receiver can branch on).
   *
   * Exceptions from the callback are not caught (fail loudly — the whole detection fails).
   */
  readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics) => void;
};

/**
 * Fetch-layer options used only by {@link Pipeline.fromPretrained} (passed through to
 * hub).
 *
 * NOTE: the description of the fetch-layer knobs, including that `headers` / `fetch` / `caches` /
 * `onRetry` are **for the HTTP source only**, is consolidated in {@link FromPretrainedHubOptions}.
 */
export type FromPretrainedOptions = PipelineOptions &
  FromPretrainedHubOptions &
  FromPretrainedComponentOptions & {
    readonly signal?: AbortSignal;
  };

/**
 * Input for building directly from fetched assets (pass the return value of hub's `fetchAssets` as
 * is).
 */
export type Assets = {
  readonly manifest: Manifest;
  readonly assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>;
};

/**
 * Turns fetched bytes into an ArrayBuffer that can be passed to `openContainer` /
 * `parseSafetensors` (the check itself is {@link readAssetBuffer}).
 */
const assetBuffer = (
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  key: string,
): ArrayBuffer => readAssetBuffer("vowel-detector", "weights / assets", assets, key);

/**
 * Component supplier of the in-memory surface (`fromAssets`) (the receiving implementation is
 * shared by the families — {@link assetComponentOpener}). A component key is either the single
 * form `krm` as one entry (`crnn`), or the part sequence of the split form (`crnn[0]` / `crnn[1]`
 * / …).
 */
const assetOpener = (assets: Assets["assets"]): Promise<ComponentOpener> =>
  assetComponentOpener("vowel-detector", assets, (key) => assetBuffer(assets, key), COMPONENT_KEYS);

/**
 * Reads the mel-basis asset (a one-tensor f32 safetensors `[80, 257]`).
 *
 * Assets are an external boundary, so the tensor name, dtype and shape are all checked before use.
 * **A transposed matrix has the same element count** (80×257 and 257×80 differ, but a different
 * basis with a mistaken `n_mels` could match in element count alone), so both axes of the shape
 * are checked.
 *
 * NOTE: `export` for tests (reaching here through `fromAssets` needs a full manifest). Not exposed
 * on `mod.ts` / the subpath surface (ADR 0008).
 */
export const parseMelBasis = (buffer: ArrayBuffer): Float32Array<ArrayBuffer> => {
  const file = parseSafetensors(buffer);
  const view = file.tensors.get(MEL_BASIS);
  if (view === undefined) {
    throw new Error(
      `vowel-detector: the mel basis asset has no tensor '${MEL_BASIS}'` +
        ` (it contains: ${[...file.tensors.keys()].join(" / ")})`,
    );
  }
  if (view.dtype !== "F32") {
    throw new Error(`vowel-detector: the mel basis is not F32 (${view.dtype})`);
  }
  if (view.shape.length !== 2 || view.shape[0] !== N_MELS || view.shape[1] !== MEL_BINS) {
    throw new Error(
      `vowel-detector: the mel basis shape is [${view.shape.join(", ")}], expected [${N_MELS}, ${MEL_BINS}]`,
    );
  }
  return new Float32Array(file.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength));
};

/**
 * Checks that the graph is baked with a **symbolic length** and fits `pipelineConfig`.
 * Returns the symbol name of the time axis (the one word `detect` uses for binding).
 *
 * MUST: do not drop it. An old-form graph baked at a fixed length has **the same input/output names
 * and ranks**, so placed in the same slot it passes construction, and at run time only "the shape
 * does not match the declaration" appears (by then it does not reach the reader that "this
 * distribution runs only at one length").
 * MUST: check down to the input's time axis being `2·T` and the output's `T`. A distribution
 * missing the factor would only stretch the `.lab` times by 2 while the shapes still hold.
 *
 * NOTE: `export` for tests that pin the rejection paths without a GPU (not exposed on `mod.ts` /
 * the subpath surface — ADR 0008).
 */
export const assertGraph = (model: GraphOwner, config: VowelDetectorPipelineConfig): string => {
  if (model.graph.inputs.length !== 1 || model.graph.outputs.length !== 1) {
    throw new Error(
      `vowelDetector.Pipeline: the graph has ${model.graph.inputs.length} inputs /` +
        ` ${model.graph.outputs.length} outputs (a CRNN with one of each is required)`,
    );
  }
  if (model.graph.symbols.length !== 1) {
    throw new Error(
      `vowelDetector.Pipeline: the graph's symbolic dimensions are [${model.graph.symbols.join(", ")}]` +
        " (only the one time axis may be symbolic)",
    );
  }
  const [symbol] = model.graph.symbols;
  const [input] = model.graph.inputs;
  if (input.name !== INPUT_NAME) {
    throw new Error(
      `vowelDetector.Pipeline: the graph input is '${input.name}' ('${INPUT_NAME}' is required)`,
    );
  }
  // The canonical notation is `coeff·sym` (a coefficient of 1 is omitted — `formatDim` in
  // `format/dims.ts`).
  const expectedInput = [1, `${TIME_STRIDE}${symbol}`, config.featureDim];
  if (
    input.shape.length !== expectedInput.length ||
    input.shape.some((dim, axis) => dim !== expectedInput[axis])
  ) {
    throw new Error(
      `vowelDetector.Pipeline: the graph input shape is ` +
        `[${input.shape.map(String).join(", ")}], expected [${expectedInput.join(", ")}]`,
    );
  }
  const name = model.graph.outputs[0];
  const value = model.graph.values[name];
  if (value === undefined) {
    throw new Error(`vowelDetector.Pipeline: graph output '${name}' is not declared`);
  }
  const expectedOutput = [1, symbol, config.classes.length];
  if (
    value.shape.length !== expectedOutput.length ||
    value.shape.some((dim, axis) => dim !== expectedOutput[axis])
  ) {
    throw new Error(
      `vowelDetector.Pipeline: the graph output shape is ` +
        `[${value.shape.map(String).join(", ")}], expected [${expectedOutput.join(", ")}]`,
    );
  }
  return symbol;
};

/**
 * Converts a 10ms frame count to seconds (shared by the messages of both checks — 2 decimal
 * places).
 */
const seconds = (frames: number): string => (frames / FRAMES_PER_SECOND).toFixed(2);

/**
 * Checks that the input length (10ms frame count) is within the operating upper limit the
 * distribution declares.
 *
 * MUST: fail when the limit is exceeded (the module doc's MUST — no silent truncation).
 *
 * MUST: compare on **the 10ms side** (`config.maxFrames` is the upper bound of the symbolic
 * dimension `T` converted to input-side units, twice `Dim(max=…)` itself). Rewriting it to compare
 * against the count on the 20ms side (`usable / TIME_STRIDE`) would silently double the effective
 * limit. The IR carries no symbol ranges, so **this is the only check**.
 *
 * NOTE: `export` for tests (so the boundary behaviour can be named without a real GPU).
 */
export const assertFrameLimit = (config: VowelDetectorPipelineConfig, frames: number): void => {
  if (frames <= config.maxFrames) return;
  // The length comes from the caller's waveform, so it is input-caused (ADR 0107 decision 2 —
  // passing it in pieces works).
  throw new ModelInputError(
    `vowelDetector.Pipeline: the audio is too long (${frames} frames of 10ms = ` +
      `${seconds(frames)} s) — the maximum this distribution was built with is ` +
      `${config.maxFrames} frames (${seconds(config.maxFrames)} s). ` +
      "Truncating would silently drop the end, so split the audio on the caller side",
  );
};

/**
 * Checks that the input length (10ms frame count) reaches the operating **lower limit** the
 * distribution declares (the counterpart of {@link assertFrameLimit}).
 *
 * MUST: fail below the lower limit (the module doc's MUST — no silent pad). The graph is baked with
 * `Dim("T", min=…)`, so an input below the lower limit would run with the symbol bound outside the
 * traced range (`bindSymbols` only checks `bound >= 0`, so the runtime does not stop it either). As
 * with the upper limit, the comparison is on **the 10ms side**.
 *
 * NOTE: `export` for tests (so the boundary behaviour can be named without a real GPU).
 */
export const assertFrameFloor = (config: VowelDetectorPipelineConfig, frames: number): void => {
  if (frames >= config.minFrames) return;
  // Like the upper side, a dimension of the caller's waveform (ADR 0107 decision 2).
  throw new ModelInputError(
    `vowelDetector.Pipeline: the audio is too short (${frames} frames of 10ms = ` +
      `${seconds(frames)} s) — the minimum length this distribution was built with is ` +
      `${config.minFrames} frames (${seconds(config.minFrames)} s). ` +
      "Filling the shortfall with zero padding on the right makes the backward GRU carry state back from the padding, " +
      "so pass audio of sufficient length from the caller side",
  );
};

/** The internal state of {@link Pipeline} (not exposed on the public surface). */
type VowelDetectorState = {
  readonly gpu: GpuContext;
  readonly ownsGpu: boolean;
  readonly config: VowelDetectorPipelineConfig;
  /** The opened CRNN graph (no Session is held here — the module doc). */
  readonly graph: ModelComponent;
  /** Symbol name of the time axis (read from the graph by `assertGraph` — the binding key). */
  readonly symbol: string;
  readonly sessionOptions: SessionOptions;
  readonly melBasis: Float32Array<ArrayBuffer>;
  readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics) => void;
};

/**
 * The execution knobs vowel-detector accepts (the manifest's quant declaration). The merge rule
 * (explicit > declared > runtime default), the value ranges, the combinations and the
 * classification of the thrown type are held by `resolveSessionOptions`, shared by every family;
 * this only decides the acceptance set. There is no explicit override (the construction options
 * have no field for execution knobs).
 *
 * No key is accepted because there is no quant that declares execution knobs (the distribution is
 * unpublished and the export is only the f32 default path) (when a declaration appears, confirm it
 * against the reference values before accepting it — do not pass it silently to the runtime).
 *
 * NOTE: `export` so the equivalence test runs every quant of the mirror through the same table (not
 * exposed on `mod.ts` / the subpath surface — ADR 0008).
 */
export const VOWEL_DETECTOR_SESSION_POLICY: FamilySessionPolicy = {
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

/** Material settled by the family admission ({@link admitVowelDetector}). */
type VowelDetectorAdmission = {
  readonly config: VowelDetectorPipelineConfig;
  readonly quantName: string;
  readonly quant: Quant;
  /**
   * The effective settings of the quant declaration passed through the acceptance table
   * ({@link VOWEL_DETECTOR_SESSION_POLICY}).
   */
  readonly sessionOptions: SessionOptions;
  /**
   * The quant declaration plus the features the effective settings need (both the request and the
   * check look at this).
   */
  readonly gpuFeatures: GpuFeaturesSpec | undefined;
  /** Symbol name of the time axis (read from the graph by `assertGraph`). */
  readonly symbol: string;
};

/**
 * Checks whether this manifest and this graph can run as vowel-detector (the family admission hook
 * of `hub/components.ts` — on the fetching surface it is called **before a single byte of the
 * weight parts is fetched**).
 *
 * MUST: gather the family's checks into this one function. Scattered into later steps, on the
 * fetching surface they would fail only **after** downloading GB-class weights (the implementation
 * would be narrower than the wording of ADR 0070 decision 5).
 * MUST: fail on manifest contract violations and **the comparison with the graph** **before going
 * to get the GPU**. If the order slips, in an environment without a GPU it turns into a different
 * exception and "what was wrong" does not reach the reader.
 *
 * NOTE: parsing the asset (`mel_basis`) cannot be placed in this hook — at admission time the bytes
 * have not been fetched yet (fetching them first would lose the position before the weight
 * prefetch).
 */
const admitVowelDetector = (
  manifest: Manifest,
  open: ComponentOpener,
  options: PipelineOptions,
): VowelDetectorAdmission => {
  const modelName = options.model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, modelName)) {
    throw new Error(
      `vowelDetector.Pipeline: model '${modelName}' is not in the manifest` +
        ` (available: ${manifest.available.models.join(" / ")})`,
    );
  }
  const entry: ModelEntry = manifest.models[modelName];
  const { name, major } = entry.pipeline;
  if (name !== VOWEL_DETECTOR_PIPELINE_NAME) {
    throw new Error(
      `vowelDetector.Pipeline: the manifest pipeline is '${name}/${major}'` +
        ` ('${VOWEL_DETECTOR_PIPELINE_NAME}/${VOWEL_DETECTOR_PIPELINE_MAJOR}' is required)`,
    );
  }
  if (major !== VOWEL_DETECTOR_PIPELINE_MAJOR) {
    // The only check that stops the silent degradation "old implementation × new repository" (ADR
    // 0038 §6).
    throw new Error(
      `vowelDetector.Pipeline: pipeline '${name}/${major}' has an unsupported major` +
        ` (this implementation reads ` +
        `${VOWEL_DETECTOR_PIPELINE_NAME}/${VOWEL_DETECTOR_PIPELINE_MAJOR})`,
    );
  }
  const config = parseVowelDetectorPipelineConfig(entry.pipelineConfig);

  const quantName = options.quant ?? entry.defaultQuant;
  if (!Object.hasOwn(entry.quants, quantName)) {
    throw new Error(
      `vowelDetector.Pipeline: quant '${quantName}' is not in the manifest` +
        ` (available: ${entry.available.quants.join(" / ")})`,
    );
  }
  const quant = entry.quants[quantName];
  // Unsupported declarations fail before opening the components (= before fetching the weight
  // parts) (the one function shared by every family).
  const sessionOptions = resolveSessionOptions(
    VOWEL_DETECTOR_SESSION_POLICY,
    quant.session,
    {},
    `vowelDetector.Pipeline: quant '${quantName}'`,
  );
  const gpuFeatures = sessionGpuFeatures(quant.gpuFeatures, sessionOptions);

  const graph = open(GRAPH_ROLE);
  const symbol = assertGraph(graph, config);

  // MUST: fail here on missing capabilities of a shared GPU (feature / device limit) — unlike the
  // case where the pipeline acquires its own, there is no reason to wait for `acquireGpu`, and
  // this is the only family check that knows before downloading the weights (the mapping between
  // request and check is the one in `session/gpu-features.ts`, and the later check calls the same
  // function).
  if (options.gpu !== undefined) {
    assertGpuFeaturesGranted(
      gpuFeatures,
      options.gpu,
      `vowelDetector.Pipeline: quant '${quantName}'`,
    );
    assertRequiredLimitsSatisfied(
      quant.requiredLimits,
      options.gpu.limits,
      `vowelDetector.Pipeline: quant '${quantName}'`,
    );
  }

  return { config, quantName, quant, symbol, sessionOptions, gpuFeatures };
};

/**
 * Builds the run state from the material that passed admission + the assets.
 *
 * MUST: fail on parsing the assets **before going to get the GPU**. If the order slips, in an
 * environment without a GPU it turns into a different exception and "what was wrong" does not reach
 * the reader. The only check allowed after acquiring the GPU is the GPU's capability (shader-f16)
 * (ADR 0028).
 */
const openVowelDetectorState = async (
  admitted: VowelDetectorAdmission,
  assets: Assets["assets"],
  open: ComponentOpener,
  options: PipelineOptions = {},
): Promise<VowelDetectorState> => {
  const { config, quantName, quant, symbol, sessionOptions, gpuFeatures } = admitted;
  // The supplier is **the same one** admission looked at (`open` only looks up the opened
  // component).
  const graph = open(GRAPH_ROLE);
  const melBasis = parseMelBasis(assetBuffer(assets, MEL_BASIS));

  // MUST: declared features can be requested only at device creation (ADR 0028). With a shared GPU
  // they cannot be requested, so fail naming the missing capability (a shared GPU was already
  // checked by {@link admitVowelDetector} with the same function, but for a device acquired here
  // this is the only check).
  const gpu = options.gpu ?? (await acquireGpu(toAcquireGpuOptions(gpuFeatures)));
  const ownsGpu = options.gpu === undefined;
  try {
    assertGpuFeaturesGranted(gpuFeatures, gpu, `vowelDetector.Pipeline: quant '${quantName}'`);
    // Also check the limits of a device acquired here — `fromAssets` has no pre-fetch check
    // (`fromPretrained` checks with the adapter's values), and without this a shortfall of the
    // declaration would be delayed until Session construction, and when the declaration is stricter
    // than the real demand the comparison would never happen at all. A shared GPU is checked by
    // `admitVowelDetector`.
    if (ownsGpu) {
      assertRequiredLimitsSatisfied(
        quant.requiredLimits,
        gpu.limits,
        `vowelDetector.Pipeline: quant '${quantName}'`,
      );
    }
  } catch (error) {
    // A GPU acquired internally can be released by no one if construction fails, so give it back
    // here.
    if (ownsGpu) gpu.destroy();
    throw error;
  }
  return {
    gpu,
    ownsGpu,
    config,
    graph,
    symbol,
    sessionOptions,
    melBasis,
    ...(options.onRunDiagnostics === undefined
      ? {}
      : { onRunDiagnostics: options.onRunDiagnostics }),
  };
};

/**
 * Creates the Session for one detection and always releases it when done.
 *
 * MUST: leave creating and tearing down to {@link withSession} — if VRAM remains after a failure
 * midway, the following detection fails to allocate and fails "somewhere other than the first
 * failure". A plain `finally` that awaits `dispose` lets dispose's exception overwrite the
 * original one when dispose also fails after a failure of run or of the diagnostics callback
 * (device lost causes both), erasing what broke first.
 * NOTE: the `run` wrapper of `withSession` does not take symbol bindings, so this runs directly on
 * the Session passed to the body, and calls the diagnostics from here in the same order (right
 * after run).
 */
const runGraph = (
  state: VowelDetectorState,
  features: Tensor,
  rows: number,
): Promise<Float32Array<ArrayBuffer>> => {
  const model = state.graph;
  return withSession(state.gpu, model, state.sessionOptions, undefined, async (_run, session) => {
    // The symbol `T` is the count on the 20ms grid. The input axis is the derived dimension `2T`,
    // so it can also be solved from the actual shape (ADR 0057), but the host passes the count it
    // computed explicitly so a disagreement fails before running.
    const outputs = await session.run({ [INPUT_NAME]: features }, { [state.symbol]: rows });
    if (state.onRunDiagnostics !== undefined) state.onRunDiagnostics(session.diagnostics());
    const name = model.graph.outputs[0];
    const tensor = outputs[name];
    if (tensor === undefined) {
      throw new Error(`vowel-detector: graph output '${name}' is missing from the run results`);
    }
    // Narrow by the discriminant (a cast to Float32Array would pass silently when the dtype is
    // off).
    if (tensor.dtype !== "f32") {
      throw new Error(`vowel-detector: the logits are not f32 (${tensor.dtype})`);
    }
    return tensor.data;
  });
};

/**
 * Runs one audio through (limit check → features → truncation of the leftover → floor check → run
 * at the actual length → post-processing).
 */
const detectAudio = async (
  state: VowelDetectorState,
  audio: Float32Array<ArrayBufferLike>,
): Promise<Result> => {
  const { config } = state;
  // The output is a 20ms grid, so the one leftover frame of an odd frame count is dropped
  // (**truncation, not pad** — adding one half frame would make that one output carry only one
  // input frame).
  const frames = featureFrameCount(audio.length);
  const usable = frames - (frames % TIME_STRIDE);
  // Compare the actual length after dropping the leftover with **both sides** of the operating
  // range the distribution declares (the lower limit is stricter than "can one output be made" —
  // it is the graph's traced range itself).
  // MUST: check the upper limit **before** building the features (the MUST of
  // {@link featureFrameCount} — do not pay for the FFT over every frame and the scan for non-finite
  // samples for a decision that depends only on the length). The lower limit stays after the
  // features — a short waveform is cheap to extract, and a length that yields not even one frame
  // is rejected by `extractFeatures` naming the sample count (the lower-limit message would show a
  // negative frame count).
  assertFrameLimit(config, usable);
  const features = extractFeatures(audio, state.melBasis);
  assertFrameFloor(config, usable);
  const rows = usable / TIME_STRIDE;

  const logits = await runGraph(
    state,
    {
      dtype: "f32",
      shape: [1, usable, config.featureDim],
      // Pass only the actual length after dropping the leftover (not a single element of pad — the
      // module doc).
      data: features.data.slice(0, usable * config.featureDim),
    },
    rows,
  );
  const expected = rows * config.classes.length;
  if (logits.length !== expected) {
    throw new Error(
      `vowel-detector: logit element count ${logits.length} differs from ` +
        `${rows}×${config.classes.length}`,
    );
  }
  const segments = logitsToSegments(logits, rows);
  return { segments, lab: toLab(segments) };
};

/**
 * The vowel-detection pipeline (audio → `.lab`).
 *
 * Construction enters only through {@link Pipeline.fromPretrained} (fetch from HF) or
 * {@link Pipeline.fromAssets} (fetched bytes) — the constructor is private so that a
 * half-built state that bypasses the manifest check and the asset comparison cannot be made (ADR
 * 0008).
 */
export class Pipeline {
  readonly #state: VowelDetectorState;
  /**
   * Serialization chain of detect and dispose (keeps "one at a time" on the public API side — the
   * module doc).
   */
  readonly #chain = createOperationChain();
  /**
   * The one dispose. **Not being undefined means "already disposed"** (holding a separate boolean
   * would make it a derived state updated independently).
   */
  #disposal: Promise<void> | undefined;

  private constructor(state: VowelDetectorState) {
    this.#state = state;
  }

  /**
   * Fetches from a distribution and builds (`loadManifest` → `resolveSelection` → fetch **only each
   * component's descriptor (part 0)** for admission → warm up the weight parts → `fetchAssets` for
   * the remaining assets → construction). Blocks are read in part order at the very moment the
   * Session is built (ADR 0109 — `src/hub/components.ts`). To replace a component with the same
   * role from another repository, use {@link FromPretrainedComponentOptions.components} (the role
   * is the one `crnn`). A string `ref` reads as `{ repo }` (= follows `main`). **`ref` is
   * required** (a source has no default — the MUST of `src/hub/repo-ref.ts`. This family has no
   * public distribution repository, so there is no pin constant either).
   *
   * A local distribution is passed as a **source handle** (`localDirectory` / `denoDirectory` of
   * `@karume/hub/deno` / `nodeDirectory` of `@karume/hub/node`). It goes through neither the
   * spelling check of the HF `owner/name`, nor the network, nor CacheStorage (the HTTP-only knobs
   * of {@link FromPretrainedOptions} have no effect).
   */
  static async fromPretrained(
    ref: string | HubRepoRef | DistributionSource,
    options: FromPretrainedOptions = {},
  ): Promise<Pipeline> {
    const source = toManifestSource(ref, "vowelDetector.Pipeline.fromPretrained");
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
    // The family's checks run in the admission hook (before fetching the weight parts —
    // `hub/components.ts`).
    const { admitted, assets, open } = await loadContainerComponents(
      "vowelDetector.Pipeline.fromPretrained",
      loaded,
      selection,
      COMPONENT_KEYS,
      async (open) => {
        const admitted = admitVowelDetector(loaded.manifest, open, buildOptions);
        // The `requiredLimits` the distribution declares are checked here **before fetching the
        // weight parts** (ADR 0089 decision 5 — the limits of a shared GPU, or the adapter's
        // measured values on the path that acquires its own).
        await assertRequiredLimitsBeforeDownload(
          admitted.quant.requiredLimits,
          buildOptions.gpu,
          `vowelDetector.Pipeline: quant '${admitted.quantName}'`,
        );
        return admitted;
      },
      {
        ...hubOptions,
        ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
        ...(options.components === undefined ? {} : { components: options.components }),
      },
    );
    return new Pipeline(await openVowelDetectorState(admitted, assets, open, buildOptions));
  }

  /**
   * Builds from a fetched manifest + assets. The contract check, the interpretation of the assets,
   * `openContainer` and the comparison with the graph are all done here (no Session is created yet
   * — the module doc's MUST).
   *
   * Component keys **accept both forms** — the bytes of the single form `krm` as one entry
   * (`crnn`), and the part sequence of the split form (`crnn[0]` / `crnn[1]` / … — in index order
   * from part 0). The split form is opened as the part sequence without concatenating the bytes.
   * Missing indices and mixing with a single-form key fail loudly (the receiving implementation is
   * the one in `src/hub/components.ts`).
   */
  static async fromAssets(input: Assets, options: PipelineOptions = {}): Promise<Pipeline> {
    const open = await assetOpener(input.assets);
    const admitted = admitVowelDetector(input.manifest, open, options);
    return new Pipeline(await openVowelDetectorState(admitted, input.assets, open, options));
  }

  /**
   * Emits the vowel sequence of one audio. Both the segment sequence and the `.lab` body are
   * returned (2 representations of the same segmentation — the same as upstream
   * `@hdae/vowel-detector`'s `detect`).
   *
   * The input is a **16kHz mono** `Float32Array` (`[-1, 1]` scale). Resampling is the caller's
   * responsibility, and **this function does not fail** on a different rate (the module doc).
   *
   * The length is bound to the symbolic dimension `T` at the actual length (not a single element
   * of pad goes in). If the 10ms frame count is odd, the one leftover frame is **truncated**.
   * Below `minFrames` and above `maxFrames` both fail loudly (same).
   *
   * When called concurrently, calls **wait and run in order** (GPU destruction and an in-flight
   * run do not interleave).
   */
  async detect(audio: Float32Array<ArrayBufferLike>): Promise<Result> {
    // The disposed check is made at call time (checked inside the chain, it would also take down
    // detections accepted before the dispose).
    if (this.#disposal !== undefined) {
      throw new Error("vowelDetector.Pipeline: cannot detect after dispose");
    }
    return await this.#chain(() => detectAudio(this.#state, audio));
  }

  /**
   * The sampling rate required of the input waveform (the distribution's declaration — the
   * caller's resampling target).
   */
  get sampleRate(): typeof SAMPLE_RATE {
    return this.#state.config.sampleRate;
  }

  /**
   * Releases. Destroys **only an internally acquired GPU** (a GpuContext passed via `options.gpu`
   * belongs to the caller and is not touched). Sessions are torn down per `detect`, so there is
   * nothing to tear down here (the module doc's MUST).
   *
   * MUST: wait for in-flight detections to complete before destroying (flush-before-destroy) —
   * putting the destruction on the chain too fixes the order of waiting and destroying in one
   * place. The second and later calls return the same completion (returning earlier would let the
   * caller treat it as "already destroyed" and move on).
   */
  dispose(): Promise<void> {
    this.#disposal ??= this.#chain(() => {
      if (this.#state.ownsGpu) this.#state.gpu.destroy();
      return Promise.resolve();
    });
    return this.#disposal;
  }

  /** `await using` support (Explicit Resource Management) — an alias of {@link dispose}. */
  [Symbol.asyncDispose](): Promise<void> {
    return this.dispose();
  }
}
