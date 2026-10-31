/**
 * `Pipeline` — the one surface for text → speech (SBV2 / Style-Bert-VITS2 JP-Extra).
 *
 * Pipeline (every NN step is Karume, no torch):
 *
 * 1. utterance (analysed mora sequence + word alignment) → phonemes / tones / word2ph / DeBERTa
 *    token sequence (no GPU, deterministic)
 * 2. `text_encoder` (DeBERTa) → the N-th from the end of hidden_states (N from `symbols.json`)
 * 3. host: tile-expand the BERT features to phoneme level with word2ph
 * 4. host: look up rows of `style_vec` / `g` from the distribution's tables (`style.ts`)
 * 5. `front` (enc_p + dp + sdp reverse) → **4 outputs** (logw_sdp / logw_dp / m_p / logs_p)
 * 6. host: durations → frame expansion idx → z_p → relative-position attention tables
 * 7. `voice` (flow + dec fused) → f32 waveform (WAV encoding is `encodeWav` — shared processing
 *    independent of the pipeline)
 *
 * ## MUST: open and close the graphs one at a time
 *
 * {@link Pipeline.fromAssets} **creates no Session at all** — it opens only up to the
 * containers (`openContainer` = parsing the 2 documents), and GPU residency is created and torn
 * down step by step inside {@link Pipeline.generate}. The 3 graphs (text_encoder / front /
 * voice) are never alive at the same time.
 *
 * MUST: this sequencing is kept **on the public API side too** — `generate` rides the serialization
 * chain (concurrent calls wait and run in order), and `dispose` waits for its completion before
 * destroying the GPU. Without it, the graphs of 2 concurrent calls would be resident at once, and a
 * dispose during generation would break flush-before-destroy.
 *
 * ## MUST: the low-precision knobs apply only to front / voice
 *
 * The quant's `session` (the Session's execution knobs) is passed only to the front / voice
 * Sessions, and the `text_encoder` Session is always created with the default execution form —
 * keeping the comparison axis of the execution knobs on the SBV2 body. This is separate from the
 * weights' **storage dtype**: `text_encoder`'s weights come as i8 or i4 depending on the quant
 * (the default `i8+bert4` and `i4` are i4 — the quant table is in the addendum of ADR 0039).
 *
 * ## MUST: numerical correctness is not guaranteed here
 *
 * Correctness is guaranteed by the golden E2E (`packages/e2e/tests/e2e_sbv2_test.ts`) and by the
 * torch comparison through `reference` of `tools/export-recipes/sbv2/demo.py`
 * (`python -m sbv2.demo reference`) (the example's dump path — `examples/sbv2/dump.ts`, removed in
 * 0.15).
 *
 * ## MUST: text analysis is not this package's responsibility (the re-ruling of 0.6.0)
 *
 * The only entry is an analysed {@link Utterance} — the dictionary, the correction dictionary
 * and the choice of readings belong to the caller (ADR 0072's injection fields (overlay /
 * givenTone / analyzeProsody) were removed). This removes both the fetching of the Japanese
 * dictionary (the only network path not going through hub) and the dependency on an analyser from
 * `packages/models`. What remains on the karume side is only the entry check
 * (`text/model-input.ts`) that verifies "the given structure is self-consistent".
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
  type GpuFeaturesSpec,
  type HubRepoRef,
  loadManifest,
  type Manifest,
  type ModelEntry,
  type Quant,
  resolveSelection,
} from "@karume/hub";
import {
  parseSbv2PipelineConfig,
  SBV2_PIPELINE_MAJOR,
  SBV2_PIPELINE_NAME,
  type Sbv2PipelineConfig,
} from "./config.ts";
import {
  assertFiniteStyleWeight,
  parseSbv2Table,
  type Sbv2Table,
  SPEAKER_TENSOR,
  speakerEmbedding,
  STYLE_TENSOR,
  styleVector,
} from "./style.ts";
import { InputError } from "./errors.ts";
import { buildSbv2ModelInput, type Sbv2ModelInput } from "./text/model-input.ts";
import type { Utterance } from "./text/utterance.ts";
import { bertHiddenOutput, tileBertToPhoneLevel, type TiledBert } from "./text/bert-tile.ts";
import { buildRelPosTables } from "./text/rel-pos-tables.ts";
import { type JpExtraRules, parseJpExtraRules, type Sbv2Knobs } from "./text/symbols.ts";
import type { DebertaTokenizer } from "./text/tokenizer.ts";
import { parseTokenizerAsset } from "./text/asset.ts";
import { durationsToFrames } from "./host/duration.ts";
import { buildZp } from "./host/latent.ts";
import { Randn } from "./host/random.ts";
import { buildRelattnTables } from "./relattn-tables.ts";
import { ModelInputError } from "../errors.ts";
import { assertAcceptableSeed } from "../request-gates.ts";
import { createOperationChain } from "../concurrency/serial.ts";
import {
  assertGpuFeaturesGranted,
  assertRequiredLimitsBeforeDownload,
  assertRequiredLimitsSatisfied,
  sessionGpuFeatures,
  toAcquireGpuOptions,
} from "../session/gpu-features.ts";
import { type FamilySessionPolicy, resolveSessionOptions } from "../session/options.ts";
import { withSession } from "../session/with-session.ts";
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
import { readAssetBuffer, readAssetJson } from "../hub/asset-readers.ts";

/**
 * Names that appear in the manifest's weights / assets tables (the conventional names of ADR 0041
 * decision 3).
 * NOTE: `style_vectors` / `speaker_embeddings` are spelled the same as the fetch keys and as the
 * safetensors tensor names ({@link STYLE_TENSOR} / {@link SPEAKER_TENSOR} in `style.ts`), but they
 * are separate vocabularies, so they are held separately.
 */
const TEXT_ENCODER = "text_encoder";
const FRONT = "front";
const VOICE = "voice";
const TOKENIZER = "tokenizer";
const SYMBOLS = "symbols";
const STYLE_VECTORS = "style_vectors";
const SPEAKER_EMBEDDINGS = "speaker_embeddings";

/** This family's weights components (also the role names the component replacement accepts). */
const COMPONENT_KEYS = [FRONT, VOICE, TEXT_ENCODER] as const;

/**
 * Number of front's graph outputs (logw_sdp / logw_dp / m_p / logs_p).
 * Exactly the 4 that {@link outputAt} looks up by position are checked at construction
 * ({@link buildSbv2State}).
 */
const FRONT_OUTPUTS = 4;

/** Generation result. `data` is an f32 mono waveform (can be passed to `encodeWav` as is). */
export type GeneratedAudio = {
  readonly sampleRate: number;
  readonly data: Float32Array<ArrayBuffer>;
};

/**
 * Knobs of one generation. Unspecified fields are filled by the manifest's
 * `pipelineConfig.defaults`.
 */
export type GenerateOptions = {
  /** Speaker name (a key of `pipelineConfig.speakers`). */
  readonly speaker?: string;
  /** Style name (a key of `pipelineConfig.styles`). */
  readonly style?: string;
  /** Style strength (1 = the selected style itself, 0 = the mean style). */
  readonly styleWeight?: number;
  /** Mixing ratio of sdp and dp (`logw = sdp·r + dp·(1−r)`). */
  readonly sdpRatio?: number;
  /** Noise scale of z_p. */
  readonly noiseScale?: number;
  /** Noise scale of the sdp reverse. */
  readonly noiseScaleW?: number;
  /** Duration scale (larger is slower). */
  readonly lengthScale?: number;
  /** Random seed (default 0 — the same seed and the same knobs give the same waveform). */
  readonly seed?: number;
};

/** Component name received by {@link PipelineOptions.onRunDiagnostics} (1 Session = 1 name). */
export type RunComponent = "text_encoder" | "front" | "voice";

/**
 * Construction options (shared by {@link Pipeline.fromAssets} /
 * {@link Pipeline.fromPretrained}).
 */
export type PipelineOptions = {
  /**
   * Shares an existing GPU. **The passing side owns it**, so {@link Pipeline.dispose} does not
   * destroy it. When omitted, the pipeline calls `acquireGpu` internally and destroys it in
   * `dispose()`.
   */
  readonly gpu?: GpuContext;
  /** Model (a key of the manifest's models). `defaultModel` when omitted. */
  readonly model?: string;
  /** Execution configuration (a key of that model's quants). `defaultQuant` when omitted. */
  readonly quant?: string;
  /**
   * Observer hook that receives the diagnostics of each `Session.run` (1 synthesis = 3 runs:
   * text_encoder → front → voice). When per-op GPU time (`lastRunTiming`) is needed, pass
   * `acquireGpu({ gpuTiming: true })` as `gpu` (ADR 0021 — no timing by default).
   * Exceptions from the callback are not caught (fail loudly — the whole synthesis fails).
   */
  readonly onRunDiagnostics?: (component: RunComponent, diagnostics: SessionDiagnostics) => void;
};

/**
 * Fetch-layer options used only by {@link Pipeline.fromPretrained} (passed through to hub).
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
 * Turns fetched bytes into an ArrayBuffer that can be passed to `openContainer` (the check itself
 * is {@link readAssetBuffer}).
 */
const assetBuffer = (
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  key: string,
): ArrayBuffer => readAssetBuffer("sbv2", "weights / assets", assets, key);

/**
 * Component supplier of the in-memory surface (`fromAssets`) (the receiving implementation is
 * shared by the families — {@link assetComponentOpener}). A component key is either the single
 * form `krm` as one entry (`voice`), or the part sequence of the split form (`voice[0]` /
 * `voice[1]` / …).
 *
 * NOTE: `export` for the same reason as {@link openSbv2State} (the dump path builds the state on
 * the in-memory surface — `examples/sbv2/dump.ts`, removed in 0.15). Not exposed on `mod.ts` / the
 * subpath surface.
 */
export const assetOpener = (assets: Assets["assets"]): Promise<ComponentOpener> =>
  assetComponentOpener("sbv2", assets, (key) => assetBuffer(assets, key), COMPONENT_KEYS);

/**
 * Reads an asset JSON (the decode / parse check is {@link readAssetJson}).
 *
 * NOTE: `export` for tests that call the check directly (reaching here through `fromAssets` needs 3
 * real IR containers). Not exposed on `mod.ts` / the subpath surface (ADR 0008).
 */
export const assetJson = (
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  key: string,
): unknown => readAssetJson("sbv2", "weights / assets", assets, key);

/**
 * Looks up the **static** dimension of one axis of a graph input.
 *
 * MUST: do not put literals on the pipeline side — the container is authoritative, and when the
 * model is replaced the value follows.
 * MUST: do not require the whole shape to be static. `bert` is `[1, 1024, P]` where P is a symbolic
 * dimension (the phoneme count depends on the data), so only the needed axis is looked at.
 */
const staticInputDim = (model: GraphOwner, inputName: string, axis: number): number => {
  const spec = model.graph.inputs.find((input) => input.name === inputName);
  if (spec === undefined) throw new Error(`Graph input '${inputName}' does not exist`);
  const dim = spec.shape[axis];
  if (typeof dim !== "number") {
    throw new Error(
      `Graph input '${inputName}' axis ${axis} is not a static dimension (${String(dim)})`,
    );
  }
  return dim;
};

/**
 * Axis 1 (feature width) of a graph input. Looks up 1024 for `bert`, 256 for `style_vec`, 512 for
 * `g`.
 */
const featureWidth = (model: GraphOwner, inputName: string): number =>
  staticInputDim(model, inputName, 1);

const asF32 = (tensor: Tensor, where: string): Float32Array<ArrayBuffer> => {
  if (tensor.dtype !== "f32") throw new Error(`${where}: not f32 (${tensor.dtype})`);
  return tensor.data;
};

const f32 = (data: Float32Array<ArrayBuffer>, shape: readonly number[]): Tensor => ({
  dtype: "f32",
  shape: [...shape],
  data,
});

const i32 = (values: readonly number[], shape: readonly number[]): Tensor => ({
  dtype: "i32",
  shape: [...shape],
  data: Int32Array.from(values),
});

const ones = (count: number): Float32Array<ArrayBuffer> => new Float32Array(count).fill(1);

/**
 * Looks up a graph output by **position** (the IR's output names are torch node names from export
 * time — meaningless names like `slice_54` / `mul_3780`, so names are not hard-coded). The count is
 * checked at construction by {@link buildSbv2State}.
 */
const outputAt = (
  model: GraphOwner,
  outputs: Readonly<Record<string, Tensor>>,
  index: number,
): Tensor => {
  const name = model.graph.outputs[index];
  if (name === undefined) {
    throw new Error(
      `Graph output ${index} does not exist (there are only ${model.graph.outputs.length})`,
    );
  }
  const tensor = outputs[name];
  if (tensor === undefined)
    throw new Error(`Graph output ${index} ('${name}') is missing from the run results`);
  return tensor;
};

/**
 * Passes the component name, bound in, to the observer hook
 * ({@link PipelineOptions.onRunDiagnostics}).
 */
const observer = (
  state: Sbv2State,
  component: RunComponent,
): ((diagnostics: SessionDiagnostics) => void) | undefined => {
  const listener = state.onRunDiagnostics;
  return listener === undefined ? undefined : (diagnostics) => listener(component, diagnostics);
};

/**
 * Checks at construction that the distribution's tables (`style_vectors` / `speaker_embeddings`)
 * fit both the name table and the graph inputs.
 *
 * MUST: do not drop the row-count comparison. A row number is the physical row of the table itself,
 * so when the name table and the table's row count drift apart the shapes still match and **the
 * voice of a different style / different speaker comes out** (the doc of `style.ts`).
 *
 * NOTE: `export` for tests that call the check directly (the same situation as
 * {@link assertTokenLimit}). Not exposed on `mod.ts` / the subpath surface (ADR 0008).
 */
export const assertTableFits = (
  table: Sbv2Table,
  names: ReadonlyMap<string, number>,
  key: string,
  cols: number,
): void => {
  if (table.rows !== names.size) {
    throw new Error(
      `sbv2.Pipeline: asset '${key}' has ${table.rows} rows, which differs from the ${names.size} entries of pipelineConfig` +
        ` (${[...names.keys()].join(" / ")}) — the row index is the physical row of the table itself`,
    );
  }
  if (table.cols !== cols) {
    throw new Error(
      `sbv2.Pipeline: asset '${key}' has ${table.cols} columns, which differs from the graph input's ${cols}`,
    );
  }
};

/**
 * Checks at construction that front's graph outputs are exactly 4, as {@link outputAt}'s positional
 * lookup expects.
 *
 * MUST: do not make it "4 or more". Only positions 0..3 are read, so extra outputs would be
 * silently ignored, and grabbing a different variant (a verification graph exposing every layer,
 * etc.) as front would go unnoticed.
 *
 * NOTE: the **order** of the outputs (logw_sdp / logw_dp / m_p / logs_p) cannot be told apart from
 * TS because the shapes are the same — that is the responsibility of the exporter (the order of
 * output names in `tools/export-recipes/sbv2/export.py`) and of the WAV sha256 check of the
 * reference environment. This looks at the count only.
 * NOTE: `export` for tests that call the check directly (the same situation as
 * {@link assertTokenLimit}).
 */
export const assertFrontOutputs = (front: GraphOwner): void => {
  if (front.graph.outputs.length !== FRONT_OUTPUTS) {
    throw new Error(
      `sbv2.Pipeline: front has ${front.graph.outputs.length} graph outputs` +
        ` (exactly ${FRONT_OUTPUTS} are required: logw_sdp / logw_dp / m_p / logs_p)`,
    );
  }
};

/**
 * Checks at construction that the width of the hidden state text_encoder emits matches the width of
 * front's graph input `bert`.
 *
 * MUST: call it **before going to get the GPU** (= before creating the text_encoder Session). Left
 * until run time, it would build the distribution's text_encoder (DeBERTa), pay one run, and then
 * fail at `Session.run`'s element-count check — with a generic message that names neither the
 * family nor the asset.
 * MUST: look at **the last axis only**. The output shape is `[1, T, dim]` where T is a symbolic
 * dimension (the token count depends on the data), so requiring the whole shape to be static would
 * reject a legitimate distribution.
 *
 * NOTE: `export` for tests that call the check directly (the same situation as
 * {@link assertTokenLimit}).
 */
export const assertBertWidth = (
  textEncoder: GraphOwner,
  front: GraphOwner,
  bertHiddenFromEnd: number,
): void => {
  const name = bertHiddenOutput(textEncoder.graph.outputs, bertHiddenFromEnd);
  const info = textEncoder.graph.values[name];
  if (info === undefined) {
    throw new Error(`sbv2.Pipeline: text_encoder output '${name}' is not declared in graph.values`);
  }
  const width = info.shape.at(-1);
  if (typeof width !== "number") {
    throw new Error(
      `sbv2.Pipeline: the last axis of text_encoder output '${name}' is not a static dimension (${String(width)})`,
    );
  }
  const expected = featureWidth(front, "bert");
  if (width !== expected) {
    throw new Error(
      `sbv2.Pipeline: text_encoder output '${name}' has width ${width}, which differs from the width of front's graph input 'bert',` +
        ` ${expected} (text_encoder and front assume different BERTs)`,
    );
  }
};

/**
 * The internal state of {@link Pipeline}.
 *
 * NOTE: not exposed on the public surface (`mod.ts` / `sbv2.ts`). Only the function is `export`ed
 * because the dump path of the torch reference comparison (`examples/sbv2/dump.ts`, removed in
 * 0.15) calls {@link synthesizeSbv2} directly — it is the only opening for observing intermediate
 * values without keeping 2 implementations of the chain, not a user story. The type itself is
 * inferred on the caller's side, so it stays inside the module.
 */
type Sbv2State = {
  readonly gpu: GpuContext;
  readonly ownsGpu: boolean;
  readonly config: Sbv2PipelineConfig;
  readonly rules: JpExtraRules;
  readonly tokenizer: DebertaTokenizer;
  readonly styles: Sbv2Table;
  readonly speakers: Sbv2Table;
  readonly textEncoder: ModelComponent;
  readonly front: ModelComponent;
  readonly voice: ModelComponent;
  readonly sessionOptions: SessionOptions;
  readonly onRunDiagnostics?: (component: RunComponent, diagnostics: SessionDiagnostics) => void;
};

/** Intermediate values of one synthesis (read only by the dump path and diagnostics). */
type Sbv2Trace = {
  readonly input: Sbv2ModelInput;
  readonly speaker: string;
  readonly style: string;
  readonly styleWeight: number;
  readonly knobs: Sbv2Knobs;
  readonly seed: number;
  /** The graph output name of the hidden_states actually taken. */
  readonly bertHiddenOutput: string;
  readonly xMask: Float32Array<ArrayBuffer>;
  readonly zNoise: Float32Array<ArrayBuffer>;
  readonly wCeil: Int32Array<ArrayBuffer>;
  readonly zpNoise: Float32Array<ArrayBuffer>;
  readonly channels: number;
  readonly frames: number;
};

/**
 * Return value of {@link synthesizeSbv2}. {@link Pipeline.generate} takes only the audio from
 * it.
 */
type Sbv2Synthesis = {
  readonly sampleRate: number;
  readonly audio: Float32Array<ArrayBuffer>;
  readonly trace: Sbv2Trace;
};

const resolveRow = (names: ReadonlyMap<string, number>, name: string, where: string): number => {
  const row = names.get(name);
  if (row === undefined) {
    throw new InputError(
      `${where} '${name}' is not in the manifest (available: ${[...names.keys()].join(" / ")})`,
    );
  }
  return row;
};

const finiteKnob = (value: number, name: string): number => {
  if (!Number.isFinite(value)) throw new InputError(`${name} ${value} is not a finite number`);
  return value;
};

/**
 * Check of the acceptance set of `seed`. It holds no condition itself; it delegates to the
 * family-wide owner {@link assertAcceptableSeed} (`request-gates.ts` — a non-negative safe integer)
 * and only re-wraps the type.
 *
 * MUST: call it at the same step as the other knobs (before creating a Session). If the only check
 * were the `Randn` constructor, a request known from the start to be unrunnable would pay for
 * building text_encoder (DeBERTa) and a whole run before failing (the same reason as the MUST on
 * the position of {@link assertTokenLimit}).
 * MUST: keep the type `InputError`. `Randn` now also throws `ModelInputError` in line with the
 * owner, so it is no longer outside the classification axis, but hosts that carry only SBV2 carve
 * out 400 with `instanceof InputError` (the line drawn by ADR 0072 decision 6), so loosening
 * this entry's type to the parent is a breaking change. `InputError` derives from
 * `ModelInputError`, so it rides the cross-family branching of hosts that carry several families
 * as is.
 *
 * NOTE: `export` for tests that call the check directly (the same situation as
 * {@link assertTokenLimit}).
 */
export const assertSeed = (seed: number): void => {
  try {
    assertAcceptableSeed(seed);
  } catch (cause) {
    // The diagnostic carries the owner's message as is (neither the acceptance set nor the message
    // is written in 2 places).
    if (cause instanceof ModelInputError) throw new InputError(cause.message, { cause });
    throw cause;
  }
};

/**
 * Check of the operating limit of the token sequence (`pipelineConfig.maxTokens`).
 *
 * MUST: call it **before** generating the relative-position tables (`c2p_pos` / `p2c_pos` are
 * 4·T² bytes each) and allocating the Session. The limit is the upper bound of the baked graph's
 * symbolic dimension itself, and a request beyond it cannot run however it is allocated.
 *
 * NOTE: `export` for tests that call the check directly (reaching here on the synthesis path needs
 * 3 real IR containers). Not exposed on `mod.ts` / the subpath surface (ADR 0008).
 */
export const assertTokenLimit = (tokens: number, maxTokens: number): void => {
  if (tokens > maxTokens) {
    throw new InputError(
      `sbv2.Pipeline: token count ${tokens} exceeds the distribution's maximum maxTokens=${maxTokens}` +
        " (split the text into shorter parts and synthesise each)",
    );
  }
};

/**
 * Check of the operating limit of the phoneme sequence (front's symbolic dimension P).
 *
 * MUST: needed **separately** from {@link assertTokenLimit}. Which of P = 2·sum(baseWord2ph)+1 and
 * T = total tokens of the word surfaces + 2 is larger **depends on the utterance**, and neither
 * implies the other — `distributePhone` gives 0 to words with fewer phonemes than characters (the
 * symbol word `{surface:"…", phones:["…"]}` splits into the 3 tokens of `"..."` under NFKC, giving
 * `[1,0,0]`), so lining up such words also gives P < T. Dropping either check lets through
 * utterances where only one side exceeds the limit.
 * MUST: call it **before** running DeBERTa. Placed after, a request already certain to exceed would
 * create and fully run text_encoder (DeBERTa) before failing on front's symbolic dimension.
 *
 * NOTE: the limit is the same single value for T and P (`pipelineConfig.maxTokens`) — that the
 * baked symbolic-dimension upper bounds form one set is fixed on the distribution side
 * (`test_the_limits_are_the_symbolic_maxima_the_export_scripts_baked` in
 * `tools/export-recipes/sbv2/tests/test_distribution.py`), so no field is added.
 * NOTE: `export` for tests that call the check directly (the same situation as
 * {@link assertTokenLimit}).
 */
export const assertPhonemeLimit = (phonemes: number, maxTokens: number): void => {
  if (phonemes > maxTokens) {
    throw new InputError(
      `sbv2.Pipeline: phoneme count ${phonemes} exceeds the distribution's maximum maxTokens=${maxTokens}` +
        " (split the text into shorter parts and synthesise each)",
    );
  }
};

/**
 * Check that sdp's `z_noise` (standard normal × `noiseScaleW`) is all finite in f32.
 *
 * MUST: the check of `noiseScaleW` is only double-precision finiteness, so a product above the f32
 * maximum (e.g. `noiseScaleW = 1e39`) becomes ±Inf the moment it enters a `Float32Array` and does
 * not surface until the output check after front has fully run (an input error shows up as an
 * internal error after wasted GPU work). The random numbers depend only on the length and the
 * seed, so the sequence drawn before running the GPU is scanned once.
 * MUST: the thrown type depends on where the value came from — a `noiseScaleW` the caller passed is
 * input-caused (`InputError`); if it was omitted and the distribution's
 * `pipelineConfig.defaults` was used, the asset is invalid.
 * NOTE: a magnitude that is finite in f32 but overflows inside the graph (e.g. 1e30) is a separate
 * problem and is not bounded here.
 * NOTE: `export` for tests that call the check directly (the same situation as
 * {@link assertTokenLimit}).
 */
export const assertFiniteNoise = (
  noise: Float32Array<ArrayBuffer>,
  noiseScaleW: number,
  fromRequest: boolean,
): void => {
  if (noise.every(Number.isFinite)) return;
  const problem = `noiseScaleW ${noiseScaleW} makes z_noise non-finite in f32`;
  if (fromRequest) throw new InputError(`sbv2.Pipeline: ${problem} (pass a smaller value)`);
  throw new Error(`sbv2.Pipeline: pipelineConfig.defaults: ${problem}`);
};

/**
 * Check of the tile-expansion result before handing it to front.
 *
 * MUST: compare **the number of columns written** ({@link TiledBert.columns}) with both the
 * allocation size and the symbolic dimension. The single check "expanded length === phoneme count"
 * would be a theorem derived on both sides from the same `Sbv2ModelInput` and thus a tautology
 * (`sum(word2ph) === P` is already enforced by `buildBaseWord2ph`), staying green even when the
 * walk breaks.
 *
 * NOTE: `export` for tests that call the check directly (the same situation as
 * {@link assertTokenLimit}).
 */
export const assertTiledBert = (tiled: TiledBert, phonemes: number): void => {
  if (tiled.data.length !== tiled.dim * tiled.columns) {
    throw new Error(
      `The ${tiled.columns} columns × dim ${tiled.dim} written by the BERT expansion do not match the allocated` +
        ` ${tiled.data.length} elements (the tile walk is broken)`,
    );
  }
  if (tiled.columns !== phonemes) {
    throw new Error(
      `BERT expansion length ${tiled.columns} differs from the phoneme count ${phonemes}`,
    );
  }
};

/**
 * The execution knobs sbv2 accepts (the manifest's quant declaration). The merge rule (explicit >
 * declared > runtime default), the value ranges, the combinations and the classification of the
 * thrown type are held by `resolveSessionOptions`, shared by every family; this only decides the
 * acceptance set. There is no explicit override (the construction options have no field for
 * execution knobs).
 *
 * Only `linearCompute` is accepted because it is the one field the distribution (`i8-a8`) declares
 * and that was confirmed against the reference values (the other fields are combinations for which
 * neither an effective slot in sbv2's graphs nor the numbers have been confirmed).
 *
 * NOTE: `export` so the equivalence test runs every quant of the mirror through the same table (not
 * exposed on `mod.ts` / the subpath surface — ADR 0008).
 */
export const SBV2_SESSION_POLICY: FamilySessionPolicy = {
  linearCompute: true,
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

/** Material settled by the family admission ({@link admitSbv2}). */
type Sbv2Admission = {
  readonly config: Sbv2PipelineConfig;
  readonly quantName: string;
  readonly quant: Quant;
  /**
   * The effective settings of the quant declaration passed through the acceptance table
   * ({@link SBV2_SESSION_POLICY}).
   */
  readonly sessionOptions: SessionOptions;
  /**
   * The quant declaration plus the features the effective settings need (both the request and the
   * check look at this).
   */
  readonly gpuFeatures: GpuFeaturesSpec | undefined;
};

/**
 * Checks whether this manifest and these graphs can run as sbv2 (the family admission hook of
 * `hub/components.ts` — on the fetching surface it is called **before a single byte of the weight
 * parts is fetched**).
 *
 * MUST: gather the family's checks into this one function. Scattered into later steps, on the
 * fetching surface they would fail only **after** downloading GB-class weights (the implementation
 * would be narrower than the wording of ADR 0070 decision 5).
 * MUST: fail on manifest contract violations **before going to get the GPU**. If the order slips,
 * in an environment without a GPU it turns into a different exception and "what was wrong" does
 * not reach the reader.
 *
 * NOTE: the comparison of the tables (`style_vectors` / `speaker_embeddings`) with the graph widths
 * cannot be placed in this hook — at admission time the asset bytes have not been fetched yet
 * (fetching them first would lose the position before the weight prefetch), so it stays in
 * {@link buildSbv2State}.
 */
const admitSbv2 = (
  manifest: Manifest,
  open: ComponentOpener,
  options: PipelineOptions,
): Sbv2Admission => {
  const modelName = options.model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, modelName)) {
    throw new Error(
      `sbv2.Pipeline: model '${modelName}' is not in the manifest` +
        ` (available: ${manifest.available.models.join(" / ")})`,
    );
  }
  const entry: ModelEntry = manifest.models[modelName];
  const { name, major } = entry.pipeline;
  if (name !== SBV2_PIPELINE_NAME) {
    throw new Error(
      `sbv2.Pipeline: the manifest pipeline is '${name}/${major}'` +
        ` ('${SBV2_PIPELINE_NAME}/${SBV2_PIPELINE_MAJOR}' is required)`,
    );
  }
  if (major !== SBV2_PIPELINE_MAJOR) {
    // The only check that stops the silent degradation "old implementation × new repository" (ADR
    // 0038 §6).
    throw new Error(
      `sbv2.Pipeline: pipeline '${name}/${major}' has an unsupported major` +
        ` (this implementation reads ${SBV2_PIPELINE_NAME}/${SBV2_PIPELINE_MAJOR})`,
    );
  }
  const config = parseSbv2PipelineConfig(entry.pipelineConfig);

  const quantName = options.quant ?? entry.defaultQuant;
  if (!Object.hasOwn(entry.quants, quantName)) {
    throw new Error(
      `sbv2.Pipeline: quant '${quantName}' is not in the manifest` +
        ` (available: ${entry.available.quants.join(" / ")})`,
    );
  }
  const quant = entry.quants[quantName];
  // Unsupported declarations fail before opening the components (= before fetching the weight
  // parts) (the one function shared by every family).
  const sessionOptions = resolveSessionOptions(
    SBV2_SESSION_POLICY,
    quant.session,
    {},
    `sbv2.Pipeline: quant '${quantName}'`,
  );
  const gpuFeatures = sessionGpuFeatures(quant.gpuFeatures, sessionOptions);

  // Check in this hook that all 3 components can be opened (the supplier is the same one as later —
  // a role that is not open fails loudly). The comparison of graph widths needs the table bytes,
  // so it stays in {@link buildSbv2State}.
  for (const key of COMPONENT_KEYS) open(key);

  // MUST: fail here on missing capabilities of a shared GPU (feature / device limit) — unlike the
  // case where the pipeline acquires its own, there is no reason to wait for `acquireGpu`, and
  // this is the only family check that knows before downloading the weights (the mapping between
  // request and check is the one in `session/gpu-features.ts`, and the later check calls the same
  // function).
  if (options.gpu !== undefined) {
    assertGpuFeaturesGranted(gpuFeatures, options.gpu, `sbv2.Pipeline: quant '${quantName}'`);
    assertRequiredLimitsSatisfied(
      quant.requiredLimits,
      options.gpu.limits,
      `sbv2.Pipeline: quant '${quantName}'`,
    );
  }

  return { config, quantName, quant, sessionOptions, gpuFeatures };
};

/**
 * Builds the run state from the material that passed admission + the assets.
 *
 * MUST: fail on **parsing the assets and comparing the tables** **before going to get the GPU**. If
 * the order slips, in an environment without a GPU it turns into a different exception and "what
 * was wrong" does not reach the reader. The only check allowed after acquiring the GPU is the GPU's
 * capability (shader-f16).
 * MUST: create no Session at all (the VRAM MUST — the module doc).
 */
const buildSbv2State = async (
  admitted: Sbv2Admission,
  assets: Assets["assets"],
  open: ComponentOpener,
  options: PipelineOptions = {},
): Promise<Sbv2State> => {
  const { config, quantName, quant, sessionOptions, gpuFeatures } = admitted;
  // The supplier is **the same one** admission looked at (`open` only looks up the opened
  // component).
  const front = open(FRONT);
  const voice = open(VOICE);
  const textEncoder = open(TEXT_ENCODER);
  const rules = parseJpExtraRules(assetJson(assets, SYMBOLS), SYMBOLS);
  const tokenizer = parseTokenizerAsset(assetJson(assets, TOKENIZER), TOKENIZER);
  const styles = parseSbv2Table(assetBuffer(assets, STYLE_VECTORS), STYLE_TENSOR);
  const speakers = parseSbv2Table(assetBuffer(assets, SPEAKER_EMBEDDINGS), SPEAKER_TENSOR);
  assertTableFits(styles, config.styles, STYLE_VECTORS, featureWidth(front, "style_vec"));
  assertTableFits(speakers, config.speakers, SPEAKER_EMBEDDINGS, featureWidth(front, "g"));
  assertFrontOutputs(front);
  assertBertWidth(textEncoder, front, rules.bertHiddenFromEnd);
  // front and voice receive the same `g`. If the widths differ, one of them asks for a different
  // speaker table.
  const voiceGin = featureWidth(voice, "g");
  if (voiceGin !== speakers.cols) {
    throw new Error(
      `sbv2.Pipeline: the width ${voiceGin} of voice's graph input 'g' differs from the speaker table's column count ${speakers.cols}` +
        " (front / voice require different speaker embeddings)",
    );
  }
  // MUST: declared features can be requested only at device creation (ADR 0028). With a shared GPU
  // they cannot be requested, so fail naming the missing capability (a shared GPU was already
  // checked by {@link admitSbv2} with the same function, but for a device acquired here this is
  // the only check).
  const gpu = options.gpu ?? (await acquireGpu(toAcquireGpuOptions(gpuFeatures)));
  const ownsGpu = options.gpu === undefined;
  try {
    assertGpuFeaturesGranted(gpuFeatures, gpu, `sbv2.Pipeline: quant '${quantName}'`);
    // Also check the limits of a device acquired here — `fromAssets` has no pre-fetch check
    // (`fromPretrained` checks with the adapter's values), and without this a shortfall of the
    // declaration would be delayed until Session construction, and when the declaration is stricter
    // than the real demand the comparison would never happen at all. A shared GPU is checked by
    // {@link admitSbv2}.
    if (ownsGpu) {
      assertRequiredLimitsSatisfied(
        quant.requiredLimits,
        gpu.limits,
        `sbv2.Pipeline: quant '${quantName}'`,
      );
    }
  } catch (error) {
    // A GPU acquired internally can be released by no one if we throw here, so give it back first.
    if (ownsGpu) gpu.destroy();
    throw error;
  }
  return {
    gpu,
    ownsGpu,
    config,
    rules,
    tokenizer,
    styles,
    speakers,
    textEncoder,
    front,
    voice,
    // The low-precision knobs apply only to the front / voice Sessions (the module doc's MUST).
    sessionOptions,
    ...(options.onRunDiagnostics === undefined
      ? {}
      : { onRunDiagnostics: options.onRunDiagnostics }),
  };
};

/**
 * Builds the run state from the manifest + assets (the body of {@link Pipeline.fromAssets} =
 * the one sequence family admission → state construction). Surfaces that have only fetched bytes
 * (`fromAssets` / the dump script) enter here — the fetching surface moves only admission forward
 * into the hook of {@link loadContainerComponents}, so it calls these 2 steps separately.
 */
export const openSbv2State = async (
  input: Assets,
  open: ComponentOpener,
  options: PipelineOptions = {},
): Promise<Sbv2State> =>
  await buildSbv2State(admitSbv2(input.manifest, open, options), input.assets, open, options);

/** Destroys only an internally acquired GPU (a passed GpuContext belongs to the caller). */
export const closeSbv2State = (state: Sbv2State): void => {
  if (state.ownsGpu) state.gpu.destroy();
};

/**
 * Turns one utterance into a waveform (with intermediate values). {@link Pipeline.generate}
 * calls this and returns only the audio.
 *
 * MUST: the order in which random numbers are consumed is fixed: `z_noise` (before front) →
 * `zp_noise` (after front). Swapping the order makes the same seed give a different waveform and
 * also breaks the correspondence between the dumped random sequence and the torch reference
 * (`host/random.ts`).
 */
export const synthesizeSbv2 = async (
  state: Sbv2State,
  utterance: Utterance,
  options: GenerateOptions = {},
): Promise<Sbv2Synthesis> => {
  if (utterance.moras.length === 0 && utterance.leadingPunctuations.length === 0) {
    throw new InputError(
      "sbv2.Pipeline: the utterance is empty (no moras and no leadingPunctuations)",
    );
  }
  const defaults = state.config.defaults;
  const speaker = options.speaker ?? defaults.speaker;
  const style = options.style ?? defaults.style;
  // The owner of the acceptance set of `styleWeight` is the table-lookup side (`style.ts`) — the
  // entry calls it without copying the condition.
  const styleWeight = options.styleWeight ?? defaults.styleWeight;
  assertFiniteStyleWeight(styleWeight);
  const knobs: Sbv2Knobs = {
    sdpRatio: finiteKnob(options.sdpRatio ?? defaults.sdpRatio, "sdpRatio"),
    noiseScale: finiteKnob(options.noiseScale ?? defaults.noiseScale, "noiseScale"),
    noiseScaleW: finiteKnob(options.noiseScaleW ?? defaults.noiseScaleW, "noiseScaleW"),
    lengthScale: finiteKnob(options.lengthScale ?? defaults.lengthScale, "lengthScale"),
  };
  if (knobs.lengthScale <= 0) {
    throw new InputError(`lengthScale ${knobs.lengthScale} is not positive`);
  }
  const seed = options.seed ?? 0;
  assertSeed(seed);
  const styleRow = resolveRow(state.config.styles, style, "Style");
  const speakerRow = resolveRow(state.config.speakers, speaker, "Speaker");

  // --- (1) text layer (no GPU, deterministic) ------------------------------
  // The self-consistency of the utterance (phoneme match of moras and words, sum(word2ph) === P) is
  // checked here.
  const input = buildSbv2ModelInput(utterance, state.tokenizer, state.rules);
  const phonemes = input.ids.phoneIds.length;
  const tokens = input.inputIds.length;
  // Fail before allocating the tables or creating any Session (the MUST of
  // {@link assertTokenLimit}). The limit is the same single value for T and P, but which is larger
  // depends on the utterance (the MUST of {@link assertPhonemeLimit}), so **both** are checked.
  assertTokenLimit(tokens, state.config.maxTokens);
  assertPhonemeLimit(phonemes, state.config.maxTokens);

  // sdp's random numbers depend only on the length (phoneme count) and the seed, so they are drawn
  // and checked before running the GPU (the MUST of {@link assertFiniteNoise}). The draw order
  // stays before the z_p noise of the later step (the same sequence for the same seed).
  const random = new Randn(seed);
  const zNoise = random.normals(2 * phonemes, knobs.noiseScaleW);
  assertFiniteNoise(zNoise, knobs.noiseScaleW, options.noiseScaleW !== undefined);

  // --- (2) text_encoder (DeBERTa; the distribution exposes only the one output used) ---
  // The quant's execution knobs are not passed (module doc "the low-precision knobs apply only to
  // front / voice").
  const hidden = await withSession(
    state.gpu,
    state.textEncoder,
    {},
    observer(state, "text_encoder"),
    async (run) => {
      // The relative-position index tables are graph inputs (baked in they would be 2MiB at
      // Tmax=512 — ADR 0045 wave 3).
      const relPos = buildRelPosTables(
        tokens,
        state.rules.bertRelPos.positionBuckets,
        state.rules.bertRelPos.maxPosition,
      );
      const outputs = await run({
        input_ids: i32(input.inputIds, [1, tokens]),
        attention_mask: i32(
          input.inputIds.map(() => 1),
          [1, tokens],
        ),
        c2p_pos: relPos.c2pPos,
        p2c_pos: relPos.p2cPos,
      });
      const name = bertHiddenOutput(state.textEncoder.graph.outputs, state.rules.bertHiddenFromEnd);
      const tensor = outputs[name];
      if (tensor === undefined) throw new Error(`text_encoder output '${name}' does not exist`);
      return { name, data: asF32(tensor, `text_encoder output '${name}'`) };
    },
  );

  // --- (3) tile-expand the BERT features to phoneme level --------------------
  const tiled = tileBertToPhoneLevel(hidden.data, tokens, input.word2ph);
  assertTiledBert(tiled, phonemes);

  // --- (4) look up style_vec / g from the tables ----------------------------
  const styleVec = styleVector(
    state.styles.data,
    state.styles.rows,
    state.styles.cols,
    styleRow,
    styleWeight,
  );
  const g = speakerEmbedding(
    state.speakers.data,
    state.speakers.rows,
    state.speakers.cols,
    speakerRow,
  );

  // --- (5) front (enc_p + dp + sdp reverse, 4 outputs) ----------------------
  const xMask = ones(phonemes);
  const predicted = await withSession(
    state.gpu,
    state.front,
    state.sessionOptions,
    observer(state, "front"),
    async (run) => {
      const outputs = await run({
        x: i32(input.ids.phoneIds, [1, phonemes]),
        x_mask: f32(xMask, [1, 1, phonemes]),
        tone: i32(input.ids.toneIds, [1, phonemes]),
        language: i32(input.ids.languageIds, [1, phonemes]),
        bert: f32(tiled.data, [1, featureWidth(state.front, "bert"), phonemes]),
        style_vec: styleVec,
        g,
        z_noise: f32(zNoise, [1, 2, phonemes]),
      });
      const mP = outputAt(state.front, outputs, 2);
      return {
        logwSdp: asF32(outputAt(state.front, outputs, 0), "front logw_sdp"),
        logwDp: asF32(outputAt(state.front, outputs, 1), "front logw_dp"),
        mP: asF32(mP, "front m_p"),
        logsP: asF32(outputAt(state.front, outputs, 3), "front logs_p"),
        channels: mP.shape[1],
      };
    },
  );

  // --- (6) host glue (durations → frame expansion → z_p → relative-position tables) ---
  // The upper limit of the total frame count is checked by `durationsToFrames` before it allocates
  // the expansion sequence (`host/duration.ts`).
  const plan = durationsToFrames(
    predicted.logwSdp,
    predicted.logwDp,
    xMask,
    knobs.sdpRatio,
    knobs.lengthScale,
    state.config.maxFrames,
  );
  const channels = predicted.channels;
  const zpNoise = random.normals(channels * plan.totalFrames);
  const zP = buildZp(
    predicted.mP,
    predicted.logsP,
    plan.expandIdx,
    channels,
    zpNoise,
    knobs.noiseScale,
  );
  const tables = buildRelattnTables(plan.totalFrames);

  // --- (7) voice (flow + dec fused) -------------------------------------------
  const audio = await withSession(
    state.gpu,
    state.voice,
    state.sessionOptions,
    observer(state, "voice"),
    async (run) => {
      const outputs = await run({
        z_p: f32(zP, [1, channels, plan.totalFrames]),
        y_mask: f32(ones(plan.totalFrames), [1, 1, plan.totalFrames]),
        g,
        idx_k: tables.idxK,
        valid: tables.valid,
      });
      return asF32(outputAt(state.voice, outputs, 0), "voice waveform output");
    },
  );

  const expectedSamples = plan.totalFrames * state.rules.hopLength;
  if (audio.length !== expectedSamples) {
    throw new Error(
      `Waveform length ${audio.length} differs from hopLength×Ty(${expectedSamples})`,
    );
  }
  // MUST: do not silently return non-finite values. WAV encoding rounds NaN to 0, so a silent wrong
  // value would come out as is as "audio that is partly silent".
  for (const sample of audio) {
    if (!Number.isFinite(sample)) throw new Error("The waveform contains non-finite values");
  }

  return {
    // MUST: the canonical sampling rate is `symbols.json` (not held in pipelineConfig — no
    // duplicate holding of the derivation source).
    sampleRate: state.rules.samplingRate,
    audio,
    trace: {
      input,
      speaker,
      style,
      styleWeight,
      knobs,
      seed,
      bertHiddenOutput: hidden.name,
      xMask,
      zNoise,
      wCeil: plan.wCeil,
      zpNoise,
      channels,
      frames: plan.totalFrames,
    },
  };
};

/**
 * SBV2's text → speech pipeline.
 *
 * Construction enters only through {@link Pipeline.fromPretrained} (fetch from HF) or
 * {@link Pipeline.fromAssets} (fetched bytes) — the constructor is private so that a half-built
 * state that bypasses the manifest check and the asset comparison cannot be made (ADR 0008).
 */
export class Pipeline {
  readonly #state: Sbv2State;
  /**
   * Serialization chain of generate and dispose (keeps the module doc's "one at a time" on the
   * public API side).
   */
  readonly #chain = createOperationChain();
  /**
   * The one dispose. **Not being undefined means "already disposed"** (holding a separate boolean
   * would make it a derived state updated independently).
   */
  #disposal: Promise<void> | undefined;

  private constructor(state: Sbv2State) {
    this.#state = state;
  }

  /**
   * Fetches from a distribution and builds (`loadManifest` → `resolveSelection` → fetch **only each
   * component's descriptor (part 0)** for admission → warm up the weight parts → `fetchAssets` for
   * the remaining assets → construction). Blocks are read in part order at the very moment the
   * Session is built (ADR 0109 — `src/hub/components.ts`). To replace a component with the same
   * role from another repository, use {@link FromPretrainedComponentOptions.components} (`front` /
   * `voice` / `text_encoder`). A string `ref` reads as `{ repo }` (= follows `main`). **`ref` is
   * required** (a source has no default — the MUST of `src/hub/repo-ref.ts`).
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
    const source = toManifestSource(
      ref,
      "sbv2.Pipeline.fromPretrained",
      'SOURCES["sbv2-jvnv"] (@karume/models/sbv2)',
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
    // The family's checks run in the admission hook (before fetching the weight parts —
    // `hub/components.ts`).
    const { admitted, assets, open } = await loadContainerComponents(
      "sbv2.Pipeline.fromPretrained",
      loaded,
      selection,
      COMPONENT_KEYS,
      async (open) => {
        const admitted = admitSbv2(loaded.manifest, open, buildOptions);
        // The `requiredLimits` the distribution declares are checked here **before fetching the
        // weight parts** (ADR 0089 decision 5 — the limits of a shared GPU, or the adapter's
        // measured values on the path that acquires its own).
        await assertRequiredLimitsBeforeDownload(
          admitted.quant.requiredLimits,
          buildOptions.gpu,
          `sbv2.Pipeline: quant '${admitted.quantName}'`,
        );
        return admitted;
      },
      {
        ...hubOptions,
        ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
        ...(options.components === undefined ? {} : { components: options.components }),
      },
    );
    return new Pipeline(await buildSbv2State(admitted, assets, open, buildOptions));
  }

  /**
   * Builds from a fetched manifest + assets. The contract check, the interpretation of the assets
   * and `openContainer` are all done here, and **no Session is created** ({@link openSbv2State}).
   *
   * Component keys **accept both forms** — the bytes of the single form `krm` as one entry
   * (`voice`), and the part sequence of the split form (`voice[0]` / `voice[1]` / … — in index
   * order from part 0). The split form is opened as the part sequence without concatenating the
   * bytes. Missing indices and mixing with a single-form key fail loudly (the receiving
   * implementation is the one in `src/hub/components.ts`).
   */
  static async fromAssets(input: Assets, options: PipelineOptions = {}): Promise<Pipeline> {
    return new Pipeline(await openSbv2State(input, await assetOpener(input.assets), options));
  }

  /**
   * Generates speech from one utterance. Text analysis (readings, accents) is **the caller's
   * responsibility**, and this accepts only an analysed {@link Utterance} (the module doc's
   * MUST).
   *
   * The same seed and the same knobs give the same waveform (the random numbers and the host glue
   * are deterministic — `host/random.ts`).
   *
   * When called concurrently, calls **wait and run in order** (no simultaneous residency of graphs
   * — the module doc).
   */
  async generate(utterance: Utterance, options: GenerateOptions = {}): Promise<GeneratedAudio> {
    // The disposed check is made at call time (checked inside the chain, it would also take down
    // generations accepted before the dispose).
    if (this.#disposal !== undefined) {
      throw new Error("sbv2.Pipeline: cannot generate after dispose");
    }
    return await this.#chain(async () => {
      const { sampleRate, audio } = await synthesizeSbv2(this.#state, utterance, options);
      return { sampleRate, data: audio };
    });
  }

  /**
   * Releases. Destroys **only an internally acquired GPU** (a GpuContext passed via `options.gpu`
   * belongs to the caller and is not touched).
   *
   * MUST: wait for in-flight generations to complete before destroying (flush-before-destroy) —
   * putting the destruction on the chain too fixes the order of waiting and destroying in one
   * place. The second and later calls return the same completion (returning earlier would let the
   * caller treat it as "already destroyed" and move on).
   */
  dispose(): Promise<void> {
    this.#disposal ??= this.#chain(() => closeSbv2State(this.#state));
    return this.#disposal;
  }

  /** `await using` support (Explicit Resource Management) — an alias of {@link dispose}. */
  [Symbol.asyncDispose](): Promise<void> {
    return this.dispose();
  }
}
