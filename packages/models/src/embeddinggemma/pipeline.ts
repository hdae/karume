/**
 * `Pipeline` — the one surface for text (and image and audio) → sentence embedding of the
 * EmbeddingGemma family (today EmbeddingGemma 2 — distribution `karume-embeddinggemma-2`, model
 * `text`, pipeline `embeddinggemma/1`; its vision tower is the optional weights role `vision`, its
 * audio tower the optional weights role `audio`).
 *
 * Per call (every step in Karume, no torch):
 *
 * 1. Host: the input's content parts → the token ids, `<bos>`, the prompt's prefix and each run of
 *    text tokenized as one string (the distribution's tokenizer asset, the Gemma tokenizer of
 *    `src/gemma/text/`), each image and each audio as its marker block, `<eos>`; the limit and the
 *    placeholder refusal (`./input.ts`). Each image is read at issue into the tower's patch rows
 *    (`./vision.ts` — resize, patches in block order, positions), each audio into its log-mel
 *    frames (`./audio.ts` — the front end in f64, the frames padded to `4n`, the frame mask)
 * 2. Only with images or audios, one tower run per block, in sequence order: `vision` per image
 *    (`pixels` `[1, 9n, 768]`, `position_x` / `position_y` `[1, 9n]` → the soft tokens
 *    `[1, n, 512]`) and `audio` per audio (`features` `[1, 4n, 128]`, `frame_mask` `[1, 4n]` → the
 *    soft tokens `[1, n, 512]`), each read back and written into `soft_embeds` at its block's soft
 *    rows, with `soft_mask` set there
 * 3. Runs `model` once (`input_ids` `[1, T]`, `pool_mask` `[1, T]` of ones, `soft_embeds`
 *    `[1, T, H]` and `soft_mask` `[1, T, 1]` — zeros and all false for a text) → the unit vector
 *    `[1, D]` (the graph holds the text tower, the masked mean pooling and the L2 normalisation; a
 *    soft row runs on its soft token instead of its token's lookup, and the placeholder id stays
 *    under it)
 * 4. Host: a shorter width is the leading slice re-normalised (`truncateEmbedding`)
 *
 * One sequence per call, no padding and no batch (the graph's sliding window is a band declared on
 * its attention, with no run-time mask — docs/limitations.md); the images and the audios of a call
 * run one at a time. Concurrent calls queue on the serialization chain.
 *
 * ## The towers are chosen at construction ({@link PipelineOptions.vision} / {@link PipelineOptions.audio})
 *
 * The Gemma family's precedent for its optional image embedder: `vision: true` fetches the
 * distribution's `vision` weights too and opens their Session beside the text one; without it not a
 * byte of them is fetched, image parts are refused with a `ModelInputError` that says how to build
 * a pipeline that takes them, and a text call is what it was. With it, a text call runs no tower
 * and binds the same inputs as without it. `audio: true` does the same for the `audio` weights and
 * audio parts; the two may be on together.
 *
 * ## MUST: the Sessions are opened at construction and kept until `dispose`
 *
 * Embedding is normally called many times; opening a Session per call would upload the whole
 * weights each time (the same residency as siglip2 / umt5).
 *
 * MUST: it still goes on the serialization chain — concurrent `embed` calls would share one Session
 * between two runs, and `dispose` would destroy the GPU without waiting for in-flight work (a
 * flush-before-destroy violation). The chain, the Sessions and the release live in {@link Embedder},
 * the part a host test drives with stand-in Sessions; `Pipeline` adds construction only.
 *
 * ## The device's token limit
 *
 * The graph's intermediates grow with the token length T, and the largest must fit one storage
 * buffer (EmbeddingGemma 2: the per-layer-embedding projection `[1, T, 12288]` f32 —
 * 49,152 B per token, 384 MiB at 8,192 tokens). The graph contract reads the largest bytes per
 * token from the declared values ({@link embeddingGemmaGraphContract}); the device's limit is its
 * binding limit over it ({@link embeddingGemmaDeviceLimit} — 5,461 tokens on a device that binds
 * 256 MiB). A longer input is refused with a `ModelInputError` that names both limits, before the
 * run (the runtime's own preflight of the intermediates stays behind it); a device whose limit is
 * at least `pipelineConfig.maxTokens` never sees it. It is not an admission check: the device still
 * runs every shorter text. An image or an audio counts its soft tokens and its two markers, as
 * toward the model's limit. The vision tower's own intermediates are per patch row (at most
 * `[1, 2520, 3072]` f32 at the 280-token budget, about 31 MB) apart from its attention scores, which
 * the row-block budget below splits, so it adds no limit of its own. Nor does the audio tower: its
 * largest intermediate is its first convolution's `[1, 128, 2n, 64]` f32 (46.9 MiB at 750 soft
 * tokens, 30 s) and its attention scores are `[8n, 1, 12]` (each query sees 12 keys); a 30 s audio
 * on a device capped at 128 MiB was measured bit-identical to the uncapped run.
 *
 * ## Cancellation (`signal`)
 *
 * Construction uses {@link FromPretrainedOptions.signal} (the fetch layer — as siglip2 / umt5);
 * a call uses `EmbedOptions.signal` (checked after the turn on the chain and before the run, after
 * yielding to the event loop once — a run is indivisible). It throws `signal.reason` as is.
 *
 * ## MUST: numerical correctness is not guaranteed here
 *
 * It is guaranteed by the real-GPU e2e over the local distribution
 * (`packages/e2e/tests/e2e_embeddinggemma2_pipeline_test.ts` — the ten golden texts against the
 * torch CPU goldens of the series, the token ids against the goldens' ids, and sha256 rows per
 * environment; `packages/e2e/tests/e2e_embeddinggemma2_image_pipeline_test.ts` — five inputs
 * holding images against upstream's SentenceTransformer vectors and token ids;
 * `packages/e2e/tests/e2e_embeddinggemma2_audio_pipeline_test.ts` — four inputs holding an audio,
 * the same way), by the tokenizer's parity with upstream (`gemma4_tokenizer_test.ts`,
 * `embeddinggemma2_pipeline_test.ts`), by the host image preprocessing's byte identity with the
 * vision series' inputs (`embeddinggemma2_image_input_test.ts`) and by the audio front end's frames
 * against upstream's feature extractor's (`embeddinggemma2_audio_input_test.ts`).
 */

import {
  acquireGpu,
  type GpuContext,
  type RunOutputs,
  type Session,
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
import { createOperationChain } from "../concurrency/serial.ts";
import { isRecord } from "../config/readers.ts";
import { ModelInputError } from "../errors.ts";
import { parseGemmaTokenizerAsset } from "../gemma/text/asset.ts";
import { Tokenizer } from "../gemma/text/tokenizer.ts";
import { readAssetBuffer } from "../hub/asset-readers.ts";
import {
  assetComponentOpener,
  type ComponentOpener,
  type GraphOwner,
  loadContainerComponents,
} from "../hub/components.ts";
import {
  type FromPretrainedComponentOptions,
  type FromPretrainedHubOptions,
  hubLoadOptions,
} from "../hub/load-options.ts";
import { toManifestSource } from "../hub/repo-ref.ts";
import { disposeSteps } from "../session/dispose-steps.ts";
import {
  assertAcquiredGpuCapabilities,
  assertGpuFeaturesGranted,
  assertRequiredLimitsBeforeDownload,
  assertRequiredLimitsSatisfied,
  sessionGpuFeatures,
  toAcquireGpuOptions,
} from "../session/gpu-features.ts";
import { type FamilySessionPolicy, resolveSessionOptions } from "../session/options.ts";
import {
  EMBEDDINGGEMMA_PIPELINE_MAJOR,
  EMBEDDINGGEMMA_PIPELINE_NAME,
  type EmbeddingGemmaPipelineConfig,
  parseEmbeddingGemmaPipelineConfig,
} from "./config.ts";
import {
  AudioFrontEnd,
  type AudioTowerInput,
  type EmbeddingGemmaAudioContract,
  embeddingGemmaAudioContract,
  runAudioTower,
} from "./audio.ts";
import {
  type EmbedContentPart,
  type EmbedOptions,
  type EmbedRequest,
  type EmbedTextRules,
  embedTextRules,
  planEmbedRequest,
  type SoftBlockCounts,
  type TokenizeOptions,
  tokenCountIncludes,
  tokenizeEmbedInput,
  tokenLimitRemedy,
  truncateEmbedding,
} from "./input.ts";
import {
  type EmbeddingGemmaVisionContract,
  embeddingGemmaVisionContract,
  type ImagePatchRows,
  imagePatchRows,
  runVisionTower,
} from "./vision.ts";

/** The message prefix (the class name). */
const OWNER = "embeddinggemma.Pipeline";

/**
 * The component names in the manifest's weights table (also the roles the replacement option
 * accepts — the graph names the dist recipe writes): the text graph, the vision tower, fetched only
 * when {@link PipelineOptions.vision} asks for it, and the audio tower, fetched only when
 * {@link PipelineOptions.audio} does.
 */
const MODEL = "model";
const VISION = "vision";
const AUDIO = "audio";

/** The towers a pipeline is built with. */
type Towers = { readonly vision: boolean; readonly audio: boolean };

/** This family's weights components for a pipeline with the towers asked for. */
const componentKeys = ({ vision, audio }: Towers): readonly string[] => [
  MODEL,
  ...(vision ? [VISION] : []),
  ...(audio ? [AUDIO] : []),
];

/** The key of the tokenizer asset (format `karume-gemma-tokenizer/1`). */
const TOKENIZER = "tokenizer";

/** The graph inputs (`GRAPH_INPUTS` of `tools/export-recipes/embeddinggemma2/export.py`). */
const INPUT_IDS = "input_ids";
const POOL_MASK = "pool_mask";
/**
 * The soft rows of a tower: a row where `soft_mask` is set runs on its `soft_embeds` row instead of
 * its token's scaled lookup (the recipe's module doc, "Soft tokens from a tower"). A text binds no
 * soft row — zeros and all false, which the graph computes as the text forward bit for bit.
 */
const SOFT_EMBEDS = "soft_embeds";
const SOFT_MASK = "soft_mask";
/** Every input the host binds, in the graph's order. */
const GRAPH_INPUTS = [INPUT_IDS, POOL_MASK, SOFT_EMBEDS, SOFT_MASK] as const;

/** The op that cuts a folded constant down to the bound length (ADR 0010). */
const SYM_PREFIX_SLICE = "sym_prefix_slice";

/**
 * The row-block budget of the attention scores — the Session's `attentionRowBlockMaxBytes`, a fixed
 * value of this family (not a caller option).
 *
 * Why: every attention node materialises its scores S, heads × T × T f32 (4 × T² × 4 bytes: 64 MiB
 * at 2,048 tokens, 1 GiB at 8,192), and without a budget the device's binding limit alone decides
 * the split — one block on a device that binds 2 GiB, so the 8,192-token plan holds a 1 GiB score
 * buffer. With this budget S runs in row blocks of at most 128 MiB. Row blocks only reassign which
 * rows a dispatch computes (ADR 0124), so the output is bit-identical to the unbudgeted run: the
 * runtime's row-block parity tests check that as u32 identity, and on 2026-11-22 the budgeted
 * outputs of the golden cases that split (3,062, 5,935 and 8,192 tokens) were measured identical,
 * word for word, to the series' unbudgeted Session (the tolerance note of the pipeline e2e). The
 * sha256 rows of the pipeline e2e pin the budgeted output itself; they do not compare it with an
 * unbudgeted run.
 *
 * Measured on the RTX 3080 Ti under Node (2026-11-22, the bf16 mirror, two rounds of each arm;
 * second run and fastest of three repeats, the plan backing, and the estimate's peak of accounted
 * bytes with the 565 MiB of weights):
 *
 * | T     | Budget  | Second run       | Fastest repeat   | Plan backing | Peak accounted | Attention dispatches |
 * | ----- | ------- | ---------------- | ---------------- | ------------ | -------------- | -------------------- |
 * | 2,048 | none    | 152 / 139 ms     | 114 / 114 ms     | 236.1 MiB    | 821.1 MiB      | 72                   |
 * | 2,048 | 128 MiB | 134 / 137 ms     | 113 / 114 ms     | 236.1 MiB    | 821.1 MiB      | 72                   |
 * | 2,048 | 64 MiB  | 141 / 137 ms     | 115 / 113 ms     | 236.1 MiB    | 821.1 MiB      | 72                   |
 * | 2,048 | 32 MiB  | 144 / 141 ms     | 117 / 117 ms     | 236.0 MiB    | 821.1 MiB      | 144                  |
 * | 8,192 | none    | 922 / 949 ms     | 909 / 881 ms     | 1,648.3 MiB  | 2,213.4 MiB    | 72                   |
 * | 8,192 | 128 MiB | 978 / 983 ms     | 927 / 940 ms     | 944.0 MiB    | 1,509.2 MiB    | 576                  |
 * | 8,192 | 64 MiB  | 1,045 / 1,055 ms | 989 / 1,007 ms   | 912.0 MiB    | 1,477.2 MiB    | 1,152                |
 * | 8,192 | 32 MiB  | 1,282 / 1,404 ms | 1,207 / 1,226 ms | 912.0 MiB    | 1,477.2 MiB    | 2,304                |
 *
 * 128 MiB takes 704 MiB off the 8,192-token plan for 2–7% of its time and leaves 2,048 tokens
 * unchanged (S fits one block). Halving it to 64 MiB takes only 32 MiB more, for 8–14%: what is left
 * (about 912 MiB) is the per-layer-embedding intermediates, which no attention budget moves, and
 * 32 MiB moves nothing more for 35–40%. 128 MiB is also the WebGPU default binding limit, so the
 * split is the same on every device that binds at least that much.
 *
 * The vision tower's Session takes the same budget: its scores are 12 heads × (9n)² f32 — 290.7 MiB
 * at the 280-token budget. Measured on the RTX 3080 Ti under Node (2026-11-22, an image of 280 soft
 * tokens, the bf16 tower — 320.3 MiB of weights): plan backing 135.1 MiB with the budget against
 * 329.1 MiB without it, the last two of four runs 131 / 131 ms against 125 / 128 ms. The audio
 * tower's Session takes it too (the settings of every Session are one value); its scores, `[8n, 1,
 * 12]`, never reach it.
 */
const ATTENTION_ROW_BLOCK_BYTES = 128 * 2 ** 20;

/** Construction options (shared by {@link Pipeline.fromAssets} / {@link Pipeline.fromPretrained}). */
export type PipelineOptions = {
  /**
   * Shares an existing GPU. **The side that passes it owns it**, so {@link Pipeline.dispose} does
   * not destroy it. When omitted, the pipeline calls `acquireGpu` internally and destroys it in
   * `dispose()`.
   */
  readonly gpu?: GpuContext;
  /** The model (a key of the manifest's models). `defaultModel` when omitted. */
  readonly model?: string;
  /** The execution configuration (a key of that model's quants). `defaultQuant` when omitted. */
  readonly quant?: string;
  /**
   * **Builds the vision tower alongside** (not built when omitted or `false` — the option name and
   * the manifest shape of the Gemma family's image embedder).
   *
   * When `true`, the distribution's `vision` weights are fetched as well and one more Session is
   * opened; `embed` and `tokenize` then take image parts (`{ type: "image", image }`, decoded RGB8).
   * A model whose `pipelineConfig` declares no `vision`, or whose graphs do not match it, fails at
   * construction, before any weight part is fetched. A non-boolean is a `ModelInputError`.
   */
  readonly vision?: boolean;
  /**
   * **Builds the audio tower alongside** (not built when omitted or `false`), as {@link vision}
   * does for images; both may be on.
   *
   * When `true`, the distribution's `audio` weights are fetched as well and one more Session is
   * opened; `embed` and `tokenize` then take audio parts (`{ type: "audio", audio }`, mono f32
   * samples at 16,000 Hz, at most 30 s). A model whose `pipelineConfig` declares no `audio`, or
   * whose graphs do not match it, fails at construction, before any weight part is fetched. A
   * non-boolean is a `ModelInputError`.
   */
  readonly audio?: boolean;
  /**
   * An observation hook that receives the diagnostics of every `model` run (one embedding = one
   * `model` run; the towers' runs are not reported, as the Gemma family's image embedder's are
   * not). For per-op GPU time (`lastRunTiming`) pass `acquireGpu({ gpuTiming: true })` as `gpu`
   * (ADR 0021 — no timing by default).
   *
   * NOTE: no component name is passed — only the one graph reports (as in siglip2).
   *
   * Exceptions from the callback are not swallowed (fail loudly — the embedding fails with it).
   */
  readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics) => void;
};

/**
 * Fetch-layer options used only by {@link Pipeline.fromPretrained} (passed through to the hub).
 * The component replacement option (`components`) has the role `model`, `vision` on a pipeline
 * built with {@link PipelineOptions.vision} and `audio` on one built with
 * {@link PipelineOptions.audio}.
 *
 * NOTE: the description of the fetch-layer knobs, including that `headers` / `fetch` / `caches` /
 * `onRetry` are **for HTTP sources only**, lives in one place: {@link FromPretrainedHubOptions}.
 */
export type FromPretrainedOptions = PipelineOptions &
  FromPretrainedHubOptions &
  FromPretrainedComponentOptions & {
    /** Cancels fetching (the manifest, the weights, the tokenizer asset). Not carried past construction. */
    readonly signal?: AbortSignal;
  };

/**
 * The input for building directly from an already-fetched manifest + assets (a shape that accepts
 * the hub's `fetchAssets` result as is): the component `model` (the key `model` of a single-form
 * `krm`, or the part list `model[0]` / `model[1]` / …), the tokenizer asset `tokenizer`, and — for
 * {@link PipelineOptions.vision} / {@link PipelineOptions.audio} — the component `vision` / `audio`
 * in the same forms.
 */
export type Assets = {
  readonly manifest: Manifest;
  readonly assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>;
};

/**
 * The execution knobs this family accepts (the manifest's quant declaration): none. Each of the
 * distribution's presets runs its storage (the checkpoint's bf16 weights, or the same weights
 * stored as per-row int8) with f32 compute and declares no knob; a declaration is accepted only
 * after it is checked against the reference values, never passed to the runtime silently.
 *
 * NOTE: the `export` is for the tests that cover the acceptance tables of every family and pin the
 * equivalence of the distribution mirror (not on `mod.ts` / the subpath surface — ADR 0008).
 */
export const EMBEDDINGGEMMA_SESSION_POLICY: FamilySessionPolicy = {
  numerics: false,
  linearCompute: false,
  convCompute: false,
  attentionCompute: false,
  attentionScoreStorage: false,
  packedStaticQuantize: false,
  integerGemv: false,
};

/** The graph contract checked at admission. */
export type EmbeddingGemmaGraphContract = {
  /** The name of the one output (`[1, width]`). */
  readonly output: string;
  /** The output width (= `pipelineConfig.dimensions[0]`). */
  readonly width: number;
  /** The row count of the embedding table `input_ids` indexes. */
  readonly vocabularySize: number;
  /** The width of a soft row (= the embedding table's width — a soft row replaces a scaled lookup). */
  readonly softWidth: number;
  /** The bytes per token of the graph's largest symbolic value (module doc "The device's token limit"). */
  readonly bytesPerToken: number;
};

/** Bytes per element (every semantic dtype is 4 bytes — ADR 0009). */
const BYTES_PER_ELEMENT = 4;

/**
 * The bytes per token of the largest value declared with `symbol`.
 *
 * MUST: every symbolic value holds `symbol` on exactly one axis, the rest static — a value with the
 * length on two axes (a `[T, T]` mask or score) grows with T², and the per-token limit would not be
 * its; that fails loudly here. A sliding window's mask is declared as the attention's band instead
 * (`karume.band.to_band_form` in the recipe), which leaves no `[T, T]` value.
 */
const graphBytesPerToken = (model: GraphOwner, symbol: string): number => {
  let largest = 0;
  for (const [name, value] of Object.entries(model.graph.values)) {
    const symbolic = value.shape.filter((dim) => typeof dim !== "number");
    if (symbolic.length === 0) continue;
    if (symbolic.length !== 1 || symbolic[0] !== symbol) {
      throw new Error(
        `${OWNER}: graph value '${name}' is [${value.shape.join(", ")}] — every symbolic value must` +
          ` hold '${symbol}' on exactly one axis (the device's token limit is read per token; a` +
          ` ${symbol} × ${symbol} mask must be declared as the attention's band instead)`,
      );
    }
    const elements = value.shape.reduce<number>(
      (product, dim) => (typeof dim === "number" ? product * dim : product),
      1,
    );
    largest = Math.max(largest, elements * BYTES_PER_ELEMENT);
  }
  return largest;
};

/** The device limits the token limit is read from (`GpuContext.limits`). */
export type EmbeddingGemmaDeviceLimits = {
  readonly maxStorageBufferBindingSize: number;
  readonly maxBufferSize: number;
};

/** The longest input one device runs (module doc "The device's token limit"). */
export type EmbeddingGemmaDeviceLimit = {
  /** The most tokens of one input, `<bos>` and `<eos>` included. */
  readonly maxTokens: number;
  /** The bytes per token of the graph's largest intermediate. */
  readonly bytesPerToken: number;
  /** The one storage buffer it must fit (the smaller of the binding and the buffer limit). */
  readonly bindingBytes: number;
};

/**
 * The device's token limit for the graph of `contract`: its binding limit over the largest bytes per
 * token.
 *
 * NOTE: the `export` is for the host tests (not on `mod.ts` / the subpath surface — ADR 0008).
 */
export const embeddingGemmaDeviceLimit = (
  contract: Pick<EmbeddingGemmaGraphContract, "bytesPerToken">,
  limits: EmbeddingGemmaDeviceLimits,
): EmbeddingGemmaDeviceLimit => {
  const bindingBytes = Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize);
  return {
    maxTokens: Math.floor(bindingBytes / contract.bytesPerToken),
    bytesPerToken: contract.bytesPerToken,
    bindingBytes,
  };
};

/** Refuses an input longer than the device runs (nothing is truncated). */
const assertDeviceTokenLimit = (
  count: number,
  limit: EmbeddingGemmaDeviceLimit,
  blocks: SoftBlockCounts,
): void => {
  if (count <= limit.maxTokens) return;
  throw new ModelInputError(
    `${OWNER}: the input is ${count} tokens (${tokenCountIncludes(blocks)}), over this device's limit of` +
      ` ${limit.maxTokens} — the graph's largest intermediate (${limit.bytesPerToken} B per token)` +
      ` must fit one storage buffer of ${limit.bindingBytes} B (the device's` +
      " maxStorageBufferBindingSize / maxBufferSize); nothing is truncated — " +
      tokenLimitRemedy(blocks),
  );
};

/** A graph input by name (fails loudly when absent). */
const graphInput = (model: GraphOwner, name: string): GraphOwner["graph"]["inputs"][number] => {
  const spec = model.graph.inputs.find((input) => input.name === name);
  if (spec === undefined) {
    const declared = model.graph.inputs.map((input) => input.name).join(" / ");
    throw new Error(`${OWNER}: graph input '${name}' is missing (declared: ${declared})`);
  }
  return spec;
};

/** Whether a declared shape is exactly `expected` (rank and every axis). */
const sameShape = (
  shape: readonly (number | string)[],
  expected: readonly (number | string)[],
): boolean =>
  shape.length === expected.length && shape.every((dim, axis) => dim === expected[axis]);

/** One integer field of a `sym_prefix_slice` slice (read through `Object.hasOwn`). */
const sliceField = (slice: Record<string, unknown>, key: string, where: string): number => {
  const value = Object.hasOwn(slice, key) ? slice[key] : undefined;
  if (typeof value !== "number" || !Number.isInteger(value)) {
    throw new Error(`${where}: slice field '${key}' is not an integer (${String(value)})`);
  }
  return value;
};

/**
 * The largest value of `symbol` the graph can run: every `sym_prefix_slice` on it cuts a constant
 * baked at the maximum (ADR 0010), so the bound is the smallest `(extent − offset) / coeff` over its
 * slices. `undefined` when no such node exists.
 *
 * NOTE: the IR declares no range for a symbol (`docs/ir-v2.md`); a folded constant is the one place
 * the bound is written, and a length past it fails at the prefix slice when the run is planned.
 */
const graphSymbolMaximum = (model: GraphOwner, symbol: string): number | undefined => {
  const { graph } = model;
  let maximum: number | undefined;
  for (const node of graph.nodes) {
    if (node.op !== SYM_PREFIX_SLICE) continue;
    const sym = Object.hasOwn(node.attrs, "sym") ? node.attrs["sym"] : undefined;
    if (sym !== symbol) continue;
    const where = `${OWNER}: the ${SYM_PREFIX_SLICE} of '${node.ins[0]}'`;
    const source = Object.hasOwn(graph.values, node.ins[0]) ? graph.values[node.ins[0]] : undefined;
    const slices = Object.hasOwn(node.attrs, "slices") ? node.attrs["slices"] : undefined;
    if (source === undefined || !Array.isArray(slices)) {
      throw new Error(`${where} has no declared source or no slices`);
    }
    const entries: readonly unknown[] = slices;
    for (const slice of entries) {
      if (!isRecord(slice)) throw new Error(`${where}: a slice is not an object`);
      const dim = sliceField(slice, "dim", where);
      const extent = source.shape[dim];
      if (typeof extent !== "number") {
        throw new Error(`${where}: axis ${dim} of the source is not static (${String(extent)})`);
      }
      const bound = Math.floor(
        (extent - sliceField(slice, "offset", where)) / sliceField(slice, "coeff", where),
      );
      maximum = maximum === undefined ? bound : Math.min(maximum, bound);
    }
  }
  return maximum;
};

/**
 * Matches the graph declaration against what the host feeds and reads, and against
 * `pipelineConfig`.
 *
 * MUST: call it at admission (before the weight parts are fetched): the host builds the four inputs
 * from its own token ids and reads one `[1, D]` output, so a mismatch would otherwise surface after
 * the weights are uploaded, in the Session's shape check.
 *
 * - One symbolic length T; exactly the inputs `input_ids` i32 `[1, T]`, `pool_mask` f32 `[1, T]`,
 *   `soft_embeds` f32 `[1, T, W]` and `soft_mask` bool `[1, T, 1]`, in that order.
 * - One f32 output `[1, D]` with D = `pipelineConfig.dimensions[0]` (the full width a caller gets).
 * - Exactly one `embedding` node reads `input_ids`, its table `[V, W]` static (V bounds the token
 *   ids the tokenizer may produce — checked when the asset arrives; W is the soft rows' width).
 * - The graph's maximum of T (read from its folded constants) equals `pipelineConfig.maxTokens`:
 *   a smaller declaration would refuse lengths the graph runs, a larger one would let a length
 *   through that fails only when the run is planned.
 * - Every symbolic value holds T on one axis; the largest of them gives the bytes per token the
 *   device's token limit is read from.
 *
 * NOTE: the `export` is for the host tests (not on `mod.ts` / the subpath surface — ADR 0008).
 */
export const embeddingGemmaGraphContract = (
  model: GraphOwner,
  config: EmbeddingGemmaPipelineConfig,
): EmbeddingGemmaGraphContract => {
  const { graph } = model;
  if (graph.symbols.length !== 1) {
    throw new Error(
      `${OWNER}: the graph's symbolic dimensions are [${graph.symbols.join(", ")}] (one token length is required)`,
    );
  }
  const [symbol] = graph.symbols;
  for (const name of GRAPH_INPUTS) graphInput(model, name);
  if (graph.inputs.length !== GRAPH_INPUTS.length) {
    throw new Error(
      `${OWNER}: the graph has ${graph.inputs.length} inputs (the host feeds only` +
        ` ${GRAPH_INPUTS.map((name) => `'${name}'`).join(", ")})`,
    );
  }
  if (graph.inputs.some((input, index) => input.name !== GRAPH_INPUTS[index])) {
    throw new Error(
      `${OWNER}: the graph's inputs are ${graph.inputs.map((input) => input.name).join(" / ")},` +
        ` not ${GRAPH_INPUTS.join(" / ")} in that order`,
    );
  }
  if (graph.outputs.length !== 1) {
    throw new Error(
      `${OWNER}: the graph has ${graph.outputs.length} outputs (one [1, D] is required)`,
    );
  }
  const [output] = graph.outputs;
  if (!Object.hasOwn(graph.values, output)) {
    throw new Error(`${OWNER}: graph output '${output}' is not declared`);
  }
  const value = graph.values[output];
  const width = config.dimensions[0];
  if (value.dtype !== "f32" || !sameShape(value.shape, [1, width])) {
    throw new Error(
      `${OWNER}: graph output '${output}' is ${value.dtype} [${value.shape.join(", ")}], expected f32` +
        ` [1, ${width}] (the full width pipelineConfig.dimensions declares)`,
    );
  }
  const lookups = graph.nodes.filter(
    (node) => node.op === "embedding" && node.ins[1] === INPUT_IDS,
  );
  if (lookups.length !== 1) {
    throw new Error(
      `${OWNER}: the graph has ${lookups.length} embedding nodes that read '${INPUT_IDS}' (expected 1)`,
    );
  }
  const table = lookups[0].ins[0];
  const tableShape = Object.hasOwn(graph.values, table) ? graph.values[table].shape : undefined;
  const vocabularySize = tableShape?.[0];
  const softWidth = tableShape?.[1];
  if (
    tableShape?.length !== 2 ||
    typeof vocabularySize !== "number" ||
    !Number.isInteger(vocabularySize) ||
    vocabularySize <= 0 ||
    typeof softWidth !== "number" ||
    !Number.isInteger(softWidth) ||
    softWidth <= 0
  ) {
    throw new Error(
      `${OWNER}: the embedding table '${table}' has shape [${tableShape?.join(", ") ?? "undeclared"}]` +
        " (expected a static [V, W])",
    );
  }
  for (const [name, dtype, shape] of [
    [INPUT_IDS, "i32", [1, symbol]],
    [POOL_MASK, "f32", [1, symbol]],
    [SOFT_EMBEDS, "f32", [1, symbol, softWidth]],
    [SOFT_MASK, "bool", [1, symbol, 1]],
  ] as const) {
    const spec = graphInput(model, name);
    if (spec.dtype !== dtype || !sameShape(spec.shape, shape)) {
      throw new Error(
        `${OWNER}: graph input '${name}' is ${spec.dtype} [${spec.shape.join(", ")}], expected` +
          ` ${dtype} [${shape.join(", ")}]`,
      );
    }
  }
  const maximum = graphSymbolMaximum(model, symbol);
  if (maximum === undefined) {
    throw new Error(
      `${OWNER}: no ${SYM_PREFIX_SLICE} bounds '${symbol}', so the graph's maximum token length cannot` +
        " be read to check pipelineConfig.maxTokens against",
    );
  }
  if (maximum !== config.maxTokens) {
    throw new Error(
      `${OWNER}: pipelineConfig.maxTokens is ${config.maxTokens}, but the graph runs '${symbol}' up to` +
        ` ${maximum} (the bound of its folded constants)`,
    );
  }
  return {
    output,
    width,
    vocabularySize,
    softWidth,
    bytesPerToken: graphBytesPerToken(model, symbol),
  };
};

/** The part of a Session the embedder uses (narrow, so a host test can stand in for the GPU). */
export type EmbeddingSession = Pick<Session, "run" | "diagnostics" | "dispose">;

/** The vision tower of an {@link Embedder} (its Session and what it reads of the graph). */
export type EmbedderTower = {
  /** The Session of `vision`, opened once; the embedder disposes it. */
  readonly session: EmbeddingSession;
  readonly contract: EmbeddingGemmaVisionContract;
};

/** The audio tower of an {@link Embedder} (its Session and what it reads of the graph). */
export type EmbedderAudioTower = {
  /** The Session of `audio`, opened once; the embedder disposes it. */
  readonly session: EmbeddingSession;
  readonly contract: EmbeddingGemmaAudioContract;
};

/** What an {@link Embedder} runs with (all of it checked before it is built). */
export type EmbedderParts = {
  /** The Session of `model`, opened once; the embedder disposes it. */
  readonly session: EmbeddingSession;
  readonly contract: EmbeddingGemmaGraphContract;
  /**
   * The rules of the input; `rules.images` is set exactly when `tower` is, `rules.audio` exactly
   * when `audioTower` is.
   */
  readonly rules: EmbedTextRules;
  /** The vision tower ({@link PipelineOptions.vision}); absent, image parts are refused. */
  readonly tower?: EmbedderTower;
  /** The audio tower ({@link PipelineOptions.audio}); absent, audio parts are refused. */
  readonly audioTower?: EmbedderAudioTower;
  /**
   * The limits of the Session's device; the embedder derives the longest input it runs from them and
   * `contract.bytesPerToken` ({@link embeddingGemmaDeviceLimit}), so the bytes per token have one
   * source.
   */
  readonly deviceLimits: EmbeddingGemmaDeviceLimits;
  /**
   * Runs after the Session is disposed, even if that fails (the pipeline destroys a GPU it
   * acquired itself here).
   */
  readonly release?: () => void;
  readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics) => void;
};

/** Reads the unit vector `[1, width]` from a run's outputs (fails loudly on anything else). */
const readEmbedding = (
  outputs: RunOutputs,
  contract: EmbeddingGemmaGraphContract,
): Float32Array<ArrayBuffer> => {
  const { output, width } = contract;
  if (!Object.hasOwn(outputs, output)) {
    throw new Error(`${OWNER}: graph output '${output}' is missing from the run results`);
  }
  const tensor = outputs[output];
  if (tensor.dtype !== "f32") {
    throw new Error(`${OWNER}: graph output '${output}' is ${tensor.dtype}, not f32`);
  }
  if (!sameShape(tensor.shape, [1, width]) || tensor.data.length !== width) {
    throw new Error(
      `${OWNER}: graph output '${output}' is [${tensor.shape.join(", ")}] with ${tensor.data.length}` +
        ` elements, not [1, ${width}]`,
    );
  }
  // MUST: never return a non-finite vector — a caller ranking by dot product would only see
  // a wrong order.
  const broken = tensor.data.findIndex((element) => !Number.isFinite(element));
  if (broken !== -1) {
    throw new Error(
      `${OWNER}: output element ${broken} is non-finite (${String(tensor.data[broken])})`,
    );
  }
  return tensor.data;
};

/**
 * One soft block of a run: its soft rows' first row and its tower's input (built at issue) — an
 * image's patch rows or an audio's frames.
 */
type TowerRun =
  | { readonly kind: "image"; readonly start: number; readonly rows: ImagePatchRows }
  | { readonly kind: "audio"; readonly start: number; readonly input: AudioTowerInput };

/**
 * The resident Sessions of one model with their serialization chain: plans each request on the
 * host (`planEmbedRequest`), runs it in turn — a tower once per image or audio, in sequence order,
 * then the text graph — and disposes the Sessions (then `release`) after the work in flight.
 *
 * NOTE: internal — {@link Pipeline} is the public entry and the only one that builds it from a
 * distribution; the `export` is for the host test that drives it with stand-in Sessions (not on
 * `mod.ts` / the subpath surface — ADR 0008).
 */
export class Embedder {
  readonly #parts: EmbedderParts;
  /** The longest input the Session's device runs (derived once from the parts). */
  readonly #deviceLimit: EmbeddingGemmaDeviceLimit;
  /** The serialization chain of embed and dispose (module doc). */
  readonly #chain = createOperationChain();
  /** The single dispose. **Not undefined means "disposed"** (no separate flag to keep in step). */
  #disposal: Promise<void> | undefined;
  /** The audio front end's tables, built on the first audio (`./audio.ts`). */
  #frontEnd: AudioFrontEnd | undefined;

  constructor(parts: EmbedderParts) {
    if ((parts.tower === undefined) !== (parts.rules.images === undefined)) {
      throw new Error(
        `${OWNER}: internal inconsistency: the image rules and the vision tower are not built together`,
      );
    }
    if ((parts.audioTower === undefined) !== (parts.rules.audio === undefined)) {
      throw new Error(
        `${OWNER}: internal inconsistency: the audio rules and the audio tower are not built together`,
      );
    }
    if (parts.tower !== undefined && parts.tower.contract.width !== parts.contract.softWidth) {
      throw new Error(
        `${OWNER}: the vision graph's soft tokens are ${parts.tower.contract.width} wide, the text` +
          ` graph's soft rows ${parts.contract.softWidth}`,
      );
    }
    if (
      parts.audioTower !== undefined &&
      parts.audioTower.contract.width !== parts.contract.softWidth
    ) {
      throw new Error(
        `${OWNER}: the audio graph's soft tokens are ${parts.audioTower.contract.width} wide, the` +
          ` text graph's soft rows ${parts.contract.softWidth}`,
      );
    }
    this.#parts = parts;
    this.#deviceLimit = embeddingGemmaDeviceLimit(parts.contract, parts.deviceLimits);
  }

  /**
   * Embeds one input (module doc). A refused request is a `ModelInputError` before the turn on the
   * chain; concurrent calls wait and run in call order.
   */
  async embed(
    input: string | readonly EmbedContentPart[],
    options: EmbedOptions = {},
  ): Promise<Float32Array<ArrayBuffer>> {
    // Checked at call time (checking inside the chain would also fail the calls accepted before
    // dispose).
    if (this.#disposal !== undefined) throw new Error(`${OWNER}: cannot embed after dispose`);
    // Refusals return without waiting for the turn (the plan is pure).
    const request = planEmbedRequest(this.#parts.rules, input, options);
    assertDeviceTokenLimit(request.ids.length, this.#deviceLimit, {
      images: request.images.length,
      audios: request.audios.length,
    });
    // The images and the audios are read here, at issue: nothing written into their pixels or
    // samples afterwards reaches this call (the Gemma family's rule for its image rows).
    const budget = this.#parts.rules.images?.softTokenBudget;
    const images: TowerRun[] = request.images.map(({ image, start, softTokens }) => {
      if (budget === undefined) {
        throw new Error(
          `${OWNER}: internal inconsistency: an image was planned without image rules`,
        );
      }
      const rows = imagePatchRows(image, budget);
      if (rows.softTokens !== softTokens) {
        throw new Error(
          `${OWNER}: internal inconsistency: the image's ${rows.softTokens} patch blocks are not the` +
            ` ${softTokens} soft tokens its sequence holds`,
        );
      }
      return { kind: "image", start, rows };
    });
    const audios: TowerRun[] = request.audios.map(({ audio, start, softTokens }) => {
      this.#frontEnd ??= new AudioFrontEnd();
      const frames = this.#frontEnd.towerInput(audio.data);
      if (frames.softTokens !== softTokens) {
        throw new Error(
          `${OWNER}: internal inconsistency: the audio's ${frames.softTokens} soft tokens of frames are` +
            ` not the ${softTokens} its sequence holds`,
        );
      }
      return { kind: "audio", start, input: frames };
    });
    // In sequence order: an image before an audio runs first, as it stands first.
    const runs = [...images, ...audios].sort((left, right) => left.start - right.start);
    return await this.#chain(() => this.#run(request, runs, options.signal));
  }

  /**
   * The token ids `embed` would run for this input and prompt, on the host (`tokenizeEmbedInput` —
   * no run, not on the chain); refused like `embed` after dispose.
   */
  tokenize(
    input: string | readonly EmbedContentPart[],
    options: TokenizeOptions = {},
  ): Int32Array<ArrayBuffer> {
    if (this.#disposal !== undefined) throw new Error(`${OWNER}: cannot tokenize after dispose`);
    return tokenizeEmbedInput(this.#parts.rules, input, options);
  }

  /**
   * One call: the abort check, one tower run per image or audio (its soft tokens written into the
   * soft rows), the text run, the output check, the width.
   */
  async #run(
    request: EmbedRequest,
    runs: readonly TowerRun[],
    signal: AbortSignal | undefined,
  ): Promise<Float32Array<ArrayBuffer>> {
    // A cancellation that arrived while waiting for the turn also takes effect here (a run is
    // indivisible — cancellation works only before it, and between the runs of one call).
    await settleAbort(signal);
    const { session, contract, onRunDiagnostics } = this.#parts;
    const length = request.ids.length;
    const width = contract.softWidth;
    // A text binds no soft row: zeros and all false, which the graph runs as its token lookups
    // alone (`SOFT_EMBEDS`).
    const softEmbeds = new Float32Array(length * width);
    const softMask = new Uint32Array(length);
    for (const run of runs) {
      const soft = await this.#runTower(run);
      const softTokens = run.kind === "image" ? run.rows.softTokens : run.input.softTokens;
      // Copied before the tower's next run (a readback lasts until then; its shape was checked
      // against `softTokens`). The placeholder id stays under each soft row; the mask makes the
      // graph take the soft token there, unscaled.
      softEmbeds.set(soft, run.start * width);
      softMask.fill(1, run.start, run.start + softTokens);
      signal?.throwIfAborted();
    }
    const outputs = await session.run({
      [INPUT_IDS]: { dtype: "i32", shape: [1, length], data: request.ids },
      // All ones: the pooling averages every token, the prompt's and the soft ones included
      // (upstream's `include_prompt`); the mask is not wired to attention.
      [POOL_MASK]: { dtype: "f32", shape: [1, length], data: new Float32Array(length).fill(1) },
      [SOFT_EMBEDS]: { dtype: "f32", shape: [1, length, width], data: softEmbeds },
      [SOFT_MASK]: { dtype: "bool", shape: [1, length, 1], data: softMask },
    });
    onRunDiagnostics?.(session.diagnostics());
    return truncateEmbedding(readEmbedding(outputs, contract), request.dimension);
  }

  /** One soft block's tower run: its soft tokens `[n · width]` (shape-checked by the tower's run). */
  async #runTower(run: TowerRun): Promise<Float32Array<ArrayBuffer>> {
    const { tower, audioTower } = this.#parts;
    if (run.kind === "image") {
      if (tower === undefined) {
        throw new Error(
          `${OWNER}: internal inconsistency: an image reached a run without the vision tower`,
        );
      }
      return await runVisionTower(tower.session, run.rows, tower.contract);
    }
    if (audioTower === undefined) {
      throw new Error(
        `${OWNER}: internal inconsistency: an audio reached a run without the audio tower`,
      );
    }
    return await runAudioTower(audioTower.session, run.input, audioTower.contract);
  }

  /**
   * Disposes the Sessions (the text graph's, then the vision tower's, then the audio tower's), then
   * runs `release`, after the calls in flight (flush-before-destroy — the disposal is queued on the
   * same chain). Later calls return the same completion.
   *
   * MUST: every step runs even if an earlier one fails ({@link disposeSteps}) — a GPU acquired by
   * the pipeline would otherwise never be returned.
   */
  dispose(): Promise<void> {
    this.#disposal ??= this.#chain(() =>
      disposeSteps([
        () => this.#parts.session.dispose(),
        () => this.#parts.tower?.session.dispose(),
        () => this.#parts.audioTower?.session.dispose(),
        () => this.#parts.release?.(),
      ]),
    );
    return this.#disposal;
  }
}

/**
 * A graph whose Session the embedder opens (a container component — narrow, so a host test can
 * stand in for one).
 */
export type EmbeddingGraph = {
  readonly createSession: (gpu: GpuContext, options: SessionOptions) => Promise<EmbeddingSession>;
};

/** What {@link openEmbedder} opens on the GPU in hand and builds the embedder from. */
export type EmbedderOpening = Omit<
  EmbedderParts,
  "session" | "tower" | "audioTower" | "deviceLimits"
> & {
  readonly gpu: GpuContext;
  /** The checks of the GPU in hand, run before any Session is opened. */
  readonly assertGpu: () => void;
  readonly model: EmbeddingGraph;
  /** The vision tower's graph and contract ({@link PipelineOptions.vision}). */
  readonly tower?: {
    readonly graph: EmbeddingGraph;
    readonly contract: EmbeddingGemmaVisionContract;
  };
  /** The audio tower's graph and contract ({@link PipelineOptions.audio}). */
  readonly audioTower?: {
    readonly graph: EmbeddingGraph;
    readonly contract: EmbeddingGemmaAudioContract;
  };
  /** The quant's effective settings; every Session takes them with the row-block budget. */
  readonly sessionOptions: SessionOptions;
};

/**
 * Opens the Sessions on the GPU in hand — the text graph's, then the vision tower's, then the audio
 * tower's, each with the family's row-block budget ({@link ATTENTION_ROW_BLOCK_BYTES}) — and builds
 * the {@link Embedder}.
 *
 * MUST: on any failure (the GPU checks, a Session, the embedder itself) the Sessions opened so far
 * are disposed and then `release` runs — no one else holds them, and a GPU the pipeline acquired
 * would never be returned. If that clean-up fails too, both failures are thrown in an
 * `AggregateError`; otherwise the original failure as it is.
 *
 * NOTE: the `export` is for the host test that drives it with stand-in graphs (not on `mod.ts` /
 * the subpath surface — ADR 0008).
 */
export const openEmbedder = async (opening: EmbedderOpening): Promise<Embedder> => {
  const { gpu, assertGpu, model, tower, audioTower, sessionOptions, ...parts } = opening;
  // Every Session splits the attention scores in row blocks of the family's budget (the vision
  // tower's are 12 heads × (9n)² f32, about 291 MiB unsplit at 280 soft tokens).
  const settings: SessionOptions = {
    ...sessionOptions,
    attentionRowBlockMaxBytes: ATTENTION_ROW_BLOCK_BYTES,
  };
  let session: EmbeddingSession | undefined;
  let towerSession: EmbeddingSession | undefined;
  let audioSession: EmbeddingSession | undefined;
  try {
    assertGpu();
    session = await model.createSession(gpu, settings);
    if (tower !== undefined) towerSession = await tower.graph.createSession(gpu, settings);
    if (audioTower !== undefined) {
      audioSession = await audioTower.graph.createSession(gpu, settings);
    }
    return new Embedder({
      ...parts,
      session,
      ...(towerSession === undefined || tower === undefined
        ? {}
        : { tower: { session: towerSession, contract: tower.contract } }),
      ...(audioSession === undefined || audioTower === undefined
        ? {}
        : { audioTower: { session: audioSession, contract: audioTower.contract } }),
      deviceLimits: gpu.limits,
    });
  } catch (error) {
    // The Sessions first — flush before destroy.
    try {
      await disposeSteps([
        () => session?.dispose(),
        () => towerSession?.dispose(),
        () => audioSession?.dispose(),
        () => parts.release?.(),
      ]);
    } catch (cleanup) {
      throw new AggregateError(
        [error, cleanup],
        `${OWNER}: construction failed, and so did its clean-up`,
      );
    }
    throw error;
  }
};

/** What the family admission ({@link admitEmbeddingGemma}) settles. */
type EmbeddingGemmaAdmission = {
  readonly config: EmbeddingGemmaPipelineConfig;
  readonly contract: EmbeddingGemmaGraphContract;
  readonly quantName: string;
  readonly quant: Quant;
  /** The effective settings after the quant declaration passed {@link EMBEDDINGGEMMA_SESSION_POLICY}. */
  readonly sessionOptions: SessionOptions;
  /** The quant declaration plus the features the effective settings need. */
  readonly gpuFeatures: GpuFeaturesSpec | undefined;
  /** The tower graph's contract, settled only when {@link PipelineOptions.vision} asks for it. */
  readonly vision?: EmbeddingGemmaVisionContract;
  /** The audio graph's contract, settled only when {@link PipelineOptions.audio} asks for it. */
  readonly audio?: EmbeddingGemmaAudioContract;
};

/**
 * The towers {@link PipelineOptions.vision} / {@link PipelineOptions.audio} ask for (a non-boolean
 * fails loudly).
 */
const wantedTowers = (where: string, options: PipelineOptions): Towers => {
  const asked = (name: "vision" | "audio"): boolean => {
    const value: unknown = options[name];
    if (value !== undefined && typeof value !== "boolean") {
      throw new ModelInputError(`${where}: ${name} ${JSON.stringify(value)} is not a boolean`);
    }
    return value === true;
  };
  return { vision: asked("vision"), audio: asked("audio") };
};

/**
 * The component supplier of the whole-bytes surface (`fromAssets`) (the receiving implementation is
 * shared by every family — {@link assetComponentOpener}).
 *
 * MUST: do not copy with `slice` ({@link readAssetBuffer}).
 */
const assetOpener = (assets: Assets["assets"], towers: Towers): Promise<ComponentOpener> =>
  assetComponentOpener(
    OWNER,
    assets,
    (key) => readAssetBuffer(OWNER, "weights / assets", assets, key),
    componentKeys(towers),
  );

/**
 * Checks a `vision: true` load against the declaration and the two graphs (before any weight part
 * is fetched): `pipelineConfig.vision` is declared, the tower graph matches it
 * ({@link embeddingGemmaVisionContract}), its soft tokens are the text graph's soft rows' width, and
 * the three image ids are rows of the text graph's table.
 *
 * MUST: each mismatch fails here and not on the first image — a width disagreement would otherwise
 * surface after the weights are uploaded, and an id outside the table would read another token's row.
 */
const admitVision = (
  open: ComponentOpener,
  config: EmbeddingGemmaPipelineConfig,
  contract: EmbeddingGemmaGraphContract,
): EmbeddingGemmaVisionContract => {
  const { vision } = config;
  if (vision === undefined) {
    throw new Error(
      `${OWNER}: vision was requested, but this model's pipelineConfig declares no 'vision' (its` +
        " distribution ships no vision tower)",
    );
  }
  if (vision.hiddenSize !== contract.softWidth) {
    throw new Error(
      `${OWNER}: pipelineConfig.vision.hiddenSize ${vision.hiddenSize} is not the width of the text` +
        ` graph's soft rows (${contract.softWidth})`,
    );
  }
  for (const key of ["imageTokenId", "beginImageTokenId", "endImageTokenId"] as const) {
    if (vision[key] >= contract.vocabularySize) {
      throw new Error(
        `${OWNER}: pipelineConfig.vision.${key} ${vision[key]} is outside the embedding table's` +
          ` ${contract.vocabularySize} rows`,
      );
    }
  }
  return embeddingGemmaVisionContract(open(VISION), vision);
};

/**
 * Checks an `audio: true` load against the declaration and the two graphs (before any weight part
 * is fetched): `pipelineConfig.audio` is declared, the tower graph matches it
 * ({@link embeddingGemmaAudioContract}), its soft tokens are the text graph's soft rows' width, the
 * three audio ids are rows of the text graph's table, and the graph's bound of its soft-token count
 * (its folded constants) equals `pipelineConfig.audio.maxSoftTokens`.
 *
 * MUST: each mismatch fails here and not on the first audio, for the reasons of {@link admitVision};
 * a declared bound above the graph's would let an audio through that fails only when the run is
 * planned, one below it would refuse audios the graph runs.
 */
const admitAudio = (
  open: ComponentOpener,
  config: EmbeddingGemmaPipelineConfig,
  contract: EmbeddingGemmaGraphContract,
): EmbeddingGemmaAudioContract => {
  const { audio } = config;
  if (audio === undefined) {
    throw new Error(
      `${OWNER}: audio was requested, but this model's pipelineConfig declares no 'audio' (its` +
        " distribution ships no audio tower)",
    );
  }
  if (audio.hiddenSize !== contract.softWidth) {
    throw new Error(
      `${OWNER}: pipelineConfig.audio.hiddenSize ${audio.hiddenSize} is not the width of the text` +
        ` graph's soft rows (${contract.softWidth})`,
    );
  }
  for (const key of ["audioTokenId", "beginAudioTokenId", "endAudioTokenId"] as const) {
    if (audio[key] >= contract.vocabularySize) {
      throw new Error(
        `${OWNER}: pipelineConfig.audio.${key} ${audio[key]} is outside the embedding table's` +
          ` ${contract.vocabularySize} rows`,
      );
    }
  }
  const tower = open(AUDIO);
  const audioContract = embeddingGemmaAudioContract(tower, audio);
  const maximum = graphSymbolMaximum(tower, audioContract.symbol);
  if (maximum === undefined) {
    throw new Error(
      `${OWNER}: no ${SYM_PREFIX_SLICE} of the audio graph bounds '${audioContract.symbol}', so its` +
        " maximum soft-token count cannot be read to check pipelineConfig.audio.maxSoftTokens against",
    );
  }
  if (maximum !== audio.maxSoftTokens) {
    throw new Error(
      `${OWNER}: pipelineConfig.audio.maxSoftTokens is ${audio.maxSoftTokens}, but the audio graph` +
        ` runs '${audioContract.symbol}' up to ${maximum} (the bound of its folded constants)`,
    );
  }
  return audioContract;
};

/**
 * Checks whether this manifest and this graph can run as this family (the family admission hook of
 * `hub/components.ts` — on the fetching surface it is called **before a single byte of the weight
 * parts is fetched**).
 *
 * MUST: gather the family checks in this one function (pipeline name / major, `pipelineConfig`,
 * quant, the tokenizer asset declaration, the quant's `session`, the shared GPU's capabilities, the
 * graph contract). Spread over later stages, on the fetching surface they would fail only after the
 * weights are downloaded (ADR 0070 decision 5).
 * MUST: manifest contract violations and the match against the graph fail **before the GPU is
 * acquired** (the same order as the other families).
 */
const admitEmbeddingGemma = (
  manifest: Manifest,
  open: ComponentOpener,
  options: PipelineOptions,
): EmbeddingGemmaAdmission => {
  const modelName = options.model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, modelName)) {
    throw new Error(
      `${OWNER}: model '${modelName}' is not in the manifest` +
        ` (available: ${manifest.available.models.join(" / ")})`,
    );
  }
  const entry: ModelEntry = manifest.models[modelName];
  const { name, major } = entry.pipeline;
  if (name !== EMBEDDINGGEMMA_PIPELINE_NAME) {
    throw new Error(
      `${OWNER}: the manifest's pipeline is '${name}/${major}'` +
        ` ('${EMBEDDINGGEMMA_PIPELINE_NAME}/${EMBEDDINGGEMMA_PIPELINE_MAJOR}' is required)`,
    );
  }
  if (major !== EMBEDDINGGEMMA_PIPELINE_MAJOR) {
    // The only check that stops the silent degradation of "old implementation × new repository"
    // (ADR 0038 §6).
    throw new Error(
      `${OWNER}: pipeline '${name}/${major}' has an unsupported major` +
        ` (this implementation reads ${EMBEDDINGGEMMA_PIPELINE_NAME}/${EMBEDDINGGEMMA_PIPELINE_MAJOR})`,
    );
  }
  const config = parseEmbeddingGemmaPipelineConfig(entry.pipelineConfig);

  const quantName = options.quant ?? entry.defaultQuant;
  if (!Object.hasOwn(entry.quants, quantName)) {
    throw new Error(
      `${OWNER}: quant '${quantName}' is not in the manifest` +
        ` (available: ${entry.available.quants.join(" / ")})`,
    );
  }
  const quant = entry.quants[quantName];
  // The asset's contents are checked once it arrives ({@link buildEmbedder}); a missing
  // declaration fails before the weights are fetched.
  if (!Object.hasOwn(entry.assets, TOKENIZER)) {
    throw new Error(
      `${OWNER}: the manifest assets of model '${modelName}' have no tokenizer asset '${TOKENIZER}'`,
    );
  }
  // An unsupported declaration fails before the components are opened (the one implementation
  // shared by every family).
  const sessionOptions = resolveSessionOptions(
    EMBEDDINGGEMMA_SESSION_POLICY,
    quant.session,
    {},
    `${OWNER}: quant '${quantName}'`,
  );
  const gpuFeatures = sessionGpuFeatures(quant.gpuFeatures, sessionOptions);

  const contract = embeddingGemmaGraphContract(open(MODEL), config);
  const towers = wantedTowers(OWNER, options);
  const vision = towers.vision ? admitVision(open, config, contract) : undefined;
  const audio = towers.audio ? admitAudio(open, config, contract) : undefined;

  // MUST: a shared GPU lacking a capability (feature / device limit) fails here — the only family
  // check that can tell before the weights are downloaded (the later check calls the same
  // function).
  if (options.gpu !== undefined) {
    assertGpuFeaturesGranted(gpuFeatures, options.gpu, `${OWNER}: quant '${quantName}'`);
    assertRequiredLimitsSatisfied(
      quant.requiredLimits,
      options.gpu.limits,
      `${OWNER}: quant '${quantName}'`,
    );
  }
  return {
    config,
    contract,
    quantName,
    quant,
    sessionOptions,
    gpuFeatures,
    ...(vision === undefined ? {} : { vision }),
    ...(audio === undefined ? {} : { audio }),
  };
};

/**
 * Builds the embedder from the admitted material + assets (parse and match the tokenizer asset →
 * acquire the GPU → open the Session, and the towers' with {@link PipelineOptions.vision} /
 * {@link PipelineOptions.audio}).
 *
 * MUST: parse and match the asset **before acquiring the GPU** (so a different exception in an
 * environment without a GPU does not hide the real cause, a broken asset).
 * MUST: the only checks allowed after acquiring the GPU are GPU capabilities (features — ADR 0028 /
 * device limits — ADR 0089 decision 5).
 */
const buildEmbedder = async (
  admitted: EmbeddingGemmaAdmission,
  assets: Assets["assets"],
  open: ComponentOpener,
  options: PipelineOptions,
): Promise<Embedder> => {
  const { config, contract, quantName, quant, sessionOptions, gpuFeatures, vision, audio } =
    admitted;
  const tokenizer = new Tokenizer(
    parseGemmaTokenizerAsset(
      new Uint8Array(readAssetBuffer(OWNER, "weights / assets", assets, TOKENIZER)),
    ),
  );
  const rules = embedTextRules(tokenizer, config, {
    images: vision !== undefined,
    audio: audio !== undefined,
  });
  // An id past the table would read a valid row of another token on the GPU, not fail.
  if (tokenizer.maxTokenId >= contract.vocabularySize) {
    throw new Error(
      `${OWNER}: the tokenizer's largest token id ${tokenizer.maxTokenId} is outside the embedding` +
        ` table's ${contract.vocabularySize} rows (a tokenizer of another vocabulary)`,
    );
  }
  // MUST: declared features can be requested only at device creation (ADR 0028).
  const gpu = options.gpu ?? (await acquireGpu(toAcquireGpuOptions(gpuFeatures)));
  const ownsGpu = options.gpu === undefined;
  // The supplier is **the same one** admission looked at (`open` only looks up the opened component).
  return await openEmbedder({
    gpu,
    // The features and limits of the GPU in hand — one helper for every family.
    assertGpu: () =>
      assertAcquiredGpuCapabilities(
        gpuFeatures,
        quant.requiredLimits,
        gpu,
        `${OWNER}: quant '${quantName}'`,
      ),
    model: open(MODEL),
    ...(vision === undefined ? {} : { tower: { graph: open(VISION), contract: vision } }),
    ...(audio === undefined ? {} : { audioTower: { graph: open(AUDIO), contract: audio } }),
    sessionOptions,
    contract,
    rules,
    release: () => {
      if (ownsGpu) gpu.destroy();
    },
    ...(options.onRunDiagnostics === undefined
      ? {}
      : { onRunDiagnostics: options.onRunDiagnostics }),
  });
};

/**
 * EmbeddingGemma text (and image and audio) → sentence embedding (resident Sessions: the text graph,
 * the vision tower with {@link PipelineOptions.vision} and the audio tower with
 * {@link PipelineOptions.audio}).
 *
 * ```ts
 * const pipeline = await embeddinggemma.Pipeline.fromPretrained(ref, { gpu, vision: true, audio: true });
 * const vector = await pipeline.embed("how do I reset my password", { prompt: "query" });
 * const picture = await pipeline.embed([{ type: "image", image }]); // image: decoded RGB8
 * const clip = await pipeline.embed([{ type: "audio", audio }]); // audio: { data, sampleRate: 16000 }
 * ```
 *
 * Construction has only two entries: {@link Pipeline.fromPretrained} (fetch from the
 * distribution) and {@link Pipeline.fromAssets} (already-fetched bytes) — the constructor is
 * private so that no half-built state can bypass the manifest check and the asset matching
 * (ADR 0008).
 */
export class Pipeline {
  readonly #embedder: Embedder;

  private constructor(embedder: Embedder) {
    this.#embedder = embedder;
  }

  /**
   * Fetches from the distribution and builds (`loadManifest` → `resolveSelection` → fetch **only the
   * component's descriptor (part 0)** and run admission → warm up the weight parts → `fetchAssets`
   * of the tokenizer asset → build). Blocks are read in part order at the moment the Session is built
   * (ADR 0109 — `src/hub/components.ts`). The weights fetched are the roles the pipeline runs:
   * `model`, `vision` only with {@link PipelineOptions.vision} and `audio` only with
   * {@link PipelineOptions.audio}. To replace a component with the
   * same role from another repository, use {@link FromPretrainedComponentOptions.components} (those
   * roles — the tokenizer is not replaced).
   *
   * **`ref` is required** (there is no default source — the MUST in `src/hub/repo-ref.ts`). The
   * source this package version verified is `"embeddinggemma-2"` in {@link SOURCES}
   * (`./config.ts`) — to pin reproducibility yourself, the caller gives `{ repo, revision }`
   * explicitly. A string `ref` is read as `{ repo }` (= following `main`). A local distribution (`models/karume-embeddinggemma-2`) is
   * passed as a **source handle** (`nodeDirectory` of `@karume/hub/node`, `denoDirectory` of
   * `@karume/hub/deno`) — it bypasses the check of the HF `owner/name` spelling, the network and
   * CacheStorage (the HTTP-only knobs of {@link FromPretrainedOptions} have no effect).
   */
  static async fromPretrained(
    ref: string | HubRepoRef | DistributionSource,
    options: FromPretrainedOptions = {},
  ): Promise<Pipeline> {
    const source = toManifestSource(
      ref,
      `${OWNER}.fromPretrained`,
      'SOURCES["embeddinggemma-2"] (@karume/models/embeddinggemma)',
    );
    const towers = wantedTowers(`${OWNER}.fromPretrained`, options);
    const hubOptions = hubLoadOptions(options);
    const loaded = await loadManifest(source, hubOptions);
    const choice = {
      ...(options.model === undefined ? {} : { model: options.model }),
      ...(options.quant === undefined ? {} : { quant: options.quant }),
    };
    // MUST: narrow the weights to fetch by role — every preset maps the `vision` and `audio` roles,
    // so without this their descriptors and parts would be fetched for a pipeline that never runs
    // them.
    const keys = componentKeys(towers);
    const selection = resolveSelection(loaded.manifest, { ...choice, weights: keys });
    const buildOptions: PipelineOptions = {
      ...(options.gpu === undefined ? {} : { gpu: options.gpu }),
      ...choice,
      ...(towers.vision ? { vision: true } : {}),
      ...(towers.audio ? { audio: true } : {}),
      ...(options.onRunDiagnostics === undefined
        ? {}
        : { onRunDiagnostics: options.onRunDiagnostics }),
    };
    // The family checks run in the admission hook (before the weight parts are fetched —
    // `hub/components.ts`).
    const { admitted, assets, open } = await loadContainerComponents(
      `${OWNER}.fromPretrained`,
      loaded,
      selection,
      keys,
      async (open) => {
        const admitted = admitEmbeddingGemma(loaded.manifest, open, buildOptions);
        // The `requiredLimits` the distribution declares are checked here **before the weight parts
        // are fetched** (ADR 0089 decision 5).
        await assertRequiredLimitsBeforeDownload(
          admitted.quant.requiredLimits,
          buildOptions.gpu,
          `${OWNER}: quant '${admitted.quantName}'`,
        );
        return admitted;
      },
      {
        ...hubOptions,
        ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
        ...(options.components === undefined ? {} : { components: options.components }),
      },
    );
    return new Pipeline(await buildEmbedder(admitted, assets, open, buildOptions));
  }

  /**
   * Builds from an already-fetched manifest + assets ({@link Assets}). Opens the container (and the
   * towers' with {@link PipelineOptions.vision} / {@link PipelineOptions.audio}), goes through the
   * same admission and build as the fetching surface, and returns with **the Sessions opened** (the
   * MUST in the module doc).
   */
  static async fromAssets(input: Assets, options: PipelineOptions = {}): Promise<Pipeline> {
    const open = await assetOpener(input.assets, wantedTowers(`${OWNER}.fromAssets`, options));
    const admitted = admitEmbeddingGemma(input.manifest, open, options);
    return new Pipeline(await buildEmbedder(admitted, input.assets, open, options));
  }

  /**
   * Embeds one input and returns a unit vector (f32, of the requested width).
   *
   * `input` is a string or an array of content parts — text parts, image parts
   * (`{ type: "image", image }`, decoded RGB8) on a pipeline built with
   * {@link PipelineOptions.vision}, and audio parts (`{ type: "audio", audio }`, mono f32 samples
   * at 16,000 Hz) on one built with {@link PipelineOptions.audio}: `<bos>`, each run of adjacent
   * text (the prompt's prefix in front) tokenized as one string, each image and each audio as its
   * soft tokens between their two markers, `<eos>`. `options.prompt` prepends a declared prompt's prefix, `options.dimension` asks for one
   * of the declared widths (default the full width; a shorter one is the leading slice re-normalised
   * to unit length), `options.signal` cancels before the runs (and between them).
   *
   * Refused with a `ModelInputError`, before the GPU is touched: an empty part array, a part of a
   * type the pipeline does not accept (an image part without the vision tower, an audio part
   * without the audio tower — the message says how to build one with it), an image that is not RGB8
   * pixels, an audio that is not 16,000 Hz (nothing is resampled), longer than 30 s (nothing is
   * truncated), of at most 160 samples (no mel frame) or holding a non-finite sample, an unknown
   * prompt, an undeclared width, a text whose tokens include an image / audio / video placeholder
   * id, and a sequence longer than the model's limit or than this device runs (its storage-buffer
   * limit — the module doc; `<bos>`, `<eos>` and every image's and audio's soft tokens and markers
   * included — nothing is truncated).
   *
   * Concurrent calls **wait and run in order** (one Session is never shared by two runs).
   */
  async embed(
    input: string | readonly EmbedContentPart[],
    options: EmbedOptions = {},
  ): Promise<Float32Array<ArrayBuffer>> {
    return await this.#embedder.embed(input, options);
  }

  /**
   * Returns the token ids {@link embed} would run for the same input and prompt: `<bos>`, the
   * prompt's prefix and the parts (each image's and each audio's soft rows as its placeholder id
   * between its two markers), `<eos>` — its length is the count the limits apply to. Synchronous,
   * on the host (no GPU, an image read for its size only, an audio for its rate, length and samples)
   * and not queued behind the calls in flight.
   *
   * Refused with the same `ModelInputError` as `embed`: an input that is neither a string nor an
   * array, an empty part array, a part of a type the model does not accept, an unknown prompt.
   * A sequence longer than the model's limit and a text whose tokens include a placeholder id are
   * **returned, not refused**, so a caller can show the count and the ids first — `embed` still
   * refuses both. Fails after {@link dispose}, as `embed` does.
   */
  tokenize(
    input: string | readonly EmbedContentPart[],
    options: TokenizeOptions = {},
  ): Int32Array<ArrayBuffer> {
    return this.#embedder.tokenize(input, options);
  }

  /**
   * Releases. Always disposes the Sessions and destroys **only a GPU acquired internally** (a
   * GpuContext passed in `options.gpu` belongs to the caller and is not touched).
   *
   * MUST: waits for the calls in flight before destroying (flush-before-destroy). Later calls
   * return the same completion; `embed` and `tokenize` after it fail.
   */
  dispose(): Promise<void> {
    return this.#embedder.dispose();
  }

  /** `await using` support (Explicit Resource Management) — an alias of {@link dispose}. */
  [Symbol.asyncDispose](): Promise<void> {
    return this.dispose();
  }
}
