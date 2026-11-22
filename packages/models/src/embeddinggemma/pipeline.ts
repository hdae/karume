/**
 * `Pipeline` — the one surface for text → sentence embedding of the EmbeddingGemma family
 * (today the text model of EmbeddingGemma 2 — distribution `karume-embeddinggemma-2`, model
 * `text`, pipeline `embeddinggemma/1`).
 *
 * Per call (every step in Karume, no torch):
 *
 * 1. Host: the input's content parts → one text, the prompt's prefix in front → token ids with
 *    `<bos>` … `<eos>` (the distribution's tokenizer asset, the Gemma tokenizer of
 *    `src/gemma/text/`); the limit and the placeholder refusal (`./input.ts`)
 * 2. Runs `model` once (`input_ids` `[1, T]`, `pool_mask` `[1, T]` of ones) → the unit vector
 *    `[1, D]` (the graph holds the text tower, the masked mean pooling and the L2 normalisation)
 * 3. Host: a shorter width is the leading slice re-normalised (`truncateEmbedding`)
 *
 * One sequence per call, no padding and no batch (the graph's sliding window is a band declared on
 * its attention, with no run-time mask — docs/limitations.md). Concurrent calls queue on the
 * serialization chain.
 *
 * ## MUST: one Session is opened at construction and kept until `dispose`
 *
 * Embedding is normally called many times; opening a Session per call would upload the whole
 * weights each time (the same residency as siglip2 / umt5).
 *
 * MUST: it still goes on the serialization chain — concurrent `embed` calls would share one Session
 * between two runs, and `dispose` would destroy the GPU without waiting for in-flight work (a
 * flush-before-destroy violation). The chain, the Session and the release live in {@link Embedder},
 * the part a host test drives with a stand-in Session; `Pipeline` adds construction only.
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
 * runs every shorter text.
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
 * environment) and by the tokenizer's parity with upstream (`gemma4_tokenizer_test.ts`,
 * `embeddinggemma2_pipeline_test.ts`).
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
  type EmbedContentPart,
  type EmbedOptions,
  type EmbedRequest,
  type EmbedTextRules,
  embedTextRules,
  planEmbedRequest,
  type TokenizeOptions,
  tokenizeEmbedInput,
  truncateEmbedding,
} from "./input.ts";

/** The message prefix (the class name). */
const OWNER = "embeddinggemma.Pipeline";

/**
 * The component name in the manifest's weights table (also the role the replacement option accepts
 * — the graph name the dist recipe writes).
 */
const MODEL = "model";

/** This family's weights components. */
const COMPONENT_KEYS = [MODEL] as const;

/** The key of the tokenizer asset (format `karume-gemma-tokenizer/1`). */
const TOKENIZER = "tokenizer";

/** The graph inputs (`GRAPH_INPUTS` of `tools/export-recipes/embeddinggemma2/export.py`). */
const INPUT_IDS = "input_ids";
const POOL_MASK = "pool_mask";

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
   * An observation hook that receives the diagnostics of every run (one embedding = one `model`
   * run). For per-op GPU time (`lastRunTiming`) pass `acquireGpu({ gpuTiming: true })` as `gpu`
   * (ADR 0021 — no timing by default).
   *
   * NOTE: no component name is passed — there is one graph (as in siglip2).
   *
   * Exceptions from the callback are not swallowed (fail loudly — the embedding fails with it).
   */
  readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics) => void;
};

/**
 * Fetch-layer options used only by {@link Pipeline.fromPretrained} (passed through to the hub).
 * The component replacement option (`components`) has the single role `model`.
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
 * `krm`, or the part list `model[0]` / `model[1]` / …) and the tokenizer asset `tokenizer`.
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
const assertDeviceTokenLimit = (count: number, limit: EmbeddingGemmaDeviceLimit): void => {
  if (count <= limit.maxTokens) return;
  throw new ModelInputError(
    `${OWNER}: the input is ${count} tokens (<bos> and <eos> included), over this device's limit of` +
      ` ${limit.maxTokens} — the graph's largest intermediate (${limit.bytesPerToken} B per token)` +
      ` must fit one storage buffer of ${limit.bindingBytes} B (the device's` +
      " maxStorageBufferBindingSize / maxBufferSize); nothing is truncated — shorten the text or" +
      " split it",
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
 * MUST: call it at admission (before the weight parts are fetched): the host builds the two inputs
 * from its own token ids and reads one `[1, D]` output, so a mismatch would otherwise surface after
 * the weights are uploaded, in the Session's shape check.
 *
 * - One symbolic length T; exactly the inputs `input_ids` i32 `[1, T]` and `pool_mask` f32 `[1, T]`.
 * - One f32 output `[1, D]` with D = `pipelineConfig.dimensions[0]` (the full width a caller gets).
 * - Exactly one `embedding` node reads `input_ids`, its table `[V, W]` static (V bounds the token
 *   ids the tokenizer may produce — checked when the asset arrives).
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
  const ids = graphInput(model, INPUT_IDS);
  const mask = graphInput(model, POOL_MASK);
  if (graph.inputs.length !== 2) {
    throw new Error(
      `${OWNER}: the graph has ${graph.inputs.length} inputs (the host feeds only '${INPUT_IDS}' and '${POOL_MASK}')`,
    );
  }
  for (const [spec, dtype] of [
    [ids, "i32"],
    [mask, "f32"],
  ] as const) {
    if (spec.dtype !== dtype || !sameShape(spec.shape, [1, symbol])) {
      throw new Error(
        `${OWNER}: graph input '${spec.name}' is ${spec.dtype} [${spec.shape.join(", ")}], expected` +
          ` ${dtype} [1, ${symbol}]`,
      );
    }
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
  if (
    tableShape?.length !== 2 ||
    typeof vocabularySize !== "number" ||
    !Number.isInteger(vocabularySize) ||
    vocabularySize <= 0
  ) {
    throw new Error(
      `${OWNER}: the embedding table '${table}' has shape [${tableShape?.join(", ") ?? "undeclared"}]` +
        " (expected a static [V, W])",
    );
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
  return { output, width, vocabularySize, bytesPerToken: graphBytesPerToken(model, symbol) };
};

/** The part of a Session the embedder uses (narrow, so a host test can stand in for the GPU). */
export type EmbeddingSession = Pick<Session, "run" | "diagnostics" | "dispose">;

/** What an {@link Embedder} runs with (all of it checked before it is built). */
export type EmbedderParts = {
  /** The Session of `model`, opened once; the embedder disposes it. */
  readonly session: EmbeddingSession;
  readonly contract: EmbeddingGemmaGraphContract;
  readonly rules: EmbedTextRules;
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
 * The resident Session of one model with its serialization chain: plans each request on the host
 * (`planEmbedRequest`), runs it in turn, and disposes the Session (then `release`) after the work
 * in flight.
 *
 * NOTE: internal — {@link Pipeline} is the public entry and the only one that builds it from a
 * distribution; the `export` is for the host test that drives it with a stand-in Session (not on
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

  constructor(parts: EmbedderParts) {
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
    assertDeviceTokenLimit(request.ids.length, this.#deviceLimit);
    return await this.#chain(() => this.#run(request, options.signal));
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

  /** One run (the abort check, the run, the output check, the width). */
  async #run(
    request: EmbedRequest,
    signal: AbortSignal | undefined,
  ): Promise<Float32Array<ArrayBuffer>> {
    // A cancellation that arrived while waiting for the turn also takes effect here (a run is
    // indivisible — cancellation works only before it).
    await settleAbort(signal);
    const { session, contract, onRunDiagnostics } = this.#parts;
    const length = request.ids.length;
    const outputs = await session.run({
      [INPUT_IDS]: { dtype: "i32", shape: [1, length], data: request.ids },
      // All ones: the pooling averages every token, the prompt's included (upstream's
      // `include_prompt`); the mask is not wired to attention.
      [POOL_MASK]: { dtype: "f32", shape: [1, length], data: new Float32Array(length).fill(1) },
    });
    onRunDiagnostics?.(session.diagnostics());
    return truncateEmbedding(readEmbedding(outputs, contract), request.dimension);
  }

  /**
   * Disposes the Session, then runs `release`, after the calls in flight (flush-before-destroy —
   * the disposal is queued on the same chain). Later calls return the same completion.
   *
   * MUST: `release` runs even if `Session.dispose()` fails ({@link disposeSteps}) — a GPU acquired
   * by the pipeline would otherwise never be returned.
   */
  dispose(): Promise<void> {
    this.#disposal ??= this.#chain(() =>
      disposeSteps([() => this.#parts.session.dispose(), () => this.#parts.release?.()]),
    );
    return this.#disposal;
  }
}

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
};

/**
 * The component supplier of the whole-bytes surface (`fromAssets`) (the receiving implementation is
 * shared by every family — {@link assetComponentOpener}).
 *
 * MUST: do not copy with `slice` ({@link readAssetBuffer}).
 */
const assetOpener = (assets: Assets["assets"]): Promise<ComponentOpener> =>
  assetComponentOpener(
    OWNER,
    assets,
    (key) => readAssetBuffer(OWNER, "weights / assets", assets, key),
    COMPONENT_KEYS,
  );

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
  return { config, contract, quantName, quant, sessionOptions, gpuFeatures };
};

/**
 * Builds the embedder from the admitted material + assets (parse and match the tokenizer asset →
 * acquire the GPU → open the Session).
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
  const { config, contract, quantName, quant, sessionOptions, gpuFeatures } = admitted;
  const tokenizer = new Tokenizer(
    parseGemmaTokenizerAsset(
      new Uint8Array(readAssetBuffer(OWNER, "weights / assets", assets, TOKENIZER)),
    ),
  );
  const rules = embedTextRules(tokenizer, config);
  // An id past the table would read a valid row of another token on the GPU, not fail.
  if (tokenizer.maxTokenId >= contract.vocabularySize) {
    throw new Error(
      `${OWNER}: the tokenizer's largest token id ${tokenizer.maxTokenId} is outside the embedding` +
        ` table's ${contract.vocabularySize} rows (a tokenizer of another vocabulary)`,
    );
  }
  // The supplier is **the same one** admission looked at (`open` only looks up the opened component).
  const model = open(MODEL);
  // MUST: declared features can be requested only at device creation (ADR 0028).
  const gpu = options.gpu ?? (await acquireGpu(toAcquireGpuOptions(gpuFeatures)));
  const ownsGpu = options.gpu === undefined;
  try {
    // The features and limits of the GPU in hand — one helper for every family.
    assertAcquiredGpuCapabilities(
      gpuFeatures,
      quant.requiredLimits,
      gpu,
      `${OWNER}: quant '${quantName}'`,
    );
    return new Embedder({
      session: await model.createSession(gpu, {
        ...sessionOptions,
        attentionRowBlockMaxBytes: ATTENTION_ROW_BLOCK_BYTES,
      }),
      contract,
      rules,
      deviceLimits: gpu.limits,
      release: () => {
        if (ownsGpu) gpu.destroy();
      },
      ...(options.onRunDiagnostics === undefined
        ? {}
        : { onRunDiagnostics: options.onRunDiagnostics }),
    });
  } catch (error) {
    // A GPU acquired internally can be released by no one else if construction fails.
    if (ownsGpu) gpu.destroy();
    throw error;
  }
};

/**
 * EmbeddingGemma text → sentence embedding (one graph, resident Session).
 *
 * ```ts
 * const pipeline = await embeddinggemma.Pipeline.fromPretrained(ref, { gpu });
 * const vector = await pipeline.embed("how do I reset my password", { prompt: "query" });
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
   * (ADR 0109 — `src/hub/components.ts`). To replace the component with the same role from another
   * repository, use {@link FromPretrainedComponentOptions.components} (the single role `model` —
   * the tokenizer is not replaced).
   *
   * **`ref` is required** (there is no default source — the MUST in `src/hub/repo-ref.ts`; this
   * family has no published distribution, so there is no pin table either). A string `ref` is read
   * as `{ repo }` (= following `main`). A local distribution (`models/karume-embeddinggemma-2`) is
   * passed as a **source handle** (`nodeDirectory` of `@karume/hub/node`, `denoDirectory` of
   * `@karume/hub/deno`) — it bypasses the check of the HF `owner/name` spelling, the network and
   * CacheStorage (the HTTP-only knobs of {@link FromPretrainedOptions} have no effect).
   */
  static async fromPretrained(
    ref: string | HubRepoRef | DistributionSource,
    options: FromPretrainedOptions = {},
  ): Promise<Pipeline> {
    const source = toManifestSource(ref, `${OWNER}.fromPretrained`);
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
    // The family checks run in the admission hook (before the weight parts are fetched —
    // `hub/components.ts`).
    const { admitted, assets, open } = await loadContainerComponents(
      `${OWNER}.fromPretrained`,
      loaded,
      selection,
      COMPONENT_KEYS,
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
   * Builds from an already-fetched manifest + assets ({@link Assets}). Opens the container, goes
   * through the same admission and build as the fetching surface, and returns with **the `model`
   * Session opened** (the MUST in the module doc).
   */
  static async fromAssets(input: Assets, options: PipelineOptions = {}): Promise<Pipeline> {
    const open = await assetOpener(input.assets);
    const admitted = admitEmbeddingGemma(input.manifest, open, options);
    return new Pipeline(await buildEmbedder(admitted, input.assets, open, options));
  }

  /**
   * Embeds one input and returns a unit vector (f32, of the requested width).
   *
   * `input` is a string or an array of content parts (today text parts only, concatenated in order
   * with nothing between them). `options.prompt` prepends a declared prompt's prefix,
   * `options.dimension` asks for one of the declared widths (default the full width; a shorter one
   * is the leading slice re-normalised to unit length), `options.signal` cancels before the run.
   *
   * Refused with a `ModelInputError`, before the GPU is touched: an empty part array, a part of a
   * type the model does not accept, an unknown prompt, an undeclared width, a text whose tokens
   * include an image / audio / video placeholder id, and a sequence longer than the model's limit
   * or than this device runs (its storage-buffer limit — the module doc; `<bos>` and `<eos>`
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
   * Returns the token ids {@link embed} would run for the same input and prompt: the prompt's
   * prefix, the joined text parts, `<bos>` … `<eos>`. Synchronous, on the host (no GPU) and not
   * queued behind the calls in flight.
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
   * Releases. Always disposes the Session and destroys **only a GPU acquired internally** (a
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
