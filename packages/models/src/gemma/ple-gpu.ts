/**
 * The **option that keeps Gemma 4's PLE (per-layer embeddings) resident on the GPU and gathers
 * it on the GPU** (the default stays the host — ADR
 * [0085](../../../../docs/decisions/0085-ple-host-gather.md) addendum (2026-09-19)
 * "GPU residency").
 *
 * ## What it removes
 *
 * The host path (`./ple.ts`) **dequantizes `per_layer_inputs[1,M,layers,dim]` on the host and
 * writeBuffers it** on every run. Even one decode token needs a 35,840 B transfer and the
 * dequantization of one row, and a 768-row prefill transfers 27.5 MB. This option removes both —
 * the quantized PLE bytes go up to the GPU **once at load**, and a run passes only token ids and
 * gathers on the GPU.
 *
 * ## Mechanism (the same technique as `./greedy-output.ts`)
 *
 * A Session holding a small gather IR (`embedding` → `mul embed_scale`) is created, its run is
 * **enqueued first in the same batch** as the target's run, and its output is written to a resident
 * tensor (`copyOutputs`). The target takes that resident tensor as the **resident input**
 * `per_layer_inputs` (baked into the bind group without a single writeBuffer), so the fence count
 * per token stays one, as before.
 *
 * ## MUST: bit-identical to the host path (ADR 0085 decision 4)
 *
 * The weight is held as `[tokens×layers, dim]` with index `id×layers + layer`. In this layout **one
 * row = the dim elements of one layer**, so whatever the storage dtype, `scales`' `[tokens,
 * layers]` is the per-row scale sequence as is (i8 / i2 use row scales, i4 uses group scales with
 * `group_size = dim`; both share the flat index `id×layers + layer`). Dequantization is one f32
 * multiplication `f32(q) × scale` in `embedding`, and `embed_scale` is applied **afterwards** in a
 * separate node — the same two steps and the same rounding points as the host's `Math.fround(q ×
 * scale) × embedScale`.
 *
 * ## MUST: single binding (the maintainer's ruling of 2026-09-19, option A)
 *
 * `values` goes into one GPU buffer, unsplit. **A distribution that does not fit the adapter's
 * binding limits (`maxStorageBufferBindingSize` / `maxBufferSize`) cannot use this option** — it
 * does not silently fall back to the host path but fails loudly, stating how many bytes are short.
 *
 * ## Loading (writeBuffer per piece — ADR 0085 decision 2 / ADR 0109 decision 4)
 *
 * The host does not concatenate everything into one huge ArrayBuffer. The `values` blocks the index
 * points at are passed as they are as **pieces of an in-memory container** (`pieces` of
 * `openMemoryContainer`). Session construction calls `read()` once per piece, `queue.writeBuffer`s
 * it at its row offset into one GPU buffer for the whole parent, and **lets go of the reference in
 * that iteration** (`writeBuffer` copies the bytes at the call — ADR 0070 decision 3, 2026-09-24
 * addendum). The in-memory container puts one piece in one part, so a fence is raised once per
 * part, but the fence governs only staging, not host RAM. The bytes of a piece are allocated only
 * for that one read (no buffer is reused; each piece gets its own), so the host RAM taken by
 * `values` stays within "one largest block". Even with a scan-type source, a PLE block is a
 * dedicated part with 1 block = 1 part (container-v1 §4.2), so the holding window holds only that
 * block's part (if the range is not exactly the part, the copy in `readAssetRange` adds one more).
 *
 * The companion scale lives in the same part as piece 1 by contract (container-v1 §5, rule ③), and
 * the runtime writes the whole scale into one GPU buffer at piece 1, so **only the scale is
 * collected whole first**. That is the dominant term of the construction peak — `scales` is `tokens
 * × layers × 4` bytes, which for E2B (262,144 tokens × 35 layers) collects 35 MiB into one buffer.
 * The container holds that buffer as is for the duration of construction (no copy). `ple` is
 * consumed only by `embedding`'s weight slot, so it loads compressed, and the "copy of the carried
 * scale" that an expanded weight's piece list takes (container-v1 §11) is not made either. So the
 * host RAM peak is "the whole scale table + one largest block" (no copy, so the scale counts once).
 * Estimating with "one largest block" alone drops the dominant term entirely. The scale table is
 * not subject to the `krm` block limit (32 MiB) — the in-memory container does not compare the
 * length of a synthesized block with the limit.
 *
 * MUST: **do not hold a reader longer than one read** (the same MUST as `./ple.ts`). The
 * `AssetReader` of an unverified source keeps the block for the reader's lifetime, so holding
 * readers breaks the "stream one at a time" shape as far as host RAM is concerned.
 *
 * MUST: zero module side effects (nothing happens until these functions are called).
 */

import {
  type AssetReader,
  type BatchScope,
  type BoundContainer,
  type CodecName,
  type GpuContext,
  type IrDeclaration,
  type MemoryEncoding,
  type MemoryTensor,
  openMemoryContainer,
  parseIrDeclarationValue,
  prepareContainer,
  type PreparedModel,
  type ResidentTensor,
  type RunInput,
  type RunInputs,
  type Session,
} from "@karume/runtime";
import { disposeSteps } from "../session/dispose-steps.ts";
import { readAssetRange } from "../hub/asset-readers.ts";
import {
  type Gemma4PleBlock,
  gemma4PleBlockBytes,
  type Gemma4PleIndex,
  type Gemma4PleTable,
  packFactor,
} from "./ple-index.ts";

/** Where PLE is placed (the value range of {@link Gemma4PipelineOptions.pleResidency}). */
export type Gemma4PleResidency = "host" | "gpu";

const PLE_RESIDENCIES: readonly Gemma4PleResidency[] = ["host", "gpu"];

/**
 * Checks and returns the option value (fails synchronously **before a single asset byte is read**).
 *
 * MUST: do not route a misspelling from outside the type to the default — if a value like `"GPU"`
 * silently ran on the host path, "GPU residency did not make it faster" would remain with neither
 * an exception nor a warning.
 */
export const assertGemma4PleResidency = (where: string, value: unknown): Gemma4PleResidency => {
  if (typeof value !== "string" || !PLE_RESIDENCIES.includes(value as Gemma4PleResidency)) {
    throw new Error(
      `${where}: pleResidency ${JSON.stringify(value)} is not ${PLE_RESIDENCIES.join(" / ")}`,
    );
  }
  return value as Gemma4PleResidency;
};

/**
 * Names in the gather IR (this one place is canonical — both the in-memory container and the
 * Session use them).
 */
const INDEX_INPUT = "ple_index";
const GATHER_OUTPUT = "per_layer";
const RAW_VALUE = "ple_raw";
const WEIGHT_TENSOR = "ple";
const EMBED_SCALE_TENSOR = "embed_scale";
/**
 * Name of the graph placed in the in-memory container (a namespace separate from the initializer
 * name {@link WEIGHT_TENSOR}).
 */
const GATHER_GRAPH = "ple";
/**
 * Symbol of the physical row count (token rows) — bound only from the shape of {@link INDEX_INPUT}.
 */
const ROW_SYMBOL = "M";

/**
 * Storage dtype → registered name in the codec ledger (how `values` is packed — container-v1 §6.3).
 */
const pleCodec = (index: Gemma4PleIndex): CodecName =>
  index.storage === "i2" ? "int2-off" : index.storage === "i4" ? "int4-sym-g" : "int8-sym";

/**
 * The bytes GPU residency needs (**decided by the index alone** — known before reading any bytes).
 *
 * `values` is the single-binding body and `scales` the companion scale. Both the binding-limit
 * check and the estimate derive from this one function.
 */
export const gemma4PleGpuBytes = (
  index: Gemma4PleIndex,
): { readonly values: number; readonly scales: number; readonly total: number } => {
  const values = index.tokens * index.values.rowBytes;
  const scales = index.tokens * index.scales.rowBytes;
  return { values, scales, total: values + scales };
};

/**
 * The gather IR (`embedding` → `mul embed_scale`).
 *
 * The index input is `[1, M, layers]`, so the output is directly `[1, M, layers, dim]` = the
 * declared shape of `per_layer_inputs` (no reshape needed — `embedding`'s output shape is "the
 * index shape + the weight's last dimension").
 *
 * MUST: `embed_scale` is applied in a **separate node** (not folded into `embedding`). Matching the
 * rounding points of the host path's `Math.fround(q × scale) × embedScale` is the only purpose of
 * these two steps (ADR 0085 decision 4).
 *
 * MUST: pass the built declaration through `parseIrDeclarationValue` — it goes through neither the
 * exporter nor the container reader, so without it none of the rules decided by the graph alone
 * (SSA, topological order, `requires.ops` matching the ops used, unknown keys, canonical symbol
 * names) would be checked.
 *
 * MUST NOT: add the optional `states` to the literal below. Leaving it out keeps the literal from
 * typing as an {@link IrDeclaration}, so only the return value of `parseIrDeclarationValue` can be
 * this function's return value (an edit that removes the wrapper fails the type check, `vp check`).
 */
export const gemma4PleGatherGraph = (index: Gemma4PleIndex): IrDeclaration => {
  const rows = index.tokens * index.layers;
  return parseIrDeclarationValue({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["embedding", "mul"] },
    symbols: [ROW_SYMBOL],
    // An IR v2 declaration carries names only (storage belongs to the binding table = the in-memory
    // container's `encoding`). No `shared` = the side that owns the data (`false` cannot be spelled
    // — it is the same declaration as an absent field).
    initializers: {
      [WEIGHT_TENSOR]: {},
      [EMBED_SCALE_TENSOR]: {},
    },
    inputs: [{ name: INDEX_INPUT, dtype: "i32", shape: [1, ROW_SYMBOL, index.layers] }],
    outputs: [GATHER_OUTPUT],
    values: {
      [WEIGHT_TENSOR]: { dtype: "f32", shape: [rows, index.dim] },
      [EMBED_SCALE_TENSOR]: { dtype: "f32", shape: [1] },
      [RAW_VALUE]: { dtype: "f32", shape: [1, ROW_SYMBOL, index.layers, index.dim] },
      [GATHER_OUTPUT]: { dtype: "f32", shape: [1, ROW_SYMBOL, index.layers, index.dim] },
    },
    nodes: [
      {
        op: "embedding",
        ins: [WEIGHT_TENSOR, INDEX_INPUT],
        outs: [RAW_VALUE],
        // `padding_idx` is a required attr of the contract table but has no effect in forward
        // (ADR 0012). The sentinel −1 = unspecified, giving the same "plain gather" as the host
        // path's row lookup.
        attrs: { padding_idx: -1 },
      },
      { op: "mul", ins: [RAW_VALUE, EMBED_SCALE_TENSOR], outs: [GATHER_OUTPUT], attrs: {} },
    ],
  });
};

/**
 * Reads a range of one block (the reader lives only for **this one read** — the MUST in the module
 * doc).
 */
const readBlockRange = async (
  openBlock: (asset: string) => AssetReader,
  block: Gemma4PleBlock,
  offset: number,
  length: number,
): Promise<Uint8Array<ArrayBuffer>> =>
  new Uint8Array(await readAssetRange(openBlock(block.asset), offset, length));

/**
 * Reads the `scales` blocks in index order and lays them out as one matrix (the whole companion
 * scale).
 */
const readScaleTable = async (
  table: Gemma4PleTable,
  openBlock: (asset: string) => AssetReader,
  total: number,
): Promise<Uint8Array<ArrayBuffer>> => {
  const bytes = new Uint8Array(new ArrayBuffer(total));
  for (const block of table.blocks) {
    const payload = await readBlockRange(openBlock, block, 0, gemma4PleBlockBytes(table, block));
    bytes.set(payload, block.start * table.rowBytes);
  }
  return bytes;
};

/**
 * The host's GPU-resident PLE (the surface that builds the **resident input passed to the target's
 * run**).
 *
 * MUST: `enqueue` goes into **the same batch** as the target's enqueue (no extra fence).
 */
export type Gemma4PleResident = {
  /** Name of the graph input the target receives (`per_layer_inputs`). */
  readonly inputName: string;
  /** Name of the graph input that carries the run's token ids (`input_ids`). */
  readonly idsName: string;
  /**
   * Adds to `batch` the enqueue that gathers the PLE of `ids` on the GPU and writes it to a
   * resident tensor.
   *
   * Returns the resident tensor bound to the target's `per_layer_inputs`; one is reused per row
   * count (runs of the same shape hit the same buffer, so the bind group is not rebaked either).
   */
  enqueue(batch: BatchScope, ids: Int32Array<ArrayBuffer>): Promise<ResidentTensor>;
  /**
   * Estimate of the resident PLE (values + scales) and the gather Session (counted in
   * auxiliaryBytes).
   */
  readonly extraBytes: number;
  /** The resident quantized bytes themselves (values + scales). */
  readonly residentBytes: number;
  dispose(): Promise<void>;
};

export type Gemma4PleResidentOptions = {
  readonly gpu: GpuContext;
  readonly index: Gemma4PleIndex;
  /** Opens the reader of one block the index points at (`open(MODEL).asset` itself). */
  readonly openBlock: (asset: string) => AssetReader;
  /** The main embedding's vocab row count (cross-check of the id spaces — ADR 0085 decision 5). */
  readonly vocabSize: number;
  /** Name of the graph input the target receives. */
  readonly inputName: string;
  /** Name of the graph input that carries the token ids. */
  readonly idsName: string;
  /**
   * Physical row counts this pipeline can run (the estimate's upper bound — decode 1 + buckets +
   * chunk length).
   */
  readonly rows: readonly number[];
  /** The subject of diagnostics (`Gemma4Pipeline` / `Gemma4QatPipeline`). */
  readonly entry: string;
};

/**
 * Builds an in-memory container from the blocks the index points at (**without touching the GPU
 * once**).
 *
 * One `values` block = one piece, and a piece's `read()` issues its range read only at that moment
 * (this layer holds no bytes at all). The row range is `[block.start × layers, block.stop ×
 * layers)` — one IR row = one layer, so the token range maps directly to a row range.
 *
 * The companion scale sits in the same part as piece 1 by rule ③, so only the scale is collected
 * whole first. An index with a single block cannot be split into `pieces` (2 or more, MUST), so it
 * is passed whole as one.
 *
 * NOTE: `export` is there to pin the synthesized cross-check without a GPU (not exposed on `mod.ts`
 * / the subpath surface).
 */
export const buildGemma4PleGatherContainer = async (
  index: Gemma4PleIndex,
  openBlock: (asset: string) => AssetReader,
  entry: string,
): Promise<BoundContainer> => {
  const factor = packFactor(index);
  const rowBytes = index.dim / factor;
  if (!Number.isSafeInteger(rowBytes) || rowBytes % 4 !== 0) {
    throw new Error(
      `${entry}: one PLE layer of ${index.dim} elements (storage ${index.storage}) is` +
        ` ${rowBytes} bytes, not 4-aligned (the container's block end alignment cannot be met)`,
    );
  }
  const bytes = gemma4PleGpuBytes(index);
  const scale = await readScaleTable(index.scales, openBlock, bytes.scales);
  // i4 has group = row length (meeting ADR 0069 decision 2's "power of two ≥ 16"); i8 / i2 are
  // per-channel with groupSize = row length. Either way there is one group, so the scale is exactly
  // the f32 column `[tokens × layers, 1]` (= the order `readScaleTable` laid out).
  const encoding: MemoryEncoding = {
    codec: pleCodec(index),
    rowAxis: 0,
    groupSize: index.dim,
    scale,
  };
  const table = index.values;
  const readBlock = (block: Gemma4PleBlock) => (): Promise<Uint8Array<ArrayBuffer>> =>
    readBlockRange(openBlock, block, 0, gemma4PleBlockBytes(table, block));
  const values: MemoryTensor =
    table.blocks.length < 2
      ? { encoding, bytes: await readBlock(table.blocks[0])() }
      : {
          encoding,
          pieces: table.blocks.map((block) => ({
            rows: [block.start * index.layers, block.stop * index.layers] as const,
            read: readBlock(block),
          })),
        };
  return openMemoryContainer({
    graphs: { [GATHER_GRAPH]: gemma4PleGatherGraph(index) },
    tensors: {
      [GATHER_GRAPH]: {
        [WEIGHT_TENSOR]: values,
        [EMBED_SCALE_TENSOR]: {
          encoding: { codec: "f32" },
          bytes: new Uint8Array(Float32Array.of(index.embedScale).buffer),
        },
      },
    },
  });
};

/**
 * Checks that it fits the binding limits (**before a single weight byte is fetched**).
 *
 * MUST: do not silently fall back to the host path. Tell the user who chose this option explicitly
 * "what" is short and "by how much" (`acquireGpu` requests the adapter's measured values as they
 * are, so a value short here is that machine's limit itself).
 */
const assertBindingLimits = (options: Gemma4PleResidentOptions, valuesBytes: number): void => {
  const { limits } = options.gpu;
  const shortfalls = (
    [
      ["maxStorageBufferBindingSize", limits.maxStorageBufferBindingSize],
      ["maxBufferSize", limits.maxBufferSize],
    ] as const
  ).filter(([, granted]) => granted < valuesBytes);
  if (shortfalls.length === 0) return;
  throw new Error(
    `${options.entry}: pleResidency: "gpu" puts the PLE's ${valuesBytes} bytes of quantized data` +
      ` in a single binding, but this device has ` +
      shortfalls.map(([key, granted]) => `${key} ${granted}`).join(" / ") +
      ` (short by ${shortfalls
        .map(([key, granted]) => `${key} ${valuesBytes - granted}`)
        .join(
          " / ",
        )} bytes) — read it with the default pleResidency: "host", or open it on a device with a larger binding limit`,
  );
};

/**
 * Makes PLE resident on the GPU and returns the surface of the on-GPU gather.
 *
 * The order is "index check → binding-limit check → scale collection → sequential consumption of
 * pieces", and only the last step touches the GPU (the same ordering as ADR 0070 decision 5's
 * graph-first).
 *
 * NOTE: only an index with a single `values` block (a small distribution whose whole table fits the
 * block limit) cannot be split into pieces, so values are also read whole in the same step as the
 * scale collection — the only form read in full before the capability check (`prepareContainer`).
 */
export const createGemma4PleResident = async (
  options: Gemma4PleResidentOptions,
): Promise<Gemma4PleResident> => {
  const { gpu, index, entry } = options;
  // (1) the PLE index's row count and (2) the main embedding's vocab row count (ADR 0085
  // decision 5's cross-check — the same relation as the check in the host path's `createGemma4Ple`,
  // applied in this option too).
  if (index.tokens !== options.vocabSize) {
    throw new Error(
      `${entry}: the PLE index's row count ${index.tokens} differs from the main embedding's vocab row count` +
        ` ${options.vocabSize} (a combination baked with a different vocabulary)`,
    );
  }
  const bytes = gemma4PleGpuBytes(index);
  assertBindingLimits(options, bytes.values);
  const bound = await buildGemma4PleGatherContainer(index, options.openBlock, entry);

  const prepared: PreparedModel = prepareContainer(bound, GATHER_GRAPH);
  // One resident output per run shape (the set of row counts is bounded: decode 1 + buckets + chunk
  // length).
  const outputBytes = [...new Set(options.rows)].reduce(
    (total, rows) => total + rows * index.layers * index.dim * 4,
    0,
  );
  // MUST: pass **the same budget** to the Session and the estimate (if only one fell back to the
  // default, the reported peak would claim a different budget from the set actually held). Each
  // shape's backing is two buffers, "gather intermediate + output", so pass a value that can hold
  // every shape explicitly (with the default 256 MiB, this small graph's report would be the budget
  // figure itself).
  const backingBudget = outputBytes * 2;
  const maxRows = options.rows.reduce((largest, rows) => Math.max(largest, rows), 1);
  const estimate = prepared.estimate({
    bindings: { [ROW_SYMBOL]: maxRows },
    planBackingBudgetBytes: backingBudget,
  });
  const session: Session = await prepared.createContainerSession(gpu, {
    planBackingBudgetBytes: backingBudget,
  });

  /**
   * Resident output and index buffer per row count (runs of the same shape hit the same object).
   */
  const outputs = new Map<number, ResidentTensor>();
  const indices = new Map<number, Int32Array<ArrayBuffer>>();
  let disposal: Promise<void> | undefined;

  const outputFor = async (rows: number): Promise<ResidentTensor> => {
    const cached = outputs.get(rows);
    if (cached !== undefined) return cached;
    const resident = await gpu.createResident(
      rows * index.layers * index.dim * 4,
      `ple-gather-${rows}`,
    );
    outputs.set(rows, resident);
    return resident;
  };

  return {
    inputName: options.inputName,
    idsName: options.idsName,
    extraBytes: estimate.peakAccountedBytes + outputBytes,
    residentBytes: bytes.total,
    async enqueue(batch: BatchScope, ids: Int32Array<ArrayBuffer>): Promise<ResidentTensor> {
      if (disposal !== undefined) throw new Error(`${entry}: a disposed PLE cannot be looked up`);
      const rows = ids.length;
      if (rows < 1) throw new Error(`${entry}: the token sequence of the PLE gather is empty`);
      let expanded = indices.get(rows);
      if (expanded === undefined) {
        expanded = new Int32Array(new ArrayBuffer(rows * index.layers * 4));
        indices.set(rows, expanded);
      }
      for (let row = 0; row < rows; row += 1) {
        const id = ids[row];
        // Out of range is not OOB but "a valid row of another token" / NaN contamination, so this
        // is the fail-loudly point (the same check as (3) of the host path's `Gemma4Ple.gather`).
        if (id < 0 || id >= index.tokens) {
          throw new Error(
            `${entry}: token id[${row}] ${id} is outside the PLE index's 0..${index.tokens - 1}`,
          );
        }
        const base = id * index.layers;
        for (let layer = 0; layer < index.layers; layer += 1) {
          expanded[row * index.layers + layer] = base + layer;
        }
      }
      const output = await outputFor(rows);
      await session.enqueue(
        {
          [INDEX_INPUT]: {
            dtype: "i32",
            shape: [1, rows, index.layers],
            data: expanded,
          },
        },
        { batch, copyOutputs: { [GATHER_OUTPUT]: output } },
      );
      return output;
    },
    dispose(): Promise<void> {
      // Order: Session → resident tensors (return the baked references before releasing —
      // ADR 0004).
      disposal ??= disposeSteps([
        () => session.dispose(),
        ...[...outputs.values()].map((resident) => () => resident.dispose()),
      ]).then(() => {
        outputs.clear();
        indices.clear();
      });
      return disposal;
    },
  };
};

/**
 * Takes the token ids out of a run's inputs (the only source of the on-GPU gather's indices).
 *
 * MUST: check that it is a host `Tensor` (i32). A resident input (`ResidentTensor`) has no
 * host-side values, so reading it silently would let "PLE gathered with empty indices" flow on
 * without an exception.
 */
export const gemma4PleGatherIds = (
  where: string,
  inputs: RunInputs,
  name: string,
): Int32Array<ArrayBuffer> => {
  const input: RunInput | undefined = Object.hasOwn(inputs, name) ? inputs[name] : undefined;
  if (input === undefined || !("dtype" in input) || input.dtype !== "i32") {
    throw new Error(
      `${where}: input '${name}' is not an i32 host tensor (the indices of the on-GPU gather)`,
    );
  }
  return input.data;
};
