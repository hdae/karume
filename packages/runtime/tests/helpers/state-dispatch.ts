import { stateStatsPvWgsl } from "../../src/kernels/state-attention-stats-pv.ts";
// **Direct dispatch harness** of gpu_state_attention_test.ts (the 3 stages of states-form
// attention + `state_append`). Wiring into the executor / recipe-builder belongs to step D-3, so
// the real-GPU checks of this step dispatch the kernels directly with `createComputePipeline` +
// hand-built buffers. Dependencies go one way, helpers → src.
//
// MUST: pipelines are cached by **the final WGSL string** (no `PipelineCache`). Fault injection
// dispatches **a different string** with rewritten WGSL, so caching by key (= variant) would hand
// the correct version's pipeline to the mutant's dispatch, and the fault injection would be
// "always green".
// MUST: S is **filled with a poison value** every time before ①QK is dispatched (S is a scratch
// buffer, and in real use the previous run's leftovers are in it). Both the bug where ① does not
// write the whole live range and the bug where ②③ read beyond the live range show up in values
// only through this poison.

import {
  STATE_STATS_STRIDE,
  stateAttentionParams,
  statePvParallelReadonlyWgsl,
  statePvParallelReadonlyWorkgroups,
  statePvParallelWgsl,
  statePvParallelWorkgroups,
  statePvTiledParams,
  statePvTiledWorkgroups,
  statePvWgsl,
  statePvWorkgroups,
  stateQkFoldedWgsl,
  stateQkFoldedWorkgroups,
  stateQkParallelReadonlyWgsl,
  stateQkParallelReadonlyWorkgroups,
  stateQkParallelWgsl,
  stateQkParallelWorkgroups,
  stateQkTiledParams,
  stateQkTiledWorkgroups,
  stateQkWgsl,
  stateQkWorkgroups,
  stateSliding,
  stateStatsParams,
  stateStatsReadonlyWgsl,
  stateStatsReadonlyWorkgroups,
  stateStatsWgsl,
  stateStatsWorkgroups,
} from "../../src/kernels/state-attention.ts";
import { statePvTiledWgsl, stateQkTiledWgsl } from "../../src/kernels/gemm.ts";
import type { StateStorage } from "../../src/kernels/state-storage.ts";
import { tiledWorkgroups } from "../../src/codegen/dispatch.ts";
import { attentionQkParams, attentionQkWgsl } from "../../src/kernels/attention.ts";
import { defaultGemmGeometry, gemmTileM, gemmTileN } from "../../src/kernels/gemm-geometry.ts";
import { BUFFER_USAGE, MAP_MODE } from "../../src/gpu/webgpu-constants.ts";
import {
  stateAppendParams,
  stateAppendWgsl,
  stateAppendWorkgroups,
} from "../../src/kernels/state-append.ts";

/** Shape and logical lengths of one case (symbols per ADR 0067 decision 4). */
export type StateCase = {
  readonly name: string;
  /** `B`. */
  readonly batch: number;
  /** `H`. */
  readonly heads: number;
  /** `Hkv` (`H % Hkv == 0`). */
  readonly kvHeads: number;
  /** `M` (physical chunk row count). */
  readonly chunkRows: number;
  /** `D`. */
  readonly depth: number;
  /** `C` (row capacity of the slot). */
  readonly capacity: number;
  /** `W` (`0` = full). */
  readonly window: number;
  /** `P`. */
  readonly past: number;
  /** `Q`. */
  readonly query: number;
  /**
   * Rows handled by one dispatch (default: all `M` at once). Cases split into several blocks are
   * mixed in to exercise the row-block execution of ADR 0067 decision 7.
   */
  readonly rowsBlock?: number;
};

/**
 * Column stride of S (full = `C` / sliding = `(W−1) + M` — the resident range of ADR 0067 decision
 * 4).
 */
export const caseColCap = (spec: StateCase): number =>
  stateSliding(spec.window) ? spec.window - 1 + spec.chunkRows : spec.capacity;

/** Half scale (`√(1/√D)` — the same contract as the existing attention). */
export const halfScale = (depth: number): number => Math.fround(Math.sqrt(1 / Math.sqrt(depth)));

/** Deterministic data (no random numbers — a failure would not reproduce). */
export const seeded = (
  count: number,
  generator: (index: number) => number,
): Float32Array<ArrayBuffer> => {
  const data = new Float32Array(count);
  for (let index = 0; index < count; index += 1) data[index] = generator(index);
  return data;
};

/**
 * The 5 inputs ①③ read (the slots hold "the contents up to the previous step" — non-resident rows
 * are filled too).
 */
export type StateInputs = {
  readonly q: Float32Array<ArrayBuffer>;
  readonly insK: Float32Array<ArrayBuffer>;
  readonly insV: Float32Array<ArrayBuffer>;
  readonly slotK: Float32Array<ArrayBuffer>;
  readonly slotV: Float32Array<ArrayBuffer>;
};

/**
 * Fault injection that replaces a kernel's WGSL. A mutation returning `undefined` means "the
 * replacement target was not found", and the caller fails it with {@link assertMutated} (so a
 * mutation cannot silently become a no-op when the text changes).
 */
export type StateMutation = (kernel: "qk" | "stats" | "pv" | "append", wgsl: string) => string;

/**
 * Cache of final WGSL string → pipeline (sharing it across cases needs only one shader compile).
 * MUST: the key is **the WGSL string, not the variant key** (if a fault-injection mutant got the
 * correct version's pipeline, the injection would be "always green").
 */
export type StatePipelineCache = Map<string, GPUComputePipeline>;

const STORAGE = BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST | BUFFER_USAGE.COPY_SRC;
const UNIFORM = BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST;

const storageBuffer = (device: GPUDevice, data: Float32Array<ArrayBuffer>): GPUBuffer => {
  const buffer = device.createBuffer({ size: Math.max(4, data.byteLength), usage: STORAGE });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

const uniformBuffer = (device: GPUDevice, data: Uint32Array<ArrayBuffer>): GPUBuffer => {
  const buffer = device.createBuffer({ size: data.byteLength, usage: UNIFORM });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

/**
 * A slot's buffer: as given, or packed the way f16 storage holds it (element 0 in the low half of the
 * word). The harness rounds on the host, which is what the f16 readers must be equal to.
 */
const slotBuffer = (
  device: GPUDevice,
  slot: Float32Array<ArrayBuffer>,
  storage: StateStorage,
): GPUBuffer => {
  if (storage === "f32") return storageBuffer(device, slot);
  const packed = new Float16Array(slot);
  const buffer = device.createBuffer({ size: Math.max(4, packed.byteLength), usage: STORAGE });
  // As bytes: Dawn under Node takes a Float16Array view to an element size of 0 and dies on SIGFPE
  device.queue.writeBuffer(buffer, 0, new Uint8Array(packed.buffer));
  return buffer;
};

/**
 * Logical-length uniform (8 bytes of `{past, query}` — the same shape `GenerationContext` holds).
 */
const lengthsBuffer = (
  device: GPUDevice,
  past: number,
  query: number,
  block?: { readonly start: number; readonly end: number },
): GPUBuffer =>
  uniformBuffer(
    device,
    new Uint32Array(
      block === undefined ? [past, query] : [past, query, past + block.start, past + block.end],
    ),
  );

const readRaw = async (
  device: GPUDevice,
  buffer: GPUBuffer,
  count: number,
): Promise<ArrayBuffer> => {
  const size = Math.max(4, count * 4);
  const staging = device.createBuffer({
    size,
    usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(MAP_MODE.READ);
  const copy = staging.getMappedRange().slice(0);
  staging.unmap();
  staging.destroy();
  return copy;
};

const readFloats = async (
  device: GPUDevice,
  buffer: GPUBuffer,
  count: number,
): Promise<Float32Array<ArrayBuffer>> =>
  new Float32Array(await readRaw(device, buffer, count), 0, count);

/**
 * Reads back **as words (u32)**. The bit-identity check of ①ₜ against ① cannot be built from f32
 * comparisons — `-inf`, the poison and NaN, values `===` cannot compare, legitimately appear, so
 * comparing words is the only form that works.
 */
const readWords = async (
  device: GPUDevice,
  buffer: GPUBuffer,
  count: number,
): Promise<Uint32Array<ArrayBuffer>> =>
  new Uint32Array(await readRaw(device, buffer, count), 0, count);

const pipelineOf = (
  device: GPUDevice,
  cache: Map<string, GPUComputePipeline>,
  wgsl: string,
): GPUComputePipeline => {
  const hit = cache.get(wgsl);
  if (hit !== undefined) return hit;
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: wgsl }), entryPoint: "main" },
  });
  cache.set(wgsl, pipeline);
  return pipeline;
};

const bind = (
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  buffers: readonly GPUBuffer[],
): GPUBindGroup =>
  device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });

/**
 * Poison value S is filled with before ①QK writes it (if a reader reads beyond the live range, amax
 * is dominated by it).
 *
 * MUST: **a value exactly representable in f32** (a power of 2). A decimal literal like `1e30` is
 * rounded when stored into a Float32Array, and the `===` comparison with the read-back value is
 * **always false** — the check that "no poison is left" would be a no-op altogether (the magnitude
 * only has to clearly exceed the tolerance, so the exactly comparable choice is taken).
 */
export const STATE_S_POISON = 2 ** 100;

export type StateRunResult = {
  /** `[B,H,M,D]`. */
  readonly out: Float32Array<ArrayBuffer>;
  /**
   * S of **the last row block** `[B·H, rowsBlock, colCap]` (for checking the predicates directly).
   */
  readonly scores: Float32Array<ArrayBuffer>;
  /** Row statistics of the last row block `[B·H·rowsBlock, 2]`. */
  readonly stats: Float32Array<ArrayBuffer>;
};

/**
 * Dispatches the 3 stages of states-form attention directly, per row block.
 *
 * MUST: the output buffer is **initialized with the poison value** (the detector for the full-write
 * invariant that ③ writes every row; an implementation that skips pad rows shows up here as
 * leftover poison).
 */
export const runStateAttention = async (
  device: GPUDevice,
  spec: StateCase,
  inputs: StateInputs,
  options: {
    readonly mutate?: StateMutation;
    readonly cache?: StatePipelineCache;
    /** Checks ② / ③' fused, directly. */
    readonly statsPvFusion?: boolean;
    /**
     * Reduction form of ③ (default sequential = reference path; parallel = the KV-parallel
     * reduction variant ③').
     */
    readonly pvReduce?: "sequential" | "parallel";
    /**
     * Reduction form of ① (default sequential = reference path; parallel = the D-parallel
     * reduction variant ①').
     *
     * NOTE: the Session option (`stateAttentionReduce`) switches ① and ③ **together**, but this
     * direct dispatch harness can pick the two stages **separately**. Without pinning ③ to the
     * reference path while measuring the band of ①', a difference could not be attributed to ①' or
     * to ③'.
     */
    readonly qkReduce?: "sequential" | "parallel";
  } = {},
): Promise<StateRunResult> => {
  const { mutate } = options;
  const parallel = options.pvReduce === "parallel";
  const qkParallel = options.qkReduce === "parallel";
  const sliding = stateSliding(spec.window);
  const gqa = spec.heads !== spec.kvHeads;
  const colCap = caseColCap(spec);
  const rowsBlock = spec.rowsBlock ?? spec.chunkRows;
  const batchHeads = spec.batch * spec.heads;
  const cache = options.cache ?? new Map<string, GPUComputePipeline>();
  const wgsl = (kernel: "qk" | "stats" | "pv", source: string): string =>
    mutate === undefined ? source : mutate(kernel, source);
  const outCount = batchHeads * spec.chunkRows * spec.depth;
  const buffers: GPUBuffer[] = [];
  const track = <T extends GPUBuffer>(buffer: T): T => {
    buffers.push(buffer);
    return buffer;
  };
  try {
    const q = track(storageBuffer(device, inputs.q));
    const insK = track(storageBuffer(device, inputs.insK));
    const insV = track(storageBuffer(device, inputs.insV));
    const slotK = track(storageBuffer(device, inputs.slotK));
    const slotV = track(storageBuffer(device, inputs.slotV));
    const out = track(
      storageBuffer(
        device,
        seeded(outCount, () => STATE_S_POISON),
      ),
    );
    const lengths = track(lengthsBuffer(device, spec.past, spec.query));
    const sCount = batchHeads * rowsBlock * colCap;
    const scores = track(
      storageBuffer(
        device,
        seeded(sCount, () => STATE_S_POISON),
      ),
    );
    const stats = track(
      storageBuffer(
        device,
        seeded(batchHeads * rowsBlock * STATE_STATS_STRIDE, () => STATE_S_POISON),
      ),
    );
    for (let rowOffset = 0; rowOffset < spec.chunkRows; rowOffset += rowsBlock) {
      const block = Math.min(rowsBlock, spec.chunkRows - rowOffset);
      const geometry = {
        rowsBlock: block,
        rowOffset,
        chunkRows: spec.chunkRows,
        depth: spec.depth,
        kvRepeat: spec.heads / spec.kvHeads,
        window: spec.window,
        capacity: spec.capacity,
        colCap,
        scale: halfScale(spec.depth),
      };
      const dispatchGeometry = {
        batchHeads,
        rowsBlock: block,
        rowOffset,
        depth: spec.depth,
        window: spec.window,
      };
      const limit = device.limits.maxComputeWorkgroupsPerDimension;
      const params = track(uniformBuffer(device, stateAttentionParams(geometry)));
      const statsParams = track(
        uniformBuffer(device, stateStatsParams(batchHeads, block, rowOffset, colCap, spec.window)),
      );
      // MUST: reset to the poison per block (with the previous block's S left in place, the bug of
      // not writing the whole live range would be painted over by "the previous block's values" and
      // go undetected)
      device.queue.writeBuffer(
        scores,
        0,
        seeded(batchHeads * block * colCap, () => STATE_S_POISON),
      );
      const qk = pipelineOf(
        device,
        cache,
        wgsl("qk", qkParallel ? stateQkParallelWgsl(sliding, gqa) : stateQkWgsl(sliding, gqa)),
      );
      const st = options.statsPvFusion
        ? undefined
        : pipelineOf(device, cache, wgsl("stats", stateStatsWgsl(sliding)));
      const pv = pipelineOf(
        device,
        cache,
        wgsl(
          "pv",
          options.statsPvFusion
            ? stateStatsPvWgsl(sliding, gqa)
            : parallel
              ? statePvParallelWgsl(sliding, gqa)
              : statePvWgsl(sliding, gqa),
        ),
      );
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(qk);
      pass.setBindGroup(0, bind(device, qk, [params, q, insK, slotK, scores, lengths]));
      const qkGroups = qkParallel
        ? stateQkParallelWorkgroups(dispatchGeometry, spec.past, spec.query, limit, spec.name)
        : stateQkWorkgroups(dispatchGeometry, spec.past, spec.query, limit, spec.name);
      pass.dispatchWorkgroups(qkGroups[0], qkGroups[1], qkGroups[2]);
      if (st !== undefined) {
        pass.setPipeline(st);
        pass.setBindGroup(0, bind(device, st, [statsParams, scores, stats, lengths]));
        const statsGroups = stateStatsWorkgroups(dispatchGeometry, spec.query, limit, spec.name);
        pass.dispatchWorkgroups(statsGroups[0], statsGroups[1], statsGroups[2]);
      }
      pass.setPipeline(pv);
      pass.setBindGroup(
        0,
        options.statsPvFusion
          ? device.createBindGroup({
              layout: pv.getBindGroupLayout(0),
              entries: [params, scores, stats, insV, slotV, out, lengths]
                .map((buffer, binding) => ({ binding, resource: { buffer } }))
                .filter((entry) => entry.binding !== 2),
            })
          : bind(device, pv, [params, scores, stats, insV, slotV, out, lengths]),
      );
      const pvGroups = parallel
        ? statePvParallelWorkgroups(dispatchGeometry, limit, spec.name)
        : statePvWorkgroups(dispatchGeometry, limit, spec.name);
      pass.dispatchWorkgroups(pvGroups[0], pvGroups[1], pvGroups[2]);
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    const lastBlock = spec.chunkRows - Math.floor((spec.chunkRows - 1) / rowsBlock) * rowsBlock;
    return {
      out: await readFloats(device, out, outCount),
      scores: await readFloats(device, scores, batchHeads * lastBlock * colCap),
      stats: await readFloats(device, stats, batchHeads * lastBlock * STATE_STATS_STRIDE),
    };
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
};

/**
 * The 3 paths of ①QK (`sequential` = ① reference path / `parallel` = ①' D-parallel reduction /
 * `tiled` = ①ₜ shared K-row tiles). The binding numbers are the same on all 3 paths; only the WGSL,
 * the params and the workgroup computation change.
 * `folded` is ①' with the F = r query heads of one kv plane per workgroup (perf-ledger K-55 — GQA
 * full-attention cases only; the same bindings and params as ①').
 */
export type StateQkVariant = "sequential" | "parallel" | "tiled" | "folded";

/**
 * Dispatches **only** ①QK per row block and returns each block's S **as words (u32)**.
 *
 * ②③ are not stacked on top because the match of ①ₜ against ① is checked on S itself (going
 * through to O, the softmax normalization would flatten the differences, and neither leftovers in
 * `[live, col_cap)` nor unwritten pad rows could be observed).
 *
 * MUST: reset S to the poison per block (the previous block's leftovers would paint over "the bug
 * of not writing everything").
 * MUST: pipelines are cached by **the final WGSL string** (as in {@link runStateAttention} — if a
 * fault-injection mutant got the correct version's pipeline, the injection would always be green).
 */
export const runStateQk = async (
  device: GPUDevice,
  spec: StateCase,
  inputs: StateInputs,
  variant: StateQkVariant,
  options: {
    readonly mutate?: StateMutation;
    readonly cache?: StatePipelineCache;
    readonly storage?: StateStorage;
    /** The tiled form's predicate of `GenerationRun.bidirectionalRows` (ADR 0144 decision 3). */
    readonly bidirectionalRows?: { readonly start: number; readonly end: number };
  } = {},
): Promise<readonly Uint32Array<ArrayBuffer>[]> => {
  const { mutate, storage = "f32", bidirectionalRows } = options;
  const sliding = stateSliding(spec.window);
  const gqa = spec.heads !== spec.kvHeads;
  const colCap = caseColCap(spec);
  const rowsBlock = spec.rowsBlock ?? spec.chunkRows;
  const batchHeads = spec.batch * spec.heads;
  const cache = options.cache ?? new Map<string, GPUComputePipeline>();
  const kvRepeat = spec.heads / spec.kvHeads;
  if (bidirectionalRows !== undefined && variant !== "tiled") {
    throw new Error(`bidirectionalRows exists on the tiled form only, not ${variant}`);
  }
  const source =
    variant === "parallel"
      ? stateQkParallelWgsl(sliding, gqa, storage)
      : variant === "tiled"
        ? stateQkTiledWgsl(sliding, gqa, spec.chunkRows, storage, bidirectionalRows !== undefined)
        : variant === "folded"
          ? stateQkFoldedWgsl(kvRepeat, storage)
          : stateQkWgsl(sliding, gqa, storage);
  const buffers: GPUBuffer[] = [];
  const track = <T extends GPUBuffer>(buffer: T): T => {
    buffers.push(buffer);
    return buffer;
  };
  try {
    const q = track(storageBuffer(device, inputs.q));
    const insK = track(storageBuffer(device, inputs.insK));
    const slotK = track(slotBuffer(device, inputs.slotK, storage));
    const lengths = track(lengthsBuffer(device, spec.past, spec.query, bidirectionalRows));
    const scores = track(
      storageBuffer(
        device,
        seeded(batchHeads * rowsBlock * colCap, () => STATE_S_POISON),
      ),
    );
    const pipeline = pipelineOf(
      device,
      cache,
      mutate === undefined ? source : mutate("qk", source),
    );
    const limit = device.limits.maxComputeWorkgroupsPerDimension;
    const blocks: Uint32Array<ArrayBuffer>[] = [];
    for (let rowOffset = 0; rowOffset < spec.chunkRows; rowOffset += rowsBlock) {
      const block = Math.min(rowsBlock, spec.chunkRows - rowOffset);
      const geometry = {
        rowsBlock: block,
        rowOffset,
        chunkRows: spec.chunkRows,
        depth: spec.depth,
        kvRepeat,
        window: spec.window,
        capacity: spec.capacity,
        colCap,
        scale: halfScale(spec.depth),
      };
      const dispatchGeometry = {
        batchHeads,
        rowsBlock: block,
        rowOffset,
        depth: spec.depth,
        window: spec.window,
      };
      const params = track(
        uniformBuffer(
          device,
          variant === "tiled" ? stateQkTiledParams(geometry) : stateAttentionParams(geometry),
        ),
      );
      const words = batchHeads * block * colCap;
      device.queue.writeBuffer(
        scores,
        0,
        seeded(words, () => STATE_S_POISON),
      );
      const groups =
        variant === "parallel"
          ? stateQkParallelWorkgroups(dispatchGeometry, spec.past, spec.query, limit, spec.name)
          : variant === "tiled"
            ? stateQkTiledWorkgroups(
                dispatchGeometry,
                spec.chunkRows,
                spec.past,
                spec.query,
                limit,
                spec.name,
              )
            : variant === "folded"
              ? stateQkFoldedWorkgroups(
                  dispatchGeometry,
                  kvRepeat,
                  spec.past,
                  spec.query,
                  limit,
                  spec.name,
                )
              : stateQkWorkgroups(dispatchGeometry, spec.past, spec.query, limit, spec.name);
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind(device, pipeline, [params, q, insK, slotK, scores, lengths]));
      pass.dispatchWorkgroups(groups[0], groups[1], groups[2]);
      pass.end();
      device.queue.submit([encoder.finish()]);
      blocks.push(await readWords(device, scores, words));
    }
    return blocks;
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
};

/**
 * The 3 paths of ③PV (`sequential` = ③ reference path / `parallel` = ③' KV-parallel reduction /
 * `tiled` = ③ₜ shared V-row tiles). The binding numbers are the same on all 3 paths; only the WGSL,
 * the params and the workgroup computation change.
 */
export type StatePvVariant = "sequential" | "parallel" | "tiled";

/**
 * Dispatches ①②③ per row block and returns **the whole O buffer as words (u32)**. ①② always take
 * the reference path (①QK sequential + ②stats) and only ③ switches — in the comparison of ③ₜ
 * against ③, this guarantees by construction that a difference can come from nowhere but ③ (① is
 * deterministic too, so the S / row statistics the two runs consume are identical).
 *
 * MUST: O is **initialized with the poison value** and not repainted across row blocks (the
 * full-write invariant of ③ — an implementation that skips pad rows shows up in the word
 * comparison as leftover poison).
 * MUST: pipelines are cached by **the final WGSL string** (as in {@link runStateAttention} — if a
 * fault-injection mutant got the correct version's pipeline, the injection would always be green).
 */
export const runStatePv = async (
  device: GPUDevice,
  spec: StateCase,
  inputs: StateInputs,
  variant: StatePvVariant,
  options: {
    readonly mutate?: StateMutation;
    readonly cache?: StatePipelineCache;
    readonly storage?: StateStorage;
  } = {},
): Promise<Uint32Array<ArrayBuffer>> => {
  const { mutate, storage = "f32" } = options;
  const sliding = stateSliding(spec.window);
  const gqa = spec.heads !== spec.kvHeads;
  const colCap = caseColCap(spec);
  const rowsBlock = spec.rowsBlock ?? spec.chunkRows;
  const batchHeads = spec.batch * spec.heads;
  const cache = options.cache ?? new Map<string, GPUComputePipeline>();
  const wgsl = (kernel: "qk" | "stats" | "pv", source: string): string =>
    mutate === undefined ? source : mutate(kernel, source);
  const pvSource =
    variant === "parallel"
      ? statePvParallelWgsl(sliding, gqa, storage)
      : variant === "tiled"
        ? statePvTiledWgsl(sliding, gqa, spec.chunkRows, storage)
        : statePvWgsl(sliding, gqa, storage);
  const outCount = batchHeads * spec.chunkRows * spec.depth;
  const buffers: GPUBuffer[] = [];
  const track = <T extends GPUBuffer>(buffer: T): T => {
    buffers.push(buffer);
    return buffer;
  };
  try {
    const q = track(storageBuffer(device, inputs.q));
    const insK = track(storageBuffer(device, inputs.insK));
    const insV = track(storageBuffer(device, inputs.insV));
    const slotK = track(slotBuffer(device, inputs.slotK, storage));
    const slotV = track(slotBuffer(device, inputs.slotV, storage));
    const out = track(
      storageBuffer(
        device,
        seeded(outCount, () => STATE_S_POISON),
      ),
    );
    const lengths = track(lengthsBuffer(device, spec.past, spec.query));
    const scores = track(
      storageBuffer(
        device,
        seeded(batchHeads * rowsBlock * colCap, () => STATE_S_POISON),
      ),
    );
    const stats = track(
      storageBuffer(
        device,
        seeded(batchHeads * rowsBlock * STATE_STATS_STRIDE, () => STATE_S_POISON),
      ),
    );
    const qk = pipelineOf(device, cache, wgsl("qk", stateQkWgsl(sliding, gqa, storage)));
    const st = pipelineOf(device, cache, wgsl("stats", stateStatsWgsl(sliding)));
    const pv = pipelineOf(device, cache, wgsl("pv", pvSource));
    const limit = device.limits.maxComputeWorkgroupsPerDimension;
    for (let rowOffset = 0; rowOffset < spec.chunkRows; rowOffset += rowsBlock) {
      const block = Math.min(rowsBlock, spec.chunkRows - rowOffset);
      const geometry = {
        rowsBlock: block,
        rowOffset,
        chunkRows: spec.chunkRows,
        depth: spec.depth,
        kvRepeat: spec.heads / spec.kvHeads,
        window: spec.window,
        capacity: spec.capacity,
        colCap,
        scale: halfScale(spec.depth),
      };
      const dispatchGeometry = {
        batchHeads,
        rowsBlock: block,
        rowOffset,
        depth: spec.depth,
        window: spec.window,
      };
      const params = track(uniformBuffer(device, stateAttentionParams(geometry)));
      const pvParams =
        variant === "tiled" ? track(uniformBuffer(device, statePvTiledParams(geometry))) : params;
      const statsParams = track(
        uniformBuffer(device, stateStatsParams(batchHeads, block, rowOffset, colCap, spec.window)),
      );
      // MUST: reset S / row statistics to the poison per block (the previous block's leftovers
      // would paint over "the bug of not writing everything"). O alone accumulates the writes of
      // every block, so it is left alone.
      device.queue.writeBuffer(
        scores,
        0,
        seeded(batchHeads * block * colCap, () => STATE_S_POISON),
      );
      device.queue.writeBuffer(
        stats,
        0,
        seeded(batchHeads * block * STATE_STATS_STRIDE, () => STATE_S_POISON),
      );
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(qk);
      pass.setBindGroup(0, bind(device, qk, [params, q, insK, slotK, scores, lengths]));
      const qkGroups = stateQkWorkgroups(dispatchGeometry, spec.past, spec.query, limit, spec.name);
      pass.dispatchWorkgroups(qkGroups[0], qkGroups[1], qkGroups[2]);
      pass.setPipeline(st);
      pass.setBindGroup(0, bind(device, st, [statsParams, scores, stats, lengths]));
      const statsGroups = stateStatsWorkgroups(dispatchGeometry, spec.query, limit, spec.name);
      pass.dispatchWorkgroups(statsGroups[0], statsGroups[1], statsGroups[2]);
      pass.setPipeline(pv);
      pass.setBindGroup(0, bind(device, pv, [pvParams, scores, stats, insV, slotV, out, lengths]));
      const pvGroups =
        variant === "parallel"
          ? statePvParallelWorkgroups(dispatchGeometry, limit, spec.name)
          : variant === "tiled"
            ? statePvTiledWorkgroups(dispatchGeometry, spec.chunkRows, limit, spec.name)
            : statePvWorkgroups(dispatchGeometry, limit, spec.name);
      pass.dispatchWorkgroups(pvGroups[0], pvGroups[1], pvGroups[2]);
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    return await readWords(device, out, outCount);
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
};

export type StateAppendCase = {
  readonly kvPlanes: number;
  readonly chunkRows: number;
  readonly depth: number;
  readonly capacity: number;
  readonly window: number;
  readonly past: number;
  readonly query: number;
};

/** Dispatches `state_append` directly and reads back **the whole slot after the write**. */
export const runStateAppend = async (
  device: GPUDevice,
  spec: StateAppendCase,
  x: Float32Array<ArrayBuffer>,
  slot: Float32Array<ArrayBuffer>,
  options: { readonly mutate?: StateMutation; readonly cache?: StatePipelineCache } = {},
): Promise<Float32Array<ArrayBuffer>> => {
  const { mutate } = options;
  const cache = options.cache ?? new Map<string, GPUComputePipeline>();
  const source = stateAppendWgsl(stateSliding(spec.window));
  const buffers: GPUBuffer[] = [];
  try {
    const params = uniformBuffer(device, stateAppendParams(spec));
    const xBuffer = storageBuffer(device, x);
    const slotBuffer = storageBuffer(device, slot);
    const lengths = lengthsBuffer(device, spec.past, spec.query);
    buffers.push(params, xBuffer, slotBuffer, lengths);
    const pipeline = pipelineOf(
      device,
      cache,
      mutate === undefined ? source : mutate("append", source),
    );
    const groups = stateAppendWorkgroups(
      spec,
      spec.query,
      device.limits.maxComputeWorkgroupsPerDimension,
      "state_append",
    );
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind(device, pipeline, [params, xBuffer, slotBuffer, lengths]));
    pass.dispatchWorkgroups(groups[0], groups[1], groups[2]);
    pass.end();
    device.queue.submit([encoder.finish()]);
    return await readFloats(device, slotBuffer, slot.length);
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
};

/**
 * Dispatches the f16 `state_append` directly and reads back **the whole slot as words**, each holding
 * two elements. `poison` fills the slot first, so a word the kernel must not write shows.
 */
export const runStateAppendF16 = async (
  device: GPUDevice,
  spec: StateAppendCase,
  x: Float32Array<ArrayBuffer>,
  words: number,
  poison: number,
  options: { readonly mutate?: StateMutation; readonly cache?: StatePipelineCache } = {},
): Promise<Uint32Array<ArrayBuffer>> => {
  const { mutate } = options;
  const cache = options.cache ?? new Map<string, GPUComputePipeline>();
  const source = stateAppendWgsl(false, "f16");
  const buffers: GPUBuffer[] = [];
  try {
    const params = uniformBuffer(device, stateAppendParams(spec, "f16"));
    const xBuffer = storageBuffer(device, x);
    const slot = device.createBuffer({ size: Math.max(4, words * 4), usage: STORAGE });
    device.queue.writeBuffer(slot, 0, new Uint32Array(words).fill(poison));
    const lengths = lengthsBuffer(device, spec.past, spec.query);
    buffers.push(params, xBuffer, slot, lengths);
    const pipeline = pipelineOf(
      device,
      cache,
      mutate === undefined ? source : mutate("append", source),
    );
    const groups = stateAppendWorkgroups(
      { kvPlanes: spec.kvPlanes, depth: spec.depth / 2 },
      spec.query,
      device.limits.maxComputeWorkgroupsPerDimension,
      "state_append",
    );
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind(device, pipeline, [params, xBuffer, slot, lengths]));
    pass.dispatchWorkgroups(groups[0], groups[1], groups[2]);
    pass.end();
    device.queue.submit([encoder.finish()]);
    return await readWords(device, slot, words);
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
};

/**
 * The words of S after fused ①QK with f16 S (`attentionQkWgsl`, f32 compute, the default geometry),
 * filled with a poison value first. Not a states-form kernel, but it needs the same direct dispatch.
 */
export const runAttentionQkS16 = async (
  device: GPUDevice,
  shape: { readonly rows: number; readonly columns: number; readonly depth: number },
  q: Float32Array<ArrayBuffer>,
  k: Float32Array<ArrayBuffer>,
  options: { readonly mutate?: (wgsl: string) => string } = {},
): Promise<Uint32Array<ArrayBuffer>> => {
  const { rows, columns, depth } = shape;
  const words = (rows * columns) / 2;
  const source = attentionQkWgsl(true, "f32", "f16");
  const buffers: GPUBuffer[] = [];
  try {
    const params = uniformBuffer(device, attentionQkParams(rows, columns, depth, 1));
    const qBuffer = storageBuffer(device, q);
    const kBuffer = storageBuffer(device, k);
    const scores = device.createBuffer({ size: words * 4, usage: STORAGE });
    device.queue.writeBuffer(scores, 0, new Uint32Array(words).fill(0xdeadbeef));
    buffers.push(params, qBuffer, kBuffer, scores);
    const pipeline = pipelineOf(
      device,
      new Map(),
      options.mutate === undefined ? source : options.mutate(source),
    );
    const limit = device.limits.maxComputeWorkgroupsPerDimension;
    const geometry = defaultGemmGeometry();
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind(device, pipeline, [params, qBuffer, kBuffer, scores]));
    pass.dispatchWorkgroups(
      tiledWorkgroups(columns, gemmTileN(geometry), limit, "score store ①QK"),
      tiledWorkgroups(rows, gemmTileM(geometry), limit, "score store ①QK"),
      1,
    );
    pass.end();
    device.queue.submit([encoder.finish()]);
    return await readWords(device, scores, words);
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
};

/**
 * Check that the fault-injection replacement **actually took effect**.
 *
 * MUST: if a mutation is a no-op (the replacement target disappeared through a text change), the
 * mutant is identical to the correct version, and "green under fault injection = not detected"
 * could no longer be told apart from "there was no mutation".
 */
export const assertMutated = (before: string, after: string, label: string): void => {
  if (before === after) {
    throw new Error(
      `fault injection '${label}' was a no-op (the replacement target is not in the WGSL)`,
    );
  }
};

/** Inputs of readonly (reads only past) — `q [B,H,1,D]` and the lender's two slots. */
export type StateReadonlyInputs = {
  readonly q: Float32Array<ArrayBuffer>;
  readonly slotK: Float32Array<ArrayBuffer>;
  readonly slotV: Float32Array<ArrayBuffer>;
};

/**
 * Dispatches the readonly variants (①' ro → ② ro → ③' ro) directly. `spec.chunkRows` /
 * `spec.query` are fixed at 1 (the contract of the borrowing context), and `spec.past` is the
 * lender's logical length P.
 *
 * MUST: the output and S are initialized with the poison value (the same detector as
 * {@link runStateAttention} — that the empty rows of P = 0 are written as exact 0 and that nothing
 * beyond the live range is read are visible only through the poison).
 */
export const runStateAttentionReadonly = async (
  device: GPUDevice,
  spec: StateCase,
  inputs: StateReadonlyInputs,
  options: {
    readonly mutate?: StateMutation;
    readonly cache?: StatePipelineCache;
    readonly storage?: StateStorage;
  } = {},
): Promise<StateRunResult> => {
  if (spec.chunkRows !== 1 || spec.query !== 1) {
    throw new Error(
      `readonly is fixed at M = Q = 1 (${spec.name}: M=${spec.chunkRows} Q=${spec.query})`,
    );
  }
  const { mutate, storage = "f32" } = options;
  const sliding = stateSliding(spec.window);
  const gqa = spec.heads !== spec.kvHeads;
  const colCap = caseColCap(spec);
  const batchHeads = spec.batch * spec.heads;
  const cache = options.cache ?? new Map<string, GPUComputePipeline>();
  const wgsl = (kernel: "qk" | "stats" | "pv", source: string): string =>
    mutate === undefined ? source : mutate(kernel, source);
  const outCount = batchHeads * spec.depth;
  const buffers: GPUBuffer[] = [];
  const track = <T extends GPUBuffer>(buffer: T): T => {
    buffers.push(buffer);
    return buffer;
  };
  try {
    const q = track(storageBuffer(device, inputs.q));
    const slotK = track(slotBuffer(device, inputs.slotK, storage));
    const slotV = track(slotBuffer(device, inputs.slotV, storage));
    const out = track(
      storageBuffer(
        device,
        seeded(outCount, () => STATE_S_POISON),
      ),
    );
    const lengths = track(lengthsBuffer(device, spec.past, 1));
    const sCount = batchHeads * colCap;
    const scores = track(
      storageBuffer(
        device,
        seeded(sCount, () => STATE_S_POISON),
      ),
    );
    const stats = track(
      storageBuffer(
        device,
        seeded(batchHeads * STATE_STATS_STRIDE, () => STATE_S_POISON),
      ),
    );
    const geometry = {
      rowsBlock: 1,
      rowOffset: 0,
      chunkRows: 1,
      depth: spec.depth,
      kvRepeat: spec.heads / spec.kvHeads,
      window: spec.window,
      capacity: spec.capacity,
      colCap,
      scale: halfScale(spec.depth),
    };
    const dispatchGeometry = {
      batchHeads,
      rowsBlock: 1,
      rowOffset: 0,
      depth: spec.depth,
      window: spec.window,
    };
    const limit = device.limits.maxComputeWorkgroupsPerDimension;
    const params = track(uniformBuffer(device, stateAttentionParams(geometry)));
    const statsParams = track(
      uniformBuffer(device, stateStatsParams(batchHeads, 1, 0, colCap, spec.window)),
    );
    const qk = pipelineOf(
      device,
      cache,
      wgsl("qk", stateQkParallelReadonlyWgsl(sliding, gqa, storage)),
    );
    const st = pipelineOf(device, cache, wgsl("stats", stateStatsReadonlyWgsl(sliding)));
    const pv = pipelineOf(
      device,
      cache,
      wgsl("pv", statePvParallelReadonlyWgsl(sliding, gqa, storage)),
    );
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(qk);
    pass.setBindGroup(0, bind(device, qk, [params, q, slotK, scores, lengths]));
    const qkGroups = stateQkParallelReadonlyWorkgroups(
      dispatchGeometry,
      spec.past,
      limit,
      spec.name,
    );
    pass.dispatchWorkgroups(qkGroups[0], qkGroups[1], qkGroups[2]);
    pass.setPipeline(st);
    pass.setBindGroup(0, bind(device, st, [statsParams, scores, stats, lengths]));
    const statsGroups = stateStatsReadonlyWorkgroups(dispatchGeometry, limit, spec.name);
    pass.dispatchWorkgroups(statsGroups[0], statsGroups[1], statsGroups[2]);
    pass.setPipeline(pv);
    pass.setBindGroup(0, bind(device, pv, [params, scores, stats, slotV, out, lengths]));
    const pvGroups = statePvParallelReadonlyWorkgroups(dispatchGeometry, limit, spec.name);
    pass.dispatchWorkgroups(pvGroups[0], pvGroups[1], pvGroups[2]);
    pass.end();
    device.queue.submit([encoder.finish()]);
    return {
      out: await readFloats(device, out, outCount),
      scores: await readFloats(device, scores, sCount),
      stats: await readFloats(device, stats, batchHeads * STATE_STATS_STRIDE),
    };
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
};
