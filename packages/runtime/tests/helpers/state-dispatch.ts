// gpu_state_attention_test.ts の**直接 dispatch ハーネス**（states 形 attention 3 段 +
// `state_append`）。executor / recipe-builder への結線は波 D-3 の担当なので、本波の実 GPU 検証は
// `createComputePipeline` + 手組みバッファでカーネルを直接撃つ。依存は helpers → src の一方向。
//
// MUST: パイプラインは**最終 WGSL 文字列**でキャッシュする（`PipelineCache` を使わない）。
// 故障注入は WGSL を書き換えた**別の文字列**を撃つので、キー（= 変種）でキャッシュすると
// 正常版のパイプラインが変異版の dispatch に配られ、故障注入が「常に緑」になる。
// MUST: S は毎回**毒値で埋めてから** ①QK を撃つ（S は一時バッファで、実運用では前回の残骸が
// 居る）。①が live 範囲を書き切らない誤りと、②③ が live の外まで読む誤りの両方が、この
// 毒値でしか値に出ない。

import {
  STATE_STATS_STRIDE,
  stateAttentionParams,
  statePvWgsl,
  statePvWorkgroups,
  stateQkWgsl,
  stateQkWorkgroups,
  stateSliding,
  stateStatsParams,
  stateStatsWgsl,
  stateStatsWorkgroups,
} from "../../src/kernels/state-attention.ts";
import {
  stateAppendParams,
  stateAppendWgsl,
  stateAppendWorkgroups,
} from "../../src/kernels/state-append.ts";

/** 1 ケースぶんの形と論理長（記号は ADR 0067 決定 4）。 */
export type StateCase = {
  readonly name: string;
  /** `B`。 */
  readonly batch: number;
  /** `H`。 */
  readonly heads: number;
  /** `Hkv`（`H % Hkv == 0`）。 */
  readonly kvHeads: number;
  /** `M`（物理 chunk 行数）。 */
  readonly chunkRows: number;
  /** `D`。 */
  readonly depth: number;
  /** `C`（スロットの行容量）。 */
  readonly capacity: number;
  /** `W`（`0` = full）。 */
  readonly window: number;
  /** `P`。 */
  readonly past: number;
  /** `Q`。 */
  readonly query: number;
  /**
   * 1 dispatch が担当する行数（省略時は `M` 一括）。ADR 0067 決定 7 の行ブロック実行を
   * 踏ませるために、複数ブロックへ割るケースを混ぜる。
   */
  readonly rowsBlock?: number;
};

/** S の列ストライド（full = `C` / sliding = `(W−1) + M` — ADR 0067 決定 4 の resident 範囲）。 */
export const caseColCap = (spec: StateCase): number =>
  stateSliding(spec.window) ? spec.window - 1 + spec.chunkRows : spec.capacity;

/** 半スケール（`√(1/√D)` — 既存 attention と同じ契約）。 */
export const halfScale = (depth: number): number => Math.fround(Math.sqrt(1 / Math.sqrt(depth)));

/** 決定的なデータ列（乱数は使わない — 失敗が再現しないため）。 */
export const seeded = (count: number, generator: (index: number) => number): Float32Array<
  ArrayBuffer
> => {
  const data = new Float32Array(count);
  for (let index = 0; index < count; index += 1) data[index] = generator(index);
  return data;
};

/** ①③ が読む 5 本の入力（スロットは「前 step までの内容」— 非 resident 行も埋まっている）。 */
export type StateInputs = {
  readonly q: Float32Array<ArrayBuffer>;
  readonly insK: Float32Array<ArrayBuffer>;
  readonly insV: Float32Array<ArrayBuffer>;
  readonly slotK: Float32Array<ArrayBuffer>;
  readonly slotV: Float32Array<ArrayBuffer>;
};

/**
 * カーネルの WGSL を差し替える故障注入。`undefined` を返した変異は「置換対象が見つからない」
 * ので、呼び手が {@link assertMutated} で落とす（文言が変わって変異が空振りするのを防ぐ）。
 */
export type StateMutation = (kernel: "qk" | "stats" | "pv" | "append", wgsl: string) => string;

/**
 * 最終 WGSL 文字列 → パイプラインのキャッシュ（ケース間で共有すると shader compile が 1 回で
 * 済む）。MUST: キーは**変種キーではなく WGSL 文字列**（故障注入の変異版が正常版のパイプライン
 * を引くと、注入が「常に緑」になる）。
 */
export type StatePipelineCache = Map<string, GPUComputePipeline>;

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
const UNIFORM = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

const storageBuffer = (
  device: GPUDevice,
  data: Float32Array<ArrayBuffer>,
): GPUBuffer => {
  const buffer = device.createBuffer({ size: Math.max(4, data.byteLength), usage: STORAGE });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

const uniformBuffer = (device: GPUDevice, data: Uint32Array<ArrayBuffer>): GPUBuffer => {
  const buffer = device.createBuffer({ size: data.byteLength, usage: UNIFORM });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

/** 論理長 uniform（`{past, query}` の 8 バイト — `GenerationContext` が持つのと同じ形）。 */
const lengthsBuffer = (device: GPUDevice, past: number, query: number): GPUBuffer =>
  uniformBuffer(device, new Uint32Array([past, query]));

const readFloats = async (
  device: GPUDevice,
  buffer: GPUBuffer,
  count: number,
): Promise<Float32Array<ArrayBuffer>> => {
  const size = Math.max(4, count * 4);
  const staging = device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const copy = staging.getMappedRange().slice(0);
  staging.unmap();
  staging.destroy();
  return new Float32Array(copy, 0, count);
};

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

/** ①QK が S へ書く前に埋める毒値（読者が live の外を読んだら amax がこれに支配される）。 */
export const STATE_S_POISON = 1e30;

export type StateRunResult = {
  /** `[B,H,M,D]`。 */
  readonly out: Float32Array<ArrayBuffer>;
  /** **最後の行ブロック**の S `[B·H, rowsBlock, colCap]`（述語の直接検査用）。 */
  readonly scores: Float32Array<ArrayBuffer>;
  /** 最後の行ブロックの行統計 `[B·H·rowsBlock, 2]`。 */
  readonly stats: Float32Array<ArrayBuffer>;
};

/**
 * states 形 attention の 3 段を行ブロックごとに直接 dispatch する。
 *
 * MUST: 出力バッファは**毒値で初期化**する（③ が全行を書く = full-write 不変条件の検出線。
 * pad 行を書かない実装はここで残った毒値として見える）。
 */
export const runStateAttention = async (
  device: GPUDevice,
  spec: StateCase,
  inputs: StateInputs,
  options: { readonly mutate?: StateMutation; readonly cache?: StatePipelineCache } = {},
): Promise<StateRunResult> => {
  const { mutate } = options;
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
      storageBuffer(device, seeded(outCount, () => STATE_S_POISON)),
    );
    const lengths = track(lengthsBuffer(device, spec.past, spec.query));
    const sCount = batchHeads * rowsBlock * colCap;
    const scores = track(storageBuffer(device, seeded(sCount, () => STATE_S_POISON)));
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
        depth: spec.depth,
        window: spec.window,
      };
      const limit = device.limits.maxComputeWorkgroupsPerDimension;
      const params = track(uniformBuffer(device, stateAttentionParams(geometry)));
      const statsParams = track(
        uniformBuffer(device, stateStatsParams(batchHeads * block, colCap, spec.window)),
      );
      // MUST: ブロックごとに毒値へ戻す（前ブロックの S が残っていると、live 範囲を書き切らない
      // 誤りが「前ブロックの値」で塗り潰されて検出できない）
      device.queue.writeBuffer(
        scores,
        0,
        seeded(batchHeads * block * colCap, () => STATE_S_POISON),
      );
      const qk = pipelineOf(device, cache, wgsl("qk", stateQkWgsl(sliding, gqa)));
      const st = pipelineOf(device, cache, wgsl("stats", stateStatsWgsl(sliding)));
      const pv = pipelineOf(device, cache, wgsl("pv", statePvWgsl(sliding, gqa)));
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(qk);
      pass.setBindGroup(0, bind(device, qk, [params, q, insK, slotK, scores, lengths]));
      const qkGroups = stateQkWorkgroups(dispatchGeometry, spec.past, spec.query, limit, spec.name);
      pass.dispatchWorkgroups(qkGroups[0], qkGroups[1], qkGroups[2]);
      pass.setPipeline(st);
      pass.setBindGroup(0, bind(device, st, [statsParams, scores, stats, lengths]));
      const statsGroups = stateStatsWorkgroups(dispatchGeometry, limit, spec.name);
      pass.dispatchWorkgroups(statsGroups[0], statsGroups[1], statsGroups[2]);
      pass.setPipeline(pv);
      pass.setBindGroup(0, bind(device, pv, [params, scores, stats, insV, slotV, out, lengths]));
      const pvGroups = statePvWorkgroups(dispatchGeometry, limit, spec.name);
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

export type StateAppendCase = {
  readonly kvPlanes: number;
  readonly chunkRows: number;
  readonly depth: number;
  readonly capacity: number;
  readonly window: number;
  readonly past: number;
  readonly query: number;
};

/** `state_append` を直接 dispatch し、**書き込み後のスロット全体**を読み戻す。 */
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
 * 故障注入の置換が**実際に効いた**ことの門。
 *
 * MUST: 変異が空振り（置換対象が文言変更で消えた）だと、変異版が正常版と同一になり
 * 「故障注入しても緑 = 検出できていない」を「変異が無かった」と区別できなくなる。
 */
export const assertMutated = (before: string, after: string, label: string): void => {
  if (before === after) {
    throw new Error(`故障注入 '${label}' が空振りした（置換対象が WGSL に無い）`);
  }
};
