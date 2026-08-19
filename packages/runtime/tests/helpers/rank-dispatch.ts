// argmax / topk の**直接 dispatch ハーネス**（ADR 0068 受入条件③ の故障注入用）。通常の門は
// Session 経由（executor + `PipelineCache`）で撃つが、故障注入は WGSL を書き換えた**別の文字列**
// を撃つ必要があるため、ここでは `createComputePipeline` + 手組みバッファで直接撃つ。
// 依存は helpers → src の一方向。
//
// MUST: パイプラインは**最終 WGSL 文字列**でキャッシュする（`PipelineCache` を使わない —
// tests/helpers/state-dispatch.ts 冒頭と同じ理由）。キー（= 変種）でキャッシュすると正常版の
// パイプラインが変異版の dispatch に配られ、故障注入が「常に緑」になる。

import { gridStrideWorkgroups } from "../../src/codegen/dispatch.ts";
import { ARGMAX_WGSL, argmaxParams } from "../../src/kernels/argmax.ts";
import { topkParams, topkWgsl } from "../../src/kernels/topk.ts";

/** WGSL を書き換える故障注入（空振りの検出は呼び手の `assertMutated`）。 */
export type RankMutation = (wgsl: string) => string;

/** 最終 WGSL 文字列 → パイプライン（キーではなく**文字列**がキー — 上の MUST）。 */
export type RankPipelineCache = Map<string, GPUComputePipeline>;

/** 平坦化した入力の形（`[rows, dim]` — 先行次元は呼び手が畳んでおく）。 */
export type RankShape = {
  readonly rows: number;
  readonly dim: number;
};

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

const pipelineOf = (
  device: GPUDevice,
  cache: RankPipelineCache,
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

const inputBuffer = (device: GPUDevice, data: Float32Array<ArrayBuffer>): GPUBuffer => {
  const buffer = device.createBuffer({ size: Math.max(4, data.byteLength), usage: STORAGE });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

const uniformBuffer = (device: GPUDevice, data: Uint32Array<ArrayBuffer>): GPUBuffer => {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

const outputBuffer = (device: GPUDevice, count: number): GPUBuffer =>
  device.createBuffer({ size: Math.max(4, count * 4), usage: STORAGE });

const readBytes = async (
  device: GPUDevice,
  buffer: GPUBuffer,
  count: number,
): Promise<ArrayBuffer> => {
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
  return copy;
};

/** 1 dispatch を撃つ（束縛は binding 1 以降が引数の順 — 0 は params 固定）。 */
const dispatch = (
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  buffers: readonly GPUBuffer[],
  groups: number,
): void => {
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(groups, 1, 1);
  pass.end();
  device.queue.submit([encoder.finish()]);
};

/** argmax を直接撃って添字列（`[rows]`）を返す。 */
export const runArgmaxDirect = async (
  device: GPUDevice,
  shape: RankShape,
  x: Float32Array<ArrayBuffer>,
  options: { readonly mutate?: RankMutation; readonly cache?: RankPipelineCache } = {},
): Promise<Int32Array<ArrayBuffer>> => {
  const cache = options.cache ?? new Map<string, GPUComputePipeline>();
  const wgsl = options.mutate === undefined ? ARGMAX_WGSL : options.mutate(ARGMAX_WGSL);
  const buffers: GPUBuffer[] = [];
  try {
    const params = uniformBuffer(device, argmaxParams(shape.rows, shape.dim));
    const input = inputBuffer(device, x);
    const out = outputBuffer(device, shape.rows);
    buffers.push(params, input, out);
    // 1 行 = 1 workgroup（上限超過は行 grid-stride 側で回る — recipe-builder と同じ式）
    const groups = gridStrideWorkgroups(
      shape.rows,
      1,
      device.limits.maxComputeWorkgroupsPerDimension,
    );
    dispatch(device, pipelineOf(device, cache, wgsl), [params, input, out], groups);
    return new Int32Array(await readBytes(device, out, shape.rows), 0, shape.rows);
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
};

/** topk の 2 出力（値 f32 の降順 + 添字 i32・どちらも `[rows, k]`）。 */
export type TopkDirectResult = {
  readonly values: Float32Array<ArrayBuffer>;
  readonly indices: Int32Array<ArrayBuffer>;
};

/** topk を直接撃って値列と添字列を返す（`k` は WGSL に焼かれるので shape と別に渡す）。 */
export const runTopkDirect = async (
  device: GPUDevice,
  shape: RankShape & { readonly k: number },
  x: Float32Array<ArrayBuffer>,
  options: { readonly mutate?: RankMutation; readonly cache?: RankPipelineCache } = {},
): Promise<TopkDirectResult> => {
  const cache = options.cache ?? new Map<string, GPUComputePipeline>();
  const source = topkWgsl(shape.k);
  const wgsl = options.mutate === undefined ? source : options.mutate(source);
  const count = shape.rows * shape.k;
  const buffers: GPUBuffer[] = [];
  try {
    const params = uniformBuffer(device, topkParams(shape.rows, shape.dim));
    const input = inputBuffer(device, x);
    const values = outputBuffer(device, count);
    const indices = outputBuffer(device, count);
    buffers.push(params, input, values, indices);
    const groups = gridStrideWorkgroups(
      shape.rows,
      1,
      device.limits.maxComputeWorkgroupsPerDimension,
    );
    dispatch(
      device,
      pipelineOf(device, cache, wgsl),
      [params, input, values, indices],
      groups,
    );
    return {
      values: new Float32Array(await readBytes(device, values, count), 0, count),
      indices: new Int32Array(await readBytes(device, indices, count), 0, count),
    };
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
};
