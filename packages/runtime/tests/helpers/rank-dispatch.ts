// **Direct dispatch harness** for argmax / topk (for the fault injection of ADR 0068 acceptance
// criterion ③). The regular checks dispatch through a Session (executor + `PipelineCache`), but
// fault injection has to dispatch **a different string** with rewritten WGSL, so this dispatches
// directly with `createComputePipeline` + hand-built buffers. Dependencies go one way, helpers →
// src.
//
// MUST: pipelines are cached by **the final WGSL string** (no `PipelineCache` — for the same reason
// as the top of tests/helpers/state-dispatch.ts). Caching by key (= variant) would hand the correct
// version's pipeline to the mutant's dispatch, and the fault injection would be "always green".

import { gridStrideWorkgroups } from "../../src/codegen/dispatch.ts";
import { ARGMAX_WGSL, argmaxParams } from "../../src/kernels/argmax.ts";
import { topkParams, topkWgsl } from "../../src/kernels/topk.ts";

/** Fault injection that rewrites WGSL (the caller's `assertMutated` detects a no-op rewrite). */
export type RankMutation = (wgsl: string) => string;

/** Final WGSL string → pipeline (the key is **the string**, not a cache key — the MUST above). */
export type RankPipelineCache = Map<string, GPUComputePipeline>;

/** Shape of the flattened input (`[rows, dim]` — the caller folds the leading dimensions). */
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

/** Issues one dispatch (bindings from 1 on follow the argument order — 0 is always params). */
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

/** Dispatches argmax directly and returns the indices (`[rows]`). */
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
    // 1 row = 1 workgroup (beyond the limit the row grid-stride loops — the same formula as
    // recipe-builder)
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

/** The two outputs of topk (f32 values in descending order + i32 indices, both `[rows, k]`). */
export type TopkDirectResult = {
  readonly values: Float32Array<ArrayBuffer>;
  readonly indices: Int32Array<ArrayBuffer>;
};

/**
 * Dispatches topk directly and returns values and indices (`k` is baked into the WGSL, so it is
 * passed apart from the shape).
 */
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
    dispatch(device, pipelineOf(device, cache, wgsl), [params, input, values, indices], groups);
    return {
      values: new Float32Array(await readBytes(device, values, count), 0, count),
      indices: new Int32Array(await readBytes(device, indices, count), 0, count),
    };
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
};
