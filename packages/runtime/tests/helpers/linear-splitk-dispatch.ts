/**
 * One linear through the reference GEMM and through the split-K GEMM (the split partial at a slice
 * count, then the combine), by raw dispatches at an explicit tile geometry, with every output word
 * read back — the harness of the split-K checks (the runtime's GPU test and
 * `@karume/kernel-checks`).
 *
 * The kernels are the generators' own WGSL (`linearWgsl`, `linearSplitKGemmWgsl`,
 * `linearSplitKCombineWgsl`), pipelines cached by the final WGSL string, so a fault injection that
 * rewrites the split partial's text (`mutate`) runs its own string. Every buffer the kernels write
 * starts filled with {@link SPLITK_UNWRITTEN} and is read back whole, past the words they should
 * write, so a store past `m · n` (or past `slices · m · n` in the partials) shows.
 *
 * MUST NOT run two cases at once on one device: each case's GPU work runs inside one validation
 * error scope, and error scopes are a per-device stack.
 *
 * Browser-safe (no `node:` import, no WebGPU global read at module scope — the usage flags come
 * from the runtime's constants).
 */

import type { GpuContext } from "../../src/gpu/device.ts";
import { BUFFER_USAGE as U, MAP_MODE } from "../../src/gpu/webgpu-constants.ts";
import { gemmUsesVec4 } from "../../src/kernels/gemm.ts";
import { type GemmGeometry, gemmTileM, gemmTileN } from "../../src/kernels/gemm-geometry.ts";
import { LINEAR_SCALE_BINDING, linearWgsl } from "../../src/kernels/linear.ts";
import {
  LINEAR_SPLITK_COMBINE_GROUP,
  linearSplitKCombineWgsl,
  linearSplitKGemmWgsl,
  linearSplitKParams,
} from "../../src/kernels/linear-splitk.ts";
import type { EncodedLinearWeight } from "./linear-storage-weights.ts";

/** The word every buffer a kernel writes starts with (a quiet NaN). */
export const SPLITK_UNWRITTEN = 0x7fc00000;

/** Words read past the end of each output: one 128-column tile row, more than any store could miss by. */
export const SPLITK_TAIL = 128;

export type SplitKDispatchCase = {
  readonly geometry: GemmGeometry;
  readonly m: number;
  readonly n: number;
  readonly k: number;
  readonly slices: number;
  readonly weight: EncodedLinearWeight;
  /** `[m, k]`. */
  readonly x: Float32Array<ArrayBuffer>;
  /** `[n]`. */
  readonly bias: Float32Array<ArrayBuffer>;
};

export type SplitKDispatchRun = {
  /** The reference GEMM's `m · n` output words. */
  readonly reference: Uint32Array<ArrayBuffer>;
  /** The split's `m · n` output words (after the combine). */
  readonly split: Uint32Array<ArrayBuffer>;
  /** The split's output words of a second dispatch of the same input. */
  readonly again: Uint32Array<ArrayBuffer>;
  /** The words past `m · n` of the reference's and the split's output buffers (must stay unwritten). */
  readonly tails: Uint32Array<ArrayBuffer>;
  /** The partial pass's `slices · m · n` words (slice-major), then its tail. */
  readonly partials: Uint32Array<ArrayBuffer>;
};

export type SplitKRunnerOptions = {
  /**
   * Rewrites the split partial's WGSL before it is compiled (a fault injection). A rewrite that
   * changes nothing throws, so a stale anchor cannot make an injected fault pass.
   */
  readonly mutate?: (wgsl: string) => string;
};

/**
 * A runner over one device: `run(case)` dispatches the reference GEMM, then the split twice, and
 * reads every written buffer back. Pipelines are cached by WGSL for the runner's lifetime.
 */
export const splitKRunner = (
  gpu: GpuContext,
  options: SplitKRunnerOptions = {},
): ((testCase: SplitKDispatchCase) => Promise<SplitKDispatchRun>) => {
  const { device } = gpu;
  const pipelines = new Map<string, Promise<GPUComputePipeline>>();
  const pipelineOf = (code: string): Promise<GPUComputePipeline> => {
    const cached = pipelines.get(code);
    if (cached !== undefined) return cached;
    const created = device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
    });
    pipelines.set(code, created);
    return created;
  };
  const partialWgsl = (wgsl: string): string => {
    if (options.mutate === undefined) return wgsl;
    const mutated = options.mutate(wgsl);
    if (mutated === wgsl) throw new Error("split-K: the mutation changed nothing in the WGSL");
    return mutated;
  };
  return async ({ geometry, m, n, k, slices, weight, x, bias }) => {
    const v4 = gemmUsesVec4(k, n);
    const owned: GPUBuffer[] = [];
    const make = (value: number | ArrayBufferView<ArrayBuffer>, usage: number): GPUBuffer => {
      const size = typeof value === "number" ? value : value.byteLength;
      const buffer = device.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage });
      owned.push(buffer);
      if (typeof value !== "number") device.queue.writeBuffer(buffer, 0, value);
      return buffer;
    };
    const input = (value: ArrayBufferView<ArrayBuffer>): GPUBuffer =>
      make(value, U.STORAGE | U.COPY_DST);
    const written = (words: number): GPUBuffer =>
      make(new Uint32Array(words).fill(SPLITK_UNWRITTEN), U.STORAGE | U.COPY_SRC | U.COPY_DST);
    const work = async (): Promise<SplitKDispatchRun> => {
      const params = make(linearSplitKParams(m, n, k, slices), U.UNIFORM | U.COPY_DST);
      const xs = input(x);
      const ws = input(weight.words);
      const biases = input(bias);
      const scale = weight.scale === undefined ? undefined : input(weight.scale);
      const outWords = m * n + SPLITK_TAIL;
      const partialWords = slices * m * n + SPLITK_TAIL;
      const outReference = written(outWords);
      const outSplit = written(outWords);
      const outAgain = written(outWords);
      const partials = written(partialWords);
      const [reference, partial, combine] = await Promise.all([
        pipelineOf(linearWgsl(weight.storage, v4, "f32", m, weight.groupSize, geometry)),
        pipelineOf(
          partialWgsl(
            linearSplitKGemmWgsl(weight.storage, v4, slices, m, weight.groupSize, geometry),
          ),
        ),
        pipelineOf(linearSplitKCombineWgsl()),
      ]);
      const scaleEntry: [number, GPUBuffer][] =
        scale === undefined ? [] : [[LINEAR_SCALE_BINDING, scale]];
      const group = (target: GPUComputePipeline, entries: readonly [number, GPUBuffer][]) =>
        device.createBindGroup({
          layout: target.getBindGroupLayout(0),
          entries: entries.map(([binding, buffer]) => ({ binding, resource: { buffer } })),
        });
      const tiles = [Math.ceil(n / gemmTileN(geometry)), Math.ceil(m / gemmTileM(geometry))];
      const combineGroups = Math.min(
        Math.ceil((m * n) / LINEAR_SPLITK_COMBINE_GROUP),
        gpu.limits.maxComputeWorkgroupsPerDimension,
      );
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(reference);
      pass.setBindGroup(
        0,
        group(reference, [
          [0, params],
          [1, xs],
          [2, ws],
          [3, biases],
          [4, outReference],
          ...scaleEntry,
        ]),
      );
      pass.dispatchWorkgroups(tiles[0], tiles[1], 1);
      for (const out of [outSplit, outAgain]) {
        pass.setPipeline(partial);
        pass.setBindGroup(
          0,
          group(partial, [[0, params], [1, xs], [2, ws], [4, partials], ...scaleEntry]),
        );
        pass.dispatchWorkgroups(tiles[0], tiles[1], slices);
        pass.setPipeline(combine);
        pass.setBindGroup(
          0,
          group(combine, [
            [0, params],
            [1, partials],
            [2, biases],
            [3, out],
          ]),
        );
        pass.dispatchWorkgroups(combineGroups);
      }
      pass.end();
      const readback = (source: GPUBuffer, words: number): GPUBuffer => {
        const mapped = make(words * 4, U.MAP_READ | U.COPY_DST);
        encoder.copyBufferToBuffer(source, 0, mapped, 0, words * 4);
        return mapped;
      };
      const mapped = [
        readback(outReference, outWords),
        readback(outSplit, outWords),
        readback(outAgain, outWords),
        readback(partials, partialWords),
      ];
      device.queue.submit([encoder.finish()]);
      const [referenceWords, splitWords, againWords, partialsWords] = await Promise.all(
        mapped.map(async (buffer) => {
          await buffer.mapAsync(MAP_MODE.READ);
          const words = new Uint32Array(buffer.getMappedRange().slice(0));
          buffer.unmap();
          return words;
        }),
      );
      const tails = new Uint32Array(3 * SPLITK_TAIL);
      for (const [index, words] of [referenceWords, splitWords, againWords].entries()) {
        tails.set(words.subarray(m * n, outWords), index * SPLITK_TAIL);
      }
      return {
        reference: referenceWords.slice(0, m * n),
        split: splitWords.slice(0, m * n),
        again: againWords.slice(0, m * n),
        tails,
        partials: partialsWords.slice(0, partialWords),
      };
    };
    // A validation error (an invalid shader, a bad binding) throws instead of reading the fill.
    device.pushErrorScope("validation");
    const settled = await work().then(
      (value) => ({ ok: true, value }) as const,
      (error: unknown) => ({ ok: false, error }) as const,
    );
    const error = await device.popErrorScope();
    await device.queue.onSubmittedWorkDone();
    for (const buffer of owned) buffer.destroy();
    if (error !== null) throw new Error(`split-K dispatch: ${error.message}`);
    if (!settled.ok) throw settled.error;
    return settled.value;
  };
};
