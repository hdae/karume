/**
 * Check 9 — `expert_linear` with f16 steps at the shapes of Gemma 4 26B A4B (ADR 0145 decision 4).
 *
 * The kernel reuses `linear`'s unrolled GEMV text with the weight row chosen per pick, so its oracle
 * is `linear`'s own sequential GEMV (`linearGemvWgsl`, i4) on the picked expert's rows with the
 * steps widened to f32 (exact — ADR 0069 decision 3): every output word of every pick MUST equal
 * that GEMV's word. A word that differs is a different expert row, x row, step or summation order.
 *
 * The two 26B projections: gate_up (n 1408, k 2816, x shared by the picks) and down (n 2816, k 704,
 * one x row per pick), each over E = 8 experts, T = 3 tokens of k = 8 picks whose ids repeat within
 * and across tokens. Every column of the expert output starts as a NaN pattern, so an unwritten word
 * mismatches too.
 */

import type { GpuContext } from "../../runtime/src/gpu/device.ts";
import { BUFFER_USAGE as U, MAP_MODE } from "../../runtime/src/gpu/webgpu-constants.ts";
import { f16BitsToF32 } from "../../runtime/src/format/f16.ts";
import {
  expertLinearDispatch,
  expertLinearParams,
  expertLinearWgsl,
} from "../../runtime/src/kernels/expert-linear.ts";
import { linearGemvParams, linearGemvWgsl } from "../../runtime/src/kernels/linear-gemv.ts";
import {
  type CheckCase,
  type CheckOptions,
  type CheckPart,
  compareWords,
  type ExpectedCases,
  scopedGpuWork,
  throwIfAborted,
} from "./result.ts";

const GROUP = 32;
const EXPERTS = 8;
const TOKENS = 3;
const PICKS = 8;
const NAN_FILL = 0x7fc00000;
const NEGATIVE_ZERO = 0x80000000;

/** The ids of the T × k picks: repeats within a token (t0 j0 / j5, t2 j1 / j6) and across tokens. */
const IDS: readonly number[] = [
  [3, 0, 7, 1, 6, 3, 2, 5],
  [5, 4, 1, 0, 2, 7, 6, 3],
  [2, 6, 6, 0, 4, 1, 5, 6],
].flat();

type Shape = {
  readonly name: string;
  readonly n: number;
  readonly k: number;
  /** x is `[T, k]` shared by the picks (otherwise `[T, picks, k]`). */
  readonly sharedX: boolean;
};

const SHAPES: readonly Shape[] = [
  { name: "26b gate_up n1408 k2816 shared x", n: 1408, k: 2816, sharedX: true },
  { name: "26b down n2816 k704 per-pick x", n: 2816, k: 704, sharedX: false },
];

export const EXPERT_LINEAR_CASES = {
  "expert-linear": SHAPES.length * TOKENS * PICKS,
} as const satisfies ExpectedCases;

/** Deterministic packed codes, f16 steps (positive normals, 26B's step range) and x. */
const fixtureOf = (shape: Shape) => {
  const { n, k, sharedX } = shape;
  const steps = (EXPERTS * n * k) / GROUP;
  const stepBits = Uint16Array.from(
    { length: steps + (steps % 2) },
    (_, i) => 0x1400 + ((i * 131) % 0x1000),
  );
  return {
    packed: Uint32Array.from(
      { length: (EXPERTS * n * k) / 8 },
      (_, i) => Math.imul(i + 7, 0x9e3779b9) >>> 0,
    ),
    stepWords: new Uint32Array(stepBits.buffer),
    widened: Float32Array.from(stepBits.subarray(0, steps), (bits) => f16BitsToF32(bits)),
    x: Float32Array.from(
      { length: (sharedX ? TOKENS : TOKENS * PICKS) * k },
      (_, i) => Math.cos(i * 0.029) * 0.9,
    ),
  };
};

/**
 * Runs the check. Needs no optional feature (`unpack2x16float` is core WGSL).
 *
 * @param mutate Rewrites the expert_linear WGSL (a fault for the tests that show the check fails).
 */
export const checkExpertLinear = async (
  gpu: GpuContext,
  options: CheckOptions & { readonly mutate?: (wgsl: string) => string } = {},
): Promise<CheckPart> => {
  const { device } = gpu;
  const cases: CheckCase[] = [];
  const total = EXPERT_LINEAR_CASES["expert-linear"];
  const compile = async (code: string): Promise<GPUComputePipeline> =>
    await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
    });
  const wgsl = expertLinearWgsl("i4f16", GROUP);
  const mutated = options.mutate === undefined ? wgsl : options.mutate(wgsl);
  if (options.mutate !== undefined && mutated === wgsl) {
    throw new Error("expert-linear: the mutation left the WGSL unchanged");
  }
  for (const shape of SHAPES) {
    throwIfAborted(options);
    const { n, k, sharedX } = shape;
    const data = fixtureOf(shape);
    const owned: GPUBuffer[] = [];
    const make = (
      value: number | ArrayBufferView<ArrayBuffer>,
      usage = U.STORAGE | U.COPY_DST,
    ): GPUBuffer => {
      const size = typeof value === "number" ? value : value.byteLength;
      const buffer = device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage });
      owned.push(buffer);
      if (typeof value !== "number") device.queue.writeBuffer(buffer, 0, value);
      return buffer;
    };
    const readBack = async (source: GPUBuffer, words: number, encoder: GPUCommandEncoder) => {
      const mapped = make(words * 4, U.MAP_READ | U.COPY_DST);
      encoder.copyBufferToBuffer(source, 0, mapped, 0, words * 4);
      device.queue.submit([encoder.finish()]);
      await mapped.mapAsync(MAP_MODE.READ);
      const out = new Uint32Array(mapped.getMappedRange().slice(0));
      mapped.unmap();
      return out;
    };
    const dispatch = (
      pipeline: GPUComputePipeline,
      buffers: readonly GPUBuffer[],
      workgroups: readonly [number, number, number],
    ): GPUCommandEncoder => {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(
        0,
        device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
        }),
      );
      pass.dispatchWorkgroups(...workgroups);
      pass.end();
      return encoder;
    };
    await scopedGpuWork(gpu, `expert-linear ${shape.name}`, async () => {
      try {
        const rows = TOKENS * PICKS;
        const out = make(rows * n * 4, U.STORAGE | U.COPY_SRC | U.COPY_DST);
        device.queue.writeBuffer(out, 0, new Uint32Array(rows * n).fill(NAN_FILL));
        const kernel = await compile(mutated);
        const got = await readBack(
          out,
          rows * n,
          dispatch(
            kernel,
            [
              make(
                expertLinearParams(
                  "i4f16",
                  { tokens: TOKENS, picks: PICKS, n, k, experts: EXPERTS, sharedX },
                  GROUP,
                ),
                U.UNIFORM | U.COPY_DST,
              ),
              make(data.x),
              make(Int32Array.from(IDS)),
              make(data.packed),
              out,
              make(data.stepWords),
            ],
            expertLinearDispatch(rows, n, gpu.limits.maxComputeWorkgroupsPerDimension),
          ),
        );
        // The oracle: linear's M = 1 GEMV over one expert's rows (bias 0, steps widened).
        const gemv = await compile(linearGemvWgsl("i4", GROUP));
        const experts = Array.from({ length: EXPERTS }, (_, expert) => ({
          weight: make(data.packed.slice((expert * n * k) / 8, ((expert + 1) * n * k) / 8)),
          steps: make(data.widened.slice((expert * n * k) / GROUP, ((expert + 1) * n * k) / GROUP)),
        }));
        const params = make(linearGemvParams("i4", 1, n, k, GROUP), U.UNIFORM | U.COPY_DST);
        const bias = make(new Float32Array(n));
        for (let row = 0; row < rows; row += 1) {
          throwIfAborted(options);
          const expert = IDS[row];
          const xRow = sharedX ? Math.floor(row / PICKS) : row;
          const reference = make(n * 4, U.STORAGE | U.COPY_SRC | U.COPY_DST);
          const expected = await readBack(
            reference,
            n,
            dispatch(
              gemv,
              [
                params,
                make(data.x.slice(xRow * k, (xRow + 1) * k)),
                experts[expert].weight,
                bias,
                reference,
                experts[expert].steps,
              ],
              [Math.ceil(n / 32), 1, 1],
            ),
          );
          // linear adds its zero bias, so a −0 sum comes back +0 there.
          const words = got
            .slice(row * n, (row + 1) * n)
            .map((word) => (word === NEGATIVE_ZERO ? 0 : word));
          cases.push({
            label: `${shape.name} t${Math.floor(row / PICKS)} j${row % PICKS} (expert ${expert})`,
            criteria: [
              compareWords("expert_linear vs linear's GEMV on that expert (u32)", words, expected),
            ],
          });
          options.onProgress?.({ part: "expert-linear", done: cases.length, total });
        }
      } finally {
        await device.queue.onSubmittedWorkDone();
        for (const buffer of owned) buffer.destroy();
      }
    });
  }
  return {
    id: "expert-linear",
    title: "expert_linear (f16 steps): against linear's GEMV on each picked expert",
    cases,
    requirements: [],
  };
};
