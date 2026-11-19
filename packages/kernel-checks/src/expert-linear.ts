/**
 * Check 9 — `expert_linear` with f16 steps at the shapes of Gemma 4 26B A4B (ADR 0145 decision 4).
 *
 * The kernel reuses `linear`'s unrolled GEMV text with the weight row chosen per pick, so its oracle
 * is `linear`'s own sequential GEMV (`linearGemvWgsl`, i4) on the picked expert's rows with the
 * steps widened to f32 (exact — ADR 0069 decision 3): every output word of every pick MUST equal
 * that GEMV's word. A word that differs is a different expert row, x row, step or summation order.
 * Three kernels are held to it at every shape: the one-row form, the grouped form (the rows that
 * picked one expert share the walk of its weight row, which a run of two or more tokens takes for
 * gate_up's long rows), and the grouped form over a pool (the form the expert cache runs). The
 * grouped form calls a word function where the one-row form and `linear`'s GEMV have the same text
 * inline, so that a backend's compiler rounds both alike is what this check shows per backend.
 *
 * The two 26B projections: gate_up (n 1408, k 2816, x shared by the picks) and down (n 2816, k 704,
 * one x row per pick), and a third shape that is no model's (n 1408, k 2880, x shared): a weight row
 * a Session groups whose word count is not a multiple of the unrolled loop's four, so the grouped
 * form's loops over the words left are compared where a Session would run them. Each is over
 * E = 8 experts, T = 9 tokens of k = 8 picks. The experts are picked
 * 9, 16, 12, 7, 10, 6, 3 and 9 times ({@link SHARERS}), so the grouped form takes every walk it
 * generates, alone and after another: 8 + 1, two scans of 8, 8 + 4, 4 + 2 + 1, 8 + 2, 4 + 2 and
 * 2 + 1 rows; an id repeats within a token too. In the pool the experts sit in another order than
 * their ids (a slot table maps them), so an addressing slip reads another expert. Every column of
 * the expert output starts as a NaN pattern, so an unwritten word mismatches too.
 */

import type { GpuContext } from "../../runtime/src/gpu/device.ts";
import { BUFFER_USAGE as U, MAP_MODE } from "../../runtime/src/gpu/webgpu-constants.ts";
import { f16BitsToF32 } from "../../runtime/src/format/f16.ts";
import {
  expertLinearDispatch,
  expertLinearParams,
  expertLinearWgsl,
} from "../../runtime/src/kernels/expert-linear.ts";
import { DEFAULT_GEOMETRY_PROFILE } from "../../runtime/src/kernels/geometry-profile.ts";
import {
  defaultLinearGemvVariant,
  linearGemvParams,
  linearGemvWgsl,
} from "../../runtime/src/kernels/linear-gemv.ts";
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
const TOKENS = 9;
const PICKS = 8;
/** How many of the T × k pick rows pick each expert (module doc). */
const SHARERS: readonly number[] = [9, 16, 12, 7, 10, 6, 3, 9];
/** The slot of each expert in the pool of the pooled form: a permutation, no expert in its own slot. */
const SLOT_OF: readonly number[] = [3, 6, 1, 4, 7, 2, 5, 0];
const NAN_FILL = 0x7fc00000;
const NEGATIVE_ZERO = 0x80000000;

/**
 * The ids of the T × k picks: {@link SHARERS}' ids in order (expert 0 nine times, expert 1 sixteen
 * times, …) spread over the rows by a fixed permutation (row `i` takes entry `29 i + 5 mod 72`), so
 * the rows of one expert are scattered over the tokens and repeat within some of them.
 */
const IDS: readonly number[] = (() => {
  const sorted = SHARERS.flatMap((count, expert) => new Array<number>(count).fill(expert));
  return sorted.map((_, row) => sorted[(row * 29 + 5) % sorted.length]);
})();

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
  // No model's shape: a weight row long enough for a Session to group it (90 words) that leaves
  // words after the unrolled loop (90 = 22 × 4 + 2), which neither projection above does in a Session
  { name: "left words n1408 k2880 shared x", n: 1408, k: 2880, sharedX: true },
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
 * @param mutate Rewrites the one-row WGSL (a fault for the tests that show the check fails).
 * @param mutateGrouped Rewrites the grouped WGSL (likewise).
 */
export const checkExpertLinear = async (
  gpu: GpuContext,
  options: CheckOptions & {
    readonly mutate?: (wgsl: string) => string;
    readonly mutateGrouped?: (wgsl: string) => string;
  } = {},
): Promise<CheckPart> => {
  const { device } = gpu;
  const cases: CheckCase[] = [];
  const total = EXPERT_LINEAR_CASES["expert-linear"];
  const compile = async (code: string): Promise<GPUComputePipeline> =>
    await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
    });
  const mutation = (
    wgsl: string,
    mutate: ((wgsl: string) => string) | undefined,
    form: string,
  ): string => {
    const mutated = mutate === undefined ? wgsl : mutate(wgsl);
    if (mutate !== undefined && mutated === wgsl) {
      throw new Error(`expert-linear: the mutation left the ${form} WGSL unchanged`);
    }
    return mutated;
  };
  const mutated = mutation(expertLinearWgsl("i4f16", GROUP), options.mutate, "one-row");
  const mutatedGrouped = mutation(
    expertLinearWgsl("i4f16", GROUP, false, true),
    options.mutateGrouped,
    "grouped",
  );
  const pooledGrouped = expertLinearWgsl("i4f16", GROUP, true, true);
  if (IDS.length !== TOKENS * PICKS) throw new Error("expert-linear: the routing is not T × k ids");
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
        const bound = [
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
        ];
        const workgroups = expertLinearDispatch(
          rows,
          n,
          gpu.limits.maxComputeWorkgroupsPerDimension,
        );
        const got = await readBack(
          out,
          rows * n,
          dispatch(await compile(mutated), bound, workgroups),
        );
        device.queue.writeBuffer(out, 0, new Uint32Array(rows * n).fill(NAN_FILL));
        const gotGrouped = await readBack(
          out,
          rows * n,
          dispatch(await compile(mutatedGrouped), bound, workgroups),
        );
        // The pooled grouped form: every expert resident, in the slot SLOT_OF gives it.
        const pooledWords = (per: number, source: Uint32Array<ArrayBuffer>) => {
          const pool = new Uint32Array(source.length);
          SLOT_OF.forEach((slot, expert) => {
            pool.set(source.subarray(expert * per, (expert + 1) * per), slot * per);
          });
          return pool;
        };
        // An expert's f16 steps are a whole number of u32 words (n · k / 32 is even at every shape).
        const stepWordsPerExpert = (n * k) / GROUP / 2;
        if (!Number.isInteger(stepWordsPerExpert)) {
          throw new Error("expert-linear: an expert's f16 steps do not fill whole words");
        }
        device.queue.writeBuffer(out, 0, new Uint32Array(rows * n).fill(NAN_FILL));
        const gotPooled = await readBack(
          out,
          rows * n,
          dispatch(
            await compile(pooledGrouped),
            [
              make(
                expertLinearParams(
                  "i4f16",
                  {
                    tokens: TOKENS,
                    picks: PICKS,
                    n,
                    k,
                    experts: EXPERTS,
                    sharedX,
                    slots: EXPERTS,
                  },
                  GROUP,
                ),
                U.UNIFORM | U.COPY_DST,
              ),
              bound[1],
              bound[2],
              make(pooledWords((n * k) / 8, data.packed)),
              out,
              make(pooledWords(stepWordsPerExpert, data.stepWords)),
              make(Int32Array.from(SLOT_OF)),
            ],
            workgroups,
          ),
        );
        // The oracle: linear's M = 1 GEMV over one expert's rows (bias 0, steps widened), in the
        // default profile's variant (built without a Session, so no adapter's table applies).
        const gemv = await compile(
          linearGemvWgsl(
            "i4",
            GROUP,
            defaultLinearGemvVariant(DEFAULT_GEOMETRY_PROFILE.linearGemv),
          ),
        );
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
          const wordsOf = (all: Uint32Array<ArrayBuffer>): Uint32Array<ArrayBuffer> =>
            all.slice(row * n, (row + 1) * n).map((word) => (word === NEGATIVE_ZERO ? 0 : word));
          cases.push({
            label: `${shape.name} t${Math.floor(row / PICKS)} j${row % PICKS} (expert ${expert})`,
            criteria: [
              compareWords(
                "expert_linear vs linear's GEMV on that expert (u32)",
                wordsOf(got),
                expected,
              ),
              compareWords(
                "the grouped form vs linear's GEMV on that expert (u32)",
                wordsOf(gotGrouped),
                expected,
              ),
              compareWords(
                "the pooled grouped form vs linear's GEMV on that expert (u32)",
                wordsOf(gotPooled),
                expected,
              ),
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
    title:
      "expert_linear (f16 steps), one-row, grouped and pooled grouped: against linear's GEMV on each picked expert",
    cases,
    requirements: [],
  };
};
