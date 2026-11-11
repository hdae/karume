/**
 * The prefetch form of the 256-wide rms_norm and of the RMS → add fusion (v2, perf-ledger K-99)
 * against v1's runtime-trip output loop: the plain kernel against RMS_NORM_WGSL (still the text of
 * every width v2 does not take) and the add forms against their v1 WGSL kept test-only in the
 * runtime's tests/fixtures/wgsl-baseline/ (the generator no longer emits it). u32 identity with NaN by
 * payload, one part per compared kernel.
 *
 * Every width v2 takes (the multiples of 256 up to RMS_NORM_PREFETCH_MAX_DIM) at rows 1 / 4 / 8 / 33,
 * a prefill-sized 768 rows at three widths, each once with one workgroup per row and once on three
 * workgroups (the forced grid-stride, where the weight registers outlive the row). The rows are
 * adversarial: all zero (signed), ±inf, NaN payloads, subnormals only, sums of squares that overflow
 * to inf, one huge element, magnitudes far below eps and at eps, raw random words and spread
 * exponents; the weights and residuals carry ±0, ±inf, NaN, subnormals and f32 max. The output is
 * poisoned with 0xdeadbeef before each dispatch.
 *
 * Fault injection (2026-10-31, RTX 3080 Ti, Node + Dawn): reordering the value to
 * `xv[t] * wv[t] * inv` fails the plain kernel (first at dim 256, 8 rows); dropping the add's XOR
 * rounding barrier fails either order on its own (first at dim 256, 33 rows — the contraction the
 * barrier exists for); prefetching the weight with a stride of 255 fails from the first width with
 * two trips (dim 512, 1 row).
 */

import type { GpuContext } from "../../runtime/src/gpu/device.ts";
import { BUFFER_USAGE as U, MAP_MODE } from "../../runtime/src/gpu/webgpu-constants.ts";
import {
  RMS_NORM_PREFETCH_MAX_DIM,
  RMS_NORM_WGSL,
  type RmsNormAddOrder,
  rmsNormAddWgsl,
  rmsNormParams,
  rmsNormPrefetchTrip,
  rmsNormWorkgroupWgsl,
} from "../../runtime/src/kernels/rms-norm.ts";
import addNormResidualV1 from "../../runtime/tests/fixtures/wgsl-baseline/rms_norm_add_nr_v1.wgsl?raw";
import addResidualNormV1 from "../../runtime/tests/fixtures/wgsl-baseline/rms_norm_add_rn_v1.wgsl?raw";
import {
  type CheckCase,
  type CheckOptions,
  type CheckPart,
  compareWords,
  type ExpectedCases,
  scopedGpuWork,
  throwIfAborted,
} from "./result.ts";

const V1_ADD: Readonly<Record<RmsNormAddOrder, string>> = {
  "norm-residual": addNormResidualV1,
  "residual-norm": addResidualNormV1,
};

const EPS = Math.fround(1e-6);

/** The word the output buffer is filled with before each dispatch. */
const POISON = 0xdeadbeef;

const f32Bits = (value: number): number => new Uint32Array(Float32Array.of(value).buffer)[0];

const SPECIAL_WORDS = [
  0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00001, 0x7fa00000, 1, 0x80000001,
  0x007fffff, 0x7f7fffff, 0xff7fffff, 0x00800000,
];

/** A deterministic word source (no Math.random — a failure must reproduce). */
const words = (seed: number) => {
  let state = seed >>> 0;
  const next = (): number => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state;
  };
  const uniform = (): number => next() / 2 ** 32;
  const gauss = (): number =>
    Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform());
  return { next, uniform, gauss };
};

/** `rows × dim` adversarial x words: row r takes pattern r % 12. */
const rowsOf = (rows: number, dim: number): Uint32Array<ArrayBuffer> => {
  const source = words(0x9e3779b9 ^ (rows * 4099 + dim));
  const out = new Uint32Array(rows * dim);
  for (let r = 0; r < rows; r += 1) {
    const row = out.subarray(r * dim, (r + 1) * dim);
    const finite = (scale: number) => {
      for (let i = 0; i < dim; i += 1) row[i] = f32Bits(source.gauss() * scale);
    };
    switch (r % 12) {
      case 0: // all zero, both signs
        for (let i = 0; i < dim; i += 1) row[i] = i % 3 === 0 ? 0x80000000 : 0;
        break;
      case 1:
        finite(1);
        row[(r * 7) % dim] = 0x7f800000;
        break;
      case 2:
        finite(1);
        row[(r * 11) % dim] = 0xff800000;
        break;
      case 3:
        finite(1);
        row[(r * 13) % dim] = 0x7fc00000 | (source.next() & 0x3fffff);
        break;
      case 4: // subnormals only
        for (let i = 0; i < dim; i += 1) row[i] = source.next() & 0x807fffff;
        break;
      case 5: // the sum of squares overflows to inf
        finite(1e19);
        break;
      case 6: // one huge element among ordinary ones
        finite(1);
        row[(r * 17) % dim] = f32Bits(3e38);
        break;
      case 7: // far below eps: inv is eps's
        finite(1e-20);
        break;
      case 8: // the mean of squares at eps
        for (let i = 0; i < dim; i += 1) row[i] = f32Bits(i % 2 === 0 ? 1e-3 : -1e-3);
        break;
      case 9: // raw words (NaN and inf among them)
        for (let i = 0; i < dim; i += 1) row[i] = source.next();
        break;
      case 10: // spread exponents
        for (let i = 0; i < dim; i += 1) {
          row[i] = f32Bits(source.gauss() * 10 ** (12 * source.uniform() - 6));
        }
        break;
      default:
        finite(3);
    }
  }
  return out;
};

/** Ordinary values with the special words sprinkled through (weight or residual). */
const operand = (length: number, seed: number): Uint32Array<ArrayBuffer> => {
  const source = words(seed);
  const out = new Uint32Array(length);
  for (let i = 0; i < length; i += 1) {
    out[i] =
      i % 61 === 0
        ? SPECIAL_WORDS[(i / 61) % SPECIAL_WORDS.length]
        : f32Bits(1 + 0.3 * source.gauss());
  }
  return out;
};

const WIDTHS = Array.from(
  { length: RMS_NORM_PREFETCH_MAX_DIM / 256 },
  (_, index) => (index + 1) * 256,
);

/** (dim, rows) cases: every width at rows 1 / 4 / 8 / 33, and 768 rows at three widths. */
const GRID: readonly (readonly [number, number])[] = [
  ...WIDTHS.flatMap((dim) => [1, 4, 8, 33].map((rows) => [dim, rows] as const)),
  ...[256, 1536, 4096].map((dim) => [dim, 768] as const),
];

/** The dispatch group counts of a case: one workgroup per row, and three (forced grid-stride). */
const groupCounts = (rows: number): readonly number[] => [...new Set([rows, Math.min(rows, 3)])];

/**
 * The cases each part of {@link checkRmsNormPrefetch} returns: per kernel, 16 widths × (1 row on
 * one group count + 3 row counts on two) and 3 widths × 768 rows on two group counts.
 */
export const RMS_NORM_PREFETCH_CASES = {
  "rms-prefetch-plain": 118,
  "rms-prefetch-add-norm-residual": 118,
  "rms-prefetch-add-residual-norm": 118,
} as const satisfies ExpectedCases;

type PartId = keyof typeof RMS_NORM_PREFETCH_CASES;

type Kernel = {
  readonly part: PartId;
  readonly title: string;
  readonly add: boolean;
  readonly prefetch: (dim: number) => string;
  readonly runtimeTrip: string;
};

/** An RMS → add order's kernel: its prefetch form against its v1 WGSL. */
const addKernel = (part: PartId, order: RmsNormAddOrder): Kernel => ({
  part,
  title: `rms_norm + add (${order}): prefetch form vs the v1 WGSL`,
  add: true,
  prefetch: (dim) => rmsNormAddWgsl(order, dim),
  runtimeTrip: V1_ADD[order],
});

const KERNELS: readonly Kernel[] = [
  {
    part: "rms-prefetch-plain",
    title: "rms_norm: prefetch form vs RMS_NORM_WGSL",
    add: false,
    prefetch: rmsNormWorkgroupWgsl,
    runtimeTrip: RMS_NORM_WGSL,
  },
  addKernel("rms-prefetch-add-norm-residual", "norm-residual"),
  addKernel("rms-prefetch-add-residual-norm", "residual-norm"),
];

/** Runs the check: the plain kernel and the two RMS → add orders (needs no optional feature). */
export const checkRmsNormPrefetch = async (
  gpu: GpuContext,
  options: CheckOptions = {},
): Promise<readonly CheckPart[]> => {
  const device = gpu.device;
  const pipelines = new Map<string, GPUComputePipeline>();
  const pipeline = async (code: string): Promise<GPUComputePipeline> => {
    const cached = pipelines.get(code);
    if (cached !== undefined) return cached;
    const created = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
    });
    pipelines.set(code, created);
    return created;
  };
  const parts = KERNELS.map((kernel) => ({ kernel, cases: new Array<CheckCase>() }));
  for (const [dim, rows] of GRID) {
    // A width the generator does not take v2 at would compare v1 with itself
    if (rmsNormPrefetchTrip(dim) !== dim / 256) {
      throw new Error(`rms_norm prefetch check: dim ${dim} does not take the prefetch form`);
    }
    const x = rowsOf(rows, dim);
    const weight = operand(dim, dim);
    const residual = operand(rows * dim, rows * 31 + dim);
    const run = async (
      code: string,
      groups: number,
      add: boolean,
    ): Promise<Uint32Array<ArrayBuffer>> => {
      const owned: GPUBuffer[] = [];
      const make = (data: Uint32Array<ArrayBuffer> | number, usage: number): GPUBuffer => {
        const size = typeof data === "number" ? data : data.byteLength;
        const buffer = device.createBuffer({ size, usage: usage | U.COPY_DST });
        owned.push(buffer);
        if (typeof data !== "number") device.queue.writeBuffer(buffer, 0, data);
        return buffer;
      };
      try {
        const bytes = rows * dim * 4;
        const out = make(bytes, U.STORAGE | U.COPY_SRC);
        const staging = make(bytes, U.MAP_READ);
        const compiled = await pipeline(code);
        const buffers = [
          make(rmsNormParams(rows, dim, EPS), U.UNIFORM),
          make(x, U.STORAGE),
          make(weight, U.STORAGE),
          ...(add ? [make(residual, U.STORAGE)] : []),
          out,
        ];
        const group = device.createBindGroup({
          layout: compiled.getBindGroupLayout(0),
          entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
        });
        device.queue.writeBuffer(out, 0, new Uint32Array(rows * dim).fill(POISON));
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(compiled);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(groups);
        pass.end();
        encoder.copyBufferToBuffer(out, 0, staging, 0, bytes);
        device.queue.submit([encoder.finish()]);
        await staging.mapAsync(MAP_MODE.READ);
        const result = new Uint32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        return result;
      } finally {
        await device.queue.onSubmittedWorkDone();
        for (const buffer of owned) buffer.destroy();
      }
    };
    for (const groups of groupCounts(rows)) {
      const label = `dim ${dim} rows ${rows} groups ${groups}`;
      for (const { kernel, cases } of parts) {
        throwIfAborted(options);
        const [prefetch, runtimeTrip] = await scopedGpuWork(
          gpu,
          `${kernel.part} ${label}`,
          async () => [
            await run(kernel.prefetch(dim), groups, kernel.add),
            await run(kernel.runtimeTrip, groups, kernel.add),
          ],
        );
        cases.push({
          label,
          criteria: [
            compareWords("prefetch form vs runtime-trip form (u32)", prefetch, runtimeTrip),
          ],
        });
        options.onProgress?.({
          part: kernel.part,
          done: cases.length,
          total: RMS_NORM_PREFETCH_CASES[kernel.part],
        });
      }
    }
  }
  return parts.map(({ kernel, cases }) => ({
    id: kernel.part,
    title: kernel.title,
    cases,
    requirements: [],
  }));
};
