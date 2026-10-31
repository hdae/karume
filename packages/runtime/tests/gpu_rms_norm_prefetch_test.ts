// Bit identity of the prefetch form of the 256-wide rms_norm and of the RMS → add fusion (v2,
// perf-ledger K-99) with v1's runtime-trip output loop on the GPU: the plain kernel against
// RMS_NORM_WGSL (still the text of every width v2 does not take) and the add forms against their v1
// WGSL kept test-only in fixtures/wgsl-baseline/ (the generator no longer emits it).
//
// Every width v2 takes (the multiples of 256 up to RMS_NORM_PREFETCH_MAX_DIM) at rows 1 / 4 / 8 / 33,
// a prefill-sized 768 rows at three widths, each once with one workgroup per row and once on three
// workgroups (the forced grid-stride, where the weight registers outlive the row). The rows are
// adversarial: all zero (signed), ±inf, NaN payloads, subnormals only, sums of squares that overflow
// to inf, one huge element, magnitudes far below eps and at eps, raw random words and spread
// exponents; the weights and residuals carry ±0, ±inf, NaN, subnormals and f32 max.
//
// Fault injection (2026-10-31, RTX 3080 Ti, Node + Dawn): reordering the value to
// `xv[t] * wv[t] * inv` fails the plain kernel (first at dim 256, 8 rows); dropping the add's XOR
// rounding barrier fails either order on its own (first at dim 256, 33 rows — the contraction the
// barrier exists for); prefetching the weight with a stride of 255 fails from the first width with
// two trips (dim 512, 1 row).
import { readFileSync } from "node:fs";
import { assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { acquireGpu } from "../src/gpu/device.ts";
import { BUFFER_USAGE as U, MAP_MODE } from "../src/gpu/webgpu-constants.ts";
import {
  RMS_NORM_PREFETCH_MAX_DIM,
  RMS_NORM_WGSL,
  type RmsNormAddOrder,
  rmsNormAddWgsl,
  rmsNormParams,
  rmsNormPrefetchTrip,
  rmsNormWorkgroupWgsl,
} from "../src/kernels/rms-norm.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const baseline = (name: string): string =>
  readFileSync(new URL(`./fixtures/wgsl-baseline/${name}_v1.wgsl`, import.meta.url), "utf8");

const V1_ADD: Readonly<Record<RmsNormAddOrder, string>> = {
  "norm-residual": baseline("rms_norm_add_nr"),
  "residual-norm": baseline("rms_norm_add_rn"),
};

const EPS = Math.fround(1e-6);

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
const CASES: readonly (readonly [number, number])[] = [
  ...WIDTHS.flatMap((dim) => [1, 4, 8, 33].map((rows) => [dim, rows] as const)),
  ...[256, 1536, 4096].map((dim) => [dim, 768] as const),
];

describe.skipIf(!GPU_AVAILABLE)(
  "the rms_norm prefetch form on a real GPU (perf-ledger K-99)",
  () => {
    it("writes v1's words in the plain kernel and both RMS → add orders, natural and grid-strided", async () => {
      const gpu = await acquireGpu();
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
      try {
        for (const [dim, rows] of CASES) {
          assertEquals(rmsNormPrefetchTrip(dim), dim / 256, `dim ${dim} takes v2`);
          const x = rowsOf(rows, dim);
          const weight = operand(dim, dim);
          const residual = operand(rows * dim, rows * 31 + dim);
          const run = async (
            code: string,
            groups: number,
            add: boolean,
          ): Promise<Uint32Array<ArrayBuffer>> => {
            device.pushErrorScope("validation");
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
              device.queue.writeBuffer(out, 0, new Uint32Array(rows * dim).fill(0xdeadbeef));
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
              assertEquals(await device.popErrorScope(), null);
            }
          };
          for (const groups of new Set([rows, Math.min(rows, 3)])) {
            const label = `dim ${dim} rows ${rows} groups ${groups}`;
            assertEquals(
              await run(rmsNormWorkgroupWgsl(dim), groups, false),
              await run(RMS_NORM_WGSL, groups, false),
              `plain ${label}`,
            );
            for (const order of ["norm-residual", "residual-norm"] as const) {
              assertEquals(
                await run(rmsNormAddWgsl(order, dim), groups, true),
                await run(V1_ADD[order], groups, true),
                `${order} ${label}`,
              );
            }
          }
        }
      } finally {
        gpu.destroy();
      }
    });
  },
);
