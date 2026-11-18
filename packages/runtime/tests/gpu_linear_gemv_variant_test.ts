import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { acquireGpu } from "../src/gpu/device.ts";
import {
  DEFAULT_GEOMETRY_PROFILE,
  resolveGeometryProfile,
} from "../src/kernels/geometry-profile.ts";
import { BUILTIN_GEOMETRY_PROFILES } from "../src/kernels/geometry-profiles/index.ts";
import {
  defaultLinearGemvVariant,
  linearGemvParams,
  type LinearGemvShapeVariant,
  linearGemvUnit,
  type LinearGemvVariant,
  linearGemvWgsl,
} from "../src/kernels/linear-gemv.ts";
import type { WeightStorage } from "../src/kernels/weight-storage.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { GEMM_TOLERANCE } from "../src/reference/op-tolerance.ts";

// The WGSL chosen for the large-vocabulary shape does not bake N. The edge columns and the edge of
// the prefetched words are checked at a small shape; the real shape 262144×1536 is verified by the
// Chrome real-weight comparison and Gemma's existing logits / golden checks.
test.skipIf(!GPU_AVAILABLE)(
  "the INT8 c16 variant is bit-identical to c32 and writes the column and prefetch edges correctly (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const device = gpu.device;
    // The kernels are built without a Session: the default profile's variants are the subject.
    const gemv = DEFAULT_GEOMETRY_PROFILE.linearGemv;
    const selected = defaultLinearGemvVariant(gemv, { storage: "i8", n: 262144, k: 1536 });
    assertEquals(selected.cols, 16);
    try {
      for (const [n, k] of [
        [68, 1536],
        [4, 16],
        [36, 80],
      ]) {
        device.pushErrorScope("validation");
        const owned: GPUBuffer[] = [];
        const make = (
          data: number | ArrayBufferView<ArrayBuffer>,
          usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        ): GPUBuffer => {
          const buffer = device.createBuffer({
            size: typeof data === "number" ? data : data.byteLength,
            usage,
          });
          owned.push(buffer);
          if (typeof data !== "number") device.queue.writeBuffer(buffer, 0, data);
          return buffer;
        };
        try {
          const packed = Uint32Array.from(
            { length: (n * k) / 4 },
            (_, i) => Math.imul(i + 1, 0x9e3779b9) >>> 0,
          );
          const input = Float32Array.from({ length: k }, (_, i) => Math.sin(i * 0.037) * 0.75);
          const scale = Float32Array.from({ length: n }, (_, i) => ((i % 17) + 1) * 0.00017);
          const bias = Float32Array.from({ length: n }, (_, i) => ((i % 7) - 3) * 0.11);
          const buffers = [
            make(linearGemvParams("i8", 1, n, k), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
            make(input),
            make(packed),
            make(bias),
            make(n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC),
            make(scale),
          ];
          const mapped = make(n * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
          let expected: Uint32Array<ArrayBuffer> | undefined;
          for (const variant of [defaultLinearGemvVariant(gemv), selected]) {
            const pipeline = await device.createComputePipelineAsync({
              layout: "auto",
              compute: {
                module: device.createShaderModule({
                  code: linearGemvWgsl("i8", undefined, variant),
                }),
                entryPoint: "main",
              },
            });
            const bindings = device.createBindGroup({
              layout: pipeline.getBindGroupLayout(0),
              entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
            });
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindings);
            pass.dispatchWorkgroups(Math.ceil(n / variant.cols));
            pass.end();
            encoder.copyBufferToBuffer(buffers[4], 0, mapped, 0, n * 4);
            device.queue.submit([encoder.finish()]);
            await mapped.mapAsync(GPUMapMode.READ);
            const actual = new Uint32Array(mapped.getMappedRange().slice(0));
            mapped.unmap();
            expected ??= actual;
            assertEquals(actual, expected, `c${variant.cols} N=${n} K=${k}`);
            const values = new Float32Array(actual.buffer);
            for (let row = 0; row < n; row++) {
              let reference = 0;
              for (let col = 0; col < k; col++) {
                const index = row * k + col;
                const q = (packed[Math.floor(index / 4)] << (24 - (index % 4) * 8)) >> 24;
                reference += input[col] * Math.fround(q * scale[row]);
              }
              reference += bias[row];
              assert(Number.isFinite(values[row]));
              assert(
                Math.abs(values[row] - reference) <=
                  GEMM_TOLERANCE.atol + GEMM_TOLERANCE.rtol * Math.abs(reference),
                `CPU reference N=${n} K=${k} row=${row}`,
              );
            }
          }
        } finally {
          await device.queue.onSubmittedWorkDone();
          for (const buffer of owned) buffer.destroy();
          const error = await device.popErrorScope();
          assertEquals(error, null);
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

// Every per-shape exception of the sequential variant that can run — the default profile's rows and
// the rows of every registered table that carries GEMV choices of its own — against the general
// variant of its profile: on the row's own shape and on two small shapes, one that leaves the last
// workgroup partial and a remainder after the read-ahead loop of the exception and of the general
// variant, and one of a single group per row. An exception is a choice of speed that a Session at
// `exact` runs, so its words MUST be the general variant's. The rows are read from the profiles: a
// row added later, or a table that starts to carry the field, is checked without an edit here.
//
// A row names no group length, so an i4 row selects its variant for a weight of any group: an i4
// row is compared at the group of the shipped weights (32, where the smallest shape is one word per
// row and never enters the read-ahead loop) and at a second one (64).
//
// The CPU reference keeps the comparison from being vacuous (two kernels that write nothing, or the
// same wrong words): every column of the small shapes, 256 columns spread over a real one.
//
// Mutations it detects (fault injection, 2026-11-18, the 27 rows of that day — 23 of the default
// profile and 4 more of nvidia-ampere, 3 of them i4 —, 90 shapes):
// - in the exception's WGSL alone, the first two multiply-adds of every run of them swapped (the
//   same products, another order of addition) → the u32 comparison fails on the real shape of every
//   row at every group (30) and on 34 of the 60 small shapes, while the CPU reference holds on all
//   90 — the check means nothing unless it is exact u32 equality;
// - an exception's kernel that returns before it writes → the u32 comparison and the CPU reference
//   fail on all 90 shapes. They do because the output buffer is cleared before each kernel: without
//   the clear this mutation passed, the buffer still holding the general variant's words.

/** The group lengths an i4 row is compared at: the shipped weights' and a second one. */
const I4_GROUPS = [32, 64] as const;

const BITS = new Uint32Array(1);
const VALUE = new Float32Array(BITS.buffer);

/**
 * A finite bf16 pattern from 16 bits: any sign and mantissa, the exponent in a band of 16 (magnitudes
 * of 2⁻²⁰ to 2⁻⁴), so no element is an infinity or a NaN and a sum over the longest row stays inside
 * the tolerance of the CPU reference.
 */
const bf16Pattern = (bits: number): number =>
  (bits & 0x8000) | ((107 + ((bits >> 7) & 0xf)) << 7) | (bits & 0x7f);

/** The packed weight words of `[n, k]` (16 B per `vec4`, as the kernel binds them). */
const weightWords = (storage: WeightStorage, n: number, k: number): Uint32Array<ArrayBuffer> => {
  const bitsPerElement = storage === "i4" ? 4 : storage === "i8" ? 8 : 16;
  const words = new Uint32Array((n * k * bitsPerElement) / 32);
  for (let index = 0; index < words.length; index++) {
    const word = Math.imul(index + 1, 0x9e3779b9) >>> 0;
    words[index] =
      storage === "bf16"
        ? ((bf16Pattern(word >>> 16) << 16) | bf16Pattern(word & 0xffff)) >>> 0
        : word;
  }
  return words;
};

/**
 * Element `[row, col]` of the weight as the kernel decodes it: the bf16 value, or the code times its
 * scale (`group`: the i4 group length).
 */
const weightAt = (
  storage: WeightStorage,
  words: Uint32Array<ArrayBuffer>,
  scale: Float32Array<ArrayBuffer>,
  group: number | undefined,
  k: number,
  row: number,
  col: number,
): number => {
  const index = row * k + col;
  if (storage === "bf16") {
    const word = words[index >> 1];
    BITS[0] = (index & 1) === 0 ? word << 16 : word & 0xffff0000;
    return VALUE[0];
  }
  if (storage === "i8" || group === undefined) {
    return Math.fround(((words[index >> 2] << (24 - (index & 3) * 8)) >> 24) * scale[row]);
  }
  // i4: element 2i is the low nibble of byte i and element 2i + 1 its high one; the code is q + 8
  const code = (words[index >> 3] >>> ((index & 7) * 4)) & 0xf;
  return Math.fround((code - 8) * scale[row * (k / group) + Math.floor(col / group)]);
};

/** One exception to compare: its row, the general variant of its profile, the profiles that hold it. */
type ExceptionRow = {
  readonly exception: LinearGemvShapeVariant;
  readonly general: LinearGemvVariant;
  readonly profiles: string[];
};

/**
 * The exceptions of the default profile and of every registered table that carries GEMV choices,
 * each once (a table holds the default's rows too).
 */
const exceptionRows = (): ExceptionRow[] => {
  const rows = new Map<string, ExceptionRow>();
  const carrying = BUILTIN_GEOMETRY_PROFILES.filter((profile) => profile.linearGemv !== undefined);
  for (const profile of [DEFAULT_GEOMETRY_PROFILE, ...carrying]) {
    const { sequential } = resolveGeometryProfile(profile).linearGemv;
    for (const exception of sequential.shapes) {
      const key = JSON.stringify([exception, sequential.variant]);
      const row = rows.get(key) ?? { exception, general: sequential.variant, profiles: [] };
      row.profiles.push(profile.id);
      rows.set(key, row);
    }
  }
  return [...rows.values()];
};

test.skipIf(!GPU_AVAILABLE)(
  "every per-shape exception of the sequential variant, of the default profile and of every registered table that carries GEMV choices, writes the words of the general variant on its shape (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const device = gpu.device;
    const rows = exceptionRows();
    assert(rows.length > 0, "no profile holds a per-shape exception");
    try {
      for (const { exception, general, profiles } of rows) {
        const { storage, variant } = exception;
        const row = `${profiles.join(" / ")}: ${storage} ${exception.n}×${exception.k} c${variant.cols}u${variant.unroll}`;
        // The storages this test builds weights for: another one needs its words and its decode above
        assert(storage === "i4" || storage === "i8" || storage === "bf16", `${row}: storage`);
        assert(
          variant.cols !== general.cols || variant.unroll !== general.unroll,
          `${row}: the row names the general variant`,
        );
        const unit = linearGemvUnit(storage);
        for (const group of storage === "i4" ? I4_GROUPS : [undefined]) {
          const where = group === undefined ? row : `${row} at group ${group}`;
          // The elements of the smallest row: one word, or one group where a group is longer
          const least = group ?? unit;
          for (const [n, k] of [
            [exception.n, exception.k],
            [68, least * (3 * variant.unroll - 1)],
            [4, least],
          ]) {
            device.pushErrorScope("validation");
            const owned: GPUBuffer[] = [];
            const make = (
              data: number | ArrayBufferView<ArrayBuffer>,
              usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            ): GPUBuffer => {
              const buffer = device.createBuffer({
                size: typeof data === "number" ? data : data.byteLength,
                usage,
              });
              owned.push(buffer);
              if (typeof data !== "number") device.queue.writeBuffer(buffer, 0, data);
              return buffer;
            };
            try {
              const words = weightWords(storage, n, k);
              const input = Float32Array.from({ length: k }, (_, i) => Math.sin(i * 0.037) * 0.75);
              const scales = group !== undefined ? n * (k / group) : storage === "i8" ? n : 0;
              const scale = new Float32Array(scales);
              for (let index = 0; index < scales; index++)
                scale[index] = ((index % 17) + 1) * 0.00017;
              const bias = new Float32Array(n);
              for (let index = 0; index < n; index++) bias[index] = ((index % 7) - 3) * 0.11;
              const buffers = [
                make(
                  linearGemvParams(storage, 1, n, k, group),
                  GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                ),
                make(input),
                make(words),
                make(bias),
                // COPY_DST: the output is cleared before each kernel
                make(
                  n * 4,
                  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
                ),
                // A float-stored weight has no scale binding
                ...(scales === 0 ? [] : [make(scale)]),
              ];
              const mapped = make(n * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
              const columns =
                n <= 256
                  ? Array.from({ length: n }, (_, column) => column)
                  : Array.from({ length: 256 }, (_, step) => Math.floor((step * (n - 1)) / 255));
              let expected: Uint32Array<ArrayBuffer> | undefined;
              for (const one of [general, variant]) {
                const ran = `${where} at N=${n} K=${k}, c${one.cols}u${one.unroll}`;
                const pipeline = await device.createComputePipelineAsync({
                  layout: "auto",
                  compute: {
                    module: device.createShaderModule({
                      code: linearGemvWgsl(storage, group, one),
                    }),
                    entryPoint: "main",
                  },
                });
                const bindings = device.createBindGroup({
                  layout: pipeline.getBindGroupLayout(0),
                  entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
                });
                const encoder = device.createCommandEncoder();
                // Both kernels write one buffer: cleared, a kernel that writes nothing leaves zeros
                // and not the other kernel's words
                encoder.clearBuffer(buffers[4]);
                const pass = encoder.beginComputePass();
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, bindings);
                pass.dispatchWorkgroups(Math.ceil(n / one.cols));
                pass.end();
                encoder.copyBufferToBuffer(buffers[4], 0, mapped, 0, n * 4);
                device.queue.submit([encoder.finish()]);
                await mapped.mapAsync(GPUMapMode.READ);
                const actual = new Uint32Array(mapped.getMappedRange().slice(0));
                mapped.unmap();
                expected ??= actual;
                let differing = 0;
                let first = -1;
                for (let column = 0; column < n; column++) {
                  if (actual[column] === expected[column]) continue;
                  differing += 1;
                  if (first < 0) first = column;
                }
                assertEquals(
                  differing,
                  0,
                  `${ran}: ${differing} of ${n} words differ from the general variant's (the first at column ${first})`,
                );
                const values = new Float32Array(actual.buffer);
                for (const column of columns) {
                  let reference = 0;
                  for (let element = 0; element < k; element++) {
                    reference +=
                      input[element] * weightAt(storage, words, scale, group, k, column, element);
                  }
                  reference += bias[column];
                  assert(
                    Math.abs(values[column] - reference) <=
                      GEMM_TOLERANCE.atol + GEMM_TOLERANCE.rtol * Math.abs(reference),
                    `${ran}: column ${column} is ${values[column]}, the CPU reference ${reference}`,
                  );
                }
              }
            } finally {
              await device.queue.onSubmittedWorkDone();
              for (const buffer of owned) buffer.destroy();
              const error = await device.popErrorScope();
              assertEquals(error, null);
            }
          }
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);
