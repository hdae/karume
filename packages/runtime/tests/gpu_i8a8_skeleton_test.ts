/**
 * The vec4 skeleton of the i8a8 linear (perf-ledger K-85 — the optional geometry field
 * `skeleton: "vec4"`) against the same geometry without it, as Uint32 words, on a real GPU.
 *
 * The i8a8 reduction is exact i32 addition, so a different fill (one `vec4<u32>` of global memory
 * per item instead of one `u32`) and a pack loop unrolled in codegen must return the very same
 * words: any difference is a fill that left a hole or put a word in the wrong place. The kernels
 * are dispatched directly (the WGSL the generator returns, pipelines cached by **the final WGSL
 * string**), so that the fault injection below — text mutations of the generated WGSL, not
 * switches in the generator — runs its own string (the MUST of tests/helpers/rank-dispatch.ts).
 *
 * Shapes: the Wan2.2 TI2V-5B DiT's K and N (3,072 → 3,072, 3,072 → 14,336, 14,336 → 3,072) with M
 * cut to a few tiles plus a partial one, and the tails — `k % 32 == 16` (a partial last K32 tile),
 * `k % 16 == 8` (no vec4 fill: the word fill with the unrolled loop), `k = 16`, `m = 1`, m and n off
 * the tile edges, `n % 4 != 0` (the scalar store). The linear has no row window. Both dot-product
 * variants (dp4a where the device lists it, and the emulated one), and w4a8 at group 32 and 64
 * with the unrolled loop (w4a8 has no vec4 fill).
 *
 * Without an adapter every case SKIPs explicitly (ADR 0005).
 */

import { assert, assertEquals, assertNotEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import {
  dp4aAvailable,
  linearI8a8Params,
  linearI8a8UsesVec4,
  linearI8a8UsesVecFill,
  linearI8a8Wgsl,
} from "../src/kernels/linear-i8a8.ts";
import {
  type I8a8Geometry,
  i8a8GeometryKeyPart,
  i8a8TileM,
  i8a8TileN,
} from "../src/kernels/i8a8-geometry.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

/** The spike's L1c tile (128×64, r16×4, wg16×8) at K32. */
const L1C_K32: I8a8Geometry = { regM: 16, regN: 4, wgX: 16, wgY: 8, tileK: 32, skeleton: "vec4" };

/** Tile 96 rows over 64 rows per vec4 slot at K32: the A side's second slot is partial (guarded). */
const PARTIAL_K32: I8a8Geometry = { ...L1C_K32, regM: 12 };

/** The skeleton geometries (the spike's measured tiles at K32 / K64, and one with a partial slot). */
const GEOMETRIES: readonly I8a8Geometry[] = [
  L1C_K32,
  { ...L1C_K32, tileK: 64 },
  { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 64, skeleton: "vec4" },
  { ...PARTIAL_K32, tileK: 64 },
  { ...L1C_K32, regM: 8 },
  PARTIAL_K32,
];

type Shape = { readonly name: string; readonly m: number; readonly n: number; readonly k: number };

/** The DiT shapes, with M cut to 300 (2 tiles of 128 + 44, 3 of 96 + 12, 4 of 64 + 44). */
const DIT_SHAPES: readonly Shape[] = [
  { name: "qkvo m300 n3072 k3072", m: 300, n: 3072, k: 3072 },
  { name: "up m300 n14336 k3072", m: 300, n: 14336, k: 3072 },
  { name: "down m300 n3072 k14336", m: 300, n: 3072, k: 14336 },
];

const TAIL_SHAPES: readonly Shape[] = [
  { name: "m1 n3072 k3072", m: 1, n: 3072, k: 3072 },
  { name: "k%32=16 m77 n136 k4112", m: 77, n: 136, k: 4112 },
  { name: "k%32=16 m129 n68 k48", m: 129, n: 68, k: 48 },
  { name: "k%16=8 m200 n136 k200 (word fill)", m: 200, n: 136, k: 200 },
  { name: "k16 m1 n64 k16", m: 1, n: 64, k: 16 },
  { name: "n%4=2 m33 n70 k208 (scalar store)", m: 33, n: 70, k: 208 },
  { name: "n%4=1 m200 n193 k64 (scalar store)", m: 200, n: 193, k: 64 },
];

const SHAPES = [...DIT_SHAPES, ...TAIL_SHAPES];

/** xorshift32 words (deterministic — a failure reproduces). */
const words = (count: number, seed: number): Uint32Array<ArrayBuffer> => {
  const out = new Uint32Array(Math.max(1, count));
  let state = seed >>> 0 || 1;
  for (let index = 0; index < out.length; index += 1) {
    state = (state ^ (state << 13)) >>> 0;
    state = (state ^ (state >>> 17)) >>> 0;
    state = (state ^ (state << 5)) >>> 0;
    out[index] = state;
  }
  return out;
};

/** Positive f32 scales near 1/127 and signed f32 bias values (deterministic). */
const floats = (count: number, seed: number, scale: boolean): Float32Array<ArrayBuffer> => {
  const source = words(count, seed);
  const out = new Float32Array(Math.max(1, count));
  for (let index = 0; index < out.length; index += 1) {
    const unit = source[index] / 2 ** 32;
    out[index] = scale ? (0.5 + unit) / 127 : unit * 2 - 1;
  }
  return out;
};

/** The device, its pipelines by final WGSL string, and the readback. */
type Bench = {
  readonly gpu: GpuContext;
  readonly pipelines: Map<string, GPUComputePipeline>;
};

const pipelineOf = (bench: Bench, wgsl: string): GPUComputePipeline => {
  const hit = bench.pipelines.get(wgsl);
  if (hit !== undefined) return hit;
  const { device } = bench.gpu;
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: wgsl }), entryPoint: "main" },
  });
  bench.pipelines.set(wgsl, pipeline);
  return pipeline;
};

const upload = (bench: Bench, data: ArrayBufferView<ArrayBuffer>, usage = STORAGE): GPUBuffer => {
  const buffer = bench.gpu.device.createBuffer({ size: Math.max(16, data.byteLength), usage });
  bench.gpu.device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

/** The inputs of one linear (bindings 1..6 of linear-i8a8.ts, the uniform dims at 0). */
type Inputs = {
  readonly shape: Shape;
  readonly groupSize?: number;
  readonly buffers: readonly GPUBuffer[];
  readonly out: GPUBuffer;
};

const createInputs = (bench: Bench, shape: Shape, groupSize?: number): Inputs => {
  const { m, n, k } = shape;
  const seed = m * 31 + n * 7 + k;
  const weightWords = groupSize === undefined ? n * (k / 4) : n * (k / 8);
  const scales = groupSize === undefined ? n : n * (k / groupSize);
  const buffers = [
    upload(
      bench,
      linearI8a8Params(m, n, k, groupSize),
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    ),
    upload(bench, words(m * (k / 4), seed + 1)),
    upload(bench, words(weightWords, seed + 2)),
    upload(bench, floats(n, seed + 3, false)),
  ];
  const out = bench.gpu.device.createBuffer({ size: Math.max(16, m * n * 4), usage: STORAGE });
  buffers.push(
    out,
    upload(bench, floats(scales, seed + 4, true)),
    upload(bench, floats(m, seed + 5, true)),
  );
  return { shape, ...(groupSize === undefined ? {} : { groupSize }), buffers, out };
};

const destroyInputs = (inputs: Inputs): void => {
  for (const buffer of inputs.buffers) buffer.destroy();
};

/** Runs one WGSL over the inputs (the output reset to 0 first) and reads the output words back. */
const run = async (
  bench: Bench,
  inputs: Inputs,
  wgsl: string,
  geometry: I8a8Geometry,
): Promise<Uint32Array<ArrayBuffer>> => {
  const { device } = bench.gpu;
  const { m, n } = inputs.shape;
  const bytes = Math.max(16, m * n * 4);
  device.pushErrorScope("validation");
  const pipeline = pipelineOf(bench, wgsl);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: inputs.buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const staging = device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.clearBuffer(inputs.out);
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(
    Math.ceil(n / i8a8TileN(geometry)),
    Math.ceil(m / i8a8TileM(geometry)),
    1,
  );
  pass.end();
  encoder.copyBufferToBuffer(inputs.out, 0, staging, 0, bytes);
  device.queue.submit([encoder.finish()]);
  const error = await device.popErrorScope();
  assertEquals(error, null, `validation: ${error?.message ?? ""}`);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Uint32Array(staging.getMappedRange().slice(0, m * n * 4));
  staging.unmap();
  staging.destroy();
  return out;
};

/** The words where two outputs differ (0 = identical). */
const differing = (left: Uint32Array, right: Uint32Array): number => {
  assertEquals(left.length, right.length, "output lengths");
  let count = 0;
  for (let index = 0; index < left.length; index += 1) if (left[index] !== right[index]) count += 1;
  return count;
};

/** The first differing index and its two words, for the failure message. */
const firstDifference = (left: Uint32Array, right: Uint32Array): string => {
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) {
      return `[${index}] ${left[index].toString(16)} vs ${right[index].toString(16)}`;
    }
  }
  return "none";
};

const withoutSkeleton = ({ skeleton: _skeleton, ...geometry }: I8a8Geometry): I8a8Geometry =>
  geometry;

const name = (geometry: I8a8Geometry): string => i8a8GeometryKeyPart(geometry, false);

/** The dot-product variants this device runs (dp4a only where it is listed). */
const dotVariants = (gpu: GpuContext): readonly boolean[] =>
  dp4aAvailable(gpu.wgslLanguageFeatures) ? [true, false] : [false];

const withBench = async (body: (bench: Bench) => Promise<void>): Promise<void> => {
  const gpu = await acquireGpu();
  try {
    await body({ gpu, pipelines: new Map() });
  } finally {
    gpu.destroy();
  }
};

/** The skeleton form of `geometry` for one i8 shape, and whether it took the vec4 fill. */
const skeletonWgsl = (
  geometry: I8a8Geometry,
  shape: Shape,
  dp4a: boolean,
): { readonly wgsl: string; readonly vecFill: boolean } => {
  const vecFill = linearI8a8UsesVecFill(geometry, "i8", shape.k);
  return {
    wgsl: linearI8a8Wgsl(linearI8a8UsesVec4(shape.n), dp4a, geometry, "i8", undefined, vecFill),
    vecFill,
  };
};

describe.skipIf(!GPU_AVAILABLE)(
  "the vec4 skeleton of the i8a8 linear returns the words of the same geometry without it (real GPU)",
  () => {
    it("i8 weights: every skeleton geometry × shape × dot-product variant, vec4 fill where k % 16 == 0 and the word fill elsewhere", async () => {
      await withBench(async (bench) => {
        if (!dp4aAvailable(bench.gpu.wgslLanguageFeatures)) {
          console.warn("[karume] dot4I8Packed is not listed: the dp4a variant is not checked here");
        }
        const forms = new Set<string>();
        for (const dp4a of dotVariants(bench.gpu)) {
          for (const shape of SHAPES) {
            const inputs = createInputs(bench, shape);
            try {
              const v4 = linearI8a8UsesVec4(shape.n);
              for (const geometry of GEOMETRIES) {
                const base = withoutSkeleton(geometry);
                const expected = await run(bench, inputs, linearI8a8Wgsl(v4, dp4a, base), base);
                const { wgsl, vecFill } = skeletonWgsl(geometry, shape, dp4a);
                const actual = await run(bench, inputs, wgsl, geometry);
                const label = `${name(geometry)} ${shape.name} dp4a=${dp4a} fill16=${vecFill}`;
                assertEquals(
                  differing(actual, expected),
                  0,
                  `${label}: first difference ${firstDifference(actual, expected)}`,
                );
                // A constant output would make the comparison vacuous
                assert(new Set(expected.subarray(0, 4096)).size > 1, `${label}: constant output`);
                forms.add(`${vecFill}:${v4}`);
              }
            } finally {
              destroyInputs(inputs);
            }
          }
        }
        // Both fills and both stores ran (vec4 fill × v4 / scalar store, word fill × v4)
        assertEquals([...forms].sort(), ["false:true", "true:false", "true:true"]);
      });
    });

    it("w4a8 (group 32 and 64): the unrolled loop with the word fill returns the words of the loop", async () => {
      const cases: readonly { readonly groupSize: number; readonly shape: Shape }[] = [
        { groupSize: 32, shape: { name: "m300 n3072 k3072", m: 300, n: 3072, k: 3072 } },
        { groupSize: 32, shape: { name: "m77 n136 k1024", m: 77, n: 136, k: 1024 } },
        { groupSize: 32, shape: { name: "m33 n70 k256 (scalar store)", m: 33, n: 70, k: 256 } },
        { groupSize: 64, shape: { name: "m300 n3072 k3072", m: 300, n: 3072, k: 3072 } },
        { groupSize: 64, shape: { name: "m1 n64 k64", m: 1, n: 64, k: 64 } },
        { groupSize: 64, shape: { name: "m33 n70 k256 (scalar store)", m: 33, n: 70, k: 256 } },
      ];
      await withBench(async (bench) => {
        let checked = 0;
        for (const dp4a of dotVariants(bench.gpu)) {
          for (const { groupSize, shape } of cases) {
            const inputs = createInputs(bench, shape, groupSize);
            try {
              const v4 = linearI8a8UsesVec4(shape.n);
              for (const geometry of GEOMETRIES.filter((g) => groupSize % g.tileK === 0)) {
                assertEquals(linearI8a8UsesVecFill(geometry, "i4", shape.k), false);
                const base = withoutSkeleton(geometry);
                const expected = await run(
                  bench,
                  inputs,
                  linearI8a8Wgsl(v4, dp4a, base, "i4", groupSize),
                  base,
                );
                const actual = await run(
                  bench,
                  inputs,
                  linearI8a8Wgsl(v4, dp4a, geometry, "i4", groupSize),
                  geometry,
                );
                const label = `${name(geometry)} g${groupSize} ${shape.name} dp4a=${dp4a}`;
                assertEquals(
                  differing(actual, expected),
                  0,
                  `${label}: first difference ${firstDifference(actual, expected)}`,
                );
                assert(new Set(expected.subarray(0, 4096)).size > 1, `${label}: constant output`);
                checked += 1;
              }
            } finally {
              destroyInputs(inputs);
            }
          }
        }
        // group 32 takes the three K32 geometries, group 64 all six
        assertEquals(checked, dotVariants(bench.gpu).length * (3 * 3 + 3 * 6));
      });
    });
  },
);

/** A text mutation of the generated WGSL (the fault the comparison must see). */
type Mutation = {
  readonly name: string;
  readonly geometry: I8a8Geometry;
  readonly mutate: (wgsl: string) => string;
};

/** Applies `pattern` once and fails if it did not match (a no-op mutation would be green). */
const replaceOnce = (wgsl: string, pattern: RegExp | string, replacement: string): string => {
  const mutated = wgsl.replace(pattern, replacement);
  assertNotEquals(mutated, wgsl, `the mutation ${String(pattern)} did not apply`);
  return mutated;
};

const MUTATIONS: readonly Mutation[] = [
  {
    // One of the four whole-word stores of a loaded vec4 is missing (a hole in the tile)
    name: "a dropped lane store (A slot 0, lane y)",
    geometry: L1C_K32,
    mutate: (wgsl) => replaceOnce(wgsl, /^ *sa\[sa_at \+ \d+u\] = av0\.y;\n/m, ""),
  },
  {
    // The global read of slot 0 takes the next quad (16 i8 further) instead of its own
    name: "a wrong quad offset (A slot 0 reads the next quad)",
    geometry: L1C_K32,
    mutate: (wgsl) =>
      replaceOnce(wgsl, "av0 = xq[arow_base0 + aquad];", "av0 = xq[arow_base0 + aquad + 1u];"),
  },
  {
    // The partial slot stores past the tile edge (into the next pack's rows)
    name: "a removed partial-slot guard (A slot 1 of the 96-row tile)",
    geometry: PARTIAL_K32,
    mutate: (wgsl) =>
      replaceOnce(wgsl, /^( *)if \(ar \+ 64u < 96u\) \{\n((?: *sa\[.*\n){4})\1\}\n/m, "$2"),
  },
  {
    // One block of the unrolled pack loop reads the previous pack's B words
    name: "a wrong pack offset in the unrolled loop (pack 1 reads pack 0 of B)",
    geometry: L1C_K32,
    mutate: (wgsl) =>
      replaceOnce(wgsl, "let bcol = 64u + lid.x * 4u;", "let bcol = 0u + lid.x * 4u;"),
  },
];

/**
 * The shapes each fault runs on (the vec4 fill runs on all of them). It is pinned red on the DiT
 * shapes; the tails are logged (a read past the end of a binding is clamped into it by the
 * robustness of the backend, so a fault there may read the right word by accident).
 */
const FAULT_SHAPES: readonly Shape[] = [
  ...DIT_SHAPES,
  { name: "m1 n3072 k3072", m: 1, n: 3072, k: 3072 },
  { name: "k%32=16 m77 n136 k4112", m: 77, n: 136, k: 4112 },
  { name: "k16 m1 n64 k16", m: 1, n: 64, k: 16 },
];

describe.skipIf(!GPU_AVAILABLE)(
  "fault injection: each mutation of the vec4 fill or the unrolled loop turns the comparison red (real GPU)",
  () => {
    it("a dropped lane store, a wrong quad offset, a removed partial-slot guard and a wrong pack offset each differ from the word form", async () => {
      await withBench(async (bench) => {
        const dp4a = dotVariants(bench.gpu)[0];
        const caught = new Map<string, string[]>();
        for (const shape of FAULT_SHAPES) {
          const inputs = createInputs(bench, shape);
          try {
            for (const mutation of MUTATIONS) {
              const base = withoutSkeleton(mutation.geometry);
              const v4 = linearI8a8UsesVec4(shape.n);
              const expected = await run(bench, inputs, linearI8a8Wgsl(v4, dp4a, base), base);
              const { wgsl, vecFill } = skeletonWgsl(mutation.geometry, shape, dp4a);
              assert(vecFill, `${shape.name}: the vec4 fill does not run`);
              // Control: the unmutated form is identical
              assertEquals(
                differing(await run(bench, inputs, wgsl, mutation.geometry), expected),
                0,
              );
              const count = differing(
                await run(bench, inputs, mutation.mutate(wgsl), mutation.geometry),
                expected,
              );
              const entry = caught.get(mutation.name) ?? [];
              entry.push(`${shape.name}: ${count}/${expected.length}`);
              caught.set(mutation.name, entry);
              if (DIT_SHAPES.includes(shape)) {
                assert(count > 0, `${mutation.name} is not seen on ${shape.name}`);
              }
            }
          } finally {
            destroyInputs(inputs);
          }
        }
        for (const [mutation, shapes] of caught) {
          console.log(`[karume] K-85 fault '${mutation}' differing words: ${shapes.join("; ")}`);
        }
        assertEquals(
          [...caught.keys()],
          MUTATIONS.map((mutation) => mutation.name),
        );
      });
    });
  },
);
