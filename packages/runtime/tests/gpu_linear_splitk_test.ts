/**
 * The split-K linear (`src/kernels/linear-splitk.ts` — the split of the GEMM skeleton) on a real GPU
 * — a `reorder` strategy: the products are the reference kernel's and only the order of the
 * additions differs, so its checks are bands, structure and one exact identity, not bit equality
 * with the reference kernel at two or more slices.
 *
 * 1. Against the f64 reference: every output inside the op's band (`GEMM_TOLERANCE`).
 * 2. Against the reference kernel on the same input: inside {@link SPLITK_AB_BAND} times that band
 *    (a declaration — its derivation is the constant's doc).
 * 3. Structure, checked exactly: with one slice the two dispatches give the reference GEMM's words
 *    on every tile geometry the linear GEMM has, scalar and vec4 (the split shares the GEMM's loop
 *    and only regroups it); on inputs whose products and sums are exact in f32 the order of the
 *    additions cannot matter, so the output equals the reference kernel's bit for bit (a dropped
 *    term, a wrong slice or a missed edge element shows here); a NaN in one input row reaches every
 *    output of that row and no other; two runs of one Session give the same bits.
 * 4. Fault injection (recorded 2026-11-28, RTX 3080 Ti, Dawn / Vulkan): with every slice starting
 *    one K tile late (`t_first = wid.z * slice_tiles + min(wid.z, slice_extra) + 1u` in the
 *    generator, so each slice drops its first tile and reads the next slice's first) the f64 band
 *    failed at the first shape (k = 512: worst |error| 2.57, about 12,100 times the band), the exact
 *    check failed, and the one-slice identity failed at the first geometry it compared.
 *
 * It also checks where the strategy applies: only under the private option `LINEAR_SPLIT_K`, for an
 * f32-stored weight whose reduction is long enough to cut, and never for a single row (the GEMV
 * family keeps it).
 */

import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import type { GpuContext } from "../src/gpu/device.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { BUFFER_USAGE as U, MAP_MODE } from "../src/gpu/webgpu-constants.ts";
import { type GemmGeometry, gemmTileM, gemmTileN } from "../src/kernels/gemm-geometry.ts";
import { gemmUsesVec4 } from "../src/kernels/gemm.ts";
import { linearKey, linearWgsl } from "../src/kernels/linear.ts";
import {
  LINEAR_SPLITK_COMBINE_GROUP,
  linearSplitKCombineKey,
  linearSplitKCombineWgsl,
  linearSplitKGemmWgsl,
  linearSplitKParams,
  linearSplitKSlices,
} from "../src/kernels/linear-splitk.ts";
import { GEMM_TOLERANCE } from "../src/reference/op-tolerance.ts";
import { createSessionFromContainer, LINEAR_SPLIT_K } from "../src/runtime/executor.ts";
import type { TensorInput } from "./helpers/container-write.ts";
import { linearGemmGeometries } from "./helpers/geometry.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import {
  type DeclarationJson,
  f32Bytes,
  GRAPH_NAME,
  openModelBytes,
} from "./helpers/model-fixture.ts";

/**
 * The A/B band against the reference kernel, as a multiple of the op band at the f64 value
 * (`GEMM_TOLERANCE.atol + rtol · |f64|`). A declaration (ADR 0147 decision 2: at least 4 × the worst
 * measured, at most 2 × the f64 band):
 *
 * | Shape (m × k × n, slices) | Worst \|split − reference\| / op band |
 * | --- | --- |
 * | 3 × 512 × 5, 2 | 0.028 |
 * | 16 × 1536 × 64, 4 | 0.144 |
 * | 9 × 8960 × 37, 16 | 0.393 |
 *
 * Measured 2026-11-28 on an RTX 3080 Ti (Dawn / Vulkan, the nvidia-ampere profile) with the
 * {@link irregular} values: the worst is 0.393, so the band is at least 1.57 and at most 2.
 */
const SPLITK_AB_BAND = 2;

type Shape = { readonly m: number; readonly n: number; readonly k: number };

type Values = {
  readonly x: Float32Array<ArrayBuffer>;
  readonly w: Float32Array<ArrayBuffer>;
  readonly bias: Float32Array<ArrayBuffer>;
};

/** Values with mixed signs and magnitudes, so the partial sums cancel and round. */
const irregular = ({ m, n, k }: Shape): Values => ({
  x: Float32Array.from({ length: m * k }, (_, i) => Math.sin(i * 0.037) * 0.75),
  w: Float32Array.from({ length: n * k }, (_, i) => Math.cos(i * 0.011 + 0.3) * 0.5),
  bias: Float32Array.from({ length: n }, (_, i) => ((i % 7) - 3) * 0.11),
});

/**
 * Small integers: every product and every partial sum is an integer far below 2^24, so each is exact
 * in f32 and the sum does not depend on the order of the additions.
 */
const exact = ({ m, n, k }: Shape): Values => ({
  x: Float32Array.from({ length: m * k }, (_, i) => ((i * 7) % 5) - 2),
  w: Float32Array.from({ length: n * k }, (_, i) => ((i * 11) % 7) - 3),
  bias: Float32Array.from({ length: n }, (_, i) => (i % 9) - 4),
});

/** The op's meaning: the sum in f64 over the f32 inputs, then the bias. */
const referenceF64 = ({ m, n, k }: Shape, { x, w, bias }: Values): Float64Array<ArrayBuffer> => {
  const y = new Float64Array(m * n);
  for (let row = 0; row < m; row += 1) {
    for (let col = 0; col < n; col += 1) {
      let sum = 0;
      for (let inner = 0; inner < k; inner += 1) sum += x[row * k + inner] * w[col * k + inner];
      y[row * n + col] = sum + bias[col];
    }
  }
  return y;
};

type Run = {
  readonly y: Float32Array<ArrayBuffer>;
  /** The pipeline keys of the run's dispatches. */
  readonly keys: readonly string[];
  /** The same request once more on the same Session. */
  readonly again: Float32Array<ArrayBuffer>;
};

const runLinear = async (
  gpu: GpuContext,
  { m, n, k }: Shape,
  { x, w, bias }: Values,
  splitK: boolean,
): Promise<Run> => {
  const graph: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [m, k] }],
    outputs: ["y"],
    initializers: { w: {}, b: {} },
    values: {
      w: { dtype: "f32", shape: [n, k] },
      b: { dtype: "f32", shape: [n] },
      y: { dtype: "f32", shape: [m, n] },
    },
    nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
  };
  const tensors: readonly TensorInput[] = [
    { graph: GRAPH_NAME, initializer: "w", bytes: f32Bytes(w), encoding: { codec: "f32" } },
    { graph: GRAPH_NAME, initializer: "b", bytes: f32Bytes(bias), encoding: { codec: "f32" } },
  ];
  const session = await createSessionFromContainer(
    gpu,
    await openModelBytes(graph, tensors),
    GRAPH_NAME,
    splitK ? { [LINEAR_SPLIT_K]: true } : {},
  );
  try {
    const request = { x: { dtype: "f32", shape: [m, k], data: x } } as const;
    const first = (await session.run(request)).y;
    assertEquals(first.shape, [m, n]);
    const keys = (session.diagnostics().lastRunPipelines ?? []).map((row) => row.key);
    const again = (await session.run(request)).y;
    assert(first.data instanceof Float32Array && again.data instanceof Float32Array);
    return { y: first.data.slice(), keys, again: again.data.slice() };
  } finally {
    await session.dispose();
  }
};

const bits = (values: Float32Array<ArrayBuffer>): Uint32Array<ArrayBuffer> =>
  new Uint32Array(values.buffer, values.byteOffset, values.length).slice();

/** Whether the run took the split (its partial at `slices` and the combine) and no plain GEMM. */
const usesSplitK = (keys: readonly string[], slices?: number): boolean =>
  keys.some(
    (key) =>
      key.startsWith("linear_splitk_gemm:") &&
      (slices === undefined || key.endsWith(`:s${slices}`)),
  ) &&
  keys.includes(linearSplitKCombineKey()) &&
  !keys.some((key) => key.startsWith("linear:"));

/** The op band at an f64 value. */
const band = (reference: number): number =>
  GEMM_TOLERANCE.atol + GEMM_TOLERANCE.rtol * Math.abs(reference);

/**
 * The worst |value − against| and the worst ratio of it to the op band at the f64 value of that
 * element.
 */
const worst = (
  values: Float32Array<ArrayBuffer>,
  against: ArrayLike<number>,
  reference: Float64Array<ArrayBuffer>,
): { readonly maxAbs: number; readonly bandUse: number } => {
  let maxAbs = 0;
  let bandUse = 0;
  for (let i = 0; i < values.length; i += 1) {
    const error = Math.abs(values[i] - against[i]);
    maxAbs = Math.max(maxAbs, error);
    bandUse = Math.max(bandUse, Number.isFinite(error) ? error / band(reference[i]) : Infinity);
  }
  return { maxAbs, bandUse };
};

/** m and n off the reference geometry's tiles; k for 2, 4 and 16 slices. */
const SHAPES: readonly Shape[] = [
  { m: 3, n: 5, k: 512 },
  { m: 16, n: 64, k: 1536 },
  { m: 9, n: 37, k: 8960 },
];

describe.skipIf(!GPU_AVAILABLE)("the split-K linear (real GPU)", () => {
  it("stays inside the op's band against the f64 reference and inside the declared A/B band against the reference kernel", async () => {
    const gpu = await acquireGpu();
    try {
      for (const shape of SHAPES) {
        const values = irregular(shape);
        const reference = referenceF64(shape, values);
        const plain = await runLinear(gpu, shape, values, false);
        const split = await runLinear(gpu, shape, values, true);
        const slices = linearSplitKSlices(shape.k);
        const where = `[${shape.m},${shape.k}] × [${shape.n},${shape.k}]`;
        assert(usesSplitK(split.keys, slices), `${where}: ${split.keys.join(" ")}`);
        const ofSplit = worst(split.y, reference, reference);
        const ofPlain = worst(plain.y, reference, reference);
        const between = worst(split.y, plain.y, reference);
        console.log(
          `[linear-splitk] ${where} (${slices} slices): |error| against f64 — ` +
            `split-K ${ofSplit.maxAbs.toExponential(2)} (${(ofSplit.bandUse * 100).toFixed(1)}% of the band), ` +
            `reference kernel ${ofPlain.maxAbs.toExponential(2)} (${(ofPlain.bandUse * 100).toFixed(1)}%); ` +
            `between the two ${between.maxAbs.toExponential(2)} (${(between.bandUse * 100).toFixed(1)}%)`,
        );
        assert(ofSplit.bandUse <= 1, `${where}: outside the band against f64 (${ofSplit.maxAbs})`);
        assert(
          between.bandUse <= SPLITK_AB_BAND,
          `${where}: outside the A/B band against the reference kernel (${between.maxAbs})`,
        );
      }
    } finally {
      gpu.destroy();
    }
  });

  it("equals the reference kernel bit for bit on inputs whose sums are exact, edges included", async () => {
    const gpu = await acquireGpu();
    try {
      for (const shape of SHAPES) {
        const values = exact(shape);
        const plain = await runLinear(gpu, shape, values, false);
        const split = await runLinear(gpu, shape, values, true);
        assert(usesSplitK(split.keys));
        assertEquals(bits(split.y), bits(plain.y));
        // The exact value itself, so that two kernels wrong in the same way do not pass.
        assertEquals(Array.from(split.y), Array.from(referenceF64(shape, values)));
      }
    } finally {
      gpu.destroy();
    }
  });

  it("carries a NaN of one input row to every output of that row and to no other, and repeats its bits", async () => {
    const gpu = await acquireGpu();
    try {
      const shape = { m: 9, n: 37, k: 1536 };
      const values = irregular(shape);
      const poisoned = 4;
      // In the last slice, so the combine pass is what carries it.
      values.x[poisoned * shape.k + shape.k - 1] = Number.NaN;
      const split = await runLinear(gpu, shape, values, true);
      assert(usesSplitK(split.keys));
      for (let row = 0; row < shape.m; row += 1) {
        for (let col = 0; col < shape.n; col += 1) {
          assertEquals(
            Number.isNaN(split.y[row * shape.n + col]),
            row === poisoned,
            `row ${row} col ${col}`,
          );
        }
      }
      assertEquals(bits(split.again), bits(split.y));
    } finally {
      gpu.destroy();
    }
  });

  it("is taken only under the private option, for a reduction long enough to cut, and never for one row", async () => {
    const gpu = await acquireGpu();
    try {
      const long = { m: 16, n: 64, k: 1536 };
      assertEquals(usesSplitK((await runLinear(gpu, long, irregular(long), false)).keys), false);
      const short = { m: 16, n: 64, k: 448 };
      assertEquals(linearSplitKSlices(short.k), undefined);
      assertEquals(usesSplitK((await runLinear(gpu, short, irregular(short), true)).keys), false);
      const oneRow = { m: 1, n: 64, k: 1536 };
      assertEquals(usesSplitK((await runLinear(gpu, oneRow, irregular(oneRow), true)).keys), false);
    } finally {
      gpu.destroy();
    }
  });
});

/** The word an output buffer is filled with before a dispatch (a quiet NaN). */
const UNWRITTEN = 0x7fc00000;

/**
 * One linear through the reference GEMM and through the split (partial + combine) at `slices`, by
 * raw dispatches at an explicit geometry: the output words of both, nothing else.
 */
const runRaw = async (
  gpu: GpuContext,
  geometry: GemmGeometry,
  { m, n, k }: Shape,
  { x, w, bias }: Values,
  slices: number,
): Promise<{ readonly reference: Uint32Array; readonly split: Uint32Array }> => {
  const { device } = gpu;
  const v4 = gemmUsesVec4(k, n);
  const owned: GPUBuffer[] = [];
  const make = (value: number | ArrayBufferView<ArrayBuffer>, usage = U.STORAGE | U.COPY_DST) => {
    const buffer = device.createBuffer({
      size: typeof value === "number" ? value : value.byteLength,
      usage,
    });
    owned.push(buffer);
    if (typeof value !== "number") device.queue.writeBuffer(buffer, 0, value);
    return buffer;
  };
  const pipeline = (code: string): Promise<GPUComputePipeline> =>
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
    });
  const work = async (): Promise<{
    readonly reference: Uint32Array;
    readonly split: Uint32Array;
  }> => {
    const params = make(linearSplitKParams(m, n, k, slices), U.UNIFORM | U.COPY_DST);
    const xs = make(x);
    const ws = make(w);
    const biases = make(bias);
    const unwritten = new Uint32Array(m * n).fill(UNWRITTEN);
    const outReference = make(unwritten, U.STORAGE | U.COPY_SRC | U.COPY_DST);
    const outSplit = make(unwritten, U.STORAGE | U.COPY_SRC | U.COPY_DST);
    const partials = make(slices * m * n * 4, U.STORAGE);
    const [reference, partial, combine] = await Promise.all([
      pipeline(linearWgsl("f32", v4, "f32", m, undefined, geometry)),
      pipeline(linearSplitKGemmWgsl("f32", v4, slices, m, undefined, geometry)),
      pipeline(linearSplitKCombineWgsl()),
    ]);
    const group = (target: GPUComputePipeline, entries: readonly [number, GPUBuffer][]) =>
      device.createBindGroup({
        layout: target.getBindGroupLayout(0),
        entries: entries.map(([binding, buffer]) => ({ binding, resource: { buffer } })),
      });
    const tiles = [Math.ceil(n / gemmTileN(geometry)), Math.ceil(m / gemmTileM(geometry))] as const;
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
      ]),
    );
    pass.dispatchWorkgroups(tiles[0], tiles[1], 1);
    pass.setPipeline(partial);
    pass.setBindGroup(
      0,
      group(partial, [
        [0, params],
        [1, xs],
        [2, ws],
        [4, partials],
      ]),
    );
    pass.dispatchWorkgroups(tiles[0], tiles[1], slices);
    pass.setPipeline(combine);
    pass.setBindGroup(
      0,
      group(combine, [
        [0, params],
        [1, partials],
        [2, biases],
        [3, outSplit],
      ]),
    );
    pass.dispatchWorkgroups(Math.ceil((m * n) / LINEAR_SPLITK_COMBINE_GROUP));
    pass.end();
    const read = (source: GPUBuffer): GPUBuffer => {
      const mapped = make(m * n * 4, U.MAP_READ | U.COPY_DST);
      encoder.copyBufferToBuffer(source, 0, mapped, 0, m * n * 4);
      return mapped;
    };
    const mappedReference = read(outReference);
    const mappedSplit = read(outSplit);
    device.queue.submit([encoder.finish()]);
    const words = async (mapped: GPUBuffer): Promise<Uint32Array> => {
      await mapped.mapAsync(MAP_MODE.READ);
      const out = new Uint32Array(mapped.getMappedRange().slice(0));
      mapped.unmap();
      return out;
    };
    return { reference: await words(mappedReference), split: await words(mappedSplit) };
  };
  // A validation error (an invalid shader, a bad binding) throws instead of reading the initial fill.
  device.pushErrorScope("validation");
  const settled = await work().then(
    (value) => ({ ok: true, value }) as const,
    (error: unknown) => ({ ok: false, error }) as const,
  );
  const error = await device.popErrorScope();
  await device.queue.onSubmittedWorkDone();
  for (const buffer of owned) buffer.destroy();
  if (error !== null) throw new Error(`split-K raw dispatch: ${error.message}`);
  if (!settled.ok) throw settled.error;
  return settled.value;
};

describe.skipIf(!GPU_AVAILABLE)("the split-K linear at one slice (real GPU)", () => {
  it("gives the reference GEMM's words on every tile geometry the linear GEMM has, scalar and vec4", async () => {
    const gpu = await acquireGpu();
    try {
      const geometries = linearGemmGeometries();
      // The embedded profiles hold this many today; a new one joins the loop by itself.
      assert(geometries.length >= 10, `${geometries.length} geometries`);
      for (const geometry of geometries) {
        // Two row tiles and two column tiles with ragged edges, and a ragged last K tile.
        const rows = gemmTileM(geometry) + 3;
        for (const shape of [
          { m: rows, n: gemmTileN(geometry) + 4, k: 100 },
          { m: rows, n: gemmTileN(geometry) + 5, k: 101 },
        ]) {
          const where = `${linearKey("f32", gemmUsesVec4(shape.k, shape.n), "f32", undefined, undefined, geometry)} ${shape.m}×${shape.n}×${shape.k}`;
          const { reference, split } = await runRaw(gpu, geometry, shape, irregular(shape), 1);
          assert(!reference.includes(UNWRITTEN), `${where}: the reference left outputs unwritten`);
          assertEquals(split, reference, where);
        }
      }
    } finally {
      gpu.destroy();
    }
  });
});

describe("linearSplitKSlices", () => {
  it("takes the most slices that divide k and leave each at least 256 terms", () => {
    assertEquals(linearSplitKSlices(8960), 16);
    assertEquals(linearSplitKSlices(2048), 8);
    assertEquals(linearSplitKSlices(1536), 4);
    assertEquals(linearSplitKSlices(512), 2);
    assertEquals(linearSplitKSlices(511), undefined);
    assertEquals(linearSplitKSlices(513), undefined);
  });
});
