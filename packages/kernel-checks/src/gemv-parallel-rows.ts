/**
 * The row-blocked parallel GEMV (M = 2..8 of linear at the level `reorder`, where a profile row
 * carries a row block) against the plain parallel GEMV at the same lanes: the same words of every
 * row in the same order, the same multiply-add text and the same tree, so every output word MUST be
 * identical. Not a speed measurement.
 *
 * Per case (storage, K, lanes, rows a thread, M): the row-blocked output equals the parallel output
 * word for word; both leave the rows past M untouched (a block reaching past M stores only the rows
 * inside it); row 0 equals the parallel GEMV's row 0 at M = 1 (decode and verify agree); and every
 * output is within the GEMM band of an FP64 reference.
 *
 * Fault injection: the node GPU test (packages/runtime/tests/gpu_linear_gemv_parallel_rows_test.ts)
 * runs this check on mutated WGSL through {@link GemvParallelRowsCheckOptions.mutate} and asserts
 * that the identity criteria fail.
 */

import type { GpuContext } from "../../runtime/src/gpu/device.ts";
import { BUFFER_USAGE as U, MAP_MODE } from "../../runtime/src/gpu/webgpu-constants.ts";
import {
  LINEAR_GEMV_PARALLEL_ROWS,
  type LinearGemvParallelLanes,
  type LinearGemvParallelRows,
  linearGemvParallelRowsFits,
  linearGemvParallelRowsWgsl,
  linearGemvParallelWgsl,
  linearGemvParams,
  linearGemvUnit,
} from "../../runtime/src/kernels/linear-gemv.ts";
import { GEMM_TOLERANCE } from "../../runtime/src/reference/op-tolerance.ts";
import {
  type CheckCase,
  type CheckOptions,
  type CheckPart,
  compareFill,
  compareWords,
  type Criterion,
  type ExpectedCases,
  f32Word,
  scopedGpuWork,
  throwIfAborted,
  type WordFailure,
} from "./result.ts";

type Storage = "i2" | "i4" | "i8";

/** The word the output buffer is filled with before each dispatch (a quiet NaN). */
const UNWRITTEN = 0x7fc00000;
/** Output rows of the buffers (the largest M). */
const ROWS = 8;
/**
 * Output columns: a workgroup holds 128 / lanes columns (64, 32, 16, 8 or 4), and 37 is a multiple
 * of none, so the last workgroup of every lane count has columns past N (the edge-column guard).
 */
const COLUMNS = 37;
const LANES: readonly LinearGemvParallelLanes[] = [2, 4, 8, 16, 32];
const STORAGES: readonly Storage[] = ["i2", "i4", "i8"];
const PART = "gemv-parallel-rows";

/**
 * The row counts a block of `rows` rows is run at: every M a row's row block can be selected at,
 * from one block to 8 — whole blocks, and every number of rows the last y tile can be left with.
 */
const rowCountsOf = (rows: LinearGemvParallelRows): number[] =>
  Array.from({ length: ROWS - rows + 1 }, (_, i) => rows + i);

/**
 * The (K, group) shapes per storage: one unit (fewer words than lanes), five units, 4096 (several
 * chunks at every lane count), plus i4's large groups.
 */
const shapesOf = (storage: Storage): readonly { k: number; group?: number }[] => {
  const unit = linearGemvUnit(storage);
  const group = storage === "i4" ? 32 : undefined;
  const shapes = [
    { k: unit, group },
    { k: unit * 5, group },
    { k: 4096, group },
  ];
  if (storage === "i4") {
    shapes.push({ k: 1536, group: 512 }, { k: 6144, group: 2048 }, { k: 4096, group: 4096 });
  }
  return shapes;
};

/** The (lanes, rows) pairs the kernel exists at for a storage. */
const blocksOf = (
  storage: Storage,
): { readonly lanes: LinearGemvParallelLanes; readonly rows: LinearGemvParallelRows }[] =>
  LANES.flatMap((lanes) =>
    LINEAR_GEMV_PARALLEL_ROWS.filter((rows) =>
      linearGemvParallelRowsFits(storage, lanes, rows),
    ).map((rows) => ({ lanes, rows })),
  );

/** Deterministic inputs that cross packed-word boundaries, per-group scales and mixed signs. */
const fixture = (
  storage: Storage,
  n: number,
  k: number,
  group?: number,
): {
  packed: Uint32Array<ArrayBuffer>;
  input: Float32Array<ArrayBuffer>;
  bias: Float32Array<ArrayBuffer>;
  scale: Float32Array<ArrayBuffer>;
  reference: Float64Array<ArrayBuffer>;
} => {
  const bits = storage === "i2" ? 2 : storage === "i4" ? 4 : 8;
  const packed = Uint32Array.from(
    { length: (n * k * bits) / 32 },
    (_, i) => Math.imul(i + 1, 0x9e3779b9) >>> 0,
  );
  // Every row differs from every other, so a row read from another row's activations shows.
  const input = Float32Array.from({ length: ROWS * k }, (_, i) => Math.sin(i * 0.037) * 0.75);
  const bias = Float32Array.from({ length: n }, (_, i) => ((i % 7) - 3) * 0.11);
  const scale = Float32Array.from(
    { length: n * (group === undefined ? 1 : k / group) },
    (_, i) => ((i % 17) + 1) * 0.00017,
  );
  const reference = new Float64Array(ROWS * n);
  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < n; col++) {
      let sum = 0;
      for (let inner = 0; inner < k; inner++) {
        const index = col * k + inner;
        const code =
          (packed[Math.floor((index * bits) / 32)] >>> ((index * bits) % 32)) & ((1 << bits) - 1);
        const q = bits === 8 ? (code << 24) >> 24 : code - (1 << (bits - 1));
        const s = scale[group === undefined ? col : col * (k / group) + Math.floor(inner / group)];
        sum += input[row * k + inner] * Math.fround(q * s);
      }
      reference[row * n + col] = sum + bias[col];
    }
  }
  return { packed, input, bias, scale, reference };
};

/** Every output finite and within the GEMM band of the FP64 reference. */
const withinBand = (
  name: string,
  actual: Uint32Array<ArrayBuffer>,
  reference: Float64Array<ArrayBuffer>,
): Criterion => {
  const values = new Float32Array(actual.buffer, actual.byteOffset, actual.length);
  let failures = 0;
  let worst: WordFailure | undefined;
  let worstError = -1;
  for (let i = 0; i < values.length; i++) {
    const expected = reference[i];
    const error = Math.abs(values[i] - expected);
    const bound = GEMM_TOLERANCE.atol + GEMM_TOLERANCE.rtol * Math.abs(expected);
    // A non-finite output fails whatever the error (NaN compares false).
    const finite = Number.isFinite(values[i]);
    if (finite && error <= bound) continue;
    failures += 1;
    const rank = finite ? error : Number.POSITIVE_INFINITY;
    if (rank > worstError) {
      worstError = rank;
      worst = {
        index: i,
        expected: f32Word(expected),
        actual: actual[i],
        at: "worst",
        note: `FP64 ${expected}, got ${values[i]} (band ${bound})`,
      };
    }
  }
  return worst === undefined
    ? { name, words: values.length, failures }
    : { name, words: values.length, failures, failure: worst };
};

/** The number of cases {@link checkGemvParallelRows} runs. */
const caseCount = (): number =>
  STORAGES.reduce(
    (total, storage) =>
      total +
      shapesOf(storage).length *
        blocksOf(storage).reduce((sum, { rows }) => sum + rowCountsOf(rows).length, 0),
    0,
  );

/**
 * The cases {@link checkGemvParallelRows} returns: per storage its shapes (i2 / i8 3, i4 6) × the
 * (lanes, rows) pairs the kernel exists at × the row counts of each block height.
 */
export const GEMV_PARALLEL_ROWS_CASES = {
  "gemv-parallel-rows": 675,
} as const satisfies ExpectedCases;

export type GemvParallelRowsCheckOptions = CheckOptions & {
  /**
   * Rewrites the row-blocked kernel's WGSL before it is compiled (a fault injection; the node test
   * only). A rewrite that leaves a text unchanged throws: the anchor no longer matches.
   */
  readonly mutate?: (wgsl: string) => string;
};

/**
 * Runs the check on any device (the kernel needs no optional feature). The same check runs in the
 * node GPU test and on the workbench's `kernel-check` entry.
 */
export const checkGemvParallelRows = async (
  gpu: GpuContext,
  options: GemvParallelRowsCheckOptions = {},
): Promise<CheckPart> => {
  const device = gpu.device;
  const cases: CheckCase[] = [];
  const total = caseCount();
  for (const storage of STORAGES) {
    for (const { k, group } of shapesOf(storage)) {
      const n = COLUMNS,
        data = fixture(storage, n, k, group),
        owned: GPUBuffer[] = [];
      const make = (
        value: number | ArrayBufferView<ArrayBuffer>,
        usage = U.STORAGE | U.COPY_DST,
      ): GPUBuffer => {
        const buffer = device.createBuffer({
          size: typeof value === "number" ? value : value.byteLength,
          usage,
        });
        owned.push(buffer);
        if (typeof value !== "number") device.queue.writeBuffer(buffer, 0, value);
        return buffer;
      };
      await scopedGpuWork(gpu, `${storage} K${k}`, async () => {
        try {
          const buffers = [
            make(linearGemvParams(storage, ROWS, n, k, group), U.UNIFORM | U.COPY_DST),
            make(data.input),
            make(data.packed),
            make(data.bias),
            make(ROWS * n * 4, U.STORAGE | U.COPY_SRC | U.COPY_DST),
            make(data.scale),
          ];
          const mapped = make(ROWS * n * 4, U.MAP_READ | U.COPY_DST);
          /** One dispatch of `code` over `m` rows in y tiles of `tile` rows: the whole output buffer. */
          const run = async (
            code: string,
            lanes: number,
            m: number,
            tile: number,
          ): Promise<Uint32Array<ArrayBuffer>> => {
            device.queue.writeBuffer(buffers[0], 0, linearGemvParams(storage, m, n, k, group));
            device.queue.writeBuffer(buffers[4], 0, new Uint32Array(ROWS * n).fill(UNWRITTEN));
            const pipeline = await device.createComputePipelineAsync({
              layout: "auto",
              compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
            });
            const bindings = device.createBindGroup({
              layout: pipeline.getBindGroupLayout(0),
              entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
            });
            const encoder = device.createCommandEncoder(),
              pass = encoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindings);
            pass.dispatchWorkgroups(Math.ceil(n / (128 / lanes)), Math.ceil(m / tile));
            pass.end();
            encoder.copyBufferToBuffer(buffers[4], 0, mapped, 0, ROWS * n * 4);
            device.queue.submit([encoder.finish()]);
            await mapped.mapAsync(MAP_MODE.READ);
            const out = new Uint32Array(mapped.getMappedRange().slice(0));
            mapped.unmap();
            return out;
          };
          for (const lanes of LANES) {
            const plain = linearGemvParallelWgsl(storage, group, lanes);
            // The parallel GEMV's outputs by M, taken once per lane count (M = 1 is the decode row)
            const parallel = new Map<number, Uint32Array<ArrayBuffer>>();
            const parallelAt = async (m: number): Promise<Uint32Array<ArrayBuffer>> => {
              const known = parallel.get(m);
              if (known !== undefined) return known;
              const out = await run(plain, lanes, m, 1);
              parallel.set(m, out);
              return out;
            };
            for (const { rows } of blocksOf(storage).filter((block) => block.lanes === lanes)) {
              const generated = linearGemvParallelRowsWgsl(storage, group, lanes, rows);
              const blocked = options.mutate === undefined ? generated : options.mutate(generated);
              if (options.mutate !== undefined && blocked === generated) {
                throw new Error(
                  `gemv-parallel-rows: the mutation left the kernel of ${storage} L${lanes} R${rows} unchanged`,
                );
              }
              for (const m of rowCountsOf(rows)) {
                throwIfAborted(options);
                const decode = await parallelAt(1);
                const expected = await parallelAt(m);
                const actual = await run(blocked, lanes, m, rows);
                cases.push({
                  label: `${storage} K${k} L${lanes} R${rows} M${m}`,
                  criteria: [
                    compareWords(
                      "row-blocked vs parallel (u32)",
                      actual.slice(0, m * n),
                      expected.slice(0, m * n),
                    ),
                    compareWords(
                      "row 0 vs the parallel GEMV's row at M = 1 (decode / verify, u32)",
                      actual.slice(0, n),
                      decode.slice(0, n),
                    ),
                    compareFill(
                      "row-blocked: rows past M left unwritten",
                      actual.slice(m * n),
                      UNWRITTEN,
                    ),
                    compareFill(
                      "parallel: rows past M left unwritten",
                      expected.slice(m * n),
                      UNWRITTEN,
                    ),
                    withinBand(
                      "row-blocked within the GEMM band of FP64",
                      actual.slice(0, m * n),
                      data.reference,
                    ),
                  ],
                });
                options.onProgress?.({ part: PART, done: cases.length, total });
              }
            }
          }
        } finally {
          await device.queue.onSubmittedWorkDone();
          for (const buffer of owned) buffer.destroy();
        }
      });
    }
  }
  return {
    id: PART,
    title: "Parallel GEMV: row blocks vs one row per workgroup",
    cases,
    requirements: [],
  };
};
