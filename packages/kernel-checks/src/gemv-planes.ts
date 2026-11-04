/**
 * The byte-plane GEMV (`linear_gemv_planes`, src/kernels/linear-gemv-planes.ts): the contract class R
 * check of ADR 0110, and a single-kernel timing against the sequential GEMV the QAT lm_head takes
 * without it.
 *
 * Check, per case (K, column count, fixture, variant, M) — points 1 to 3 of ADR 0110 §3:
 *
 * 1. The f64 band: every output within the reference op band (`GEMM_TOLERANCE`) of an f64
 *    evaluation of `(Σ_k q · x) · scale + bias` — the claim of class R.
 * 2. The A/B band, where the sequential GEMV runs (M = 1, K a multiple of its 64-element word):
 *    every output within twice that band of the sequential GEMV's output on the same input (the
 *    theoretical bound of ADR 0110 §4 — both sides within the f64 band).
 * 3. Structure: a second dispatch of the same input writes the same words; nothing past the M × n
 *    outputs is written (rows past M, and columns past n, which would land past the last row); a
 *    NaN at one element of a row reaches every output of that row.
 *
 * The fixtures: random codes with activations in [−2, 2), and at the largest K the saturated
 * fixture — every code 0 (value −2) against activations of one sign, so no term cancels and the
 * sums, with the rounding of each addition, are as large as K allows. The column counts are
 * multiples of no workgroup's column count, so the edge workgroup reads the last channel's words
 * for its extra columns and must not store them.
 *
 * Point 4 (a fault injection that turns point 1 red) is not part of the check: the node GPU test
 * (packages/runtime/tests/gpu_linear_gemv_planes_test.ts) runs this check on mutated WGSL through
 * {@link PlanesGemvCheckOptions.mutate} and asserts that the f64 band fails.
 *
 * Timing ({@link benchPlanesGemv}): not a check. It times the sequential GEMV and each variant on
 * the QAT E2B lm_head shape, reading a weight buffer larger than any GPU cache.
 */

import type { GpuContext } from "../../runtime/src/gpu/device.ts";
import { BUFFER_USAGE as U, MAP_MODE } from "../../runtime/src/gpu/webgpu-constants.ts";
import {
  defaultLinearGemvVariant,
  linearGemvParams,
  linearGemvUnit,
  linearGemvWgsl,
} from "../../runtime/src/kernels/linear-gemv.ts";
import {
  planesGemvParams,
  type PlanesGemvVariant,
  planesGemvWgsl,
  planesGemvWorkgroups,
} from "../../runtime/src/kernels/linear-gemv-planes.ts";
import { GEMM_TOLERANCE } from "../../runtime/tests/helpers/op-tolerance.ts";
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

/** The word the output buffer is filled with before a dispatch (a quiet NaN). */
const UNWRITTEN = 0x7fc00000;
/**
 * The fill before the NaN dispatch: finite (1.0), so a row that dispatch left unwritten does not
 * read as the NaN it should hold.
 */
const FINITE_FILL = 0x3f800000;
/** Activation rows of the buffers (the largest M). */
const ROWS = 8;
const MS = [1, 8] as const;
/** Multiples of no workgroup's column count ((128 / lanes) · rows = 4..128). */
const COLUMNS = [37, 1029] as const;
/** Words past the ROWS × n outputs: one workgroup's columns, the most a kernel could write past n. */
const TAIL = 128;
/** Elements per weight word (16 i2 codes per `u32`). */
const WORD = 16;

/** Every (lanes, rows) the generator accepts. */
const VARIANTS: readonly PlanesGemvVariant[] = ([4, 8, 16, 32] as const).flatMap((lanes) =>
  ([1, 2, 4, 8] as const).filter((rows) => rows <= lanes).map((rows) => ({ lanes, rows })),
);

type Fixture = "mixed" | "saturated";

/**
 * K = 16 (one word: every lane past the first idle), 80 (five words, a multiple of no lane count)
 * and 1536 (the lm_head's K), at both column counts; the saturated fixture at K = 1536.
 */
const RUNS: readonly { readonly k: number; readonly n: number; readonly kind: Fixture }[] = [
  ...[16, 80, 1536].flatMap((k) => COLUMNS.map((n) => ({ k, n, kind: "mixed" as const }))),
  ...COLUMNS.map((n) => ({ k: 1536, n, kind: "saturated" as const })),
];

export const GEMV_PLANES_CASES = {
  "gemv-planes": RUNS.length * VARIANTS.length * MS.length,
} as const satisfies ExpectedCases;

/** The name of point 1's criterion (the node test's fault injection reads it). */
export const GEMV_PLANES_F64_BAND = "planes within the GEMM band of f64";

/** A 32-bit mix of `i` (lowbias32): every bit of the result depends on every bit of `i`. */
const hash = (i: number): number => {
  let h = i >>> 0;
  h ^= h >>> 16;
  h = Math.imul(h, 0x7feb352d);
  h ^= h >>> 15;
  h = Math.imul(h, 0x846ca68b);
  h ^= h >>> 16;
  return h >>> 0;
};

/** A uniform value in [0, 1) drawn from `i`. */
const uniform = (i: number): number => hash(i) / 2 ** 32;

/** The signed weight value of element `j` of channel `column` (stored code − 2). */
const weightValue = (
  words: Uint32Array<ArrayBuffer>,
  k: number,
  column: number,
  j: number,
): number => ((words[(column * k + j) >>> 4] >>> ((j & 15) * 2)) & 3) - 2;

type FixtureData = {
  readonly x: Float32Array<ArrayBuffer>;
  readonly words: Uint32Array<ArrayBuffer>;
  readonly wscale: Float32Array<ArrayBuffer>;
  readonly bias: Float32Array<ArrayBuffer>;
  /** `(Σ_k q · x) · scale + bias` in f64, ROWS × n. */
  readonly reference: Float64Array<ArrayBuffer>;
};

/**
 * Weights as bound (element `j` of channel `c` at bits 2·(j mod 16) of word (c·k + j) / 16), one
 * scale per channel, and ROWS rows of activations. The f64 evaluation reads the same f32 values.
 */
const fixtureOf = (n: number, k: number, kind: Fixture): FixtureData => {
  const words = new Uint32Array((n * k) / WORD);
  // Saturated: every stored code 0 (value −2), i.e. every word 0.
  if (kind === "mixed") for (let i = 0; i < words.length; i++) words[i] = hash(i + 1);
  const x = Float32Array.from({ length: ROWS * k }, (_, i) =>
    kind === "saturated" ? 0.5 + 1.5 * uniform(i + 7) : 4 * uniform(i + 7) - 2,
  );
  const wscale = Float32Array.from({ length: n }, (_, c) => ((c % 17) + 1) * 0.0031);
  const bias = Float32Array.from({ length: n }, (_, c) => ((c % 7) - 3) * 0.11);
  const reference = new Float64Array(ROWS * n);
  for (let row = 0; row < ROWS; row++) {
    for (let column = 0; column < n; column++) {
      let sum = 0;
      for (let j = 0; j < k; j++) sum += weightValue(words, k, column, j) * x[row * k + j];
      reference[row * n + column] = sum * wscale[column] + bias[column];
    }
  }
  return { x, words, wscale, bias, reference };
};

/** The reference op band around an f64 value (class R: the GEMM row of op-tolerance.ts). */
const band = (reference: number): number =>
  GEMM_TOLERANCE.atol + GEMM_TOLERANCE.rtol * Math.abs(reference);

/**
 * Every output finite and within `factor` × the band of `reference` around `against` (the f64
 * values themselves for point 1, the sequential GEMV's outputs for point 2).
 */
const withinBand = (
  name: string,
  actual: Uint32Array<ArrayBuffer>,
  against: ArrayLike<number>,
  reference: Float64Array<ArrayBuffer>,
  factor: number,
): Criterion => {
  const values = new Float32Array(actual.buffer, actual.byteOffset, actual.length);
  let failures = 0;
  let worst: WordFailure | undefined;
  let worstError = -1;
  for (let i = 0; i < values.length; i++) {
    const error = Math.abs(values[i] - against[i]);
    const bound = factor * band(reference[i]);
    // A non-finite output fails whatever the error (NaN compares false).
    const finite = Number.isFinite(values[i]);
    if (finite && error <= bound) continue;
    failures += 1;
    const rank = finite ? error : Number.POSITIVE_INFINITY;
    if (rank > worstError) {
      worstError = rank;
      worst = {
        index: i,
        expected: f32Word(against[i]),
        actual: actual[i],
        at: "worst",
        note: `against ${against[i]}, got ${values[i]} (bound ${bound}, f64 ${reference[i]})`,
      };
    }
  }
  return worst === undefined
    ? { name, words: values.length, failures }
    : { name, words: values.length, failures, failure: worst };
};

/** Every word a NaN. */
const allNaN = (name: string, actual: Uint32Array<ArrayBuffer>): Criterion => {
  const values = new Float32Array(actual.buffer, actual.byteOffset, actual.length);
  let failures = 0;
  let failure: WordFailure | undefined;
  for (let i = 0; i < values.length; i++) {
    if (Number.isNaN(values[i])) continue;
    failures += 1;
    failure ??= { index: i, expected: UNWRITTEN, actual: actual[i], at: "first", note: "not NaN" };
  }
  return failure === undefined
    ? { name, words: values.length, failures }
    : { name, words: values.length, failures, failure };
};

export type PlanesGemvCheckOptions = CheckOptions & {
  /**
   * Rewrites the byte-plane GEMV's WGSL before it is compiled (the node test's fault injection;
   * absent = the generator's own WGSL). A rewrite that changes nothing throws, so a stale anchor
   * cannot make an injected fault pass for the real kernel.
   */
  readonly mutate?: (wgsl: string) => string;
};

/** Runs the check. Needs no optional feature (`unpack4x8unorm` and `dot` are core). */
export const checkPlanesGemv = async (
  gpu: GpuContext,
  options: PlanesGemvCheckOptions = {},
): Promise<CheckPart> => {
  const device = gpu.device;
  const cases: CheckCase[] = [];
  const total = GEMV_PLANES_CASES["gemv-planes"];
  const pipelineOf = async (code: string): Promise<GPUComputePipeline> =>
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
    });
  const planes = new Map<PlanesGemvVariant, Promise<GPUComputePipeline>>();
  const planesPipeline = (variant: PlanesGemvVariant): Promise<GPUComputePipeline> => {
    const cached = planes.get(variant);
    if (cached !== undefined) return cached;
    const wgsl = planesGemvWgsl(variant);
    const code = options.mutate === undefined ? wgsl : options.mutate(wgsl);
    if (code === wgsl && options.mutate !== undefined) {
      throw new Error(
        `gemv-planes L${variant.lanes}R${variant.rows}: the mutation changed nothing in the WGSL`,
      );
    }
    const pipeline = pipelineOf(code);
    planes.set(variant, pipeline);
    return pipeline;
  };
  const sequentialVariant = defaultLinearGemvVariant();
  let sequentialPipeline: Promise<GPUComputePipeline> | undefined;
  for (const { k, n, kind } of RUNS) {
    const data = fixtureOf(n, k, kind);
    const owned: GPUBuffer[] = [];
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
    const label = `K${k} N${n} ${kind}`;
    await scopedGpuWork(gpu, `planes ${label}`, async () => {
      try {
        const outWords = ROWS * n + TAIL;
        const params = make(16, U.UNIFORM | U.COPY_DST);
        const x = make(data.x);
        const w = make(data.words);
        const bias = make(data.bias);
        const out = make(outWords * 4, U.STORAGE | U.COPY_SRC | U.COPY_DST);
        const wscale = make(data.wscale);
        const mapped = make(outWords * 4, U.MAP_READ | U.COPY_DST);
        /** One dispatch over `fill`ed output; returns every word of the output buffer. */
        const dispatch = async (
          pipeline: GPUComputePipeline,
          uniform: GPUBuffer,
          input: GPUBuffer,
          grid: readonly [number, number],
          fill: number,
        ): Promise<Uint32Array<ArrayBuffer>> => {
          device.queue.writeBuffer(out, 0, new Uint32Array(outWords).fill(fill));
          const bindings = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [uniform, input, w, bias, out, wscale].map((buffer, binding) => ({
              binding,
              resource: { buffer },
            })),
          });
          const encoder = device.createCommandEncoder();
          const pass = encoder.beginComputePass();
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bindings);
          pass.dispatchWorkgroups(grid[0], grid[1]);
          pass.end();
          encoder.copyBufferToBuffer(out, 0, mapped, 0, outWords * 4);
          device.queue.submit([encoder.finish()]);
          await mapped.mapAsync(MAP_MODE.READ);
          const words = new Uint32Array(mapped.getMappedRange().slice(0));
          mapped.unmap();
          return words;
        };
        // The sequential GEMV (M = 1 only) runs where its 64-element word divides K.
        let sequential: Float32Array<ArrayBuffer> | undefined;
        if (k % linearGemvUnit("i2") === 0) {
          sequentialPipeline ??= pipelineOf(linearGemvWgsl("i2", undefined, sequentialVariant));
          const words = await dispatch(
            await sequentialPipeline,
            make(linearGemvParams("i2", 1, n, k), U.UNIFORM | U.COPY_DST),
            x,
            [Math.ceil(n / sequentialVariant.cols), 1],
            UNWRITTEN,
          );
          sequential = new Float32Array(words.buffer, 0, n);
        }
        for (const m of MS) {
          device.queue.writeBuffer(params, 0, planesGemvParams(m, n, k));
          // A NaN at the last element of row m − 1 (read by one lane, then the workgroup tree).
          const poisoned = Float32Array.from(data.x);
          poisoned[(m - 1) * k + k - 1] = Number.NaN;
          const xNaN = make(poisoned);
          const reference = data.reference.subarray(0, m * n);
          for (const variant of VARIANTS) {
            throwIfAborted(options);
            const pipeline = await planesPipeline(variant);
            const grid = [planesGemvWorkgroups(n, variant), m] as const;
            const first = await dispatch(pipeline, params, x, grid, UNWRITTEN);
            const second = await dispatch(pipeline, params, x, grid, UNWRITTEN);
            const withNaN = await dispatch(pipeline, params, xNaN, grid, FINITE_FILL);
            const outputs = first.slice(0, m * n);
            cases.push({
              label: `${label} L${variant.lanes}R${variant.rows} M${m}`,
              criteria: [
                withinBand(GEMV_PLANES_F64_BAND, outputs, reference, reference, 1),
                ...(m === 1 && sequential !== undefined
                  ? [
                      withinBand(
                        "planes within twice the f64 band of the sequential GEMV",
                        outputs,
                        sequential,
                        reference,
                        2,
                      ),
                    ]
                  : []),
                compareWords(
                  "second dispatch of the same input (u32)",
                  second.slice(0, m * n),
                  outputs,
                ),
                compareFill(
                  "nothing written past the M × n outputs (rows past M, columns past n)",
                  first.slice(m * n),
                  UNWRITTEN,
                ),
                allNaN(
                  "a NaN in x reaches every output of its row",
                  withNaN.slice((m - 1) * n, m * n),
                ),
              ],
            });
            options.onProgress?.({ part: "gemv-planes", done: cases.length, total });
          }
        }
      } finally {
        await device.queue.onSubmittedWorkDone();
        for (const buffer of owned) buffer.destroy();
      }
    });
  }
  return {
    id: "gemv-planes",
    title: "Byte-plane GEMV: f64 band, A/B against the sequential GEMV, structure",
    cases,
    requirements: [],
  };
};

export type PlanesGemvBenchRow = {
  readonly kernel: string;
  /** Median over the rounds of (pass time / dispatches), µs. */
  readonly microseconds: number;
  /** The fastest round, µs (the least disturbed by other work on the machine). */
  readonly minimumMicroseconds: number;
  /** Weight bytes per dispatch / the median time. */
  readonly gigabytesPerSecond: number;
};

/** The QAT E2B lm_head (i2, once per decode token): 100 MB of weights, larger than any GPU cache. */
const BENCH_N = 262144;
const BENCH_K = 1536;

/**
 * Times the sequential GEMV (the lm_head's kernel with the option off) and each byte-plane variant
 * on the QAT E2B lm_head, M = 1. `device` needs `timestamp-query`. Each round is one compute pass of
 * `dispatches` dispatches per kernel, the kernels taking turns within the round so a slow phase of
 * the machine lands on all of them; round 0 warms up and is not kept.
 *
 * @param variants The byte-plane variants to time; by default every variant whose grid fits the
 * device's `maxComputeWorkgroupsPerDimension` (L32R1 needs 65536 workgroups). A variant passed
 * explicitly that does not fit throws.
 */
export const benchPlanesGemv = async (
  device: GPUDevice,
  variants?: readonly PlanesGemvVariant[],
  { dispatches = 10, rounds = 15 }: { dispatches?: number; rounds?: number } = {},
): Promise<readonly PlanesGemvBenchRow[]> => {
  if (!device.features.has("timestamp-query")) {
    throw new Error("benchPlanesGemv needs a device with timestamp-query");
  }
  const limit = device.limits.maxComputeWorkgroupsPerDimension;
  const fits = (variant: PlanesGemvVariant): boolean =>
    planesGemvWorkgroups(BENCH_N, variant) <= limit;
  const timed = variants ?? VARIANTS.filter(fits);
  for (const variant of timed) {
    if (!fits(variant)) {
      throw new Error(
        `benchPlanesGemv: L${variant.lanes}R${variant.rows} needs ${planesGemvWorkgroups(
          BENCH_N,
          variant,
        )} workgroups, past the device's limit ${limit}`,
      );
    }
  }
  const owned: GPUBuffer[] = [];
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
  const querySet = device.createQuerySet({ type: "timestamp", count: 2 });
  const rows: PlanesGemvBenchRow[] = [];
  try {
    const resolve = make(16, U.QUERY_RESOLVE | U.COPY_SRC);
    const readback = make(16, U.MAP_READ | U.COPY_DST);
    const weightBytes = (BENCH_N * BENCH_K) / 4;
    const words = new Uint32Array(weightBytes / 4);
    for (let i = 0; i < words.length; i++) words[i] = hash(i + 1);
    const w = make(words);
    const x = make(Float32Array.from({ length: BENCH_K }, (_, i) => 4 * uniform(i + 7) - 2));
    const bias = make(new Float32Array(BENCH_N));
    const wscale = make(Float32Array.from({ length: BENCH_N }, (_, c) => ((c % 17) + 1) * 0.0031));
    const out = make(BENCH_N * 4, U.STORAGE);
    const sequential = defaultLinearGemvVariant({ storage: "i2", n: BENCH_N, k: BENCH_K });
    const kernels = [
      {
        name: `sequential c${sequential.cols}u${sequential.unroll}`,
        code: linearGemvWgsl("i2", undefined, sequential),
        params: linearGemvParams("i2", 1, BENCH_N, BENCH_K),
        grid: Math.ceil(BENCH_N / sequential.cols),
      },
      ...timed.map((variant) => ({
        name: `planes L${variant.lanes}R${variant.rows}`,
        code: planesGemvWgsl(variant),
        params: planesGemvParams(1, BENCH_N, BENCH_K),
        grid: planesGemvWorkgroups(BENCH_N, variant),
      })),
    ];
    const prepared = [];
    for (const kernel of kernels) {
      const pipeline = await device.createComputePipelineAsync({
        layout: "auto",
        compute: { module: device.createShaderModule({ code: kernel.code }), entryPoint: "main" },
      });
      const params = make(kernel.params, U.UNIFORM | U.COPY_DST);
      const group = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [params, x, w, bias, out, wscale].map((buffer, binding) => ({
          binding,
          resource: { buffer },
        })),
      });
      prepared.push({ kernel, pipeline, group, samples: [] as number[] });
    }
    for (let round = 0; round <= rounds; round++) {
      for (const { kernel, pipeline, group, samples } of prepared) {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass({
          timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
        });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, group);
        for (let d = 0; d < dispatches; d++) pass.dispatchWorkgroups(kernel.grid, 1);
        pass.end();
        encoder.resolveQuerySet(querySet, 0, 2, resolve, 0);
        encoder.copyBufferToBuffer(resolve, 0, readback, 0, 16);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(MAP_MODE.READ);
        const [begin, end] = new BigUint64Array(readback.getMappedRange().slice(0));
        readback.unmap();
        if (round > 0) samples.push(Number(end - begin) / 1000 / dispatches);
      }
    }
    for (const { kernel, samples } of prepared) {
      samples.sort((a, b) => a - b);
      const microseconds = samples[Math.floor(samples.length / 2)];
      rows.push({
        kernel: kernel.name,
        microseconds,
        minimumMicroseconds: samples[0],
        gigabytesPerSecond: weightBytes / (microseconds * 1000),
      });
    }
  } finally {
    await device.queue.onSubmittedWorkDone();
    querySet.destroy();
    for (const buffer of owned) buffer.destroy();
  }
  return rows;
};
