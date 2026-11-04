/**
 * The integer GEMV (`linear_gemv_integer`, src/kernels/linear-gemv-integer.ts): the mirror-class (E)
 * check against its CPU reference, and a single-kernel timing against the parallel GEMV it replaces.
 *
 * Check: per case (weight form, K, variant, M, with / without the SRQ epilogue) every output word
 * equals the reference's, and the rows past M stay unwritten. The fixtures include saturated inputs
 * at the largest K of each storage (every activation code −128 against the largest stored code),
 * where the lane sums sit closest to 2^24 and each step's `255 · s` is largest — the two places the
 * exactness rests on. The i8 one is offset at one element so its sum is past 2^24 and 3 mod 4: the
 * integer-to-f32 conversion then has to round a tie, and round-to-nearest-even rounds it up (a
 * truncating conversion would round it down).
 *
 * Timing ({@link benchIntegerGemv}): not a check. It times one dispatch of each kernel on the real
 * E2B QAT shapes, cycling over enough weight copies that the weights come from memory, not a cache.
 */

import type { GpuContext } from "../../runtime/src/gpu/device.ts";
import { BUFFER_USAGE as U, MAP_MODE } from "../../runtime/src/gpu/webgpu-constants.ts";
import {
  type IntegerGemvLanes,
  integerGemvParams,
  type IntegerGemvRows,
  type IntegerGemvVariant,
  type IntegerGemvWeight,
  integerGemvWgsl,
  integerGemvWorkgroups,
} from "../../runtime/src/kernels/linear-gemv-integer.ts";
import {
  linearGemvPackedEligible,
  linearGemvParallelEligible,
  linearGemvStaticQuantizePackedParams,
  linearGemvStaticQuantizePackedWgsl,
  linearGemvStaticQuantizeParams,
  linearGemvStaticQuantizeWgsl,
} from "../../runtime/src/kernels/linear-gemv.ts";
import { referenceIntegerGemv } from "../../runtime/src/reference/linear-gemv-integer.ts";
import {
  type CheckCase,
  type CheckOptions,
  type CheckPart,
  compareFill,
  compareWords,
  type ExpectedCases,
  scopedGpuWork,
  throwIfAborted,
} from "./result.ts";

const UNWRITTEN = 0x7fc00000;
const ROWS = 8;
/** Not a multiple of any workgroup's channel count, so the edge workgroup is exercised. */
const COLUMNS = 37;
const MS = [1, 8] as const;
const X_SCALE = Math.fround(0.0137);
const OUTPUT_SCALE = Math.fround(0.021);

/** A (lanes, rows) pair; the weight form is joined per run. */
type Geometry = { readonly lanes: IntegerGemvLanes; readonly rows: IntegerGemvRows };

/** Every (lanes, rows) the generator accepts. */
const GEOMETRIES: readonly Geometry[] = ([4, 8, 16, 32] as const).flatMap((lanes) =>
  ([1, 2, 4, 8] as const).filter((rows) => rows <= lanes).map((rows) => ({ lanes, rows })),
);
/** The corners and the production-like middle, for the forms where every geometry would be slow. */
const CORNERS: readonly Geometry[] = [
  { lanes: 4, rows: 1 },
  { lanes: 8, rows: 4 },
  { lanes: 16, rows: 4 },
  { lanes: 32, rows: 1 },
  { lanes: 32, rows: 8 },
];
const EXTREMES: readonly Geometry[] = [
  { lanes: 4, rows: 1 },
  { lanes: 32, rows: 8 },
];

type Fixture = "mixed" | "saturated";

const i4 = (groupSize: number, groups: number): IntegerGemvWeight => ({
  storage: "i4",
  groupSize,
  groups,
});

/**
 * The runs: small K with every geometry, the widths in use with the corners, and each storage's
 * largest K saturated at the extremes of the geometry table.
 */
const RUNS: readonly {
  weight: IntegerGemvWeight;
  k: number;
  kind: Fixture;
  geometries: readonly Geometry[];
}[] = [
  ...[16, 80, 1536, 12288].map((k) => ({
    weight: { storage: "i2" } as const,
    k,
    kind: "mixed" as const,
    geometries: GEOMETRIES,
  })),
  { weight: { storage: "i2" }, k: 12288, kind: "saturated", geometries: EXTREMES },
  { weight: i4(16, 1), k: 16, kind: "mixed", geometries: GEOMETRIES },
  { weight: i4(16, 3), k: 48, kind: "mixed", geometries: GEOMETRIES },
  { weight: i4(32, 2), k: 64, kind: "mixed", geometries: GEOMETRIES },
  { weight: i4(512, 3), k: 1536, kind: "mixed", geometries: CORNERS },
  { weight: i4(2048, 3), k: 6144, kind: "mixed", geometries: CORNERS },
  { weight: i4(4096, 1), k: 4096, kind: "mixed", geometries: CORNERS },
  { weight: i4(2048, 3), k: 6144, kind: "saturated", geometries: EXTREMES },
  { weight: i4(4096, 1), k: 4096, kind: "saturated", geometries: EXTREMES },
  ...[16, 80, 1536].map((k) => ({
    weight: { storage: "i8" } as const,
    k,
    kind: "mixed" as const,
    geometries: GEOMETRIES,
  })),
  { weight: { storage: "i8" }, k: 1536, kind: "saturated", geometries: EXTREMES },
];

export const GEMV_INTEGER_CASES = {
  "gemv-integer": RUNS.reduce((total, run) => total + run.geometries.length, 0) * MS.length * 2,
} as const satisfies ExpectedCases;

const weightLabel = (weight: IntegerGemvWeight): string =>
  weight.storage === "i4" ? `i4 g${weight.groupSize}x${weight.groups}` : weight.storage;

const packCodes = (codes: Int8Array<ArrayBuffer>): Uint32Array<ArrayBuffer> => {
  const words = new Uint32Array(codes.length / 4);
  for (let i = 0; i < codes.length; i++) words[i >> 2] |= (codes[i] & 0xff) << ((i & 3) * 8);
  return words;
};

/** Bits per stored element, and the largest stored code (the saturated fixture's). */
const STORED: Readonly<Record<IntegerGemvWeight["storage"], { bits: number; top: number }>> = {
  i2: { bits: 2, top: 3 },
  i4: { bits: 4, top: 15 },
  i8: { bits: 8, top: 0x80 },
};

/**
 * Weights as bound: `bits` per element, low bits first (i2 stored = value + 2, i4 stored = value +
 * 8, i8 two's complement). The saturated i8 fixture stores −127 and reads −125 at element 0 of each
 * row: 1535 · 16384 + 15875 = 25165315, past 2^24 and 3 mod 4.
 */
const fixtureOf = (
  weight: IntegerGemvWeight,
  n: number,
  k: number,
  kind: Fixture,
): {
  codes: Uint32Array<ArrayBuffer>;
  weights: Uint32Array<ArrayBuffer>;
  wscale: Float32Array<ArrayBuffer>;
  bias: Float32Array<ArrayBuffer>;
} => {
  const { bits, top } = STORED[weight.storage];
  const oddI8 = kind === "saturated" && weight.storage === "i8";
  const activation = Int8Array.from({ length: ROWS * k }, (_, i) =>
    kind === "saturated"
      ? oddI8 && i % k === 0
        ? -125
        : -128
      : ((Math.imul(i + 7, 0x2545f491) >>> 13) % 256) - 128,
  );
  const perWord = 32 / bits;
  const mask = (1 << bits) - 1;
  const weights = new Uint32Array((n * k) / perWord);
  for (let i = 0; i < n * k; i++) {
    const stored =
      kind === "saturated"
        ? oddI8 && i % k === 0
          ? 0x81
          : top
        : (Math.imul(i + 1, 0x9e3779b9) >>> 11) & mask;
    weights[Math.floor(i / perWord)] |= stored << ((i % perWord) * bits);
  }
  const groups = weight.storage === "i4" ? weight.groups : 1;
  return {
    codes: packCodes(activation),
    weights,
    wscale: Float32Array.from({ length: n * groups }, (_, i) => ((i % 17) + 1) * 0.00031),
    bias: Float32Array.from({ length: n }, (_, i) => ((i % 7) - 3) * 0.11),
  };
};

/** Runs the mirror check. Needs no optional feature (`unpack4xI8` / `unpack4x8unorm` are core). */
export const checkIntegerGemv = async (
  gpu: GpuContext,
  options: CheckOptions = {},
): Promise<CheckPart> => {
  const device = gpu.device;
  const cases: CheckCase[] = [];
  const total = GEMV_INTEGER_CASES["gemv-integer"];
  const n = COLUMNS;
  const pipelines = new Map<string, GPUComputePipeline>();
  const pipelineOf = async (
    variant: IntegerGemvVariant,
    quantize: boolean,
  ): Promise<GPUComputePipeline> => {
    const key = `${weightLabel(variant.weight)}:${variant.lanes}:${variant.rows}:${quantize}`;
    const cached = pipelines.get(key);
    if (cached !== undefined) return cached;
    const pipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: integerGemvWgsl(variant, quantize) }),
        entryPoint: "main",
      },
    });
    pipelines.set(key, pipeline);
    return pipeline;
  };
  for (const { weight, k, kind, geometries } of RUNS) {
    const data = fixtureOf(weight, n, k, kind);
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
    const label = `${weightLabel(weight)} K${k} ${kind}`;
    await scopedGpuWork(gpu, `integer ${label}`, async () => {
      try {
        const params = make(268 * 4, U.UNIFORM | U.COPY_DST);
        const buffers = [
          params,
          make(data.codes),
          make(data.weights),
          make(data.bias),
          make(ROWS * n * 4, U.STORAGE | U.COPY_SRC | U.COPY_DST),
          make(data.wscale),
        ];
        const mapped = make(ROWS * n * 4, U.MAP_READ | U.COPY_DST);
        for (const quantize of [false, true]) {
          for (const m of MS) {
            const expected = referenceIntegerGemv({
              weight,
              m,
              n,
              k,
              codes: data.codes.slice(0, (m * k) / 4),
              xScale: X_SCALE,
              weights: data.weights,
              wscale: data.wscale,
              bias: data.bias,
              ...(quantize ? { outputScale: OUTPUT_SCALE } : {}),
            });
            for (const geometry of geometries) {
              throwIfAborted(options);
              const variant: IntegerGemvVariant = { weight, ...geometry };
              const pipeline = await pipelineOf(variant, quantize);
              device.queue.writeBuffer(
                params,
                0,
                integerGemvParams(variant, m, n, k, X_SCALE, quantize ? OUTPUT_SCALE : undefined),
              );
              device.queue.writeBuffer(buffers[4], 0, new Uint32Array(ROWS * n).fill(UNWRITTEN));
              const bindings = device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
              });
              const encoder = device.createCommandEncoder();
              const pass = encoder.beginComputePass();
              pass.setPipeline(pipeline);
              pass.setBindGroup(0, bindings);
              pass.dispatchWorkgroups(integerGemvWorkgroups(n, variant), m);
              pass.end();
              encoder.copyBufferToBuffer(buffers[4], 0, mapped, 0, ROWS * n * 4);
              device.queue.submit([encoder.finish()]);
              await mapped.mapAsync(MAP_MODE.READ);
              const out = new Uint32Array(mapped.getMappedRange().slice(0));
              mapped.unmap();
              cases.push({
                label: `${label} L${variant.lanes}R${variant.rows} M${m}${quantize ? " SRQ" : ""}`,
                criteria: [
                  compareWords(
                    "integer GEMV vs CPU reference (u32)",
                    out.slice(0, m * n),
                    expected,
                  ),
                  compareFill("rows past M left unwritten", out.slice(m * n), UNWRITTEN),
                ],
              });
              options.onProgress?.({ part: "gemv-integer", done: cases.length, total });
            }
          }
        }
      } finally {
        await device.queue.onSubmittedWorkDone();
        for (const buffer of owned) buffer.destroy();
      }
    });
  }
  return {
    id: "gemv-integer",
    title: "Integer GEMV: against the CPU reference",
    cases,
    requirements: [],
  };
};

export type IntegerGemvBenchRow = {
  readonly shape: string;
  readonly kernel: string;
  /** Median over the rounds of (pass time / dispatches), µs. */
  readonly microseconds: number;
  /** The fastest round, µs (the least disturbed by other work on the machine). */
  readonly minimumMicroseconds: number;
  /** Weight bytes per dispatch / the median time. */
  readonly gigabytesPerSecond: number;
};

type BenchShape = {
  readonly name: string;
  readonly weight: IntegerGemvWeight;
  readonly n: number;
  readonly k: number;
};

/** The E2B QAT decode linears (per token: count in the name). */
export const INTEGER_GEMV_BENCH_SHAPES: readonly BenchShape[] = [
  { name: "i2 gate/up 12288x1536 (x40)", weight: { storage: "i2" }, n: 12288, k: 1536 },
  { name: "i2 down 1536x12288 (x20)", weight: { storage: "i2" }, n: 1536, k: 12288 },
  { name: "i4 q 2048x1536 (x28)", weight: i4(512, 3), n: 2048, k: 1536 },
  { name: "i4 q 4096x1536 (x7)", weight: i4(512, 3), n: 4096, k: 1536 },
  { name: "i4 k/v 256x1536 (x24)", weight: i4(512, 3), n: 256, k: 1536 },
  { name: "i4 k/v 512x1536 (x6)", weight: i4(512, 3), n: 512, k: 1536 },
  { name: "i4 gate/up 6144x1536 (x30)", weight: i4(512, 3), n: 6144, k: 1536 },
  { name: "i4 o 1536x2048 (x28)", weight: i4(2048, 1), n: 1536, k: 2048 },
  { name: "i4 o 1536x4096 (x7)", weight: i4(4096, 1), n: 1536, k: 4096 },
  { name: "i4 down 1536x6144 (x15)", weight: i4(2048, 3), n: 1536, k: 6144 },
  { name: "i8 PLE gate 256x1536 (x35)", weight: { storage: "i8" }, n: 256, k: 1536 },
  { name: "i8 PLE proj 1536x256 (x35)", weight: { storage: "i8" }, n: 1536, k: 256 },
];

/**
 * Times the parallel GEMV the shape takes today (its eligibility functions decide lanes and the
 * packed form) and each integer geometry, on the E2B QAT shapes, with the SRQ epilogue as in
 * production. `device` needs `timestamp-query`. Each round is one compute pass of `dispatches`
 * dispatches cycling over weight copies of 40 MB or more in total.
 *
 * @param shapes A filter on the shape names (substring); all when absent.
 */
export const benchIntegerGemv = async (
  device: GPUDevice,
  geometries: readonly Geometry[] = GEOMETRIES,
  {
    dispatches = 40,
    rounds = 15,
    shapes,
  }: { dispatches?: number; rounds?: number; shapes?: string } = {},
): Promise<readonly IntegerGemvBenchRow[]> => {
  if (!device.features.has("timestamp-query")) {
    throw new Error("benchIntegerGemv needs a device with timestamp-query");
  }
  const rows: IntegerGemvBenchRow[] = [];
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
  try {
    const resolve = make(16, U.QUERY_RESOLVE | U.COPY_SRC);
    const readback = make(16, U.MAP_READ | U.COPY_DST);
    for (const shape of INTEGER_GEMV_BENCH_SHAPES) {
      if (shapes !== undefined && !shape.name.includes(shapes)) continue;
      const { weight, n, k } = shape;
      const weightBytes = (n * k * STORED[weight.storage].bits) / 8;
      const copies = Math.max(2, Math.ceil(40e6 / weightBytes));
      const data = fixtureOf(weight, n, k, "mixed");
      const codes = make(data.codes.slice(0, k / 4));
      const values = make(
        Float32Array.from({ length: k }, (_, i) => {
          const code = ((data.codes[i >> 2] >>> ((i & 3) * 8)) << 24) >> 24;
          return Math.fround(code * X_SCALE);
        }),
      );
      const weights = Array.from({ length: copies }, () => make(data.weights));
      const bias = make(new Float32Array(n));
      const wscale = make(data.wscale);
      const out = make(n * 4, U.STORAGE);
      const { storage } = weight;
      const group = weight.storage === "i4" ? weight.groupSize : undefined;
      const lanes = linearGemvParallelEligible(storage, 1, n, k, group);
      const packed = linearGemvPackedEligible(storage, 1, n, k, group) !== undefined;
      const parallel =
        lanes === undefined
          ? []
          : [
              {
                name: `parallel l${lanes}${packed ? " packed" : ""}`,
                code: packed
                  ? linearGemvStaticQuantizePackedWgsl(storage, group, lanes)
                  : linearGemvStaticQuantizeWgsl(storage, group, lanes),
                params: packed
                  ? linearGemvStaticQuantizePackedParams(
                      storage,
                      1,
                      n,
                      k,
                      X_SCALE,
                      OUTPUT_SCALE,
                      group,
                    )
                  : linearGemvStaticQuantizeParams(storage, 1, n, k, OUTPUT_SCALE, group),
                x: packed ? codes : values,
                grid: Math.ceil(n / (128 / lanes)),
              },
            ];
      const kernels = [
        ...parallel,
        ...geometries.map((geometry) => {
          const variant: IntegerGemvVariant = { weight, ...geometry };
          return {
            name: `integer L${geometry.lanes}R${geometry.rows}`,
            code: integerGemvWgsl(variant, true),
            params: integerGemvParams(variant, 1, n, k, X_SCALE, OUTPUT_SCALE),
            x: codes,
            grid: integerGemvWorkgroups(n, variant),
          };
        }),
      ];
      // Kernels take turns within each round, so a slow phase of the machine lands on all of them.
      const prepared = [];
      for (const kernel of kernels) {
        const pipeline = await device.createComputePipelineAsync({
          layout: "auto",
          compute: { module: device.createShaderModule({ code: kernel.code }), entryPoint: "main" },
        });
        const params = make(kernel.params, U.UNIFORM | U.COPY_DST);
        const groups = weights.map((buffer) =>
          device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [params, kernel.x, buffer, bias, out, wscale].map((resource, binding) => ({
              binding,
              resource: { buffer: resource },
            })),
          }),
        );
        prepared.push({ kernel, pipeline, groups, samples: [] as number[] });
      }
      for (let round = 0; round <= rounds; round++) {
        for (const { kernel, pipeline, groups, samples } of prepared) {
          const encoder = device.createCommandEncoder();
          const pass = encoder.beginComputePass({
            timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
          });
          pass.setPipeline(pipeline);
          for (let d = 0; d < dispatches; d++) {
            pass.setBindGroup(0, groups[d % copies]);
            pass.dispatchWorkgroups(kernel.grid, 1);
          }
          pass.end();
          encoder.resolveQuerySet(querySet, 0, 2, resolve, 0);
          encoder.copyBufferToBuffer(resolve, 0, readback, 0, 16);
          device.queue.submit([encoder.finish()]);
          await readback.mapAsync(MAP_MODE.READ);
          const [begin, end] = new BigUint64Array(readback.getMappedRange().slice(0));
          readback.unmap();
          // Round 0 warms the pipelines and the caches; it is not kept.
          if (round > 0) samples.push(Number(end - begin) / 1000 / dispatches);
        }
      }
      for (const { kernel, samples } of prepared) {
        samples.sort((a, b) => a - b);
        const microseconds = samples[Math.floor(samples.length / 2)];
        rows.push({
          shape: shape.name,
          kernel: kernel.name,
          microseconds,
          minimumMicroseconds: samples[0],
          gigabytesPerSecond: weightBytes / (microseconds * 1000),
        });
      }
    }
  } finally {
    await device.queue.onSubmittedWorkDone();
    querySet.destroy();
    for (const buffer of owned) buffer.destroy();
  }
  return rows;
};
