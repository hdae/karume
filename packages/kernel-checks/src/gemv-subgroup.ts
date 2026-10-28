/**
 * Check 1 — the parallel GEMV's subgroup32 reduction (`linearGemvReduce: "parallel-subgroup32"`,
 * ADR 0101) against the plain parallel reduction: the same rounding and the same addition order, so
 * every output word MUST be identical. Not a speed measurement.
 *
 * Per case (storage, K, lanes, M): the subgroup output equals the parallel output word for word; both
 * leave the rows past M untouched; row 0 is the same at every M (decode and verify agree); and every
 * output is within the GEMM band of an FP64 reference.
 */

import type { GpuContext } from "../../runtime/src/gpu/device.ts";
import { BUFFER_USAGE as U, MAP_MODE } from "../../runtime/src/gpu/webgpu-constants.ts";
import {
  linearGemvParallelWgsl,
  linearGemvParams,
  linearGemvSubgroupWgsl,
  linearGemvUnit,
} from "../../runtime/src/kernels/linear-gemv.ts";
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

type Storage = "i2" | "i4" | "i8";

/** The word the output buffer is filled with before each dispatch (a quiet NaN). */
const UNWRITTEN = 0x7fc00000;
/** Output rows of the buffers (the largest M). */
const ROWS = 8;
/** Output columns. */
const COLUMNS = 36;
const LANES = [2, 4, 8, 16, 32] as const;
const MS = [1, 4, 8] as const;

/** The (K, group) shapes per storage: one unit, five units, 4096, plus i4's large groups. */
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

const STORAGES = ["i2", "i4", "i8"] as const;

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

/**
 * The cases {@link checkGemvSubgroup} returns: i2 / i8 3 shapes and i4 6 shapes, each × 5 lane
 * counts × M = 1 / 4 / 8.
 */
export const GEMV_SUBGROUP_CASES = { "gemv-subgroup32": 180 } as const satisfies ExpectedCases;

/** The number of cases {@link checkGemvSubgroup} runs. */
const caseCount = (): number =>
  STORAGES.reduce((total, storage) => total + shapesOf(storage).length, 0) *
  LANES.length *
  MS.length;

/**
 * Runs the check on a device with the subgroup features (`subgroup32Available`). The same check runs
 * in the node GPU test and on the playground's `/kernel-check` page.
 */
export const checkGemvSubgroup = async (
  gpu: GpuContext,
  options: CheckOptions = {},
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
          for (const lanes of LANES) {
            let first: Uint32Array<ArrayBuffer> | undefined;
            for (const m of MS) {
              throwIfAborted(options);
              const outputs: Uint32Array<ArrayBuffer>[] = [];
              const untouched: Criterion[] = [];
              for (const [shaderName, shader] of [
                ["parallel", linearGemvParallelWgsl],
                ["subgroup32", linearGemvSubgroupWgsl],
              ] as const) {
                device.queue.writeBuffer(buffers[0], 0, linearGemvParams(storage, m, n, k, group));
                device.queue.writeBuffer(buffers[4], 0, new Uint32Array(ROWS * n).fill(UNWRITTEN));
                const pipeline = await device.createComputePipelineAsync({
                  layout: "auto",
                  compute: {
                    module: device.createShaderModule({ code: shader(storage, group, lanes) }),
                    entryPoint: "main",
                  },
                });
                const bindings = device.createBindGroup({
                  layout: pipeline.getBindGroupLayout(0),
                  entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
                });
                const encoder = device.createCommandEncoder(),
                  pass = encoder.beginComputePass();
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, bindings);
                pass.dispatchWorkgroups(Math.ceil(n / (128 / lanes)), m);
                pass.end();
                encoder.copyBufferToBuffer(buffers[4], 0, mapped, 0, ROWS * n * 4);
                device.queue.submit([encoder.finish()]);
                await mapped.mapAsync(MAP_MODE.READ);
                const out = new Uint32Array(mapped.getMappedRange().slice(0));
                mapped.unmap();
                untouched.push(
                  compareFill(
                    `${shaderName}: rows past M left unwritten`,
                    out.slice(m * n),
                    UNWRITTEN,
                  ),
                );
                outputs.push(out.slice(0, m * n));
              }
              first ??= outputs[1].slice(0, n);
              cases.push({
                label: `${storage} K${k} L${lanes} M${m}`,
                criteria: [
                  compareWords("subgroup32 vs parallel (u32)", outputs[1], outputs[0]),
                  compareWords(
                    "row 0 vs row 0 at M = 1 (decode / verify, u32)",
                    outputs[1].slice(0, n),
                    first,
                  ),
                  ...untouched,
                  withinBand("subgroup32 within the GEMM band of FP64", outputs[1], data.reference),
                ],
              });
              options.onProgress?.({ part: "gemv-subgroup32", done: cases.length, total });
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
    id: "gemv-subgroup32",
    title: "Parallel GEMV: subgroup32 vs parallel",
    cases,
    requirements: [],
  };
};
