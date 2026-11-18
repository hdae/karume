/**
 * The parallel GEMV on float-stored weights (f32 / f16 / bf16) on a real GPU: the kernel against an
 * f64 reference and against the sequential GEMV, and through a Session whose geometry profile names
 * a float shape.
 *
 * The parallel form is of the reorder class (another summation order, held to the band of `linear`),
 * so its checks are the four points of that class:
 * 1. the band against an f64 reference of the stored values;
 * 2. the band against the reference path's kernel (the sequential GEMV) on the same inputs;
 * 3. exact structure — M = 1 / 4 / 8 of one shape give the same words for the same row, the staged
 *    form gives the plain form's words, outputs past M stay unwritten, edge columns and spare lanes
 *    write nothing;
 * 4. faults that turn them red, recorded here (2026-11-18, RTX 3080 Ti, Dawn): with the last level
 *    of the partial-sum tree left out (`width > 0u` → `width > 1u` in `parallelWgsl`, so half of every
 *    column's words are dropped) point 1 fails as `bf16 K40 L2 M1 @0: 1.6825… / 3.0501…` and the
 *    Session case's point 2 as `bf16 @0: 3.7438… / 7.5835…`; with the staged form staging the next
 *    chunk's quads (`let v = chunk * … + q` → `+ q + 1u`) point 3 fails as `bf16 K8 L2 M1: staged`.
 *
 * The worst share of the band any output of the kernel case used on that device: 0.195 against the
 * f64 reference, 0.172 against the sequential GEMV (the test prints both).
 */
import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import type { CodecName } from "../src/format/container/codecs.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { DEFAULT_GEOMETRY_PROFILE, type GeometryProfile } from "../src/kernels/geometry-profile.ts";
import {
  defaultLinearGemvVariant,
  type LinearGemvParallelLanes,
  linearGemvParallelParams,
  linearGemvParallelStagedWgsl,
  linearGemvParallelWgsl,
  linearGemvParams,
  linearGemvUnit,
  linearGemvWgsl,
} from "../src/kernels/linear-gemv.ts";
import { GEMM_TOLERANCE } from "../src/reference/op-tolerance.ts";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import { quantizeBf16 } from "./helpers/bf16.ts";
import type { TensorInput } from "./helpers/container-write.ts";
import { quantizeF16 } from "./helpers/f16.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import {
  type DeclarationJson,
  f32Bytes,
  GRAPH_NAME,
  openModelBytes,
} from "./helpers/model-fixture.ts";
import { countDispatches, requireCensus, SEAT_SIGNATURES } from "./helpers/pipeline-census.ts";

type Storage = "f32" | "f16" | "bf16";

const STORAGES: readonly Storage[] = ["bf16", "f16", "f32"];

/**
 * A weight `[n, k]` in `storage`: its stored bytes and the values those bytes hold (decoded by the
 * production decoders, never by the encoder). Signs mix, and the magnitudes run over three decades
 * below 1 — the range the band of `linear` is stated for (its absolute term is 2e-4, and the terms of
 * a 4096-long reduction of weights up to 10 already round further apart than that in f32).
 */
const weightOf = (
  storage: Storage,
  n: number,
  k: number,
): { readonly bytes: Uint8Array<ArrayBuffer>; readonly values: Float32Array<ArrayBuffer> } => {
  const raw = Float32Array.from(
    { length: n * k },
    (_, i) => Math.sin(i * 0.0113 + 0.4) * 10 ** ((i % 4) - 3),
  );
  if (storage === "bf16") return quantizeBf16(raw);
  if (storage === "f16") return quantizeF16(raw);
  return { bytes: new Uint8Array(raw.buffer.slice(0)), values: raw };
};

const inputOf = (rows: number, k: number): Float32Array<ArrayBuffer> =>
  Float32Array.from({ length: rows * k }, (_, i) => Math.sin(i * 0.037) * 0.75);

const biasOf = (n: number): Float32Array<ArrayBuffer> =>
  Float32Array.from({ length: n }, (_, i) => ((i % 7) - 3) * 0.11);

/** `Σ_k x[r,k] · w[c,k] + bias[c]` in f64, over the stored values. */
const referenceOf = (
  rows: number,
  n: number,
  k: number,
  x: Float32Array<ArrayBuffer>,
  w: Float32Array<ArrayBuffer>,
  bias: Float32Array<ArrayBuffer>,
): Float64Array<ArrayBuffer> => {
  const out = new Float64Array(rows * n);
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < n; col += 1) {
      let sum = 0;
      for (let inner = 0; inner < k; inner += 1) sum += x[row * k + inner] * w[col * k + inner];
      out[row * n + col] = sum + bias[col];
    }
  }
  return out;
};

describe.skipIf(!GPU_AVAILABLE)(
  "the parallel GEMV on float-stored weights (real GPU, the kernel alone)",
  () => {
    it("stays inside the band of linear against the f64 reference and against the sequential GEMV, gives the same words at M = 1, 4 and 8 and staged or not, and writes nothing else", async () => {
      const gpu = await acquireGpu();
      const device = gpu.device;
      // The largest share of the band any output used: against the f64 reference, and against the
      // sequential GEMV
      const worst = { reference: 0, sequential: 0 };
      try {
        for (const storage of STORAGES) {
          const unit = linearGemvUnit(storage);
          // One word, a count no lane count divides, and the reduction lengths of Gemma 4's linears
          for (const k of [unit, unit * 5, 1536, 4096]) {
            // 36 columns: no workgroup's column count (4 to 64) divides it
            const n = 36;
            const weight = weightOf(storage, n, k);
            const input = inputOf(8, k);
            const bias = biasOf(n);
            const reference = referenceOf(8, n, k, input, weight.values, bias);
            const owned: GPUBuffer[] = [];
            device.pushErrorScope("validation");
            const make = (
              value: number | ArrayBufferView<ArrayBuffer>,
              usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            ): GPUBuffer => {
              const buffer = device.createBuffer({
                size: typeof value === "number" ? value : value.byteLength,
                usage,
              });
              owned.push(buffer);
              if (typeof value !== "number") device.queue.writeBuffer(buffer, 0, value);
              return buffer;
            };
            try {
              const params = make(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
              const buffers = [
                params,
                make(input),
                make(weight.bytes),
                make(bias),
                make(
                  8 * n * 4,
                  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
                ),
              ];
              const mapped = make(8 * n * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
              const run = async (
                code: string,
                dims: Uint32Array<ArrayBuffer>,
                cols: number,
                m: number,
              ): Promise<Uint32Array<ArrayBuffer>> => {
                const uniform = new Uint32Array(4);
                uniform.set(dims);
                device.queue.writeBuffer(params, 0, uniform);
                // Do not let earlier valid values hide unwritten outputs.
                device.queue.writeBuffer(buffers[4], 0, new Uint32Array(8 * n).fill(0x7fc00000));
                const pipeline = await device.createComputePipelineAsync({
                  layout: "auto",
                  compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
                });
                const bindings = device.createBindGroup({
                  layout: pipeline.getBindGroupLayout(0),
                  entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
                });
                const encoder = device.createCommandEncoder();
                const pass = encoder.beginComputePass();
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, bindings);
                pass.dispatchWorkgroups(Math.ceil(n / cols), m);
                pass.end();
                encoder.copyBufferToBuffer(buffers[4], 0, mapped, 0, 8 * n * 4);
                device.queue.submit([encoder.finish()]);
                await mapped.mapAsync(GPUMapMode.READ);
                const out = new Uint32Array(mapped.getMappedRange().slice(0));
                mapped.unmap();
                for (const untouched of out.subarray(m * n)) assertEquals(untouched, 0x7fc00000);
                return out.slice(0, m * n);
              };
              const sequential = defaultLinearGemvVariant(DEFAULT_GEOMETRY_PROFILE.linearGemv);
              const baseline = new Float32Array(
                (
                  await run(
                    linearGemvWgsl(storage, undefined, sequential),
                    linearGemvParams(storage, 1, n, k),
                    sequential.cols,
                    1,
                  )
                ).buffer,
              );
              for (const lanes of [2, 4, 8, 16, 32] satisfies LinearGemvParallelLanes[]) {
                let first: Uint32Array<ArrayBuffer> | undefined;
                for (const m of [1, 4, 8]) {
                  const where = `${storage} K${k} L${lanes} M${m}`;
                  const dims = linearGemvParallelParams(storage, m, n, k);
                  const out = await run(
                    linearGemvParallelWgsl(storage, undefined, lanes),
                    dims,
                    128 / lanes,
                    m,
                  );
                  first ??= out;
                  assertEquals(out.subarray(0, n), first, `${where}: row 0 at every M`);
                  const staged = await run(
                    linearGemvParallelStagedWgsl(storage, undefined, lanes),
                    dims,
                    128 / lanes,
                    m,
                  );
                  assertEquals(staged, out, `${where}: staged`);
                  const values = new Float32Array(out.buffer);
                  for (let i = 0; i < values.length; i += 1) {
                    const expected = reference[i];
                    const tolerance =
                      GEMM_TOLERANCE.atol + GEMM_TOLERANCE.rtol * Math.abs(expected);
                    assert(Number.isFinite(values[i]), `${where} @${i}: ${values[i]}`);
                    assert(
                      Math.abs(values[i] - expected) <= tolerance,
                      `${where} @${i}: ${values[i]} / ${expected}`,
                    );
                    worst.reference = Math.max(
                      worst.reference,
                      Math.abs(values[i] - expected) / tolerance,
                    );
                    if (i < n) {
                      assert(
                        Math.abs(values[i] - baseline[i]) <= tolerance,
                        `${where} @${i}: against the sequential GEMV ${baseline[i]}`,
                      );
                      worst.sequential = Math.max(
                        worst.sequential,
                        Math.abs(values[i] - baseline[i]) / tolerance,
                      );
                    }
                  }
                }
              }
            } finally {
              await device.queue.onSubmittedWorkDone();
              for (const buffer of owned) buffer.destroy();
              assertEquals(await device.popErrorScope(), null);
            }
          }
        }
      } finally {
        gpu.destroy();
      }
      // The checks above are not vacuous: the reordered sums do differ from the sequential ones
      assert(worst.sequential > 0, "no output differs from the sequential GEMV");
      console.info(
        `[karume] float parallel GEMV: worst share of the band of linear — ${worst.reference.toFixed(3)} against f64, ${worst.sequential.toFixed(3)} against the sequential GEMV`,
      );
    });
  },
);

// Through a Session: a float shape takes the parallel GEMV only where the profile names it, only for
// a Session that asked for `reorder`, and only at 1 to 8 rows.

const CODEC: Readonly<Record<Storage, CodecName>> = { f32: "f32", f16: "f16", bf16: "bf16" };

/** One linear `[m, k] × [n, k]ᵀ + [n]` with its weight in `storage`. */
const linearModel = async (storage: Storage, m: number, n: number, k: number) => {
  const tensors: readonly TensorInput[] = [
    {
      graph: GRAPH_NAME,
      initializer: "w",
      bytes: weightOf(storage, n, k).bytes,
      encoding: { codec: CODEC[storage] },
    },
    {
      graph: GRAPH_NAME,
      initializer: "b",
      bytes: f32Bytes(Array.from(biasOf(n))),
      encoding: { codec: "f32" },
    },
  ];
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
  return await openModelBytes(graph, tensors);
};

/** The shapes of the Session cases: one row of the table per storage, with a column edge at 300. */
const SESSION_SHAPES = [
  { storage: "bf16", n: 300, k: 1536, lanes: 4, staged: false },
  { storage: "f16", n: 300, k: 2048, lanes: 32, staged: true },
  { storage: "f32", n: 300, k: 512, lanes: 8, staged: false },
] as const;

/** The default profile with one parallel row per case above (nothing else moved). */
const FLOAT_ROWS_PROFILE: GeometryProfile = {
  ...DEFAULT_GEOMETRY_PROFILE,
  id: "test-float-parallel-rows",
  linearGemv: {
    ...DEFAULT_GEOMETRY_PROFILE.linearGemv,
    parallelShapes: [
      ...DEFAULT_GEOMETRY_PROFILE.linearGemv.parallelShapes,
      ...SESSION_SHAPES.map(({ storage, n, k, lanes, staged }) => ({
        storage,
        n,
        k,
        lanes,
        packedActivations: false,
        ...(staged ? { stagedActivations: true as const } : {}),
      })),
    ],
  },
};

describe.skipIf(!GPU_AVAILABLE)(
  "where the parallel GEMV applies to a float-stored weight (real GPU, through a Session)",
  () => {
    it("runs for a shape the profile names at reorder and 1 to 8 rows with one summation order, and leaves exact Sessions, other row counts and other shapes on their kernels", async () => {
      const gpu = await acquireGpu({ geometryProfile: FLOAT_ROWS_PROFILE });
      try {
        for (const { storage, n, k, staged } of SESSION_SHAPES) {
          let parallelRow: Uint32Array<ArrayBuffer> | undefined;
          let exactRow: Uint32Array<ArrayBuffer> | undefined;
          for (const m of [1, 4, 8, 9]) {
            for (const reorder of [false, true]) {
              const where = `${storage} M${m} ${reorder ? "reorder" : "exact"}`;
              const session = await createSessionFromContainer(
                gpu,
                await linearModel(storage, m, n, k),
                GRAPH_NAME,
                reorder ? { numerics: { ops: { linear: "reorder" } } } : {},
              );
              try {
                const output = (
                  await session.run({ x: { dtype: "f32", shape: [m, k], data: inputOf(m, k) } })
                ).y;
                assertEquals(output.shape, [m, n]);
                const census = requireCensus(session.diagnostics().lastRunPipelines, where);
                const keys = census.map((row) => row.key);
                const parallel = reorder && m <= 8;
                assertEquals(
                  countDispatches(census, SEAT_SIGNATURES.linearGemvReduce.parallel) > 0,
                  parallel,
                  `${where} ${keys}`,
                );
                assertEquals(
                  keys.some((key) => key.endsWith(":staged-x")),
                  parallel && staged,
                  `${where} ${keys}`,
                );
                const row = new Uint32Array(output.data.buffer, output.data.byteOffset, n).slice();
                if (parallel) {
                  parallelRow ??= row;
                  assertEquals(row, parallelRow, `${where}: row 0 at every M`);
                } else {
                  // The sequential GEMV (M = 1) and the GEMM (M ≥ 2) sum in ascending k: one row,
                  // the same words — at exact, and at reorder past the parallel range
                  exactRow ??= row;
                  assertEquals(row, exactRow, `${where}: the reference order`);
                }
              } finally {
                await session.dispose();
              }
            }
          }
          assert(parallelRow !== undefined && exactRow !== undefined);
          const reference = new Float32Array(exactRow.buffer);
          const reordered = new Float32Array(parallelRow.buffer);
          for (let i = 0; i < n; i += 1) {
            assert(
              Math.abs(reordered[i] - reference[i]) <=
                GEMM_TOLERANCE.atol + GEMM_TOLERANCE.rtol * Math.abs(reference[i]),
              `${storage} @${i}: ${reordered[i]} / ${reference[i]}`,
            );
          }
        }
        // A shape the profile does not name keeps the sequential GEMV at reorder
        const session = await createSessionFromContainer(
          gpu,
          await linearModel("bf16", 1, 296, 1536),
          GRAPH_NAME,
          { numerics: { ops: { linear: "reorder" } } },
        );
        try {
          await session.run({ x: { dtype: "f32", shape: [1, 1536], data: inputOf(1, 1536) } });
          const census = requireCensus(session.diagnostics().lastRunPipelines, "no row");
          assertEquals(countDispatches(census, SEAT_SIGNATURES.linearGemvReduce.parallel), 0);
        } finally {
          await session.dispose();
        }
      } finally {
        gpu.destroy();
      }
    });
  },
);
