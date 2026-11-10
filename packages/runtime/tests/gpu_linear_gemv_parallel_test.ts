import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import type { CodecName } from "../src/format/container/codecs.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import {
  type LinearGemvParallelLanes,
  linearGemvParallelWgsl,
  linearGemvParams,
  linearGemvUnit,
  linearGemvWgsl,
} from "../src/kernels/linear-gemv.ts";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import type { TensorInput } from "./helpers/container-write.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import {
  type DeclarationJson,
  f32Bytes,
  GRAPH_NAME,
  openModelBytes,
} from "./helpers/model-fixture.ts";
import { GEMM_TOLERANCE } from "./helpers/op-tolerance.ts";
import { countDispatches, requireCensus, SEAT_SIGNATURES } from "./helpers/pipeline-census.ts";

type Storage = "i2" | "i4" | "i8";

/** Deterministic inputs that cross packed-word boundaries, use per-group scales and mix signs. */
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
  const input = Float32Array.from({ length: 8 * k }, (_, i) => Math.sin(i * 0.037) * 0.75);
  const bias = Float32Array.from({ length: n }, (_, i) => ((i % 7) - 3) * 0.11);
  const scale = Float32Array.from(
    { length: n * (group === undefined ? 1 : k / group) },
    (_, i) => ((i % 17) + 1) * 0.00017,
  );
  const reference = new Float64Array(8 * n);
  for (let row = 0; row < 8; row++) {
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

describe.skipIf(!GPU_AVAILABLE)("parallel accumulation of the quantized GEMV (real GPU)", () => {
  it("matches the FP64 reference within the error band, and keeps edge columns, spare lanes and identical accumulation for M1/4/8", async () => {
    const gpu = await acquireGpu();
    const device = gpu.device;
    try {
      for (const storage of ["i2", "i4", "i8"] as const) {
        const unit = linearGemvUnit(storage);
        const shapes = [
          { k: unit, group: storage === "i4" ? 32 : undefined },
          {
            k: unit * 5,
            group: storage === "i4" ? 32 : undefined,
          },
          { k: 4096, group: storage === "i4" ? 32 : undefined },
        ];
        if (storage === "i4") {
          shapes.push(
            { k: 1536, group: 512 },
            { k: 6144, group: 2048 },
            {
              k: 4096,
              group: 4096,
            },
          );
        }
        // The reduction length × group of the E4B rows (2026-09-26; lanes copied from E2B, not
        // measured). The arithmetic of 1 output depends only on (storage, k, group, lanes) and n
        // changes only the dispatch count, so every lanes value runs with the same 36 columns as
        // above (including an edge that the workgroup's column count does not divide). i2 has no
        // E4B rows.
        if (storage === "i4") {
          shapes.push(
            { k: 2560, group: 32 },
            { k: 10240, group: 32 },
            { k: 2048, group: 32 },
            { k: 256, group: 32 },
            { k: 2560, group: 512 },
            { k: 2048, group: 2048 },
            { k: 10240, group: 2048 },
          );
        }
        if (storage === "i8") {
          shapes.push({ k: 2560, group: undefined }, { k: 256, group: undefined });
        }
        for (const { k, group } of shapes) {
          const n = 36;
          const data = fixture(storage, n, k, group);
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
            const buffers = [
              make(
                linearGemvParams(storage, 8, n, k, group),
                GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
              ),
              make(data.input),
              make(data.packed),
              make(data.bias),
              make(
                8 * n * 4,
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
              ),
              make(data.scale),
            ];
            const mapped = make(8 * n * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
            const run = async (
              code: string,
              cols: number,
              m: number,
            ): Promise<Uint32Array<ArrayBuffer>> => {
              device.queue.writeBuffer(buffers[0], 0, linearGemvParams(storage, m, n, k, group));
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
            const baseline = new Float32Array(
              (await run(linearGemvWgsl(storage, group), 32, 1)).buffer,
            );
            for (const lanes of [2, 4, 8, 16, 32] satisfies LinearGemvParallelLanes[]) {
              let first: Uint32Array<ArrayBuffer> | undefined;
              for (const m of [1, 4, 8]) {
                const out = await run(
                  linearGemvParallelWgsl(storage, group, lanes),
                  128 / lanes,
                  m,
                );
                first ??= out;
                assertEquals(
                  out.subarray(0, n),
                  first,
                  `${storage} K${k} L${lanes} M${m}: decode/verify`,
                );
                const values = new Float32Array(out.buffer);
                for (let i = 0; i < values.length; i++) {
                  const expected = data.reference[i];
                  const tolerance = GEMM_TOLERANCE.atol + GEMM_TOLERANCE.rtol * Math.abs(expected);
                  assert(Number.isFinite(values[i]));
                  assert(
                    Math.abs(values[i] - expected) <= tolerance,
                    `${storage} K${k} L${lanes} M${m} @${i}: ${values[i]} / ${expected}`,
                  );
                  if (i < n) {
                    // This fixed input does not widen the band; the A/B is also checked within the
                    // existing linear error band.
                    assert(
                      Math.abs(values[i] - baseline[i]) <= tolerance,
                      `${storage} K${k} L${lanes} @${i}: A/B`,
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
  });
});

// Through a session: the `parallel` opt-in reaches only the requested linears (pipeline census).

/** Storage name → name registered in the codec ledger (i2 is the per-channel `int2-off`). */
const CODEC: Readonly<Record<"i2" | "i4" | "i8", CodecName>> = {
  i2: "int2-off",
  i4: "int4-sym-g",
  i8: "int8-sym",
};

describe.skipIf(!GPU_AVAILABLE)("where the parallel GEMV accumulation applies (real GPU)", () => {
  it("applies only to the explicitly requested targets and keeps every output column u32-identical for M1/4/8", async () => {
    const gpu = await acquireGpu();
    try {
      for (const shape of [
        { storage: "i2", bits: 2, n: 12288, k: 1536, group: undefined },
        { storage: "i4", bits: 4, n: 256, k: 1536, group: 32 },
        { storage: "i4", bits: 4, n: 256, k: 1536, group: 512 },
        { storage: "i4", bits: 4, n: 1536, k: 2048, group: 2048 },
        { storage: "i4", bits: 4, n: 1536, k: 4096, group: 4096 },
        { storage: "i8", bits: 8, n: 256, k: 1536, group: undefined },
        // Representatives of the E4B rows (2026-09-26; lanes copied from E2B, not measured) —
        // l32 / l4 × g32 / g512 / g2048 / i8. Every output column is checked at the real n.
        { storage: "i4", bits: 4, n: 512, k: 2560, group: 32 },
        { storage: "i4", bits: 4, n: 10240, k: 2560, group: 512 },
        { storage: "i4", bits: 4, n: 2560, k: 10240, group: 2048 },
        { storage: "i8", bits: 8, n: 2560, k: 256, group: undefined },
        // Representatives of the 12B rows (2026-11-06) — l4 / l32 / l16 at the real n.
        { storage: "i4", bits: 4, n: 15360, k: 3840, group: 32 },
        { storage: "i4", bits: 4, n: 512, k: 3840, group: 32 },
        { storage: "i4", bits: 4, n: 3840, k: 15360, group: 32 },
        { storage: "i4", bits: 4, n: 262144, k: 3840, group: 32 },
        // Representatives of the 26B rows (2026-11-10) — l4 / l32 / l32 staged / the l4 head at
        // the real n.
        { storage: "i4", bits: 4, n: 8192, k: 2816, group: 32 },
        { storage: "i4", bits: 4, n: 128, k: 2816, group: 32 },
        { storage: "i4", bits: 4, n: 2816, k: 2112, group: 32 },
        { storage: "i4", bits: 4, n: 262144, k: 2816, group: 32 },
      ] as const) {
        const { storage, n, k, bits, group } = shape;
        const w = Uint8Array.from(
          { length: (n * k * bits) / 8 },
          (_, i) => Math.imul(i + 1, 0x9e3779b9) >>> 24,
        );
        // Per-channel (i2 / i8) has groups = 1; group quantization (i4) has row length / group.
        const groups = group === undefined ? 1 : k / group;
        const tensors: readonly TensorInput[] = [
          {
            graph: GRAPH_NAME,
            initializer: "w",
            bytes: w,
            encoding: {
              codec: CODEC[storage],
              groupSize: group ?? k,
              scale: {
                dtype: "f32",
                bytes: f32Bytes(
                  Array.from({ length: n * groups }, (_, i) => ((i % 17) + 1) * 0.00017),
                ),
              },
            },
          },
          {
            graph: GRAPH_NAME,
            initializer: "b",
            bytes: f32Bytes(Array.from({ length: n }, (_, i) => ((i % 7) - 3) * 0.11)),
            encoding: { codec: "f32" },
          },
        ];
        let expected: Uint32Array<ArrayBuffer> | undefined;
        for (const m of [1, 4, 8, 9]) {
          for (const parallel of m === 1 ? [false, true] : [true]) {
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
            const session = await createSessionFromContainer(
              gpu,
              await openModelBytes(graph, tensors),
              GRAPH_NAME,
              parallel ? { linearGemvReduce: "parallel" } : {},
            );
            try {
              const input = Float32Array.from(
                { length: m * k },
                (_, i) => Math.sin(i * 0.037) * 0.75,
              );
              const output = (
                await session.run({ x: { dtype: "f32", shape: [m, k], data: input } })
              ).y;
              assertEquals(output.shape, [m, n]);
              for (const value of output.data) assert(Number.isFinite(value));
              const where = `${storage} M${m}`;
              const census = requireCensus(session.diagnostics().lastRunPipelines, where);
              assertEquals(
                countDispatches(census, SEAT_SIGNATURES.linearGemvReduce.parallel) > 0,
                parallel && m <= 8,
                `${where} ${census.map((row) => row.key)}`,
              );
              if (parallel && m <= 8) {
                const first = new Uint32Array(
                  output.data.buffer,
                  output.data.byteOffset,
                  n,
                ).slice();
                expected ??= first;
                assertEquals(first, expected);
              }
            } finally {
              await session.dispose();
            }
          }
        }
      }
    } finally {
      gpu.destroy();
    }
  });
});
