import { assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import type { CodecName } from "../src/format/container/codecs.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import {
  type LinearGemvParallelLanes,
  linearGemvParallelPackedParams,
  linearGemvParallelPackedStagedWgsl,
  linearGemvParallelPackedWgsl,
  linearGemvParallelStagedKey,
  linearGemvParallelStagedWgsl,
  linearGemvParallelWgsl,
  linearGemvParams,
  linearGemvStaticQuantizePackedParams,
  linearGemvStaticQuantizePackedStagedWgsl,
  linearGemvStaticQuantizePackedWgsl,
  linearGemvStaticQuantizeParams,
  linearGemvStaticQuantizeStagedWgsl,
  linearGemvStaticQuantizeWgsl,
  linearGemvUnit,
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
import { requireCensus } from "./helpers/pipeline-census.ts";

type Storage = "i2" | "i4" | "i8";

/** Deterministic weights, scales, bias and 8 activation rows whose signs and magnitudes vary. */
const fixture = (storage: Storage, n: number, k: number, group?: number) => {
  const bits = storage === "i2" ? 2 : storage === "i4" ? 4 : 8;
  return {
    packed: Uint32Array.from(
      { length: (n * k * bits) / 32 },
      (_, i) => Math.imul(i + 1, 0x9e3779b9) >>> 0,
    ),
    input: Float32Array.from({ length: 8 * k }, (_, i) => Math.sin(i * 0.037) * 0.75),
    bias: Float32Array.from({ length: n }, (_, i) => ((i % 7) - 3) * 0.11),
    scale: Float32Array.from(
      { length: n * (group === undefined ? 1 : k / group) },
      (_, i) => ((i % 17) + 1) * 0.00017,
    ),
  };
};

describe.skipIf(!GPU_AVAILABLE)("staged-activation parallel GEMV (K-90, real GPU)", () => {
  it("writes the same u32 words as the plain parallel GEMV for every column, lane count and M = 1 / 4 / 8", async () => {
    const gpu = await acquireGpu();
    const device = gpu.device;
    try {
      for (const storage of ["i2", "i4", "i8"] as const) {
        const unit = linearGemvUnit(storage);
        const group = storage === "i4" ? 32 : undefined;
        // K: fewer words than lanes, a word count that is not a multiple of 32 (a partly filled last
        // chunk) and the production lengths of the marked rows.
        const shapes: { k: number; group: number | undefined }[] = [
          { k: unit * 5, group },
          { k: unit * 45, group },
          { k: 2048, group },
          { k: 6144, group },
        ];
        if (storage === "i4")
          shapes.push({ k: 12288, group: 32 }, { k: 4096, group: 512 }, { k: 2048, group: 2048 });
        for (const { k, group: shapeGroup } of shapes) {
          // 36 columns: not a multiple of any lane count's columns per workgroup (edge columns).
          const n = 36;
          const data = fixture(storage, n, k, shapeGroup);
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
                linearGemvParams(storage, 8, n, k, shapeGroup),
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
              lanes: number,
              m: number,
            ): Promise<Uint32Array<ArrayBuffer>> => {
              device.queue.writeBuffer(
                buffers[0],
                0,
                linearGemvParams(storage, m, n, k, shapeGroup),
              );
              // A NaN fill, so an unwritten word cannot hide behind an earlier correct value.
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
              pass.dispatchWorkgroups(Math.ceil(n / (128 / lanes)), m);
              pass.end();
              encoder.copyBufferToBuffer(buffers[4], 0, mapped, 0, 8 * n * 4);
              device.queue.submit([encoder.finish()]);
              await mapped.mapAsync(GPUMapMode.READ);
              const out = new Uint32Array(mapped.getMappedRange().slice(0));
              mapped.unmap();
              for (const untouched of out.subarray(m * n)) assertEquals(untouched, 0x7fc00000);
              return out.slice(0, m * n);
            };
            for (const lanes of [2, 4, 16, 32] satisfies LinearGemvParallelLanes[]) {
              for (const m of [1, 4, 8]) {
                const where = `${storage} K${k} g${shapeGroup ?? "-"} L${lanes} M${m}`;
                const plain = await run(
                  linearGemvParallelWgsl(storage, shapeGroup, lanes),
                  lanes,
                  m,
                );
                const staged = await run(
                  linearGemvParallelStagedWgsl(storage, shapeGroup, lanes),
                  lanes,
                  m,
                );
                assertEquals(staged, plain, where);
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

const CODEC: Readonly<Record<Storage, CodecName>> = {
  i2: "int2-off",
  i4: "int4-sym-g",
  i8: "int8-sym",
};

describe.skipIf(!GPU_AVAILABLE)("staged-activation parallel GEMV in a Session (real GPU)", () => {
  it("runs only on the marked rows under linearGemvReduce: 'parallel', with the same words for M = 1 / 4 / 8", async () => {
    const gpu = await acquireGpu();
    try {
      for (const shape of [
        // Marked (gemma4 E2B o projection): takes the staged key.
        { storage: "i4", bits: 4, n: 1536, k: 2048, group: 32, staged: true },
        // Not marked (E2B k / v, K = 1536 measured slower staged): keeps the plain key.
        { storage: "i4", bits: 4, n: 256, k: 1536, group: 32, staged: false },
        // Marked by its packed-activation row instead (ADR 0105), not staged.
        { storage: "i4", bits: 4, n: 1536, k: 2048, group: 2048, staged: false },
      ] as const) {
        const { storage, n, k, bits, group } = shape;
        const w = Uint8Array.from(
          { length: (n * k * bits) / 8 },
          (_, i) => Math.imul(i + 1, 0x9e3779b9) >>> 24,
        );
        const tensors: readonly TensorInput[] = [
          {
            graph: GRAPH_NAME,
            initializer: "w",
            bytes: w,
            encoding: {
              codec: CODEC[storage],
              groupSize: group,
              scale: {
                dtype: "f32",
                bytes: f32Bytes(
                  Array.from({ length: n * (k / group) }, (_, i) => ((i % 17) + 1) * 0.00017),
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
        for (const m of [1, 4, 8]) {
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
            {
              linearGemvReduce: "parallel",
            },
          );
          try {
            const input = Float32Array.from(
              { length: m * k },
              (_, i) => Math.sin(i * 0.037) * 0.75,
            );
            const output = (await session.run({ x: { dtype: "f32", shape: [m, k], data: input } }))
              .y;
            const where = `${storage} n${n} k${k} g${group} M${m}`;
            const keys = requireCensus(session.diagnostics().lastRunPipelines, where).map(
              (row) => row.key,
            );
            assertEquals(
              keys.includes(linearGemvParallelStagedKey(storage, group, 32)),
              shape.staged,
              `${where} ${keys}`,
            );
            const first = new Uint32Array(output.data.buffer, output.data.byteOffset, n).slice();
            expected ??= first;
            assertEquals(first, expected, where);
          } finally {
            await session.dispose();
          }
        }
      }
    } finally {
      gpu.destroy();
    }
  });
});

type StagedPair = {
  readonly name: string;
  readonly packed: boolean;
  readonly unstaged: (
    storage: Storage,
    group: number | undefined,
    lanes: LinearGemvParallelLanes,
  ) => string;
  readonly staged: (
    storage: Storage,
    group: number | undefined,
    lanes: LinearGemvParallelLanes,
  ) => string;
  readonly params: (
    storage: Storage,
    m: number,
    n: number,
    k: number,
    group: number | undefined,
  ) => Uint32Array<ArrayBuffer>;
};

// Exact in f32, so the params' guards accept them.
const X_SCALE = Math.fround(0.0123);
const SRQ_SCALE = Math.fround(0.0371);

const STAGED_PAIRS: readonly StagedPair[] = [
  {
    name: "SRQ-fused",
    packed: false,
    unstaged: linearGemvStaticQuantizeWgsl,
    staged: linearGemvStaticQuantizeStagedWgsl,
    params: (storage, m, n, k, group) =>
      linearGemvStaticQuantizeParams(storage, m, n, k, SRQ_SCALE, group),
  },
  {
    name: "packed",
    packed: true,
    unstaged: linearGemvParallelPackedWgsl,
    staged: linearGemvParallelPackedStagedWgsl,
    params: (storage, m, n, k, group) =>
      linearGemvParallelPackedParams(storage, m, n, k, X_SCALE, group),
  },
  {
    name: "SRQ-fused packed",
    packed: true,
    unstaged: linearGemvStaticQuantizePackedWgsl,
    staged: linearGemvStaticQuantizePackedStagedWgsl,
    params: (storage, m, n, k, group) =>
      linearGemvStaticQuantizePackedParams(storage, m, n, k, X_SCALE, SRQ_SCALE, group),
  },
];

describe.skipIf(!GPU_AVAILABLE)(
  "staged forms of the SRQ-fused and packed parallel GEMVs (real GPU)",
  () => {
    it("write the same u32 words as their unstaged bodies for every column, lane count and M = 1 / 4 / 8", async () => {
      const gpu = await acquireGpu();
      const device = gpu.device;
      try {
        for (const pair of STAGED_PAIRS) {
          for (const storage of ["i2", "i4", "i8"] as const) {
            const unit = linearGemvUnit(storage);
            // K: fewer words than lanes, a partly filled last chunk, and the QAT rows' lengths (K = 2048
            // / 12288 for the packed rows, 2560 for the E4B unpacked lanes-32 rows).
            const shapes: { k: number; group: number | undefined }[] =
              storage === "i4"
                ? [
                    { k: unit * 5, group: 32 },
                    { k: unit * 45, group: 32 },
                    { k: 2048, group: 2048 },
                    { k: 2560, group: 512 },
                    { k: 4096, group: 4096 },
                  ]
                : [
                    { k: unit * 5, group: undefined },
                    { k: unit * 45, group: undefined },
                    { k: 2560, group: undefined },
                    { k: 12288, group: undefined },
                  ];
            for (const { k, group } of shapes) {
              const n = 36;
              const data = fixture(storage, n, k, group);
              // Packed activations: k / 4 words per row of four int8 codes (ADR 0105).
              const input = pair.packed
                ? Uint32Array.from(
                    { length: (8 * k) / 4 },
                    (_, i) => Math.imul(i + 7, 0x85ebca6b) >>> 0,
                  )
                : data.input;
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
                    pair.params(storage, 8, n, k, group),
                    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                  ),
                  make(input),
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
                  lanes: number,
                  m: number,
                ): Promise<Uint32Array<ArrayBuffer>> => {
                  device.queue.writeBuffer(buffers[0], 0, pair.params(storage, m, n, k, group));
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
                  pass.dispatchWorkgroups(Math.ceil(n / (128 / lanes)), m);
                  pass.end();
                  encoder.copyBufferToBuffer(buffers[4], 0, mapped, 0, 8 * n * 4);
                  device.queue.submit([encoder.finish()]);
                  await mapped.mapAsync(GPUMapMode.READ);
                  const out = new Uint32Array(mapped.getMappedRange().slice(0));
                  mapped.unmap();
                  for (const untouched of out.subarray(m * n)) assertEquals(untouched, 0x7fc00000);
                  return out.slice(0, m * n);
                };
                for (const lanes of [2, 4, 16, 32] satisfies LinearGemvParallelLanes[]) {
                  for (const m of [1, 4, 8]) {
                    const where = `${pair.name} ${storage} K${k} g${group ?? "-"} L${lanes} M${m}`;
                    const unstaged = await run(pair.unstaged(storage, group, lanes), lanes, m);
                    const staged = await run(pair.staged(storage, group, lanes), lanes, m);
                    assertEquals(staged, unstaged, where);
                  }
                }
              } finally {
                await device.queue.onSubmittedWorkDone();
                for (const buffer of owned) buffer.destroy();
                assertEquals(await device.popErrorScope(), null);
              }
            }
          }
        }
      } finally {
        gpu.destroy();
      }
    });
  },
);
