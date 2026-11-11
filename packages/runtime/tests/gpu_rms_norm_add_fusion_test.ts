import { assertEquals, assertRejects } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { checkFusionTwin } from "@karume/kernel-checks";
import { acquireGpu } from "../src/gpu/device.ts";
import {
  createSessionFromContainer,
  type SessionOptions,
  type Tensor,
} from "../src/runtime/executor.ts";
import { BUFFER_USAGE as U, MAP_MODE } from "../src/gpu/webgpu-constants.ts";
import { rmsNormAddWgsl, rmsNormParams } from "../src/kernels/rms-norm.ts";
import { fill, GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckCase, assertWordIdentity } from "./helpers/kernel-check-assert.ts";
import { rmsNormAddGraph } from "./helpers/rms-norm-add-graph.ts";

const words = (t: Tensor): Uint32Array<ArrayBuffer> =>
  new Uint32Array(t.data.buffer, t.data.byteOffset, t.data.length);

describe.skipIf(!GPU_AVAILABLE)("RMS→add fusion (real GPU)", () => {
  for (const dim of [256, 1536, 2560, 2816, 3840]) {
    for (const normFirst of [false, true]) {
      it(`dim=${dim} / normFirst=${normFirst}: matches the reference across row counts and cancellation`, async () => {
        const gpu = await acquireGpu();
        try {
          let firstRow: Uint32Array<ArrayBuffer> | undefined;
          for (const rows of [1, 4, 8, 40, 64]) {
            const options = { rows, dim, normFirst };
            const fused = await createSessionFromContainer(
              gpu,
              await openModelBytes(rmsNormAddGraph(options), []),
              GRAPH_NAME,
              { fuseRmsNormAdd: true },
            );
            const reference = await createSessionFromContainer(
              gpu,
              await openModelBytes(rmsNormAddGraph({ ...options, normOutput: true }), []),
              GRAPH_NAME,
            );
            try {
              for (const pattern of ["finite", "cancel", "special", "zeros"]) {
                const x = fill([rows, dim], (i) => Math.sin((i % dim) * 0.713 + 0.19) * 4);
                const w = fill([dim], (i) => 0.2 + Math.cos(i * 0.17));
                const r = fill([rows, dim], (i) => Math.cos((i % dim) * 0.531) * 3);
                if (pattern === "special") {
                  words(x).set([
                    0, 0x80000000, 1, 0x80000001, 0x00800000, 0x80800000, 0x7f800000, 0xff800000,
                    0x7fc00123,
                  ]);
                }
                if (pattern === "zeros") {
                  words(x).set(
                    Uint32Array.from({ length: rows * dim }, (_, i) => (i % 2 ? 0x80000000 : 0)),
                  );
                }
                if (pattern === "cancel") {
                  const n = (await reference.run({ x, w, r })).n;
                  for (let i = 0; i < n.data.length; i++) r.data[i] = -n.data[i];
                }
                // y and the late x consumer (`copy`); the reference also outputs the norm.
                assertCheckCase(
                  await checkFusionTwin(gpu, {
                    label: `${rows}/${pattern}`,
                    fused,
                    plain: reference,
                    feeds: { x, w, r },
                    counter: "rmsNormAdd",
                    hits: 1,
                    nan: "class",
                    outputs: ["y", "copy"],
                    dispatches: { fused: 2 },
                  }),
                );
                if (pattern === "finite") {
                  // The fused output, from a run of its own (the harness returns a case).
                  const row = words((await fused.run({ x, w, r })).y).slice(0, dim);
                  if (firstRow === undefined) firstRow = row;
                  else {
                    assertWordIdentity(
                      row,
                      firstRow,
                      "class",
                      "M=1 matches the first row of the multi-row run",
                    );
                  }
                }
              }
            } finally {
              await fused.dispose();
              await reference.dispose();
            }
          }
        } finally {
          gpu.destroy();
        }
      });
    }
  }
  it("forced grid-stride writes every row and executes the uniform XOR", async () => {
    const gpu = await acquireGpu(),
      d = gpu.device,
      rows = 9,
      dim = 1536;
    const owned: GPUBuffer[] = [];
    d.pushErrorScope("validation");
    const make = (data: ArrayBufferView<ArrayBuffer> | number, usage: number): GPUBuffer => {
      const buffer = d.createBuffer({
        size: typeof data === "number" ? data : data.byteLength,
        usage,
      });
      owned.push(buffer);
      if (typeof data !== "number") d.queue.writeBuffer(buffer, 0, data);
      return buffer;
    };
    try {
      const x = fill([rows, dim], (i) => Math.sin(i * 0.713 + 0.19) * 4),
        w = fill([dim], () => 1),
        r = fill([rows, dim], () => 0);
      const reference = await createSessionFromContainer(
        gpu,
        await openModelBytes(rmsNormAddGraph({ rows, dim }), []),
        GRAPH_NAME,
      );
      try {
        const expected = words((await reference.run({ x, w, r })).y);
        const params = make(rmsNormParams(rows, dim, 1e-6), U.UNIFORM | U.COPY_DST);
        const buffers = [
          params,
          ...[x, w, r].map((t) => make(t.data, U.STORAGE | U.COPY_DST)),
          make(rows * dim * 4, U.STORAGE | U.COPY_SRC | U.COPY_DST),
        ];
        const stage = make(rows * dim * 4, U.COPY_DST | U.MAP_READ);
        const pipeline = await d.createComputePipelineAsync({
          layout: "auto",
          compute: {
            module: d.createShaderModule({ code: rmsNormAddWgsl("residual-norm", dim) }),
            entryPoint: "main",
          },
        });
        const group = d.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
        });
        for (const mask of [0, 1, 0x80000000]) {
          d.queue.writeBuffer(params, 12, new Uint32Array([mask]));
          d.queue.writeBuffer(buffers[4], 0, new Uint32Array(rows * dim).fill(0xdeadbeef));
          const e = d.createCommandEncoder(),
            p = e.beginComputePass();
          p.setPipeline(pipeline);
          p.setBindGroup(0, group);
          p.dispatchWorkgroups(2);
          p.end();
          e.copyBufferToBuffer(buffers[4], 0, stage, 0, rows * dim * 4);
          d.queue.submit([e.finish()]);
          await stage.mapAsync(MAP_MODE.READ);
          const actual = new Uint32Array(stage.getMappedRange().slice(0));
          stage.unmap();
          assertWordIdentity(
            actual,
            expected.map((v) => (v ^ mask) >>> 0),
            "class",
            `mask=${mask}`,
          );
        }
      } finally {
        await reference.dispose();
      }
    } finally {
      d.queue.submit([]);
      await d.queue.onSubmittedWorkDone();
      const error = await d.popErrorScope();
      for (const b of owned) b.destroy();
      gpu.destroy();
      assertEquals(error, null);
    }
  });
  it("does not silently accept a non-boolean", async () => {
    const gpu = await acquireGpu();
    const opened = await openModelBytes(rmsNormAddGraph(), []);
    try {
      for (const value of [null, 0, "true"]) {
        const options: SessionOptions = {};
        // Inject an invalid value at the public boundary; the type alone is not a guard.
        Reflect.set(options, "fuseRmsNormAdd", value);
        await assertRejects(
          () => createSessionFromContainer(gpu, opened, GRAPH_NAME, options),
          Error,
          "fuseRmsNormAdd",
        );
      }
    } finally {
      gpu.destroy();
    }
  });
});
