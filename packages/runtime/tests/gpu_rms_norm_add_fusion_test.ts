// The twin cases of the RMS→add fusion live in @karume/kernel-checks (checkRmsNormAddTwins), which
// the playground's /kernel-check page runs on the browser's backend too; the row identity across
// row counts, the forced grid-stride dispatch and the option's rejection stay here.

import { assertEquals, assertRejects } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  checkRmsNormAddTwins,
  RMS_NORM_ADD_DIMS,
  RMS_NORM_ADD_ORDERS,
  RMS_NORM_ADD_ROWS,
  RMS_NORM_ADD_TWIN_CASES,
  rmsNormAddInputs,
} from "@karume/kernel-checks";
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
import { assertCheckPart, assertWordIdentity } from "./helpers/kernel-check-assert.ts";
import { rmsNormAddGraph } from "./helpers/rms-norm-add-graph.ts";

const words = (t: Tensor): Uint32Array<ArrayBuffer> =>
  new Uint32Array(t.data.buffer, t.data.byteOffset, t.data.length);

describe.skipIf(!GPU_AVAILABLE)("RMS→add fusion (real GPU)", () => {
  it("matches the reference across widths, orders, row counts and cancellation", async () => {
    const gpu = await acquireGpu();
    try {
      const part = await checkRmsNormAddTwins(gpu);
      assertEquals([part.id, part.cases.length], Object.entries(RMS_NORM_ADD_TWIN_CASES)[0]);
      assertCheckPart(part);
    } finally {
      gpu.destroy();
    }
  });
  for (const dim of RMS_NORM_ADD_DIMS) {
    for (const normFirst of RMS_NORM_ADD_ORDERS) {
      it(`dim=${dim} / normFirst=${normFirst}: the M=1 row matches the first row of every multi-row run`, async () => {
        const gpu = await acquireGpu();
        try {
          let firstRow: Uint32Array<ArrayBuffer> | undefined;
          for (const rows of RMS_NORM_ADD_ROWS) {
            const fused = await createSessionFromContainer(
              gpu,
              await openModelBytes(rmsNormAddGraph({ rows, dim, normFirst }), []),
              GRAPH_NAME,
              { fuseRmsNormAdd: true },
            );
            try {
              const row = words((await fused.run(rmsNormAddInputs(rows, dim, "finite"))).y).slice(
                0,
                dim,
              );
              if (firstRow === undefined) firstRow = row;
              else {
                assertWordIdentity(
                  row,
                  firstRow,
                  "class",
                  "M=1 matches the first row of the multi-row run",
                );
              }
            } finally {
              await fused.dispose();
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
