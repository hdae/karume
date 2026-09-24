// DenoのGPUテストとChromeの実走ハーネスで同じ数値・寿命の検査を使う。
import { assert, assertEquals } from "@std/assert";
import type { GpuContext } from "../../src/gpu/device.ts";
import { BUFFER_USAGE as U, MAP_MODE } from "../../src/gpu/webgpu-constants.ts";
import { rmsNormParams } from "../../src/kernels/rms-norm.ts";
import { rmsNormSubgroupWgsl } from "../../src/kernels/rms-norm-subgroup.ts";
import { compareTensors, formatAllclose } from "../../src/reference/allclose.ts";
import { applyReferenceOp, refTensor } from "../../src/reference/ops.ts";
import { createSessionFromContainer, type Tensor } from "../../src/runtime/executor.ts";
import { fill, GRAPH_NAME, openModelBytes, singleOpDeclaration } from "./model-fixture.ts";
import { rmsNormAddGraph } from "./rms-norm-add-graph.ts";
import { opTolerance } from "./op-tolerance.ts";

const bits = (t: Tensor): Uint32Array<ArrayBuffer> =>
  new Uint32Array(t.data.buffer, t.data.byteOffset, t.data.length);
const close = (actual: Tensor, expected: Tensor): void => {
  const report = compareTensors(actual, expected, opTolerance("rms_norm"));
  assert(report.pass, formatAllclose(report));
};

/** 端数幅・先頭行・両加算順・借用入力の寿命を実Sessionで検査する。 */
export const checkRmsSubgroup = async (
  gpu: GpuContext,
): Promise<{ cases: number; elements: number }> => {
  let cases = 0, elements = 0;
  for (const dim of [1, 17, 128, 129, 255, 256, 257, 1024, 1536, 1537, 2560, 2561]) {
    let first: Uint32Array<ArrayBuffer> | undefined;
    for (const rows of [1, 4, 8, 19]) {
      const x = fill(
        [rows, dim],
        (i) => Math.sin((i % dim) * .713 + .19) * (1 + Math.floor(i / dim)),
      );
      const w = fill([dim], (i) => .2 + Math.cos(i * .17));
      const graph = singleOpDeclaration("rms_norm", [[rows, dim], [dim]], [[rows, dim]], {
        attrs: { eps: 1e-6 },
      });
      const session = await createSessionFromContainer(
        gpu,
        await openModelBytes(graph, []),
        GRAPH_NAME,
        { rmsNormReduce: "subgroup32" },
      );
      try {
        const { y } = await session.run({ x0: x, x1: w });
        close(y, applyReferenceOp("rms_norm", [x, w], { eps: 1e-6 }));
        const row = bits(y).slice(0, dim);
        if (first === undefined) first = row;
        else assertEquals(row, first, `dim=${dim}: M=1/4/8/19の先頭行`);
        cases++;
        elements += rows * dim;
      } finally {
        await session.dispose();
      }
    }
  }
  for (const dim of [256, 1536, 2560]) {
    for (const normFirst of [false, true]) {
      const rows = 19;
      const x = fill([rows, dim], (i) => Math.sin(i * .713 + .19) * 4);
      const w = fill([dim], (i) => .2 + Math.cos(i * .17));
      const r = fill([rows, dim], (i) => Math.cos(i * .531) * 3);
      const model = await openModelBytes(rmsNormAddGraph({ rows, dim, normFirst }), []);
      const session = await createSessionFromContainer(gpu, model, GRAPH_NAME, {
        rmsNormReduce: "subgroup32",
        fuseRmsNormAdd: true,
      });
      const separate = await createSessionFromContainer(gpu, model, GRAPH_NAME, {
        rmsNormReduce: "subgroup32",
      });
      try {
        for (const cancel of [false, true]) {
          if (cancel) {
            const norm = applyReferenceOp("rms_norm", [x, w], { eps: 1e-6 });
            for (let i = 0; i < r.data.length; i++) r.data[i] = -norm.data[i];
          }
          const actual = await session.run({ x, w, r }), expected = await separate.run({ x, w, r });
          assertEquals(
            bits(actual.y),
            bits(expected.y),
            "同じ縮約の融合・非融合は整数丸め障壁で一致",
          );
          assertEquals(bits(actual.copy), bits(expected.copy), "後続consumerのxの寿命");
          assertEquals(session.diagnostics().lastRunFusions?.rmsNormAdd, 1);
          cases++;
          elements += rows * dim;
        }
      } finally {
        await session.dispose();
        await separate.dispose();
      }
    }
  }
  // 2 workgroupで19行を処理し、共有部分和の読み終わりより早い上書きを検出する。
  const d = gpu.device, rows = 19, dim = 1537, owned: GPUBuffer[] = [];
  let error: GPUError | null = null;
  d.pushErrorScope("validation");
  const buffer = (data: ArrayBufferView<ArrayBuffer> | number, usage: number): GPUBuffer => {
    const b = d.createBuffer({ size: typeof data === "number" ? data : data.byteLength, usage });
    owned.push(b);
    if (typeof data !== "number") d.queue.writeBuffer(b, 0, data);
    return b;
  };
  try {
    const x = Float32Array.from({ length: rows * dim }, (_, i) => Math.sin(i * .23) * (1 + i % 7));
    const w = Float32Array.from({ length: dim }, (_, i) => Math.cos(i * .37));
    const resources = [
      buffer(rmsNormParams(rows, dim, 1e-6), U.UNIFORM | U.COPY_DST),
      buffer(x, U.STORAGE | U.COPY_DST),
      buffer(w, U.STORAGE | U.COPY_DST),
      buffer(x.byteLength, U.STORAGE | U.COPY_SRC),
    ];
    const staging = buffer(x.byteLength, U.COPY_DST | U.MAP_READ);
    const pipeline = await d.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module: d.createShaderModule({ code: rmsNormSubgroupWgsl() }),
        entryPoint: "main",
      },
    });
    const group = d.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: resources.map((b, binding) => ({ binding, resource: { buffer: b } })),
    });
    const encoder = d.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(2);
    pass.end();
    encoder.copyBufferToBuffer(resources[3], 0, staging, 0, x.byteLength);
    d.queue.submit([encoder.finish()]);
    await staging.mapAsync(MAP_MODE.READ);
    const actual = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    close(
      refTensor([rows, dim], actual),
      applyReferenceOp("rms_norm", [refTensor([rows, dim], x), refTensor([dim], w)], { eps: 1e-6 }),
    );
    cases++;
    elements += x.length;
  } finally {
    d.queue.submit([]);
    await d.queue.onSubmittedWorkDone();
    error = await d.popErrorScope();
    for (const b of owned) b.destroy();
  }
  if (error) throw Error(error.message);
  return { cases, elements };
};
