/**
 * Check 2 — the subgroup32 RMS reduction (`rmsNormReduce: "subgroup32"`, ADR 0100) through real
 * Sessions:
 *
 * - rows: ragged widths, M = 1 / 4 / 8 / 19 — within the `rms_norm` tolerance of the CPU reference,
 *   and row 0 word-identical at every M;
 * - fusion: the fused rms_norm + add against the same reduction unfused, both addition orders, with
 *   and without cancellation — word-identical outputs, the borrowed input's later consumer intact,
 *   and exactly one fusion in the run;
 * - grid-stride: 19 rows on 2 workgroups, which catches a shared partial sum overwritten before
 *   every lane has read it — within the tolerance of the CPU reference.
 */

import type { GpuContext } from "../../runtime/src/gpu/device.ts";
import { BUFFER_USAGE as U, MAP_MODE } from "../../runtime/src/gpu/webgpu-constants.ts";
import { rmsNormParams } from "../../runtime/src/kernels/rms-norm.ts";
import { rmsNormSubgroupWgsl } from "../../runtime/src/kernels/rms-norm-subgroup.ts";
import { compareTensors, formatAllclose } from "../../runtime/src/reference/allclose.ts";
import { applyReferenceOp, refTensor } from "../../runtime/src/reference/ops.ts";
import { createSessionFromContainer, type Tensor } from "../../runtime/src/runtime/executor.ts";
import {
  fill,
  GRAPH_NAME,
  openModelBytes,
  singleOpDeclaration,
} from "../../runtime/tests/helpers/model-fixture.ts";
import { opTolerance } from "../../runtime/tests/helpers/op-tolerance.ts";
import { rmsNormAddGraph } from "../../runtime/tests/helpers/rms-norm-add-graph.ts";
import {
  type CheckCase,
  type CheckOptions,
  type CheckPart,
  compareCount,
  compareWords,
  type Criterion,
  type ExpectedCases,
  f32Word,
  scopedGpuWork,
  throwIfAborted,
  wordsOf,
} from "./result.ts";

const ROW_DIMS = [1, 17, 128, 129, 255, 256, 257, 1024, 1536, 1537, 2560, 2561] as const;
const ROW_COUNTS = [1, 4, 8, 19] as const;
const FUSION_DIMS = [256, 1536, 2560] as const;
const FUSION_ROWS = 19;

/**
 * The cases each part of {@link checkRmsSubgroup} returns: 12 widths × 4 row counts; 3 widths × 2
 * addition orders × with and without cancellation; one grid-stride run.
 */
export const RMS_SUBGROUP_CASES = {
  "rms-rows": 48,
  "rms-fusion": 12,
  "rms-grid-stride": 1,
} as const satisfies ExpectedCases;

/** Within the `rms_norm` tolerance of the reference (`compareTensors`; its worst element on failure). */
const withinTolerance = (
  name: string,
  actual: Pick<Tensor, "dtype" | "data">,
  expected: Pick<Tensor, "dtype" | "data">,
): Criterion => {
  const report = compareTensors(actual, expected, opTolerance("rms_norm"));
  const words = expected.data.length;
  if (report.pass) return { name, words, failures: 0 };
  const index = report.worstIndex;
  return {
    name,
    words,
    failures: Math.max(report.failCount, 1),
    failure: {
      index,
      expected: f32Word(expected.data[index]),
      actual: wordsOf(actual.data)[index],
      at: "worst",
      note: formatAllclose(report),
    },
  };
};

/** Runs the check on a device with the subgroup features (`subgroup32Available`). */
export const checkRmsSubgroup = async (
  gpu: GpuContext,
  options: CheckOptions = {},
): Promise<readonly CheckPart[]> => {
  const rows = await rowsPart(gpu, options);
  const fusion = await fusionPart(gpu, options);
  const gridStride = await gridStridePart(gpu, options);
  return [rows, fusion, gridStride];
};

const rowsPart = async (gpu: GpuContext, options: CheckOptions): Promise<CheckPart> => {
  const cases: CheckCase[] = [];
  const total = ROW_DIMS.length * ROW_COUNTS.length;
  for (const dim of ROW_DIMS) {
    let first: Uint32Array<ArrayBuffer> | undefined;
    for (const rows of ROW_COUNTS) {
      throwIfAborted(options);
      const x = fill(
        [rows, dim],
        (i) => Math.sin((i % dim) * 0.713 + 0.19) * (1 + Math.floor(i / dim)),
      );
      const w = fill([dim], (i) => 0.2 + Math.cos(i * 0.17));
      const graph = singleOpDeclaration("rms_norm", [[rows, dim], [dim]], [[rows, dim]], {
        attrs: { eps: 1e-6 },
      });
      const label = `dim=${dim} M=${rows}`;
      const y = await scopedGpuWork(gpu, `RMS ${label}`, async () => {
        const session = await createSessionFromContainer(
          gpu,
          await openModelBytes(graph, []),
          GRAPH_NAME,
          { rmsNormReduce: "subgroup32" },
        );
        try {
          return (await session.run({ x0: x, x1: w })).y;
        } finally {
          await session.dispose();
        }
      });
      const row = wordsOf(y.data).slice(0, dim);
      first ??= row;
      cases.push({
        label,
        criteria: [
          withinTolerance(
            "within the rms_norm tolerance of the reference",
            y,
            applyReferenceOp("rms_norm", [x, w], { eps: 1e-6 }),
          ),
          compareWords("row 0 vs row 0 at M = 1 (u32)", row, first),
        ],
      });
      options.onProgress?.({ part: "rms-rows", done: cases.length, total });
    }
  }
  return {
    id: "rms-rows",
    title: "RMS subgroup32: ragged widths and row counts",
    cases,
    requirements: [],
  };
};

const fusionPart = async (gpu: GpuContext, options: CheckOptions): Promise<CheckPart> => {
  const cases: CheckCase[] = [];
  const total = FUSION_DIMS.length * 2 * 2;
  for (const dim of FUSION_DIMS) {
    for (const normFirst of [false, true]) {
      const rows = FUSION_ROWS;
      const x = fill([rows, dim], (i) => Math.sin(i * 0.713 + 0.19) * 4);
      const w = fill([dim], (i) => 0.2 + Math.cos(i * 0.17));
      const r = fill([rows, dim], (i) => Math.cos(i * 0.531) * 3);
      const model = await openModelBytes(rmsNormAddGraph({ rows, dim, normFirst }), []);
      await scopedGpuWork(gpu, `RMS fusion dim=${dim}`, async () => {
        const session = await createSessionFromContainer(gpu, model, GRAPH_NAME, {
          rmsNormReduce: "subgroup32",
          fuseRmsNormAdd: true,
        });
        const separate = await createSessionFromContainer(gpu, model, GRAPH_NAME, {
          rmsNormReduce: "subgroup32",
        });
        try {
          for (const cancel of [false, true]) {
            throwIfAborted(options);
            if (cancel) {
              const norm = applyReferenceOp("rms_norm", [x, w], { eps: 1e-6 });
              for (let i = 0; i < r.data.length; i++) r.data[i] = -norm.data[i];
            }
            const actual = await session.run({ x, w, r }),
              expected = await separate.run({ x, w, r });
            cases.push({
              label: `dim=${dim} ${normFirst ? "norm + residual" : "residual + norm"}${
                cancel ? " cancelling" : ""
              }`,
              criteria: [
                // The same reduction fused and unfused agrees through the integer rounding barrier.
                compareWords(
                  "fused vs unfused y (u32)",
                  wordsOf(actual.y.data),
                  wordsOf(expected.y.data),
                ),
                compareWords(
                  "fused vs unfused copy of x (the later consumer's x, u32)",
                  wordsOf(actual.copy.data),
                  wordsOf(expected.copy.data),
                ),
                compareCount(
                  "rms_norm + add fusions in the run",
                  session.diagnostics().lastRunFusions?.rmsNormAdd,
                  1,
                ),
              ],
            });
            options.onProgress?.({ part: "rms-fusion", done: cases.length, total });
          }
        } finally {
          await session.dispose();
          await separate.dispose();
        }
      });
    }
  }
  return {
    id: "rms-fusion",
    title: "RMS subgroup32: fused rms_norm + add vs unfused",
    cases,
    requirements: [],
  };
};

const gridStridePart = async (gpu: GpuContext, options: CheckOptions): Promise<CheckPart> => {
  throwIfAborted(options);
  const d = gpu.device,
    rows = 19,
    dim = 1537,
    owned: GPUBuffer[] = [];
  const x = Float32Array.from({ length: rows * dim }, (_, i) => Math.sin(i * 0.23) * (1 + (i % 7)));
  const w = Float32Array.from({ length: dim }, (_, i) => Math.cos(i * 0.37));
  const buffer = (data: ArrayBufferView<ArrayBuffer> | number, usage: number): GPUBuffer => {
    const b = d.createBuffer({ size: typeof data === "number" ? data : data.byteLength, usage });
    owned.push(b);
    if (typeof data !== "number") d.queue.writeBuffer(b, 0, data);
    return b;
  };
  const actual = await scopedGpuWork(gpu, "RMS grid-stride", async () => {
    try {
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
      const encoder = d.createCommandEncoder(),
        pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(2);
      pass.end();
      encoder.copyBufferToBuffer(resources[3], 0, staging, 0, x.byteLength);
      d.queue.submit([encoder.finish()]);
      await staging.mapAsync(MAP_MODE.READ);
      const out = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      return out;
    } finally {
      d.queue.submit([]);
      await d.queue.onSubmittedWorkDone();
      for (const b of owned) b.destroy();
    }
  });
  const criterion = withinTolerance(
    "within the rms_norm tolerance of the reference",
    refTensor([rows, dim], actual),
    applyReferenceOp("rms_norm", [refTensor([rows, dim], x), refTensor([dim], w)], { eps: 1e-6 }),
  );
  options.onProgress?.({ part: "rms-grid-stride", done: 1, total: 1 });
  return {
    id: "rms-grid-stride",
    title: "RMS subgroup32: 19 rows on 2 workgroups (grid-stride)",
    cases: [{ label: `dim=${dim} M=${rows} on 2 workgroups`, criteria: [criterion] }],
    requirements: [],
  };
};
