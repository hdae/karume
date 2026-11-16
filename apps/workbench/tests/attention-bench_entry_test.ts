import {
  type AttentionBenchReport,
  type AttentionBenchRoute,
  type AttentionBenchShapeResult,
  WAN22_ATTENTION_SHAPES,
} from "@karume/runtime/tune";
import { describe, expect, it } from "vite-plus/test";

import { benchShapes, SHAPE_IDS } from "../src/attention-bench/shapes.ts";
import { attentionBenchSummary, benchProgressText } from "../src/attention-bench/summary.ts";
import attentionBench from "../src/entries/attention-bench.ts";
import { ParameterError, parseParameters } from "../src/entry.ts";

const [SELF_33F] = WAN22_ATTENTION_SHAPES;
if (SELF_33F === undefined) throw new Error("no Wan2.2 shape");

const stage = (name: string, ms: number) => ({
  stage: name,
  key: `${name}:key`,
  workgroups: [1, 1, 1] as const,
  msPerDispatch: ms,
});

const S16 = {
  eligible: true,
  rows: 2640,
  productionDispatches: 3,
  productionRows: 2640,
  rowWindow: true,
  stages: [stage("qk", 2), stage("stats", 0.5), stage("pv", 1.5)],
  msPerDispatch: 4,
  msPer1000Rows: 1.25,
  estimatedNodeMs: 12,
} satisfies AttentionBenchRoute;

const RESULT = {
  shape: SELF_33F,
  s16: S16,
  online: { eligible: false, reason: "the shape is not eligible" },
  quantization: { stages: [stage("quantize_q", 0.25)], msPerNode: 0.25 },
} satisfies AttentionBenchShapeResult;

const REPORT = {
  format: "karume-attention-bench/2",
  startedAt: "2026-11-16T10:00:00.000Z",
  date: "2026-11-16T10:01:00.000Z",
  aborted: false,
  userAgent: "Node.js/24",
  adapter: { vendor: "nvidia", architecture: "ampere", device: "", description: "" },
  timing: { unit: "ns", gpuTimingEnabled: true },
  variants: {
    geometryProfile: "nvidia-ampere",
    attentionQkGeometry: { regM: 4, regN: 4, wgX: 16, wgY: 16, tileK: 16 },
    attentionPvGeometry: { regM: 4, regN: 4, wgX: 16, wgY: 16, tileK: 16 },
    dp4a: {
      listed: true,
      dot: "emu",
      canary: { branch: "emu-exact", exact: true, maxAbsError: 0, maxBandRatio: 0 },
    },
    device: {
      subgroups: true,
      maxComputeWorkgroupStorageSize: 32768,
      subgroupLanesLinear: true,
      features: { subgroups: true, subgroupSizeControl: true, subgroupId: true },
      maxStorageBufferBindingSize: 2 ** 31,
      maxComputeWorkgroupsPerDimension: 65535,
    },
  },
  settings: {
    rounds: 5,
    quantization: true,
    onlineTiled: false,
    targetPassMs: 80,
    maxReps: 16384,
    warmupNs: 1,
    warmupMinRuns: 1,
  },
  shapes: [RESULT],
} satisfies AttentionBenchReport;

describe("the attention-bench entry", () => {
  it("declares the shapes, the rows per dispatch, the rounds, the quantization and the candidate, with the page's defaults", () => {
    expect(attentionBench.id).toBe("attention-bench");
    expect(parseParameters(attentionBench.parameters, {})).toEqual({
      shapes: WAN22_ATTENTION_SHAPES.map(({ id }) => id),
      rowsPerDispatch: 0,
      rounds: 5,
      quantization: true,
      onlineTiled: false,
    });
  });

  it.each([
    [{ rowsPerDispatch: "-1" }, "rowsPerDispatch"],
    [{ rounds: "0" }, "rounds"],
    [{ shapes: "wan22-self-9f" }, "shapes"],
  ])("refuses %o, naming the parameter", (given, parameter) => {
    expect(() => parseParameters(attentionBench.parameters, given)).toThrow(
      `Parameter ${parameter}:`,
    );
  });

  it("throws before acquiring a GPU when no shape is chosen", async () => {
    const run = attentionBench.run({
      parameters: parseParameters(attentionBench.parameters, { shapes: "" }),
      gpuOptions: {},
      signal: new AbortController().signal,
      progress: () => undefined,
    });
    await expect(run).rejects.toThrow(ParameterError);
  });
});

describe("benchShapes", () => {
  it("keeps the chosen shapes in the bench's order with production's split for 0", () => {
    expect(benchShapes([...SHAPE_IDS].reverse(), 0)).toEqual([...WAN22_ATTENTION_SHAPES]);
    expect(benchShapes(["wan22-self-121f"], 0).map(({ id }) => id)).toEqual(["wan22-self-121f"]);
  });

  it("sets the rows per dispatch, capped at each shape's query count", () => {
    const shapes = benchShapes(SHAPE_IDS, 10_000);
    expect(shapes.map(({ id, rowsPerDispatch }) => [id, rowsPerDispatch])).toEqual(
      WAN22_ATTENTION_SHAPES.map(({ id, queries }) => [id, Math.min(10_000, queries)]),
    );
  });
});

describe("attentionBenchSummary", () => {
  it("is the status facts, then one line per shape and route", () => {
    expect(attentionBenchSummary(REPORT)).toEqual([
      "timing: GPU timestamps (ns) · integer dot: emulated (canary emu-exact) · subgroups true (features true / size control true / subgroup_id true / lanes linear true) · geometry profile nvidia-ampere",
      "wan22-self-33f s16: 4.00 ms / dispatch · 1.25 ms / 1,000 rows · node 12.00 ms · 3 blocks × 2,640 rows · row window",
      "wan22-self-33f online: not eligible: the shape is not eligible",
      "wan22-self-33f quantization: node 0.250 ms · quantize_q 0.250",
    ]);
  });

  it("puts the online / s16 ratio on the online line, and the wall clock in the status line", () => {
    const online = {
      ...S16,
      rowWindow: undefined,
      reduce: "sg32",
      workgroupStorageBytes: 33792,
    } satisfies AttentionBenchRoute;
    const report = {
      ...REPORT,
      timing: { unit: "wall", gpuTimingEnabled: false },
      shapes: [{ ...RESULT, online, onlineVsS16: 0.7, quantization: undefined }],
    } satisfies AttentionBenchReport;
    const [status, , onlineLine] = attentionBenchSummary(report);
    expect(status).toMatch(/^timing: wall clock · /);
    expect(onlineLine).toBe(
      "wan22-self-33f online: 4.00 ms / dispatch · 1.25 ms / 1,000 rows · node 12.00 ms · online / s16 ×0.700 · 3 windows × 2,640 rows · sg32 · 33,792 B",
    );
  });

  it("adds one line per shape of the candidate: both ratios and the identity", () => {
    const tiled = {
      ...RESULT,
      onlineTiled: {
        eligible: true,
        rows: 2640,
        kernel: {
          stages: ["online", "onlineTiled"],
          keys: ["online:key", "online:key:ktile"],
          workgroups: [
            [1, 1, 1],
            [1, 1, 1],
          ],
          rounds: [
            {
              a: [6, 6],
              b: [5.7, 5.7],
              wallA: [6, 6],
              wallB: [5.7, 5.7],
              clamped: 0,
              ratio: 0.95,
            },
          ],
          ratio: 0.95,
        },
        identity: {
          identical: false,
          words: 10,
          differingWords: 1,
          first: { index: 3, a: 0x3f800000, b: 0x3f800001 },
        },
      },
      tiledVsOnline: 0.95,
      tiledVsOnlineWithProducer: 0.96,
    } satisfies AttentionBenchShapeResult;
    const notRun = {
      ...RESULT,
      onlineTiled: { eligible: false, reason: "the online route does not run here" },
    } satisfies AttentionBenchShapeResult;
    const lines = attentionBenchSummary({ ...REPORT, shapes: [tiled, notRun] });
    expect(lines.slice(-2)).toEqual([
      "wan22-self-33f tiled ÷ online: kernel ×0.950 · kernel + producer ×0.960 · 1 rounds · differs — first differing O word 3: 0x3f800000 vs 0x3f800001",
      "wan22-self-33f tiled ÷ online: not run: the online route does not run here",
    ]);
  });
});

describe("benchProgressText", () => {
  it("shows the shapes done with the runtime's message, or the shape that finished", () => {
    expect(benchProgressText({ kind: "status", message: "s16 qk" }, 0, 3)).toBe(
      "0 / 3 shapes · s16 qk",
    );
    expect(benchProgressText({ kind: "shape", result: RESULT }, 1, 3)).toBe(
      "1 / 3 shapes · wan22-self-33f finished",
    );
  });
});
