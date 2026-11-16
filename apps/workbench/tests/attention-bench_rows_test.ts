import type { AttentionBenchRoute, AttentionBenchShapeResult } from "@karume/runtime/tune";
import { describe, expect, test } from "vite-plus/test";

import {
  benchTableRows,
  candidateRows,
  formatCount,
  formatMs,
  formatRatio,
} from "../src/attention-bench/rows.ts";

const SHAPE = {
  id: "wan22-self-121f",
  heads: 24,
  queries: 27280,
  keys: 27280,
  depth: 128,
  scale: 0.2973017692565918,
};

const stage = (name: string, ms: number) => ({
  stage: name,
  key: `${name}:key`,
  workgroups: [1, 1, 1] as const,
  reps: 2,
  msPerDispatch: ms,
  wallMsPerDispatch: ms + 1,
});

const S16 = {
  eligible: true,
  rows: 3032,
  productionDispatches: 9,
  productionRows: 3032,
  rowWindow: true,
  stages: [stage("qk", 20), stage("stats", 5), stage("pv", 15)],
  msPerDispatch: 40,
  msPer1000Rows: 13.2,
  tops: 50,
  estimatedNodeMs: 360,
} satisfies AttentionBenchRoute;

const RESULT = {
  shape: SHAPE,
  s16: S16,
  online: {
    eligible: true,
    rows: 6528,
    productionDispatches: 5,
    productionRows: 6528,
    reduce: "sg32",
    workgroupStorageBytes: 33792,
    stages: [stage("online", 60)],
    msPerDispatch: 60,
    msPer1000Rows: 9.2,
    tops: 70,
    estimatedNodeMs: 251,
  },
  onlineVsS16: 0.697,
  quantization: {
    stages: [
      stage("quantize_q", 0.5),
      stage("quantize_k", 0.5),
      stage("permute_v", 1),
      stage("quantize_vt", 0.4),
    ],
    msPerNode: 2.4,
  },
} satisfies AttentionBenchShapeResult;

describe("attention bench table rows", () => {
  test("one row per route plus the quantization, with the ratio on the online row", () => {
    const rows = benchTableRows([RESULT]);
    expect(rows.map(({ route, ratio }) => [route, ratio])).toEqual([
      ["s16", undefined],
      ["online", 0.697],
      ["quantization", undefined],
    ]);
    const [s16, online, quantization] = rows;
    expect(s16.split).toEqual("9 blocks × 3,032 rows");
    expect(s16.stages).toEqual("qk 20.00 · stats 5.00 · pv 15.00");
    expect(s16.variant).toEqual("row window");
    expect(online.split).toEqual("5 windows × 6,528 rows");
    expect(online.variant).toEqual("sg32 · 33,792 B");
    expect(online.stages).toEqual(undefined);
    expect(quantization.nodeMs).toEqual(2.4);
    expect(quantization.stages).toEqual(
      "quantize_q 0.500 · quantize_k 0.500 · permute_v 1.00 · quantize_vt 0.400",
    );
  });

  test("a route that does not run shows its reason and no times", () => {
    const { quantization: _, ...unquantized } = RESULT;
    const [, online] = benchTableRows([
      { ...unquantized, online: { eligible: false, reason: "the shape is not eligible" } },
    ]);
    expect(online.note).toEqual("not eligible: the shape is not eligible");
    expect(online.msPerDispatch).toEqual(undefined);
  });

  test("a failed stage is named in the note", () => {
    const [s16] = benchTableRows([
      {
        ...RESULT,
        s16: {
          ...S16,
          stages: [stage("qk", 20), { ...stage("stats", 5), error: "lost" }],
          error: "stats: lost",
        },
      },
    ]);
    expect(s16.note).toEqual("stats: lost");
  });

  test("labels and numbers", () => {
    expect(formatCount(27280)).toEqual("27,280");
    expect([formatMs(12.345), formatMs(0.01234), formatMs(undefined)]).toEqual([
      "12.35",
      "0.0123",
      "—",
    ]);
  });
});

const pairedRound = (a: number, b: number, ratio?: number) => ({
  a: [a, a] as const,
  b: [b, b] as const,
  wallA: [a, a] as const,
  wallB: [b, b] as const,
  clamped: ratio === undefined ? 1 : 0,
  ...(ratio === undefined ? {} : { ratio }),
});

const TILED = {
  ...RESULT,
  onlineTiled: {
    eligible: true,
    rows: 6528,
    kernel: {
      stages: ["online", "onlineTiled"],
      keys: ["online:key", "online:key:ktile"],
      workgroups: [
        [102, 24, 1],
        [102, 24, 1],
      ],
      reps: 2,
      rounds: [pairedRound(60, 57, 0.95), pairedRound(61, 0), pairedRound(62, 58.9, 0.95)],
      aMsPerDispatch: 60,
      bMsPerDispatch: 57,
      ratio: 0.95,
    },
    onlineNodeMs: 250.7,
    tiledNodeMs: 238.2,
    identity: { identical: true, words: 83804160, differingWords: 0, differingScaleWords: 0 },
  },
  tiledVsOnline: 0.95,
  tiledVsOnlineWithProducer: 0.957,
  quantization: {
    ...RESULT.quantization,
    tiledVt: {
      stages: ["quantize_vt", "quantize_vt_tiled"],
      keys: ["quantize_vt:key", "quantize_vt:key:tile64"],
      workgroups: [
        [3072, 1, 1],
        [3072, 1, 1],
      ],
      reps: 4,
      aMsPerDispatch: 0.4,
      bMsPerDispatch: 0.5,
      ratio: 1.25,
    },
  },
} satisfies AttentionBenchShapeResult;

describe("the candidate's table (tiled key axis)", () => {
  test("one row per shape that carries the candidate, with both ratios, the rounds and the identity", () => {
    const rows = candidateRows([RESULT, TILED]);
    expect(rows.length).toEqual(1);
    const [row] = rows;
    expect(row.shape).toEqual("wan22-self-121f");
    expect([row.kernelRatio, row.withProducerRatio]).toEqual([0.95, 0.957]);
    // The round with a clamped pass gives no ratio
    expect(row.rounds).toEqual(2);
    expect(row.identical).toEqual(true);
    expect(row.note).toEqual(undefined);
  });

  test("a shape where the candidate did not run shows why", () => {
    const [row] = candidateRows([
      { ...RESULT, onlineTiled: { eligible: false, reason: "the online route does not run here" } },
    ]);
    expect(row.note).toEqual("not run: the online route does not run here");
    expect(row.identical).toEqual(undefined);
  });

  test("a differing identity names the first differing O word", () => {
    const [row] = candidateRows([
      {
        ...TILED,
        onlineTiled: {
          ...TILED.onlineTiled,
          identity: {
            identical: false,
            words: 10,
            differingWords: 1,
            first: { index: 1234, a: 0x3f800000, b: 0x3f800001 },
          },
        },
      },
    ]);
    expect(row.identical).toEqual(false);
    expect(row.note).toEqual("first differing O word 1,234: 0x3f800000 vs 0x3f800001");
  });

  test("ratios read as ×0.951", () => {
    expect([formatRatio(0.95123), formatRatio(undefined)]).toEqual(["×0.951", "—"]);
  });
});
