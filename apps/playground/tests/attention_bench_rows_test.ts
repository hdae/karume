import { assertEquals } from "@std/assert";
import type { AttentionBenchRoute, AttentionBenchShapeResult } from "@karume/runtime/tune";
import { describe, test } from "vite-plus/test";

import {
  benchTableRows,
  candidateRows,
  formatCount,
  formatMs,
  formatRatio,
  shapeLabel,
} from "../src/models/attention-bench/rows.ts";

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
    assertEquals(
      rows.map(({ route, ratio }) => [route, ratio]),
      [
        ["s16", undefined],
        ["online", 0.697],
        ["quantization", undefined],
      ],
    );
    const [s16, online, quantization] = rows;
    assertEquals(s16.split, "9 blocks × 3,032 rows");
    assertEquals(s16.stages, "qk 20.00 · stats 5.00 · pv 15.00");
    assertEquals(s16.variant, "row window");
    assertEquals(online.split, "5 windows × 6,528 rows");
    assertEquals(online.variant, "sg32 · 33,792 B");
    assertEquals(online.stages, undefined);
    assertEquals(quantization.nodeMs, 2.4);
    assertEquals(
      quantization.stages,
      "quantize_q 0.500 · quantize_k 0.500 · permute_v 1.00 · quantize_vt 0.400",
    );
  });

  test("a route that does not run shows its reason and no times", () => {
    const { quantization: _, ...unquantized } = RESULT;
    const [, online] = benchTableRows([
      { ...unquantized, online: { eligible: false, reason: "the shape is not eligible" } },
    ]);
    assertEquals(online.note, "not eligible: the shape is not eligible");
    assertEquals(online.msPerDispatch, undefined);
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
    assertEquals(s16.note, "stats: lost");
  });

  test("labels and numbers", () => {
    assertEquals(shapeLabel(SHAPE), "wan22-self-121f (24 × 27,280 → 27,280, D 128)");
    assertEquals(formatCount(27280), "27,280");
    assertEquals(
      [formatMs(12.345), formatMs(0.01234), formatMs(undefined)],
      ["12.35", "0.0123", "—"],
    );
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
  test("one row per shape that carries the candidate, with both ratios, the producer pair and the identity", () => {
    const rows = candidateRows([RESULT, TILED]);
    assertEquals(rows.length, 1);
    const [row] = rows;
    assertEquals(row.shape, "wan22-self-121f");
    assertEquals([row.kernelRatio, row.withProducerRatio], [0.95, 0.957]);
    assertEquals([row.onlineMs, row.tiledMs], [60, 57]);
    assertEquals(row.producer, "today 0.400 · tiled 0.500 (×1.250)");
    // The round with a clamped pass gives no ratio
    assertEquals(row.rounds, 2);
    assertEquals(row.identical, true);
    assertEquals(row.note, undefined);
  });

  test("a shape where the candidate did not run shows why", () => {
    const [row] = candidateRows([
      { ...RESULT, onlineTiled: { eligible: false, reason: "the online route does not run here" } },
    ]);
    assertEquals(row.note, "not run: the online route does not run here");
    assertEquals(row.identical, undefined);
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
    assertEquals(row.identical, false);
    assertEquals(row.note, "first differing O word 1,234: 0x3f800000 vs 0x3f800001");
  });

  test("ratios read as ×0.951", () => {
    assertEquals([formatRatio(0.95123), formatRatio(undefined)], ["×0.951", "—"]);
  });
});
