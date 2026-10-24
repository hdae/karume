import { assertEquals } from "@std/assert";
import type { AttentionBenchRoute, AttentionBenchShapeResult } from "@karume/runtime/tune";
import { describe, test } from "vite-plus/test";

import {
  benchTableRows,
  formatCount,
  formatMs,
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
