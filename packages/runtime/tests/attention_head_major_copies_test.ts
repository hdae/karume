/**
 * The head-major copies a fused attention makes for the permutes it absorbed (perf-ledger F-3 A4 /
 * F-8 — `attentionHeadMajorCopies` of src/runtime/recipe-builders/attention.ts, shared by the recipe
 * and the estimate). No GPU: which operand each route copies, and the bytes, by hand.
 */

import { describe, it } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import type { AttentionRoute } from "../src/runtime/attention-route.ts";
import type { AttentionTokenMajor } from "../src/runtime/fusion.ts";
import {
  attentionHeadMajorCopies,
  attentionHeadMajorShapes,
} from "../src/runtime/recipe-builders/attention.ts";

const ALL: AttentionTokenMajor = { q: true, k: true, v: true, o: true };
/** B 2, S 5, N 6, H 3, D 8: q / O 2·5·3·8·4 = 960 B, k / v 2·6·3·8·4 = 1,152 B. */
const SHAPES = {
  inputs: [
    [2, 5, 3, 8],
    [2, 6, 3, 8],
    [2, 6, 3, 8],
  ],
  output: [2, 5, 3, 8],
} as const;

const ONLINE: AttentionRoute = { qk: "i8a8", pv: "i8a8", score: "none", online: "wg" };
const A8: AttentionRoute = { qk: "i8a8", pv: "i8a8", score: "f16" };
const DENSE: AttentionRoute = { qk: "dense", pv: "dense", score: "f32" };
const QK_DENSE: AttentionRoute = { qk: "dense", pv: "i8a8", score: "f32" };
const PV_DENSE: AttentionRoute = { qk: "i8a8", pv: "dense", score: "f32" };

describe("the head-major copies of an attention's absorbed permutes", () => {
  it("the online route copies nothing: the quantisation reads q / k / v token-major and the kernel stores O token-major", () => {
    assertEquals(attentionHeadMajorCopies(ALL, ONLINE, SHAPES), {
      q: undefined,
      k: undefined,
      v: undefined,
      o: undefined,
    });
  });

  it("the three a8 dispatches copy only O (③PV stores head-major)", () => {
    assertEquals(attentionHeadMajorCopies(ALL, A8, SHAPES), {
      q: undefined,
      k: undefined,
      v: undefined,
      o: 960,
    });
  });

  it("a dense route copies every absorbed operand, and a mixed route the operands of its dense stage", () => {
    assertEquals(attentionHeadMajorCopies(ALL, DENSE, SHAPES), {
      q: 960,
      k: 1152,
      v: 1152,
      o: 960,
    });
    assertEquals(attentionHeadMajorCopies(ALL, QK_DENSE, SHAPES), {
      q: 960,
      k: 1152,
      v: undefined,
      o: 960,
    });
    assertEquals(attentionHeadMajorCopies(ALL, PV_DENSE, SHAPES), {
      q: undefined,
      k: undefined,
      v: 1152,
      o: 960,
    });
  });

  it("only absorbed operands are copied, and nothing without the annotation", () => {
    assertEquals(
      attentionHeadMajorCopies({ q: false, k: false, v: true, o: true }, DENSE, SHAPES),
      { q: undefined, k: undefined, v: 1152, o: 960 },
    );
    assertEquals(attentionHeadMajorCopies(undefined, DENSE, SHAPES), {
      q: undefined,
      k: undefined,
      v: undefined,
      o: undefined,
    });
  });

  it("the head-major shapes swap the middle axes of the absorbed operands only", () => {
    assertEquals(
      attentionHeadMajorShapes(SHAPES.inputs, { q: true, k: false, v: true, o: false }),
      [
        [2, 3, 5, 8],
        [2, 6, 3, 8],
        [2, 3, 6, 8],
      ],
    );
  });
});
