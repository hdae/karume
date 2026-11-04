/**
 * The params of the row kernels reject rows whose in-row loop index would wrap (no GPU).
 *
 * The row kernels walk one row as `i = lane; while (i < dim) { …; i = i + step; }` with a constant
 * step (the workgroup size or the lane count). The last addition reaches `dim − 1 + step`, so when
 * `dim + step` passes 2^32 the u32 index wraps to a value below `dim` and the loop never ends. The
 * loop is inside a row, so the dispatch-count check of grid-stride kernels never sees it. Each
 * builder is pinned at its own edge: `dim = 2^32 − step` passes and one more fails.
 */

import { assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { CodegenError } from "../src/codegen/errors.ts";
import { assertRowLoopBound } from "../src/codegen/params.ts";
import {
  AXIS_REDUCE_WORKGROUP_SIZE,
  axisReduceParams,
  reduceParams,
} from "../src/codegen/reduce.ts";
import {
  ARGMAX_SPLIT_SPAN,
  ARGMAX_WORKGROUP_SIZE,
  argmaxParams,
  argmaxSplitGroups,
  argmaxSplitParams,
} from "../src/kernels/argmax.ts";
import { ATTENTION_STATS_WORKGROUP_SIZE, attentionStatsParams } from "../src/kernels/attention.ts";
import { LAYER_NORM_WORKGROUP_SIZE, layerNormParams } from "../src/kernels/layer-norm.ts";
import {
  QUANTIZE_ROWS_WORKGROUP_SIZE,
  quantizeRowsPaddedParams,
  quantizeRowsParams,
} from "../src/kernels/quantize-rows.ts";
import { rmsNormRopeParams } from "../src/kernels/rms-norm-rope.ts";
import { RMS_NORM_WORKGROUP_SIZE, rmsNormParams } from "../src/kernels/rms-norm.ts";
import { SOFTMAX_WORKGROUP_SIZE, softmaxParams } from "../src/kernels/softmax.ts";
import {
  STATE_PV_KV_LANES,
  STATE_PV_UNROLL,
  STATE_STATS_WORKGROUP_SIZE,
  stateAttentionParams,
  stateStatsParams,
} from "../src/kernels/state-attention.ts";
import { TOPK_WORKGROUP_SIZE, topkParams } from "../src/kernels/topk.ts";

const TWO_32 = 2 ** 32;
const EPS = 1e-6;
const MESSAGE = "in-row step";

describe("assertRowLoopBound", () => {
  it("accepts bound + step = 2^32 and rejects one more, naming the bound", () => {
    assertRowLoopBound("where", { dim: TWO_32 - 256 }, 256);
    assertThrows(
      () => assertRowLoopBound("where", { dim: TWO_32 - 255 }, 256),
      CodegenError,
      "where: dim 4294967041 + in-row step 256 exceeds 2^32",
    );
  });

  it("u32-checks the bound first (NaN would pass the sum comparison)", () => {
    assertThrows(() => assertRowLoopBound("where", { dim: Number.NaN }, 256), CodegenError, "u32");
  });
});

/**
 * One row kernel: the builder for a row of `bound` elements, its step, and the edge values of
 * `bound` it accepts and rejects (the edges differ only where the builder also needs `dim % 4`).
 */
type RowCase = {
  readonly name: string;
  readonly build: (bound: number) => unknown;
  readonly accepts: number;
  readonly rejects: number;
};

/** The edge for a builder with no other constraint on the bound. */
const edge = (name: string, step: number, build: (bound: number) => unknown): RowCase => ({
  name,
  build,
  accepts: TWO_32 - step,
  rejects: TWO_32 - step + 1,
});

const STATE_GEOMETRY = {
  rowsBlock: 1,
  rowOffset: 0,
  chunkRows: 1,
  depth: 4,
  kvRepeat: 1,
  window: 0,
  capacity: 1,
  colCap: 1,
  scale: 0.5,
};

const CASES: readonly RowCase[] = [
  edge("rms_norm", RMS_NORM_WORKGROUP_SIZE, (dim) => rmsNormParams(1, dim, EPS)),
  // head_dim 2 divides every even dim; the rejected edge is made even to keep that check quiet.
  {
    name: "rms_norm_rope",
    build: (dim) => rmsNormRopeParams(1, dim, EPS, 2),
    accepts: TWO_32 - RMS_NORM_WORKGROUP_SIZE,
    rejects: TWO_32 - RMS_NORM_WORKGROUP_SIZE + 2,
  },
  edge("layer_norm (and adaLN)", LAYER_NORM_WORKGROUP_SIZE, (dim) => layerNormParams(1, dim, EPS)),
  edge("softmax", SOFTMAX_WORKGROUP_SIZE, (dim) => softmaxParams(1, dim)),
  edge("safe_softmax", SOFTMAX_WORKGROUP_SIZE, (dim) => softmaxParams(1, dim, true)),
  edge("argmax", ARGMAX_WORKGROUP_SIZE, (dim) => argmaxParams(1, dim)),
  edge("argmax split", ARGMAX_SPLIT_SPAN, (dim) =>
    argmaxSplitParams(1, dim, argmaxSplitGroups(dim)),
  ),
  edge("topk", TOPK_WORKGROUP_SIZE, (dim) => topkParams(1, dim)),
  edge("attention_stats", ATTENTION_STATS_WORKGROUP_SIZE, (dim) => attentionStatsParams(1, dim)),
  // The plain forms need dim % 4 == 0, the padded form dim % 4 != 0.
  {
    name: "quantize_rows",
    build: (dim) => quantizeRowsParams(1, dim),
    accepts: TWO_32 - QUANTIZE_ROWS_WORKGROUP_SIZE,
    rejects: TWO_32 - QUANTIZE_ROWS_WORKGROUP_SIZE + 4,
  },
  {
    name: "quantize_rows padded",
    build: (dim) => quantizeRowsPaddedParams(1, dim),
    accepts: TWO_32 - QUANTIZE_ROWS_WORKGROUP_SIZE - 1,
    rejects: TWO_32 - QUANTIZE_ROWS_WORKGROUP_SIZE + 1,
  },
  edge("reduce", AXIS_REDUCE_WORKGROUP_SIZE, (dim) => reduceParams(1, dim)),
  edge("axis reduce", AXIS_REDUCE_WORKGROUP_SIZE, (axisLen) => axisReduceParams(1, axisLen, 1)),
  // The live columns of a row are at most col_cap.
  edge("attention_state_stats", STATE_STATS_WORKGROUP_SIZE, (colCap) =>
    stateStatsParams(1, 1, 0, colCap, 0),
  ),
  edge("attention_state ③ scan", STATE_PV_UNROLL * STATE_PV_KV_LANES, (colCap) =>
    stateAttentionParams({ ...STATE_GEOMETRY, colCap }),
  ),
];

describe("row-kernel params: dim + in-row step ≤ 2^32", () => {
  for (const { name, build, accepts, rejects } of CASES) {
    it(`${name} accepts ${accepts} and rejects ${rejects}`, () => {
      build(accepts);
      assertThrows(() => build(rejects), CodegenError, MESSAGE);
    });
  }
});
