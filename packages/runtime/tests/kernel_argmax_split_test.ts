/**
 * Checks of the pure-function side of the two-phase argmax split ("two-phase split" in
 * src/kernels/argmax.ts — MTP stage 4-B ③).
 *
 * Two things are checked: ① the path choice is a pure function of the shape, deterministic on both
 * sides of the threshold (the same shape always takes the same path = the same key); ② params
 * reject a call that disagrees with the split shape (passing another `groups` value would make the
 * partial write target and the merge read source diverge, a silent wrong value). Values on the GPU
 * (bit identity, tie-breaks, NaN, all −inf rows) are checked by the argmax check of
 * gpu_ops_test.ts, including two-phase cases.
 */

import { describe, it, test } from "vite-plus/test";
import { assertEquals, assertThrows } from "@std/assert";
import { CodegenError } from "../src/codegen/errors.ts";
import {
  ARGMAX_SPLIT_MIN_DIM,
  ARGMAX_SPLIT_SPAN,
  argmaxSplitGroups,
  argmaxSplitParams,
  argmaxSplitPartialBytes,
  topkOneSplitGroups,
} from "../src/kernels/argmax.ts";

describe("the path choice of the two-phase argmax split is a pure function of the row length", () => {
  it("below the threshold it is the one-dispatch form (0 spans)", () => {
    for (const dim of [1, 256, 4096, ARGMAX_SPLIT_MIN_DIM - 1]) {
      assertEquals(argmaxSplitGroups(dim), 0, `dim=${dim}`);
    }
  });

  it("at or above the threshold there are ceil(dim / span) spans (a partial span counts as one)", () => {
    assertEquals(argmaxSplitGroups(ARGMAX_SPLIT_MIN_DIM), 4);
    assertEquals(argmaxSplitGroups(ARGMAX_SPLIT_MIN_DIM + 1), 5);
    // gemma4 vocabulary length (the drafter's lm_head output)
    assertEquals(argmaxSplitGroups(262144), 262144 / ARGMAX_SPLIT_SPAN);
  });

  it("the scratch is [rows, groups] × 8 bytes (value + index)", () => {
    assertEquals(argmaxSplitPartialBytes(3, 64), 3 * 64 * 8);
  });
});

describe("the params of the two-phase argmax split fail when they do not match the split shape", () => {
  it("the correct form is 4 words (rows, dim, groups, −inf)", () => {
    const params = argmaxSplitParams(2, 262144, 64);
    assertEquals([...params.slice(0, 3)], [2, 262144, 64]);
    assertEquals(params[3], 0xff800000);
  });

  it("fails when groups differs from the pure function's value (for a one-dispatch dim too)", () => {
    assertThrows(
      () => argmaxSplitParams(1, 262144, 63),
      CodegenError,
      "does not match the split shape",
    );
    assertThrows(
      () => argmaxSplitParams(1, 4096, 1),
      CodegenError,
      "does not match the split shape",
    );
    assertThrows(
      () => argmaxSplitParams(1, 262144, 0),
      CodegenError,
      "does not match the split shape",
    );
  });
});

test("the topk k=1 split respects the threshold and the dispatch limit, and leaves huge rows on the existing path", () => {
  assertEquals(topkOneSplitGroups(ARGMAX_SPLIT_MIN_DIM - 1), 0);
  assertEquals(topkOneSplitGroups(ARGMAX_SPLIT_MIN_DIM), 4);
  assertEquals(topkOneSplitGroups(ARGMAX_SPLIT_MIN_DIM + 1), 5);
  assertEquals(topkOneSplitGroups(262144), 64);
  assertEquals(topkOneSplitGroups(65535 * ARGMAX_SPLIT_SPAN), 65535);
  assertEquals(topkOneSplitGroups(65535 * ARGMAX_SPLIT_SPAN + 1), 0);
});
