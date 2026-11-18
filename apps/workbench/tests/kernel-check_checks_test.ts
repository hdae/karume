import { describe, expect, it } from "vite-plus/test";

import { KERNEL_CHECKS } from "../src/kernel-check/checks.ts";
import { KERNEL_CHECK_IDS } from "../src/kernel-check/verdict.ts";

describe("the kernel-check entry's checks", () => {
  it("has one check per id, in the order the verdicts and the record list them", () => {
    expect(KERNEL_CHECKS.map(({ id }) => id)).toEqual([...KERNEL_CHECK_IDS]);
  });

  it("needs subgroups only for the two subgroup32 reductions", () => {
    expect(
      Object.fromEntries(KERNEL_CHECKS.map(({ id, needsSubgroups }) => [id, needsSubgroups])),
    ).toEqual({
      "gemv-subgroup32": true,
      "rms-subgroup32": true,
      "rms-norm-prefetch": false,
      "state-attention-rewrites": false,
      "gemv-integer": false,
      "gemv-planes": false,
      "gemv-parallel-rows": false,
      "state-kv16": false,
      "score-f16-store": false,
      "state-bidirectional": false,
      "expert-linear": false,
      "fusion-twins": false,
      "attention-online-tiled": false,
    });
  });
});
