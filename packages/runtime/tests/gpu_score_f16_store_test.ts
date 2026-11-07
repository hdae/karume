// The rounding of the f16 attention scores on a real GPU. The cases and the host rounding live in
// @karume/kernel-checks, one implementation with the playground's /kernel-check page (the same check
// on a browser's backend); this file asserts the counts it returns.

import { describe, it } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import {
  checkScoreF16Store,
  SCORE_F16_CASES,
  scoreF16StoreSwapMutant,
} from "@karume/kernel-checks";
import { acquireGpu } from "../src/gpu/device.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckPart } from "./helpers/kernel-check-assert.ts";

describe.skipIf(!GPU_AVAILABLE)("the store of f16 attention scores (real GPU)", () => {
  it("stores what the host's rounding stores", async () => {
    const gpu = await acquireGpu();
    try {
      const part = await checkScoreF16Store(gpu);
      assertEquals(part.cases.length, SCORE_F16_CASES["score-f16-store"]);
      assertCheckPart(part);
    } finally {
      gpu.destroy();
    }
  });

  it("the comparison can fail: a store that swaps the halves mismatches", async () => {
    const gpu = await acquireGpu();
    try {
      assert((await scoreF16StoreSwapMutant(gpu)) > 0);
    } finally {
      gpu.destroy();
    }
  });
});
