// f16 storage of a full state slot (ADR 0143), on a real GPU. The cases, the host rounding and the
// mutants live in @karume/kernel-checks, one implementation with the playground's /kernel-check page
// (the same check on a browser's backend); this file asserts the counts it returns.

import { describe, it } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import {
  checkStateKv16,
  checkStateKv16Mutants,
  STATE_KV16_CASES,
  STATE_KV16_MUTANTS,
} from "@karume/kernel-checks";
import { acquireGpu } from "../src/gpu/device.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckPart } from "./helpers/kernel-check-assert.ts";

describe.skipIf(!GPU_AVAILABLE)("f16 storage of a full state slot (real GPU)", () => {
  it("every reader equals its f32 form over a rounded slot, and the append packs the host's rounding", async () => {
    const gpu = await acquireGpu();
    try {
      const parts = await checkStateKv16(gpu);
      assertEquals(
        Object.fromEntries(parts.map((part) => [part.id, part.cases.length])),
        STATE_KV16_CASES,
      );
      for (const part of parts) assertCheckPart(part);
    } finally {
      gpu.destroy();
    }
  });

  it("the mutants mismatch (the word comparison is not vacuous)", async () => {
    const gpu = await acquireGpu();
    try {
      const mutants = await checkStateKv16Mutants(gpu);
      assertEquals(mutants.length, STATE_KV16_MUTANTS);
      for (const { label, differingWords, words } of mutants) {
        assert(differingWords > 0, `${label}: matched the correct form word for word`);
        console.log(`mutant '${label}': ${differingWords} of ${words} words differ`);
      }
    } finally {
      gpu.destroy();
    }
  });
});
