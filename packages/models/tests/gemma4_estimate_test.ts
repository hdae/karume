// The wiring of the generation settings (`gemma4EstimateGenerations`) that the gemma4 estimate
// (`Pipeline.estimateSessionMemory`) passes to `estimateGraphMemory`. Uses no GPU.
//
// On the real gemma4 E2B assets the bucket shapes do not dominate the peak, so the check on `peakAccountedBytes`
// (`e2e_gemma4_drafter_test.ts` ②) does not break even if the bucket wiring is removed. The wiring is pinned here directly.

import { assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { gemma4EstimateGenerations } from "../src/gemma/pipeline.ts";
import { fakeSession, programOf } from "./helpers/generation-fake.ts";

/** A pipeline with chunk length 16 and bucket ladder [4, 8] (capacity symbol `C`, row symbol `R`). */
const wiring = () =>
  programOf(fakeSession(), { chunkLength: 16, chunkBuckets: [4, 8], capacity: 64 });

describe("the generation settings the gemma4 estimate passes to the main model", () => {
  it("passes the pipeline's chunkBuckets that are below the estimated chunkLength", () => {
    const at = (chunkLength: number) =>
      gemma4EstimateGenerations(wiring(), { capacity: 64, chunkLength }).target.chunkBuckets;
    assertEquals(
      at(16),
      [4, 8],
      "at the same chunk length as the pipeline the ladder is passed as is",
    );
    assertEquals(
      at(8),
      [4],
      "a bucket equal to the chunk length is not in the accepted set (below the chunk length)",
    );
    assertEquals(at(4), [], "an empty accepted set passes no bucket");
  });

  it("even with an assumed chunkLength larger than the pipeline's, the buckets passed stay the pipeline's ladder", () => {
    const { target } = gemma4EstimateGenerations(wiring(), { capacity: 64, chunkLength: 32 });
    assertEquals(target.chunkBuckets, [4, 8]);
  });

  it("passes the estimated chunk length and capacity as is", () => {
    const { target } = gemma4EstimateGenerations(wiring(), { capacity: 48, chunkLength: 16 });
    assertEquals(target.chunkLength, 16);
    assertEquals(target.bindings, { C: 48 });
  });

  it("adds no extra scenario for a pipeline without a drafter", () => {
    const { target } = gemma4EstimateGenerations(wiring(), { capacity: 64, chunkLength: 16 });
    assertEquals(target.scenarios, undefined);
  });

  it("with speculation, adds verify with k+1 rows and physical rows rounded up to a bucket", () => {
    const { target } = gemma4EstimateGenerations(wiring(), {
      capacity: 64,
      chunkLength: 16,
      verifyQueryLength: 3,
    });
    assertEquals(target.scenarios, [{ name: "verify", chunkLength: 4, bindings: { R: 3 } }]);
  });

  it("with an assumed chunkLength larger than the pipeline's and k+1 beyond the ladder's largest bucket, verify runs at the assumed chunk length", () => {
    // Rounding with the pipeline's chunk length 16 would pass 16, which is not in the allowed set {1, 4, 8, 32} of
    // the estimated context (chunk length 32, ladder [4, 8]), and the estimate would become an ExecutionError.
    const { target } = gemma4EstimateGenerations(wiring(), {
      capacity: 64,
      chunkLength: 32,
      verifyQueryLength: 10,
    });
    assertEquals(target.scenarios, [{ name: "verify", chunkLength: 32, bindings: { R: 10 } }]);
  });
});

describe("the generation settings the gemma4 estimate passes to the drafter (the borrowing context)", () => {
  it("chunk length 1, the same capacity as the lender, and no buckets", () => {
    const { borrower } = gemma4EstimateGenerations(wiring(), {
      capacity: 48,
      chunkLength: 16,
      verifyQueryLength: 4,
    });
    assertEquals(borrower, { chunkLength: 1, bindings: { C: 48 } });
  });
});
