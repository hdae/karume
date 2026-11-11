// Bit identity of the prefetch form of the 256-wide rms_norm and of the RMS → add fusion (v2,
// perf-ledger K-99) with v1's runtime-trip output loop on the GPU. The cases, the inputs and the
// fault-injection record live in @karume/kernel-checks (checkRmsNormPrefetch), which the
// playground's /kernel-check page runs on the browser's backend too.
import { assertEquals } from "@std/assert";
import { checkRmsNormPrefetch, RMS_NORM_PREFETCH_CASES } from "@karume/kernel-checks";
import { describe, it } from "vite-plus/test";
import { acquireGpu } from "../src/gpu/device.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckPart } from "./helpers/kernel-check-assert.ts";

describe.skipIf(!GPU_AVAILABLE)(
  "the rms_norm prefetch form on a real GPU (perf-ledger K-99)",
  () => {
    it("writes v1's words in the plain kernel and both RMS → add orders, natural and grid-strided", async () => {
      const gpu = await acquireGpu();
      try {
        const parts = await checkRmsNormPrefetch(gpu);
        assertEquals(
          parts.map(({ id, cases }) => [id, cases.length]),
          Object.entries(RMS_NORM_PREFETCH_CASES),
        );
        for (const part of parts) assertCheckPart(part);
      } finally {
        gpu.destroy();
      }
    });
  },
);
