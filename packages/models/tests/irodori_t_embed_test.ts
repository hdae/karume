// Asset-free structural checks of the host-side `t_embed` (no golden, no GPU). The torch golden
// comparison of the `t_embed` table and of the t schedule reads a golden of the test assets, so it
// lives in packages/e2e/tests/e2e_irodori_t_embed_test.ts.

import { test } from "vite-plus/test";
import { assertEquals, assertThrows } from "@std/assert";
import { timestepEmbedding, timestepFrequencies } from "../src/irodori/host/t-embed.ts";

// ---- Structural checks (no golden needed) ----------------------------------
//
// NOTE: the golden comparison referred to below is in
// packages/e2e/tests/e2e_irodori_t_embed_test.ts.
//
// Both golden comparisons above are `ignore: !AVAILABLE`, so on a machine without assets
// `t_embed` would have no checks at all. anima's counterpart `timestepsProj` has asset-free
// structural tests (`anima_sampler_test.ts`), so this side gets the same coverage.

test("timestepEmbedding: at t=0 the first half is all 1 and the second half all 0 (cos first, sin second)", () => {
  // Both halves have the range [-1,1], so swapping them changes neither the shape nor the
  // statistics (the doc's MUST). A swap gives `[0,0,0,0,1,1,1,1]` and fails here.
  assertEquals([...timestepEmbedding(0, timestepFrequencies(8))], [1, 1, 1, 1, 0, 0, 0, 0]);
});

test("timestepFrequencies: k=0 is exactly 1000 and the rest follow the upstream order of operations", () => {
  // Forgetting to multiply by `FREQ_SCALE` (1000) makes k=0 equal 1 and fails here (measured
  // in the golden comparison: max absolute difference 2.00e+0). k=1 is
  // `1000·exp(f32(-ln(10000)·1)/2)` evaluated in f32 one operation at a time.
  const expected = Math.fround(
    1000 * Math.fround(Math.exp(Math.fround(Math.fround(-Math.log(10000) * 1) / 2))),
  );
  assertEquals([...timestepFrequencies(4)], [1000, expected]);
});

test("timestepFrequencies: accepted widths (positive even numbers)", () => {
  // The width is split into a cos half and a sin half, so an odd width cannot be built (the same
  // condition as `timestepEmbedDim` in `config.ts`).
  for (const dim of [0, -2, 3, 2.5]) {
    assertThrows(() => timestepFrequencies(dim), RangeError, "not a positive even number");
  }
});
