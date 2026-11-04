/**
 * Fusion hit counts of the MiniCPM5-1B decode series graph (no GPU — `planFusions` on part 0 only;
 * the shared planning side and the rationale live in `runtime/tests/helpers/fusion-counts.ts`).
 *
 * MUST: the asset is `outputs/series/minicpm5-1b-decode/` (untracked local asset); where it is
 * missing the case SKIPs explicitly with a reason (ADR 0005). MiniCPM5 has no distribution, so this
 * is an implementation-time series check (ADR 0139 decision 6 — KARUME_SERIES=1).
 */

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import type { FusionCounts } from "../../runtime/src/runtime/fusion.ts";
import { readContainerGraph } from "../../runtime/tests/helpers/container-graph.ts";
import { SERIES_CHECKS, SERIES_CHECKS_OPT_IN } from "../../models/tests/helpers/series-checks.ts";
import { decodeFusionCounts, exists, NONE, seriesSource } from "./helpers/fusion-counts.ts";

// A thunk: the series directory is not listed at all without KARUME_SERIES=1.
const minicpm5DecodeModel = () => seriesSource("minicpm5-1b-decode");

const MINICPM5_DECODE_AVAILABLE = SERIES_CHECKS && (await exists(minicpm5DecodeModel().part0));
if (SERIES_CHECKS && !MINICPM5_DECODE_AVAILABLE) {
  console.warn(
    `[karume] ${minicpm5DecodeModel().part0.pathname} is missing, so the MiniCPM5 decode fusion` +
      " hit counts are SKIPped",
  );
}

/**
 * MiniCPM5-1B decode (24 layers × q/k = 48 chains, no KV sharing). Unlike gemma4, **every chain
 * matches independently of M** — this is also the control for the gemma4 side, showing how far
 * the same table-lookup RoPE can diverge (if only one of the two moves, the pair tells which
 * emitted form changed). silu 24 is the gate MLP of the 24 layers.
 */
test.skipIf(!MINICPM5_DECODE_AVAILABLE)(
  `the real-asset MiniCPM5 decode catches rope 48 / silu 24 (independent of M)${SERIES_CHECKS_OPT_IN}`,
  async () => {
    const source = minicpm5DecodeModel();
    const graph = await readContainerGraph(source.part0, source.graph);
    const expected: FusionCounts = { ...NONE, rope: 48, silu: 24 };
    for (const rows of [1, 32]) {
      assertEquals(decodeFusionCounts(graph, rows), expected, `M=${rows}`);
    }
  },
);
