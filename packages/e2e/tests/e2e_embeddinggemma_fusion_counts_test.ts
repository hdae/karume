/**
 * Fusion hit counts of the EmbeddingGemma-300m series graph (no GPU — `planFusions` on part 0 only;
 * the shared planning side and the rationale live in `runtime/tests/helpers/fusion-counts.ts`).
 *
 * MUST: the asset is `outputs/series/embeddinggemma-300m/` (untracked local asset); where it is
 * missing the case SKIPs explicitly with a reason (ADR 0005). EmbeddingGemma has no published
 * distribution yet, so this is an implementation-time series check (ADR 0139 decision 6 —
 * KARUME_SERIES=1).
 *
 * NOTE: EmbeddingGemma gets a published distribution (ADR 0139 Rulings D2, staging step 7); this
 * case then reads part 0 from `models/<repo>/` through the manifest and returns to the default lane.
 */

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import type { FusionCounts } from "../../runtime/src/runtime/fusion.ts";
import { readContainerGraph } from "../../runtime/tests/helpers/container-graph.ts";
import { SERIES_CHECKS, SERIES_CHECKS_OPT_IN } from "../../models/tests/helpers/series-checks.ts";
import { exists, fusionCounts, NONE, seriesSource } from "./helpers/fusion-counts.ts";

// A thunk: the series directory is not listed at all without KARUME_SERIES=1.
const gemmaModel = () => seriesSource("embeddinggemma-300m");

const GEMMA_AVAILABLE = SERIES_CHECKS && (await exists(gemmaModel().part0));
if (SERIES_CHECKS && !GEMMA_AVAILABLE) {
  console.warn(
    `[karume] ${gemmaModel().part0.pathname} is missing, so the EmbeddingGemma fusion hit counts are SKIPPED`,
  );
}

/**
 * EmbeddingGemma-300m (24 layers × 2 for q / k = 48 chains). Unlike Anima the head width is 256,
 * and the cos / sin tables are cut down to T by `sym_prefix_slice` per θ family (local 1e4 /
 * global 1e6). Their first two uses fall into gaps in a chain, so if the matcher cannot step over
 * an in-window passthrough, 2 of the 48 silently drop out — **the hit count 48 is a check that
 * covers that too**.
 *
 * identityExpand is **0**. In an asset that keeps SDPA (ADR 0023 revision — attention with a mask)
 * the identity expands that came from the decomposition disappear from the IR altogether (the
 * same mechanism as decision 6; with the decomposed asset it was 96).
 */
test.skipIf(!GEMMA_AVAILABLE)(
  `the real EmbeddingGemma asset matches rope 48 (head width 256, in-window passthrough included)${SERIES_CHECKS_OPT_IN}`,
  async () => {
    const source = gemmaModel();
    const graph = await readContainerGraph(source.part0, source.graph);
    const expected: FusionCounts = { ...NONE, rope: 48 };
    // The hit count does not depend on T (two points inside Tmax = 512).
    for (const sequence of [12, 318]) {
      assertEquals(
        fusionCounts(graph, { input_ids: [1, sequence], pool_mask: [1, sequence] }),
        expected,
        `T=${sequence}`,
      );
    }
  },
);
