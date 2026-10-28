/**
 * The **public distribution repo tables** of all families folded into one table (ADR 0092).
 *
 * The per-family tables (`<FAMILY>_SOURCES`) are canonical; this only makes their union available
 * from the barrel in one place. Values **and keys are the same as on the family side**, so
 * `KARUME_SOURCES["anima"]` and `anima.SOURCES["anima"]` point at the same entry.
 *
 * It serves the side that "walks every published distribution repo" (connectivity smoke tests,
 * cache prefetch, listings); a consumer that uses only one family does better for tree-shaking by
 * looking up that family's table on its subpath (importing the barrel pulls in the configs of 9
 * families).
 *
 * MUST: fold the family tables **as they are** — re-spelling them here would create a shape where
 * the barrel keeps shipping an old revision after the family side is fixed (derivable values held
 * twice). Overlapping keys would silently drop one entry (last one wins), so the absence of
 * duplicates is checked by `tests/sources_test.ts` (keys derive from the HF repo name, so a duplicate
 * occurs only when "two families hold the same repo").
 */

import type { HubRepoRef } from "@karume/hub";

import * as anima from "./anima/config.ts";
import { IRODORI_SOURCES } from "./irodori/config.ts";
import { SBV2_SOURCES } from "./sbv2/config.ts";
import { GEMMA4_SOURCES } from "./gemma/config.ts";
import { SIGLIP2_SOURCES } from "./siglip2/config.ts";
import { DEPTH_ANYTHING_SOURCES } from "./depth-anything/config.ts";
import { BIREFNET_SOURCES } from "./birefnet/config.ts";
import { UMT5_SOURCES } from "./umt5/config.ts";
import { WAN_SOURCES } from "./wan/config.ts";

export const KARUME_SOURCES = {
  ...anima.SOURCES,
  ...IRODORI_SOURCES,
  ...SBV2_SOURCES,
  ...GEMMA4_SOURCES,
  ...SIGLIP2_SOURCES,
  ...DEPTH_ANYTHING_SOURCES,
  ...BIREFNET_SOURCES,
  ...UMT5_SOURCES,
  ...WAN_SOURCES,
} as const satisfies Record<string, HubRepoRef>;
