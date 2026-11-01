/**
 * The **public distribution repo tables** of all families folded into one table (ADR 0092).
 *
 * The per-family tables (`SOURCES` of each family's subpath) are canonical; this only makes their union available
 * from the barrel in one place. Values **and keys are the same as on the family side**, so
 * `KARUME_SOURCES["anima"]` and `anima.SOURCES["anima"]` point at the same entry.
 *
 * It serves the side that "walks every published distribution repo" (connectivity smoke tests,
 * cache prefetch, listings); a consumer that uses only one family does better for tree-shaking by
 * looking up that family's table on its subpath (importing the barrel pulls in the config modules of 11
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
import * as irodori from "./irodori/config.ts";
import * as sbv2 from "./sbv2/config.ts";
import * as gemma from "./gemma/config.ts";
import * as siglip2 from "./siglip2/config.ts";
import * as depthAnything from "./depth-anything/config.ts";
import * as birefnet from "./birefnet/config.ts";
import * as umt5 from "./umt5/config.ts";
import * as wan from "./wan/config.ts";
import * as moss from "./moss/config.ts";

export const KARUME_SOURCES = {
  ...anima.SOURCES,
  ...irodori.SOURCES,
  ...sbv2.SOURCES,
  ...gemma.SOURCES,
  ...gemma.QAT_SOURCES,
  ...siglip2.SOURCES,
  ...depthAnything.SOURCES,
  ...birefnet.SOURCES,
  ...umt5.SOURCES,
  ...wan.SOURCES,
  ...moss.SOURCES,
} as const satisfies Record<string, HubRepoRef>;
