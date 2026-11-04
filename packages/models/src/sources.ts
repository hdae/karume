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

// The spread sits in a pure-annotated arrow call: a bundler cannot prove an object spread free of
// side effects (a getter could run), so a bare spread kept every family's table in any barrel
// import that does not read this (ADR 0037 §4 addendum 2026-11-04). The type is spelled from the
// family tables so that JSR's generated declaration carries them (the published 0.16.0 declaration
// of the bare spread was `{}`); a table added to the spread MUST be added to the type as well.
export const KARUME_SOURCES: typeof anima.SOURCES &
  typeof irodori.SOURCES &
  typeof sbv2.SOURCES &
  typeof gemma.SOURCES &
  typeof gemma.QAT_MOBILE_SOURCES &
  typeof siglip2.SOURCES &
  typeof depthAnything.SOURCES &
  typeof birefnet.SOURCES &
  typeof umt5.SOURCES &
  typeof wan.SOURCES &
  typeof moss.SOURCES = /* @__PURE__ */ (() =>
  ({
    ...anima.SOURCES,
    ...irodori.SOURCES,
    ...sbv2.SOURCES,
    ...gemma.SOURCES,
    ...gemma.QAT_MOBILE_SOURCES,
    ...siglip2.SOURCES,
    ...depthAnything.SOURCES,
    ...birefnet.SOURCES,
    ...umt5.SOURCES,
    ...wan.SOURCES,
    ...moss.SOURCES,
  }) as const satisfies Record<string, HubRepoRef>)();
