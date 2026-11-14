/**
 * The distributions the dev server mounts: `/models/<name>/` → a directory under `models/` (relative
 * to the repository root — docs/assets-layout.md). One table for both sides: `vite.config.ts` hands
 * it to `distributionsPlugin`, and the page registry names its mounts with {@link MountName} (a type
 * import, so the browser bundle never loads this module).
 *
 * The object form also serves the mount on the HF-compatible route (`/hf/`) under the public repo name
 * of the family's `SOURCES` table, so a page can read it through the hub's CacheStorage. A
 * directory only reached as a cross-repo reference (umT5) is mounted for the direct route only.
 */

import * as anima from "@karume/models/anima";
import * as birefnet from "@karume/models/birefnet";
import * as depthAnything from "@karume/models/depth-anything";
import * as gemma from "@karume/models/gemma";
import * as gemma4QatMobile from "@karume/models/gemma4-qat-mobile";
import * as irodori from "@karume/models/irodori";
import * as moss from "@karume/models/moss";
import * as sbv2 from "@karume/models/sbv2";
import * as siglip2 from "@karume/models/siglip2";
import * as wan from "@karume/models/wan";

import type { DistributionsPluginOptions } from "../vite-plugin-distributions.ts";

export const MOUNTS = {
  wan22: { directory: "models/karume-wan2.2", hfRepo: wan.SOURCES["wan2.2"].repo },
  // The cross-repo reference of Wan's manifest (its text encoder).
  umt5: "models/karume-umt5-xxl",
  // karume-anima-extra takes its text encoder and VAE from karume-anima at the revision its manifest
  // declares, while this mount is served at its own revision: the opt-in lets the declared revision
  // alias this directory for exactly the files the reference names (vite-plugin-distributions.ts).
  anima: {
    directory: "models/karume-anima",
    hfRepo: anima.SOURCES["anima"].repo,
    aliasReferencedRevisions: true,
  },
  "anima-extra": {
    directory: "models/karume-anima-extra",
    hfRepo: anima.SOURCES["anima-extra"].repo,
  },
  "birefnet-hr": {
    directory: "models/karume-birefnet-hr",
    hfRepo: birefnet.SOURCES["birefnet-hr"].repo,
  },
  lucida: { directory: "models/karume-lucida", hfRepo: birefnet.SOURCES["lucida"].repo },
  "depth-anything": {
    directory: "models/karume-depth-anything-v2",
    hfRepo: depthAnything.SOURCES["depth-anything-v2"].repo,
  },
  siglip2: { directory: "models/karume-siglip2", hfRepo: siglip2.SOURCES["siglip2"].repo },
  // karume-irodori-v4.1-small takes the components it shares with v4-small from
  // karume-irodori-v4-small, and karume-irodori-v4.1-small-mf (the MeanFlow model, which carries only
  // its DiT) takes its duration predictor from karume-irodori-v4.1-small and the rest from
  // karume-irodori-v4-small, each at the revision its manifest declares (the same opt-in as `anima`).
  "irodori-v4": {
    directory: "models/karume-irodori-v4-small",
    hfRepo: irodori.SOURCES["irodori-v4-small"].repo,
    aliasReferencedRevisions: true,
  },
  "irodori-v4.1": {
    directory: "models/karume-irodori-v4.1-small",
    hfRepo: irodori.SOURCES["irodori-v4.1-small"].repo,
    aliasReferencedRevisions: true,
  },
  "irodori-v4.1-mf": {
    directory: "models/karume-irodori-v4.1-small-mf",
    hfRepo: irodori.SOURCES["irodori-v4.1-small-mf"].repo,
  },
  sbv2: { directory: "models/karume-sbv2-jvnv", hfRepo: sbv2.SOURCES["sbv2-jvnv"].repo },
  gemma4: { directory: "models/karume-gemma4", hfRepo: gemma.SOURCES["gemma4"].repo },
  // The q4_0 QAT line (read by the regular entry point, with MTP drafters) and the mobile QAT
  // distribution (gemma4QatMobile.Pipeline). MUST keep them on the `/hf/` route: on the direct
  // route every PLE row the host gather misses is an HTTP range request per token.
  "gemma4-qat": {
    directory: "models/karume-gemma4-qat",
    hfRepo: gemma.SOURCES["gemma4-qat"].repo,
  },
  "gemma4-qat-mobile": {
    directory: "models/karume-gemma4-qat-mobile",
    hfRepo: gemma4QatMobile.SOURCES["gemma4-qat-mobile"].repo,
  },
  "moss-image": {
    directory: "models/karume-moss-image",
    hfRepo: moss.SOURCES["moss-image"].repo,
  },
} as const satisfies DistributionsPluginOptions["distributions"];

/** The name of a mount (`/models/<name>/`, the key of `/models/index.json`). */
export type MountName = keyof typeof MOUNTS;
