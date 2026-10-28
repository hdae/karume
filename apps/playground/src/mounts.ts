/**
 * The distributions the dev server mounts: `/models/<name>/` → a directory under `models/` (relative
 * to the repository root — docs/assets-layout.md). One table for both sides: `vite.config.ts` hands
 * it to `distributionsPlugin`, and the page registry names its mounts with {@link MountName} (a type
 * import, so the browser bundle never loads this module).
 *
 * The object form also serves the mount on the HF-compatible route (`/hf/`) under the public repo name
 * of the family's `*_SOURCES` table, so a page can read it through the hub's CacheStorage. A
 * directory without a public repo (Gemma 4 QAT) or one only reached as a cross-repo reference (umT5)
 * is mounted for the direct route only.
 */

import { ANIMA_SOURCES } from "@karume/models/anima";
import { BIREFNET_SOURCES } from "@karume/models/birefnet";
import { DEPTH_ANYTHING_SOURCES } from "@karume/models/depth-anything";
import { GEMMA4_SOURCES } from "@karume/models/gemma";
import { IRODORI_SOURCES } from "@karume/models/irodori";
import { SBV2_SOURCES } from "@karume/models/sbv2";
import { SIGLIP2_SOURCES } from "@karume/models/siglip2";
import { WAN_SOURCES } from "@karume/models/wan";

import type { DistributionsPluginOptions } from "../vite-plugin-distributions.ts";

export const MOUNTS = {
  wan22: { directory: "models/karume-wan2.2", hfRepo: WAN_SOURCES["wan2.2"].repo },
  // The cross-repo reference of Wan's manifest (its text encoder).
  umt5: "models/karume-umt5-xxl",
  // karume-anima-extra takes its text encoder and VAE from karume-anima at the revision its manifest
  // declares, while this mount is served at its own revision: the opt-in lets the declared revision
  // alias this directory for exactly the files the reference names (vite-plugin-distributions.ts).
  anima: {
    directory: "models/karume-anima",
    hfRepo: ANIMA_SOURCES["anima"].repo,
    aliasReferencedRevisions: true,
  },
  "anima-extra": {
    directory: "models/karume-anima-extra",
    hfRepo: ANIMA_SOURCES["anima-extra"].repo,
  },
  "birefnet-hr": {
    directory: "models/karume-birefnet-hr",
    hfRepo: BIREFNET_SOURCES["birefnet-hr"].repo,
  },
  lucida: { directory: "models/karume-lucida", hfRepo: BIREFNET_SOURCES["lucida"].repo },
  "depth-anything": {
    directory: "models/karume-depth-anything-v2",
    hfRepo: DEPTH_ANYTHING_SOURCES["depth-anything-v2"].repo,
  },
  siglip2: { directory: "models/karume-siglip2", hfRepo: SIGLIP2_SOURCES["siglip2"].repo },
  "irodori-v4": {
    directory: "models/karume-irodori-v4-small",
    hfRepo: IRODORI_SOURCES["irodori-v4-small"].repo,
  },
  "irodori-v4.1": {
    directory: "models/karume-irodori-v4.1-small",
    hfRepo: IRODORI_SOURCES["irodori-v4.1-small"].repo,
  },
  sbv2: { directory: "models/karume-sbv2-jvnv", hfRepo: SBV2_SOURCES["sbv2-jvnv"].repo },
  gemma4: { directory: "models/karume-gemma4", hfRepo: GEMMA4_SOURCES["gemma4"].repo },
  // No public repo (GEMMA4_SOURCES holds only `gemma4`): the direct route only.
  "gemma4-qat": "models/karume-gemma4-qat",
} as const satisfies DistributionsPluginOptions["distributions"];

/** The name of a mount (`/models/<name>/`, the key of `/models/index.json`). */
export type MountName = keyof typeof MOUNTS;
