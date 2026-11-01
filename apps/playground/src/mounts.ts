/**
 * The distributions the dev server mounts: `/models/<name>/` → a directory under `models/` (relative
 * to the repository root — docs/assets-layout.md). One table for both sides: `vite.config.ts` hands
 * it to `distributionsPlugin`, and the page registry names its mounts with {@link MountName} (a type
 * import, so the browser bundle never loads this module).
 *
 * The object form also serves the mount on the HF-compatible route (`/hf/`) under the public repo name
 * of the family's `SOURCES` table, so a page can read it through the hub's CacheStorage. A
 * directory without a public repo (Gemma 4 QAT) or one only reached as a cross-repo reference (umT5)
 * is mounted for the direct route only.
 */

import * as anima from "@karume/models/anima";
import * as birefnet from "@karume/models/birefnet";
import * as depthAnything from "@karume/models/depth-anything";
import * as gemma from "@karume/models/gemma";
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
  // karume-irodori-v4-small at the revision its manifest declares (the same opt-in as `anima`).
  "irodori-v4": {
    directory: "models/karume-irodori-v4-small",
    hfRepo: irodori.SOURCES["irodori-v4-small"].repo,
    aliasReferencedRevisions: true,
  },
  "irodori-v4.1": {
    directory: "models/karume-irodori-v4.1-small",
    hfRepo: irodori.SOURCES["irodori-v4.1-small"].repo,
  },
  sbv2: { directory: "models/karume-sbv2-jvnv", hfRepo: sbv2.SOURCES["sbv2-jvnv"].repo },
  gemma4: { directory: "models/karume-gemma4", hfRepo: gemma.SOURCES["gemma4"].repo },
  // No public repo (gemma.SOURCES holds only `gemma4`): the direct route only.
  "gemma4-qat": "models/karume-gemma4-qat",
  "moss-image": {
    directory: "models/karume-moss-image",
    hfRepo: moss.SOURCES["moss-image"].repo,
  },
} as const satisfies DistributionsPluginOptions["distributions"];

/** The name of a mount (`/models/<name>/`, the key of `/models/index.json`). */
export type MountName = keyof typeof MOUNTS;
