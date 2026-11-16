/**
 * The browser's `openDistribution`: a mount read through the dev server the page came from
 * (`openMountRoute` of `@karume/local-distributions/browser`) — over `/hf/` into the hub's
 * CacheStorage (`cached`), so a second run does not move the weights again, or with plain fetch
 * from `/models/<mount>/` (`direct`) — and the facts its record carries: the sha256 of the manifest
 * it read and how it was read. Every URL is resolved against `base`, the page's `document.baseURI`.
 */

import { openMountRoute } from "@karume/local-distributions/browser";

import type { OpenDistribution } from "../entry.ts";
import { sha256Hex } from "../sha256.ts";

export const browserDistributions =
  (base: string): OpenDistribution =>
  async (mount, source) => {
    const opened = await openMountRoute(base, mount, source);
    return {
      ref: opened.ref,
      facts: {
        directory: opened.directory,
        manifestSha256: await sha256Hex(opened.manifestBytes),
        read: source,
      },
    };
  };
