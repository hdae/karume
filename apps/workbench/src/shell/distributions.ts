/**
 * The browser's `openDistribution`: a mount read through the dev server the page came from
 * (`@karume/local-distributions/plugin`) — over `/hf/` into the hub's CacheStorage (`cached`), so a
 * second run does not move the weights again, or with plain fetch from `/models/<mount>/`
 * (`direct`). Every URL is resolved against `base`, the page's `document.baseURI`.
 */

import { type DistributionSource, MANIFEST_FILENAME, parseManifest } from "@karume/hub";
import {
  crossRepoHolders,
  distributionSource,
  fetchDistributionIndex,
  hfRepoRef,
  MountUnavailableError,
} from "@karume/local-distributions/browser";

import type { DistributionRead, OpenDistribution } from "../entry.ts";

/** Lower-case hex sha256 of `bytes` (WebCrypto). */
const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> => {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(digest, (byte) => byte.toString(16).padStart(2, "0")).join("");
};

export const browserDistributions =
  (base: string): OpenDistribution =>
  async (mount, source) => {
    const index = await fetchDistributionIndex(base);
    const status = index[mount];
    if (!status.available) {
      throw new MountUnavailableError(mount, `its mirror ${status.directory} has no karume.json`);
    }
    const path = `models/${mount}/${MANIFEST_FILENAME}`;
    const response = await fetch(new URL(path, base).href);
    if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    const manifestSha256 = await sha256Hex(bytes);
    const facts = (read: DistributionRead) => ({
      directory: status.directory,
      manifestSha256,
      read,
    });
    if (source === "cached") {
      if (status.hf === undefined) {
        throw new MountUnavailableError(
          mount,
          "the dev server serves it on no HF-compatible route (/hf/), so it cannot be read cached; read it direct",
        );
      }
      return { ref: hfRepoRef(status.hf, base), facts: facts("cached") };
    }
    const manifest = parseManifest(new TextDecoder().decode(bytes));
    const crossRepo: Record<string, DistributionSource> = Object.fromEntries(
      Object.entries(crossRepoHolders(manifest, index)).map(([repo, holder]) => [
        repo,
        distributionSource(holder, base),
      ]),
    );
    return { ref: distributionSource(mount, base, crossRepo), facts: facts("direct") };
  };
