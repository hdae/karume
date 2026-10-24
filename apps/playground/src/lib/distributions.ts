/**
 * Reading the distributions the dev server mounts under `/models/<name>/`
 * (`vite-plugin-distributions.ts`). URLs are relative, so the page works behind any base path.
 *
 * Two ways to read one: {@link distributionSource} (`localDirectory` over plain `fetch`, nothing
 * cached) and {@link hfRepoRef} (the HF-compatible route `/hf/`, read by the hub's HF source into
 * its CacheStorage).
 */

import { type DistributionSource, type HubRepoRef, localDirectory } from "@karume/hub";

/** The HF-compatible coordinates of a mount (the revision follows its `karume.json`). */
export type HfCoordinates = { readonly repo: string; readonly revision: string };

/** One mount as `/models/index.json` reports it. */
export type DistributionStatus = {
  readonly directory: string;
  readonly available: boolean;
  /** Present when the mount is also served on the HF-compatible route. */
  readonly hf?: HfCoordinates;
};
export type DistributionIndex = Readonly<Record<string, DistributionStatus>>;

const isHfCoordinates = (value: unknown): value is HfCoordinates =>
  typeof value === "object" &&
  value !== null &&
  "repo" in value &&
  typeof value.repo === "string" &&
  "revision" in value &&
  typeof value.revision === "string" &&
  /^[0-9a-f]{40}$/.test(value.revision);

const isStatus = (value: unknown): value is DistributionStatus =>
  typeof value === "object" &&
  value !== null &&
  "directory" in value &&
  typeof value.directory === "string" &&
  "available" in value &&
  typeof value.available === "boolean" &&
  (!("hf" in value) || isHfCoordinates(value.hf));

/** Fetches `/models/index.json` (which mounts exist and whether their directory is there). */
export const fetchDistributionIndex = async (signal?: AbortSignal): Promise<DistributionIndex> => {
  const response = await fetch("models/index.json", signal === undefined ? {} : { signal });
  if (!response.ok) throw new Error(`models/index.json: HTTP ${response.status}`);
  const body: unknown = await response.json();
  if (typeof body !== "object" || body === null) {
    throw new Error("models/index.json is not an object");
  }
  const index: Record<string, DistributionStatus> = {};
  for (const [name, status] of Object.entries(body)) {
    if (!isStatus(status)) throw new Error(`models/index.json: malformed entry '${name}'`);
    index[name] = status;
  }
  return index;
};

const withSignal = (signal: AbortSignal | undefined): RequestInit =>
  signal === undefined ? {} : { signal };

/**
 * A `@karume/hub` source over the mount `name`: whole files with `fetch`, range reads with
 * `Range` (a `206` with exactly `length` bytes, anything else fails). `crossRepo` maps the repos a
 * manifest references (umT5 for Wan) to other mounts.
 */
export const distributionSource = (
  name: string,
  crossRepo: Readonly<Record<string, DistributionSource>> = {},
): DistributionSource =>
  localDirectory(
    {
      readFile: async (path, { signal }) => {
        const response = await fetch(`models/${name}/${path}`, withSignal(signal));
        if (!response.ok) throw new Error(`models/${name}/${path}: HTTP ${response.status}`);
        return new Uint8Array(await response.arrayBuffer());
      },
      readFileRange: async (path, offset, length, { signal }) => {
        const response = await fetch(`models/${name}/${path}`, {
          ...withSignal(signal),
          headers: { Range: `bytes=${offset}-${offset + length - 1}` },
        });
        if (response.status !== 206) {
          throw new Error(
            `models/${name}/${path}: range ${offset}+${length} answered HTTP ${response.status}, expected 206`,
          );
        }
        const bytes = new Uint8Array(await response.arrayBuffer());
        if (bytes.byteLength !== length) {
          throw new Error(
            `models/${name}/${path}: range ${offset}+${length} returned ${bytes.byteLength} bytes`,
          );
        }
        return bytes;
      },
    },
    { label: `/models/${name}/`, crossRepo },
  );

/**
 * The `HubRepoRef` of a mount on the HF-compatible route: `hubUrl` is the page's `/hf` and the
 * revision is passed explicitly (a commit-shaped revision issues no resolution request).
 */
export const hfRepoRef = (hf: HfCoordinates, base: string): HubRepoRef => ({
  repo: hf.repo,
  revision: hf.revision,
  hubUrl: new URL("hf", base).href,
});
