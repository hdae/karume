/**
 * Reading the distributions the dev server mounts under `/models/<name>/`
 * (`vite-plugin-distributions.ts`). URLs are relative, so the page works behind any base path.
 *
 * Two ways to read one: {@link distributionSource} (`localDirectory` over plain `fetch`, nothing
 * cached) and {@link hfRepoRef} (the HF-compatible route `/hf/`, read by the hub's HF source into
 * its CacheStorage). A page picks one with a {@link PipelineSource} and hands {@link pretrainedRef}
 * to the family's `fromPretrained`.
 */

import {
  type DistributionSource,
  type HubRepoRef,
  localDirectory,
  type Manifest,
  resolveSelection,
} from "@karume/hub";

import type { MountName } from "@/mounts";

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

/** What `/models/index.json` says about one mount (`undefined` while loading or for an unknown name). */
export const mountStatus = (
  index: DistributionIndex | undefined,
  name: MountName,
): DistributionStatus | undefined => index?.[name];

/** Where a page reads a distribution: the hub's cache over `/hf/`, or plain fetch from `/models/`. */
export type SourceKind = "cached" | "direct";

/**
 * The source a pipeline is built from: `cached` = the HF-compatible route through the hub's
 * CacheStorage (later runs read the weights from the browser's disk); `direct` = `localDirectory`
 * over plain `fetch`, nothing cached. MUST be stable (memoized — {@link usePipelineSource} does it):
 * a page that rebuilds its pipeline when the source changes would rebuild on every render.
 */
export type PipelineSource =
  | { readonly kind: "direct" }
  | {
      readonly kind: "cached";
      readonly ref: HubRepoRef;
      /** Bumped when the cache is cleared, so the pipeline (and the readers it holds) is rebuilt. */
      readonly cacheGeneration: number;
    };

/**
 * The first argument of a family's `fromPretrained` for `source`: the `HubRepoRef` of the cached
 * route as is (it reaches cross-repo references on the same route), or the mount `name` over plain
 * fetch with `crossRepo` mapping each referenced repo to the mount that holds it (the direct route
 * has no revision to follow — `{ "hdae/karume-anima": "anima" }` for the Anima fine-tunes).
 */
export const pretrainedRef = (
  source: PipelineSource,
  name: MountName,
  crossRepo: Readonly<Record<string, MountName>> = {},
): HubRepoRef | DistributionSource =>
  source.kind === "cached"
    ? source.ref
    : distributionSource(
        name,
        Object.fromEntries(
          Object.entries(crossRepo).map(([repo, mount]) => [repo, distributionSource(mount)]),
        ),
      );

/**
 * The mount that holds each repository a page's manifests may reference across repositories, keyed
 * by the public repo name (the name the mount is served under on `/hf/` — a family's `*_SOURCES`
 * entry): `{ "hdae/karume-anima": "anima" }` for the Anima fine-tunes,
 * `{ "hdae/karume-irodori-v4-small": "irodori-v4" }` for Irodori v4.1-small.
 */
export type RepoHolders<M extends MountName = MountName> = Readonly<Record<string, M>>;

/**
 * The `crossRepo` map of the direct route ({@link pretrainedRef}) for one selection: every
 * repository its containers and assets reference, mapped to the mount that holds it — empty for a
 * selection that references none (a self-contained distribution). Throws when a referenced
 * repository has no mount (the direct route would fail on the first part it cannot read).
 */
export const crossRepoMounts = <M extends MountName>(
  manifest: Manifest,
  model: string,
  quant: string,
  holders: RepoHolders<M>,
): Readonly<Record<string, M>> => {
  const selection = resolveSelection(manifest, { model, quant });
  const files = [
    ...Object.values(selection.containers).flatMap(({ parts }) => parts),
    ...Object.values(selection.assets),
  ];
  const map: Record<string, M> = {};
  for (const { repo } of files) {
    if (repo === undefined || Object.hasOwn(map, repo)) continue;
    if (!Object.hasOwn(holders, repo)) {
      throw new Error(
        `Model '${model}' (${quant}) references ${repo}, which no mount of this page holds`,
      );
    }
    map[repo] = holders[repo];
  }
  return map;
};
