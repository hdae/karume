/**
 * `@karume/local-distributions/browser` — reading the distributions a dev server mounts under
 * `/models/<name>/` (`plugin.ts`) from a page. Web APIs only. Every URL is resolved against the
 * `base` the page passes (its `document.baseURI`), so the page works behind any base path and
 * nothing here reads the page's location.
 *
 * Two ways to read one: {@link distributionSource} (`localDirectory` over plain `fetch`, nothing
 * cached) and {@link hfRepoRef} (the HF-compatible route `/hf/`, read by the hub's HF source into
 * its CacheStorage). A page picks one with a {@link PipelineSource} and hands {@link pretrainedRef}
 * to the family's `fromPretrained`; {@link openMountRoute} opens a mount on the route a
 * {@link SourceKind} names, with its manifest's bytes (the browser twin of the Node side's
 * `openMountDirectory`).
 */

import {
  type DistributionSource,
  type FileRef,
  type HubRepoRef,
  localDirectory,
  MANIFEST_FILENAME,
  type Manifest,
  parseManifest,
  resolveSelection,
} from "@karume/hub";

import type { MountName } from "./mounts.ts";

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
export const fetchDistributionIndex = async (
  base: string,
  signal?: AbortSignal,
): Promise<DistributionIndex> => {
  const response = await fetch(
    new URL("models/index.json", base).href,
    signal === undefined ? {} : { signal },
  );
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
  base: string,
  crossRepo: Readonly<Record<string, DistributionSource>> = {},
): DistributionSource =>
  localDirectory(
    {
      readFile: async (path, { signal }) => {
        const response = await fetch(
          new URL(`models/${name}/${path}`, base).href,
          withSignal(signal),
        );
        if (!response.ok) throw new Error(`models/${name}/${path}: HTTP ${response.status}`);
        return new Uint8Array(await response.arrayBuffer());
      },
      readFileRange: async (path, offset, length, { signal }) => {
        const response = await fetch(new URL(`models/${name}/${path}`, base).href, {
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
 * over plain `fetch`, nothing cached. MUST be stable (memoized — the playground's `usePipelineSource`
 * does it): a page that rebuilds its pipeline when the source changes would rebuild on every render.
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
 * fetch below `base` with `crossRepo` mapping each referenced repo to the mount that holds it (the
 * direct route has no revision to follow — `{ "hdae/karume-anima": "anima" }` for the Anima
 * fine-tunes; a mount is named as `/models/index.json` keys it, which is how
 * {@link crossRepoHolders} names it).
 */
export const pretrainedRef = (
  source: PipelineSource,
  name: MountName,
  base: string,
  crossRepo: Readonly<Record<string, string>> = {},
): HubRepoRef | DistributionSource =>
  source.kind === "cached"
    ? source.ref
    : distributionSource(
        name,
        base,
        Object.fromEntries(
          Object.entries(crossRepo).map(([repo, mount]) => [repo, distributionSource(mount, base)]),
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

/** The last path element (`hdae/karume-umt5-xxl` → `karume-umt5-xxl`). */
const lastElement = (path: string): string => path.split(/[/\\]/).at(-1) ?? path;

/**
 * What {@link crossRepoHolders} reads of each mount: its directory and the repository it is served
 * under on `/hf/`. A {@link DistributionIndex} is one; the Node side builds it from the mount table.
 */
type HolderIndex = Readonly<
  Record<string, { readonly directory: string; readonly hf?: Pick<HfCoordinates, "repo"> }>
>;

/**
 * The mount of `index` that holds `repo`, by the rule the `/hf/` route serves a cross-repo reference
 * with (`plugin.ts`): the mount served under that repository's name, else the mount whose directory
 * is named after the repository's last element (`hdae/karume-umt5-xxl` → `models/karume-umt5-xxl`).
 */
const holderOf = (repo: string, index: HolderIndex): string | undefined => {
  const mounts = Object.entries(index);
  return (mounts.find(([, status]) => status.hf?.repo === repo) ??
    mounts.find(([, status]) => lastElement(status.directory) === lastElement(repo)))?.[0];
};

/**
 * The `crossRepo` map of the direct route for a whole distribution: every repository `manifest`
 * references across repositories (any model, container or asset), mapped to the mount of `index`
 * that holds it ({@link holderOf}). The Node side maps a mirror's references through this same
 * function. A repository no mount holds is left out: the hub fails loudly, naming it, when a read
 * needs it, so a selection that does not read it still opens.
 */
export const crossRepoHolders = (
  manifest: Manifest,
  index: HolderIndex,
): Readonly<Record<string, string>> => {
  const refs: readonly FileRef[] = Object.values(manifest.models).flatMap((model) => [
    ...Object.values(model.weights).flatMap((dtypes) =>
      Object.values(dtypes).flatMap(({ container }) => container.parts),
    ),
    ...Object.values(model.assets),
  ]);
  const holders: Record<string, string> = {};
  for (const { repo } of refs) {
    if (repo === undefined || Object.hasOwn(holders, repo)) continue;
    const holder = holderOf(repo, index);
    if (holder !== undefined) holders[repo] = holder;
  }
  return holders;
};

/**
 * A mount that cannot be read the way it was asked for: its mirror is missing, or it has no
 * HF-compatible coordinates for the cached route.
 */
export class MountUnavailableError extends Error {
  readonly mount: string;

  constructor(mount: string, reason: string) {
    super(`Distribution mount ${mount}: ${reason}`);
    this.name = "MountUnavailableError";
    this.mount = mount;
  }
}

/** A mount opened by {@link openMountRoute}. */
export type OpenedMountRoute = {
  /** What `loadManifest` and a family's `fromPretrained` take ({@link pretrainedRef}). */
  readonly ref: HubRepoRef | DistributionSource;
  /** The mount's directory relative to the repository root, as `/models/index.json` names it. */
  readonly directory: string;
  /** The bytes of its `karume.json`, read from `/models/<name>/` when it was opened. */
  readonly manifestBytes: Uint8Array<ArrayBuffer>;
};

/**
 * Opens mount `name` of the dev server below `base` the way `source` names: `cached` = the
 * `HubRepoRef` of the HF-compatible route, `direct` = plain fetch from `/models/<name>/` with each
 * repository the manifest references read from the mount that holds it ({@link crossRepoHolders} —
 * the mapping the Node side's `openMountDirectory` uses too). The manifest's bytes are read from
 * `/models/<name>/` either way. Throws {@link MountUnavailableError} when the mirror is missing or,
 * for `cached`, when the dev server serves the mount on no HF-compatible route.
 */
export const openMountRoute = async (
  base: string,
  name: MountName,
  source: SourceKind,
): Promise<OpenedMountRoute> => {
  const index = await fetchDistributionIndex(base);
  const status = index[name];
  if (!status.available) {
    throw new MountUnavailableError(name, `its mirror ${status.directory} has no karume.json`);
  }
  const path = `models/${name}/${MANIFEST_FILENAME}`;
  const response = await fetch(new URL(path, base).href);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  const manifestBytes = new Uint8Array(await response.arrayBuffer());
  if (source === "cached") {
    if (status.hf === undefined) {
      throw new MountUnavailableError(
        name,
        "the dev server serves it on no HF-compatible route (/hf/), so it cannot be read cached; read it direct",
      );
    }
    return { ref: hfRepoRef(status.hf, base), directory: status.directory, manifestBytes };
  }
  const manifest = parseManifest(new TextDecoder().decode(manifestBytes));
  return {
    ref: pretrainedRef({ kind: "direct" }, name, base, crossRepoHolders(manifest, index)),
    directory: status.directory,
    manifestBytes,
  };
};
