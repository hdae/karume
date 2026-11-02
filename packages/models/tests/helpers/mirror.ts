/**
 * The distribution mirrors under `models/` as the real-weight tests read them (ADR 0139 decision
 * 1): one root resolver, one source with the cross-repo references resolved to sibling mirrors, and
 * one manifest-based opener for a component's container and for a declared asset.
 *
 * A mirror container is opened **through its manifest** (`karume.json` → selection →
 * `container.parts`), never from a representative path plus numbering: the part names and the
 * layout are the distribution's (`<model>/<graph>/model.<dtype>-N-of-M.krm`, shared parts under
 * `shared/` — ADR 0041 decision 9), and the manifest also carries the expected sha256 of the two
 * documents that `openContainer` checks.
 *
 * NOTE: only the public surfaces of hub and runtime are used here (`@karume/hub`,
 * `@karume/hub/node`, `@karume/runtime`) — the same rule as `helpers/memory-cache.ts`.
 */

import { isNotFound, readTextFileSync } from "@karume/test-support/fs";
import {
  type DistributionSource,
  fetchAssets,
  type FileRef,
  type LoadedManifest,
  loadManifest,
  parseManifest,
  type ResolvedSelection,
  resolveSelection,
  openContainerSource,
  type ContainerRef,
} from "@karume/hub";
import { nodeDirectory } from "@karume/hub/node";
import {
  openContainer,
  type OpenedContainer,
  prepareContainer,
  type PreparedModel,
} from "@karume/runtime";

/** The root of the distribution mirrors (one directory per Hugging Face repository). */
export const MODELS_ROOT: URL = new URL("../../../../models/", import.meta.url);

/** The manifest major this version reads (older majors are not read — ADR 0109 decision 1). */
export const MIRROR_FORMAT = "karume/5";

/** The owner of the published distributions: a cross-repo reference `hdae/<name>` is `models/<name>`. */
const MIRROR_OWNER = "hdae";

/** The selection axes of a mirror (the manifest's defaults when omitted). */
export type MirrorChoice = {
  readonly model?: string;
  readonly quant?: string;
};

/** The mirror of one repository, e.g. `mirrorRoot("karume-birefnet-hr")` → `models/karume-birefnet-hr/`. */
export const mirrorRoot = (repoDir: string): URL => new URL(`${repoDir}/`, MODELS_ROOT);

/**
 * Whether `root` holds a distribution **this version can read** (`karume.json` declares
 * `karume/5`) together with every `hdae/<name>` mirror it references across repositories.
 *
 * MUST: a mirror whose cross-repo sibling is absent is not present — opening it would throw in
 * {@link mirrorDirectory}, so a test guarded by this check SKIPs instead of erroring at module
 * evaluation (the distribution gate owns the red for an incomplete set of mirrors).
 *
 * MUST: decide by the format, not by "the file exists" — a machine that still has a mirror of an
 * older major would otherwise run the tests against a manifest `parseManifest` rejects, and the
 * gate would be red without saying why (ADR 0109 decision 9). Only NotFound means "absent"; any
 * other failure propagates.
 */
export const mirrorPresent = (root: URL): boolean => mirrorReadable(root, []);

const mirrorReadable = (root: URL, chain: readonly string[]): boolean => {
  let text: string;
  try {
    text = readTextFileSync(new URL("karume.json", root));
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
  if ((JSON.parse(text) as { readonly format?: unknown }).format !== MIRROR_FORMAT) return false;
  // A reference outside hdae/ or a cycle is left to mirrorDirectory, which fails by name.
  return [...crossRepos(root)].every((repo) => {
    const [owner, name] = repo.split("/");
    if (owner !== MIRROR_OWNER || name === undefined || chain.includes(repo)) return true;
    return mirrorReadable(new URL(`../${name}/`, root), [...chain, repo]);
  });
};

/** Every repository a manifest references across repositories (parts of every container and assets). */
const crossRepos = (root: URL): ReadonlySet<string> => {
  const manifest = parseManifest(readTextFileSync(new URL("karume.json", root)));
  const refs: FileRef[] = [];
  for (const entry of Object.values(manifest.models)) {
    for (const dtypes of Object.values(entry.weights)) {
      for (const { container } of Object.values(dtypes)) refs.push(...container.parts);
    }
    refs.push(...Object.values(entry.assets));
  }
  return new Set(refs.flatMap((ref) => (ref.repo === undefined ? [] : [ref.repo])));
};

/**
 * A `nodeDirectory` source of the mirror at `root` whose cross-repo references (read from its
 * manifest) resolve to the sibling mirrors: `hdae/<name>` → `<root>/../<name>/` (under `models/`,
 * `models/<name>/`). A sibling's own cross-repo references resolve the same way.
 *
 * MUST: a referenced mirror that is absent fails here, loudly and by name — an unmapped cross-repo
 * target would otherwise surface later as a fetch failure of whichever file is read first.
 */
export const mirrorDirectory = (root: URL): DistributionSource => mirrorSource(root, []);

const mirrorSource = (root: URL, chain: readonly string[]): DistributionSource => {
  const crossRepo: Record<string, DistributionSource> = {};
  for (const repo of crossRepos(root)) {
    const [owner, name] = repo.split("/");
    if (owner !== MIRROR_OWNER || name === undefined) {
      throw new Error(
        `test: ${root.pathname} references '${repo}', which is not a ${MIRROR_OWNER}/<name> mirror`,
      );
    }
    if (chain.includes(repo)) {
      throw new Error(`test: cross-repo references form a cycle: ${[...chain, repo].join(" → ")}`);
    }
    const sibling = new URL(`../${name}/`, root);
    if (!mirrorPresent(sibling)) {
      throw new Error(
        `test: ${root.pathname} references '${repo}', but its mirror ${sibling.pathname} is ` +
          `absent (or not ${MIRROR_FORMAT})`,
      );
    }
    crossRepo[repo] = mirrorSource(sibling, [...chain, repo]);
  }
  return nodeDirectory(root.pathname, { crossRepo });
};

/** The loaded manifest of a mirror, the selection, and the container of `key` in it. */
export type MirrorSelection = {
  readonly loaded: LoadedManifest;
  readonly selection: ResolvedSelection;
  readonly container: ContainerRef;
};

/** Loads the mirror's manifest and resolves `choice` with the weights subset `[key]`. */
export const selectMirrorComponent = async (
  root: URL,
  key: string,
  choice: MirrorChoice = {},
): Promise<MirrorSelection> => {
  const loaded = await loadManifest(mirrorDirectory(root));
  const selection = resolveSelection(loaded.manifest, {
    ...(choice.model === undefined ? {} : { model: choice.model }),
    ...(choice.quant === undefined ? {} : { quant: choice.quant }),
    weights: [key],
  });
  const container = selection.containers[key];
  if (container === undefined) {
    throw new Error(`test: the manifest of ${root.pathname} has no container for '${key}'`);
  }
  return { loaded, selection, container };
};

/** Opens a selected container with range reads (part 0 now, blocks when a Session reads them). */
export const openSelectedContainer = async ({
  loaded,
  container,
}: MirrorSelection): Promise<OpenedContainer> =>
  await openContainer(
    { kind: "source", source: openContainerSource(loaded, container) },
    container.descriptor,
  );

/**
 * Opens the component `key` of the mirror with range reads and returns its declaration and the
 * Session entry point.
 *
 * A component is one container; its blocks are read in part order when the Session is built
 * (ADR 0109 decision 7). The graph name is the manifest's weights key (the writer's convention —
 * container-v1 §2.1).
 */
export const openMirrorComponent = async (
  root: URL,
  key: string,
  choice: MirrorChoice = {},
): Promise<{
  readonly graph: PreparedModel["graph"];
  readonly createSession: PreparedModel["createContainerSession"];
  readonly opened: OpenedContainer;
}> => {
  const opened = await openSelectedContainer(await selectMirrorComponent(root, key, choice));
  const prepared = prepareContainer(opened, key);
  return {
    graph: prepared.graph,
    createSession: (gpu, options) => prepared.createContainerSession(gpu, options),
    opened,
  };
};

/**
 * Reads the declared asset `name` of the mirror (a tokenizer, Wan's text embeddings) as bytes,
 * through the manifest: the source checks the declared size, and a cross-repo asset is read from
 * its sibling mirror.
 */
export const readMirrorAsset = async (
  root: URL,
  name: string,
  choice: MirrorChoice = {},
): Promise<Uint8Array<ArrayBuffer>> => {
  const loaded = await loadManifest(mirrorDirectory(root));
  const { assets, model } = resolveSelection(loaded.manifest, {
    ...(choice.model === undefined ? {} : { model: choice.model }),
    ...(choice.quant === undefined ? {} : { quant: choice.quant }),
    weights: [],
  });
  const ref = assets[name];
  if (ref === undefined) {
    throw new Error(
      `test: model '${model}' of ${root.pathname} declares no asset '${name}' ` +
        `(declared: ${Object.keys(assets).join(" / ")})`,
    );
  }
  const fetched = await fetchAssets(loaded, { [name]: ref });
  return fetched[name];
};
