/**
 * `@karume/local-distributions/node` — the mounts of the mount table read straight from their
 * directories under Node: the facts a page gets from `/models/index.json`
 * ({@link localDistributionIndex}) and a mount opened as a hub source with `nodeDirectory`
 * ({@link openMountDirectory}). Cross-repo references are mapped to the mounts that hold them by
 * `crossRepoHolders`, the function the direct route of a browser maps them with, so both read the
 * same repositories from the same mounts.
 */

import { readFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { type DistributionSource, MANIFEST_FILENAME, parseManifest } from "@karume/hub";
import { nodeDirectory } from "@karume/hub/node";

import { crossRepoHolders, type DistributionIndex, MountUnavailableError } from "./browser.ts";
import { MOUNTS, type MountName } from "./mounts.ts";
import { distributionIndex, distributionMounts } from "./plugin.ts";

/** What `/models/index.json` answers, read from the directories under `repositoryRoot` now. */
export const localDistributionIndex = (repositoryRoot: string): Promise<DistributionIndex> =>
  distributionIndex(distributionMounts(repositoryRoot, MOUNTS));

/** A mount opened by {@link openMountDirectory}. */
export type OpenedMountDirectory = {
  /** What `loadManifest` and a family's `fromPretrained` take. */
  readonly source: DistributionSource;
  /** The mount's directory relative to the repository root, as `/models/index.json` names it. */
  readonly directory: string;
  /** The bytes of its `karume.json`, read when it was opened. */
  readonly manifestBytes: Uint8Array<ArrayBuffer>;
};

/**
 * Opens mount `name` from its directory under `repositoryRoot` (the mount table's directory). Each
 * repository its manifest references is read from the directory of the mount that holds it.
 * Throws {@link MountUnavailableError} when the mirror is missing.
 */
export const openMountDirectory = async (
  repositoryRoot: string,
  name: MountName,
): Promise<OpenedMountDirectory> => {
  const index = await localDistributionIndex(repositoryRoot);
  const status = index[name];
  if (!status.available) {
    throw new MountUnavailableError(name, `its mirror ${status.directory} has no karume.json`);
  }
  const directoryOf = (mount: string): string => resolve(repositoryRoot, index[mount].directory);
  const manifestBytes = new Uint8Array(await readFile(join(directoryOf(name), MANIFEST_FILENAME)));
  const manifest = parseManifest(new TextDecoder().decode(manifestBytes));
  const crossRepo = Object.fromEntries(
    Object.entries(crossRepoHolders(manifest, index)).map(([repo, mount]) => [
      repo,
      nodeDirectory(directoryOf(mount)),
    ]),
  );
  return {
    source: nodeDirectory(directoryOf(name), { crossRepo }),
    directory: status.directory,
    manifestBytes,
  };
};
