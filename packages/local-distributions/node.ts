/**
 * `@karume/local-distributions/node` — a mount of the mount table opened as a hub source with
 * `nodeDirectory`, straight from its directory under Node ({@link openMountDirectory}). Cross-repo
 * references are mapped to the mounts that hold them by `crossRepoHolders`, the function the direct
 * route of a browser maps them with (`openMountRoute`), so both read the same repositories from the
 * same mounts.
 *
 * It reads the mount table and the opened mount's own mirror only: the HF-compatible route's
 * revisions (`hfServingTable`, which reads every mirror's manifest) are the dev server's, and a Node
 * run serves no `/hf/` route, so a broken mirror of another mount does not stop it.
 */

import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { type DistributionSource, MANIFEST_FILENAME, parseManifest } from "@karume/hub";
import { nodeDirectory } from "@karume/hub/node";

import { crossRepoHolders, MountUnavailableError } from "./browser.ts";
import { MOUNTS, type MountName } from "./mounts.ts";
import { type DistributionMount, distributionMounts, realRoot } from "./plugin.ts";

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
 * repository its manifest references is read from the directory of the mount that holds it: the
 * mount served under that repository's name on `/hf/` (the table's `hfRepo`), else the mount whose
 * directory is named after it. Throws {@link MountUnavailableError} when the mirror is missing (the
 * dev server's rule: no `karume.json` in the directory).
 */
export const openMountDirectory = async (
  repositoryRoot: string,
  name: MountName,
): Promise<OpenedMountDirectory> => {
  const mounts = distributionMounts(repositoryRoot, MOUNTS);
  const byName: Readonly<Record<string, DistributionMount>> = Object.fromEntries(
    mounts.map((mount) => [mount.name, mount]),
  );
  const mount = byName[name];
  if ((await realRoot(mount)) === undefined) {
    throw new MountUnavailableError(name, `its mirror ${mount.label} has no karume.json`);
  }
  const manifestBytes = new Uint8Array(await readFile(join(mount.root, MANIFEST_FILENAME)));
  const manifest = parseManifest(new TextDecoder().decode(manifestBytes));
  // What `crossRepoHolders` reads of each mount, from the table: its directory and its `/hf/` repo.
  const holders = Object.fromEntries(
    mounts.map((entry) => [
      entry.name,
      {
        directory: entry.label,
        ...(entry.hfRepo === undefined ? {} : { hf: { repo: entry.hfRepo } }),
      },
    ]),
  );
  const crossRepo = Object.fromEntries(
    Object.entries(crossRepoHolders(manifest, holders)).map(([repo, holder]) => [
      repo,
      nodeDirectory(byName[holder].root),
    ]),
  );
  return {
    source: nodeDirectory(mount.root, { crossRepo }),
    directory: mount.label,
    manifestBytes,
  };
};
