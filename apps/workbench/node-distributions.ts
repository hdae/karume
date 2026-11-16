/**
 * The Node run's `openDistribution` (`run.ts`): a mount read from its mirror's directory under the
 * repository root, whatever way the entry asks a browser to read it — under Node there is no dev
 * server, and the directory is the distribution itself.
 */

import { openMountDirectory } from "@karume/local-distributions/node";

import type { OpenDistribution } from "./src/entry.ts";
import { sha256Hex } from "./src/sha256.ts";

export const nodeDistributions =
  (repositoryRoot: string): OpenDistribution =>
  async (mount) => {
    const opened = await openMountDirectory(repositoryRoot, mount);
    return {
      ref: opened.source,
      facts: {
        directory: opened.directory,
        manifestSha256: await sha256Hex(opened.manifestBytes),
        read: "directory",
      },
    };
  };
