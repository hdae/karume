/**
 * A **temporary manifest** for fault injection (a source that borrows the distribution's directory
 * through symlinks and replaces only `karume.json`). The technique is the same as the temporary
 * manifest of `e2e_gemma4_quant_test.ts`; the original distribution and its weights are not
 * changed by a single byte.
 *
 * NOTE: only tests use `Deno.symlink` / `Deno.makeTempDir` (the package itself uses Web standard
 * APIs only — cross-cutting invariant).
 */

import { makeTempDir, readDir, remove, symlink, writeTextFile } from "@karume/test-support/fs";
import type { DistributionSource, LocalDirectoryOptions } from "@karume/hub";
import { nodeDirectory } from "@karume/hub/node";

/**
 * Runs `body` with a source that replaces the manifest of the distribution at `root` by `manifest`
 * (the temporary directory is removed on exit — only the symlinks are removed; the files they
 * point to are not touched).
 *
 * `options` go to the source as they are — a distribution with cross-repository references needs
 * its `crossRepo` mapping on the overridden source too.
 */
export const withManifestOverride = async <T>(
  root: URL,
  manifest: unknown,
  body: (source: DistributionSource) => Promise<T>,
  options: LocalDirectoryOptions = {},
): Promise<T> => {
  const temporary = await makeTempDir({ prefix: "karume-ab-manifest-" });
  try {
    for await (const entry of readDir(root)) {
      if (entry.name === "karume.json") continue;
      await symlink(new URL(entry.name, root), `${temporary}/${entry.name}`);
    }
    await writeTextFile(`${temporary}/karume.json`, JSON.stringify(manifest), {
      createNew: true,
    });
    return await body(nodeDirectory(temporary, options));
  } finally {
    await remove(temporary, { recursive: true });
  }
};
