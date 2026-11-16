/**
 * The repository's formatter (Oxfmt through Vite+, with the settings of the root `vite.config.ts`)
 * applied to TypeScript text. `vp fmt --stdin-filepath=<path>` reads the text from stdin and writes
 * the formatted text to stdout: the file name picks the parser and the ignore rules, and the
 * configuration is the one found from the working directory, so it runs at the repository root.
 * The `vp` started is the workspace's own (the version the repository pins), not whichever is on
 * PATH.
 */

import { execFileSync } from "node:child_process";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

/** The formatter refused the text or could not run. */
export class FormatError extends Error {
  override name = "FormatError";
}

/** The repository root (this file is `tools/geometry-profiles/format.ts`). */
export const REPOSITORY_ROOT = fileURLToPath(new URL("../../", import.meta.url)).replace(
  /[\\/]$/,
  "",
);

const vpBin = (): string =>
  join(dirname(createRequire(import.meta.url).resolve("vite-plus/package.json")), "bin", "vp");

/**
 * Formats `text` as the file `path` (repository-relative) would be formatted. The file itself is
 * neither read nor written.
 */
export const formatTypeScript = (text: string, path: string): string => {
  try {
    return execFileSync(process.execPath, [vpBin(), "fmt", `--stdin-filepath=${path}`], {
      cwd: REPOSITORY_ROOT,
      input: text,
      encoding: "utf8",
      stdio: ["pipe", "pipe", "pipe"],
    });
  } catch (cause) {
    const stderr =
      typeof cause === "object" && cause !== null && "stderr" in cause ? String(cause.stderr) : "";
    throw new FormatError(`vp fmt failed on ${path}: ${stderr.trim()}`, { cause });
  }
};
