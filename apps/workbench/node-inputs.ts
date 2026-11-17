/**
 * The Node run's `openInput` (`run.ts`): a file under the repository's `inputs/` read from disk, one
 * positional read per range (the file is opened for each read, so nothing is left open when a run
 * ends). The real path is checked to stay inside `inputs/`, which may itself be a link.
 *
 * SPEC: docs/spec/workbench.md#input-files
 */

import { open, realpath, stat } from "node:fs/promises";
import { join, sep } from "node:path";

import type { OpenInput } from "./src/entry.ts";
import { inputSegments } from "./src/inputs.ts";

export const nodeInputs =
  (repositoryRoot: string): OpenInput =>
  async (path) => {
    const segments = inputSegments(path);
    const root = await realpath(join(repositoryRoot, "inputs"));
    const file = await realpath(join(root, ...segments));
    if (!file.startsWith(`${root}${sep}`)) {
      throw new Error(`input file inputs/${path}: its real path leaves inputs/`);
    }
    const { size } = await stat(file);
    return {
      file: {
        size,
        read: async (offset, length) => {
          const bytes = new Uint8Array(new ArrayBuffer(length));
          const handle = await open(file, "r");
          try {
            const { bytesRead } = await handle.read(bytes, 0, length, offset);
            if (bytesRead !== length) {
              throw new Error(
                `input file inputs/${path}: the range [${offset}, ${offset + length}) read ${bytesRead} bytes`,
              );
            }
          } finally {
            await handle.close();
          }
          return bytes;
        },
      },
      facts: { path: `inputs/${segments.join("/")}`, size, read: "directory" },
    };
  };
