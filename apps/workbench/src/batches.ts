/**
 * The tracked batches: every file `apps/workbench/batches/<id>.json`, as text, in the dev server
 * and in a production build alike. Adding a batch is adding a file; nothing else lists them.
 *
 * SPEC: docs/spec/workbench.md#batches
 */

import type { BatchFile } from "./batch.ts";

const FILES = import.meta.glob<string>("../batches/*.json", {
  eager: true,
  query: "?raw",
  import: "default",
});

/** The tracked batch files in file-name order, each with its id (the file name without `.json`). */
export const trackedBatchFiles = (): readonly BatchFile[] =>
  Object.entries(FILES)
    .map(([path, text]) => ({ id: path.slice(path.lastIndexOf("/") + 1, -".json".length), text }))
    .sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
