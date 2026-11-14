import { isNotFound, readFile, readTextFile } from "@karume/test-support/fs";
/**
 * Reads a local asset "if it is present" (used by the checks that take a missing file —
 * `undefined` — as the source of their SKIP decision).
 *
 * MUST: anything other than NotFound propagates — reading a permission error, a directory or a
 * failure part-way through a read as "the asset is missing" would turn a check that never ran into
 * a silent green SKIP (ADR 0005's explicit SKIP; fail loudly).
 *
 * NOTE: only tests read local files (`readFile` of `@karume/test-support/fs`). The package itself
 * uses Web standard APIs only (cross-cutting invariant).
 */

const undefinedIfNotFound = (cause: unknown): undefined => {
  if (isNotFound(cause)) return undefined;
  throw cause;
};

/** Reads as text: `undefined` when the path is missing; any other failure throws. */
export const readTextIfPresent = (path: URL): Promise<string | undefined> =>
  readTextFile(path).catch(undefinedIfNotFound);

/** Reads as bytes: `undefined` when the path is missing; any other failure throws. */
export const readFileIfPresent = (path: URL): Promise<Uint8Array<ArrayBuffer> | undefined> =>
  readFile(path).catch(undefinedIfNotFound);
