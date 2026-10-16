/**
 * `@karume/hub/node` — a distribution source backed by the Node.js file system.
 *
 * MUST: **this subpath is the only place in hub that depends on `node:` modules** (`node:fs`). It is
 * an opt-in carve-out from the invariant "runtime dependencies are Web standard APIs only" that
 * `mod.ts` keeps; browser code must not import it (there, pass OPFS / IndexedDB / File System
 * Access handles as a `DirectoryAdapter` to {@link ../mod.ts localDirectory}).
 *
 * Only "how to read local bytes" lives here. The semantics of a source (generations, verification,
 * cross-repo references, progress) stay with `localDirectory` in the main module. The Deno
 * counterpart is `./deno.ts`; the two adapters MUST behave the same from the main module's view.
 */

import { open, readFile } from "node:fs/promises";
import {
  type DirectoryAdapter,
  type DistributionSource,
  localDirectory,
  type LocalDirectoryOptions,
} from "./mod.ts";

/**
 * Location of an entry under `root`. A string root is **concatenated as is** — manifest paths are
 * POSIX-relative (`transformer/model-00001-of-00004.krm`) and `..` was rejected at parse time
 * (`ManifestPathError`), so no `URL` normalisation is needed. A `URL` root resolves relative to it.
 */
const locate = (base: string | URL, path: string): string | URL =>
  typeof base === "string" ? `${base}${path}` : new URL(path, base);

/** Guarantee a trailing `/` (without it `URL` resolution points at a sibling and concatenation runs into the path). */
const asDirectory = (root: string | URL): string | URL => {
  const href = typeof root === "string" ? root : root.href;
  const slashed = href.endsWith("/") ? href : `${href}/`;
  return typeof root === "string" ? slashed : new URL(slashed);
};

/** `Buffer` views may share a pooled `ArrayBuffer`; the adapter contract wants bytes that own theirs. */
const owned = (buffer: Uint8Array): Uint8Array<ArrayBuffer> => {
  const backing = buffer.buffer;
  if (
    backing instanceof ArrayBuffer && buffer.byteOffset === 0 &&
    backing.byteLength === buffer.byteLength
  ) {
    return new Uint8Array(backing);
  }
  const copy = new Uint8Array(new ArrayBuffer(buffer.byteLength));
  copy.set(buffer);
  return copy;
};

/**
 * Use a directory on the Node.js file system as a distribution source. `root` is a directory path
 * (relative paths resolve against the current working directory) or a `file:` URL.
 */
export const nodeDirectory = (
  root: string | URL,
  options: LocalDirectoryOptions = {},
): DistributionSource => {
  const base = asDirectory(root);
  const adapter: DirectoryAdapter = {
    readFile: async (path, { signal }) => {
      const at = locate(base, path);
      try {
        return owned(await readFile(at, { ...(signal === undefined ? {} : { signal }) }));
      } catch (error) {
        // MUST: an abort surfaces as the caller's reason, untouched (the common layer tells "cancelled"
        // from "failed"; Node wraps it in an AbortError whose cause is the reason).
        if (signal?.aborted === true) throw signal.reason;
        // Missing, unreadable, or a directory: all fail loudly as "cannot read". Naming the real
        // path is this adapter's responsibility — the common layer only knows the manifest path.
        throw new Error(`@karume/hub/node: cannot read ${at}`, { cause: error });
      }
    },
    // Range read (`DirectoryAdapter.readFileRange` contract): positional reads return exactly
    // `[offset, offset + length)`. The declared size was checked by the source (`sources/local.ts`),
    // so the only failure left here is "the file is shorter than declared".
    readFileRange: async (path, offset, length, { signal }) => {
      // Check before opening a handle: an abort that is only noticed at the top of the loop would
      // slip through for a zero-length read.
      signal?.throwIfAborted();
      const at = locate(base, path);
      let file;
      try {
        file = await open(at, "r");
      } catch (error) {
        throw new Error(`@karume/hub/node: cannot read ${at}`, { cause: error });
      }
      try {
        const target = new Uint8Array(new ArrayBuffer(length));
        let filled = 0;
        while (filled < length) {
          signal?.throwIfAborted();
          const { bytesRead } = await file.read(target, filled, length - filled, offset + filled);
          if (bytesRead === 0) {
            throw new Error(
              `@karume/hub/node: ${at} is shorter than offset ${offset} + ${length} bytes`,
            );
          }
          filled += bytesRead;
        }
        return target;
      } finally {
        await file.close();
      }
    },
  };
  return localDirectory(adapter, { label: String(root), ...options });
};
