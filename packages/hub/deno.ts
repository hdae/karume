/**
 * `@karume/hub/deno` — the subpath that uses Deno's file system as a source.
 *
 * MUST: **only this subpath depends on the Deno API** (`Deno.readFile`). It is an opt-in carve-out
 * from the invariant of the hub's main entry (`mod.ts`) that "runtime dependencies are Web standard
 * APIs only", and must not be imported from browser code — as long as only the main entry is used,
 * `Deno` never appears (in a browser, pass an OPFS / IndexedDB / File System Access picker as the
 * `DirectoryAdapter` of {@link ../mod.ts localDirectory}).
 *
 * Only "how to read the local files" belongs here; the semantics as a source (generations,
 * verification, cross-repository references, progress) are owned by `localDirectory` in the main
 * entry.
 */

import {
  type DirectoryAdapter,
  type DistributionSource,
  localDirectory,
  type LocalDirectoryOptions,
} from "./mod.ts";

/**
 * The slice of Deno's API this subpath uses. The repository is type-checked with Node's type set,
 * where the `Deno` global does not exist; under Deno the real global shadows this declaration.
 */
declare const Deno: {
  readFile(
    path: string | URL,
    options?: { signal?: AbortSignal },
  ): Promise<Uint8Array<ArrayBuffer>>;
  open(path: string | URL): Promise<DenoFsFile>;
  readonly SeekMode: { readonly Start: 0 };
};
type DenoFsFile = {
  seek(offset: number, whence: 0): Promise<number>;
  read(buffer: Uint8Array<ArrayBuffer>): Promise<number | null>;
  close(): void;
};

/**
 * The location of a file under root. A string root is **concatenated as is** — manifest paths are
 * POSIX-relative (`transformer/model-00001-of-00004.krm`) and `..` is rejected at parse time
 * (`ManifestPathError`), so there is no need to normalise through `URL`. Conversely, going through
 * `URL` would need a spelling that maps a Windows drive letter to `file:`, so keeping the string is
 * less fragile.
 */
const locate = (base: string | URL, path: string): string | URL =>
  typeof base === "string" ? `${base}${path}` : new URL(path, base);

/**
 * Ensures a trailing `/` (without it, `URL` resolution points at a sibling and string concatenation
 * runs into the path).
 */
const asDirectory = (root: string | URL): string | URL => {
  const href = typeof root === "string" ? root : root.href;
  const slashed = href.endsWith("/") ? href : `${href}/`;
  return typeof root === "string" ? slashed : new URL(slashed);
};

/**
 * Uses a directory under Deno as a source. `root` is a directory path (relative to the cwd when
 * relative) or a `file:` URL.
 *
 * ```ts ignore
 * import { loadManifest } from "@karume/hub";
 * import { denoDirectory } from "@karume/hub/deno";
 *
 * const loaded = await loadManifest(denoDirectory("./models/karume-gemma4"));
 * ```
 *
 * The only permission needed is `--allow-read` under root (neither the network nor CacheStorage is
 * used).
 */
export const denoDirectory = (
  root: string | URL,
  options: LocalDirectoryOptions = {},
): DistributionSource => {
  const base = asDirectory(root);
  const adapter: DirectoryAdapter = {
    readFile: async (path, { signal }) => {
      const at = locate(base, path);
      try {
        return await Deno.readFile(at, { ...(signal === undefined ? {} : { signal }) });
      } catch (error) {
        // MUST: rethrow an abort as is (otherwise the common layer cannot tell "cancelled" from
        // "fetch failed").
        if (signal?.aborted === true) throw error;
        // Missing, permission denied, or a directory: all fail the same way, as "cannot read"
        // (fail loudly). Naming **the real path** is this code's responsibility alone — the common
        // layer knows only the relative path in the manifest; only the adapter knows which
        // directory it looked under.
        throw new Error(`@karume/hub/deno: cannot read ${at}`, { cause: error });
      }
    },
    // Range read (the contract of `DirectoryAdapter.readFileRange`): a positional read that returns
    // exactly `[offset, offset + length)`. `Deno.open` / `read` do not take a signal, so abort is
    // checked between reads (close in finally, name the real path). The source
    // (`sources/local.ts`) has already checked against the declared size, so this fails only when
    // "the file is shorter than declared".
    readFileRange: async (path, offset, length, { signal }) => {
      // Do not open an fd when already aborted (noticing it at the top of the loop after opening
      // would let the abort slip through a zero-length read — check once before touching the file).
      signal?.throwIfAborted();
      const at = locate(base, path);
      let file: DenoFsFile;
      try {
        file = await Deno.open(at);
      } catch (error) {
        throw new Error(`@karume/hub/deno: cannot read ${at}`, { cause: error });
      }
      try {
        await file.seek(offset, Deno.SeekMode.Start);
        const target = new Uint8Array(new ArrayBuffer(length));
        let filled = 0;
        while (filled < length) {
          signal?.throwIfAborted();
          const read = await file.read(target.subarray(filled));
          if (read === null) {
            throw new Error(
              `@karume/hub/deno: ${at} ends before offset ${offset} plus ${length} bytes`,
            );
          }
          filled += read;
        }
        return target;
      } finally {
        file.close();
      }
    },
  };
  return localDirectory(adapter, { label: String(root), ...options });
};
