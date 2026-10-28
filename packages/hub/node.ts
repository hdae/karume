/**
 * `@karume/hub/node` — the Node.js file system behind hub: a distribution source that reads a
 * local directory ({@link nodeDirectory}) and a `CacheStorage` for the fetch layer behind the
 * Hugging Face source ({@link nodeCacheStorage} — Node has no Cache API of its own).
 *
 * MUST: **this subpath is the only place in hub that depends on `node:` modules** (`node:fs`,
 * `node:crypto`, `node:url`). It is an opt-in carve-out from the invariant "runtime dependencies are
 * Web standard APIs only" that `mod.ts` keeps; browser code must not import it (there, pass OPFS /
 * IndexedDB / File System Access handles as a `DirectoryAdapter` to {@link ../mod.ts localDirectory},
 * and the browser's own `caches`).
 *
 * Only "how to read and write local bytes" lives here. The semantics of a source (generations,
 * verification, cross-repo references, progress) stay with `localDirectory` in the main module, and
 * the semantics of the cache (keys, the recorded hash, self-heal) stay with the fetch layer. The
 * Deno counterpart of the directory source is `./deno.ts`; the two adapters MUST behave the same
 * from the main module's view.
 */

import { createHash } from "node:crypto";
import {
  type FileHandle,
  mkdir,
  open,
  readdir,
  readFile,
  rename,
  rm,
  stat,
  unlink,
} from "node:fs/promises";
import { fileURLToPath } from "node:url";
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
    backing instanceof ArrayBuffer &&
    buffer.byteOffset === 0 &&
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

// ---------------------------------------------------------------------------------------------
// nodeCacheStorage — a file-backed Cache Storage for the fetch layer
// ---------------------------------------------------------------------------------------------

/**
 * Directory name of a namespace is this prefix + `encodeURIComponent(name)`. The fixed prefix keeps
 * every name inside the root (`""`, `"."` and `".."` encode to themselves) while the usual names
 * stay readable (`cache-fetch-cache`).
 */
const NAMESPACE_PREFIX = "cache-";
/** Suffix of an entry still being written; listings skip these. */
const TEMPORARY_SUFFIX = ".tmp";
/** The header line (URL, status, headers as one JSON object) MUST fit in this many bytes. */
const HEADER_LIMIT = 64 * 1024;
/** Size of one positional read while streaming a body. */
const CHUNK_BYTES = 1 << 20;

const isNotFound = (error: unknown): boolean =>
  (error as { readonly code?: unknown }).code === "ENOENT";

const urlOf = (request: RequestInfo | URL): string =>
  typeof request === "string" ? request : request instanceof URL ? request.href : request.url;

/**
 * Entry key: the request URL normalised the way the Cache API normalises keys (parsed, fragment
 * dropped). A URL that does not parse fails loudly; the fetch layer passes absolute URLs only.
 */
const keyOf = (request: RequestInfo | URL): string => {
  const url = new URL(urlOf(request));
  url.hash = "";
  return url.href;
};

const rejectQueryOptions = (options: CacheQueryOptions | undefined): void => {
  if (
    options?.ignoreSearch === true ||
    options?.ignoreMethod === true ||
    options?.ignoreVary === true
  ) {
    throw new Error(
      "@karume/hub/node: cache query options (ignoreSearch / ignoreMethod / ignoreVary) are not supported",
    );
  }
};

const unsupported = (member: string): Promise<never> =>
  Promise.reject(
    new Error(
      `@karume/hub/node: ${member} is not supported (the fetch layer needs match / put / delete / keys only)`,
    ),
  );

/** First line of an entry file; the body follows the newline. */
type EntryHeader = {
  readonly url: string;
  readonly status: number;
  readonly statusText: string;
  readonly headers: Readonly<Record<string, string>>;
};

const isEntryHeader = (value: unknown): value is EntryHeader => {
  if (typeof value !== "object" || value === null) return false;
  const { url, status, statusText, headers } = value as Record<string, unknown>;
  return (
    typeof url === "string" &&
    typeof status === "number" &&
    typeof statusText === "string" &&
    typeof headers === "object" &&
    headers !== null &&
    Object.values(headers).every((item) => typeof item === "string")
  );
};

/** An entry opened for reading. Reads are positional, so the handle has no cursor to keep. */
type OpenedEntry = {
  readonly handle: FileHandle;
  readonly header: EntryHeader;
  readonly bodyOffset: number;
  readonly size: number;
};

/**
 * Open an entry file; `undefined` when there is none. A file without a parsable header line is
 * corrupt and fails loudly (a silent miss would hide a broken cache behind re-downloads).
 */
const openEntry = async (path: string): Promise<OpenedEntry | undefined> => {
  let handle: FileHandle;
  try {
    handle = await open(path, "r");
  } catch (error) {
    if (isNotFound(error)) return undefined;
    throw error;
  }
  try {
    const { size } = await handle.stat();
    const head = new Uint8Array(new ArrayBuffer(Math.min(HEADER_LIMIT, size)));
    const { bytesRead } = await handle.read(head, 0, head.byteLength, 0);
    const newline = head.subarray(0, bytesRead).indexOf(0x0a);
    if (newline === -1) {
      throw new Error(
        `@karume/hub/node: ${path} is not a cache entry (no header line in the first ${HEADER_LIMIT} bytes)`,
      );
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(new TextDecoder().decode(head.subarray(0, newline)));
    } catch (error) {
      throw new Error(`@karume/hub/node: ${path} has a corrupt header line`, { cause: error });
    }
    if (!isEntryHeader(parsed)) {
      throw new Error(`@karume/hub/node: ${path} has a corrupt header line`);
    }
    return { handle, header: parsed, bodyOffset: newline + 1, size };
  } catch (error) {
    await handle.close();
    throw error;
  }
};

/**
 * The body as a stream of positional reads. MUST close the handle on every exit (end, error,
 * cancel): the fetch layer cancels the bodies it does not consume, and a handle left open is one
 * leaked descriptor per `match`.
 */
const bodyStream = (entry: OpenedEntry): ReadableStream<Uint8Array<ArrayBuffer>> => {
  let position = entry.bodyOffset;
  let closed = false;
  const close = async (): Promise<void> => {
    if (closed) return;
    closed = true;
    await entry.handle.close();
  };
  return new ReadableStream({
    pull: async (controller) => {
      const remaining = entry.size - position;
      if (remaining <= 0) {
        await close();
        controller.close();
        return;
      }
      const chunk = new Uint8Array(new ArrayBuffer(Math.min(CHUNK_BYTES, remaining)));
      let bytesRead: number;
      try {
        ({ bytesRead } = await entry.handle.read(chunk, 0, chunk.byteLength, position));
      } catch (error) {
        await close();
        controller.error(error);
        return;
      }
      if (bytesRead === 0) {
        // The inode shrank under an open handle: nothing renames into that state, so the entry is
        // corrupt rather than merely short.
        await close();
        controller.error(
          new Error(`@karume/hub/node: cache entry ended ${remaining} bytes before its size`),
        );
        return;
      }
      position += bytesRead;
      controller.enqueue(chunk.subarray(0, bytesRead));
    },
    cancel: close,
  });
};

const writeAll = async (handle: FileHandle, bytes: Uint8Array<ArrayBuffer>): Promise<void> => {
  let written = 0;
  while (written < bytes.byteLength) {
    const { bytesWritten } = await handle.write(bytes, written, bytes.byteLength - written);
    written += bytesWritten;
  }
};

const nodeCache = (directory: string): Cache => {
  const pathOf = (request: RequestInfo | URL): string =>
    `${directory}/${createHash("sha256").update(keyOf(request)).digest("hex")}`;
  const entryPaths = async (): Promise<string[]> => {
    const names = await readdir(directory).catch((error: unknown) => {
      if (isNotFound(error)) return [];
      throw error;
    });
    return names
      .filter((name) => !name.endsWith(TEMPORARY_SUFFIX))
      .map((name) => `${directory}/${name}`);
  };
  // Spread into the result rather than written there: the DOM lib's `Cache` requires these members
  // while Deno's lib `Cache` does not declare them, so as written properties they fail Deno's
  // excess-property check (`deno publish` type-checks with Deno's lib). Spread members are still
  // checked against the DOM `Cache`.
  const unsupportedMembers = {
    add: () => unsupported("Cache.add"),
    addAll: () => unsupported("Cache.addAll"),
    matchAll: () => unsupported("Cache.matchAll"),
  };
  return {
    match: async (request, options) => {
      rejectQueryOptions(options);
      const entry = await openEntry(pathOf(request));
      if (entry === undefined) return undefined;
      return new Response(bodyStream(entry), {
        status: entry.header.status,
        statusText: entry.header.statusText,
        headers: entry.header.headers,
      });
    },
    // Write next to the entry, then rename over it: the entry is either absent or complete, a
    // failed write (quota, an aborted transfer, a mismatch the fetch layer detects in flight and
    // turns into a stream error) leaves the previous entry untouched, and a reader that opened the
    // previous file keeps reading that file. Header and body share the file so they are replaced
    // together — a header from one write with the body of another would pass the fetch layer's
    // recorded-hash check.
    put: async (request, response) => {
      if (response.bodyUsed) {
        throw new TypeError("@karume/hub/node: Cache.put: the response body was already consumed");
      }
      const header: EntryHeader = {
        url: keyOf(request),
        status: response.status,
        statusText: response.statusText,
        headers: Object.fromEntries(response.headers),
      };
      const path = pathOf(request);
      const temporary = `${path}.${crypto.randomUUID()}${TEMPORARY_SUFFIX}`;
      const handle = await open(temporary, "wx");
      try {
        await writeAll(handle, new TextEncoder().encode(`${JSON.stringify(header)}\n`));
        if (response.body !== null) {
          for await (const chunk of response.body) await writeAll(handle, chunk);
        }
        await handle.sync();
        await handle.close();
        await rename(temporary, path);
      } catch (error) {
        await handle.close().catch(() => {});
        await unlink(temporary).catch(() => {});
        throw error;
      }
    },
    delete: async (request, options) => {
      rejectQueryOptions(options);
      try {
        await unlink(pathOf(request));
        return true;
      } catch (error) {
        if (isNotFound(error)) return false;
        throw error;
      }
    },
    keys: async (request, options) => {
      rejectQueryOptions(options);
      const paths = request === undefined ? await entryPaths() : [pathOf(request)];
      const requests: Request[] = [];
      for (const path of paths) {
        const entry = await openEntry(path);
        if (entry === undefined) continue;
        await entry.handle.close();
        requests.push(new Request(entry.header.url));
      }
      return requests;
    },
    ...unsupportedMembers,
  };
};

const nodeCacheStorageAt = (root: string): CacheStorage => {
  const directoryOf = (name: string): string =>
    `${root}/${NAMESPACE_PREFIX}${encodeURIComponent(name)}`;
  const has = async (name: string): Promise<boolean> => {
    try {
      return (await stat(directoryOf(name))).isDirectory();
    } catch (error) {
      if (isNotFound(error)) return false;
      throw error;
    }
  };
  return {
    open: async (name) => {
      const directory = directoryOf(name);
      await mkdir(directory, { recursive: true });
      return nodeCache(directory);
    },
    has,
    delete: async (name) => {
      if (!(await has(name))) return false;
      await rm(directoryOf(name), { recursive: true, force: true });
      return true;
    },
    keys: async () => {
      const entries = await readdir(root, { withFileTypes: true }).catch((error: unknown) => {
        if (isNotFound(error)) return [];
        throw error;
      });
      return entries
        .filter((entry) => entry.isDirectory() && entry.name.startsWith(NAMESPACE_PREFIX))
        .map((entry) => decodeURIComponent(entry.name.slice(NAMESPACE_PREFIX.length)));
    },
    match: () => unsupported("CacheStorage.match"),
  };
};

/**
 * A `CacheStorage` on the Node.js file system for the fetch layer (`@hdae/fetch-cache`) behind
 * hub's Hugging Face source. Node has no Cache API, so a Node application passes this as `caches`
 * to `loadManifest` / `fetchAssets` / `prefetchAssets` / `openAsset` / `clearHubCache`. `root` is a
 * directory path (relative paths resolve against the current working directory) or a `file:` URL;
 * it is created on first use.
 *
 * Layout: one directory per namespace, one file per entry named by the SHA-256 of the key URL. The
 * file holds a JSON header line (URL, status, headers — the fetch layer's recorded hash travels in
 * the headers) followed by the body; `match` streams the body back in 1 MiB reads, and `put`
 * writes a temporary file and renames it over the entry (see `nodeCache`). Several processes may
 * share one root: writers of the same key each complete a whole file and the last rename wins,
 * readers keep the file they opened.
 *
 * Supported: `caches.open / has / delete / keys` and `cache.match / put / delete / keys`, without
 * query options. `add`, `addAll`, `matchAll` and `caches.match` reject — the fetch layer never
 * calls them.
 */
export const nodeCacheStorage = (root: string | URL): CacheStorage =>
  nodeCacheStorageAt(typeof root === "string" ? root : fileURLToPath(root));
