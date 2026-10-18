/**
 * The local source adapter — the source for when "the distribution is already here". Pure Web
 * standard (this file references no runtime-specific API). Reading the files is delegated to the
 * one surface {@link DirectoryAdapter}, into which Deno / Node / OPFS / IndexedDB / the File System
 * Access picker are plugged.
 *
 * The difference from the HF source (`hf.ts`) is in the capabilities themselves:
 *
 * - **No generations** — a local directory has only "what is there now", with neither a mutable ref
 *   nor a commit SHA. No implicit-`main` warning either (there is nothing to pin).
 * - **No phase 1 (prefetch)** — it does not go through CacheStorage, so "warming" means nothing
 *   (`source.ts` (4)). Adding not a single copy of the bytes is the local source's greatest
 *   advantage; copying into a cache could only do harm.
 * - **Range reads (`source.ts` (8)) are possible** — `openFile` exists only when the adapter has
 *   positional reads (`readFileRange`), and the cost class is `"seek"` (independent of the offset).
 *   (8) checks **only the bounds of the declared size**; neither the whole-file size check
 *   (`sizeViolation`) nor sha256 applies (reading only a range, there is no whole file at hand to
 *   compare with the declaration) — corruption of the file shows up as the values of the rows read.
 * - **Verification is an exact size match only** (the sha256 is trusted). Local files are the
 *   user's assets, not "fetched copies" of what the distributor serves, and there is no threat that
 *   justifies hashing the whole thing (several GiB) on every start. The size is known for free once
 *   reading finishes, so it stays as a check (a copy cut off midway or a mix-up with another quant
 *   fails here).
 * - **Cross-repo references by explicit mapping only** — no guessing of an adjacent directory of
 *   the same name (`originFor`).
 * - **No inventory deletion** — only "what was fetched and kept" can be deleted, and a directory's
 *   contents are the user's assets. The inventory query (`source.ts` (6)) answers "everything is
 *   there".
 *
 * MUST NOT: assemble errors here (the common layer holds the diagnostic context — `context.ts`).
 * The exception is a bare `Error` reporting that "the caller's configuration is incomplete" (an
 * unmapped cross-repo target); the common layer wraps it as a fetch failure, keeping it as `cause`.
 */

import { type FileRef, fileRefKey, MANIFEST_FILENAME, MAX_MANIFEST_BYTES } from "../manifest.ts";
import type { LoadManifestOptions } from "../session.ts";
import {
  type AssetRangeReader,
  DistributionSource,
  driverOf,
  type PinnedSource,
  type SourceDriver,
  type SourceOrigin,
} from "../source.ts";

/**
 * The reader of one directory. **The only required method is `readFile`**; besides `denoDirectory`
 * of `@karume/hub/deno` (`Deno.readFile`) and `nodeDirectory` of `@karume/hub/node`, in browsers
 * OPFS (`FileSystemDirectoryHandle` → `File.arrayBuffer()`), IndexedDB and the File System Access
 * picker fit the same shape.
 *
 * MUST: the `Uint8Array` that `readFile` returns **spans its whole buffer** (a tight view) — the
 * common layer passes the bytes received here as is to the `bytes` input of `openContainer`, so
 * returning a view with slack would double the RAM peak with a compensating `slice` (the common
 * layer's tight view check fails it on the spot).
 * MUST: a missing file fails loudly (never return `undefined` or empty bytes). The error carries
 * **the path of the file** — the common layer can add only the relative path in the manifest, and
 * only the adapter knows "under which directory it looked".
 * MUST: pass `signal` through (reads of large parts must be cancellable).
 *
 * `readFileRange` (optional) is the file side of **range reads** (`source.ts` (8)): it returns
 * exactly `length` bytes of `[offset, offset + length)` (throwing if there are not enough — never a
 * short return). Directory files support positional reads, so an adapter with this method gets a
 * reader with cost class `"seek"`. With an adapter without it (a reader that cannot fetch just a
 * range cheaply), `openFile` does not exist at all, and the consumer falls back to a full read.
 */
export type DirectoryAdapter = {
  readonly readFile: (
    path: string,
    options: { readonly signal?: AbortSignal },
  ) => Promise<Uint8Array<ArrayBuffer>>;
  readonly readFileRange?: (
    path: string,
    offset: number,
    length: number,
    options: { readonly signal?: AbortSignal },
  ) => Promise<Uint8Array<ArrayBuffer>>;
};

/** Settings of {@link localDirectory}. */
export type LocalDirectoryOptions = {
  /**
   * The directory name shown in diagnostics (it appears in every error message as the directory
   * label). When omitted, a fixed placeholder (see `localDirectory`) — an adapter that knows the
   * path should always pass it (`denoDirectory` and `nodeDirectory` put the `root` they were given
   * here).
   */
  readonly label?: string;
  /**
   * The sources for cross-repo references (`repo` + `revision` of a `FileRef` — ADR 0038 §7). The
   * key is the `"owner/name"` the manifest declares; the value is a source that provides that repo
   * **as a whole**.
   *
   * MUST: look only at explicit mappings — a guess such as "if there is a same-named directory next
   * door, use it" would silently read mixed-up bytes (they pass if the size matches). Without a
   * declaration, fail.
   *
   * NOTE: there is no field for a fallback delegate for cross-repo targets missing from the mapping
   * (a source that accepts any repo). The delegate would be an HF repo source, but there is no
   * public API that builds one as a {@link DistributionSource} handle (`loadManifest` /
   * `fromPretrained` resolve a `HubRepoRef` internally), so even with the field, no value the
   * public surface can create would work as a delegate. It is added once a public factory exists.
   */
  readonly crossRepo?: Readonly<Record<string, DistributionSource>>;
};

/** Opens a cross-repo target at the declared (repo, revision) coordinates. */
const pinTarget = (
  target: DistributionSource,
  revision: string,
  options: LoadManifestOptions,
): PinnedSource => driverOf(target).pin(revision, options);

const missingCrossRepo = (label: string, repo: string, revision: string): Error =>
  new Error(
    `@karume/hub: ローカル取得元（ディレクトリ ${label}）に repo '${repo}' の越境先が無い。` +
      `localDirectory の crossRepo に { "${repo}": <取得元> } を渡すこと` +
      `（宣言 revision ${revision} — 隣接する同名ディレクトリを推測して読むことはしない）`,
  );

/**
 * (8) The range reader (`source.ts`). **Opening does not touch the file** — reads happen per
 * `read`, so a reader opened and never used runs no I/O at all.
 *
 * The range check is **not pushed down to the adapter** because the adapter knows only the file:
 * the file can be longer than the declared `size` (a mixed-up quant, a copy still being written),
 * so checking against the file length would silently read bytes outside the manifest. Only this
 * place (the source) holds the declared size.
 */
const localRangeReader = (
  readRange: NonNullable<DirectoryAdapter["readFileRange"]>,
  ref: FileRef,
): AssetRangeReader => ({
  // Directory files support positional reads, so the cost does not depend on the offset.
  cost: "seek",
  read: async (offset, length, options = {}) => {
    if (
      !Number.isInteger(offset) ||
      !Number.isInteger(length) ||
      offset < 0 ||
      length < 0 ||
      offset + length > ref.size
    ) {
      throw new Error(
        `hub: ${ref.path} の区間 [${offset}, ${offset + length}) が不正` +
          `（宣言 size ${ref.size} — 非負整数で size に収まる区間だけを読める）`,
      );
    }
    const { signal } = options;
    const bytes = await readRange(ref.path, offset, length, {
      ...(signal === undefined ? {} : { signal }),
    });
    // MUST: do not let a length mismatch through silently (if short, the consumer reads zero-filled
    // rows as valid values; if long, bytes outside the request slip in). Short and long have
    // different causes, so the messages differ.
    if (bytes.byteLength !== length) {
      const cause =
        bytes.byteLength < length
          ? `実体が宣言 size ${ref.size} より短い`
          : "アダプターが要求より長い戻りを返した（`readFileRange` の契約違反）";
      throw new Error(
        `hub: ${ref.path} の区間読みが offset ${offset} の ${length} バイト要求に` +
          ` ${bytes.byteLength} バイトを返した（${cause}）`,
      );
    }
    return bytes;
  },
});

const pinnedLocalSource = (
  adapter: DirectoryAdapter,
  settings: LocalDirectoryOptions & { readonly label: string },
  options: LoadManifestOptions,
): PinnedSource => {
  // Check the capability once here (binding it keeps the type narrowed inside the reader too).
  const readRange = adapter.readFileRange;
  const origin: SourceOrigin = {
    label: `ディレクトリ ${settings.label}`,
    // A failure origin that returns the same bytes when read again (no room for retry, unlike the
    // network).
    integrity: "local",
  };
  return {
    origin,

    readManifest: async ({ parse, signal, sizeViolation }) => {
      const bytes = await adapter.readFile(MANIFEST_FILENAME, {
        ...(signal === undefined ? {} : { signal }),
      });
      // NOTE: the check runs after reading everything (the adapter has only full reads). It is a
      // format check on a local file, not a check that assumes a malicious sender, so failing after
      // reading it all is fine (for the HF source's `karume.json` too, the fetch layer has no cap
      // without exact match, so `parseManifest` checks it after the full receive —
      // `sources/hf.ts`).
      if (bytes.byteLength > MAX_MANIFEST_BYTES) {
        throw sizeViolation(bytes.byteLength, origin.integrity);
      }
      // MUST: let parse's throw out as is (locally there is no cache to evict, so a broken manifest
      // correctly fails with the same ManifestFormatError every time).
      parse(bytes);
    },

    readFile: async (ref, { signal, sizeViolation }) => {
      const abort = signal === undefined ? {} : { signal };
      const bytes = await adapter.readFile(ref.path, abort);
      // Verification is an exact size match only (the sha256 is trusted). onProgress is never
      // called — there is no "midway through receiving" state, so the common layer closes with the
      // single complete.
      if (bytes.byteLength !== ref.size) {
        throw sizeViolation(bytes.byteLength, origin.integrity);
      }
      return bytes;
    },

    // No phase 1 (prefetchFile) — see the module doc above.

    // (8) Range reads exist only when the adapter has positional reads (an optional capability in
    // the same style as `prefetchFile` — with an adapter without it the reader does not appear at
    // all, and the consumer falls back to a full read).
    ...(readRange === undefined
      ? {}
      : {
          // The cancellation check on open is a convention of the common layer (`openAsset` in
          // `fetch.ts`), so it is not here. Per-read cancellation passes the signal of
          // `AssetRangeReader.read` through to the adapter.
          openFile: (ref: FileRef) => Promise.resolve(localRangeReader(readRange, ref)),
        }),

    originFor: (repo, revision) => {
      const mapped = settings.crossRepo;
      if (mapped === undefined || !Object.hasOwn(mapped, repo)) {
        throw missingCrossRepo(settings.label, repo, revision);
      }
      return pinTarget(mapped[repo], revision, options);
    },

    // (6) The inventory is always "everything passed in is there" — by the same reasoning as having
    // no phase 1, for a source that reads directly "a later read is cheap" holds from the start. A
    // missing file fails when read (walking the directory on an inventory query would pay I/O per
    // asset on every query).
    inventory: (refs) => Promise.resolve(new Set(refs.map(fileRefKey))),
    // No (7) deletion — a directory's contents are the user's assets, not fetched copies.
  };
};

/**
 * Makes a local directory a source. It fits the same surfaces as the HF source, in the form
 * `loadManifest(localDirectory(adapter), …)` / `fromPretrained(localDirectory(adapter))`.
 *
 * ```ts ignore
 * import { localDirectory, loadManifest } from "@karume/hub";
 * import { denoDirectory } from "@karume/hub/deno";
 *
 * const loaded = await loadManifest(denoDirectory("./models/karume-gemma4"));
 * ```
 *
 * For relative subdirectories of cross-repo references, **create another source and pass it**
 * (`crossRepo: { "owner/name": denoDirectory("./models/other") }`) — keeping one source = one
 * directory lets a mapping value be swapped for a remote source as is.
 */
export const localDirectory = (
  adapter: DirectoryAdapter,
  options: LocalDirectoryOptions = {},
): DistributionSource => {
  const settings = { ...options, label: options.label ?? "(ローカル)" };
  const driver: SourceDriver = {
    origin: { label: `ディレクトリ ${settings.label}`, integrity: "local" },
    // There is no notion of generations, so no resolution or request happens (no implicit-main
    // warning either — there is nothing to pin).
    resolveGeneration: () => Promise.resolve(""),
    pin: (_generation, callOptions) => pinnedLocalSource(adapter, settings, callOptions),
  };
  return new DistributionSource(driver);
};
