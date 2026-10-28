/**
 * Dev-server plugin that serves local Karume distributions read-only under `/models/<name>/`.
 *
 * The page reads a distribution through `localDirectory` of `@karume/hub` over `fetch` (whole files
 * and `Range` reads), so the server only has to hand out bytes: no listing, no writes, nothing
 * outside the mounted directories. The request handling is a pure function over (method, url,
 * headers) → response description ({@link handleDistributionRequest}) so the rules are testable
 * without a server; the plugin only adapts it to Node's `http` objects and streams file ranges.
 *
 * Cross-origin isolation: every response of the dev server carries COOP / COEP (the page needs
 * `crossOriginIsolated` for SharedArrayBuffer and precise timers, as the Deno-era gpu-lab page did),
 * and the distribution files carry CORP `same-origin` so they load under `require-corp`.
 *
 * A missing distribution directory does not stop the server: it logs a warning at startup, answers
 * 404 for that prefix, and `/models/index.json` reports it as unavailable so the page can show a
 * disabled state that names the directory.
 *
 * ## The HF-compatible route (`/hf/`)
 *
 * A mount with an `hfRepo` is also served under `/hf/` in the two shapes the HF source of
 * `@karume/hub` requests (`resolveHfRevision` / `hfResolveUrl` of `@hdae/fetch-cache/hf`):
 * `/hf/api/models/{repo}/revision/{ref}` and `/hf/{repo}/resolve/{revision}/{path}`. With
 * `hubUrl = <origin>/hf` the page then reads the distribution through the hub's CacheStorage, so a
 * second generation does not move the weights over the network again.
 *
 * The revision of such a repo is the first 40 hex digits of the sha256 of its `karume.json`
 * ({@link hfRevisionOf}), recomputed on every request. Re-assembling the mirror therefore changes the
 * revision at once, and the hub's manifest cache entry (keyed by the revision-pinned URL) can never
 * hand out a stale manifest. Asset entries are keyed by content (repo + path + sha256) and stay valid
 * across revisions. Cross-repo references of the manifest (umT5 for Wan) are served at the
 * (repo, revision) the manifest declares, from the mount whose directory is named after the last
 * element of the repo name (`hdae/karume-umt5-xxl` → `models/karume-umt5-xxl`).
 *
 * ### Referenced revisions (aliases)
 *
 * One repo can be both served from its own mount and referenced by another served manifest at a
 * different revision: `hdae/karume-anima-extra` takes the text encoder, the VAE and the tokenizers
 * from `hdae/karume-anima` at the revision its manifest declares, while the `karume-anima` mount is
 * served at its own computed revision. By default that fails loudly (one directory holds one
 * revision). A mount with `aliasReferencedRevisions` lets each such declared revision **alias** its
 * directory, under two MUSTs:
 *
 * - every file a manifest takes from the repo at that revision is listed in the mount's own manifest
 *   (as its own file, not a cross-repo one) with the same path, size and sha256 — otherwise the table
 *   fails loudly, because the alias would serve different bytes under the declared revision;
 * - the alias serves only those files (anything else at that revision answers 404), so it never
 *   claims to hold more of the old revision than the references prove.
 *
 * The primary revision (the computed one) is unchanged: the revision API answers it for any ref
 * except an alias revision, which it answers as itself. The aliases are logged at startup. The direct
 * route (`localDirectory` over `/models/`) needs none of this: it maps cross-repo references to mounts
 * explicitly (`crossRepo`) and checks no revision.
 */

import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { access, readFile, realpath, stat } from "node:fs/promises";
import type { IncomingMessage, ServerResponse } from "node:http";
import { basename, dirname, join, relative, resolve, sep } from "node:path";
import { type FileRef, MANIFEST_FILENAME, type Manifest, parseManifest } from "@karume/hub";
import type { Plugin } from "vite-plus";

/** One served distribution: `/models/<name>/…` → files under `root`. */
export type DistributionMount = {
  readonly name: string;
  /** Absolute path of the distribution directory (may be a symlink). */
  readonly root: string;
  /** What the page and the logs show for the directory (relative to the repository root). */
  readonly label: string;
  /** Also serve the mount on the HF-compatible route under this repo name (`owner/name`). */
  readonly hfRepo?: string;
  /**
   * With `hfRepo`: let the revisions other served manifests reference this repo at alias the
   * directory, for the files they name (module doc, "Referenced revisions").
   */
  readonly aliasReferencedRevisions?: boolean;
};

export type DistributionRequest = {
  readonly method: string;
  /** The raw request target (`/models/wan22/karume.json?x`), not normalized. */
  readonly url: string;
  readonly headers: Readonly<Record<string, string | undefined>>;
};

/** What `/models/index.json` returns: one entry per mount. */
export type DistributionIndex = Readonly<
  Record<
    string,
    {
      readonly directory: string;
      readonly available: boolean;
      /** Only for an available mount with an `hfRepo`: what the page passes as its `HubRepoRef`. */
      readonly hf?: { readonly repo: string; readonly revision: string };
    }
  >
>;

export type DistributionResponse =
  /** Not a distribution URL: Vite answers it; the plugin only adds `headers`. */
  | { readonly kind: "pass"; readonly headers: Readonly<Record<string, string>> }
  | {
      readonly kind: "respond";
      readonly status: number;
      readonly headers: Readonly<Record<string, string>>;
      /** `undefined` = no body (HEAD, errors without text). `end` is inclusive. */
      readonly body?:
        | {
            readonly kind: "file";
            readonly path: string;
            readonly start: number;
            readonly end: number;
          }
        | { readonly kind: "text"; readonly text: string };
    };

/** Headers on every dev-server response (cross-origin isolation of the page). */
export const ISOLATION_HEADERS: Readonly<Record<string, string>> = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

const RESOURCE_HEADERS: Readonly<Record<string, string>> = {
  ...ISOLATION_HEADERS,
  "Cross-Origin-Resource-Policy": "same-origin",
  "X-Content-Type-Options": "nosniff",
};

const PREFIX = "/models/";
export const INDEX_PATH = "/models/index.json";
export const HF_PREFIX = "/hf/";
/** `{hubUrl}/api/models/{repo}/revision/{ref}` (`resolveHfRevision`). */
const HF_REVISION_ROUTE = /^\/hf\/api\/models\/(.+)\/revision\/([^/]+)$/;
/** `{hubUrl}/{repo}/resolve/{revision}/{path}` (`hfResolveUrl`). */
const HF_RESOLVE_ROUTE = /^\/hf\/(.+?)\/resolve\/([^/]+)\/(.+)$/;

const contentType = (path: string): string => {
  if (path.endsWith(".json")) return "application/json";
  if (path.endsWith(".md")) return "text/markdown; charset=utf-8";
  return "application/octet-stream";
};

const respond = (
  status: number,
  extra: Readonly<Record<string, string>> = {},
  body?: Extract<DistributionResponse, { kind: "respond" }>["body"],
): DistributionResponse => ({
  kind: "respond",
  status,
  headers: { ...RESOURCE_HEADERS, ...extra },
  ...(body === undefined ? {} : { body }),
});

const notFound = (text: string): DistributionResponse =>
  respond(404, { "Content-Type": "text/plain; charset=utf-8" }, { kind: "text", text });

const isMissing = (error: unknown): boolean =>
  error instanceof Error &&
  "code" in error &&
  (error.code === "ENOENT" || error.code === "ENOTDIR");

/** Absolute real path of the mount, or `undefined` when it has no `karume.json`. */
const realRoot = async (mount: DistributionMount): Promise<string | undefined> => {
  try {
    const root = await realpath(mount.root);
    await access(join(root, "karume.json"));
    return root;
  } catch (error) {
    if (isMissing(error)) return undefined;
    throw error;
  }
};

/**
 * Splits the path below a mount into decoded segments; `undefined` for anything that could leave the
 * directory or name it ambiguously (empty, `.`, `..`, separators or NUL inside a segment).
 */
const safeSegments = (rest: string): readonly string[] | undefined => {
  let segments: string[];
  try {
    segments = rest.split("/").map((segment) => decodeURIComponent(segment));
  } catch {
    return undefined;
  }
  const unsafe = segments.some(
    (segment) =>
      segment === "" ||
      segment === "." ||
      segment === ".." ||
      segment.includes("/") ||
      segment.includes("\\") ||
      segment.includes("\0"),
  );
  return unsafe ? undefined : segments;
};

/** `bytes=a-b` / `bytes=a-` / `bytes=-n` → inclusive [start, end]; `undefined` = unsatisfiable. */
const parseRange = (
  header: string,
  size: number,
): { readonly start: number; readonly end: number } | undefined => {
  const match = /^bytes=(\d*)-(\d*)$/.exec(header.trim());
  if (match === null) return undefined;
  const [, first = "", last = ""] = match;
  if (first === "" && last === "") return undefined;
  if (first === "") {
    // Suffix range: the last n bytes.
    const length = Number(last);
    if (!Number.isSafeInteger(length) || length === 0 || size === 0) return undefined;
    return { start: Math.max(0, size - length), end: size - 1 };
  }
  const start = Number(first);
  const end = last === "" ? size - 1 : Math.min(Number(last), size - 1);
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end)) return undefined;
  if (start >= size || start > end) return undefined;
  return { start, end };
};

/**
 * Serves one file below `root` (GET / HEAD, with `Range`). `segments` went through
 * {@link safeSegments}; the real path is checked again because a symlink inside the directory could
 * point anywhere.
 */
const serveFile = async (
  root: string,
  segments: readonly string[],
  request: DistributionRequest,
  cacheControl: string,
): Promise<DistributionResponse> => {
  let file: string;
  try {
    file = await realpath(join(root, ...segments));
  } catch (error) {
    if (isMissing(error)) return notFound("Asset not found");
    throw error;
  }
  // MUST: check the real path (a symlink inside the mount could point anywhere).
  if (!file.startsWith(`${root}${sep}`)) return notFound("Asset outside the distribution");
  const info = await stat(file);
  if (!info.isFile()) return notFound("Not a file");

  const size = info.size;
  const base = {
    "Content-Type": contentType(file),
    "Accept-Ranges": "bytes",
    "Cache-Control": cacheControl,
  };
  const rangeHeader = request.headers["range"];
  let start = 0;
  let end = size - 1;
  let status = 200;
  const extra: Record<string, string> = { ...base };
  if (rangeHeader !== undefined) {
    const range = parseRange(rangeHeader, size);
    if (range === undefined) return respond(416, { ...base, "Content-Range": `bytes */${size}` });
    ({ start, end } = range);
    status = 206;
    extra["Content-Range"] = `bytes ${start}-${end}/${size}`;
  }
  // MUST: declare the length. The hub's received-byte check needs it to match the manifest's `size`.
  extra["Content-Length"] = String(end - start + 1);
  if (request.method === "HEAD" || size === 0) return respond(status, extra);
  return respond(status, extra, { kind: "file", path: file, start, end });
};

/** The revision a mirror is served at: the first 40 hex digits of the sha256 of its `karume.json`. */
export const hfRevisionOf = (manifestBytes: Uint8Array): string =>
  createHash("sha256").update(manifestBytes).digest("hex").slice(0, 40);

/** One repo of the HF-compatible route: the revision it answers and the directory it reads. */
type HfServedRepo = {
  readonly revision: string;
  readonly mount: DistributionMount;
  /** The real path of the mount's directory. */
  readonly root: string;
  /** Alias revision → the paths it serves (module doc, "Referenced revisions"). */
  readonly aliases: ReadonlyMap<string, ReadonlySet<string>>;
};

/** A referenced revision served as an alias of a mount's directory (what the startup log names). */
export type HfAlias = {
  readonly repo: string;
  readonly revision: string;
  /** The revision the repo is served at (sha256 of its `karume.json`). */
  readonly servedRevision: string;
  /** The mount's directory as the logs show it. */
  readonly directory: string;
  /** The repos whose manifests reference the repo at this revision. */
  readonly referencedBy: readonly string[];
  /** How many files the alias serves. */
  readonly files: number;
};

/** The HF-compatible serving table, plus the aliases and the cross-repo references no mount holds. */
export type HfServingTable = {
  readonly repos: ReadonlyMap<string, HfServedRepo>;
  readonly aliases: readonly HfAlias[];
  /** `repo@revision` of each cross-repo reference without a mount (requests for it answer 404). */
  readonly unserved: readonly string[];
};

/** Every file reference of the manifest (all models × weights × dtypes, plus the assets). */
const allFileRefs = (manifest: Manifest): readonly FileRef[] =>
  Object.values(manifest.models).flatMap((model) => [
    ...Object.values(model.weights).flatMap((entry) =>
      Object.values(entry).flatMap((dtype) => dtype.container.parts),
    ),
    ...Object.values(model.assets),
  ]);

/**
 * Builds the serving table from the mounts as they are on disk now (called per request, so a
 * re-assembled mirror is picked up at once). Each available mount with an `hfRepo` is served under
 * that name at {@link hfRevisionOf}; each cross-repo reference of its manifest is served at the
 * declared (repo, revision) from the mount whose directory is named after the last element of the
 * repo name.
 *
 * MUST: two different revisions for one repo fail loudly — one directory holds one revision, and
 * answering both would serve the wrong bytes under one of them. The one exception is a mount with
 * `aliasReferencedRevisions`, and only while every referenced file matches its own manifest (module
 * doc, "Referenced revisions").
 */
export const hfServingTable = async (
  mounts: readonly DistributionMount[],
): Promise<HfServingTable> => {
  /** A repo being built: `ownFiles` only for a mount served under its own `hfRepo`. */
  type Building = {
    readonly revision: string;
    readonly mount: DistributionMount;
    readonly root: string;
    readonly ownFiles?: ReadonlyMap<string, FileRef>;
    readonly aliases: Map<string, { files: Set<string>; referencedBy: Set<string> }>;
  };
  const repos = new Map<string, Building>();
  const manifests: { readonly repo: string; readonly manifest: Manifest }[] = [];
  for (const mount of mounts) {
    if (mount.hfRepo === undefined) continue;
    const root = await realRoot(mount);
    if (root === undefined) continue;
    if (repos.has(mount.hfRepo)) {
      throw new Error(`Two distribution mounts claim the HF repo ${mount.hfRepo}`);
    }
    const bytes = await readFile(join(root, MANIFEST_FILENAME));
    const manifest = parseManifest(new TextDecoder().decode(bytes));
    const ownFiles = new Map(
      allFileRefs(manifest)
        .filter((ref) => ref.repo === undefined)
        .map((ref) => [ref.path, ref] as const),
    );
    repos.set(mount.hfRepo, {
      revision: hfRevisionOf(bytes),
      mount,
      root,
      ownFiles,
      aliases: new Map(),
    });
    manifests.push({ repo: mount.hfRepo, manifest });
  }
  const unserved = new Set<string>();
  for (const { repo: referrer, manifest } of manifests) {
    for (const ref of allFileRefs(manifest)) {
      if (ref.repo === undefined || ref.revision === undefined) continue;
      const known = repos.get(ref.repo);
      if (known !== undefined) {
        if (known.revision === ref.revision) continue;
        if (known.mount.aliasReferencedRevisions !== true || known.ownFiles === undefined) {
          throw new Error(
            `HF repo ${ref.repo} is referenced at ${ref.revision} but served at ${known.revision} — one directory holds one revision`,
          );
        }
        const own = known.ownFiles.get(ref.path);
        if (own === undefined || own.size !== ref.size || own.sha256 !== ref.sha256) {
          throw new Error(
            `HF repo ${ref.repo} is referenced at ${ref.revision} by ${referrer} for ${ref.path} (${ref.size} B, sha256 ${ref.sha256}), but the manifest of ${known.mount.label} ${
              own === undefined
                ? "does not list that file as its own"
                : `lists it as ${own.size} B, sha256 ${own.sha256}`
            } — aliasing the revision would serve different bytes`,
          );
        }
        const alias = known.aliases.get(ref.revision) ?? {
          files: new Set(),
          referencedBy: new Set(),
        };
        alias.files.add(ref.path);
        alias.referencedBy.add(referrer);
        known.aliases.set(ref.revision, alias);
        continue;
      }
      const directory = ref.repo.slice(ref.repo.lastIndexOf("/") + 1);
      const mount = mounts.find(({ root }) => basename(root) === directory);
      const root = mount === undefined ? undefined : await realRoot(mount);
      if (mount === undefined || root === undefined) unserved.add(`${ref.repo}@${ref.revision}`);
      else repos.set(ref.repo, { revision: ref.revision, mount, root, aliases: new Map() });
    }
  }
  const served = new Map<string, HfServedRepo>();
  const aliases: HfAlias[] = [];
  for (const [repo, building] of repos) {
    served.set(repo, {
      revision: building.revision,
      mount: building.mount,
      root: building.root,
      aliases: new Map([...building.aliases].map(([revision, { files }]) => [revision, files])),
    });
    for (const [revision, { files, referencedBy }] of building.aliases) {
      aliases.push({
        repo,
        revision,
        servedRevision: building.revision,
        directory: building.mount.label,
        referencedBy: [...referencedBy],
        files: files.size,
      });
    }
  }
  return { repos: served, aliases, unserved: [...unserved] };
};

const decodeComponent = (text: string): string | undefined => {
  try {
    return decodeURIComponent(text);
  } catch {
    return undefined;
  }
};

/**
 * `/hf/…`: the revision API answers `{ "sha": <revision> }` for any ref of a served repo, and the
 * resolve URL serves a file only at the repo's current revision. Responses are `no-store`: the hub
 * keeps its own copy in CacheStorage, and a browser HTTP-cache copy of several GiB would only double
 * the disk use.
 */
const handleHfRequest = async (
  path: string,
  request: DistributionRequest,
  mounts: readonly DistributionMount[],
): Promise<DistributionResponse> => {
  const { repos } = await hfServingTable(mounts);
  const revisionRoute = HF_REVISION_ROUTE.exec(path);
  if (revisionRoute !== null) {
    const served = repos.get(decodeComponent(revisionRoute[1] ?? "") ?? "");
    if (served === undefined) return notFound("Unknown repository");
    const ref = decodeComponent(revisionRoute[2] ?? "") ?? "";
    const text = JSON.stringify({ sha: served.aliases.has(ref) ? ref : served.revision });
    return respond(
      200,
      { "Content-Type": "application/json", "Cache-Control": "no-store" },
      request.method === "HEAD" ? undefined : { kind: "text", text },
    );
  }
  const resolveRoute = HF_RESOLVE_ROUTE.exec(path);
  if (resolveRoute === null) return notFound("Unknown route");
  const [, repo = "", revision = "", rest = ""] = resolveRoute;
  const served = repos.get(decodeComponent(repo) ?? "");
  if (served === undefined) return notFound("Unknown repository");
  const requested = decodeComponent(revision) ?? "";
  // An alias revision serves only the files the references name (module doc).
  const aliasFiles = requested === served.revision ? undefined : served.aliases.get(requested);
  if (requested !== served.revision && aliasFiles === undefined) {
    return notFound("Unknown revision");
  }
  const segments = safeSegments(rest);
  if (segments === undefined) return notFound("Invalid asset path");
  if (aliasFiles !== undefined && !aliasFiles.has(segments.join("/"))) {
    return notFound("Not a file the aliased revision is referenced for");
  }
  return await serveFile(served.root, segments, request, "no-store");
};

/**
 * The whole request policy. `/models/index.json` lists the mounts; `/models/<name>/<path>` serves a
 * file of that mount (GET / HEAD, with `Range`); `/hf/…` is the HF-compatible route
 * ({@link handleHfRequest}); everything else passes to Vite with the isolation headers.
 */
export const handleDistributionRequest = async (
  request: DistributionRequest,
  mounts: readonly DistributionMount[],
): Promise<DistributionResponse> => {
  const path = request.url.split(/[?#]/, 1)[0] ?? "";
  const hf = path.startsWith(HF_PREFIX);
  if (!hf && !path.startsWith(PREFIX)) return { kind: "pass", headers: ISOLATION_HEADERS };
  if (request.method !== "GET" && request.method !== "HEAD") {
    return respond(405, { Allow: "GET, HEAD" });
  }
  if (hf) return await handleHfRequest(path, request, mounts);

  if (path === INDEX_PATH) {
    const { repos } = await hfServingTable(mounts);
    const index: Record<string, DistributionIndex[string]> = {};
    for (const mount of mounts) {
      const served = mount.hfRepo === undefined ? undefined : repos.get(mount.hfRepo);
      index[mount.name] = {
        directory: mount.label,
        available: (await realRoot(mount)) !== undefined,
        ...(mount.hfRepo === undefined || served === undefined
          ? {}
          : { hf: { repo: mount.hfRepo, revision: served.revision } }),
      };
    }
    const text = JSON.stringify(index);
    return respond(
      200,
      { "Content-Type": "application/json", "Cache-Control": "no-store" },
      request.method === "HEAD" ? undefined : { kind: "text", text },
    );
  }

  const rest = path.slice(PREFIX.length);
  const slash = rest.indexOf("/");
  const mount = slash < 0 ? undefined : mounts.find(({ name }) => name === rest.slice(0, slash));
  if (mount === undefined) return notFound("Unknown distribution");
  const segments = safeSegments(rest.slice(slash + 1));
  if (segments === undefined) return notFound("Invalid asset path");

  const root = await realRoot(mount);
  if (root === undefined) return notFound(`Distribution directory ${mount.label} is missing`);
  return await serveFile(root, segments, request, "no-cache");
};

/** The nearest ancestor of `from` (inclusive) that holds `pnpm-workspace.yaml`. */
const findRepoRoot = async (from: string): Promise<string> => {
  let directory = resolve(from);
  for (;;) {
    try {
      await access(join(directory, "pnpm-workspace.yaml"));
      return directory;
    } catch {
      const parent = dirname(directory);
      if (parent === directory) {
        throw new Error(`No pnpm-workspace.yaml above ${from} — cannot place the distributions`);
      }
      directory = parent;
    }
  }
};

export type DistributionsPluginOptions = {
  /**
   * `/models/<name>/` → directory relative to the repository root (the `pnpm-workspace.yaml`). The
   * object form also serves the mount on the HF-compatible route as `hfRepo` (module doc).
   */
  readonly distributions: Readonly<
    Record<
      string,
      | string
      | {
          readonly directory: string;
          readonly hfRepo: string;
          readonly aliasReferencedRevisions?: boolean;
        }
    >
  >;
};

/** Serves {@link DistributionsPluginOptions.distributions} on the dev and preview servers. */
export const distributionsPlugin = (options: DistributionsPluginOptions): Plugin => {
  let mounts: readonly DistributionMount[] = [];

  const middleware = (
    req: IncomingMessage,
    res: ServerResponse,
    next: (error?: unknown) => void,
  ): void => {
    const headers: Record<string, string | undefined> = {};
    for (const [key, value] of Object.entries(req.headers)) {
      headers[key] = Array.isArray(value) ? value.join(", ") : value;
    }
    handleDistributionRequest(
      { method: req.method ?? "GET", url: req.url ?? "/", headers },
      mounts,
    ).then(
      (response) => {
        for (const [key, value] of Object.entries(response.headers)) res.setHeader(key, value);
        if (response.kind === "pass") {
          next();
          return;
        }
        res.statusCode = response.status;
        const { body } = response;
        if (body === undefined) res.end();
        else if (body.kind === "text") res.end(body.text);
        else {
          const stream = createReadStream(body.path, { start: body.start, end: body.end });
          stream.on("error", (error) => res.destroy(error));
          res.on("close", () => stream.destroy());
          stream.pipe(res);
        }
      },
      (error: unknown) => next(error),
    );
  };

  const resolveMounts = async (
    appRoot: string,
    log: (message: string) => void,
    info: (message: string) => void,
  ): Promise<void> => {
    const repoRoot = await findRepoRoot(appRoot);
    mounts = Object.entries(options.distributions).map(([name, entry]) => {
      const directory = typeof entry === "string" ? entry : entry.directory;
      return {
        name,
        root: resolve(repoRoot, directory),
        label: relative(repoRoot, resolve(repoRoot, directory)),
        ...(typeof entry === "string" ? {} : { hfRepo: entry.hfRepo }),
        ...(typeof entry === "string" || entry.aliasReferencedRevisions !== true
          ? {}
          : { aliasReferencedRevisions: true }),
      };
    });
    for (const mount of mounts) {
      if ((await realRoot(mount)) === undefined) {
        log(
          `[distributions] ${mount.label} has no karume.json — /models/${mount.name}/ answers 404 and the pages that need it are disabled`,
        );
      }
    }
    // The table is rebuilt per request; this pass only reports problems early. A broken manifest
    // does not stop the server (the requests that need the table answer the error).
    try {
      const table = await hfServingTable(mounts);
      for (const alias of table.aliases) {
        info(
          `[distributions] ${alias.repo}@${alias.revision} aliases ${alias.directory} (served at ${alias.servedRevision}) for the ${alias.files} files ${alias.referencedBy.join(", ")} references there — each matches its manifest`,
        );
      }
      for (const reference of table.unserved) {
        log(
          `[distributions] no mount holds the cross-repo reference ${reference} — its ${HF_PREFIX} requests answer 404`,
        );
      }
    } catch (error) {
      log(`[distributions] the ${HF_PREFIX} serving table cannot be built: ${String(error)}`);
    }
  };

  return {
    name: "karume-distributions",
    async configureServer(server) {
      await resolveMounts(
        server.config.root,
        (message) => server.config.logger.warn(message),
        (message) => server.config.logger.info(message),
      );
      server.middlewares.use(middleware);
    },
    async configurePreviewServer(server) {
      await resolveMounts(
        server.config.root,
        (message) => server.config.logger.warn(message),
        (message) => server.config.logger.info(message),
      );
      server.middlewares.use(middleware);
    },
  };
};
