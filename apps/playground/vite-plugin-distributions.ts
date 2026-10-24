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
 */

import { createReadStream } from "node:fs";
import { access, realpath, stat } from "node:fs/promises";
import type { IncomingMessage, ServerResponse } from "node:http";
import { dirname, join, relative, resolve, sep } from "node:path";
import type { Plugin } from "vite-plus";

/** One served distribution: `/models/<name>/…` → files under `root`. */
export type DistributionMount = {
  readonly name: string;
  /** Absolute path of the distribution directory (may be a symlink). */
  readonly root: string;
  /** What the page and the logs show for the directory (relative to the repository root). */
  readonly label: string;
};

export type DistributionRequest = {
  readonly method: string;
  /** The raw request target (`/models/wan22/karume.json?x`), not normalized. */
  readonly url: string;
  readonly headers: Readonly<Record<string, string | undefined>>;
};

/** What `/models/index.json` returns: one entry per mount. */
export type DistributionIndex = Readonly<
  Record<string, { readonly directory: string; readonly available: boolean }>
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
 * The whole request policy. `/models/index.json` lists the mounts; `/models/<name>/<path>` serves a
 * file of that mount (GET / HEAD, with `Range`); everything else passes to Vite with the isolation
 * headers.
 */
export const handleDistributionRequest = async (
  request: DistributionRequest,
  mounts: readonly DistributionMount[],
): Promise<DistributionResponse> => {
  const path = request.url.split(/[?#]/, 1)[0] ?? "";
  if (!path.startsWith(PREFIX)) return { kind: "pass", headers: ISOLATION_HEADERS };
  if (request.method !== "GET" && request.method !== "HEAD") {
    return respond(405, { Allow: "GET, HEAD" });
  }

  if (path === INDEX_PATH) {
    const index: Record<string, { directory: string; available: boolean }> = {};
    for (const mount of mounts) {
      index[mount.name] = {
        directory: mount.label,
        available: (await realRoot(mount)) !== undefined,
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
    "Cache-Control": "no-cache",
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
  extra["Content-Length"] = String(end - start + 1);
  if (request.method === "HEAD" || size === 0) return respond(status, extra);
  return respond(status, extra, { kind: "file", path: file, start, end });
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
  /** `/models/<name>/` → directory relative to the repository root (the `pnpm-workspace.yaml`). */
  readonly distributions: Readonly<Record<string, string>>;
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

  const resolveMounts = async (appRoot: string, log: (message: string) => void): Promise<void> => {
    const repoRoot = await findRepoRoot(appRoot);
    mounts = Object.entries(options.distributions).map(([name, directory]) => ({
      name,
      root: resolve(repoRoot, directory),
      label: relative(repoRoot, resolve(repoRoot, directory)),
    }));
    for (const mount of mounts) {
      if ((await realRoot(mount)) === undefined) {
        log(
          `[distributions] ${mount.label} has no karume.json — /models/${mount.name}/ answers 404 and the pages that need it are disabled`,
        );
      }
    }
  };

  return {
    name: "karume-distributions",
    async configureServer(server) {
      await resolveMounts(server.config.root, (message) => server.config.logger.warn(message));
      server.middlewares.use(middleware);
    },
    async configurePreviewServer(server) {
      await resolveMounts(server.config.root, (message) => server.config.logger.warn(message));
      server.middlewares.use(middleware);
    },
  };
};
