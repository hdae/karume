/**
 * A throwaway server that serves a local distribution directory (one holding `karume.json`) over
 * **HF-shaped HTTP**.
 *
 * ## Its consumers are only those that go through the real download path itself
 *
 * The demos' `--source <local path>` does not come through here — a local distribution can now be
 * read directly through a source handle (`nodeDirectory` of `@karume/hub/node` — `local-source.ts`
 * (removed in 0.15)), which needs neither a port nor a copy into the persistent cache. The two
 * remaining consumers below both exist to go through the **real download path** (revision
 * resolution → resolve URL → received-byte check → persistent cache), so they need this server
 * that speaks pseudo-HF:
 *
 * - `packages/e2e/tests/e2e_gemma4_pretrained_test.ts` — the check that the path works end to end
 * - `--state cold` / `warm` of `tools/ram-peak/measure.ts` (removed in 0.15) — measures the host
 *   RAM peak and the cache behaviour of that path
 *
 * ## Why not read whole files even over HTTP
 *
 * Components are already **split into parts** in the distribution (container — ADR 0109). Whole
 * reading (`local-assets.ts` (removed in 0.15) + `from*Assets`) can read the split form too, but
 * that surface **puts every part in host RAM at once** (a 3.7GiB DiT stays resident as is). Going
 * through the fetch layer lets the fetch surface (descriptor → admission → warming the weight
 * parts → range reads of blocks) take effect as is, and only the neighbourhood of the block being
 * read sits in RAM.
 *
 * It speaks only the 2 paths the hub actually hits (the revision resolution API and the resolve
 * URL — the source of truth for their spelling is `resolveHfRevision` / `hfResolveUrl` of
 * `@hdae/fetch-cache/hf`). Neither Range nor HEAD is needed (the fetch layer fires only plain
 * GETs).
 *
 * ## Cross-repository references are served from the neighbouring mirror
 *
 * A distribution can point outside its own repository (`repo` / `revision` of `FileRef` — ADR 0038
 * §7. Example: the text stack of `karume-anima-extra` points at 1 commit of `hdae/karume-anima`).
 * The fetch layer fetches that one **from the declared (repo, revision), not from the session's
 * (repo, SHA)**, so a server that names only one pair answers 404 for the whole cross-repository
 * share. The serving table is therefore made of **multiple entries** "(repo, revision) → served
 * directory", and a cross-repository target is served from **the adjacent directory named after
 * the last element of the repo name** (the layout where distribution mirrors sit side by side in
 * `models/`).
 *
 * NOTE: the fetch layer copies assets into the **persistent cache** (the directory of `nodeCacheStorage` on Node), so a
 * local distribution is held twice on disk (the second and later launches read from that cache).
 * Clean up with `clearHubCache` of `@karume/hub`.
 */

import { open, type FileHandle } from "node:fs/promises";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { pipeline } from "node:stream/promises";
import {
  type FileRef,
  type HubRepoRef,
  type Manifest,
  MANIFEST_FILENAME,
  parseManifest,
} from "../../hub/mod.ts";
import { isNotFound, readTextFileSync, statSync } from "./fs.ts";

/**
 * The repository name the throwaway server names **as its main repository**. The fetch layer's
 * cache key includes the repo (an asset's key is the content key repo + path + sha256), so it is
 * spelled so as not to collide with a real HF repository. Cross-repository targets are not
 * synthesised: the server names the real repo the manifest named, as is.
 */
const REPO = "karume-local/dist";

/**
 * The fixed SHA the main repository's revision resolution API returns. One distribution directory
 * has only one revision, so the value itself carries no meaning (the fetch layer only requires 40
 * lowercase hex digits).
 */
const REVISION_SHA = "0".repeat(40);

/** `{hubUrl}/api/models/{repo}/revision/{ref}` (`resolveHfRevision`). */
const REVISION_ROUTE = /^\/api\/models\/(.+)\/revision\/(.+)$/;
/** `{hubUrl}/{repo}/resolve/{revision}/{path}` (`hfResolveUrl`). */
const RESOLVE_ROUTE = /^\/(.+?)\/resolve\/([^/]+)\/(.+)$/;

// Same status, body and content-type as the Deno version's `new Response("not found", ...)`.
const notFound = (response: ServerResponse): void => {
  response.writeHead(404, { "content-type": "text/plain;charset=UTF-8" }).end("not found");
};

/**
 * Whether a relative path points only **inside** the distribution directory. Throwaway or not,
 * this is a real server opened on 127.0.0.1, so `..` and absolute paths are not passed straight
 * to a file open.
 */
const isInsideDist = (path: string): boolean =>
  path.length > 0 &&
  !path.startsWith("/") &&
  path.split("/").every((segment) => segment !== "" && segment !== "." && segment !== "..");
/**
 * One entry of the serving table = for one repo, "the revision it names" and "the directory it
 * serves from".
 */
type ServedRepo = {
  readonly revision: string;
  readonly dir: string;
};

/**
 * Every `FileRef` across the manifest (the container parts of every model × every weights entry ×
 * every dtype, plus assets).
 *
 * It does not narrow to one (model, quant) pair with `resolveSelection` because the server comes
 * up **before the selection** (`serveLocalDist` receives only `dir`; model / quant are decided
 * later by `fromPretrained`). Putting too much in the table does no harm: it only adds repos that
 * are never actually fetched.
 *
 * Zero-length parts are dropped (the fetch layer does not fetch them either — ADR 0109 decision 3).
 */
const allFileRefs = (manifest: Manifest): readonly FileRef[] => {
  const refs: FileRef[] = [];
  for (const model of Object.values(manifest.models)) {
    for (const entry of Object.values(model.weights)) {
      for (const dtype of Object.values(entry)) {
        refs.push(...dtype.container.parts.filter((part) => part.size > 0));
      }
    }
    refs.push(...Object.values(model.assets));
  }
  return refs;
};

/**
 * Confirms **at startup** that the body of each cross-repository reference exists in the mirror.
 *
 * MUST: do not let it fail through a 404 mid-run. The fetch layer is a surface that notices a gap
 * only after streaming hundreds of MiB, so rather than failing after making the caller wait, fail
 * at startup.
 */
const assertMirrored = (ref: FileRef, mirror: string): void => {
  try {
    statSync(`${mirror}/${ref.path}`);
  } catch (error) {
    if (!isNotFound(error)) throw error;
    throw new Error(
      `no mirror for a cross-repo reference: ${ref.repo}@${ref.revision} ${ref.path} is not found under ${mirror}` +
        ` (put a distribution mirror in a directory named after the last element of the repo name)`,
    );
  }
};

/**
 * Builds the serving table. The main repository serves `dir` under the synthetic name
 * {@link REPO} + the fixed SHA; a cross-repository target serves the **adjacent mirror of the same
 * name** under the declared (repo, revision).
 *
 * The revision resolution API of cross-repository targets is only provided and never actually hit
 * — a cross-repository `revision` is always a 40-digit commit SHA, and the fetch layer uses the
 * SHA as is (no resolution request occurs). Keeping the table lookup the same for the 2 routes
 * means fewer branches, so it is not special-cased but put in the same table.
 */
const buildRoutes = (dir: string): ReadonlyMap<string, ServedRepo> => {
  const routes = new Map<string, ServedRepo>([[REPO, { revision: REVISION_SHA, dir }]]);
  const manifest = parseManifest(readTextFileSync(`${dir}/${MANIFEST_FILENAME}`));
  for (const ref of allFileRefs(manifest)) {
    if (ref.repo === undefined || ref.revision === undefined) continue;
    // Reach the siblings of `dir` through `..` (works for both absolute and relative paths
    // without bringing in a path library).
    const mirror = `${dir}/../${ref.repo.slice(ref.repo.lastIndexOf("/") + 1)}`;
    const known = routes.get(ref.repo);
    if (known === undefined) routes.set(ref.repo, { revision: ref.revision, dir: mirror });
    else if (known.revision !== ref.revision) {
      throw new Error(
        `the revisions of cross-repo references to one repo disagree (${ref.repo}: ${known.revision} and ` +
          `${ref.revision}) — mirror ${mirror} can hold only one revision`,
      );
    }
    assertMirrored(ref, mirror);
  }
  return routes;
};

/**
 * A running throwaway server. Shut it down with `await using` or an explicit
 * {@link LocalDistServer.close}.
 */
export type LocalDistServer = AsyncDisposable & {
  /** The source to pass to `fromPretrained` (points at this server). */
  readonly source: HubRepoRef;
  /** Shuts the server down (no need to call it when received with `await using`). */
  readonly close: () => Promise<void>;
};

/** Handles one request (the spelling and the 404 conditions are identical to the Deno version). */
const handle = async (
  routes: ReadonlyMap<string, ServedRepo>,
  request: IncomingMessage,
  response: ServerResponse,
): Promise<void> => {
  const { pathname } = new URL(request.url ?? "/", "http://127.0.0.1");
  const revision = REVISION_ROUTE.exec(pathname);
  if (revision !== null) {
    const served = routes.get(revision[1]);
    if (served === undefined) return notFound(response);
    response
      .writeHead(200, { "content-type": "application/json" })
      .end(JSON.stringify({ sha: served.revision }));
    return;
  }
  const resolved = RESOLVE_ROUTE.exec(pathname);
  if (resolved === null) return notFound(response);
  const [, repo, fixed, encoded] = resolved;
  const path = decodeURIComponent(encoded);
  const served = routes.get(repo);
  // The revision is matched against "the value declared for that repo" (the fixed SHA for the
  // main repository, the commit SHA the manifest named for a cross-repository target).
  if (served === undefined || fixed !== served.revision || !isInsideDist(path)) {
    return notFound(response);
  }
  let file: FileHandle;
  try {
    file = await open(`${served.dir}/${path}`);
  } catch (error) {
    if (isNotFound(error)) return notFound(response);
    throw error;
  }
  let size: number;
  try {
    ({ size } = await file.stat());
  } catch (error) {
    await file.close();
    throw error;
  }
  // MUST: declare content-length. The hub's received-byte check requires the declared length to
  // match the manifest's `size` exactly, so omitting it lets the check pass as "nothing declared"
  // (the real HF always declares it).
  response.writeHead(200, { "content-length": String(size) });
  // The read stream owns the handle from here (autoClose), also when the pipeline fails.
  await pipeline(file.createReadStream(), response);
};

/**
 * Starts serving `dir` (and the cross-repository mirrors its manifest points at) HF-style (port
 * auto-assigned, bound to 127.0.0.1). A missing mirror fails here ({@link buildRoutes}).
 *
 * `source.revision` is set to `"main"` **explicitly** so that the "main can be moved" warning the
 * hub issues for calls without a revision (guidance meant for real HF repositories) does not fire
 * on this server, which has only one revision. Resolution goes through the revision resolution API
 * exactly once, as in production.
 *
 * Node binds a host-qualified port only asynchronously (`listen(0, "127.0.0.1")` resolves the
 * host first), so the server is handed out once it listens. The distribution checks above still
 * run synchronously: a broken mirror throws from this call itself, before any port is opened.
 */
export const serveLocalDist = (dir: string): Promise<LocalDistServer> => {
  const routes = buildRoutes(dir);
  const server = createServer((request, response) => {
    handle(routes, request, response).catch((error: unknown) => {
      // Like Deno.serve's default onError: log and answer 500 while that is still possible.
      if (response.headersSent) {
        response.destroy();
        return;
      }
      console.error(error);
      response
        .writeHead(500, { "content-type": "text/plain;charset=UTF-8" })
        .end("Internal Server Error");
    });
  });
  let closing: Promise<void> | undefined;
  // Idempotent like Deno's `shutdown()`. `close()` alone waits for keep-alive sockets held by
  // the client's connection pool, so every remaining connection is closed as well.
  const close = (): Promise<void> =>
    (closing ??= new Promise((resolve, reject) => {
      server.close((error) => (error === undefined ? resolve() : reject(error)));
      server.closeAllConnections();
    }));
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.off("error", reject);
      const address = server.address();
      if (address === null || typeof address === "string") {
        reject(new Error(`serveLocalDist: cannot get the TCP address (${String(address)})`));
        return;
      }
      resolve({
        source: { repo: REPO, revision: "main", hubUrl: `http://127.0.0.1:${address.port}` },
        close,
        [Symbol.asyncDispose]: close,
      });
    });
  });
};
