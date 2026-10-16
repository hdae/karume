/**
 * ローカルの配布形ディレクトリ（`karume.json` を持つ）を **HF 形の HTTP** で配る使い捨て
 * サーバ。
 *
 * ## 消費者は「実 DL 経路そのものを通す」ものだけ
 *
 * デモの `--source <ローカルのパス>` はここを通らない — 手元の配布形は取得元ハンドル
 * （`@karume/hub/node` の `nodeDirectory` — `local-source.ts`）で直に読めるようになり、
 * ポートも永続キャッシュへの複製も要らなくなった。残る消費者は次の 2 つで、どちらも**実 DL
 * 経路**（revision 解決 → resolve URL → 受信バイト門 → 永続キャッシュ）を通すことが目的なので、
 * 疑似 HF を喋るこのサーバでなければならない:
 *
 * - `packages/models/tests/e2e_gemma4_pretrained_test.ts` — その経路の疎通を見る門
 * - `tools/ram-peak/measure.ts` の `--state cold` / `warm` — その経路のホスト RAM ピークと
 *   キャッシュの挙動を測る
 *
 * ## なぜ HTTP でも全量読みにしないのか
 *
 * 部品は配布形の時点で **part 分割**されている（コンテナ — ADR 0109）。全量読み
 * （`local-assets.ts` + `from*Assets`）でも分割形は読めるが、その面は**全 part を同時に
 * ホスト RAM へ載せる**（3.7GiB の DiT がそのまま常駐する）。取得層を通せば取得面
 * （descriptor → admission → 重みの part を温める → block の区間読み）がそのまま効いて、
 * RAM に載るのは読んでいる block の周辺だけになる。
 *
 * 喋るのは hub が実際に叩く 2 経路だけ（revision 解決 API と resolve URL — 綴りの正本は
 * `@hdae/fetch-cache/hf` の `resolveHfRevision` / `hfResolveUrl`）。Range も HEAD も要らない
 * （取得層が撃つのは素の GET だけ）。
 *
 * ## 越境参照は隣のミラーから配る
 *
 * 配布形は自リポの外を指せる（`FileRef` の `repo` / `revision` — ADR 0038 §7。例:
 * `karume-anima-extra` の text stack は `hdae/karume-anima` の 1 commit を指す）。取得層は
 * その 1 本を**セッションの (repo, SHA) ではなく宣言された (repo, revision) から**取りに来る
 * ので、1 組しか名乗らないサーバでは越境ぶんが丸ごと 404 になる。そこで配信表を
 * 「(repo, revision) → 配信ディレクトリ」の**複数エントリ**にし、越境先は **repo 名の最終要素と
 * 同名の隣接ディレクトリ**（`models/` に配布形ミラーが並ぶ配置）から配る。
 *
 * NOTE: 取得層は資産を**永続キャッシュ**（Deno は DENO_DIR）へ写すので、ローカルの配布形が
 * ディスク上で二重に持たれる（2 回目以降の起動はそのキャッシュから読む）。掃除は
 * `@karume/hub` の `clearHubCache`。
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
 * 使い捨てサーバが**主リポとして**名乗るリポジトリ名。取得層のキャッシュキーには repo が入る
 * （資産のキーは repo + path + sha256 の内容キー）ので、実在の HF リポと衝突しない綴りに
 * しておく。越境先は合成せず、manifest が名指しした実在の repo をそのまま名乗る。
 */
const REPO = "karume-local/dist";

/**
 * 主リポの revision 解決 API が返す固定 SHA。配布形のディレクトリ 1 つに revision は 1 つしか
 * 無いので、値そのものに意味は無い（40 桁小文字 hex であることだけが取得層の要求）。
 */
const REVISION_SHA = "0".repeat(40);

/** `{hubUrl}/api/models/{repo}/revision/{ref}`（`resolveHfRevision`）。 */
const REVISION_ROUTE = /^\/api\/models\/(.+)\/revision\/(.+)$/;
/** `{hubUrl}/{repo}/resolve/{revision}/{path}`（`hfResolveUrl`）。 */
const RESOLVE_ROUTE = /^\/(.+?)\/resolve\/([^/]+)\/(.+)$/;

// Same status, body and content-type as the Deno version's `new Response("not found", ...)`.
const notFound = (response: ServerResponse): void => {
  response.writeHead(404, { "content-type": "text/plain;charset=UTF-8" }).end("not found");
};

/**
 * 配布形ディレクトリの**中**だけを指す相対 path か。使い捨てとはいえ 127.0.0.1 に開いた実
 * サーバなので、`..` や絶対 path をそのままファイルの open へ渡さない。
 */
const isInsideDist = (path: string): boolean =>
  path.length > 0 && !path.startsWith("/") &&
  path.split("/").every((segment) => segment !== "" && segment !== "." && segment !== "..");
/** 配信表の 1 エントリ = ある repo について「名乗る revision」と「配る元ディレクトリ」。 */
type ServedRepo = {
  readonly revision: string;
  readonly dir: string;
};

/**
 * manifest 全域の `FileRef`（全 model × 全 weights × 全 dtype のコンテナの part + assets）。
 *
 * `resolveSelection` で 1 組の (model, quant) に絞らないのは、サーバが**選択より先に**立つため
 * （`serveLocalDist` は `dir` しか受け取らず、model / quant は後段の `fromPretrained` が決める）。
 * 表に載せ過ぎても実際に取りに来ない repo が増えるだけで害は無い。
 *
 * 長さ 0 の part は落とす（取得層も取りに来ない — ADR 0109 決定 3）。
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
 * 越境参照の実体がミラーに在ることを**起動時に**確かめる。
 *
 * MUST: 実行中の 404 で不発にしない。取得層は数百 MiB を流し終えてから欠けに気づく面なので、
 * 待たせた末に落とすくらいなら立ち上がりで落とす。
 */
const assertMirrored = (ref: FileRef, mirror: string): void => {
  try {
    statSync(`${mirror}/${ref.path}`);
  } catch (error) {
    if (!isNotFound(error)) throw error;
    throw new Error(
      `越境参照のミラーが無い: ${ref.repo}@${ref.revision} の ${ref.path} が ${mirror} 配下に` +
        `見つからない（repo 名の最終要素と同名のディレクトリへ配布形ミラーを置くこと）`,
    );
  }
};

/**
 * 配信表を組む。主リポは合成名 {@link REPO} + 固定 SHA で `dir` を、越境先は宣言された
 * (repo, revision) で**隣接する同名ミラー**を配る。
 *
 * 越境先の revision 解決 API は用意するだけで実際には叩かれない — 越境の `revision` は 40 桁
 * commit SHA 固定で、取得層は SHA をそのまま使う（解決要求が発生しない）。表を引く経路を
 * 2 ルートで揃えるほうが分岐が減るので、特例にせず同じ表に載せている。
 */
const buildRoutes = (dir: string): ReadonlyMap<string, ServedRepo> => {
  const routes = new Map<string, ServedRepo>([[REPO, { revision: REVISION_SHA, dir }]]);
  const manifest = parseManifest(readTextFileSync(`${dir}/${MANIFEST_FILENAME}`));
  for (const ref of allFileRefs(manifest)) {
    if (ref.repo === undefined || ref.revision === undefined) continue;
    // `dir` の兄弟を `..` で辿る（path ライブラリを持ち込まずに絶対・相対の両方で効く）。
    const mirror = `${dir}/../${ref.repo.slice(ref.repo.lastIndexOf("/") + 1)}`;
    const known = routes.get(ref.repo);
    if (known === undefined) routes.set(ref.repo, { revision: ref.revision, dir: mirror });
    else if (known.revision !== ref.revision) {
      throw new Error(
        `越境参照の revision が 1 リポで割れている（${ref.repo}: ${known.revision} と ` +
          `${ref.revision}）— ミラー ${mirror} は 1 revision ぶんしか持てない`,
      );
    }
    assertMirrored(ref, mirror);
  }
  return routes;
};

/** 起動中の使い捨てサーバ。`await using` でも明示 {@link LocalDistServer.close} でも畳める。 */
export type LocalDistServer = AsyncDisposable & {
  /** `fromPretrained` へ渡す取得元（このサーバを指す）。 */
  readonly source: HubRepoRef;
  /** サーバを畳む（`await using` で受けたなら呼ばなくてよい）。 */
  readonly close: () => Promise<void>;
};

/** 1 リクエストを捌く（綴りと 404 の条件は Deno 版と同一）。 */
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
    response.writeHead(200, { "content-type": "application/json" })
      .end(JSON.stringify({ sha: served.revision }));
    return;
  }
  const resolved = RESOLVE_ROUTE.exec(pathname);
  if (resolved === null) return notFound(response);
  const [, repo, fixed, encoded] = resolved;
  const path = decodeURIComponent(encoded);
  const served = routes.get(repo);
  // revision は「その repo に対して宣言された値」と照合する（主リポは固定 SHA・越境は
  // manifest が名指しした commit SHA）。
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
  // MUST: content-length を宣言する。hub の受信バイト門は申告と manifest の `size` の
  // 厳密一致を見る面なので、省くと門が「申告なし」で素通りする（本番の HF は必ず申告する）。
  response.writeHead(200, { "content-length": String(size) });
  // The read stream owns the handle from here (autoClose), also when the pipeline fails.
  await pipeline(file.createReadStream(), response);
};

/**
 * `dir`（と、その manifest が指す越境ミラー）を HF 形で配り始める（ポートは自動割当・
 * 127.0.0.1 束縛）。ミラーの欠けはここで落ちる（{@link buildRoutes}）。
 *
 * `source.revision` に `"main"` を**明示**するのは、hub が revision 未指定の呼びに出す
 * 「main は付け替えられる」警告（実在の HF リポ向けの案内）を、revision が 1 つしか無い
 * このサーバで鳴らさないため。解決は本番と同じく revision 解決 API を 1 度だけ通る。
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
      response.writeHead(500, { "content-type": "text/plain;charset=UTF-8" })
        .end("Internal Server Error");
    });
  });
  let closing: Promise<void> | undefined;
  // Idempotent like Deno's `shutdown()`. `close()` alone waits for keep-alive sockets held by
  // the client's connection pool, so every remaining connection is closed as well.
  const close = (): Promise<void> =>
    closing ??= new Promise((resolve, reject) => {
      server.close((error) => (error === undefined ? resolve() : reject(error)));
      server.closeAllConnections();
    });
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.off("error", reject);
      const address = server.address();
      if (address === null || typeof address === "string") {
        reject(new Error(`serveLocalDist: TCP のアドレスが取れない（${String(address)}）`));
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
