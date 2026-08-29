/**
 * ローカルの配布形ディレクトリ（`karume.json` を持つ）を **HF 形の HTTP** で配る使い捨て
 * サーバ。デモの `--source <ローカルのパス>` はこれ越しに `fromPretrained` を通る。
 *
 * ## なぜローカル読みを HTTP へ回すのか
 *
 * 1GiB 超のコンポーネントは配布形の時点で **shard 分割**されている（shard は独立ヘッダの
 * safetensors なので連結できない）。全量読み（`local-assets.ts`）は「1 コンポーネント =
 * 1 ファイル」の前提でバイト列を `openModel` へ渡す面なので、分割された配布形をそもそも
 * 開けない。取得層を通せば shard 面（グラフ shard → `prepareModel` → 重み shard の逐次流し）が
 * そのまま効くので、デモのロード経路が**本番（hub + prefetch + streamAssets）と 1 本になり**、
 * 分割の有無を気にしなくてよくなる。
 *
 * 喋るのは hub が実際に叩く 2 経路だけ（revision 解決 API と resolve URL — 綴りの正本は
 * `@hdae/fetch-cache/hf` の `resolveHfRevision` / `hfResolveUrl`）。Range も HEAD も要らない
 * （取得層が撃つのは素の GET だけ）。
 *
 * NOTE: 取得層は資産を**永続キャッシュ**（Deno は DENO_DIR）へ写すので、ローカルの配布形が
 * ディスク上で二重に持たれる（2 回目以降の起動はそのキャッシュから読む）。掃除は
 * `@karume/hub` の `clearHubCache`。
 */

import type { HubRepoRef } from "../../packages/hub/mod.ts";

/**
 * 使い捨てサーバが名乗るリポジトリ名。取得層のキャッシュキーには repo が入る（資産のキーは
 * repo + path + sha256 の内容キー）ので、実在の HF リポと衝突しない綴りにしておく。
 */
const REPO = "karume-local/dist";

/**
 * revision 解決 API が返す固定 SHA。配布形のディレクトリ 1 つに revision は 1 つしか無いので、
 * 値そのものに意味は無い（40 桁小文字 hex であることだけが取得層の要求）。
 */
const REVISION_SHA = "0".repeat(40);

/** `{hubUrl}/api/models/{repo}/revision/{ref}`（`resolveHfRevision`）。 */
const REVISION_ROUTE = /^\/api\/models\/(.+)\/revision\/(.+)$/;
/** `{hubUrl}/{repo}/resolve/{revision}/{path}`（`hfResolveUrl`）。 */
const RESOLVE_ROUTE = /^\/(.+?)\/resolve\/([^/]+)\/(.+)$/;

const notFound = (): Response => new Response("not found", { status: 404 });

/**
 * 配布形ディレクトリの**中**だけを指す相対 path か。使い捨てとはいえ 127.0.0.1 に開いた実
 * サーバなので、`..` や絶対 path をそのまま `Deno.open` へ渡さない。
 */
const isInsideDist = (path: string): boolean =>
  path.length > 0 && !path.startsWith("/") &&
  path.split("/").every((segment) => segment !== "" && segment !== "." && segment !== "..");

/** 起動中の使い捨てサーバ。`await using` でも明示 {@link LocalDistServer.close} でも畳める。 */
export type LocalDistServer = AsyncDisposable & {
  /** `fromPretrained` へ渡す取得元（このサーバを指す）。 */
  readonly source: HubRepoRef;
  /** サーバを畳む（`await using` で受けたなら呼ばなくてよい）。 */
  readonly close: () => Promise<void>;
};

/**
 * `dir` を HF 形で配り始める（ポートは自動割当・127.0.0.1 束縛）。
 *
 * `source.revision` に `"main"` を**明示**するのは、hub が revision 未指定の呼びに出す
 * 「main は付け替えられる」警告（実在の HF リポ向けの案内）を、revision が 1 つしか無い
 * このサーバで鳴らさないため。解決は本番と同じく revision 解決 API を 1 度だけ通る。
 */
export const serveLocalDist = (dir: string): LocalDistServer => {
  const server = Deno.serve(
    { port: 0, hostname: "127.0.0.1", onListen: () => {} },
    async (request) => {
      const { pathname } = new URL(request.url);
      const revision = REVISION_ROUTE.exec(pathname);
      if (revision !== null) {
        return revision[1] === REPO ? Response.json({ sha: REVISION_SHA }) : notFound();
      }
      const resolved = RESOLVE_ROUTE.exec(pathname);
      if (resolved === null) return notFound();
      const [, repo, fixed, encoded] = resolved;
      const path = decodeURIComponent(encoded);
      if (repo !== REPO || fixed !== REVISION_SHA || !isInsideDist(path)) return notFound();
      let file: Deno.FsFile;
      try {
        file = await Deno.open(`${dir}/${path}`);
      } catch (error) {
        if (error instanceof Deno.errors.NotFound) return notFound();
        throw error;
      }
      const { size } = await file.stat();
      // MUST: content-length を宣言する。hub の受信バイト門は申告と manifest の `size` の
      // 厳密一致を見る面なので、省くと門が「申告なし」で素通りする（本番の HF は必ず申告する）。
      return new Response(file.readable, { headers: { "content-length": String(size) } });
    },
  );
  const close = (): Promise<void> => server.shutdown();
  return {
    source: { repo: REPO, revision: "main", hubUrl: `http://127.0.0.1:${server.addr.port}` },
    close,
    [Symbol.asyncDispose]: close,
  };
};
