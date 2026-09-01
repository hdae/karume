/**
 * ローカルの配布形ディレクトリ（`karume.json` を持つ）を **HF 形の HTTP** で配る使い捨て
 * サーバ。
 *
 * ## 消費者は「HTTP 疎通そのものを見る門」だけ
 *
 * デモの `--source <ローカルのパス>` はここを通らない — 手元の配布形は取得元ハンドル
 * （`@karume/hub/deno` の `denoDirectory` — `local-source.ts`）で直に読めるようになり、
 * ポートも永続キャッシュへの複製も要らなくなった。残る消費者は
 * `packages/models/tests/e2e_gemma4_pretrained_test.ts` で、あちらは**実 DL 経路**（revision
 * 解決 → resolve URL → 受信バイト門 → 永続キャッシュ）を通すことが門の目的なので、疑似 HF
 * を喋るこのサーバでなければならない。
 *
 * ## なぜ HTTP でも全量読みにしないのか
 *
 * 1GiB 超のコンポーネントは配布形の時点で **shard 分割**されている。全量読み
 * （`local-assets.ts` + `from*Assets`）でも分割形は読めるが、その面は**全 shard を同時に
 * ホスト RAM へ載せる**（3.7GiB の DiT がそのまま常駐する）。取得層を通せば shard 面
 * （グラフ shard → `prepareModel` → 重み shard の逐次流し）がそのまま効いて RAM に載るのは
 * 常に「今の 1 本」だけになる。
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

import {
  type FileRef,
  type HubRepoRef,
  type Manifest,
  MANIFEST_FILENAME,
  parseManifest,
} from "../../packages/hub/mod.ts";

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

const notFound = (): Response => new Response("not found", { status: 404 });

/**
 * 配布形ディレクトリの**中**だけを指す相対 path か。使い捨てとはいえ 127.0.0.1 に開いた実
 * サーバなので、`..` や絶対 path をそのまま `Deno.open` へ渡さない。
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
 * manifest 全域の `FileRef`（全 model × 全 weights × 全 dtype の shards / extras + assets）。
 *
 * `resolveFiles` で 1 組の (model, quant) に絞らないのは、サーバが**選択より先に**立つため
 * （`serveLocalDist` は `dir` しか受け取らず、model / quant は後段の `fromPretrained` が決める）。
 * 表に載せ過ぎても実際に取りに来ない repo が増えるだけで害は無い。
 */
const allFileRefs = (manifest: Manifest): readonly FileRef[] => {
  const refs: FileRef[] = [];
  for (const model of Object.values(manifest.models)) {
    for (const entry of Object.values(model.weights)) {
      for (const files of Object.values(entry)) {
        refs.push(...files.shards, ...Object.values(files.extras));
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
    Deno.statSync(`${mirror}/${ref.path}`);
  } catch (error) {
    if (!(error instanceof Deno.errors.NotFound)) throw error;
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
  const manifest = parseManifest(Deno.readTextFileSync(`${dir}/${MANIFEST_FILENAME}`));
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

/**
 * `dir`（と、その manifest が指す越境ミラー）を HF 形で配り始める（ポートは自動割当・
 * 127.0.0.1 束縛）。ミラーの欠けはここで落ちる（{@link buildRoutes}）。
 *
 * `source.revision` に `"main"` を**明示**するのは、hub が revision 未指定の呼びに出す
 * 「main は付け替えられる」警告（実在の HF リポ向けの案内）を、revision が 1 つしか無い
 * このサーバで鳴らさないため。解決は本番と同じく revision 解決 API を 1 度だけ通る。
 */
export const serveLocalDist = (dir: string): LocalDistServer => {
  const routes = buildRoutes(dir);
  const server = Deno.serve(
    { port: 0, hostname: "127.0.0.1", onListen: () => {} },
    async (request) => {
      const { pathname } = new URL(request.url);
      const revision = REVISION_ROUTE.exec(pathname);
      if (revision !== null) {
        const served = routes.get(revision[1]);
        return served === undefined ? notFound() : Response.json({ sha: served.revision });
      }
      const resolved = RESOLVE_ROUTE.exec(pathname);
      if (resolved === null) return notFound();
      const [, repo, fixed, encoded] = resolved;
      const path = decodeURIComponent(encoded);
      const served = routes.get(repo);
      // revision は「その repo に対して宣言された値」と照合する（主リポは固定 SHA・越境は
      // manifest が名指しした commit SHA）。
      if (served === undefined || fixed !== served.revision || !isInsideDist(path)) {
        return notFound();
      }
      let file: Deno.FsFile;
      try {
        file = await Deno.open(`${served.dir}/${path}`);
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
