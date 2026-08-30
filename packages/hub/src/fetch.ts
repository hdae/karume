/**
 * HF リポジトリからの取得（ADR 0038 §5「hub の取得層」）。土台は `@hdae/fetch-cache`
 * （実行時依存ゼロ・Web 標準 API のみ）。
 *
 * 契約:
 * - revision は**セッションあたり 1 回だけ**解決し、manifest も全ファイルも同一 commit SHA に
 *   固定して取得する（可変 ref のまま複数回解決すると manifest と重みが別コミットから来る）。
 *   例外は manifest が明示した**越境参照**（`FileRef` の `repo` / `revision` — ADR 0038 §7）
 *   だけで、その 1 本はセッションの SHA ではなく宣言された (repo, revision) から取る。
 * - キャッシュの所有は取得層（0.5.0 で名前空間が内部固定 1 個になった）。資産のキーは内容キー
 *   `["hf", kind, repo, path, sha256]` なので、revision が動いてもバイト不変のファイルはヒットの
 *   まま読める。manifest だけは事前の期待 sha が無いので SHA 固定 resolve URL がキー。
 * - **資産の完全性検証は取得層へ委ねる**（spec の `sha256` / `expectedBytes`）。取得時に検証して
 *   記録ハッシュをエントリへ焼き、以後のヒットは記録との文字列比較だけで済ませる（全量ハッシュ
 *   0 回）。記録が食い違うエントリは自動で evict → 取り直し（self-heal）。
 * - 同時取得の律速は面ごとに違う（全量面はバイト予算・逐次面の相 1 は本数 4）。`AbortSignal` は
 *   全取得へ透過。
 * - 進捗総量は content-length ではなく manifest の `size` 合計（path 一意化後）。
 */

import { clearCache } from "@hdae/fetch-cache";
import {
  fetchHfFile,
  hfResolveUrl,
  isCommitSha,
  prefetchHfFile,
  resolveHfRevision,
} from "@hdae/fetch-cache/hf";
import { createByteAdmission } from "./concurrency.ts";
import {
  HubError,
  HubFetchError,
  IntegrityError,
  ManifestFormatError,
  ManifestReferenceError,
} from "./errors.ts";
import {
  type FileRef,
  fileRefKey,
  type Manifest,
  MANIFEST_FILENAME,
  MAX_MANIFEST_BYTES,
  parseManifest,
} from "./manifest.ts";
import type { ResolvedFiles } from "./resolve.ts";
import { type ByteBudget, createGuardedFetch } from "./transport.ts";

/**
 * 取得層が名前空間を自前で持つ前（`@hdae/fetch-cache` 0.4 以前）に hub が使っていた名前空間。
 * 本体 `karume/1` と、認証隔離だった `karume/1:*` の両系列を指す。
 */
const LEGACY_CACHE_NAMESPACE = "karume/1";

/**
 * 逐次面 {@link streamAssets} 相 1 の同時取得数（数十コンポーネントの manifest で接続を
 * 破綻させない）。相 1 は body をそのままキャッシュへ流す streaming なので、受信バッファの
 * 前確保が無い＝本数だけで RAM が決まらない。
 */
const CONCURRENCY = 4;

/**
 * 全量面 {@link fetchAssets} の in-flight バイト予算（1.5GiB = 1,610,612,736 バイト）。
 *
 * 全量面は 1 ファイルにつき `ref.size` ぶんの受信バッファを**受信前に**確保するので、律速を
 * 本数にすると RAM ピークが「同時本数 × その時点で一番大きいファイル」で決まってしまう
 * （実測: anima turbo i4 の先頭 4 本で計 2.503GiB を同時前確保 — 8GB 機ターゲットでは
 * ブラウザごと落ちる）。予算は前確保の合計に課すもので、これに加えて完走済みファイルの保持と
 * 検証の一時コピー（Chrome の `crypto.subtle.digest` は入力を Blink 内部へ全量コピーする）が
 * 乗るため、単一 ArrayBuffer の上限（Chromium は 2,145,386,496 バイトで打ち切る）よりも
 * 明確に下へ置く。
 *
 * NOTE: その digest は取得層の中（network 取得の検証）にあり、hub からは直列化できない — 検証を
 * 委ねた以上、同時に走る本数を決めるのはこの予算だけになった。キャッシュヒットは記録ハッシュの
 * 文字列比較で済むので digest ごと起きない（2 回目以降の起動でこのピークは立たない）。
 *
 * 公開ノブにはしない — 「安全側へ下げる」以外の使い道が無い値であり、上げれば上のピークが
 * そのまま戻る。合わない配布が出たら定数ごと裁定し直す。
 */
const BYTE_BUDGET = 1.5 * 1024 * 1024 * 1024;

/** 取得対象リポジトリ。`hubUrl` は**アプリが明示指定した場合のみ**有効（manifest からは来ない）。 */
export type HubRepoRef = {
  /** `"owner/name"`。 */
  readonly repo: string;
  /** ブランチ / タグ / commit SHA。既定 `"main"`。SHA を渡すと解決要求が発生しない。 */
  readonly revision?: string;
  /** ミラー用。既定 `https://huggingface.co`。 */
  readonly hubUrl?: string;
};

/** cache I/O 失敗の診断（quota 超過等）。console.warn に任せず**アプリへ届ける**。 */
export type CacheDiagnostic = {
  readonly op: "open" | "match" | "put" | "delete";
  readonly url: string;
  readonly error: unknown;
};

export type LoadManifestOptions = {
  readonly signal?: AbortSignal;
  /**
   * `Authorization` 等。取得（revision 解決・ファイル）へそのまま透過する。
   *
   * NOTE: キャッシュは credential で分けない（by-design）— キーにヘッダは入らないので、認証付きで
   * 取得したバイト列は以後の無認証呼び出しにもヒットする。gated 資産の運用予定が無い以上、
   * 隔離は過剰防御だった（詳細は `docs/limitations.md`）。
   */
  readonly headers?: HeadersInit;
  /**
   * cache I/O 失敗の通知先。無指定だと「毎起動フル再 DL が黙って常態化」するため、
   * アプリは受け取って `navigator.storage.persist()` の案内などに使う。
   */
  readonly onCacheError?: (diagnostic: CacheDiagnostic) => void;
  /** `fetch` の差し替え（テスト・カスタム輸送用）。 */
  readonly fetch?: typeof globalThis.fetch;
  /** `CacheStorage` の差し替え（テスト用）。 */
  readonly caches?: CacheStorage;
};

/** 解決済み revision に固定された manifest。以降の取得は全てこの SHA で行う。 */
export type LoadedManifest = {
  readonly repo: string;
  /** 解決済み commit SHA（40 桁）。返り値・エラー・診断に必ず載る。 */
  readonly revisionSha: string;
  readonly hubUrl?: string;
  readonly manifest: Manifest;
};

/**
 * 進捗のフェーズ。`complete` は 1 ファイルの終端（bytes が確定した点）。
 *
 * MUST: 1 ファイルの phase は `downloading`* → `complete` の順にだけ進み、逆行しない
 * （`complete` はファイルごとに 1 回だけ・以降そのファイルの通知は出ない）。例外は破損キャッシュ
 * の self-heal で、取得層が拒否した後に network から取り直すためこの 1 巡が最初からやり直しに
 * なる（`complete` が終端であることは変わらない）。
 *
 * NOTE: 照合中を表す `verifying` は持たない — 資産の検証は取得層の内部（受信中のハッシュ / 記録
 * ハッシュの突合）に埋まっていて hub からは観測できないため。観測できないフェーズを推測で
 * 名乗ると、実際には終わっている照合を「進行中」と表示する嘘になる。
 */
export type AssetPhase = "downloading" | "complete";

export type AssetProgress = {
  readonly phase: AssetPhase;
  /** イベントを起こしたファイルの path。 */
  readonly path: string;
  /** 取得済みバイトの合計（全ファイル・path 一意化後）。 */
  readonly loaded: number;
  /** manifest の `size` 合計（path 一意化後）。 */
  readonly total: number;
  /**
   * `path` の**そのファイル自身**の受信済みバイト。`loaded` が全ファイルの合計なのに対し
   * こちらは 1 ファイルぶんなので、ファイル別の進捗バーはこの値と {@link fileTotal} で描く。
   *
   * `complete` は全量が揃った点なので常に `fileLoaded === fileTotal`（`downloading` が 1 度も
   * 出ないキャッシュヒットでは `complete` の 1 点だけが出る）。
   */
  readonly fileLoaded: number;
  /** `path` のファイル自身の manifest 由来サイズ（`FileRef.size`）。`total` はこれの合計。 */
  readonly fileTotal: number;
};

export type FetchAssetsOptions = LoadManifestOptions & {
  readonly onProgress?: (progress: AssetProgress) => void;
};

/** 旧名前空間（本体 `karume/1` と、認証隔離だった `karume/1:*`）の名前を列挙する。 */
const legacyCacheNames = async (storage: CacheStorage): Promise<string[]> =>
  (await storage.keys()).filter((name) =>
    name === LEGACY_CACHE_NAMESPACE || name.startsWith(`${LEGACY_CACHE_NAMESPACE}:`)
  );

/**
 * 旧名前空間を回収する。新コードは二度と読まないので、置いておいても容量を占めるだけ
 * （中身は取得層 0.4 以前の形式で、キーも記録ハッシュも現行と互換が無い）。
 *
 * キャッシュは正しさの要件ではなく最適化なので、ここは決してロードを落とさない —
 * `caches` が無い環境（Deno 等）は黙って素通りし、列挙・削除の失敗は診断へ流して続行する。
 */
const purgeLegacyCaches = async (options: LoadManifestOptions): Promise<void> => {
  const storage = options.caches ?? globalThis.caches;
  if (storage === undefined) return;
  let names: readonly string[];
  try {
    names = await legacyCacheNames(storage);
  } catch (error) {
    options.onCacheError?.({ op: "delete", url: LEGACY_CACHE_NAMESPACE, error });
    return;
  }
  for (const name of names) {
    try {
      await storage.delete(name);
    } catch (error) {
      // 失敗した名前空間そのものを名乗る（この診断の `url` はキャッシュ名 — 取得元ではない）。
      options.onCacheError?.({ op: "delete", url: name, error });
    }
  }
};

const requestInit = (headers?: HeadersInit, signal?: AbortSignal): RequestInit => ({
  ...(headers === undefined ? {} : { headers }),
  ...(signal === undefined ? {} : { signal }),
});

const hfRef = (
  ref: HubRepoRef,
  revision: string,
): { repo: string; revision: string; hubUrl?: string } => ({
  repo: ref.repo,
  revision,
  ...(ref.hubUrl === undefined ? {} : { hubUrl: ref.hubUrl }),
});

/**
 * 1 ファイルの取得元を決める。**越境参照（`repo` + `revision`）を持つ ref はその
 * (repo, revision) から取る** — セッションの解決済み revision ではない（ADR 0038 §7 の
 * 別リポ参照席。参照先は commit SHA 固定が必須なので、可変 ref が混ざる余地はない）。
 *
 * `hubUrl`（ミラー指定）は**ホストの選択**なので越境先にも同じものを効かせる。資産のキャッシュ
 * キーは内容キー（`hubUrl` を含まない）なので、同じバイト列ならミラーを跨いでも 1 エントリを
 * 共有する。別リポの同名 path はキーに `repo` が入るぶん別エントリのまま。
 */
const originFor = (
  loaded: LoadedManifest,
  session: { repo: string; revision: string; hubUrl?: string },
) =>
(ref: FileRef): { repo: string; revision: string; hubUrl?: string } =>
  ref.repo === undefined || ref.revision === undefined ? session : hfRef({
    repo: ref.repo,
    ...(loaded.hubUrl === undefined ? {} : { hubUrl: loaded.hubUrl }),
  }, ref.revision);

/**
 * エラーに載せる取得元の文脈。越境参照は**実際に取りに行った (repo, SHA)** を名乗る
 * （セッションの repo を名乗ると、そのリポには存在しない path を指す診断になる）。
 */
const fetchContext = (repo: string, revisionSha: string) => (ref: FileRef) => ({
  repo: ref.repo ?? repo,
  revisionSha: ref.revision ?? revisionSha,
});

/**
 * 中断（呼び出し側の `AbortSignal`）かどうか。中断は取得失敗ではないので `HubFetchError` に
 * 包まず素通しする — 包むと呼び出し側の「自分が止めたのか落ちたのか」の判別が壊れる。
 *
 * MUST: 既定の `abort()` が作る DOMException だけを見ない — `abort(reason)` の custom reason
 * では `fetch` が **reason 自体**で reject するため、任意の値が中断の正体になり得る。判定は
 * 「実際に取得へ渡した signal が aborted で、捕まえた値がその reason と同一」に依る（合成した
 * 場合は合成後の signal — 上流の reason はそのまま伝播する）。
 */
const isAborted = (error: unknown, signal?: AbortSignal): boolean =>
  (error instanceof DOMException && error.name === "AbortError") ||
  (signal?.aborted === true && error === signal.reason);

/**
 * `openModel` は全量 ArrayBuffer を要求するため、返す bytes は buffer 全体を占めていなければ
 * ならない（slice で辻褄を合わせると RAM ピークが倍増する）。SharedArrayBuffer 背面は
 * ここで弾く（述語が主張する `Uint8Array<ArrayBuffer>` を型の上でも嘘にしない）。
 */
const isTightView = (bytes: Uint8Array): bytes is Uint8Array<ArrayBuffer> =>
  bytes.buffer instanceof ArrayBuffer && bytes.byteOffset === 0 &&
  bytes.byteLength === bytes.buffer.byteLength;

const assertTightView = (bytes: Uint8Array, path: string): Uint8Array<ArrayBuffer> => {
  if (!isTightView(bytes)) {
    throw new Error(
      `hub: ${path} の bytes が buffer 全体を占めていない` +
        `（byteOffset ${bytes.byteOffset} / byteLength ${bytes.byteLength} /` +
        ` buffer ${bytes.buffer.byteLength}）`,
    );
  }
  return bytes;
};

/**
 * 可変 ref を commit SHA へ解決する。ここが**セッション唯一の解決点**で、以降の取得は
 * 全てこの SHA に固定される。
 */
const resolveRevision = async (
  ref: HubRepoRef,
  revision: string,
  options: LoadManifestOptions,
): Promise<string> => {
  if (isCommitSha(revision)) return revision;
  try {
    return await resolveHfRevision(hfRef(ref, revision), {
      ...(options.fetch === undefined ? {} : { fetch: options.fetch }),
      init: requestInit(options.headers, options.signal),
    });
  } catch (error) {
    if (isAborted(error, options.signal)) throw error;
    throw new HubFetchError(
      `revision '${revision}' の解決に失敗した（repo ${ref.repo}）。可変 ref はオフラインで` +
        `起動できない — revision に commit SHA を渡すと解決要求そのものが発生しない`,
      { repo: ref.repo, cause: error },
    );
  }
};

/**
 * revision を渡さずに `main` を暗黙解決したときの案内を 1 回だけ出す。
 *
 * 呼び手が revision を書いていない場合だけが対象で、`"main"` の**明示**指定・タグ・SHA 指定
 * では出さない（明示は「可変 ref でよい」という呼び手の意思表示で、警告は誤検出になる）。
 * 解決した SHA を印字するのは、その 1 行をコピーすればそのまま pin が完成するため。
 */
const warnImplicitMain = (repo: string, revisionSha: string): void => {
  console.warn(
    `@karume/hub: revision を指定していないため 'main' を解決した（repo ${repo} → ${revisionSha}）。\n` +
      `main は付け替えられるので、同じコードが次の起動で別の重みを読み得る。次のどちらかで固定すること:\n` +
      `  ① revision: "${revisionSha}" を渡す（この 1 行のコピーで pin が完成する）\n` +
      `  ② @karume/models の *_CURRENT 定数（パッケージ検証済みの pin）を使う`,
  );
};

/**
 * `karume.json` を取得して parse する。revision の解決はここで 1 回だけ行い、結果の SHA を
 * 返り値に載せる（{@link fetchAssets} はその SHA で取得する）。
 *
 * セッションの入口でもあるので、ここで旧名前空間（`karume/1` 系）を 1 回だけ回収する。
 *
 * NOTE: manifest は資産と違い**期待 sha256 を事前に持てない**（正本の根なので）。したがって
 * キーは SHA 固定 resolve URL のままで、`validate` = UTF-8 decode + parse がバイト列 →
 * `Manifest` の唯一の変換点として残る（資産側の検証だけが取得層へ移った）。
 */
export const loadManifest = async (
  ref: HubRepoRef,
  options: LoadManifestOptions = {},
): Promise<LoadedManifest> => {
  await purgeLegacyCaches(options);
  const revisionSha = await resolveRevision(ref, ref.revision ?? "main", options);
  // 解決の**後**に出す — 印字する SHA が確定するのがここで、解決に失敗した場合は警告ではなく
  // 失敗そのものが報告されるべきだから（fail loudly が先）。
  if (ref.revision === undefined) warnImplicitMain(ref.repo, revisionSha);
  const target = hfRef(ref, revisionSha);
  const url = hfResolveUrl({ ...target, path: MANIFEST_FILENAME });
  const budget: ByteBudget = {
    maxBytes: MAX_MANIFEST_BYTES,
    exact: false,
    violation: (actual, where) =>
      new ManifestFormatError(
        `manifest: ${MANIFEST_FILENAME} が上限 ${MAX_MANIFEST_BYTES} バイトを超えた` +
          `（${where} = ${actual} — repo ${ref.repo} @ ${revisionSha}）`,
      ),
  };
  // MUST: SHA 固定 URL のキャッシュヒットは network に出ない＝下層の signal 監視が効かないので、
  // 取得の前後で明示的に中断を見る（見ないと中断済みの signal で呼んでも manifest が返り、
  // 取り消したはずのロードがそのまま先へ進む）。
  options.signal?.throwIfAborted();
  // MUST: UTF-8 decode と parse は取得層の `validate` フックの中で行う — 取得の外でやると
  // 破損したキャッシュエントリが evict されず、`clearHubCache` を手で叩くまで毎回同じ
  // ManifestFormatError を返し続ける（資産側と同じ self-heal 経路に揃える）。
  // MUST NOT: このフックへ中断確認を混ぜない — フックの throw は下層で「破損」と解釈され、
  // 健全なキャッシュエントリの evict と取り直しを招く。
  let manifest: Manifest | undefined;
  const validate = (bytes: Uint8Array): void => {
    let text: string;
    try {
      text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    } catch (error) {
      throw new ManifestFormatError(`manifest: ${MANIFEST_FILENAME} が UTF-8 として読めない`, {
        cause: error,
      });
    }
    manifest = parseManifest(text);
  };
  try {
    await fetchHfFile(target, { path: MANIFEST_FILENAME, validate }, {
      init: requestInit(options.headers, options.signal),
      fetch: createGuardedFetch(options.fetch ?? globalThis.fetch, new Map([[url, budget]])),
      ...(options.caches === undefined ? {} : { caches: options.caches }),
      ...(options.onCacheError === undefined ? {} : { onCacheError: options.onCacheError }),
    });
  } catch (error) {
    if (error instanceof HubError || isAborted(error, options.signal)) throw error;
    throw new HubFetchError(
      `${MANIFEST_FILENAME} の取得に失敗した（repo ${ref.repo} @ ${revisionSha}）`,
      { repo: ref.repo, revisionSha, path: MANIFEST_FILENAME, cause: error },
    );
  }
  // 取得を抜けた直後にも見る（この後は同期の組み立てだけなので、これが返却前の最後の関門）。
  options.signal?.throwIfAborted();
  if (manifest === undefined) {
    throw new Error(
      `hub: ${MANIFEST_FILENAME} の検証フックが走っていない（取得層の不変条件破れ）`,
    );
  }
  return {
    repo: ref.repo,
    revisionSha,
    ...(ref.hubUrl === undefined ? {} : { hubUrl: ref.hubUrl }),
    manifest,
  };
};

/**
 * 進捗の `fileTotal` に載せる manifest 由来の size を引く。表は取得対象そのものから作るので
 * 取得中の ref は必ず引ける — 引けないのは内部の不変条件が破れているときだけなので落とす
 * （0 で埋めるとファイル別の進捗バーが黙って壊れた値を描く）。
 */
const fileSizeOf = (refs: ReadonlyMap<string, FileRef>, key: string): number => {
  const ref = refs.get(key);
  if (ref === undefined) {
    throw new Error(`hub: ${key} の size が取得対象の表に無い（進捗集計の不変条件破れ）`);
  }
  return ref.size;
};

/**
 * 解決済みファイル表を取得する。取得と進捗総量は **path で一意化**され、同じ path を指す
 * 複数のキーには同一のバイト列が入る。
 *
 * 検証（size / sha256）は取得層へ委ねる — manifest の `sha256` / `size` を spec に載せるので、
 * network 取得は受信中に照合され、通ったエントリには記録ハッシュが焼かれる。以後のヒットは記録
 * との文字列比較だけで済み（全量ハッシュ 0 回）、記録が食い違う・記録が無いのに実ハッシュが
 * 合わないエントリは evict → 取り直し（self-heal）になる。
 */
export const fetchAssets = async (
  loaded: LoadedManifest,
  files: ResolvedFiles,
  options: FetchAssetsOptions = {},
): Promise<Record<string, Uint8Array<ArrayBuffer>>> => {
  const { repo, revisionSha, manifest } = loaded;
  const available = manifest.available;
  const target = hfRef(loaded, revisionSha);
  const targetFor = originFor(loaded, target);
  const contextFor = fetchContext(repo, revisionSha);

  const keys = Object.keys(files);
  // MUST: 一意化は path ではなく {@link fileRefKey} で行う — 越境参照が入った以上、別リポの
  // 同名 path は別のバイト列であり、path で畳むと片方の bytes がもう片方に配られる。
  const unique = new Map<string, FileRef>();
  for (const key of keys) {
    const ref = files[key];
    const refKey = fileRefKey(ref);
    if (!unique.has(refKey)) unique.set(refKey, ref);
  }
  const targets = [...unique.values()];
  let total = 0;
  for (const ref of targets) total += ref.size;

  const received = new Map<string, number>();
  const emit = (phase: AssetPhase, ref: FileRef): void => {
    if (options.onProgress === undefined) return;
    const refKey = fileRefKey(ref);
    const fileTotal = fileSizeOf(unique, refKey);
    // complete は全量が揃った点なので size をそのまま渡す（キャッシュヒットは downloading が
    // 1 度も出ず `received` に載らないため、受信実績から引くと 0 に見える）。
    const fileLoaded = phase === "downloading" ? received.get(refKey) ?? 0 : fileTotal;
    // MUST: 全体 `loaded` にも同じ値を積む — このファイルぶんだけ `received` から引くと、
    // 同一イベントで fileLoaded が size なのに loaded がそれを数えない矛盾が出る（全ファイル
    // キャッシュ済みの起動では loaded が 0 のまま complete が並ぶ）。downloading では
    // fileLoaded が `received` の値そのものなので二重計上にはならない。
    let sum = fileLoaded;
    for (const [other, bytes] of received) {
      if (other !== refKey) sum += bytes;
    }
    options.onProgress({ phase, path: ref.path, loaded: sum, total, fileLoaded, fileTotal });
  };

  const budgets = new Map<string, ByteBudget>();
  for (const ref of targets) {
    budgets.set(hfResolveUrl({ ...targetFor(ref), path: ref.path }), {
      maxBytes: ref.size,
      exact: true,
      violation: (actual, where) =>
        new IntegrityError(
          `${ref.path}: ${where} が manifest の size と食い違う（期待 ${ref.size} / 実際 ${actual}）`,
          {
            ...contextFor(ref),
            path: ref.path,
            expected: String(ref.size),
            actual: String(actual),
            source: "network",
            available,
          },
        ),
    });
  }

  const failure = new AbortController();
  const signal = options.signal === undefined
    ? failure.signal
    : AbortSignal.any([failure.signal, options.signal]);
  const guarded = createGuardedFetch(options.fetch ?? globalThis.fetch, budgets);
  const bytesByRef = new Map<string, Uint8Array<ArrayBuffer>>();

  const fetchOne = async (ref: FileRef): Promise<void> => {
    const refKey = fileRefKey(ref);
    // MUST: キャッシュヒットは network に出ない＝下層の signal 監視が効かない区間なので、
    // 取得の前後で明示的に中断を見る（数 GB の読出しを回している最中に取り消しが効かないのは
    // 中断の透過が壊れているのと同じ — 逐次面 streamAssets と同型の確認）。
    signal.throwIfAborted();
    let bytes: Uint8Array;
    try {
      bytes = await fetchHfFile(
        targetFor(ref),
        {
          path: ref.path,
          // 検証は取得層が持つ（受信中のハッシュ / 記録ハッシュの突合 / 不一致の self-heal）。
          sha256: ref.sha256,
          // バイト数の門であり、同時に受信バッファの前確保サイズでもある。確保自体が失敗する
          // 大きさ（Chromium の単一 ArrayBuffer 上限超え）なら受信前に throw されるので、
          // 数 GB を撃ち終わってから落ちることがない。
          expectedBytes: ref.size,
        },
        {
          init: requestInit(options.headers, signal),
          fetch: guarded,
          // network 側だけ発火する（キャッシュヒットは complete の 1 点だけで進む）。
          // `loaded` が `size` を超えないことは受信バイトの門（transport.ts）が保証する。
          onProgress: (progress) => {
            received.set(refKey, progress.loaded);
            emit("downloading", ref);
          },
          ...(options.caches === undefined ? {} : { caches: options.caches }),
          ...(options.onCacheError === undefined ? {} : { onCacheError: options.onCacheError }),
        },
      );
    } catch (error) {
      if (error instanceof HubError || isAborted(error, signal)) throw error;
      const context = contextFor(ref);
      throw new HubFetchError(
        `${ref.path} の取得に失敗した（repo ${context.repo} @ ${context.revisionSha}）`,
        { ...context, path: ref.path, available, cause: error },
      );
    }
    // 取得を抜けた直後にも見る — 前段の確認だけだと「キャッシュ読出しの最中に中断された」形が
    // 観測されず、取り消したはずのファイルが complete まで進む。
    signal.throwIfAborted();
    bytesByRef.set(refKey, assertTightView(bytes, ref.path));
    received.set(refKey, ref.size);
    emit("complete", ref);
  };

  // 送出は「本数」ではなく「in-flight の `ref.size` 合計」で律速する（{@link BYTE_BUDGET}）。
  // 待つのはこのループ 1 本だけなので、`targets`（= resolveFiles の順）の head-of-line
  // blocking がそのまま送出順になる — 後続の小さいファイルに追い越させないので、同じ
  // manifest なら同じ順に出る。
  const admission = createByteAdmission(BYTE_BUDGET);
  const running: Promise<void>[] = [];
  const run = async (ref: FileRef): Promise<void> => {
    try {
      await fetchOne(ref);
    } catch (error) {
      // 1 本でも落ちたら残りを止める（fail loud — 全体が reject するのに DL を続ける意味はない）。
      // MUST: ここで reject を外へ漏らさない — 決着を待つのは全送出の後なので、漏らすと
      // 送出待ちの間に unhandled rejection になる。真の失敗理由は `failure` の reason が持つ。
      failure.abort(error);
    } finally {
      // 成否を問わず席を返す（返さないと後続が永久に待つ）。
      admission.release(ref.size);
    }
  };
  for (const ref of targets) {
    // 失敗・中断の後に新しい取得を起こさない（`fetchOne` 冒頭の確認より前に止める）。
    if (signal.aborted) break;
    await admission.admit(ref.size);
    running.push(run(ref));
  }
  // MUST: 失敗しても送出済み全本の決着を待ってから抜ける（早期 reject だと呼び出し側が
  // catch した後も取得が背後で走り続け、abort の意味が無くなる）。`run` は reject しないので
  // ここは常に成功で返る。
  await Promise.all(running);
  // MUST: 巻き添えの失敗ではなく**真の第一失敗**を上げる。`failure.abort()` は最初の 1 回だけが
  // reason を決めるので、`failure.signal.reason` は必ず最初に落ちた 1 本の理由になる。ワーカーの
  // reject をそのまま拾うと、巻き添え側が先に決着した場合にそちらが表面化する — しかも巻き添えは
  // 中断由来なので `isAborted` で `HubFetchError` に包まれず（生の AbortError）、Chrome では
  // 理由の文言まで固定文言（"BodyStreamBuffer was aborted"）へ差し替えられて真因が消える。
  // 呼び手渡しの外部 signal による中断も、その reason がそのまま `failure` に載るので素通しの
  // ままになる（中断が取得失敗に化けない）。
  if (failure.signal.aborted) throw failure.signal.reason;
  // MUST: 全ファイルがキャッシュ済みの起動は 1 度も network に出ないため、決着後にも中断を見る
  // （この後は同期の組み立てだけなので、これが返却前の最後の関門になる）。
  signal.throwIfAborted();

  let assets: Record<string, Uint8Array<ArrayBuffer>> = {};
  for (const key of keys) {
    const bytes = bytesByRef.get(fileRefKey(files[key]));
    if (bytes === undefined) {
      throw new Error(`hub: ${files[key].path} の bytes が揃っていない（取得層の不変条件破れ）`);
    }
    assets = { ...assets, [key]: bytes };
  }
  return assets;
};

/**
 * {@link streamAssets} が 1 本ずつ引き渡す shard。runtime の `ModelShard` と**構造互換**で、
 * `streamAssets(...)` をそのまま `createSessionFromShards` へ渡せる。
 */
export type StreamedAsset = {
  /**
   * id = manifest の path（`refs` に渡した順で届く）。runtime 側はこれを失敗とフェンスの
   * 帰属先に使う — 到着順の連番では配布形のどのファイルかが決まらない。
   */
  readonly id: string;
  /** 検証（size / sha256）を通ったバイト列。buffer 全体を占める。 */
  readonly bytes: Uint8Array<ArrayBuffer>;
};

/** {@link streamAssets} のオプション（全量面 {@link fetchAssets} と同じ透過）。 */
export type StreamAssetsOptions = FetchAssetsOptions;

/** 相 1（prefetch）を抜けた時点の道具一式 — 相 2 を持つ面（{@link streamAssets}）が引き継ぐ。 */
type PrefetchedPhase = {
  /** ファイル別の受信済みバイト（キーは {@link fileRefKey}）。相 2 の self-heal も同じ表を更新する。 */
  readonly received: Map<string, number>;
  /** 進捗の発火口。`total` は渡した `refs` の size 合計に固定されている。 */
  readonly emit: (phase: AssetPhase, ref: FileRef) => void;
  /** 受信バイトの門を被せた `fetch`。budget の表が同じなので相 2 もこれをそのまま使う。 */
  readonly guarded: typeof globalThis.fetch;
};

/**
 * 全 ref を永続キャッシュへ落とす**相 1** の実体。{@link streamAssets} の相 1 と
 * {@link prefetchAssets} の本体はこの 1 本（同じ機構を 2 つ書くと、同時取得数・真因の復元・
 * 進捗の綴りが片方だけ直る）。
 *
 * `emitComplete` は「この面が `complete` の発行者か」— 相 2 を持つ逐次面では引き渡しの直前が
 * 終端なので `false`（両方が出すと `downloading`* → `complete` を 1 ファイル 1 回とする
 * {@link AssetPhase} の契約が破れる）、相 2 を持たない {@link prefetchAssets} では終端がここしか
 * ないので `true`。
 *
 * `where` は入力検査の文言に載る面の名前（呼び出し側の誤りをどの面で弾いたかが分かる）。
 */
const runPrefetchPhase = async (
  where: string,
  loaded: LoadedManifest,
  refs: readonly FileRef[],
  options: FetchAssetsOptions,
  { emitComplete }: { readonly emitComplete: boolean },
): Promise<PrefetchedPhase> => {
  const { repo, revisionSha, manifest } = loaded;
  const available = manifest.available;
  const target = hfRef(loaded, revisionSha);
  const targetFor = originFor(loaded, target);
  const contextFor = fetchContext(repo, revisionSha);

  // 入力検査は取得の前に済ませる（相 1 は全 shard を落としてしまうので、network に出た後で
  // 呼び出し側の誤りに気づいても帯域が戻らない）。
  if (refs.length === 0) {
    throw new ManifestReferenceError(
      `${where}: 取得対象が 1 つも無い（repo ${repo} @ ${revisionSha}）`,
      { available },
    );
  }
  // 重複検査の表はそのまま進捗の per-file 引き当て（`fileTotal`）にも使う。同一性は
  // {@link fileRefKey}（越境参照は別リポの同名 path を別の 1 本として数える）。
  const declared = new Map<string, FileRef>();
  for (const ref of refs) {
    const refKey = fileRefKey(ref);
    if (declared.has(refKey)) {
      throw new ManifestReferenceError(
        `${where}: 参照 '${refKey}' が重複している（この面は渡された列をそのまま扱う —` +
          ` 同じ shard を 2 回渡すのは呼び出し側の誤り。全量面 fetchAssets は一意化する）`,
        { available },
      );
    }
    declared.set(refKey, ref);
  }

  let total = 0;
  for (const ref of refs) total += ref.size;

  const received = new Map<string, number>();
  const emit = (phase: AssetPhase, ref: FileRef): void => {
    if (options.onProgress === undefined) return;
    const refKey = fileRefKey(ref);
    const fileTotal = fileSizeOf(declared, refKey);
    // complete は全量が揃った点なので size をそのまま渡す（相 1 が温めた分は相 2 でキャッシュ
    // ヒットになり downloading が出ないため、受信実績から引くと 0 に見える）。
    const fileLoaded = phase === "downloading" ? received.get(refKey) ?? 0 : fileTotal;
    // MUST: 全体 `loaded` にも同じ値を積む — この shard ぶんだけ `received` から引くと、
    // 同一イベントで fileLoaded が size なのに loaded がそれを数えない矛盾が出る（全 shard が
    // 温まっている 2 回目以降の起動では loaded が 0 のまま complete が並ぶ）。downloading では
    // fileLoaded が `received` の値そのものなので二重計上にはならない。
    let sum = fileLoaded;
    for (const [other, bytes] of received) {
      if (other !== refKey) sum += bytes;
    }
    options.onProgress({ phase, path: ref.path, loaded: sum, total, fileLoaded, fileTotal });
  };

  const budgets = new Map<string, ByteBudget>();
  for (const ref of refs) {
    budgets.set(hfResolveUrl({ ...targetFor(ref), path: ref.path }), {
      maxBytes: ref.size,
      exact: true,
      violation: (actual, where) =>
        new IntegrityError(
          `${ref.path}: ${where} が manifest の size と食い違う（期待 ${ref.size} / 実際 ${actual}）`,
          {
            ...contextFor(ref),
            path: ref.path,
            expected: String(ref.size),
            actual: String(actual),
            source: "network",
            available,
          },
        ),
    });
  }

  const failure = new AbortController();
  const prefetchSignal = options.signal === undefined
    ? failure.signal
    : AbortSignal.any([failure.signal, options.signal]);
  const guarded = createGuardedFetch(options.fetch ?? globalThis.fetch, budgets);

  // 既存エントリの扱いは記録ハッシュとの突合で決まる（取得層 0.5.0）— 記録が期待 sha256 と
  // 一致すれば network に出ずそのまま温存し、記録が無い / 食い違うエントリは検証付きで温め直す。
  const prefetchOne = async (ref: FileRef): Promise<void> => {
    const refKey = fileRefKey(ref);
    // MUST: 記録ハッシュが一致するエントリは network に出ない＝下層の signal 監視が効かない
    // 区間なので、ファイルごとに明示的に中断を見る（見ないと、取り消しも第一失敗も温まっている
    // ファイルに対してだけ効かず、残り全 ref を舐め切ってから決着する）。全量面 fetchOne・
    // 相 2 と同じ綴り。
    prefetchSignal.throwIfAborted();
    try {
      await prefetchHfFile(targetFor(ref), { path: ref.path, sha256: ref.sha256 }, {
        init: requestInit(options.headers, prefetchSignal),
        fetch: guarded,
        onProgress: (progress) => {
          received.set(refKey, progress.loaded);
          emit("downloading", ref);
        },
        ...(options.caches === undefined ? {} : { caches: options.caches }),
      });
    } catch (error) {
      if (error instanceof HubError || isAborted(error, prefetchSignal)) throw error;
      // MUST: 捕まえた値の同一性だけで中断を判定しない — prefetch はバイト列を手元に持たない
      // 面なので、転送中断も put の reject として現れ、`cause` に沈めて包まれる。signal が
      // 落ちていればその reason（巻き添えなら最初の失敗そのもの）を素通しする。
      // NOTE: 上の `isAborted` を先に通る形（生の AbortError）ではここに来ないので、これだけでは
      //       真因の復元にならない。最終的な決着は相 1 の allSettled の後で reason から取る。
      if (prefetchSignal.aborted) throw prefetchSignal.reason;
      const context = contextFor(ref);
      throw new HubFetchError(
        `${ref.path} の事前取得に失敗した（repo ${context.repo} @ ${context.revisionSha}）`,
        { ...context, path: ref.path, available, cause: error },
      );
    }
    if (emitComplete) {
      // MUST: `received` にも size を書く — 書かずに complete だけ出すと、キャッシュヒット
      // （downloading が 1 度も出ない）だったファイルが後続イベントの `loaded` 合計から抜け、
      // 全体の進捗が巻き戻って見える。
      received.set(refKey, ref.size);
      emit("complete", ref);
    }
  };

  let next = 0;
  const worker = async (): Promise<void> => {
    try {
      while (next < refs.length) await prefetchOne(refs[next++]);
    } catch (error) {
      // 1 本でも落ちたら残りを止める（fail loud — 全体が reject するのに DL を続ける意味はない）。
      failure.abort(error);
      throw error;
    }
  };
  // MUST: 失敗しても全ワーカーの決着を待ってから抜ける（全量面と同じ理由 — 早期 reject だと
  // 呼び出し側が catch した後も取得が背後で走り続ける）。
  await Promise.allSettled(
    Array.from({ length: Math.min(CONCURRENCY, refs.length) }, () => worker()),
  );
  // MUST: 巻き添えではなく**真の第一失敗**を上げる（全量面と同じ理由）。ワーカーの reject を
  // 配列順に拾うと、真犯人が worker[0] 以外だったときに巻き添え側が表面化する。
  if (failure.signal.aborted) throw failure.signal.reason;

  return { received, emit, guarded };
};

/**
 * 資産を永続キャッシュへ**落とすだけ**の面（逐次面 {@link streamAssets} の相 1 単体）。重み
 * shard を先に永続キャッシュへ落とす面 — 後続の {@link streamAssets} 相 2 は network に出ない。
 *
 * 使いどころは「Session を遅延構築するパイプラインのロード時」— 構築が初回の実行まで遅れると
 * 重み shard の DL もそこまで遅れ、ロード進捗にも現れない。ここで先に落としておけば、進捗は
 * ロード中に出揃い、構築はキャッシュ読出しだけで済む。
 *
 * 機構は逐次面の相 1 と同一（同時 {@link CONCURRENCY} 本・`sha256` は通過中に照合・失敗は
 * 真因を復元して {@link HubFetchError}）。バイト列は返さない（RAM に載せない面なので、欲しい
 * ときは {@link streamAssets} / {@link fetchAssets} で読み直す — キャッシュヒットになる）。
 *
 * 進捗は `downloading`* に続けて**ファイルごとに `complete` を 1 回**出す（相 2 を伴わない
 * この面が終端 — {@link AssetPhase} の契約。キャッシュ済みのファイルは `complete` 1 点だけ）。
 *
 * NOTE: `caches` が無い環境・キャッシュ書込み失敗（quota 超過等）では **fail loud**（相 1 と
 * 同じ理由 — バイト列を手元に持たない面なので素 fetch へ縮退する余地が無い）。
 */
export const prefetchAssets = async (
  loaded: LoadedManifest,
  refs: readonly FileRef[],
  options: FetchAssetsOptions = {},
): Promise<void> => {
  await runPrefetchPhase("prefetchAssets", loaded, refs, options, { emitComplete: true });
  // MUST: 全ファイルがキャッシュ済みの呼び出しは 1 度も network に出ない＝下層の signal 監視が
  // 効かないので、決着後にも中断を見る（これが返却前の最後の関門）。
  options.signal?.throwIfAborted();
};

/**
 * shard を **2 相**で読み、1 本ずつ引き渡す逐次面（ADR 0070 決定 2）。
 *
 * - **相 1（prefetch）**: 最初の yield の前に、全 shard を永続キャッシュへ落とす
 *   （streaming — RAM に全量を載せない・同時 4 本）。`sha256` は通過中に照合され、
 *   不一致はエントリ不成立で fail loud（帯域を捨てた後に全量を握って落ちない）。通ったエントリ
 *   には記録ハッシュが焼かれ、既に記録が一致するエントリは network に出ずそのまま温存される。
 * - **相 2（逐次引き渡し）**: `refs` の順に 1 本ずつ「キャッシュから取得 → 呼び手へ渡す →
 *   参照を手放す」。相 1 が焼いた記録と期待 sha256 の突合は取得層が行い（全量ハッシュ 0 回）、
 *   記録が食い違う・バイト数が合わないエントリは self-heal で 1 往復だけ取り直す。
 *
 * 全量面 {@link fetchAssets} との違いは**渡したバイト列への参照を残さない**ことで、RAM ピークが
 * O(最大 shard) に収まる（全量ホスト保持が成立しない検収モデル級のための面）。全量面は温存して
 * あるので、小モデルは従来どおりそちらを使う。
 *
 * MUST: 相 1 は**最初の `next()` まで開始されない**（async generator の遅延）。呼んだだけでは
 * 何も起きず、`for await` に入るか `next()` を呼んだ時点で DL が始まる。空の `refs`・重複 path も
 * その時点（network に出る前）に {@link ManifestReferenceError} で弾く。
 *
 * NOTE: 相 1 は `caches` が無い環境・キャッシュ書込み失敗（quota 超過等）で **fail loud** になる
 * （下層 prefetch の契約 — バイト列を手元に持たない面なので素 fetch へ縮退する余地が無い。黙って
 * 縮退させると RAM ピークの目標が壊れる）。`onCacheError` の診断が届くのは相 2 だけで、相 1 の
 * cache I/O 失敗は {@link HubFetchError} として上がる。
 */
export const streamAssets = async function* (
  loaded: LoadedManifest,
  refs: readonly FileRef[],
  options: StreamAssetsOptions = {},
): AsyncGenerator<StreamedAsset, void, unknown> {
  const { repo, revisionSha, manifest } = loaded;
  const available = manifest.available;
  const target = hfRef(loaded, revisionSha);
  const targetFor = originFor(loaded, target);
  const contextFor = fetchContext(repo, revisionSha);

  // ---- 相 1: 全 shard を永続キャッシュへ落とす（ここを抜けるまで 1 本も yield しない）。
  // 入力検査もこの中（network に出る前）で済む。`complete` は相 2 が出すので発行しない。
  const { received, emit, guarded } = await runPrefetchPhase(
    "streamAssets",
    loaded,
    refs,
    options,
    { emitComplete: false },
  );

  // ---- 相 2: 1 本ずつ引き渡す。
  for (const ref of refs) {
    // 相 2 は大半がキャッシュ読出しで network に出ない＝下層の signal 監視が効かない区間なので、
    // shard の切れ目で明示的に中断を見る（数 GB の読出しを何本も回している最中に取り消しが
    // 効かないのは中断の透過が壊れているのと同じ）。
    options.signal?.throwIfAborted();

    const refKey = fileRefKey(ref);
    let bytes: Uint8Array;
    try {
      bytes = await fetchHfFile(
        targetFor(ref),
        // 相 1 と同じ内容キーになる spec（sha256 が一致するので相 1 が温めたエントリに当たる）。
        // 検証は取得層が記録ハッシュとの突合で済ませる — 全量ハッシュは走らない。
        { path: ref.path, sha256: ref.sha256, expectedBytes: ref.size },
        {
          init: requestInit(options.headers, options.signal),
          fetch: guarded,
          // 相 1 が温めた分はキャッシュヒットなので、ここが発火するのは self-heal の取り直しだけ。
          // NOTE: self-heal は evict してから取り直すため、その 1 巡だけ `loaded` はそのファイル
          //       ぶん巻き戻る（phase 契約が認めている「最初からやり直し」と同じ 1 巡）。
          //       `fileLoaded` も同じ 1 巡だけそのファイルの先頭から数え直しになる。
          onProgress: (progress) => {
            received.set(refKey, progress.loaded);
            emit("downloading", ref);
          },
          ...(options.caches === undefined ? {} : { caches: options.caches }),
          ...(options.onCacheError === undefined ? {} : { onCacheError: options.onCacheError }),
        },
      );
    } catch (error) {
      if (error instanceof HubError || isAborted(error, options.signal)) throw error;
      const context = contextFor(ref);
      throw new HubFetchError(
        `${ref.path} の取得に失敗した（repo ${context.repo} @ ${context.revisionSha}）`,
        { ...context, path: ref.path, available, cause: error },
      );
    }
    // MUST: yield の直前にも中断を見る — 冒頭の確認だけだと「最終 shard のキャッシュ読出しの
    // 最中に中断された」形が観測されず、取り消したはずのロードが正常完了して下流の Session
    // 構築まで走る（検証済みバイトを配ってから止まるのでは中断の意味が無い）。
    options.signal?.throwIfAborted();
    const asset = assertTightView(bytes, ref.path);
    received.set(refKey, ref.size);
    emit("complete", ref);
    // MUST: ここで手放す — 引き渡したバイト列を generator 側の表に溜めない（溜めた瞬間に
    // 全量面と同じ RAM 特性に戻り、この面の存在理由が消える）。次の反復に入れば `bytes` の
    // 束縛ごと到達不能になるので、常駐するのは「今の 1 本」だけ。
    yield { id: ref.path, bytes: asset };
  }
};

/**
 * ダウンロード済みの資産を消して容量を空ける。対象は**取得層の名前空間まるごと**と、まだ
 * 残っているなら旧名前空間（`karume/1` 系）。記録ハッシュを信じる既定を疑ったときの回復手段も
 * これ（`recheck` 相当のノブは持たない — DECIDED: `docs/decisions/0080-hub-fetch-cache-050.md`）。
 *
 * NOTE: 取得層 0.5.0 の名前空間は内部固定 1 個で、キーに「どのアプリが温めたか」は入らない。
 * したがって**同じ origin のアプリが `@hdae/fetch-cache` を直接使って温めたものも一緒に消える**
 * （分離する手段がライブラリ側に無い）。repo 単位の細粒度掃除はキャッシュ保守波へ送り、ここは
 * 「全部消す」の 1 択に留める。
 *
 * @returns 何かが実在して消えたら `true`（元から 1 つも無ければ `false`）。
 */
export const clearHubCache = async (
  options: { readonly caches?: CacheStorage } = {},
): Promise<boolean> => {
  const storage = options.caches ?? globalThis.caches;
  if (storage === undefined) {
    // 黙って no-op にしない。Cache Storage が無い環境（非セキュアオリジン等）で「消したつもり」に
    // なるのが最悪 — 消えていない写しをアプリが消えたものとして扱う。
    throw new Error(
      "hub: この環境に CacheStorage が無いためキャッシュを消せない" +
        "（options.caches で明示的に渡す）",
    );
  }
  // 旧名前空間はここでも回収する（{@link loadManifest} を 1 度も通さずに掃除だけする呼び方が
  // あるため）。こちらは掃除そのものが仕事なので、失敗を握り潰さず素通しする。
  const legacy = await legacyCacheNames(storage);
  const deleted = await Promise.all(legacy.map((name) => storage.delete(name)));
  // 名前空間の名前は取得層が所有するので、綴りを hub に焼かず掃除 API へ委ねる。
  const cleared = await clearCache({ caches: storage });
  return cleared || deleted.includes(true);
};
