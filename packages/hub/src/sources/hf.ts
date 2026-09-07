/**
 * HuggingFace 取得元アダプター（ADR 0038 §5「hub の取得層」）。土台は `@hdae/fetch-cache`
 * （実行時依存ゼロ・Web 標準 API のみ）。取得元契約 `source.ts` の実装のひとつ
 * （もう 1 つは `local.ts` — こちらだけが世代・キャッシュ・network を持つ）。
 *
 * このアダプターが持つもの（= HF / HTTP + 永続キャッシュに固有のもの）:
 * - 可変 ref → commit SHA の解決と、暗黙 `main` の pin 案内
 * - キャッシュの所有（0.5.0 で名前空間が内部固定 1 個になった）。資産のキーは内容キー
 *   `["hf", kind, repo, path, sha256]` なので、revision が動いてもバイト不変のファイルはヒットの
 *   まま読める。manifest だけは事前の期待 sha が無いので SHA 固定 resolve URL がキー。
 * - **資産の完全性検証**（spec の `sha256` / `expectedBytes` を取得層へ委ねる）。取得時に検証して
 *   記録ハッシュをエントリへ焼き、以後のヒットは記録との文字列比較だけで済ませる（全量ハッシュ
 *   0 回）。記録が食い違うエントリは自動で evict → 取り直し（self-heal）。受信バイトの上限も
 *   同じ `expectedBytes` が兼ねる — 宣言を超えた時点で取得層が受信を打ち切る（取得層 ADR 0011）。
 * - 相 1（streaming prefetch）— **RAM に載せずに永続キャッシュへ落とす** HTTP 固有の最適化。
 * - キャッシュ在庫の照会と削除（`source.ts` ⑥⑦）— 溜めているのがこの取得元なので、
 *   「何が手元にあるか」「これを消せるか」に答えられるのもここだけ。
 *
 * MUST NOT: ここでエラーを組み立てない（診断の文脈を持つのは共通層 — `context.ts`）。
 * 例外は「取得層の不変条件破れ」を告げる素の `Error` だけ。
 */

import { evict as evictKey, listKeys } from "@hdae/fetch-cache";
import { fetchHfFile, isCommitSha, prefetchHfFile, resolveHfRevision } from "@hdae/fetch-cache/hf";
import { type FileRef, fileRefKey, MANIFEST_FILENAME } from "../manifest.ts";
import type { HubRepoRef, LoadManifestOptions } from "../session.ts";
import {
  DistributionSource,
  type PinnedSource,
  type SourceDriver,
  type SourceOrigin,
} from "../source.ts";

/** HF 上の 1 つの座標（世代は解決済み）。`hubUrl` は**ホストの選択**なので越境先にも効かせる。 */
type HfTarget = { readonly repo: string; readonly revision: string; readonly hubUrl?: string };

const hfTarget = (repo: string, revision: string, hubUrl?: string): HfTarget => ({
  repo,
  revision,
  ...(hubUrl === undefined ? {} : { hubUrl }),
});

const requestInit = (headers?: HeadersInit, signal?: AbortSignal): RequestInit => ({
  ...(headers === undefined ? {} : { headers }),
  ...(signal === undefined ? {} : { signal }),
});

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
      `  ② @karume/models の *_SOURCES（公開配布リポの対応表 — パッケージ検証済みの pin）を使う`,
  );
};

/**
 * 資産 1 本のキャッシュキー（取得層 HF 層の**内容キー**）。hub は `kind` を渡さないので
 * 第 2 要素は常に `"model"`。
 *
 * MUST: 綴りをここ 1 箇所に閉じる — 在庫の照会（プレフィックス `["hf", "model", repo]`）と
 * 削除（5 要素の完全キー）が同じ式から外れると、消したつもりのエントリが残る。
 */
const contentKey = (repo: string, ref: FileRef): readonly string[] => [
  "hf",
  "model",
  repo,
  ref.path,
  ref.sha256,
];

/** 在庫の突合キー。repo はプレフィックスで絞り込み済みなので (path, sha256) の組で足りる。 */
const stockKey = (path: string, sha256: string): string => `${path} ${sha256}`;

const pinnedHfSource = (
  repo: string,
  generation: string,
  hubUrl: string | undefined,
  options: LoadManifestOptions,
): PinnedSource => {
  const target = hfTarget(repo, generation, hubUrl);
  const baseFetch = options.fetch ?? globalThis.fetch;
  const shared = {
    ...(options.caches === undefined ? {} : { caches: options.caches }),
  };
  // 診断の名乗り（HF は repo と commit SHA を持つ取得元 — 完全性検証は network 取得の側）。
  const origin: SourceOrigin = {
    label: `repo ${repo} @ ${generation}`,
    integrity: "network",
    repo,
    revisionSha: generation,
  };
  return {
    origin,

    readManifest: async ({ parse, signal }) => {
      // manifest は正本の根なので**事前の期待 sha256 も期待バイト数も持てない**。取得層に
      // 「厳密一致なしの上限」は無いので、1 MiB の上限は `parse`（`parseManifest`）が全量を
      // 受け取ってから見る（受信を途中で止める門はここには無い）。
      // MUST: UTF-8 decode と parse は取得層の `validate` フックの中で行う — 取得の外でやると
      // 破損したキャッシュエントリが evict されず、`clearHubCache` を手で叩くまで毎回同じ
      // ManifestFormatError を返し続ける（資産側と同じ self-heal 経路に揃える）。
      // MUST NOT: このフックへ中断確認を混ぜない — フックの throw は下層で「破損」と解釈され、
      // 健全なキャッシュエントリの evict と取り直しを招く。
      await fetchHfFile(target, { path: MANIFEST_FILENAME, validate: parse }, {
        init: requestInit(options.headers, signal),
        fetch: baseFetch,
        ...shared,
        ...(options.onCacheError === undefined ? {} : { onCacheError: options.onCacheError }),
        ...(options.onRetry === undefined ? {} : { onRetry: options.onRetry }),
      });
    },

    // NOTE: 区間読み（`source.ts` ⑧ `openFile`）はまだ持たない。取得層 `@hdae/fetch-cache` 0.8 の
    // `openHfFile` が入った時点で、費用の型 = 戦略（blob → "seek" / stream → "scan"）として載せる。
    // それまでこの取得元では `openAsset` が `undefined` を返し、呼び手は全量読みへ倒す。

    readFile: async (ref, { signal, onProgress, into }) => {
      return await fetchHfFile(
        target,
        {
          path: ref.path,
          // 検証は取得層が持つ（受信中のハッシュ / 記録ハッシュの突合 / 不一致の self-heal）。
          sha256: ref.sha256,
          // バイト数の門（受信の上限 + 全量受信後の厳密一致）であり、同時に受信バッファの
          // 前確保サイズでもある。確保自体が失敗する大きさ（Chromium の単一 ArrayBuffer 上限
          // 超え）なら受信前に throw されるので、数 GB を撃ち終わってから落ちることがない。
          expectedBytes: ref.size,
          // 逐次面の器（最大 shard 長 1 本）があれば取得層にそこへ書かせる — 受信もキャッシュ
          // 読出しも器の先頭へ入り、shard 毎のバッファ確保が消える（取得層 `into`・ADR 0070 追記）。
          // 器の先頭 `size` バイトを指す view が返る契約は取得層側が保証する（容量不足は throw）。
          ...(into === undefined ? {} : { into: into() }),
        },
        {
          init: requestInit(options.headers, signal),
          fetch: baseFetch,
          // network 側だけ発火する（キャッシュヒットは complete の 1 点だけで進む）。
          // `loaded` が `size` を超えないことは取得層の受信上限（`expectedBytes`）が保証する。
          onProgress: (progress) => onProgress(progress.loaded),
          ...shared,
          ...(options.onCacheError === undefined ? {} : { onCacheError: options.onCacheError }),
          ...(options.onRetry === undefined ? {} : { onRetry: options.onRetry }),
        },
      );
    },

    // 相 1: バイト列を手元に持たない面。既存エントリの扱いは記録ハッシュとの突合で決まる
    // （取得層 0.5.0）— 記録が期待 sha256 と一致すれば network に出ずそのまま温存し、記録が
    // 無い / 食い違うエントリは検証付きで温め直す。`caches` 不在・put 失敗は fail loud
    // （素 fetch へ縮退する余地が無い — 縮退させると RAM ピークの目標が壊れる）。
    prefetchFile: async (ref, { signal, onProgress }) => {
      // 相 1 でも `expectedBytes` は**受信の上限**として効く（取得層 ADR 0011）— 宣言を超えた
      // 時点で打ち切るので、バイト列を手元に持たないこの面でも上限が抜けない。
      await prefetchHfFile(
        target,
        { path: ref.path, sha256: ref.sha256, expectedBytes: ref.size },
        {
          init: requestInit(options.headers, signal),
          fetch: baseFetch,
          onProgress: (progress) => onProgress(progress.loaded),
          ...shared,
          ...(options.onRetry === undefined ? {} : { onRetry: options.onRetry }),
        },
      );
    },

    // 越境先も同じアダプター（参照先は commit SHA 固定が必須なので、越境側で解決は起きない）。
    // 資産のキャッシュキーは内容キー（`hubUrl` を含まない）なので、同じバイト列ならミラーを
    // 跨いでも 1 エントリを共有する。別リポの同名 path はキーに `repo` が入るぶん別エントリ。
    originFor: (crossRepo, crossRevision) =>
      pinnedHfSource(crossRepo, crossRevision, hubUrl, options),

    // ⑥在庫。**listKeys は repo プレフィックスで 1 回だけ引く** — 参照ごとに引くと、参照の
    // 本数だけキャッシュ全体の列挙が走る（数十コンポーネントの manifest で効く差）。
    inventory: async (refs) => {
      const keys = await listKeys(["hf", "model", repo], shared);
      // 内容キーは 5 要素固定（`contentKey`）。それ以外の形はこの取得元が書いたものではない。
      const stock = new Set(
        keys.filter((key) => key.length === 5).map((key) => stockKey(`${key[3]}`, `${key[4]}`)),
      );
      return new Set(
        refs.filter((ref) => stock.has(stockKey(ref.path, ref.sha256))).map(fileRefKey),
      );
    },

    // ⑦削除。プレフィックス意味論だが完全キーを渡すので、対象はそのエントリ 1 件だけ
    // （件数 > 0 = 実在して消えた）。
    evict: async (refs) => {
      const removed: FileRef[] = [];
      for (const ref of refs) {
        const count = await evictKey(contentKey(repo, ref), shared);
        if (count > 0) removed.push(ref);
      }
      return removed;
    },
  };
};

/**
 * HF の取得元を作る。`ref.revision` は可変 ref でよい（{@link SourceDriver.resolveGeneration}
 * が commit SHA へ解決する）。
 *
 * 取得の作法（`fetch` / `caches` / `headers`）は**面ごと**に渡る（`pin` の引数）— この factory が
 * 持つのは「どこから取るか」（repo / 要求 ref / ミラー）だけ。
 */
export const createHfSource = (ref: HubRepoRef): DistributionSource => {
  const requested = ref.revision ?? "main";
  const driver: SourceDriver = {
    // 世代解決前の名乗り。解決に失敗したときの診断はこれ 1 つで「どこの何を引きに行ったか」を
    // 言えなければならないので、要求した ref（可変 ref のまま）まで載せる。
    origin: { label: `repo ${ref.repo} @ ${requested}`, integrity: "network", repo: ref.repo },

    resolveGeneration: async (options) => {
      if (isCommitSha(requested)) return requested;
      const revisionSha = await resolveHfRevision(hfTarget(ref.repo, requested, ref.hubUrl), {
        ...(options.fetch === undefined ? {} : { fetch: options.fetch }),
        init: requestInit(options.headers, options.signal),
        ...(options.onRetry === undefined ? {} : { onRetry: options.onRetry }),
      });
      // 解決の**後**に出す — 印字する SHA が確定するのがここで、解決に失敗した場合は警告ではなく
      // 失敗そのものが報告されるべきだから（fail loudly が先）。
      if (ref.revision === undefined) warnImplicitMain(ref.repo, revisionSha);
      return revisionSha;
    },

    pin: (generation, options) => pinnedHfSource(ref.repo, generation, ref.hubUrl, options),
  };
  return new DistributionSource(driver);
};
