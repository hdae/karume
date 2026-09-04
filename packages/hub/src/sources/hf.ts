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
 *   0 回）。記録が食い違うエントリは自動で evict → 取り直し（self-heal）。
 * - 受信バイトの門（`transport.ts` の予算付き `fetch`）— content-length と実受信の両方を見る。
 * - 相 1（streaming prefetch）— **RAM に載せずに永続キャッシュへ落とす** HTTP 固有の最適化。
 *
 * MUST NOT: ここでエラーを組み立てない（診断の文脈を持つのは共通層 — `context.ts`）。
 * 例外は「取得層の不変条件破れ」を告げる素の `Error` だけ。
 */

import {
  fetchHfFile,
  hfResolveUrl,
  isCommitSha,
  prefetchHfFile,
  resolveHfRevision,
} from "@hdae/fetch-cache/hf";
import { MANIFEST_FILENAME, MAX_MANIFEST_BYTES } from "../manifest.ts";
import type { HubRepoRef, LoadManifestOptions } from "../session.ts";
import {
  DistributionSource,
  type PinnedSource,
  type SizeViolation,
  type SourceDriver,
} from "../source.ts";
import { type ByteBudget, createGuardedFetch } from "../transport.ts";

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

/** URL 1 本ぶんのバイト門を被せた `fetch`。予算の無い URL（revision 解決 API 等）は素通しする。 */
const guardedFetchFor = (
  base: typeof globalThis.fetch,
  url: string,
  budget: ByteBudget,
): typeof globalThis.fetch => createGuardedFetch(base, new Map([[url, budget]]));

/** 資産 1 本の予算: バイト数は manifest の宣言と**厳密一致**でなければならない。 */
const exactBudget = (size: number, violation: SizeViolation): ByteBudget => ({
  maxBytes: size,
  exact: true,
  violation,
});

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

  return {
    // 診断の名乗り（HF は repo と commit SHA を持つ取得元 — 完全性検証は network 取得の側）。
    origin: {
      label: `repo ${repo} @ ${generation}`,
      integrity: "network",
      repo,
      revisionSha: generation,
    },

    readManifest: async ({ parse, signal, sizeViolation }) => {
      const url = hfResolveUrl({ ...target, path: MANIFEST_FILENAME });
      // MUST: UTF-8 decode と parse は取得層の `validate` フックの中で行う — 取得の外でやると
      // 破損したキャッシュエントリが evict されず、`clearHubCache` を手で叩くまで毎回同じ
      // ManifestFormatError を返し続ける（資産側と同じ self-heal 経路に揃える）。
      // MUST NOT: このフックへ中断確認を混ぜない — フックの throw は下層で「破損」と解釈され、
      // 健全なキャッシュエントリの evict と取り直しを招く。
      await fetchHfFile(target, { path: MANIFEST_FILENAME, validate: parse }, {
        init: requestInit(options.headers, signal),
        // manifest は正本の根なので**事前の期待 sha256 を持てない**。上限だけを課す
        // （`exact: false` — 実際のバイト数は manifest ごとに違う）。
        fetch: guardedFetchFor(baseFetch, url, {
          maxBytes: MAX_MANIFEST_BYTES,
          exact: false,
          violation: sizeViolation,
        }),
        ...shared,
        ...(options.onCacheError === undefined ? {} : { onCacheError: options.onCacheError }),
      });
    },

    readFile: async (ref, { signal, onProgress, sizeViolation, into }) => {
      const url = hfResolveUrl({ ...target, path: ref.path });
      return await fetchHfFile(
        target,
        {
          path: ref.path,
          // 検証は取得層が持つ（受信中のハッシュ / 記録ハッシュの突合 / 不一致の self-heal）。
          sha256: ref.sha256,
          // バイト数の門であり、同時に受信バッファの前確保サイズでもある。確保自体が失敗する
          // 大きさ（Chromium の単一 ArrayBuffer 上限超え）なら受信前に throw されるので、
          // 数 GB を撃ち終わってから落ちることがない。
          expectedBytes: ref.size,
          // 逐次面の器（最大 shard 長 1 本）があれば取得層にそこへ書かせる — 受信もキャッシュ
          // 読出しも器の先頭へ入り、shard 毎のバッファ確保が消える（取得層 `into`・ADR 0070 追記）。
          // 器の先頭 `size` バイトを指す view が返る契約は取得層側が保証する（容量不足は throw）。
          ...(into === undefined ? {} : { into: into() }),
        },
        {
          init: requestInit(options.headers, signal),
          fetch: guardedFetchFor(baseFetch, url, exactBudget(ref.size, sizeViolation)),
          // network 側だけ発火する（キャッシュヒットは complete の 1 点だけで進む）。
          // `loaded` が `size` を超えないことは受信バイトの門（transport.ts）が保証する。
          onProgress: (progress) => onProgress(progress.loaded),
          ...shared,
          ...(options.onCacheError === undefined ? {} : { onCacheError: options.onCacheError }),
        },
      );
    },

    // 相 1: バイト列を手元に持たない面。既存エントリの扱いは記録ハッシュとの突合で決まる
    // （取得層 0.5.0）— 記録が期待 sha256 と一致すれば network に出ずそのまま温存し、記録が
    // 無い / 食い違うエントリは検証付きで温め直す。`caches` 不在・put 失敗は fail loud
    // （素 fetch へ縮退する余地が無い — 縮退させると RAM ピークの目標が壊れる）。
    prefetchFile: async (ref, { signal, onProgress, sizeViolation }) => {
      const url = hfResolveUrl({ ...target, path: ref.path });
      await prefetchHfFile(target, { path: ref.path, sha256: ref.sha256 }, {
        init: requestInit(options.headers, signal),
        fetch: guardedFetchFor(baseFetch, url, exactBudget(ref.size, sizeViolation)),
        onProgress: (progress) => onProgress(progress.loaded),
        ...shared,
      });
    },

    // 越境先も同じアダプター（参照先は commit SHA 固定が必須なので、越境側で解決は起きない）。
    // 資産のキャッシュキーは内容キー（`hubUrl` を含まない）なので、同じバイト列ならミラーを
    // 跨いでも 1 エントリを共有する。別リポの同名 path はキーに `repo` が入るぶん別エントリ。
    originFor: (crossRepo, crossRevision) =>
      pinnedHfSource(crossRepo, crossRevision, hubUrl, options),
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
