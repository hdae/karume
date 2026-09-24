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
 * - 区間読み（`source.ts` ⑧）— 溜めたエントリの中を `[offset, offset + length)` だけ開く
 *   （取得層 0.8.0 の `openHfFile`）。読めるのはキャッシュの中身だけなので、未取得の参照は
 *   相 1 と同じ温めを 1 度挟んでから開き直す。
 *
 * MUST NOT: ここでエラーを組み立てない（診断の文脈を持つのは共通層 — `context.ts`）。
 * 例外は「取得層の不変条件破れ」を告げる素の `Error` だけ。
 */

import { evict as evictKey, listKeys } from "@hdae/fetch-cache";
import {
  fetchHfFile,
  isCommitSha,
  openHfFile,
  prefetchHfFile,
  resolveHfRevision,
} from "@hdae/fetch-cache/hf";
import { type FileRef, fileRefKey, MANIFEST_FILENAME } from "../manifest.ts";
import type { HubRepoRef, LoadManifestOptions } from "../session.ts";
import {
  type AssetRangeReader,
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
  /**
   * 相 1 の実体（④`prefetchFile` と ⑧`openFile` の温め直しが共用する）。バイト列を手元に持たず
   * 永続キャッシュへ落とすだけの面で、`expectedBytes` は**受信の上限**として効く（取得層 ADR 0011）
   * — 宣言を超えた時点で打ち切るので、この面でも上限が抜けない。
   */
  const warmFile = async (
    ref: FileRef,
    { signal, onProgress }: {
      readonly signal?: AbortSignal;
      readonly onProgress?: (loaded: number) => void;
    },
  ): Promise<void> => {
    await prefetchHfFile(target, { path: ref.path, sha256: ref.sha256, expectedBytes: ref.size }, {
      init: requestInit(options.headers, signal),
      fetch: baseFetch,
      ...(onProgress === undefined
        ? {}
        : { onProgress: (progress) => onProgress(progress.loaded) }),
      ...shared,
      ...(options.onRetry === undefined ? {} : { onRetry: options.onRetry }),
    });
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

    readFile: async (ref, { signal, onProgress }) => {
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
      await warmFile(ref, { ...(signal === undefined ? {} : { signal }), onProgress });
    },

    // ⑧区間読み。開くのはキャッシュの中身だけで、`openHfFile` は**network に出ない**（revision の
    // 解決もしない）ので、未取得・記録なし・記録不一致（開く側が self-heal で消す）はどれも
    // `undefined` として返る。そこで**相 1 と同じ温め**を 1 度だけ挟んでから開き直す — 呼び手に
    // 「先に prefetchAssets を通せ」という手順を負わせないため（消費側は行を要求するだけでよい）。
    openFile: async (ref, { signal }): Promise<AssetRangeReader> => {
      const spec = { path: ref.path, sha256: ref.sha256, expectedBytes: ref.size };
      const openOptions = {
        ...shared,
        ...(options.onCacheError === undefined ? {} : { onCacheError: options.onCacheError }),
      };
      let opened = await openHfFile(target, spec, openOptions);
      if (opened === undefined) {
        await warmFile(ref, { ...(signal === undefined ? {} : { signal }) });
        opened = await openHfFile(target, spec, openOptions);
      }
      if (opened === undefined) {
        // 温めが成功した直後に開けないのは取得層の不変条件破れ（相 1 は検証付きで記録ハッシュを
        // 焼くので、そこを通れば同じ内容キーで開けるはず）か、温めと開き直しの間に在庫が
        // 消されたか（在庫削除 ⑦ との競合）のどれかなので、候補を全て名指しして落とす。
        throw new Error(
          `@karume/hub: ${ref.path} の区間読み口が温め直した直後も開かない` +
            `（記録ハッシュを保持しないキャッシュ / CacheStorage が使えない / 温めと開き直しの` +
            `間に在庫が消された〈evictCachedAssets・clearHubCache との競合〉の可能性）`,
        );
      }
      const entry = opened;
      return {
        // 費用の型は取得層の読み出し戦略そのもの: "blob" は遅延 Blob の slice（offset に依らない）・
        // "stream" は本文の読み飛ばし（offset に比例）。Deno の既定は "stream"（`blob()` が全量を
        // ヒープへ載せるため）なので、Deno では "scan"・ブラウザでは "seek" になる。
        cost: entry.strategy === "blob" ? "seek" : "scan",
        read: async (offset, length, readOptions) => {
          const bytes = await entry.read(offset, length, readOptions);
          const { buffer } = bytes;
          // 取得層の戻り型は buffer の種別を持たない（`Uint8Array<ArrayBufferLike>`）。消費側は
          // 返ったバイト列をそのまま TypedArray として読むので、SharedArrayBuffer は黙って
          // 通さない（`as` で潰すと、共有メモリ由来の view が型だけ健全に見える）。
          if (!(buffer instanceof ArrayBuffer)) {
            throw new Error(
              `@karume/hub: ${ref.path} の区間読みが ArrayBuffer 以外の buffer を返した`,
            );
          }
          // 長さ検査（`length` ちょうどか）と tight view 検査は取得層と共通層が持つので重ねない。
          return new Uint8Array(buffer, bytes.byteOffset, bytes.byteLength);
        },
      };
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
