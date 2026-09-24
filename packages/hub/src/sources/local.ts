/**
 * ローカル取得元アダプター — 「配布形がもう手元にある」場合の取得元。純 Web 標準（このファイル
 * は特定ランタイムの API を一切参照しない）。実体の読みは {@link DirectoryAdapter} 1 面へ委ね、
 * そこへ Deno / OPFS / IndexedDB / File System Access の picker を差し込む。
 *
 * HF 取得元（`hf.ts`）との違いは能力そのもの:
 *
 * - **世代を持たない** — 手元のディレクトリは「今そこにある内容」しかなく、可変 ref も commit
 *   SHA も無い。暗黙 `main` の警告も出ない（固定するものが無い）。
 * - **相 1（prefetch）を持たない** — CacheStorage を通らないので「温める」に意味がない
 *   （`source.ts` ④）。バイト列の複製が 1 つも増えないのがローカル取得元の最大の利点で、
 *   キャッシュへ写すのは害でしかない。
 * - **区間読み（`source.ts` ⑧）を持てる** — アダプターが位置読み（`readFileRange`）を持つときだけ
 *   `openFile` が生え、費用の型は `"seek"`（offset に依らない）。
 *   ⑧ が見るのは**宣言 size の境界だけ**で、全量面の size 門（`sizeViolation`）も sha256 も
 *   掛からない（区間だけを読む以上、宣言と照合できる全量が手元に無い）— 実体の破損は読んだ行の
 *   値として現れる。
 * - **検証は size 厳密一致のみ**（sha256 は信頼する）。手元の
 *   ファイルは配布元と同じ「取得物」ではなく利用者の資産で、毎起動の全量ハッシュ（数 GiB）に
 *   見合う脅威が無い。size は読み終えた時点でタダで分かるので門として残す（途中で切れた
 *   コピー・別 quant の取り違えはここで落ちる）。
 * - **越境は明示 mapping だけ** — 隣接する同名ディレクトリを推測しない（`originFor`）。
 * - **在庫の削除を持たない** — 消せるのは「取ってきて溜めたもの」だけで、ディレクトリの中身は
 *   利用者の資産。在庫の照会（`source.ts` ⑥）の方は「全部ある」と答える。
 *
 * MUST NOT: ここでエラーを組み立てない（診断の文脈を持つのは共通層 — `context.ts`）。例外は
 * 「呼び手の設定が足りない」ことを告げる素の `Error`（未 mapping の越境）で、共通層がそれを
 * 取得失敗として `cause` に残したまま包む。
 */

import { type FileRef, fileRefKey, MANIFEST_FILENAME, MAX_MANIFEST_BYTES } from "../manifest.ts";
import type { LoadManifestOptions } from "../session.ts";
import {
  type AssetRangeReader,
  DistributionSource,
  driverOf,
  type PinnedSource,
  type SourceDriver,
  type SourceOrigin,
} from "../source.ts";

/**
 * ディレクトリ 1 つぶんの読み口。**必須はこの 1 メソッド（`readFile`）だけ**で、実装は
 * `@karume/hub/deno` の `denoDirectory`（`Deno.readFile`）のほか、ブラウザでは OPFS
 * （`FileSystemDirectoryHandle` → `File.arrayBuffer()`）・IndexedDB・File System Access の
 * picker が同じ形で乗る。
 *
 * MUST: `readFile` が返す `Uint8Array` は **buffer 全体を占める**（tight view）— 共通層はここで
 * 受けたバイト列をそのまま `openContainer` の `bytes` 入力へ渡すので、余白のある view を返すと
 * 辻褄合わせの `slice` で RAM ピークが倍増する（共通層の tight view 検査がその場で落とす）。
 * MUST: 欠損は fail loudly（`undefined` や空バイト列を返さない）。エラーには**実体のパス**を
 * 載せる — 共通層が付けられるのは manifest 上の相対 path までで、「どのディレクトリの下を
 * 探したか」を知っているのはアダプターだけ。
 * MUST: `signal` を透過する（大きい part の読みは中断できなければならない）。
 *
 * `readFileRange`（任意）は**区間読み**（`source.ts` ⑧）の実体側: `[offset, offset + length)` を
 * `length` ちょうど返す（足りなければ throw — 短い戻りを返さない）。ディレクトリの実体は位置読みが
 * できるので、これを持つアダプターの読み口は費用の型 `"seek"` を名乗る。持たないアダプター
 * （区間だけを安く取れない読み口）では `openFile` ごと生えず、消費側は全量読みへ倒す。
 */
export type DirectoryAdapter = {
  readonly readFile: (
    path: string,
    options: { readonly signal?: AbortSignal },
  ) => Promise<Uint8Array<ArrayBuffer>>;
  readonly readFileRange?: (
    path: string,
    offset: number,
    length: number,
    options: { readonly signal?: AbortSignal },
  ) => Promise<Uint8Array<ArrayBuffer>>;
};

/** {@link localDirectory} の設定。 */
export type LocalDirectoryOptions = {
  /**
   * 診断に載せるディレクトリの名前（`ディレクトリ <label>` の形で全エラーの文言に出る）。
   * 省略時は `"(ローカル)"` — アダプターがパスを知っている場合は必ず渡すこと
   * （`denoDirectory` は解決済みの root をここへ入れる）。
   */
  readonly label?: string;
  /**
   * 越境参照（`FileRef` の `repo` + `revision` — ADR 0038 §7）の取得元。キーは manifest が
   * 宣言している `"owner/name"` で、値はその repo を**まるごと**提供する取得元。
   *
   * MUST: 明示 mapping だけを見る — 「隣に同名のディレクトリがあればそれ」のような推測は、
   * 取り違えたバイト列を黙って読ませる（サイズが合えば通る）。宣言が無ければ落とす。
   */
  readonly crossRepo?: Readonly<Record<string, DistributionSource>>;
  /**
   * mapping に無い越境参照の委譲先（例: 別のローカルディレクトリの取得元 — Deno なら
   * `denoDirectory`）。**明示した場合だけ**降格する（暗黙のリモート降格は禁止 —
   * オフライン前提の配布が黙って network へ出る）。
   *
   * NOTE: HF リポを {@link DistributionSource} のハンドルとして組む公開 API は無い
   * （`loadManifest` / `fromPretrained` が `HubRepoRef` を内部で解決する）ので、ここでリモートへ
   * 降格させる構成は現在の公開面では作れない。
   */
  readonly fallback?: DistributionSource;
};

/** 越境先を宣言された (repo, revision) の座標で開く。 */
const pinTarget = (
  target: DistributionSource,
  revision: string,
  options: LoadManifestOptions,
): PinnedSource => driverOf(target).pin(revision, options);

const missingCrossRepo = (label: string, repo: string, revision: string): Error =>
  new Error(
    `@karume/hub: ローカル取得元（ディレクトリ ${label}）に repo '${repo}' の越境先が無い。` +
      `localDirectory の crossRepo に { "${repo}": <取得元> } を渡すか、fallback を指定すること` +
      `（宣言 revision ${revision} — 隣接する同名ディレクトリを推測して読むことはしない）`,
  );

/**
 * ⑧区間読み口（`source.ts`）。**開く時点では実体に触れない** — 読みは `read` ごとに起きるので、
 * 開いたまま使わなければ I/O は 1 回も走らない。
 *
 * 区間の検査を**アダプターへ降ろさない**のは、アダプターが知っているのが実体だけだから: 実体は
 * 宣言 `size` より長いことがある（別 quant の取り違え・書きかけのコピー）ので、実体長で検査すると
 * manifest の外側のバイト列が黙って読める。宣言 size を持っているのはここ（取得元）だけ。
 */
const localRangeReader = (
  readRange: NonNullable<DirectoryAdapter["readFileRange"]>,
  ref: FileRef,
): AssetRangeReader => ({
  // ディレクトリの実体は位置読みができるので、費用は offset に依らない。
  cost: "seek",
  read: async (offset, length, options = {}) => {
    if (
      !Number.isInteger(offset) || !Number.isInteger(length) || offset < 0 || length < 0 ||
      offset + length > ref.size
    ) {
      throw new Error(
        `hub: ${ref.path} の区間 [${offset}, ${offset + length}) が不正` +
          `（宣言 size ${ref.size} — 非負整数で size に収まる区間だけを読める）`,
      );
    }
    const { signal } = options;
    const bytes = await readRange(ref.path, offset, length, {
      ...(signal === undefined ? {} : { signal }),
    });
    // MUST: 長さ違いを黙って通さない（短ければ消費側が 0 埋めの行を正常な値として読み、
    // 長ければ要求の外のバイト列が混じる）。短い / 長いで原因が別なので文言を分ける。
    if (bytes.byteLength !== length) {
      const cause = bytes.byteLength < length
        ? `実体が宣言 size ${ref.size} より短い`
        : "アダプターが要求より長い戻りを返した（`readFileRange` の契約違反）";
      throw new Error(
        `hub: ${ref.path} の区間読みが offset ${offset} の ${length} バイト要求に` +
          ` ${bytes.byteLength} バイトを返した（${cause}）`,
      );
    }
    return bytes;
  },
});

const pinnedLocalSource = (
  adapter: DirectoryAdapter,
  settings: LocalDirectoryOptions & { readonly label: string },
  options: LoadManifestOptions,
): PinnedSource => {
  // 能力の有無はここで 1 度だけ見る（束縛にしておくと、口の中でも型が絞れたまま使える）。
  const readRange = adapter.readFileRange;
  const origin: SourceOrigin = {
    label: `ディレクトリ ${settings.label}`,
    // 取り直しても同じバイト列が返る失敗元（network のような再試行の余地が無い）。
    integrity: "local",
  };
  return {
    origin,

    readManifest: async ({ parse, signal, sizeViolation }) => {
      const bytes = await adapter.readFile(MANIFEST_FILENAME, {
        ...(signal === undefined ? {} : { signal }),
      });
      // NOTE: 全量を読んでから門を見る（アダプターは全量読みしか持たない）。手元の実体に対する
      // 形式検査で、送出側の悪意を想定する門ではないので、読み切ってから落として構わない
      // （HF 取得元の `karume.json` も取得層に厳密一致なしの上限が無いため、同じく全量受信後に
      // `parseManifest` が見る — `sources/hf.ts`）。
      if (bytes.byteLength > MAX_MANIFEST_BYTES) {
        throw sizeViolation(bytes.byteLength, origin.integrity);
      }
      // MUST: parse の throw をそのまま外へ出す（ローカルには evict すべきキャッシュが無いので、
      // 壊れた manifest は毎回同じ ManifestFormatError で落ちるのが正しい）。
      parse(bytes);
    },

    readFile: async (ref, { signal, sizeViolation }) => {
      const abort = signal === undefined ? {} : { signal };
      const bytes = await adapter.readFile(ref.path, abort);
      // 検証は size 厳密一致だけ（sha256 は信頼する）。onProgress は 1 度も呼ばない —
      // 受信の途中という状態が無いので、共通層が complete の 1 点で閉じる。
      if (bytes.byteLength !== ref.size) {
        throw sizeViolation(bytes.byteLength, origin.integrity);
      }
      return bytes;
    },

    // 相 1（prefetchFile）は持たない — 上のモジュール doc を参照。

    // ⑧区間読みは、アダプターが位置読みを持つときだけ生やす（`prefetchFile` と同じ流儀で
    // optional 能力 — 持たないアダプターでは口ごと現れず、消費側は全量読みへ倒す）。
    ...(readRange === undefined ? {} : {
      // 開く側の中断確認は共通層（`fetch.ts` の `openAsset`）の作法なのでここには置かない。
      // 読みごとの中断は `AssetRangeReader.read` の signal がアダプターへ透過する。
      openFile: (ref: FileRef) => Promise.resolve(localRangeReader(readRange, ref)),
    }),

    originFor: (repo, revision) => {
      const mapped = settings.crossRepo;
      if (mapped !== undefined && Object.hasOwn(mapped, repo)) {
        return pinTarget(mapped[repo], revision, options);
      }
      const { fallback } = settings;
      if (fallback === undefined) throw missingCrossRepo(settings.label, repo, revision);
      // 委譲先は「他の repo も提供できる取得元」なので、自分の座標ではなく宣言された座標へ
      // 寄せてから開く（mapping と違い、fallback は特定の repo に紐付いていない）。
      return pinTarget(fallback, revision, options).originFor(repo, revision);
    },

    // ⑥在庫は常に「渡された全部がある」— 相 1 を持たないのと同じ理屈で、直接読める取得元では
    // 「後の読みが安く済む状態」が最初から満たされている。実体の欠損は読む時に落ちる
    // （在庫の照会でディレクトリを舐めると、資産数ぶんの I/O を照会のたびに払うことになる）。
    inventory: (refs) => Promise.resolve(new Set(refs.map(fileRefKey))),
    // ⑦削除は持たない — ディレクトリの中身は取得物ではなく利用者の資産。
  };
};

/**
 * 手元のディレクトリを取得元にする。`loadManifest(localDirectory(adapter), …)` /
 * `fromPretrained(localDirectory(adapter))` の形で、HF 取得元と同じ面に乗る。
 *
 * ```ts ignore
 * import { localDirectory, loadManifest } from "@karume/hub";
 * import { denoDirectory } from "@karume/hub/deno";
 *
 * const loaded = await loadManifest(denoDirectory("./models/karume-gemma4"));
 * ```
 *
 * 越境参照の相対サブディレクトリは**別の取得元を作って渡す**（`crossRepo: { "owner/name":
 * denoDirectory("./models/other") }`）— 取得元 1 つ = ディレクトリ 1 つに保つと、mapping の
 * 値がそのままリモートの取得元にも差し替わる。
 */
export const localDirectory = (
  adapter: DirectoryAdapter,
  options: LocalDirectoryOptions = {},
): DistributionSource => {
  const settings = { ...options, label: options.label ?? "(ローカル)" };
  const driver: SourceDriver = {
    origin: { label: `ディレクトリ ${settings.label}`, integrity: "local" },
    // 世代の概念が無いので解決も要求も起きない（暗黙 main の警告も出ない — 固定する対象が無い）。
    resolveGeneration: () => Promise.resolve(""),
    pin: (_generation, callOptions) => pinnedLocalSource(adapter, settings, callOptions),
  };
  return new DistributionSource(driver);
};
