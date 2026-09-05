/**
 * ローカル取得元アダプター — 「配布形がもう手元にある」場合の取得元。純 Web 標準（このファイル
 * は特定ランタイムの API を一切参照しない）。実体の読みは {@link DirectoryAdapter} 1 面へ委ね、
 * そこへ Deno / OPFS / IndexedDB / File System Access の picker を差し込む。
 *
 * HF 取得元（`hf.ts`）との違いは能力そのもの:
 *
 * - **世代を持たない** — 手元のディレクトリは「今そこにある内容」しかなく、可変 ref も commit
 *   SHA も無い。暗黙 `main` の警告も出ない（固定するものが無い）。
 * - **相 1（prefetch）を持たない** — CacheStorage を通らないので「温める」に意味がない。
 *   逐次面は相 2 だけで同じ RAM ピーク（O(最大 shard)）を満たす（`source.ts` ④）。バイト列の
 *   複製が 1 つも増えないのがローカル取得元の最大の利点で、キャッシュへ写すのは害でしかない。
 * - **検証は size 厳密一致のみ**（sha256 は信頼する）。手元の
 *   ファイルは配布元と同じ「取得物」ではなく利用者の資産で、毎起動の全量ハッシュ（数 GiB）に
 *   見合う脅威が無い。size は読み終えた時点でタダで分かるので門として残す（途中で切れた
 *   コピー・別 quant の取り違えはここで落ちる）。
 * - **越境は明示 mapping だけ** — 隣接する同名ディレクトリを推測しない（`originFor`）。
 *
 * MUST NOT: ここでエラーを組み立てない（診断の文脈を持つのは共通層 — `context.ts`）。例外は
 * 「呼び手の設定が足りない」ことを告げる素の `Error`（未 mapping の越境）で、共通層がそれを
 * 取得失敗として `cause` に残したまま包む。
 */

import { MANIFEST_FILENAME, MAX_MANIFEST_BYTES } from "../manifest.ts";
import type { LoadManifestOptions } from "../session.ts";
import {
  DistributionSource,
  driverOf,
  type PinnedSource,
  type SourceDriver,
  type SourceOrigin,
} from "../source.ts";

/**
 * ディレクトリ 1 つぶんの読み口。**この 1 メソッドが取得元の全て**で、実装は
 * `@karume/hub/deno` の `denoDirectory`（`Deno.readFile`）のほか、ブラウザでは OPFS
 * （`FileSystemDirectoryHandle` → `File.arrayBuffer()`）・IndexedDB・File System Access の
 * picker が同じ形で乗る。
 *
 * MUST: `readFile` が返す `Uint8Array` は **buffer 全体を占める**（tight view）— 共通層はここで
 * 受けたバイト列をそのまま `openModel` へ渡すので、余白のある view を返すと辻褄合わせの `slice`
 * で RAM ピークが倍増する（共通層の tight view 検査がその場で落とす）。
 * MUST: 欠損は fail loudly（`undefined` や空バイト列を返さない）。エラーには**実体のパス**を
 * 載せる — 共通層が付けられるのは manifest 上の相対 path までで、「どのディレクトリの下を
 * 探したか」を知っているのはアダプターだけ。
 * MUST: `signal` を透過する（大きい shard の読みは中断できなければならない）。
 *
 * `readFileInto`（任意）は逐次面の**器の使い回し**のための面: 実体を `target` の先頭へ読み、
 * **ファイルの実長**を返す。`target` に収まらないファイルは読まずに（または途中で止めて）実長だけ
 * を返す — size 違反を名乗るのは共通層（`sizeViolation`）で、アダプターは判定しない。持たない
 * アダプターは `readFile` だけで従来どおり動く（器は確保されない）。
 */
export type DirectoryAdapter = {
  readonly readFile: (
    path: string,
    options: { readonly signal?: AbortSignal },
  ) => Promise<Uint8Array<ArrayBuffer>>;
  readonly readFileInto?: (
    path: string,
    target: Uint8Array<ArrayBuffer>,
    options: { readonly signal?: AbortSignal },
  ) => Promise<number>;
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
   * mapping に無い越境参照の委譲先（例: リモートの取得元）。**明示した場合だけ**降格する
   * （暗黙のリモート降格は禁止 — オフライン前提の配布が黙って network へ出る）。
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

const pinnedLocalSource = (
  adapter: DirectoryAdapter,
  settings: LocalDirectoryOptions & { readonly label: string },
  options: LoadManifestOptions,
): PinnedSource => {
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
      // NOTE: 全量を読んでから門を見る（アダプターは逐次面を持たない）。HF 側の上限が「受信を
      // 途中で止める」防波堤なのに対し、こちらは手元の実体に対する形式検査 — 送出側の悪意を
      // 想定する門ではないので、読み切ってから落として構わない。
      if (bytes.byteLength > MAX_MANIFEST_BYTES) {
        throw sizeViolation(bytes.byteLength, "body", origin.integrity);
      }
      // MUST: parse の throw をそのまま外へ出す（ローカルには evict すべきキャッシュが無いので、
      // 壊れた manifest は毎回同じ ManifestFormatError で落ちるのが正しい）。
      parse(bytes);
    },

    readFile: async (ref, { signal, sizeViolation, into }) => {
      const abort = signal === undefined ? {} : { signal };
      // 器を貸されていて、アダプターが器へ読めるなら、その経路（ホスト RAM に載る shard は常に
      // 1 本）。どちらか欠ければ従来の全量読み（新しい buffer）。
      if (into !== undefined && adapter.readFileInto !== undefined) {
        const vessel = into();
        if (vessel.byteLength < ref.size) {
          throw new Error(
            `hub: ${ref.path} の器（${vessel.byteLength} バイト）が宣言 size ${ref.size} より小さい`,
          );
        }
        const actual = await adapter.readFileInto(ref.path, vessel.subarray(0, ref.size), abort);
        if (actual !== ref.size) throw sizeViolation(actual, "body", origin.integrity);
        return new Uint8Array(vessel.buffer, 0, ref.size);
      }
      const bytes = await adapter.readFile(ref.path, abort);
      // 検証は size 厳密一致だけ（sha256 は信頼する）。onProgress は 1 度も呼ばない —
      // 受信の途中という状態が無いので、共通層が complete の 1 点で閉じる。
      if (bytes.byteLength !== ref.size) {
        throw sizeViolation(bytes.byteLength, "body", origin.integrity);
      }
      return bytes;
    },

    // 相 1（prefetchFile）は持たない — 上のモジュール doc を参照。

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
