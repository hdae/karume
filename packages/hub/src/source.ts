/**
 * 取得元の内部契約（{@link SourceDriver} / {@link PinnedSource}）と、公開面が持ち回る**不透明
 * ハンドル**（{@link DistributionSource}）。契約そのものは公開面ではない — 公開面は `mod.ts` の
 * まま（ADR 0008 の薄さ）で、ここは共通層（`fetch.ts`）とアダプター（`sources/`）の間の境界。
 *
 * 現行の取得経路を畳むと、取得元が実際に答えているのは 5 つの質問しかない:
 *
 * 1. 可変 ref → 不変な世代識別子（{@link SourceDriver.resolveGeneration}）
 * 2. `karume.json` 1 本の全量バイト（{@link PinnedSource.readManifest} — 上限は
 *    {@link ../manifest.ts MAX_MANIFEST_BYTES}・sha256 の期待値は**持てない**）
 * 3. ある `FileRef` の全量バイト（{@link PinnedSource.readFile} — sha256 / size の期待値つき）
 * 4. ある `FileRef` を「RAM に載せずに、後で 3 が安く済む状態にする」
 *    （{@link PinnedSource.prefetchFile} = 逐次面の相 1。**optional 能力**であって、持たない
 *    取得元が正当 — HTTP + 永続キャッシュ固有の最適化で、直接読める取得元には意味がない）
 * 5. 越境 (repo, revision) → 別の取得元（{@link PinnedSource.originFor}）
 *
 * 取得経路の外側に、キャッシュ在庫の管理面（`inventory.ts` の照会と削除）がもう 2 つ乗る
 * （⑥{@link PinnedSource.inventory} / ⑦{@link PinnedSource.evict}）。どちらも**optional 能力**で、
 * 「取ってきたものを溜めている」取得元だけが本当に答えられる質問。
 *
 * さらに ⑧ある `FileRef` の**区間だけ**を読む（{@link PinnedSource.openFile} =
 * {@link AssetRangeReader}）。これも**optional 能力**で、「全量を読まずに数 KB を引く」ことが
 * 成立する取得元だけが名乗る（③との違いは、返すのが宣言 size 全部ではなく `[offset, offset+length)`
 * だけという点だけ）。
 *
 * MUST: 進捗・並行度（in-flight バイト予算）・中断の透過・tight view の検査・エラーの文脈は
 * 取得元固有の能力ではなく**共通層の作法**として `fetch.ts` に残す。取得元へ降ろすと、
 * 取得元が増えるたびに同じ不変条件を書き直すことになる。
 */

import type { IntegritySource } from "./errors.ts";
import { crossRefOf, type FileRef } from "./manifest.ts";
import type { LoadManifestOptions } from "./session.ts";

/**
 * 診断が名乗る取得元の身元。**エラーを組み立てるのは共通層**（`context.ts`）なので、取得元が
 * 持つのは「自分は何者か」を表すこの値だけ。
 *
 * MUST: 持っていない身元を合成しない — ローカル取得元が repo / commit SHA を名乗ると、実在
 * しないリポを指す診断（HF へ探しに行けと言う案内）を生む。持たない欄は省いてよい設計で、
 * 代わりに {@link label} が**必ず**実際に取りに行った先を名乗る。
 */
export type SourceOrigin = {
  /**
   * 診断の文言に載る 1 行の名乗り（HF: `repo owner/name @ <commit SHA>` /
   * ローカル: `ディレクトリ <ラベル>`）。取得元ごとに語彙が違ってよい唯一の欄。
   */
  readonly label: string;
  /** 完全性検証が破れたときの失敗元（{@link ../errors.ts IntegrityError} の `source`）。 */
  readonly integrity: IntegritySource;
  /** HF 語彙の構造化欄。**持たない取得元は省く**（{@link ../errors.ts HubFetchError} も同様）。 */
  readonly repo?: string;
  /** 解決済み世代識別子（commit SHA）。世代の概念を持たない取得元は省く。 */
  readonly revisionSha?: string;
};

/**
 * バイト数が宣言と食い違ったときに投げるエラーの組み立て。**取得元は組み立てない** —
 * 診断の文脈（取得元の名乗り・利用可能ラベル）を持つのは共通層なので、取得元は
 * 「いくつだったか」と**自分の失敗元**（{@link SourceOrigin.integrity}）だけを渡す
 * （組み立て点は `context.ts` の 1 箇所）。
 *
 * MUST: `integrity` は呼ぶ取得元自身のもの（`origin.integrity` — 定数を書かない）。越境参照は
 * セッションと違う取得元から来る（ローカルセッション + リモート越境は正当な構成）ので、共通層は
 * 宣言から失敗元を推定できない — 推定すると「network から来たバイト列が `"local"`（＝再試行は
 * 無駄）を名乗る」形の嘘が 1 欄だけ混じる。
 */
export type SizeViolation = (actual: number, integrity: IntegritySource) => Error;

/** 資産 1 本の読み（{@link PinnedSource.readFile} / {@link PinnedSource.prefetchFile}）の作法。 */
export type FileReadOptions = {
  readonly signal?: AbortSignal;
  /**
   * 受信途中の累積バイト。**直接読める取得元では 1 度も呼ばれなくてよい**（キャッシュヒットと
   * 同じ扱い — 共通層は `complete` の 1 点だけで進捗を閉じられる）。
   */
  readonly onProgress: (loaded: number) => void;
  readonly sizeViolation: SizeViolation;
  /**
   * 器の貸し出し（逐次面だけが渡す）。呼ぶと **`ref.size` 以上の長さの buffer** が返り、取得元は
   * そこへ実体を先頭から読み、器の prefix view（byteOffset 0 / byteLength = `ref.size`）を返して
   * よい。器は shard ごとに使い回されるので、ホスト RAM に同時に載る shard は常に 1 本になる
   * （ADR 0070 追記 — 係数 1 化）。
   *
   * 使えない取得元（取得層が自前で buffer を確保する外部実装の取得元）は**呼ばずに**従来どおり
   * tight view を返す — 呼ばなければ器は確保されない（遅延確保）。組み込みの 2 取得元
   * 〈ディレクトリ / HF〉はどちらも使う。MUST: 呼んだら器へ読む（呼んで別の buffer を返すと、
   * 器 1 本ぶんの RAM が無駄に居座る）。
   */
  readonly into?: () => Uint8Array<ArrayBuffer>;
};

/**
 * 資産 1 本の**区間読み口**（{@link PinnedSource.openFile} が返す ⑧の能力）。全量読み
 * （{@link PinnedSource.readFile}）と違い、宣言 size のうち欲しい `[offset, offset + length)` だけを
 * 返す — 数百 MiB の shard から数 KB の行だけを引く消費側（層ごとの埋め込み表の decode）のための面。
 *
 * {@link cost} が**費用の型**を名乗るのは、消費側が「行読みに切り替えてよい行数の上限」をそれで
 * 変えるから: `"scan"` の取得元では 1 行が offset に比例した読み飛ばしを伴うので、全量 1 回の方が
 * 安くなる行数がある。数値（ms / バイト毎秒）ではなく型で名乗るのは、実測値が環境（ブラウザ /
 * ランタイム / ディスク）で 2 桁動く一方、**どちらの型か**は取得元の実装で決まって動かないため。
 *
 * 読み口は**読みごとに開き直してよい**（fd や handle を保持しない — `denoDirectory` は read の
 * たびに `Deno.open` する）。実測 9,100 B × 2,000 回で 46 µs/read に対し fd 保持は 23 µs/read で、
 * 差は decode 1 token の壁（数十 ms）に対して無視できる。したがって**閉じる面は持たない** —
 * 開きっぱなしの資源が無いので、呼び手に解放の責務が生えない。
 */
export type AssetRangeReader = {
  /**
   * `"seek"` = offset に依らず小さい（ファイルの位置読み・ブラウザの遅延 Blob の `slice`）/
   * `"scan"` = offset に比例する（本文ストリームの読み飛ばし）。
   */
  readonly cost: "seek" | "scan";
  /**
   * `[offset, offset + length)` を返す。
   *
   * MUST: `length` ちょうどを返す（短く返さない）。要求が実体の外へ出る・実体が宣言より短くて
   * 埋まらない、いずれも fail loudly — 短い戻りを黙って通すと、消費側は「0 埋めされた行」を
   * 正常な値として読む。
   * MUST: buffer 全体を占める view（tight view）を返す — 消費側は返ったバイト列をそのまま
   * TypedArray として読む。
   */
  readonly read: (
    offset: number,
    length: number,
    options?: { readonly signal?: AbortSignal },
  ) => Promise<Uint8Array<ArrayBuffer>>;
};

/** manifest 1 本の読みの作法。 */
export type ManifestReadOptions = {
  readonly signal?: AbortSignal;
  readonly sizeViolation: SizeViolation;
  /**
   * バイト列 → `Manifest` の唯一の変換点。
   *
   * MUST: 取得元は**自分の完全性検証の内側**でこれを呼ぶ（外で呼ぶと、壊れたエントリが
   * evict されず毎回同じ `ManifestFormatError` を返し続ける）。この関数の throw は
   * 「取得物が壊れている」の意味で、取得元はそれを破損として扱ってよい。
   * MUST NOT: 中断の確認をここへ混ぜない（健全なエントリの evict を招く）。
   */
  readonly parse: (bytes: Uint8Array) => void;
};

/**
 * 世代を固定した取得元（②〜⑤）。世代の固定は**セッションに 1 回**で、以降の取得は全て
 * この 1 つの世代に留まる（可変 ref のまま複数回解決すると manifest と重みが別の世代から来る）。
 */
export type PinnedSource = {
  /** 診断が名乗る身元（この世代・この座標のもの）。 */
  readonly origin: SourceOrigin;
  /** ②`karume.json` を読み、`parse` を通す。バイト列そのものは共通層へ渡さない。 */
  readonly readManifest: (options: ManifestReadOptions) => Promise<void>;
  /**
   * ③1 本の全量バイト。検証は取得元が持つ（共通層は buffer 全体を占めるかだけを見る —
   * `fetch.ts` の tight view 検査）。**何を検証できるかは取得元によって違う** — HF は
   * sha256 まで照合し、ローカル取得元は size 厳密一致だけを見る（sha256 は信頼する）。
   */
  readonly readFile: (ref: FileRef, options: FileReadOptions) => Promise<Uint8Array>;
  /**
   * ④相 1（optional 能力）— RAM に載せずに、後続の {@link readFile} が安く済む状態にする。
   * **持たない取得元が正当**で、その場合は逐次面が相 2（直接逐次読み）だけで同じ RAM 目標を
   * 満たす（ADR 0070 決定 2 の読み替え）。
   */
  readonly prefetchFile?: (ref: FileRef, options: FileReadOptions) => Promise<void>;
  /**
   * ⑧区間読み口を開く（**optional 能力**）— {@link prefetchFile} と同じ流儀で、**持たない取得元が
   * 正当**（HTTP + 永続キャッシュのように、区間だけを安く取り出す口をまだ持たない取得元がある）。
   * 共通層はその場合 `openAsset` から `undefined` を返し、消費側は全量読みへ倒す。
   *
   * MUST NOT: ここでバイト列を取りに行かない — 開くのは読み口だけで、実際の読みは
   * {@link AssetRangeReader.read} が呼ばれたときに起きる。
   */
  readonly openFile?: (
    ref: FileRef,
    options: { readonly signal?: AbortSignal },
  ) => Promise<AssetRangeReader>;
  /**
   * ⑤越境参照（`FileRef` の `repo` / `revision` — ADR 0038 §7）の取得元。参照先は世代識別子
   * 固定が必須なので、越境先で世代の解決は起きない。
   *
   * 越境先を**決められない取得元は throw してよい**（ローカル取得元は明示 mapping しか持たず、
   * 未 mapping は fail loudly — 隣接同名ディレクトリの推測はしない）。共通層はこの throw を
   * 取得失敗として文脈付きで包む。
   */
  readonly originFor: (repo: string, revision: string) => PinnedSource;
  /**
   * ⑥在庫の照会（**optional 能力**）— 渡した参照のうち、**network に出ずに読める**ものの
   * {@link ../manifest.ts fileRefKey} の集合を返す。
   *
   * MUST: 渡る参照は全て**この取得元の座標のもの**（越境参照は共通層が `originFor` で origin
   * ごとに分けてから渡す）。取得元側で越境を捌き直すと、同じ参照が 2 通りのキーで数えられる。
   */
  readonly inventory?: (refs: readonly FileRef[]) => Promise<ReadonlySet<string>>;
  /**
   * ⑦在庫の削除（**optional 能力**）— 渡した参照の在庫を消し、**実際に消えたもの**を返す。
   *
   * 持たない取得元が正当（手元のディレクトリの中身は取得物ではなく利用者の資産で、hub が
   * 消してよいものが 1 つも無い）。共通層はその場合に fail loud で断る。
   */
  readonly evict?: (refs: readonly FileRef[]) => Promise<readonly FileRef[]>;
};

/**
 * 世代を解決する前の取得元（①）。公開面の {@link DistributionSource} が包んでいる実装本体。
 *
 * 面ごとの作法（`fetch` / `caches` / `headers` / `onCacheError`）は**取得元の生成時ではなく
 * {@link pin} の呼び出しごとに**渡す。取得元は「どこから取るか」だけを持ち、「どんな作法で
 * 取るか」は面のオプションから来る — こうしないと `loadManifest` に渡した `fetch` が以後の
 * `fetchAssets` にも黙って効き続け、面ごとの差し替えが効かなくなる。
 */
export type SourceDriver = {
  /** 世代解決前の名乗り（HF なら `repo owner/name @ main` のように**要求した** ref を含む）。 */
  readonly origin: SourceOrigin;
  /**
   * ①可変 ref → 不変な世代識別子。**セッション唯一の解決点**で、返り値が以降の取得を固定する。
   * 世代という概念を持たない取得元は固定値（空文字）を返してよい。
   */
  readonly resolveGeneration: (options: LoadManifestOptions) => Promise<string>;
  /** 解決済みの世代へ固定した取得元を、その呼び出しの作法で開く。 */
  readonly pin: (generation: string, options: LoadManifestOptions) => PinnedSource;
};

// MUST: クラス定義より前に置く — `static` ブロックはクラス評価時に走るので、後ろに置くと
// TDZ で ReferenceError になる（import 時に落ちる）。
let readDriver: (source: DistributionSource) => SourceDriver;

/**
 * 取得元の**公開ハンドル**。`loadManifest(source, …)` / `fromPretrained(source)` が受け取る値で、
 * 中身（{@link SourceDriver}）は hub の内部にしかない。
 *
 * MUST: 公開メンバを生やさない — 取得の実装詳細（世代の解決・pin・越境）が公開面に漏れると、
 * 取得元が増えるたびに公開面の互換を気にすることになる。判別も**同一性**（`instanceof`）で
 * 行う: ブランド欄を生やすと利用者が偽造でき、構造判別（`"repo" in value`）にすると
 * `HubRepoRef` の綴り間違いが黙って取得元として通る。
 */
export class DistributionSource {
  readonly #driver: SourceDriver;

  /**
   * MUST: 取得元アダプター（`sources/`）の factory だけが呼ぶ。`SourceDriver` は `mod.ts` が
   * 輸出しないので、利用者はこの引数を型として綴れない。
   */
  constructor(driver: SourceDriver) {
    this.#driver = driver;
  }

  // `#driver` を読めるのはクラス本体の中だけなので、モジュール内の 1 関数へ束縛して外へ出す
  // （static メンバにすると公開面に現れてしまう — このクラスは「メンバを持たない」ことが仕様）。
  static {
    readDriver = (source) => source.#driver;
  }
}

/** 公開ハンドルから実装を取り出す（hub の内部だけが呼ぶ）。 */
export const driverOf = (source: DistributionSource): SourceDriver => readDriver(source);

/**
 * 値が取得元ハンドルかを見る（`ref | source` の union を捌く**唯一の判別点**）。
 *
 * `mod.ts` は {@link DistributionSource} を**型としてしか**輸出しない（生やすメンバが無い以上、
 * クラスの値を出しても利用者にできるのは壊れた取得元の生成だけ）ので、hub の外から同一性判別を
 * 綴る手段はこの述語しかない。上の MUST のとおり、判別を利用者側に書かせると構造判別
 * （`"repo" in value`）へ流れ、`HubRepoRef` の綴り間違いが黙って取得元として通る。
 */
export const isDistributionSource = (value: unknown): value is DistributionSource =>
  value instanceof DistributionSource;

/**
 * 1 本の `FileRef` を取りに行く取得元を決める。**越境参照（`repo` + `revision` が両方ある ref）
 * だけ**がセッションの取得元ではなく宣言された (repo, revision) から来る。
 *
 * MUST: 分岐はこの 1 箇所だけに置く — 面ごとに書くと、片方だけ越境を素通ししたときに
 * 「別リポの同名 path を自リポから取る」形の取り違えが黙って成立する。
 */
export const sourceForRef = (source: PinnedSource, ref: FileRef): PinnedSource => {
  const cross = crossRefOf(ref);
  return cross === undefined ? source : source.originFor(cross.repo, cross.revision);
};
