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
 * MUST: 進捗・並行度（in-flight バイト予算）・中断の透過・tight view の検査・エラーの文脈は
 * 取得元固有の能力ではなく**共通層の作法**として `fetch.ts` に残す。取得元へ降ろすと、
 * 取得元が増えるたびに同じ不変条件を書き直すことになる。
 */

import type { IntegritySource } from "./errors.ts";
import type { FileRef } from "./manifest.ts";
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

/** バイト数の門を破ったと気づいた場所（診断の文言に載る）。 */
export type SizeViolationSite = "content-length" | "body";

/**
 * バイト数が宣言と食い違ったときに投げるエラーの組み立て。**取得元は組み立てない** —
 * 診断の文脈（取得元の名乗り・利用可能ラベル）を持つのは共通層なので、取得元は
 * 「どこで、いくつだったか」だけを渡す（組み立て点は `context.ts` の 1 箇所）。
 */
export type SizeViolation = (actual: number, where: SizeViolationSite) => Error;

/** 資産 1 本の読み（{@link PinnedSource.readFile} / {@link PinnedSource.prefetchFile}）の作法。 */
export type FileReadOptions = {
  readonly signal?: AbortSignal;
  /**
   * 受信途中の累積バイト。**直接読める取得元では 1 度も呼ばれなくてよい**（キャッシュヒットと
   * 同じ扱い — 共通層は `complete` の 1 点だけで進捗を閉じられる）。
   */
  readonly onProgress: (loaded: number) => void;
  readonly sizeViolation: SizeViolation;
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
   * ⑤越境参照（`FileRef` の `repo` / `revision` — ADR 0038 §7）の取得元。参照先は世代識別子
   * 固定が必須なので、越境先で世代の解決は起きない。
   *
   * 越境先を**決められない取得元は throw してよい**（ローカル取得元は明示 mapping しか持たず、
   * 未 mapping は fail loudly — 隣接同名ディレクトリの推測はしない）。共通層はこの throw を
   * 取得失敗として文脈付きで包む。
   */
  readonly originFor: (repo: string, revision: string) => PinnedSource;
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
export const sourceForRef = (source: PinnedSource, ref: FileRef): PinnedSource =>
  ref.repo === undefined || ref.revision === undefined
    ? source
    : source.originFor(ref.repo, ref.revision);
