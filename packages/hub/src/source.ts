/**
 * 取得元の内部契約（`DistributionSource`）。**公開面ではない** — 公開面は `mod.ts` のまま
 * （ADR 0008 の薄さ）で、ここは共通層（`fetch.ts`）とアダプター（`sources/`）の間の境界。
 *
 * 現行の取得経路を畳むと、取得元が実際に答えているのは 5 つの質問しかない:
 *
 * 1. 可変 ref → 不変な世代識別子（{@link DistributionSource.resolveGeneration}）
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

import type { FileRef } from "./manifest.ts";

/** バイト数の門を破ったと気づいた場所（診断の文言に載る）。 */
export type SizeViolationSite = "content-length" | "body";

/**
 * バイト数が宣言と食い違ったときに投げるエラーの組み立て。**取得元は組み立てない** —
 * 診断の文脈（repo / 世代識別子 / 利用可能ラベル）を持つのは共通層なので、取得元は
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
  /** ②`karume.json` を読み、`parse` を通す。バイト列そのものは共通層へ渡さない。 */
  readonly readManifest: (options: ManifestReadOptions) => Promise<void>;
  /**
   * ③1 本の全量バイト。sha256 / size の検証は取得元が持つ（共通層は buffer 全体を占めるか
   * だけを見る — `fetch.ts` の tight view 検査）。
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
   */
  readonly originFor: (repo: string, revision: string) => PinnedSource;
};

/** 世代を解決する前の取得元（①）。 */
export type DistributionSource = {
  /**
   * ①可変 ref → 不変な世代識別子。**セッション唯一の解決点**で、返り値が以降の取得を固定する。
   * 世代という概念を持たない取得元は固定値を返してよい。
   */
  readonly resolveGeneration: (options: { readonly signal?: AbortSignal }) => Promise<string>;
  /** 解決済みの世代へ固定した取得元を開く。 */
  readonly pin: (generation: string) => PinnedSource;
};

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
