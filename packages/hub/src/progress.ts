/**
 * 取得の進捗（ADR 0038 §5「進捗総量は content-length ではなく manifest の `size` 合計」）。
 *
 * 集計は取得元の能力ではなく**共通層の作法**なので、全ての面（全量面 / 温め面）が同じ 1 つの
 * 実装を共有する。面ごとに書くと `loaded` の積み方だけが片方で直る形の食い違いを再生産する。
 */

import { type FileRef, fileRefKey } from "./manifest.ts";

/**
 * ファイルの取得先の名乗り（`context.ts` の `FetchContext.originOf` の構造化欄）。進捗の
 * {@link AssetProgress.repo} / {@link AssetProgress.revision} はここから写す。
 */
type OriginIdentity = { readonly repo?: string; readonly revisionSha?: string };

/**
 * 進捗のフェーズ。`complete` は 1 ファイルの終端（bytes が確定した点）。
 *
 * MUST: 1 ファイルの phase は `downloading`* → `complete` の順にだけ進み、逆行しない
 * （`complete` はファイルごとに 1 回だけ・以降そのファイルの通知は出ない）。例外は破損キャッシュ
 * の self-heal で、取得層が拒否した後に network から取り直すためこの 1 巡が最初からやり直しに
 * なる（`complete` が終端であることは変わらない）。
 *
 * NOTE: 照合中を表す `verifying` は持たない — 資産の検証は取得元の内部（受信中のハッシュ / 記録
 * ハッシュの突合）に埋まっていて共通層からは観測できないため。観測できないフェーズを推測で
 * 名乗ると、実際には終わっている照合を「進行中」と表示する嘘になる。
 */
export type AssetPhase = "downloading" | "complete";

export type AssetProgress = {
  readonly phase: AssetPhase;
  /**
   * イベントを起こしたファイルの path（リポ内の相対 path）。
   *
   * MUST NOT: path だけをファイルの識別子にしない — 越境参照（ADR 0038 §7）や部品差し替えでは
   * 別リポの同じ path が 1 回のロードに並ぶ。ファイル別に集計するときは {@link repo} /
   * {@link revision} と組にしてキーにする。
   */
  readonly path: string;
  /**
   * ファイルを**実際に取りに行った先**のリポ（`"owner/name"`）。越境参照は宣言された越境先、
   * それ以外はセッションのリポ。repo という概念を持たない取得元（ローカルディレクトリ）の
   * ファイルでは欄ごと現れない（合成した名前を名乗らせない）。
   */
  readonly repo?: string;
  /**
   * {@link repo} の世代（解決済み commit SHA・40 桁小文字 hex）。越境参照は宣言された
   * `revision`、それ以外はセッションの解決済み SHA。世代を持たない取得元では欄ごと現れない。
   */
  readonly revision?: string;
  /** 取得済みバイトの合計（全ファイル・同一ファイル一意化後）。 */
  readonly loaded: number;
  /** manifest の `size` 合計（同一ファイル一意化後）。 */
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

/** 取得面 1 つぶんの進捗集計器。状態（受信実績）はここに閉じる。 */
export type ProgressEmitter = {
  /** 受信途中の累積バイトを記録して `downloading` を出す。 */
  readonly downloading: (ref: FileRef, loaded: number) => void;
  /** 1 ファイルの終端。 */
  readonly complete: (ref: FileRef) => void;
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
 * 取得対象の表（キーは {@link fileRefKey}）から集計器を作る。`total` はこの表の `size` 合計に
 * 固定され、以降変わらない。
 *
 * MUST: `originOf` はエラーの名乗りと**同じ関数**（`FetchContext.originOf`）を渡す — 進捗だけ
 * 別に「越境かどうか」を綴ると、診断と進捗が同じファイルに違う取得先を名乗る食い違いが出る。
 *
 * MUST: 受信実績（`received`）の記録は通知（`emit`）より前に行う — `emit` は自分のファイルぶんも
 * `received` から読むので、順序が逆だと送出値が 1 イベントぶん古くなる（downloading の初回が
 * `fileLoaded` 0 を名乗り、以降ずっと 1 つ前の値で遅れる）。`fileLoaded` と全体 `loaded` は
 * 常に同じ値を数える（`emit` が自分のキーを除いて足す）ので、食い違いはどちらの順でも出ない。
 */
export const createProgressEmitter = (
  targets: ReadonlyMap<string, FileRef>,
  originOf: (ref: FileRef) => OriginIdentity,
  onProgress?: (progress: AssetProgress) => void,
): ProgressEmitter => {
  let total = 0;
  for (const ref of targets.values()) total += ref.size;

  const received = new Map<string, number>();
  const emit = (phase: AssetPhase, ref: FileRef): void => {
    if (onProgress === undefined) return;
    const refKey = fileRefKey(ref);
    const fileTotal = fileSizeOf(targets, refKey);
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
    const origin = originOf(ref);
    onProgress({
      phase,
      path: ref.path,
      ...(origin.repo === undefined ? {} : { repo: origin.repo }),
      ...(origin.revisionSha === undefined ? {} : { revision: origin.revisionSha }),
      loaded: sum,
      total,
      fileLoaded,
      fileTotal,
    });
  };

  return {
    downloading: (ref, loaded) => {
      received.set(fileRefKey(ref), loaded);
      emit("downloading", ref);
    },
    complete: (ref) => {
      // MUST: `received` にも size を書く — 書かずに complete だけ出すと、キャッシュヒット
      // （downloading が 1 度も出ない）だったファイルが後続イベントの `loaded` 合計から抜け、
      // 全体の進捗が巻き戻って見える。
      received.set(fileRefKey(ref), ref.size);
      emit("complete", ref);
    },
  };
};
