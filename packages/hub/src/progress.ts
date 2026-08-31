/**
 * 取得の進捗（ADR 0038 §5「進捗総量は content-length ではなく manifest の `size` 合計」）。
 *
 * 集計は取得元の能力ではなく**共通層の作法**なので、全ての面（全量 / 相 1 / 逐次）が同じ 1 つの
 * 実装を共有する。面ごとに書くと `loaded` の積み方だけが片方で直る形の食い違いを再生産する。
 */

import { type FileRef, fileRefKey } from "./manifest.ts";

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
 * MUST: 受信実績の記録は `onProgress` の有無に関わらず行う — 通知しないだけで、面を跨いで
 * 引き継ぐ表（逐次面の相 1 → 相 2）は常に正しくなければならない。
 */
export const createProgressEmitter = (
  targets: ReadonlyMap<string, FileRef>,
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
    onProgress({ phase, path: ref.path, loaded: sum, total, fileLoaded, fileTotal });
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
