/**
 * 生成要求の門のうち、**家族をまたいで受理集合が同じ**ものの置き場。
 *
 * 家族ごとの門（解像度・スタイル名・chunk 長）は家族側に置く — ここに来るのは
 * 「anima でも sbv2 でも irodori でも同じ値だけが通る」と言い切れる検査だけである。
 * 受理集合を複数の家族が写して持つと必ず割れる（片方だけ緩む・片方だけ falsy 判定になる）ので、
 * **所有者を 1 本にする**のがこのモジュールの役目。
 *
 * MUST: モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 */

import { ModelInputError } from "./errors.ts";

/**
 * `seed` の受理集合（**非負の安全整数**）を見る門。anima / sbv2 / irodori が共有する。
 *
 * 上限が `Number.MAX_SAFE_INTEGER` なのは、3 家族とも受けた `seed` を `BigInt` へ落として
 * splitmix64 の状態にするため — 2^53 以上は倍精度の格子の外なので「渡した値」と「使われた値」が
 * 黙って食い違う。端数と負も `BigInt` で表せない。
 *
 * NOTE: 呼ぶのは**生成の入口**（各 `generate` の冒頭）であって、乱数生成器のコンストラクタ
 * だけではない。生成器を作るのは重みを GPU に載せた後なので、そこにしか検査が無いと
 * **GB 級のロードを待たされた末に**「seed が整数でない」で落ちる。入口で先に落とせば、
 * 打ち間違いは 1 ミリ秒で返る。
 */
export const assertAcceptableSeed = (seed: number): void => {
  if (!Number.isInteger(seed) || seed < 0 || seed > Number.MAX_SAFE_INTEGER) {
    throw new ModelInputError(`seed ${seed} が非負の安全整数でない`);
  }
};
