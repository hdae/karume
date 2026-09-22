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
 * 3 家族とも受けた `seed` を `BigInt` へ落として splitmix64 の状態にするので、条件の理由は
 * 3 つに分かれる:
 *
 * - **非負**は API の値域方針である。負も `BigInt` へは落ちる（`BigInt(-1) === -1n`）し、
 *   64 bit へ畳めば有効な状態になるが、`seed` の要求としては受けない。
 * - **上限 `Number.MAX_SAFE_INTEGER`** は、2^53 以上では隣接整数を区別できないため
 *   （2^53 自体は厳密に表せるが、2^53 + 1 は 2^53 へ潰れる）。ここを開けると「渡した値」と
 *   「使われた値」が黙って食い違う。
 * - **端数**は `BigInt` への変換そのものが `RangeError` を投げる（`BigInt(1.5)`）ので、
 *   入口で落とさないと乱数生成器の内側で別の顔の例外になる。
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
