/**
 * 生テキスト → **詰めた** token 列（グラフ入力 `[1,T]` の中身）。
 *
 * 上流は `max_text_len` / `max_caption_len` へ右 pad した固定長で backbone を呼ぶが、B=1 では
 * 「詰めきった長さで呼ぶ」ことと数学的に同値（双方向マスクで pad を落とす ⇔ pad を最初から
 * 渡さない）。同値性は exporter の `_static_scheme_evidence` が毎 emit 実測している。
 *
 * MUST: 正規化は 1 回だけ（`text/tokenizer.ts` の doc — 正本の `tokenizer.json` は normalizer が
 * null なので、トークナイザ側でもう一度掛かることは無い）。
 */

import { normalizeText } from "../text/normalize.ts";
import type { IrodoriTokenizer } from "../text/tokenizer.ts";

/**
 * `normalize_text` → `strip` → BOS 前置 + 本文を `maxLength-1` で切る、までを 1 本にする。
 *
 * MUST: 正規化後に空なら fail loudly。上流（`irodori_pipeline.py` の `_packed_ids`）も同じ位置で
 * 落ちる — BOS だけの列を通すと、**無音でも無いのに内容の無い発話**が生成される。
 *
 * NOTE: `encodePadded` の pad を落として使うのは、BOS 前置と切り詰めの綴りを 2 箇所に
 * 持たないため（マスクは必ず先頭からの連続した prefix なので、走査で実長が取れる）。
 */
export const packIds = (
  tokenizer: IrodoriTokenizer,
  text: string,
  maxLength: number,
  where: string,
): Int32Array<ArrayBuffer> => {
  const normalized = normalizeText(text).trim();
  if (normalized.length === 0) {
    throw new Error(`irodori: ${where} が正規化後に空（合成する本文が無い）`);
  }
  const padded = tokenizer.encodePadded(normalized, maxLength);
  let used = 0;
  while (used < padded.mask.length && padded.mask[used] === 1) used += 1;
  return padded.ids.slice(0, used);
};
