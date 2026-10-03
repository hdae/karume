/**
 * T5 トークナイザ（Unigram）。Anima の text_conditioner 側の入力を作る。
 *
 * 正本の経路（`t5_tokenizer/tokenizer.json`）は
 *   AddedVocabulary（103 特殊トークンを leftmost-longest で切り出す）
 *   → Precompiled 正規化（`spm-normalizer.ts`）
 *   → WhitespaceSplit → Metaspace（replacement=▁ / prepend_scheme=always / split）
 *   → Unigram（Viterbi・unk_id=2・byte_fallback なし・fuse_unk）
 *   → TemplateProcessing（末尾に `</s>`）
 *
 * 経路の本体（追加語彙・Metaspace・Unigram・`</s>`）は家族横断の `src/text/t5-tokenizer.ts` が
 * 持つ。ここが持つのは Anima 固有の 3 点だけ: Precompiled 正規化の表を資産に含むこと、
 * 語彙外は unk 1 個へ融合すること、512 を超えたら上流どおり切り詰めること。
 */

import { type T5Tables, T5UnigramTokenizer } from "../../text/t5-tokenizer.ts";
import { normalizeSpm, type SpmTables } from "./spm-normalizer.ts";

export type T5Assets = T5Tables & { readonly normalizer: SpmTables };

export class T5Tokenizer {
  readonly #inner: T5UnigramTokenizer;

  constructor(assets: T5Assets) {
    // NOTE: `tokenizer.json` の `fuse_unk` は `null` だが、`tokenizers` の Unigram は未指定でも
    // 融合する（Rust 側の既定）。融合しないと日本語プロンプトで unk が 1 文字ずつ並び、正本と
    // の突合が `japanese` ケースで落ちる。byte_fallback は false なので未知は unk 1 個になる。
    this.#inner = new T5UnigramTokenizer(assets, {
      normalize: (text) => normalizeSpm(assets.normalizer, text),
      unknown: "unk",
      overflow: "truncate",
    });
  }

  /**
   * `tokenizer([text], padding="longest", max_length=512, truncation=True)` と同じ id 列。
   * 結果は常に `</s>` で終わり、長さは必ず 1 以上（切り詰めの規則は共通層の `encode`）。
   */
  encode(text: string): number[] {
    return this.#inner.encode(text);
  }
}
