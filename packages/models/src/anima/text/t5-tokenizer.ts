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
 * Unigram 本体（Viterbi・同点処理・`fuse_unk`）はファミリ非依存なので `src/text/unigram.ts`
 * が持つ。ここが持つのは Anima 固有の前段（AddedVocabulary・正規化・Metaspace）と後段
 * （切り詰めと `</s>`）だけ。
 */

import { type UnigramModel, unigramTokenize, type UnigramVocabEntry } from "../../text/unigram.ts";
import { type CodeRanges, inCodeRanges, toCodePoints } from "./code-ranges.ts";
import { splitAddedTokens } from "./qwen2-tokenizer.ts";
import { normalizeSpm, type SpmTables } from "./spm-normalizer.ts";

/** Metaspace の置換文字（U+2581）。 */
const METASPACE = "▁";

export type T5VocabEntry = UnigramVocabEntry;

export type T5Assets = {
  readonly vocab: ReadonlyMap<string, T5VocabEntry>;
  /** 語彙**全体**の最小スコア。部分集合を渡す場合も全体の値を渡す（未知ノードの重み）。 */
  readonly minScore: number;
  /** 語彙**全体**の最長トークンのコードポイント数（前方一致の探索幅）。 */
  readonly maxTokenLength: number;
  readonly unkId: number;
  readonly eosId: number;
  readonly addedTokens: ReadonlyMap<string, number>;
  /** WhitespaceSplit の空白集合（Qwen2 側の `\s` と同一であることは emit 時に検査済み）。 */
  readonly space: CodeRanges;
  readonly normalizer: SpmTables;
  readonly maxLength: number;
};

/**
 * WhitespaceSplit → Metaspace。空白は捨て、各断片の先頭に ▁ を付ける。
 *
 * MUST: `split=true` は MergedWithNext — 区切りの ▁ は**次の**断片の先頭に付く。
 * 「前の断片の末尾」にすると分割が変わる。
 *
 * NOTE（実測）: 現在の正規化表は U+2581 を U+0020 へ写すので、`encode` の経路では正規化を
 * 生き延びた ▁ は現れず、この分割ループは踏まれない。それでも正本どおりに書いてあるのは、
 * 正規化表が**資産**（上流の tokenizer.json 由来）で、写像が変われば到達しうるため。挙動は
 * テストがこの関数を直接呼んで固定している。
 */
export const t5PreTokenize = (text: string, space: CodeRanges): string[] => {
  const out: string[] = [];
  for (const word of splitOnSpace(text, space)) {
    let piece = word.replaceAll(" ", METASPACE);
    if (!piece.startsWith(METASPACE)) piece = METASPACE + piece;
    let start = 0;
    for (let idx = 1; idx < piece.length; idx++) {
      if (piece[idx] === METASPACE) {
        out.push(piece.slice(start, idx));
        start = idx;
      }
    }
    out.push(piece.slice(start));
  }
  return out.filter((piece) => piece !== "");
};

const splitOnSpace = (text: string, space: CodeRanges): string[] => {
  const out: string[] = [];
  let buffer = "";
  for (const ch of text) {
    if (inCodeRanges(space, ch.codePointAt(0) as number)) {
      if (buffer !== "") {
        out.push(buffer);
        buffer = "";
      }
    } else {
      buffer += ch;
    }
  }
  if (buffer !== "") out.push(buffer);
  return out;
};

export class T5Tokenizer {
  readonly #assets: T5Assets;
  readonly #added: string[];

  constructor(assets: T5Assets) {
    this.#assets = assets;
    this.#added = [...assets.addedTokens.keys()];
  }

  /**
   * 1 断片を id 列へ（Unigram 本体は共有モジュールへ委譲）。連続する未知ノードは 1 トークンに
   * 融合される（`fuse_unk`）。
   *
   * NOTE: `tokenizer.json` の `fuse_unk` は `null` だが、`tokenizers` の Unigram は未指定でも
   * 融合する（Rust 側の既定）。融合しないと日本語プロンプトで unk が 1 文字ずつ並び、正本と
   * の突合が `japanese` ケースで落ちる。byte_fallback は false なので（= `byteBaseId` を
   * 渡さないので）未知は unk 1 個になる。
   */
  #tokenize(piece: string): number[] {
    // T5Assets は UnigramModel の面をそのまま満たす（byteBaseId を持たない = byte_fallback なし）。
    return unigramTokenize(this.#assets satisfies UnigramModel, toCodePoints(piece));
  }

  /**
   * `tokenizer([text], padding="longest", max_length=512, truncation=True)` と同じ id 列。
   *
   * MUST: 切り詰めは `</s>` の分を空けてから（正本の truncation は post_processor の**前**）。
   * `slice(0, maxLength)` の後に足すと 513 個になる。結果として id 列は常に `</s>` で終わり、
   * 長さは必ず 1 以上。
   */
  encode(text: string): number[] {
    const ids: number[] = [];
    for (const chunk of splitAddedTokens(text, this.#added)) {
      if (chunk.added) {
        ids.push(this.#assets.addedTokens.get(chunk.text) as number);
        continue;
      }
      const normalized = normalizeSpm(this.#assets.normalizer, chunk.text);
      for (const piece of t5PreTokenize(normalized, this.#assets.space)) {
        ids.push(...this.#tokenize(piece));
      }
    }
    return [...ids.slice(0, this.#assets.maxLength - 1), this.#assets.eosId];
  }
}
