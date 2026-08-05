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
 * Viterbi の同点処理まで正本に合わせる: 同じ位置で終わるノードは「開始位置の昇順」に積まれ、
 * 同点なら先に積まれた方（= より長い断片）が勝つ。ここがずれると、稀にだが別の分割になり
 * id 列が静かに変わる。
 */

import { type CodeRanges, inCodeRanges, toCodePoints } from "./code-ranges.ts";
import { splitAddedTokens } from "./qwen2-tokenizer.ts";
import { normalizeSpm, type SpmTables } from "./spm-normalizer.ts";

/** Metaspace の置換文字（U+2581）。 */
const METASPACE = "▁";

/** 未知ノードのスコアは「語彙全体の最小スコア − 10」（正本の `K_UNK_PENALTY`）。 */
const UNK_PENALTY = 10;

export type T5VocabEntry = { readonly id: number; readonly score: number };

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

/** Viterbi 経路のノード（開始位置とそこから始まるノードのスロット）。 */
type Slot = { readonly pos: number; readonly slot: number };

export class T5Tokenizer {
  readonly #assets: T5Assets;
  readonly #added: string[];

  constructor(assets: T5Assets) {
    this.#assets = assets;
    this.#added = [...assets.addedTokens.keys()];
  }

  /** 1 断片の最適分割（正本の `Lattice::viterbi` をコードポイント位置で写す）。 */
  #viterbi(cps: readonly number[]): { start: number; length: number }[] {
    const n = cps.length;
    const unkScore = this.#assets.minScore - UNK_PENALTY;
    // begin[pos] = そこから始まるノードの長さ（挿入順 = 長さ昇順）
    const begin: number[][] = [];
    const scores: number[][] = [];
    for (let pos = 0; pos < n; pos++) {
      const lengths: number[] = [];
      const nodeScores: number[] = [];
      let text = "";
      const limit = Math.min(this.#assets.maxTokenLength, n - pos);
      for (let length = 1; length <= limit; length++) {
        text += String.fromCodePoint(cps[pos + length - 1]);
        const entry = this.#assets.vocab.get(text);
        if (entry !== undefined) {
          lengths.push(length);
          nodeScores.push(entry.score);
        }
      }
      if (lengths[0] !== 1) {
        lengths.unshift(1);
        nodeScores.unshift(unkScore);
      }
      begin.push(lengths);
      scores.push(nodeScores);
    }
    // end[e] = そこで終わるノード。begin 昇順に積む（= 同じ e なら長い方が先）。
    const end: Slot[][] = Array.from({ length: n + 1 }, () => []);
    for (let pos = 0; pos < n; pos++) {
      for (const [slot, length] of begin[pos].entries()) end[pos + length].push({ pos, slot });
    }
    const best: number[][] = begin.map((lengths) => Array.from(lengths, () => 0));
    const prev: (Slot | undefined)[][] = begin.map((lengths) =>
      Array.from(lengths, (): Slot | undefined => undefined)
    );

    /** pos で終わる最良ノード。pos=0 は bos（スコア 0）。到達不能なら undefined。 */
    const bestLeft = (pos: number): { score: number; node: Slot | undefined } | undefined => {
      if (pos === 0) return { score: 0, node: undefined };
      let found: { score: number; node: Slot } | undefined;
      for (const node of end[pos]) {
        const candidate = best[node.pos][node.slot];
        // MUST: 厳密な `>` — 同点は先に積まれた方（より長い断片）が残る。
        if (found === undefined || candidate > found.score) found = { score: candidate, node };
      }
      return found;
    };

    for (let pos = 0; pos < n; pos++) {
      const left = bestLeft(pos);
      if (left === undefined) return [];
      for (let slot = 0; slot < begin[pos].length; slot++) {
        best[pos][slot] = left.score + scores[pos][slot];
        prev[pos][slot] = left.node;
      }
    }
    const tail = bestLeft(n);
    if (tail === undefined) return [];
    const path: { start: number; length: number }[] = [];
    let node = tail.node;
    while (node !== undefined) {
      path.push({ start: node.pos, length: begin[node.pos][node.slot] });
      node = prev[node.pos][node.slot];
    }
    return path.reverse();
  }

  /**
   * 1 断片を id 列へ。連続する未知ノードは 1 トークンに融合される（`fuse_unk`）。
   *
   * NOTE: `tokenizer.json` の `fuse_unk` は `null` だが、`tokenizers` の Unigram は未指定でも
   * 融合する（Rust 側の既定）。融合しないと日本語プロンプトで unk が 1 文字ずつ並び、正本と
   * の突合が `japanese` ケースで落ちる。byte_fallback は false なので未知はここに来る。
   */
  #tokenize(piece: string): number[] {
    const cps = toCodePoints(piece);
    const ids: number[] = [];
    let pendingUnk = false;
    for (const span of this.#viterbi(cps)) {
      let text = "";
      for (let k = span.start; k < span.start + span.length; k++) {
        text += String.fromCodePoint(cps[k]);
      }
      const entry = this.#assets.vocab.get(text);
      if (entry === undefined) {
        pendingUnk = true;
        continue;
      }
      if (pendingUnk) {
        ids.push(this.#assets.unkId);
        pendingUnk = false;
      }
      ids.push(entry.id);
    }
    if (pendingUnk) ids.push(this.#assets.unkId);
    return ids;
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
