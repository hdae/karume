/**
 * 追加語彙（特殊トークン）の切り出し。
 *
 * **ファミリ非依存**（`tokenizers` の `AddedVocabulary` はモデル種別に依らず正規化の手前に
 * 挟まる）なので `src/text/` に置く。切り出された断片は正規化も pre-token 化も通さず id を
 * 直接出す、という契約もモデル種別に依らない。
 */

/** 切り出しの 1 片（`added` なら追加語彙そのもの）。 */
type AddedTokenChunk = { text: string; added: boolean };

/** 追加語彙の照合索引（tokenizer の構築時に 1 回だけ組み、encode のたびに使い回す）。 */
export type AddedTokenMatcher = {
  /** 追加語彙を leftmost-longest で切り出す（{@link createAddedTokenMatcher} の規則）。 */
  split(text: string): AddedTokenChunk[];
};

/** 索引の節（UTF-16 単位 1 つぶんの遷移と、ここで終わる候補の有無）。 */
type TrieNode = { readonly next: Map<string, TrieNode>; terminal: boolean };

/**
 * 追加語彙の照合索引を組む。規則は leftmost-longest（正本の `AddedVocabulary` = AhoCorasick の
 * LeftmostLongest と同じ）: 位置が同じなら長い方、位置が違えば左が勝つ。
 *
 * MUST: 長い順に並べて先頭一致で採るだけでは足りない — **各位置で全候補中の最長**を採る。
 * 索引は文字単位の trie で、各位置から辿りながら最長の終端を記録する（辿る長さは最長の候補長で
 * 頭打ち）。先頭 1 文字で束ねるだけだと、全候補が共通接頭辞（`<unused0>` 等の `<`）を持つ
 * 資産で本文の `<` ごとに全候補を走査し、文字数 × 語彙数になる（EmbeddingGemma は 6,415 本）。
 *
 * NOTE: 照合の単位は UTF-16 単位（`text[i]`）。候補も同じ単位で辿るので、BMP 外の文字は
 * サロゲート対の 2 段で一致する（コードポイント単位と結果は変わらない）。
 */
export const createAddedTokenMatcher = (added: readonly string[]): AddedTokenMatcher => {
  const root: TrieNode = { next: new Map(), terminal: false };
  for (const token of added) {
    if (token === "") {
      // 空文字は全位置で一致して走査が進まない（無限ループ）。黙って読み飛ばすと切り出し規則
      // そのものが別物になるので落とす。
      throw new Error("追加語彙に空文字がある — 切り出しが進まない");
    }
    let node = root;
    for (let unit = 0; unit < token.length; unit += 1) {
      let child = node.next.get(token[unit]);
      if (child === undefined) {
        child = { next: new Map(), terminal: false };
        node.next.set(token[unit], child);
      }
      node = child;
    }
    node.terminal = true;
  }

  /** 位置 `start` から始まる最長の候補の長さ（無ければ 0）。 */
  const longestAt = (text: string, start: number): number => {
    let node = root;
    let longest = 0;
    for (let cursor = start; cursor < text.length; cursor += 1) {
      const child = node.next.get(text[cursor]);
      if (child === undefined) break;
      node = child;
      if (node.terminal) longest = cursor - start + 1;
    }
    return longest;
  };

  return {
    split(text: string): AddedTokenChunk[] {
      const out: AddedTokenChunk[] = [];
      // 追加語彙でない区間の始点（区間は 1 回の slice で切り出す）。
      let plainStart = 0;
      let i = 0;
      while (i < text.length) {
        const length = longestAt(text, i);
        if (length === 0) {
          i += 1;
          continue;
        }
        if (plainStart < i) out.push({ text: text.slice(plainStart, i), added: false });
        out.push({ text: text.slice(i, i + length), added: true });
        i += length;
        plainStart = i;
      }
      if (plainStart < text.length) out.push({ text: text.slice(plainStart), added: false });
      return out;
    },
  };
};

/**
 * 1 回きりの切り出し（索引を組んで捨てる）。encode を繰り返す tokenizer は
 * {@link createAddedTokenMatcher} を構築時に 1 回だけ呼んで使い回す。
 */
export const splitAddedTokens = (
  text: string,
  added: readonly string[],
): AddedTokenChunk[] => createAddedTokenMatcher(added).split(text);
