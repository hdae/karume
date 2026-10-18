/**
 * Cutting out the added vocabulary (special tokens).
 *
 * It is **family-independent** (`AddedVocabulary` of `tokenizers` sits before normalization
 * regardless of the model kind), so it lives in `src/text/`. The contract that cut-out pieces emit
 * their id directly, without normalization or pre-tokenization, does not depend on the model kind
 * either.
 */

/** One cut-out part (the added vocabulary entry itself when `added`). */
type AddedTokenChunk = { text: string; added: boolean };

/**
 * Matching index of the added vocabulary (built once when the tokenizer is constructed, reused for
 * every encode).
 */
export type AddedTokenMatcher = {
  /**
   * Cuts out the added vocabulary leftmost-longest (the rule of {@link createAddedTokenMatcher}).
   */
  split(text: string): AddedTokenChunk[];
};

/** A node of the index (transitions for one UTF-16 unit, and whether a candidate ends here). */
type TrieNode = { readonly next: Map<string, TrieNode>; terminal: boolean };

/**
 * Builds the matching index of the added vocabulary. The rule is leftmost-longest (the same as the
 * source of truth's `AddedVocabulary` = AhoCorasick's LeftmostLongest): at the same position the
 * longer one wins; at different positions the left one wins.
 *
 * MUST: sorting longest-first and taking the first prefix match is not enough — take **the longest
 * of all candidates at each position**. The index is a per-character trie, walked from each
 * position while recording the longest terminal (the walk is capped at the longest candidate
 * length). Grouping only by the first character would, for assets whose candidates all share a
 * prefix (the `<` of `<unused0>` etc.), scan every candidate at each `<` in the text — characters ×
 * vocabulary size (EmbeddingGemma has 6,415 entries).
 *
 * NOTE: the matching unit is the UTF-16 unit (`text[i]`). Candidates are walked in the same unit,
 * so characters outside the BMP match in two steps over the surrogate pair (the result is the same
 * as per code point).
 */
export const createAddedTokenMatcher = (added: readonly string[]): AddedTokenMatcher => {
  const root: TrieNode = { next: new Map(), terminal: false };
  for (const token of added) {
    if (token === "") {
      // The empty string matches at every position and the scan never advances (an infinite loop).
      // Silently skipping it would change the cut-out rule itself, so it fails.
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

  /** Length of the longest candidate starting at position `start` (0 if none). */
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
      // Start of the current non-added span (the span is cut out with one slice).
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
 * One-off cutting out (builds the index and throws it away). A tokenizer that encodes repeatedly
 * calls {@link createAddedTokenMatcher} once at construction and reuses it.
 */
export const splitAddedTokens = (text: string, added: readonly string[]): AddedTokenChunk[] =>
  createAddedTokenMatcher(added).split(text);
