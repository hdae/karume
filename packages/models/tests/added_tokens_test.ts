// The matching rule and the complexity of the added-token split (`src/text/added-tokens.ts`). Needs
// neither a GPU nor real assets.
//
// It pins two things:
//
// - **Matching rule**: leftmost-longest (at the same position the longer one, at different
//   positions the left one). The expected side is written out in this file in the naive form "at
//   each position, try every candidate and take the longest", and compared on random inputs with a
//   synthetic vocabulary full of shared prefixes (the check that changing how the index is held
//   does not break the rule).
// - **Complexity**: even with a vocabulary whose candidates all start with the same character
//   (`<`), matching stays within input length × the longest candidate length. Grouping only by the
//   first character would scan every candidate at each `<` of the text.

import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { createAddedTokenMatcher, splitAddedTokens } from "../src/text/added-tokens.ts";

type Chunk = { text: string; added: boolean };

/**
 * The expected side: at each position, try every candidate and take the longest (a different form
 * from the implementation).
 */
const referenceSplit = (text: string, added: readonly string[]): Chunk[] => {
  const out: Chunk[] = [];
  let buffer = "";
  let i = 0;
  while (i < text.length) {
    let hit = "";
    for (const token of added) {
      if (token.length > hit.length && text.startsWith(token, i)) hit = token;
    }
    if (hit === "") {
      buffer += text[i];
      i += 1;
      continue;
    }
    if (buffer !== "") out.push({ text: buffer, added: false });
    buffer = "";
    out.push({ text: hit, added: true });
    i += hit.length;
  }
  if (buffer !== "") out.push({ text: buffer, added: false });
  return out;
};

/** A deterministic pseudo-random number (an integer in `[0, bound)`). */
const randomIndex = (() => {
  let state = 20261010;
  return (bound: number): number => {
    // A 32-bit LCG (`Math.imul` — a plain multiplication exceeds 2^53 and loses the low bits). The
    // low bits have a short period, so the upper 16 bits are used.
    state = (Math.imul(state, 1103515245) + 12345) >>> 0;
    return (state >>> 16) % bound;
  };
})();

test("added-token split: leftmost-longest agrees with brute force on a vocabulary full of shared prefixes", () => {
  // Nested prefixes (`<a` ⊂ `<ab` ⊂ `<abc>`), candidates that match partway and then fail, and
  // characters outside the BMP.
  const added = ["<a", "<ab", "<abc>", "<b>", "<", "ab>", "<😀>", "c<a"];
  const alphabet = ["<", "a", "b", "c", ">", "😀", "x"];
  const matcher = createAddedTokenMatcher(added);
  let addedChunks = 0;
  for (let sample = 0; sample < 5_000; sample += 1) {
    const length = randomIndex(16);
    const text = Array.from({ length }, () => alphabet[randomIndex(alphabet.length)]).join("");
    const expected = referenceSplit(text, added);
    addedChunks += expected.filter((chunk) => chunk.added).length;
    assertEquals(matcher.split(text), expected, JSON.stringify(text));
    assertEquals(splitAddedTokens(text, added), expected, JSON.stringify(text));
  }
  // The random input really exercises the splitting side (if everything passed through, the
  // comparison would be vacuous).
  assert(addedChunks >= 1_000, `only ${addedChunks} chunks were split out`);
});

test("added-token split: matching is linear in the input length even when every candidate starts with the same character", () => {
  // The shape of an EmbeddingGemma-sized added vocabulary (6,400 entries) that all start with `<`.
  // Grouping only by the first character would run startsWith over 6,400 entries at each `<` of the
  // text (seconds for `<` × 65,536).
  const added = Array.from({ length: 6_400 }, (_, index) => `<unused${index}>`);
  const matcher = createAddedTokenMatcher(added);
  const text = "<".repeat(65_536);
  const started = performance.now();
  const chunks = matcher.split(text);
  const elapsed = performance.now() - started;
  assertEquals(chunks, [{ text, added: false }]);
  assert(elapsed < 500, `matching \`<\` × ${text.length} took ${elapsed.toFixed(0)} ms`);
});
