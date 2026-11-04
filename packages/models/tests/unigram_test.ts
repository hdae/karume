// Behaviour check of the shared Unigram layer (src/text/unigram.ts).
//
// Independent of real assets — a small synthetic vocabulary exercises "the tie rule of the lattice
// search", "unknown fusion" and "the byte_fallback expansion" directly. Comparison with the real
// assets (whether the id sequence matches Python) belongs to the family's parity test
// (anima_tokenizer_test.ts); this file keeps **the rules themselves** in a form that can fail.

import { assertEquals } from "@std/assert";
import { beforeAll, describe, it, test } from "vite-plus/test";
import { toCodePoints } from "../src/text/code-points.ts";
import { type UnigramModel, unigramTokenize } from "../src/text/unigram.ts";

const UNK_ID = 2;
/** The id of byte_fallback's `<0x00>`. A value that does not collide with the real asset, to tell the expansion apart. */
const BYTE_BASE = 100;

/** Builds a model from `[token, id, score]` rows. minScore / maxTokenLength are derived from the whole vocabulary. */
const buildModel = (
  vocab: readonly (readonly [string, number, number])[],
  options: { readonly byteBaseId?: number } = {},
): UnigramModel => ({
  vocab: new Map(vocab.map(([token, id, score]) => [token, { id, score }])),
  minScore: vocab.length === 0 ? 0 : Math.min(...vocab.map(([, , score]) => score)),
  maxTokenLength: Math.max(1, ...vocab.map(([token]) => toCodePoints(token).length)),
  unkId: UNK_ID,
  ...options,
});

const tokenize = (model: UnigramModel, text: string): number[] =>
  unigramTokenize(model, toCodePoints(text));

describe("lattice search: the path is decided by the sum of scores", () => {
  it("on a tie the longer piece wins (the comparison is a strict `>`)", () => {
    // a(-1) + b(-1) and ab(-2) tie exactly. The reference (tokenizers' Unigram) pushes the nodes
    // ending at the same position in ascending order of start position and, on a tie, keeps the one
    // pushed first = the longer one. With `>=` the result silently changes to [4, 5].
    const model = buildModel([
      ["a", 4, -1],
      ["b", 5, -1],
      ["ab", 6, -2],
    ]);
    assertEquals(tokenize(model, "ab"), [6]);
  });

  it("without a tie the higher-scoring path wins (not always the longest)", () => {
    // In the same shape with only ab lowered to -3, the two short pieces win on the sum. This catches
    // the rule above degenerating into "always return the longest".
    const model = buildModel([
      ["a", 4, -1],
      ["b", 5, -1],
      ["ab", 6, -3],
    ]);
    assertEquals(tokenize(model, "ab"), [4, 5]);
  });
});

describe("unknowns: without byte_fallback", () => {
  let model: UnigramModel;
  beforeAll(() => {
    model = buildModel([["a", 4, -1]]);
  });

  it("consecutive unknowns are fused into one unk (fuse_unk)", () => {
    // Without fusion the unknowns line up one character each, giving [4, 2, 2, 4].
    assertEquals(tokenize(model, "axya"), [4, UNK_ID, 4]);
  });

  it("unknowns separated by a known piece become separate unks (only consecutive ones fuse)", () => {
    assertEquals(tokenize(model, "xay"), [UNK_ID, 4, UNK_ID]);
  });

  it("a text that is entirely unknown is still one unk", () => {
    assertEquals(tokenize(model, "xyz"), [UNK_ID]);
  });
});

describe("unknowns: with byte_fallback", () => {
  let model: UnigramModel;
  beforeAll(() => {
    model = buildModel([["a", 4, -1]], { byteBaseId: BYTE_BASE });
  });

  it("a 3-byte character expands into the ids of its 3 UTF-8 bytes", () => {
    // U+3042 "あ" = E3 81 82 → 100 + each byte.
    assertEquals(tokenize(model, "あ"), [327, 229, 230]);
  });

  it("a 4-byte character (outside the BMP) gives 4 ids by the same rule", () => {
    // U+1F600 "😀" = F0 9F 98 80. Counting by surrogate pair instead of by code point gives 3 ids or
    // invalid bytes.
    assertEquals(tokenize(model, "😀"), [340, 259, 252, 228]);
  });

  it("a fused unknown piece becomes bytes in full (it does not fall back to unk)", () => {
    assertEquals(tokenize(model, "aあ😀a"), [4, 327, 229, 230, 340, 259, 252, 228, 4]);
  });

  it("a piece in the vocabulary is not expanded into bytes", () => {
    const withKana = buildModel(
      [
        ["a", 4, -1],
        ["あ", 7, -1],
      ],
      { byteBaseId: BYTE_BASE },
    );
    assertEquals(tokenize(withKana, "aあ"), [4, 7]);
  });
});

describe("degenerate cases: empty input and empty vocabulary", () => {
  it("the empty string gives an empty id sequence (no unk either)", () => {
    assertEquals(tokenize(buildModel([["a", 4, -1]]), ""), []);
    assertEquals(tokenize(buildModel([], { byteBaseId: BYTE_BASE }), ""), []);
  });

  it("with an empty vocabulary the whole text becomes one unk", () => {
    assertEquals(tokenize(buildModel([]), "abc"), [UNK_ID]);
  });

  it("with an empty vocabulary + byte_fallback the whole text becomes bytes", () => {
    // "abc" = 61 62 63 → 100 + each byte.
    assertEquals(tokenize(buildModel([], { byteBaseId: BYTE_BASE }), "abc"), [197, 198, 199]);
  });
});

test("long unknown piece: a byte sequence past the argument-spread limit still gives an id sequence", () => {
  // "あ" × 50,000 = one unknown piece of 150,000 bytes. Passing the expansion as variadic arguments
  // (`push(...ids)`) hits V8's argument limit (about 130K) with a bare RangeError, failing in a way
  // unrelated to the input's length.
  const model = buildModel([["a", 4, -1]], { byteBaseId: BYTE_BASE });
  const ids = tokenize(model, "a" + "あ".repeat(50_000) + "a");
  assertEquals(ids.length, 150_002);
  assertEquals(ids.slice(0, 4), [4, 327, 229, 230]);
  assertEquals(ids.slice(-4), [327, 229, 230, 4]);
});
