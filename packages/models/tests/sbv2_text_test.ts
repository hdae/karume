// Behaviour tests of the SBV2 text layer (the pure functions of `src/sbv2/text/`).
//
// When this layer breaks, **the shapes stay valid and only the audio degrades** — front still runs
// with a phoneme ID off by one, and a waveform still comes out with the BERT features handed to the
// wrong phonemes. Numeric correctness here is beyond what the golden E2E can guarantee, so these
// tests use fault injection to show that the invariants really catch the breakage.
//
// Only the last test depends on assets (outputs/misc/sbv2-demo/); everything else runs without
// them.

import { isNotFound, readTextFile, statSync } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assert, assertEquals, assertFalse, assertThrows } from "@std/assert";
import { InputError } from "../src/sbv2/errors.ts";
import { buildSbv2ModelInput } from "../src/sbv2/text/model-input.ts";
import {
  addBlankWord2ph,
  intersperse,
  type JpExtraRules,
  parseJpExtraRules,
  phonesToIds,
  phonesTonesToModelIds,
} from "../src/sbv2/text/symbols.ts";
import { DebertaTokenizer } from "../src/sbv2/text/tokenizer.ts";
import { buildBaseWord2ph, distributePhone } from "../src/sbv2/text/word2ph.ts";
import { bertHiddenOutput, tileBertToPhoneLevel } from "../src/sbv2/text/bert-tile.ts";
import { toBertText } from "../src/sbv2/text/bert-text.ts";

/**
 * JP-Extra rules for testing. **Not the real values** (the `symbols.json` asset is authoritative
 * for those — the last test reads it). This is a minimal setup that only exercises the shape of the
 * rules; the tone base and the language ID are deliberately non-zero (with 0, an implementation
 * that does not add them could not be told apart).
 */
const TEST_RULES_JSON = {
  symbols: ["_", "a", "k", "o", "N", ".", ","],
  pad: "_",
  punctuations: [".", ","],
  toneStart: 6,
  languageId: 1,
  numTones: 12,
  numLanguages: 3,
  addBlank: true,
  blankId: 0,
  samplingRate: 44100,
  hopLength: 512,
  bertHiddenFromEnd: 3,
  bertRelPos: { positionBuckets: 256, maxPosition: 512 },
  defaults: { sdpRatio: 0.2, noiseScale: 0.6, noiseScaleW: 0.8, lengthScale: 1 },
};

const testRules = (): JpExtraRules => parseJpExtraRules(TEST_RULES_JSON, "test rules");

/**
 * A minimal tokenizer with one code point = one token (empty clean tables = nothing is dropped).
 */
const tinyTokenizer = (tokens: readonly string[]): DebertaTokenizer =>
  DebertaTokenizer.fromVocabText(
    ["[PAD]", "[CLS]", "[SEP]", "[UNK]", ...tokens].join("\n"),
    { removed: [], spaced: [] },
    { clsId: 1, sepId: 2, unkId: 3 },
  );

test("intersperse: inserts blank between elements and at both ends, giving 2n+1", () => {
  assertEquals(intersperse([5, 7, 9], 0), [0, 5, 0, 7, 0, 9, 0]);
  // Even an empty sequence keeps the one element for "both ends" (as the reference
  // commons.intersperse does).
  assertEquals(intersperse([], 0), [0]);
  // A non-zero blank goes into the same positions (rejects an implementation that hardcodes 0).
  assertEquals(intersperse([5], 4), [4, 5, 4]);
});

test("addBlankWord2ph: doubles every element and adds 1 to the first only", () => {
  const base = [1, 2, 3, 1];
  assertEquals(addBlankWord2ph(base), [3, 4, 6, 2]);
  // Invariant: sum(word2ph) equals the phoneme length 2n+1 after add_blank.
  const phoneCount = base.reduce((sum, count) => sum + count, 0);
  const total = addBlankWord2ph(base).reduce((sum, count) => sum + count, 0);
  assertEquals(total, 2 * phoneCount + 1);
});

test("addBlankWord2ph: fault injection — dropping the leading +1 breaks the match with the phoneme length", () => {
  // Shows the test above is not vacuous. An implementation that drops `[0] += 1` gets a sum one too
  // small and fails downstream (tileBertToPhoneLevel) with "differs from the phoneme count".
  const base = [1, 2, 3, 1];
  const broken = base.map((count) => count * 2); // an implementation that forgot the +1
  const phoneCount = base.reduce((sum, count) => sum + count, 0);
  assert(broken.reduce((sum, count) => sum + count, 0) !== 2 * phoneCount + 1);
});

test("addBlankWord2ph: rejects an empty base (the sentinels at both ends would be missing)", () => {
  assertThrows(() => addBlankWord2ph([]), Error, "is empty");
});

test("phonesToIds: returns symbol-table indices and rejects unknown phonemes", () => {
  const rules = testRules();
  assertEquals(phonesToIds(rules, ["_", "k", "o", "."]), [0, 2, 3, 5]);
  // A failure reachable through the caller's utterance alone, so it is an input error (400-like) —
  // a plain Error would turn it into a 500.
  assertThrows(() => phonesToIds(rules, ["ky"]), InputError, "which is not in the symbol table");
});

test("phonesTonesToModelIds: the tone base offset, the language ID fill and add_blank all apply together", () => {
  const rules = testRules();
  const ids = phonesTonesToModelIds(rules, ["_", "k", "a", "_"], [0, 1, 1, 0]);
  assertEquals(ids.phoneIds, [0, 0, 0, 2, 0, 1, 0, 0, 0]);
  // tone is +6 (blank positions stay blankId — nothing is added).
  assertEquals(ids.toneIds, [0, 6, 0, 7, 0, 7, 0, 6, 0]);
  // MUST: even in JP-Extra, language is not all 0 (real phoneme positions carry languageId).
  assertEquals(ids.languageIds, [0, 1, 0, 1, 0, 1, 0, 1, 0]);
});

test("phonesTonesToModelIds: rejects an out-of-range tone and a length mismatch", () => {
  const rules = testRules();
  assertThrows(() => phonesTonesToModelIds(rules, ["_"], [0, 0]), Error, "differ in length");
  // tone=7 for a 12-tone model gives 6+7=13, out of range. Do not let it silently read another row.
  assertThrows(() => phonesTonesToModelIds(rules, ["_"], [7]), Error, "is outside");
});

test("parseJpExtraRules: does not silently use a broken asset", () => {
  assertThrows(() => parseJpExtraRules({ ...TEST_RULES_JSON, symbols: [] }, "x"), Error, "empty");
  assertThrows(
    () => parseJpExtraRules({ ...TEST_RULES_JSON, symbols: ["_", "_"] }, "x"),
    Error,
    "has duplicates",
  );
  assertThrows(
    () => parseJpExtraRules({ ...TEST_RULES_JSON, addBlank: false }, "x"),
    Error,
    "add_blank",
  );
  assertThrows(() => parseJpExtraRules({ ...TEST_RULES_JSON, pad: "?" }, "x"), Error, "pad");
  assertThrows(
    () => parseJpExtraRules({ ...TEST_RULES_JSON, toneStart: 1.5 }, "x"),
    Error,
    "not an integer",
  );
});

test("parseJpExtraRules: blankId is within all three tables (phoneme, tone and language)", () => {
  // The same blankId goes into the blank positions of all three ID sequences, so outside any one
  // table it becomes an out-of-range embedding gather (7 symbols, blankId 5 and numLanguages 3 give
  // languageIds [5,1,5]).
  assertThrows(
    () => parseJpExtraRules({ ...TEST_RULES_JSON, blankId: 5 }, "x"),
    Error,
    "x.blankId",
  );
  assertThrows(
    () => parseJpExtraRules({ ...TEST_RULES_JSON, blankId: 4, numTones: 4 }, "x"),
    Error,
    "x.blankId",
  );
  // Accepted when it fits all three tables (a non-zero blank is valid under the contract).
  assertEquals(parseJpExtraRules({ ...TEST_RULES_JSON, blankId: 2 }, "x").blankId, 2);
});

test("parseJpExtraRules: bertRelPos accepts only pairs for which the log-bucket formula holds", () => {
  // (2, 2) makes both the log denominator and the coefficient 0, producing NaN that turns into
  // index 0 on conversion to i32 (rejected where the asset enters).
  for (const bertRelPos of [
    { positionBuckets: 2, maxPosition: 2 },
    {
      positionBuckets: 4,
      maxPosition: 3,
    },
  ]) {
    assertThrows(
      () => parseJpExtraRules({ ...TEST_RULES_JSON, bertRelPos }, "x"),
      Error,
      "x.bertRelPos",
      JSON.stringify(bertRelPos),
    );
  }
  assertEquals(parseJpExtraRules(TEST_RULES_JSON, "x").bertRelPos, {
    positionBuckets: 256,
    maxPosition: 512,
  });
});

test("parseJpExtraRules: checks the range of the runtime constants at construction (being an integer is not enough)", () => {
  // The deciding one is samplingRate — the only field that **goes straight into the public result**
  // as `sampleRate` of `Pipeline.generate`, so a negative value or 0 raises no exception anywhere
  // and yields "a broken WAV that looks normal". The others would only fail loudly later, but their
  // checks are gathered here too.
  for (const key of [
    "samplingRate",
    "hopLength",
    "bertHiddenFromEnd",
    "numTones",
    "numLanguages",
  ]) {
    for (const bad of [0, -1, 1.5]) {
      assertThrows(
        () => parseJpExtraRules({ ...TEST_RULES_JSON, [key]: bad }, "x"),
        Error,
        `x.${key}: not a positive safe integer (${bad})`,
      );
    }
  }
  // languageId points at a row of the language embedding, so it is checked to be below numLanguages
  // (0 is valid).
  assertThrows(
    () => parseJpExtraRules({ ...TEST_RULES_JSON, languageId: TEST_RULES_JSON.numLanguages }, "x"),
    Error,
    "x.languageId: outside the language embedding",
  );
  assertThrows(
    () => parseJpExtraRules({ ...TEST_RULES_JSON, languageId: -1 }, "x"),
    Error,
    "x.languageId: outside the language embedding",
  );
  assertEquals(parseJpExtraRules({ ...TEST_RULES_JSON, languageId: 0 }, "x").languageId, 0);
});

test("distributePhone: distributes phonemes evenly over characters (the remainder from the left)", () => {
  assertEquals(distributePhone(5, 2), [3, 2]);
  assertEquals(distributePhone(1, 3), [1, 0, 0]);
  assertEquals(distributePhone(0, 2), [0, 0]);
  assertEquals(distributePhone(6, 3), [2, 2, 2]);
  assertThrows(() => distributePhone(3, 0), Error, "is not an integer >= 1");
});

/**
 * The expected side, written as the upstream `__distribute_phone` iteration as is (compared with
 * the closed-form implementation).
 */
const iterativeDistribute = (phoneCount: number, wordLength: number): number[] => {
  const perWord = new Array<number>(wordLength).fill(0);
  for (let k = 0; k < phoneCount; k += 1) {
    let minIndex = 0;
    for (let i = 1; i < wordLength; i += 1) {
      if (perWord[i] < perWord[minIndex]) minIndex = i;
    }
    perWord[minIndex] += 1;
  }
  return perWord;
};

test("distributePhone: matches the upstream iteration for every combination (phonemes 0..39 × characters 1..14)", () => {
  for (let phones = 0; phones < 40; phones += 1) {
    for (let chars = 1; chars < 15; chars += 1) {
      assertEquals(
        distributePhone(phones, chars),
        iterativeDistribute(phones, chars),
        `phones=${phones} chars=${chars}`,
      );
    }
  }
});

test("distributePhone: distributes in time proportional to the character count even for one huge word", () => {
  // A literal copy of the iteration (phonemes × characters) would make 4×10^10 comparisons for
  // 200,000 × 200,000 and never return.
  const size = 200_000;
  const started = performance.now();
  const perWord = distributePhone(size + 3, size);
  const elapsed = performance.now() - started;
  assertEquals(perWord.length, size);
  assertEquals(perWord.slice(0, 4), [2, 2, 2, 1]);
  assertEquals(
    perWord.reduce((sum, count) => sum + count, 0),
    size + 3,
  );
  assert(elapsed < 500, `distributing over ${size} characters took ${elapsed.toFixed(0)} ms`);
});

test("buildBaseWord2ph: lays out the per-word distributions with a sentinel 1 at both ends", () => {
  const tokenizer = tinyTokenizer(["こ", "ん", "に", "ち", "は", "、"]);
  const words = [
    { surface: "こんにちは", phones: ["k", "o", "N", "n", "i", "ch", "i", "w", "a"] },
    { surface: "、", phones: [","] },
  ];
  // 9 + 1 = 10 phonemes; with the PAD at both ends the given_phone length is 12.
  const word2ph = buildBaseWord2ph(words, tokenizer, 12);
  assertEquals(word2ph, [1, 2, 2, 2, 2, 1, 1, 1]);
  assertEquals(word2ph.length, tokenizer.encode("こんにちは、").length);
});

test("buildBaseWord2ph: fault injection — fails when the given_phone length is off by one", () => {
  // Without this invariant, the phoneme sequence and word2ph would reach front shifted
  // independently, and the BERT features would be handed out rotated by one phoneme (audio comes
  // out, but degraded).
  const tokenizer = tinyTokenizer(["こ", "ん"]);
  const words = [{ surface: "こん", phones: ["k", "o", "N"] }];
  assertEquals(buildBaseWord2ph(words, tokenizer, 5), [1, 2, 1, 1]);
  assertThrows(() => buildBaseWord2ph(words, tokenizer, 6), Error, "sum(word2ph)");
});

test("buildBaseWord2ph: a surface that normalises to 0 tokens fails as an input error", () => {
  // A whitespace-only surface is fixed by the caller fixing words = InputError (400-like).
  const tokenizer = tinyTokenizer(["あ"]);
  assertThrows(
    () => buildBaseWord2ph([{ surface: " ", phones: ["a"] }], tokenizer, 3),
    InputError,
    "normalised to 0 tokens",
  );
  // Counterpart: a sum(word2ph) mismatch is a broken internal invariant, so it stays a plain Error
  // (a path errors.ts explicitly puts on the 500 side — making both 400 would erase the
  // classification).
  const sumError = assertThrows(
    () => buildBaseWord2ph([{ surface: "あ", phones: ["a"] }], tokenizer, 9),
    Error,
    "sum(word2ph)",
  );
  assertFalse(sumError instanceof InputError);
});

test("buildSbv2ModelInput: a mismatch between the input_ids length and the word2ph length fails as an input error", () => {
  // A path reached only by an utterance whose symbol-word surface was rewritten to a token count
  // different from its normal form. The caller fixes it by restoring words to the analysis, so it
  // is on the 400 side.
  const rules = testRules();
  const tokenizer = tinyTokenizer([".", "あ"]);
  const utterance = {
    leadingPunctuations: ["."],
    moras: [],
    words: [{ surface: "あ あ", phones: ["."] }],
  };
  // The surface is 2 tokens and the normal form "." is 1 token, so the word2ph length 4 and the
  // input_ids length 3 disagree.
  assertThrows(
    () => buildSbv2ModelInput(utterance, tokenizer, rules),
    InputError,
    "input_ids length",
  );
});

test("DebertaTokenizer: NFKC splits one character into several tokens (… → ...)", () => {
  const tokenizer = tinyTokenizer([".", "あ"]);
  assertEquals(tokenizer.tokenize("…"), [".", ".", "."]);
  assertEquals(tokenizer.tokenize("あ…"), ["あ", ".", ".", "."]);
  assertEquals(tokenizer.encode("あ"), [1, 5, 2]);
  // A character not in the vocab becomes [UNK] (a known degradation, not a swallowed error).
  assertEquals(tokenizer.encode("ん"), [1, 3, 2]);
});

test("DebertaTokenizer: the clean tables' removal and spacing take effect", () => {
  // U+0000 is removed and U+200B (ZWSP) becomes a space. **NBSP is not used for the spacing case**
  // because the later split `/\s+/` already treats NBSP as whitespace — dropping the spaced rule
  // entirely would give the same answer and make the check vacuous (`/\s/` does not match ZWSP).
  const vocabText = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "a", "b"].join("\n");
  const special = { clsId: 1, sepId: 2, unkId: 3 };
  const tokenizer = DebertaTokenizer.fromVocabText(
    vocabText,
    { removed: [[0, 0]], spaced: [[0x200b, 0x200b]] },
    special,
  );
  assertEquals(tokenizer.tokenize("a\u0000b"), ["a", "b"]);
  assertEquals(tokenizer.tokenize("a\u200bb"), ["a", "b"]);
  // Counterpart: with empty clean tables the same input does not split (shows the two above really
  // exercise the tables).
  const bare = DebertaTokenizer.fromVocabText(vocabText, { removed: [], spaced: [] }, special);
  assertEquals(bare.tokenize("a\u0000b"), ["a", "\u0000", "b"]);
  assertEquals(bare.tokenize("a\u200bb"), ["a", "\u200b", "b"]);
});

test("DebertaTokenizer: no trailing CR remains even in a vocabulary with CRLF line endings", () => {
  // A leftover \r sends every lookup to [UNK] (no error; only the BERT features quietly break).
  const tokenizer = DebertaTokenizer.fromVocabText(
    "[PAD]\r\n[CLS]\r\n[SEP]\r\n[UNK]\r\nあ\r\n",
    { removed: [], spaced: [] },
    { clsId: 1, sepId: 2, unkId: 3 },
  );
  assertEquals(tokenizer.encode("あ"), [1, 4, 2]);
});

test("DebertaTokenizer: rejects duplicate vocabulary lines (no last-one-wins lookup of another row)", () => {
  // Line number = ID, so a duplicate line makes the earlier ID unreachable. No exception is raised
  // and only the BERT features change to "a different, legal row", so neither the id-sequence
  // length check nor the word2ph comparison catches it.
  assertThrows(
    () =>
      DebertaTokenizer.fromVocabText(
        "[PAD]\n[CLS]\n[SEP]\n[UNK]\nあ\nあ",
        { removed: [], spaced: [] },
        { clsId: 1, sepId: 2, unkId: 3 },
      ),
    Error,
    "vocabText: line 5 has the same token as line 4",
  );
});

test("toBertText: maps only symbol words to their normal form and concatenates", () => {
  const punctuations = new Set([",", "."]);
  const words = [
    { surface: "これ", phones: ["k", "o", "r", "e"] },
    { surface: "。", phones: ["."] },
  ];
  assertEquals(toBertText(words, punctuations), "これ.");
  // A word that is not a symbol (not in normal form even with a single phone) keeps its surface.
  assertEquals(toBertText([{ surface: "ん", phones: ["N"] }], punctuations), "ん");
});

test("bertHiddenOutput: chooses relative to the end (the same rule for any layer count)", () => {
  const full = Array.from({ length: 25 }, (_, index) => `h${index}`);
  assertEquals(bertHiddenOutput(full, 3), "h22");
  assertEquals(bertHiddenOutput(["h0", "h1", "h2"], 3), "h0");
  assertThrows(() => bertHiddenOutput(["h0", "h1"], 3), Error, "at least 3 are required");
  assertThrows(() => bertHiddenOutput(full, 0), Error, "is not an integer >= 1");
});

test("tileBertToPhoneLevel: repeats each token word2ph times and transposes to [dim, P]", () => {
  // 2 tokens × dim 3. The values differ by both token and dimension (dim ≠ P and asymmetric, to
  // exercise a transpose mix-up — with symmetric input, swapping the axes gives the same answer).
  const hidden = new Float32Array([1, 2, 3, 10, 20, 30]);
  const { data, dim, columns } = tileBertToPhoneLevel(hidden, 2, [1, 2]);
  assertEquals(columns, 3);
  assertEquals(dim, 3);
  // Expected: the columns are [t0, t1, t1] and the rows are dim. Row-major [dim=3][P=3].
  assertEquals([...data], [1, 10, 10, 2, 20, 20, 3, 30, 30]);
});

test("tileBertToPhoneLevel: fault injection — a different word2ph distribution changes the order of values", () => {
  // The counterpart that rejects an implementation that passes as long as the total matches (one
  // that ignores the distribution and splits evenly, for example).
  const hidden = new Float32Array([1, 2, 3, 10, 20, 30]);
  const a = tileBertToPhoneLevel(hidden, 2, [1, 2]);
  const b = tileBertToPhoneLevel(hidden, 2, [2, 1]);
  assertEquals(a.columns, b.columns, "the totals are equal (= a fault the length cannot detect)");
  assert(
    [...a.data].some((value, index) => value !== b.data[index]),
    "the values are the same for a different distribution",
  );
});

test("tileBertToPhoneLevel: rejects a mismatch between the token count and the word2ph length", () => {
  const hidden = new Float32Array([1, 2, 3, 10, 20, 30]);
  assertThrows(() => tileBertToPhoneLevel(hidden, 2, [1, 2, 1]), Error, "word2ph length");
  assertThrows(() => tileBertToPhoneLevel(hidden, 4, [1, 1, 1, 1]), Error, "is not divisible by");
});

// ---- Real assets (if present) -----------------------------------------------

const SYMBOLS_PATH = new URL("../../../outputs/misc/sbv2-demo/symbols.json", import.meta.url);
const assetAvailable = (() => {
  try {
    return statSync(SYMBOLS_PATH).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
})();

if (!assetAvailable) {
  console.warn(
    `[karume] ${SYMBOLS_PATH.pathname} is missing, so the demo asset check is SKIPPED. ` +
      "Generate: cd tools/exporter && uv run --group sbv2 python sbv2_demo.py assets",
  );
}

test.skipIf(!assetAvailable)("demo assets: symbols.json reads as JP-Extra rules", async () => {
  const rules = parseJpExtraRules(JSON.parse(await readTextFile(SYMBOLS_PATH)), "symbols.json");
  // The asset is authoritative for the values themselves (no expected-value table on the TS side —
  // that would erase the point of the asset). Only "the rules are well formed" is pinned here.
  assertEquals(rules.symbols[0], rules.pad, "the symbol table starts with PAD");
  assert(rules.toneStart + 1 < rules.numTones, "tone base + maximum tone (1) is in range");
  assert(rules.languageId < rules.numLanguages, "the language ID is in range");
  for (const punctuation of rules.punctuations) {
    assert(
      rules.symbolToId.has(punctuation),
      `punctuation ${punctuation} is not in the symbol table`,
    );
  }
  assert(rules.hopLength > 0 && rules.samplingRate > 0);
});
