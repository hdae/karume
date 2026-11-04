// Behaviour tests of SBV2 utterance input (the `toUtterance` conversion and the checks at the
// synthesis entry point).
//
// When this layer breaks, **the shapes stay valid and only the audio changes** — front still runs
// with the nucleus → tone expansion off by one step, and an utterance with replaced moras runs as
// long as the phoneme count matches. Neither raises an exception. These tests separately pin that
// the conversion follows the upstream rules and that a broken utterance is stopped by a check.
//
// The conversion and the checks are karume's pure functions, so they run without a dictionary. The
// path through the caller-side analysis (`@hdae/yomi`) runs only when a real dictionary is present
// (since 0.6.0 the analysis lives outside the package).

import { isNotFound, readDirSync, readFileSync } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { analyzeWithWords, JtdDictionary, OverlayDictionary, type OverlayEntry } from "@hdae/yomi";
import { InputError } from "../src/sbv2/errors.ts";
import { assertWordPhones, buildSbv2ModelInput } from "../src/sbv2/text/model-input.ts";
import { toSbv2PhoneTone } from "../src/sbv2/text/phone-tone.ts";
import {
  type Mora,
  type Phrases,
  type Utterance,
  type Word,
  toUtterance,
} from "../src/sbv2/text/utterance.ts";
import { type JpExtraRules, parseJpExtraRules } from "../src/sbv2/text/symbols.ts";
import { DebertaTokenizer } from "../src/sbv2/text/tokenizer.ts";

/**
 * JP-Extra rules for testing. **Not the real values** (the `symbols.json` asset is authoritative
 * for those). The tone base is non-zero (with 0, an implementation that does not add it could not
 * be told apart).
 */
const TEST_RULES_JSON = {
  symbols: ["_", "a", "e", "k", "m", "o", "r", "t", "u", ".", ","],
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

const TEXT = "香留芽";
const textTokenizer = (): DebertaTokenizer => tinyTokenizer(["香", "留", "芽", "。"]);

// ---- Nucleus → tone expansion (no dictionary) ------------------------------

/**
 * The analysis of one phrase "カルメ" (built by hand in the shape the caller-side analyser returns).
 */
const karumePhrases = (accentNucleus: number, punctuations: string[] = []): Phrases => ({
  result: {
    leadingPunctuations: [],
    accentPhrases: [
      {
        moras: [
          { kana: "カ", consonant: "k", vowel: "a" },
          { kana: "ル", consonant: "r", vowel: "u" },
          { kana: "メ", consonant: "m", vowel: "e" },
        ],
        accentNucleus,
        punctuations,
      },
    ],
  },
  // The word alignment holds each symbol as its own element (the same shape as the analyser's word
  // view).
  words: [
    { surface: TEXT, phones: ["k", "a", "r", "u", "m", "e"] },
    ...punctuations.map((punct) => ({ surface: punct, phones: [punct] })),
  ],
});

const tonesOf = (utterance: Utterance): number[] => utterance.moras.map((mora) => mora.tone);

test("toUtterance: expands the nucleus into a binary tone per mora (heiban / atamadaka / nakadaka / odaka)", () => {
  // The same rule as upstream SBV2's g2p. Off by even one step, it is a silent wrong value:
  // "synthesised with a different accent".
  assertEquals(tonesOf(toUtterance(karumePhrases(0))), [0, 1, 1], "heiban (flat)");
  assertEquals(tonesOf(toUtterance(karumePhrases(1))), [1, 0, 0], "atamadaka (head-high)");
  assertEquals(tonesOf(toUtterance(karumePhrases(2))), [0, 1, 0], "nakadaka (middle-high)");
  assertEquals(tonesOf(toUtterance(karumePhrases(3))), [0, 1, 1], "odaka (tail-high)");
});

test("toUtterance: clamps the nucleus's upper end to the mora count (no round trip that cannot be made)", () => {
  // An out-of-range nucleus comes up routinely from dictionary differences and user dictionaries.
  // Rejecting it would mean "an utterance exactly as analysed does not pass", so only the upper end
  // is clamped (clamping does not change the tone sequence = the audio is the same).
  assertEquals(tonesOf(toUtterance(karumePhrases(9))), tonesOf(toUtterance(karumePhrases(3))));
});

test("toUtterance: rejects a negative or non-integer nucleus (the lower end is not clamped)", () => {
  // Silently falling back to heiban would synthesise with an accent other than the one specified.
  assertThrows(() => toUtterance(karumePhrases(-1)), InputError, "accentNucleus = -1");
  assertThrows(() => toUtterance(karumePhrases(1.5)), InputError, "accentNucleus = 1.5");
});

test("toUtterance: a phrase's symbols go to its last mora, symbols before any phrase go to the leading field", () => {
  const withPunct = toUtterance(karumePhrases(1, ["."]));
  assertEquals(withPunct.moras[2].punctuations, ["."], "not attached to the phrase's last mora");
  assertEquals(withPunct.moras[0].punctuations, undefined, "an unrelated mora grew the field");

  // Symbols at the start of a sentence cannot be recovered from a flat phoneme sequence (exactly
  // why they are carried in the structure).
  const leading = karumePhrases(1);
  const withLeading = toUtterance({
    result: { ...leading.result, leadingPunctuations: ["…"] },
    words: [{ surface: "…", phones: ["…"] }, ...leading.words],
  });
  assertEquals(withLeading.leadingPunctuations, ["…"]);
});

// ---- Several phrases (phrase boundaries; no dictionary) ---------------------
//
// In a one-phrase fixture, "the utterance's last mora" and "the phrase's last mora" are the same,
// so the power to detect a symbol attached to the wrong mora is structurally zero. Index shifts
// across phrases, symbols leaking to another phrase, and phrases without moras are caught only here
// (the dictionary-path tests run only where the assets are present).

/**
 * Two phrases "カルメ" + "トク" (`words` built by hand to match the phrases' phonemes — one word per
 * symbol).
 */
const twoPhrases = (
  nuclei: readonly [number, number],
  punctuations: readonly [string[], string[]] = [[], []],
): Phrases => ({
  result: {
    leadingPunctuations: [],
    accentPhrases: [
      {
        moras: [
          { kana: "カ", consonant: "k", vowel: "a" },
          { kana: "ル", consonant: "r", vowel: "u" },
          { kana: "メ", consonant: "m", vowel: "e" },
        ],
        accentNucleus: nuclei[0],
        punctuations: punctuations[0],
      },
      {
        moras: [
          { kana: "ト", consonant: "t", vowel: "o" },
          { kana: "ク", consonant: "k", vowel: "u" },
        ],
        accentNucleus: nuclei[1],
        punctuations: punctuations[1],
      },
    ],
  },
  words: [
    { surface: TEXT, phones: ["k", "a", "r", "u", "m", "e"] },
    ...punctuations[0].map((punct) => ({ surface: punct, phones: [punct] })),
    { surface: "特", phones: ["t", "o", "k", "u"] },
    ...punctuations[1].map((punct) => ({ surface: punct, phones: [punct] })),
  ],
});

test("toUtterance: each phrase's nucleus is expanded independently (phrase 2 is not tainted by phrase 1's nucleus)", () => {
  // A regression that makes `at` in `tones[at]` an index over the whole utterance can never be hit
  // by a one-phrase test.
  assertEquals(tonesOf(toUtterance(twoPhrases([1, 0]))), [1, 0, 0, 0, 1]);
  assertEquals(tonesOf(toUtterance(twoPhrases([0, 0]))), [0, 1, 1, 0, 1]);
  // The expansion equals exactly "phrase 1's expansion ++ phrase 2's expansion" (the one-phrase
  // answers concatenated).
  assertEquals(tonesOf(toUtterance(twoPhrases([1, 2]))), [
    ...tonesOf(toUtterance(karumePhrases(1))),
    0,
    1,
  ]);
  // The phoneme sequence and the word alignment agree across phrases too.
  const utterance = toUtterance(twoPhrases([1, 0]));
  assertWordPhones(toSbv2PhoneTone(utterance, "_").phones, utterance.words);
});

test("toUtterance: a symbol attaches to the last mora of **its own** phrase (a middle phrase's symbol does not leak to the last phrase)", () => {
  const utterance = toUtterance(twoPhrases([1, 0], [[","], ["."]]));
  assertEquals(utterance.moras.length, 5, "moras were lost across phrases");
  assertEquals(
    utterance.moras[2].punctuations,
    [","],
    "the middle phrase's symbol is not on phrase 1's last mora",
  );
  assertEquals(
    utterance.moras[4].punctuations,
    ["."],
    "the last phrase's symbol is not on phrase 2's last mora",
  );
  assertEquals(utterance.moras[3].punctuations, undefined, "phrase 2's first mora grew the field");
  assertEquals(utterance.leadingPunctuations, []);
  // The position shows in the phoneme sequence too (a symbol leaking to the last phrase changes the
  // order).
  assertEquals(toSbv2PhoneTone(utterance, "_").phones, [
    "_",
    "k",
    "a",
    "r",
    "u",
    "m",
    "e",
    ",",
    "t",
    "o",
    "k",
    "u",
    ".",
    "_",
  ]);
  assertWordPhones(toSbv2PhoneTone(utterance, "_").phones, utterance.words);
});

test("toUtterance: the symbols of a phrase without moras go to the preceding mora (or to the leading field at the start)", () => {
  // The two branches the doc of `utterance.ts` spells out. The one-phrase fixture always has 3
  // moras, so it reaches neither branch.
  const emptyPhrase = { moras: [], accentNucleus: 0, punctuations: ["…"] };
  const karume = karumePhrases(0);

  const trailing = toUtterance({
    result: {
      leadingPunctuations: [],
      accentPhrases: [...karume.result.accentPhrases, emptyPhrase],
    },
    words: [...karume.words, { surface: "…", phones: ["…"] }],
  });
  assertEquals(trailing.moras.length, 3, "a phrase without moras grew moras");
  assertEquals(trailing.moras[2].punctuations, ["…"], "not moved to the preceding mora");
  assertEquals(trailing.leadingPunctuations, []);
  assertWordPhones(toSbv2PhoneTone(trailing, "_").phones, trailing.words);

  // Placed first, there is no mora to move to → `leadingPunctuations` (information only the
  // structure can carry).
  const leading = toUtterance({
    result: {
      leadingPunctuations: [],
      accentPhrases: [emptyPhrase, ...karume.result.accentPhrases],
    },
    words: [{ surface: "…", phones: ["…"] }, ...karume.words],
  });
  assertEquals(leading.leadingPunctuations, ["…"]);
  for (const [index, mora] of leading.moras.entries()) {
    assertEquals(mora.punctuations, undefined, `moras[${index}] grew the field`);
  }
  assertWordPhones(toSbv2PhoneTone(leading, "_").phones, leading.words);
});

test("toUtterance: folds the geminate consonant (sokuon) into its phoneme (`Mora.vowel` is the phoneme itself)", () => {
  // The upstream analysis writes the sokuon as "cl", but the word alignment's phoneme is "q".
  // Carried unfolded, the comparison of the phoneme sequence with words breaks on that one phoneme
  // (= every input with a sokuon fails).
  const utterance = toUtterance({
    result: {
      leadingPunctuations: [],
      accentPhrases: [
        {
          moras: [
            { kana: "カ", consonant: "k", vowel: "a" },
            { kana: "ッ", vowel: "cl" },
            { kana: "ト", consonant: "t", vowel: "o" },
          ],
          accentNucleus: 1,
          punctuations: [],
        },
      ],
    },
    words: [{ surface: "カット", phones: ["k", "a", "q", "t", "o"] }],
  });
  assertEquals(utterance.moras[1].vowel, "q");
  assertEquals(toSbv2PhoneTone(utterance, "_").phones, ["_", "k", "a", "q", "t", "o", "_"]);
  // The comparison passes (an implementation that does not fold fails here).
  assertWordPhones(toSbv2PhoneTone(utterance, "_").phones, utterance.words);
});

test("toUtterance: a mora without a consonant has no consonant field at all (moraic nasal, sokuon, lone vowel)", () => {
  // `Mora.consonant` is an optional property, not `string | undefined`. A field appearing would
  // break the duty "never create a field that can be passed but has no effect" (the moraic nasal N
  // is the ordinary path).
  const utterance = toUtterance({
    result: {
      leadingPunctuations: [],
      accentPhrases: [
        {
          moras: [
            { kana: "カ", consonant: "k", vowel: "a" },
            { kana: "ン", vowel: "N" },
          ],
          accentNucleus: 1,
          punctuations: [],
        },
      ],
    },
    words: [{ surface: "缶", phones: ["k", "a", "N"] }],
  });
  assertEquals(Object.keys(utterance.moras[0]).sort(), ["consonant", "kana", "tone", "vowel"]);
  assertEquals(Object.keys(utterance.moras[1]).sort(), ["kana", "tone", "vowel"]);
  // A mora without a consonant emits only one phoneme (`_` + k a N + `_`).
  assertEquals(toSbv2PhoneTone(utterance, "_").phones, ["_", "k", "a", "N", "_"]);
});

// ---- Packing the phoneme and tone sequences (no dictionary) ----------------

test("toSbv2PhoneTone: PAD at both ends, tone 0 for symbols, one tone within a mora", () => {
  const utterance = toUtterance(karumePhrases(1, ["."]));
  const { phones, tones } = toSbv2PhoneTone(utterance, "_");
  assertEquals(phones, ["_", "k", "a", "r", "u", "m", "e", ".", "_"]);
  // atamadaka: the consonant and vowel of the first mora are 1, the rest 0. Symbols and PAD are 0.
  assertEquals(tones, [0, 1, 1, 0, 0, 0, 0, 0, 0]);
});

test("toSbv2PhoneTone: rejects a tone other than 0/1 (types cannot guard calls from JS)", () => {
  // Both 2 and -1 fall within the symbol table once toneStart is added (`phonesTonesToModelIds`
  // lets them through).
  const utterance = toUtterance(karumePhrases(1));
  const broken = (tone: number): Utterance => ({
    ...utterance,
    moras: utterance.moras.map((mora, index) => (index === 1 ? ({ ...mora, tone } as Mora) : mora)),
  });
  assertThrows(() => toSbv2PhoneTone(broken(2), "_"), InputError, "moras[1] (ル) has tone = 2");
  assertThrows(() => toSbv2PhoneTone(broken(-1), "_"), InputError, "tone = -1");
  assertThrows(() => toSbv2PhoneTone(broken(0.5), "_"), InputError, "tone = 0.5");
});

// ---- Checks at the synthesis entry point (comparing moras with words) ------

test("assertWordPhones: rejects different content even when the lengths match", () => {
  const words: Word[] = [{ surface: TEXT, phones: ["k", "a", "r", "u", "m", "e"] }];
  const phones = ["_", "k", "a", "r", "u", "m", "e", "_"];
  assertWordPhones(phones, words);
  assertThrows(
    () => assertWordPhones(["_", "k", "a", "r", "u", "m", "o", "_"], words),
    InputError,
    "Phoneme[5]",
  );
  assertThrows(() => assertWordPhones(phones.slice(0, 7), words), InputError, "phoneme count 5");
});

test("editing tone directly reaches the tone IDs of the synthesis input (the phoneme side does not move)", () => {
  const rules = testRules();
  const tokenizer = textTokenizer();
  const base = toUtterance(karumePhrases(1));
  const baseInput = buildSbv2ModelInput(base, tokenizer, rules);

  // Invert it (every position changes whatever the analysis returned = an implementation that
  // ignores the edit is always caught).
  const flipped: Utterance = {
    ...base,
    moras: base.moras.map((mora) => ({ ...mora, tone: (1 - mora.tone) as 0 | 1 })),
  };
  const edited = buildSbv2ModelInput(flipped, tokenizer, rules);

  assertEquals(
    edited.tones,
    [0, 0, 0, 1, 1, 1, 1, 0],
    "the moras' tones were not handed to the phonemes",
  );
  assertEquals(edited.phones, baseInput.phones, "the phoneme sequence moved with a tone edit");
  assertEquals(edited.ids.phoneIds, baseInput.ids.phoneIds, "the phoneme IDs moved");
  assertEquals(edited.bertText, baseInput.bertText, "the BERT input does not read tones");
  assertEquals(edited.baseWord2ph, baseInput.baseWord2ph, "word2ph does not read tones");
  assert(
    edited.ids.toneIds.some((id, index) => id !== baseInput.ids.toneIds[index]),
    "the tone IDs do not reflect the edit",
  );
  // After add_blank the real phoneme positions are the odd indices (the `intersperse` convention).
  for (const [index, tone] of edited.tones.entries()) {
    assertEquals(edited.ids.toneIds[2 * index + 1], TEST_RULES_JSON.toneStart + tone);
    assertEquals(
      edited.ids.toneIds[2 * index],
      TEST_RULES_JSON.blankId,
      "a blank position was altered",
    );
  }
});

test("edits that change phonemes, adding or removing symbols, and mixed-up words fail at the entry point", () => {
  const rules = testRules();
  const tokenizer = textTokenizer();
  const base = toUtterance(karumePhrases(1));
  const withMoras = (moras: readonly Mora[]): Utterance => ({ ...base, moras });

  // ① A mora read differently (the phoneme count does not change = a length check alone lets it
  //  through).
  assertThrows(
    () =>
      buildSbv2ModelInput(
        withMoras([
          base.moras[0],
          { kana: "ロ", consonant: "r", vowel: "o", tone: base.moras[1].tone },
          base.moras[2],
        ]),
        tokenizer,
        rules,
      ),
    InputError,
    "Phoneme[3]",
  );

  // ② One mora dropped (the phoneme count changes).
  assertThrows(
    () => buildSbv2ModelInput(withMoras(base.moras.slice(0, 2)), tokenizer, rules),
    InputError,
    "phoneme count 4",
  );

  // ③ A symbol added (as when trying to insert a question's rising intonation).
  assertThrows(
    () =>
      buildSbv2ModelInput(
        withMoras([base.moras[0], base.moras[1], { ...base.moras[2], punctuations: ["."] }]),
        tokenizer,
        rules,
      ),
    InputError,
    "phoneme count 7",
  );

  // ④ words taken from a different analysis mixed in (= moras and words not from the same
  //  analysis).
  assertThrows(
    () =>
      buildSbv2ModelInput(
        { ...base, words: toUtterance(karumePhrases(1, ["."])).words },
        tokenizer,
        rules,
      ),
    InputError,
    "phoneme count",
  );
});

// ---- The caller-side analysis path (if a real dictionary is present) -------
//
// 0.6.0 moved the analysis out of the package. Pinned here: the analyser's return value can be
// passed to `toUtterance` as is, and a caller-side user dictionary reaches the final IDs.

const DICT_DIR = new URL("../../../outputs/misc/yomi/", import.meta.url);

/**
 * The JTD1 dictionary in `outputs/misc/yomi/` (found by extension, as the name changes with the
 * version).
 */
const dictUrl = ((): URL | undefined => {
  let found: string | undefined;
  try {
    for (const entry of readDirSync(DICT_DIR)) {
      if (!entry.isFile || !entry.name.endsWith(".jtd")) continue;
      if (found === undefined || entry.name < found) found = entry.name;
    }
  } catch (cause) {
    if (!isNotFound(cause)) throw cause;
  }
  return found === undefined ? undefined : new URL(found, DICT_DIR);
})();

if (dictUrl === undefined) {
  console.warn(
    `[karume] ${DICT_DIR.pathname} has no *.jtd, so the analysis-path checks are SKIPPED. ` +
      "Get: naist-jdic.jtd from the HF dataset hdae/yomi-dict (decompress the gzip and place it)",
  );
}

/**
 * The 19MB file is read only when the asset is present (every test in this section reuses the same
 * instance).
 */
const dictionary =
  dictUrl === undefined ? undefined : JtdDictionary.load(readFileSync(dictUrl).buffer);

/**
 * Narrows away undefined outside ignore (`ignore: !available` guards it, so it never happens in
 * practice).
 */
const jtd = (): JtdDictionary => {
  if (dictionary === undefined) throw new Error("a dictionary test ran without a dictionary");
  return dictionary;
};

const entry = (reading: string, accentType: number): OverlayEntry => ({
  surface: TEXT,
  reading,
  accentType,
});

test.skipIf(dictUrl === undefined)(
  "the analyser's return value becomes the utterance as is (user dictionaries are applied on the caller side)",
  () => {
    const dict = jtd();
    const rules = testRules();
    const tokenizer = textTokenizer();
    const base = buildSbv2ModelInput(toUtterance(analyzeWithWords(dict, TEXT)), tokenizer, rules);

    // Reading: the main dictionary's reading is not written as a fixed value (it changes with the
    // dictionary version — what is verified is that the caller-side overlay produces the specified
    // phoneme sequence, not how the main dictionary reads it).
    const head = new OverlayDictionary(dict, [entry("カルメ", 1)]);
    const fixed = buildSbv2ModelInput(
      toUtterance(analyzeWithWords(dict, TEXT, head)),
      tokenizer,
      rules,
    );
    assertEquals(fixed.phones, ["_", "k", "a", "r", "u", "m", "e", "_"]);
    assert(
      base.phones.join(" ") !== fixed.phones.join(" "),
      "the reading is the same as without the overlay",
    );

    // Accent: changing only the type for the same reading leaves the phoneme sequence identical and
    // moves only the tones.
    const flat = new OverlayDictionary(dict, [entry("カルメ", 0)]);
    const heiban = buildSbv2ModelInput(
      toUtterance(analyzeWithWords(dict, TEXT, flat)),
      tokenizer,
      rules,
    );
    assertEquals(heiban.phones, fixed.phones);
    assertEquals(fixed.tones, [0, 1, 1, 0, 0, 0, 0, 0], "atamadaka: only the first mora is high");
    assertEquals(heiban.tones, [0, 0, 0, 1, 1, 1, 1, 0], "heiban: rises after the first mora");

    // The difference reaches the IDs handed to front (tone base 6 added; blank positions are
    // blankId).
    assertEquals(
      fixed.ids.phoneIds,
      heiban.ids.phoneIds,
      "the phoneme IDs do not change with the type",
    );
    assert(
      fixed.ids.toneIds.some((id, index) => id !== heiban.ids.toneIds[index]),
      "the tone IDs do not reflect the difference in type",
    );
    // word2ph also comes from the same analysis (exercising the word split of an overlay folded
    // into one word).
    assertEquals(
      fixed.baseWord2ph.reduce((sum, count) => sum + count, 0),
      fixed.phones.length,
      "sum(word2ph) differs from the phoneme count",
    );
  },
);

test.skipIf(dictUrl === undefined)(
  "an utterance exactly as analysed passes the checks, and so does one with only its tones edited",
  () => {
    const dict = jtd();
    const rules = testRules();
    const tokenizer = textTokenizer();
    const overlay = new OverlayDictionary(dict, [entry("カルメ", 1)]);
    const utterance = toUtterance(analyzeWithWords(dict, TEXT, overlay));
    assertEquals(utterance.moras.length, 3, "not folded into one phrase of 3 moras");
    assertEquals(
      utterance.moras.map((mora) => mora.tone),
      [1, 0, 0],
      "not atamadaka",
    );

    // The same edit as changing the nucleus to heiban (setting the moras' tones directly).
    const flattened: Utterance = {
      ...utterance,
      moras: utterance.moras.map((mora, index) => ({ ...mora, tone: index === 0 ? 0 : 1 })),
    };
    const edited = buildSbv2ModelInput(flattened, tokenizer, rules);
    const head = buildSbv2ModelInput(utterance, tokenizer, rules);
    assertEquals(edited.tones, [0, 0, 0, 1, 1, 1, 1, 0], "not the heiban tone sequence");
    assertEquals(edited.phones, head.phones, "the phoneme sequence moved");
    assertEquals(edited.baseWord2ph, head.baseWord2ph, "word2ph does not read tones");
    assertEquals(edited.bertText, head.bertText, "the BERT input does not read tones");

    // Mixing in words taken from a different analysis (without the overlay) fails at the entry
    // point.
    assertThrows(
      () =>
        buildSbv2ModelInput(
          { ...utterance, words: toUtterance(analyzeWithWords(dict, TEXT)).words },
          tokenizer,
          rules,
        ),
      InputError,
    );
  },
);
