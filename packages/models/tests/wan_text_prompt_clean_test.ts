// Behaviour and random-sweep parity of Wan2.1's preprocessing (src/wan/text/prompt-clean.ts — the
// mirror of upstream's prompt_clean) (ADR 0119 decision 2). Needs neither a GPU nor real assets.
//
// The random-sweep expectations are the digests in the fixture `wan-text/prompt-clean-sweep.json`
// (values the recipe took from upstream's `prompt_clean` / `BADNESS_RE`). Here the same sequences are
// built with the same PRNG, pools and procedure as the recipe, and the digests of the outcomes (the
// output when accepted, the reason when rejected) are compared. At least the pools and counts of the
// investigation §3.3: 200,000 strings from the safe pool (without `&` and the literal characters of
// BADNESS_RE), 50,000 from the full pool (mixing in the 449 literal characters and `&`), and 20,000
// from the pool that exercises the entity check.
//
// The all-codepoint sweeps (BADNESS_RE `\w` / `\s` / `.`, single-character prompt_clean, NFC
// version drift) are the describe block at the end of this file.

import { describe, test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { ModelInputError } from "../src/errors.ts";
import { inCodeRanges } from "../src/text/code-ranges.ts";
import { cleanPrompt, PromptCleanError } from "../src/wan/text/prompt-clean.ts";
import {
  allCodePoints,
  chunkedDigests,
  digestBits,
  expandRanges,
  fuzzStrings,
  loadPromptCleanTables,
  loadSweepFixture,
  mismatchedIndices,
  recordOf,
  sweepBlocks,
  xorshift32,
} from "./helpers/wan-prompt-clean-fixture.ts";

const fixture = await loadSweepFixture();
const tables = await loadPromptCleanTables();
const clean = (text: string): string => cleanPrompt(tables, text);

const rejectReason = (text: string): string => {
  try {
    clean(text);
  } catch (error) {
    if (error instanceof PromptCleanError) return error.reason;
    throw error;
  }
  throw new Error(`not rejected: ${JSON.stringify(text)}`);
};

// ---- Deterministic processing (expectations confirmed against the recipe's upstream prompt_clean) ----

test("table lookups: full-width forms, curly quotes, ligatures, terminal escapes and newlines become the same string as upstream", () => {
  assertEquals(
    clean("色调艳丽，过曝"),
    "色调艳丽,过曝",
    "the full-width comma of the official negative",
  );
  assertEquals(clean("“quoted” it’s"), '"quoted" it\'s');
  // `ŉ` becomes `ʼn` (U+02BC) through the ligature table, then `'n` through uncurl_quotes in the same
  // pass (the order of the tables matters).
  assertEquals(clean("ŉ"), "'n");
  assertEquals(clean("ﬁne"), "fine");
  assertEquals(clean("\x1b[31mred\x1b[0m"), "red");
  assertEquals(clean("a\r\nb\rc d"), "a b c d");
  assertEquals(clean("ＡＢＣ　ｶﾀｶﾅ"), "ABC カタカナ");
  assertEquals(clean("a\x07b﻿c"), "abc");
  assertEquals(clean("x​y"), "x​y", "a zero-width space is not whitespace");
});

test("the digits of a terminal escape are Python's `\\d` (Unicode Nd, across planes)", () => {
  // Matching only ASCII digits would leave ESC and `[` behind, and removing control characters would
  // leave `[`.
  assertEquals(clean("\x1b[١mred"), "red", "Arabic-Indic digit");
  assertEquals(clean("\x1b[\u{1D7CE}mred"), "red", "mathematical bold digit (U+1D7CE)");
  // A sequence not closed by a letter is not removed; only ESC is removed as a control character.
  assertEquals(clean("\x1b[1;2xok\x1b["), "ok[");
});

test("chunking: NFC does not compose across a cut at 1,000,000 code points (ftfy's chunking)", () => {
  // Upstream (measured): 999,999 spaces + e + U+0301 end the first chunk at e, and U+0301 falls into
  // the next chunk, so they are not composed. With one space fewer they fall into the same chunk and
  // become é.
  assertEquals(clean(" ".repeat(999_999) + "é"), "é");
  assertEquals(clean(" ".repeat(999_998) + "é"), "é");
});

test("the checks run at the start of each pass: sequences produced by NFC or the full-width table are rejected too", () => {
  // The first pass's input does not match, but BADNESS_RE matches in the 2nd pass after NFC produced
  // à À (upstream's fix_encoding just tries and reverts = lets it through silently, but this port
  // does not port the guessing, so it rejects).
  assertEquals(rejectReason("àÀ"), "mojibake");
  // The full-width `＆` becomes `&` in the first pass, and ftfy's unescape_html decodes `&amp;` in the
  // 2nd pass.
  assertEquals(rejectReason("＆amp;"), "entity");
});

test("the entity check: rejected when a letter follows `&`, accepted when whitespace or a newline follows", () => {
  assertEquals(rejectReason("salt &amp; pepper"), "entity");
  assertEquals(
    rejectReason("R&D"),
    "entity",
    "sequences that would not form an entity are also rejected",
  );
  assertEquals(
    rejectReason("x &#65 y"),
    "entity",
    "a numeric reference without a semicolon (upstream decodes it to A)",
  );
  assertEquals(clean("salt & pepper"), "salt & pepper");
  assertEquals(clean("a &\rb"), "a & b", "CR goes through html.unescape after becoming a newline");
  assertEquals(clean("a &　b"), "a & b");
});

test("rejection: mojibake, C1 and unassigned carry the reason in a PromptCleanError (an input error)", () => {
  assertEquals(rejectReason("cafÃ©"), "mojibake");
  assertEquals(rejectReason("a\x85b"), "c1");
  assertEquals(rejectReason("a͸b"), "unassigned");
  const error = assertThrows(
    () => cleanPrompt(tables, "a\x85b", "negativePrompt"),
    ModelInputError,
  );
  assert(error instanceof PromptCleanError);
  assert(error.message.includes("negativePrompt"), error.message);
  assert(error.message.includes("U+0085"), error.message);
});

test("a lone surrogate is rejected before preprocessing (a ModelInputError, not a PromptCleanError)", () => {
  const error = assertThrows(() => clean("a\ud800b"), ModelInputError, "surrogate");
  assert(!(error instanceof PromptCleanError));
});

test("a broken table that never reaches a fixed point fails with a plain Error (no endless loop)", () => {
  // A cycle full-width A ↔ A (not in upstream's tables — a form only a corrupt asset produces).
  const cyclic = {
    ...tables,
    width: new Map([
      [0x41, "Ａ"],
      [0xff21, "A"],
    ]),
  };
  const error = assertThrows(() => cleanPrompt(cyclic, "A"), Error, "fixed point");
  assert(!(error instanceof ModelInputError), "a corrupt asset is not an input error");
});

// ---- Random sweeps (replay the recipe's sequences and compare digests) ---------------

test("PRNG: xorshift32 yields the same sequence as the recipe", () => {
  const next = xorshift32(fixture.prng.seed);
  assertEquals(Array.from({ length: fixture.prng.first.length }, next), fixture.prng.first);
});

for (const run of fixture.clean) {
  test(`random sweep [${run.name}]: the outcomes of ${run.count} strings match upstream's prompt_clean`, async () => {
    const pool = expandRanges(fixture.pools[run.pool]);
    const records: string[] = [];
    for (const text of fuzzStrings(pool, run.seed, run.count, run.maxLength)) {
      records.push(recordOf(tables, text));
    }
    const outcomes: Record<string, number> = {};
    for (const item of records) {
      const key = item.startsWith("A") ? "accepted" : item.slice(1);
      outcomes[key] = (outcomes[key] ?? 0) + 1;
    }
    assertEquals(outcomes, run.outcomes, "breakdown of accepted / rejected");
    const chunks = await chunkedDigests(records, run.chunk);
    assertEquals(
      mismatchedIndices(chunks, run.chunks),
      [],
      "chunks diverging from upstream (chunk index)",
    );
  });
}

test("every string of the safe pool is accepted (the rejection checks do not reject ordinary prompts)", () => {
  const safe = fixture.clean.find((run) => run.name === "safe");
  assertEquals(safe?.outcomes, { accepted: safe?.count ?? -1 });
});

for (const check of fixture.badness.filter((entry) => entry.template === undefined)) {
  test(`the JS translation of BADNESS_RE [random ${check.name}] matches upstream's is_bad on every string`, async () => {
    const pool = expandRanges(fixture.pools[check.pool as string]);
    let bits = "";
    for (const text of fuzzStrings(
      pool,
      check.seed as number,
      check.count as number,
      check.maxLength as number,
    )) {
      bits += tables.badness.test(text) ? "1" : "0";
    }
    assertEquals(Array.from(bits).filter((bit) => bit === "1").length, check.hits, "match count");
    assertEquals(await digestBits(bits), check.digest);
  });
}

// ---- All-codepoint sweeps ------------------------------------------------------
//
// The all-codepoint sweeps of Wan2.1's preprocessing (src/wan/text/prompt-clean.ts) (ADR 0119
// decision 2). Need neither a GPU nor real assets. The expectations are the digests of the fixture
// `wan-text/prompt-clean-sweep.json` (values the recipe took from upstream — one per block of 4,096
// code points).
//
//  - The JS translation of BADNESS_RE: every code point is put into contexts that exercise the
//    Unicode-dependent syntax (`\w`, `\s`, `.`, `[^…]`, `^`), and every hit / miss matches upstream's
//    `is_bad`
//  - Preprocessing of one character: every code point is put into `{}` and `a{}b`, and the outcome (the
//    output when accepted, the reason when rejected) matches upstream's prompt_clean (exhaustively
//    catching gaps in the tables, the whitespace set and the checks)
//  - NFC version drift: every code point assigned in Python (UCD 16.0.0) is put into 8 contexts, and the
//    JS engine's `normalize("NFC")` matches `unicodedata.normalize`. If one diverges, the characters of
//    that block need to be rejected / baked in the preprocessing tables (ADR 0119 decision 2 — the check
//    for when the execution engine's Unicode version changes)

describe("all-codepoint sweeps", () => {
  const everyCodePoint = allCodePoints();

  /**
   * Replaces **every** `{}` in `template` with one character (like Python's `str.replace` — JS's
   * `replace` only replaces the first one and also treats `$` in the replacement specially, so it is not
   * used).
   */
  const fill = (template: string, cp: number): string =>
    template.split("{}").join(String.fromCodePoint(cp));

  /** The first code point of each block (names a diverging block in the message). */
  const blockStarts = (indices: readonly number[]): string[] =>
    indices.map(
      (index) => `U+${(index * fixture.block).toString(16).toUpperCase().padStart(4, "0")}`,
    );

  for (const check of fixture.badness.filter((entry) => entry.template !== undefined)) {
    test(`the JS translation of BADNESS_RE [sweep ${check.name}: ${check.template}] matches is_bad on every code point`, async () => {
      const template = check.template as string;
      let bits = "";
      for (const cp of everyCodePoint) {
        bits += tables.badness.test(fill(template, cp)) ? "1" : "0";
      }
      assertEquals(Array.from(bits).filter((bit) => bit === "1").length, check.hits, "match count");
      assertEquals(await digestBits(bits), check.digest);
    });
  }

  for (const sweep of fixture.cleanSweeps) {
    test(`preprocessing sweep [${sweep.name}: ${sweep.template}] matches upstream's prompt_clean on every code point`, async () => {
      const blocks = await sweepBlocks(
        everyCodePoint,
        (cp) => recordOf(tables, fill(sweep.template, cp)),
        fixture.block,
      );
      assertEquals(
        blockStarts(mismatchedIndices(blocks, sweep.blocks)),
        [],
        "blocks diverging from upstream",
      );
    });
  }

  const assigned = everyCodePoint.filter((cp) => inCodeRanges(tables.assigned, cp));

  for (const sweep of fixture.nfcSweeps) {
    test(`NFC sweep [${sweep.name}] matches Python (UCD 16.0.0) on every code point`, async () => {
      const blocks = await sweepBlocks(
        assigned,
        (cp) => fill(sweep.template, cp).normalize("NFC"),
        fixture.block,
      );
      assertEquals(
        blockStarts(mismatchedIndices(blocks, sweep.blocks)),
        [],
        "blocks diverging from Python",
      );
    });
  }
});
