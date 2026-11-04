// Behaviour checks of the shared BPE layer (src/text/bpe.ts) and the incremental decoder
// (src/text/detokenizer.ts).
//
// It does not depend on real assets — a small synthetic vocabulary looks directly at "the priority
// of rank and position", "invalidating stale pairs", "expanding byte_fallback" and "how byte runs
// fold". The comparison with real assets (whether the id sequence matches upstream `tokenizers`)
// belongs to the family side (gemma4_tokenizer_test.ts); this file sets out the **rules
// themselves** and the **complexity** in a form that can fail.

import { beforeAll, describe, it, test } from "vite-plus/test";
import { assertEquals, assertThrows } from "@std/assert";
import { bpeEncode, type BpeModel, createBpeModel } from "../src/text/bpe.ts";
import { detokenize, StreamingDetokenizer } from "../src/text/detokenizer.ts";

/** The spelling of the byte_fallback vocabulary (the same convention as the assets). */
const byteToken = (byte: number): string =>
  `<0x${byte.toString(16).toUpperCase().padStart(2, "0")}>`;

/**
 * Builds a model from a list of spellings and `[left, right]` merge rules.
 *
 * The 256 byte entries are appended **after** the plain vocabulary (so that each test does not have
 * to write out the 256 entries `createBpeModel` requires).
 */
const buildModel = (
  pieces: readonly string[],
  merges: readonly (readonly [string, string])[] = [],
): BpeModel => {
  const tokens = [...pieces, ...Array.from({ length: 256 }, (_, byte) => byteToken(byte))];
  const idOf = new Map(tokens.map((token, id) => [token, id] as const));
  return createBpeModel({
    vocab: tokens.map((token, id) => [id, token] as const),
    merges: merges.map(
      ([left, right], rank) => [idOf.get(left) as number, idOf.get(right) as number, rank] as const,
    ),
    byteIds: Array.from({ length: 256 }, (_, byte) => idOf.get(byteToken(byte)) as number),
  });
};

/** Maps an id sequence back to spellings (to compare in a form whose intent reads). */
const spell = (model: BpeModel, ids: readonly number[]): string[] =>
  ids.map((id) => model.tokenOf.get(id) as string);

describe("asset checks: rejects shapes that would produce silently wrong values", () => {
  it("fails when byteIds does not have 256 entries", () => {
    assertThrows(
      () => createBpeModel({ vocab: [[0, "a"]], merges: [], byteIds: [0] }),
      Error,
      "does not have 256 entries",
    );
  });

  it("fails when a spelling byteIds points at is not `<0xHH>`", () => {
    // The consecutive `base + byte` numbering is a measured fact about the real assets, not a
    // schema guarantee. Letting a shifted table through would make the byte expansion point at
    // another valid row (a broken distribution that raises no exception).
    const tokens = ["x", ...Array.from({ length: 256 }, (_, byte) => byteToken(byte))];
    assertThrows(
      () =>
        createBpeModel({
          vocab: tokens.map((token, id) => [id, token] as const),
          merges: [],
          // Shifted one back = a table where everything is off by one byte.
          byteIds: Array.from({ length: 256 }, (_, byte) => byte),
        }),
      Error,
      "the spelling of",
    );
  });

  it("fails when the same pair has two merge rules (upstream silently lets the last one win)", () => {
    assertThrows(
      () =>
        buildModel(
          ["a", "aa"],
          [
            ["a", "a"],
            ["a", "a"],
          ],
        ),
      Error,
      "duplicate",
    );
  });

  it("fails when the merged spelling is not in the vocabulary", () => {
    assertThrows(() => buildModel(["a", "b"], [["a", "b"]]), Error, "is not in the vocabulary");
  });

  it("fails when vocabulary spellings repeat (row number = id could no longer be looked up)", () => {
    assertThrows(
      () =>
        createBpeModel({
          vocab: [
            [0, "a"],
            [1, "a"],
          ],
          merges: [],
          byteIds: [],
        }),
      Error,
      "duplicate",
    );
  });
});

describe("merge order: smallest rank first, ties from the left", () => {
  it("the pair with the smaller rank collapses first, not a greedy left-to-right take", () => {
    // ("b","c") is rank 0, ("a","b") rank 1. Taking from the left would give [ab, c], but the
    // source of truth goes by rank, so [a, bc] is correct.
    const model = buildModel(
      ["a", "b", "c", "ab", "bc"],
      [
        ["b", "c"],
        ["a", "b"],
      ],
    );
    assertEquals(spell(model, bpeEncode(model, "abc")), ["a", "bc"]);
  });

  it("when the same rank stands at several positions, the left one wins", () => {
    // "aaa" has ("a","a") rank 0 at both position 0 and 1. If the left wins, [aa, a] → ("aa","a")
    // folds all the way to [aaa]. Taking the right first stops at [a, aa] (("a","aa") is not a
    // rule).
    const model = buildModel(
      ["a", "aa", "aaa"],
      [
        ["a", "a"],
        ["aa", "a"],
      ],
    );
    assertEquals(spell(model, bpeEncode(model, "aaa")), ["aaa"]);
  });

  it("stale entries at positions changed by a merge are discarded", () => {
    // When ("y","z") rank 0 collapses first, the partner of ("x","y") rank 1 queued at position 0
    // is no longer `y`. Without matching by identity, `x` and `yz` would be folded as `xy`.
    const model = buildModel(
      ["x", "y", "z", "yz", "xy"],
      [
        ["y", "z"],
        ["x", "y"],
      ],
    );
    assertEquals(spell(model, bpeEncode(model, "xyz")), ["x", "yz"]);
  });

  it("entries at vanished positions are skipped", () => {
    // ("a","b") → ab removes position 1, but ("b","c") is still in the queue.
    const model = buildModel(
      ["a", "b", "c", "d", "ab", "cd", "bc", "abcd"],
      [
        ["a", "b"],
        ["c", "d"],
        ["b", "c"],
        ["ab", "cd"],
      ],
    );
    assertEquals(spell(model, bpeEncode(model, "abcd")), ["abcd"]);
  });
});

describe("byte_fallback: characters not in the vocabulary split into UTF-8 bytes", () => {
  let model: BpeModel;
  beforeAll(() => {
    model = buildModel(["a"]);
  });

  it("a multi-byte character becomes one symbol per byte", () => {
    assertEquals(spell(model, bpeEncode(model, "あ")), ["<0xE3>", "<0x81>", "<0x82>"]);
  });

  it("characters in the vocabulary are not split", () => {
    assertEquals(spell(model, bpeEncode(model, "aあ")).slice(0, 1), ["a"]);
  });

  it("an empty chunk gives an empty id sequence", () => {
    assertEquals(bpeEncode(model, ""), []);
  });

  it("rejects an unpaired surrogate (a value the reference's str cannot hold)", () => {
    assertThrows(() => bpeEncode(model, "\ud800"), Error, "surrogate");
  });
});

test("merge queue: the complexity is close to linear on long text (an O(n²) full scan fails)", () => {
  // A vocabulary that doubles by powers of two is synthesised, and 65,536 copies of one Japanese
  // character are fed as a single chunk. That is 65,535 merges, and the naive "scan every adjacent
  // pair → splice the minimum rank" walks the symbol sequence at each merge, a search on the order
  // of 10^9 steps (minutes). A merge queue needs pushes and pops of a constant multiple of the
  // symbol count.
  //
  // MUST: this check is not vacuous — rewriting the implementation back to a full scan reliably
  // fails it (the time budget keeps a margin of more than 100× the measurement, so machine speed
  // differences do not fail it).
  const LEVELS = 16;
  const pieces = Array.from({ length: LEVELS + 1 }, (_, level) => "あ".repeat(2 ** level));
  const merges = Array.from(
    { length: LEVELS },
    (_, level) => [pieces[level], pieces[level]] as const,
  );
  const model = buildModel(pieces, merges);

  const started = performance.now();
  const ids = bpeEncode(model, "あ".repeat(2 ** LEVELS));
  const elapsed = performance.now() - started;

  assertEquals(spell(model, ids), [pieces[LEVELS]]);
  const BUDGET_MS = 5_000;
  if (elapsed > BUDGET_MS) {
    throw new Error(
      `merging ${2 ** LEVELS} symbols took ${elapsed.toFixed(0)}ms (budget ${BUDGET_MS}ms) — ` +
        "check whether it regressed to a full scan",
    );
  }
});

/** Synthesises a decode source (byte vocabulary = `<0xHH>`, ids in `skip` return no spelling). */
const buildSource = (
  tokens: ReadonlyMap<number, string>,
  skip: ReadonlySet<number> = new Set(),
) => ({
  byteOf: (id: number): number | undefined => {
    const token = tokens.get(id);
    if (token === undefined || !/^<0x[0-9A-F]{2}>$/.test(token)) return undefined;
    return Number.parseInt(token.slice(3, 5), 16);
  },
  textOf: (id: number): string | undefined => {
    if (skip.has(id)) return undefined;
    const token = tokens.get(id);
    if (token === undefined) throw new Error(`decode: id ${id} is not in the vocabulary`);
    return token;
  },
});

/** A decode table that assigns `<0xHH>` to id 200+byte and the ordinary spellings to 0..9. */
const DECODE_TOKENS = new Map<number, string>([
  [0, "hello"],
  [1, "<eos>"],
  [2, "world"],
  ...Array.from({ length: 256 }, (_, byte) => [200 + byte, byteToken(byte)] as const),
]);

describe("incremental decode: holds back no token except in a byte run", () => {
  it("a non-byte token is emitted as soon as it is pushed", () => {
    // A "fake streaming" that holds everything until `finish()` fails here (ADR 0084 decision 4).
    const detokenizer = new StreamingDetokenizer(buildSource(DECODE_TOKENS));
    assertEquals(detokenizer.push(0), "hello");
    assertEquals(detokenizer.push(2), "world");
    assertEquals(detokenizer.finish(), "");
  });

  it("a byte run is not emitted until it closes (close = a non-byte token or finish)", () => {
    // It cannot be "emit once three bytes are in". The source of truth folds the whole run as one
    // unit, so a following byte can turn **even the valid leading part** into replacement
    // characters (the section on how runs fold below).
    const detokenizer = new StreamingDetokenizer(buildSource(DECODE_TOKENS));
    assertEquals(detokenizer.push(200 + 0xe3), "");
    assertEquals(detokenizer.push(200 + 0x81), "");
    assertEquals(detokenizer.push(200 + 0x82), "");
    assertEquals(detokenizer.finish(), "あ");
  });

  it("is emitted together with the non-byte token right after the run", () => {
    const detokenizer = new StreamingDetokenizer(buildSource(DECODE_TOKENS));
    detokenizer.push(200 + 0xe3);
    detokenizer.push(200 + 0x81);
    detokenizer.push(200 + 0x82);
    assertEquals(detokenizer.push(0), "あhello");
  });
});

describe("incremental decode: byte runs fold the way the reference's String::from_utf8 does", () => {
  let source: ReturnType<typeof buildSource>;
  beforeAll(() => {
    source = buildSource(DECODE_TOKENS);
  });

  it("a run that is invalid as a whole gives one replacement character per byte", () => {
    // It does not take the form "extract only the valid leading part".
    assertEquals(detokenize(source, [200 + 0xe3, 200 + 0x81]), "��");
  });

  it("a run split by a non-byte token is folded separately", () => {
    assertEquals(detokenize(source, [200 + 0xe3, 200 + 0x81, 0, 200 + 0x82]), "��hello�");
  });

  it("a BOM is not stripped (TextDecoder drops it by default)", () => {
    // Removing `ignoreBOM: true` makes only the `EF BB BF` run vanish quietly.
    assertEquals(detokenize(source, [200 + 0xef, 200 + 0xbb, 200 + 0xbf]), "﻿");
  });

  it("a trailing run is emitted at finish", () => {
    const detokenizer = new StreamingDetokenizer(source);
    detokenizer.push(200 + 0x41);
    assertEquals(detokenizer.finish(), "A");
  });
});

test("incremental decode: a skipped special token does not break a byte run", () => {
  // The source of truth skips at the stage that builds "id sequence → spelling sequence", and the
  // decoder chain runs after that. Breaking the run would turn a character into two replacement
  // characters just because a special token sat in between.
  const source = buildSource(DECODE_TOKENS, new Set([1]));
  assertEquals(detokenize(source, [200 + 0xe3, 1, 200 + 0x81, 200 + 0x82]), "あ");
});

test("incremental decode: rejects an id not in the vocabulary (the source of truth silently skips it)", () => {
  assertThrows(
    () => detokenize(buildSource(DECODE_TOKENS), [999]),
    Error,
    "is not in the vocabulary",
  );
});
