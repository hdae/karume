import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import {
  DimError,
  evalDim,
  formatDim,
  isSymbolName,
  parseDim,
  solveDim,
  tryParseDim,
} from "../src/format/dims.ts";

// The source of truth for the grammar is tests/fixtures/dim-grammar.json (docs/ir-v2.md). This file
// only runs every row of that table and does not build a second definition of the accepted set in
// TS.

type ValidCase = {
  readonly text: string;
  readonly coeff: number;
  readonly sym: string;
  readonly offset: number;
};
type EvalCase = {
  readonly expr: string;
  readonly bindings: Record<string, number>;
  readonly value?: number;
  readonly throws: boolean;
};
type Fixture = {
  readonly valid: readonly ValidCase[];
  readonly invalid: readonly string[];
  readonly eval: readonly EvalCase[];
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const section = (root: Record<string, unknown>, key: string): readonly unknown[] => {
  const value = root[key];
  if (!Array.isArray(value)) throw new Error(`fixture: section '${key}' is not an array`);
  const list: readonly unknown[] = value;
  if (list.length === 0) throw new Error(`fixture: section '${key}' is empty`);
  return list;
};

const asValidCase = (raw: unknown): ValidCase => {
  if (
    !isRecord(raw) ||
    typeof raw["text"] !== "string" ||
    typeof raw["coeff"] !== "number" ||
    typeof raw["sym"] !== "string" ||
    typeof raw["offset"] !== "number"
  ) {
    throw new Error(`fixture: malformed valid case ${JSON.stringify(raw)}`);
  }
  return { text: raw["text"], coeff: raw["coeff"], sym: raw["sym"], offset: raw["offset"] };
};

const asEvalCase = (raw: unknown): EvalCase => {
  if (!isRecord(raw) || typeof raw["expr"] !== "string" || !isRecord(raw["bindings"])) {
    throw new Error(`fixture: malformed eval case ${JSON.stringify(raw)}`);
  }
  const bindings: Record<string, number> = {};
  for (const [symbol, bound] of Object.entries(raw["bindings"])) {
    if (typeof bound !== "number") {
      throw new Error(`fixture: binding ${symbol} is not a number`);
    }
    bindings[symbol] = bound;
  }
  const throws = raw["throws"] === true;
  const value = raw["value"];
  if (throws === (typeof value === "number")) {
    throw new Error(
      `fixture: an eval case has exactly one of value or throws ${JSON.stringify(raw)}`,
    );
  }
  return {
    expr: raw["expr"],
    bindings,
    value: typeof value === "number" ? value : undefined,
    throws,
  };
};

const loadFixture = async (): Promise<Fixture> => {
  const text = await readTextFile(new URL("./fixtures/dim-grammar.json", import.meta.url));
  const raw: unknown = JSON.parse(text);
  if (!isRecord(raw)) throw new Error("fixture: not an object");
  return {
    valid: section(raw, "valid").map(asValidCase),
    invalid: section(raw, "invalid").map((item) => {
      if (typeof item !== "string")
        throw new Error("fixture: the invalid section is a list of strings");
      return item;
    }),
    eval: section(raw, "eval").map(asEvalCase),
  };
};

test("dim-grammar valid: canonical notation parses, and the round trip with format is string-identical", async () => {
  const fixture = await loadFixture();
  for (const testCase of fixture.valid) {
    const expected = { coeff: testCase.coeff, sym: testCase.sym, offset: testCase.offset };
    assertEquals(parseDim(testCase.text), expected, `parseDim('${testCase.text}')`);
    assertEquals(formatDim(expected), testCase.text, `formatDim → '${testCase.text}'`);
    assert(isSymbolName(testCase.sym), `symbol name '${testCase.sym}'`);
  }
});

test("dim-grammar invalid: non-canonical and non-matching text is rejected", async () => {
  const fixture = await loadFixture();
  for (const text of fixture.invalid) {
    assertEquals(tryParseDim(text), undefined, `tryParseDim('${text}')`);
    assertThrows(() => parseDim(text), DimError, undefined, `parseDim('${text}')`);
  }
});

test("dim-grammar eval: evaluation against binding tables (prototype-pollution cases included)", async () => {
  const fixture = await loadFixture();
  for (const testCase of fixture.eval) {
    const expr = parseDim(testCase.expr);
    if (testCase.throws) {
      assertThrows(() => evalDim(expr, testCase.bindings), DimError, undefined, testCase.expr);
      continue;
    }
    assertEquals(evalDim(expr, testCase.bindings), testCase.value, testCase.expr);
  }
});

test("evalDim: a value reached through the prototype chain is not treated as a binding", () => {
  // A case JSON cannot express (inherited rather than an own property). Only a hasOwn check rejects
  // it.
  const inherited: Record<string, number> = Object.create({ T: 5 });
  assertThrows(() => evalDim(parseDim("T"), inherited), DimError);
  assertEquals(
    inherited["T"],
    5,
    "the inherited value itself is readable (an === undefined check lets it through)",
  );
});

test("formatDim: a non-canonical decomposition is not assembled", () => {
  assertThrows(() => formatDim({ coeff: 0, sym: "T", offset: 0 }), DimError);
  assertThrows(() => formatDim({ coeff: 1, sym: "T", offset: -1 }), DimError);
  assertThrows(() => formatDim({ coeff: 1, sym: "2T", offset: 0 }), DimError);
});

test("evalDim: fails when the result exceeds the safe-integer range", () => {
  assertThrows(() => evalDim({ coeff: 8, sym: "T", offset: 0 }, { T: 2 ** 52 }), DimError);
});

test("solveDim: running the eval rows of the conformance table backwards returns the same bindings", async () => {
  // Checks that it is the inverse of `evalDim` on the grammar's source-of-truth table itself (no
  // second table).
  const fixture = await loadFixture();
  for (const { expr: text, bindings, value } of fixture.eval) {
    if (value === undefined) continue;
    const expr = parseDim(text);
    assertEquals(solveDim(expr, value), bindings[expr.sym], `${text} = ${value}`);
  }
});

test("solveDim: an actual size that does not have the declared form has no solution (no rounding)", () => {
  // This is where the vowel-detection CRNN's "the 10 ms frame count is even" fails (ADR 0057
  // decision 2).
  assertEquals(solveDim(parseDim("2T"), 285), undefined);
  assertEquals(solveDim(parseDim("2T"), 284), 142);
  // An actual size below the offset (7 for `T+8`) and negative or non-integer actual sizes.
  assertEquals(solveDim(parseDim("T+8"), 7), undefined);
  assertEquals(solveDim(parseDim("T+8"), 8), 0);
  assertEquals(solveDim(parseDim("T"), -1), undefined);
  assertEquals(solveDim(parseDim("T"), 1.5), undefined);
});
