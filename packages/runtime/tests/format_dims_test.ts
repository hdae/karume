import { assert, assertEquals, assertThrows } from "@std/assert";
import {
  DimError,
  evalDim,
  formatDim,
  isSymbolName,
  parseDim,
  tryParseDim,
} from "../src/format/dims.ts";

// 文法の正本は tests/fixtures/dim-grammar.json（docs/ir-v1.md）。ここは表を全件回すだけで、
// TS 側に受理集合の第 2 の定義を作らない。

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
  if (!Array.isArray(value)) throw new Error(`fixture: 節 '${key}' が配列でない`);
  const list: readonly unknown[] = value;
  if (list.length === 0) throw new Error(`fixture: 節 '${key}' が空`);
  return list;
};

const asValidCase = (raw: unknown): ValidCase => {
  if (
    !isRecord(raw) || typeof raw["text"] !== "string" || typeof raw["coeff"] !== "number" ||
    typeof raw["sym"] !== "string" || typeof raw["offset"] !== "number"
  ) {
    throw new Error(`fixture: valid ケースの形が不正 ${JSON.stringify(raw)}`);
  }
  return { text: raw["text"], coeff: raw["coeff"], sym: raw["sym"], offset: raw["offset"] };
};

const asEvalCase = (raw: unknown): EvalCase => {
  if (!isRecord(raw) || typeof raw["expr"] !== "string" || !isRecord(raw["bindings"])) {
    throw new Error(`fixture: eval ケースの形が不正 ${JSON.stringify(raw)}`);
  }
  const bindings: Record<string, number> = {};
  for (const [symbol, bound] of Object.entries(raw["bindings"])) {
    if (typeof bound !== "number") {
      throw new Error(`fixture: 束縛 ${symbol} が数値でない`);
    }
    bindings[symbol] = bound;
  }
  const throws = raw["throws"] === true;
  const value = raw["value"];
  if (throws === (typeof value === "number")) {
    throw new Error(`fixture: eval ケースは value か throws のどちらか一方 ${JSON.stringify(raw)}`);
  }
  return {
    expr: raw["expr"],
    bindings,
    value: typeof value === "number" ? value : undefined,
    throws,
  };
};

const loadFixture = async (): Promise<Fixture> => {
  const text = await Deno.readTextFile(new URL("./fixtures/dim-grammar.json", import.meta.url));
  const raw: unknown = JSON.parse(text);
  if (!isRecord(raw)) throw new Error("fixture: オブジェクトでない");
  return {
    valid: section(raw, "valid").map(asValidCase),
    invalid: section(raw, "invalid").map((item) => {
      if (typeof item !== "string") throw new Error("fixture: invalid 節は文字列の列");
      return item;
    }),
    eval: section(raw, "eval").map(asEvalCase),
  };
};

Deno.test("dim-grammar valid: 正準表記を分解でき、format との往復が文字列同一", async () => {
  const fixture = await loadFixture();
  for (const testCase of fixture.valid) {
    const expected = { coeff: testCase.coeff, sym: testCase.sym, offset: testCase.offset };
    assertEquals(parseDim(testCase.text), expected, `parseDim('${testCase.text}')`);
    assertEquals(formatDim(expected), testCase.text, `formatDim → '${testCase.text}'`);
    assert(isSymbolName(testCase.sym), `シンボル名 '${testCase.sym}'`);
  }
});

Deno.test("dim-grammar invalid: 非正準・非該当は受理しない", async () => {
  const fixture = await loadFixture();
  for (const text of fixture.invalid) {
    assertEquals(tryParseDim(text), undefined, `tryParseDim('${text}')`);
    assertThrows(() => parseDim(text), DimError, undefined, `parseDim('${text}')`);
  }
});

Deno.test("dim-grammar eval: 束縛表の評価（prototype 汚染ケースを含む）", async () => {
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

Deno.test("evalDim: prototype チェーン越しの値は束縛と見なさない", () => {
  // JSON では表現できないケース（自前プロパティではなく継承）。hasOwn 判定でのみ落ちる。
  const inherited: Record<string, number> = Object.create({ T: 5 });
  assertThrows(() => evalDim(parseDim("T"), inherited), DimError);
  assertEquals(inherited["T"], 5, "継承値自体は読める（=== undefined 判定では素通りする）");
});

Deno.test("formatDim: 非正準な分解結果は組み立てない", () => {
  assertThrows(() => formatDim({ coeff: 0, sym: "T", offset: 0 }), DimError);
  assertThrows(() => formatDim({ coeff: 1, sym: "T", offset: -1 }), DimError);
  assertThrows(() => formatDim({ coeff: 1, sym: "2T", offset: 0 }), DimError);
});

Deno.test("evalDim: 評価結果が安全整数を超えたら落とす", () => {
  assertThrows(() => evalDim({ coeff: 8, sym: "T", offset: 0 }, { T: 2 ** 52 }), DimError);
});
