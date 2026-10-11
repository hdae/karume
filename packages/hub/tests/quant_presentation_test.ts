import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { ManifestFormatError, parseManifest } from "../mod.ts";
import { FETCH_MANIFEST_TEXT } from "./helpers/fixture.ts";

// quant の表示欄（label / description — ADR 0075）の受理集合は hub の parser が正本
// （2026-10-11 裁定）。表 tests/fixtures/quant-presentation.json は Python の `karume dist`
// （tools/exporter/tests/test_dist.py）も同じ 1 ファイルで回す — 片側だけ動かすと両方の
// どちらかが赤になる。ここは表を全件回すだけで、TS 側に第 2 の定義を作らない。

type PresentationCase = {
  readonly name: string;
  readonly field: string;
  readonly hasValue: boolean;
  readonly value: unknown;
  readonly accepted: boolean;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const loadCases = async (): Promise<readonly PresentationCase[]> => {
  const raw: unknown = JSON.parse(
    await Deno.readTextFile(new URL("./fixtures/quant-presentation.json", import.meta.url)),
  );
  if (!isRecord(raw) || !Array.isArray(raw["cases"])) throw new Error("fixture: cases が無い");
  const cases = raw["cases"].map((entry: unknown): PresentationCase => {
    if (
      !isRecord(entry) || typeof entry["name"] !== "string" ||
      typeof entry["field"] !== "string" || typeof entry["accepted"] !== "boolean"
    ) {
      throw new Error(`fixture: ケースの形が不正 ${JSON.stringify(entry)}`);
    }
    return {
      name: entry["name"],
      field: entry["field"],
      hasValue: Object.hasOwn(entry, "value"),
      value: entry["value"],
      accepted: entry["accepted"],
    };
  });
  if (cases.length === 0) throw new Error("fixture: ケースが空");
  return cases;
};

const cases = await loadCases();

/** 正当な manifest の、既定モデルの最初の quant 席に表示欄 1 つを書いた（または消した）綴り。 */
const manifestWith = (testCase: PresentationCase): string => {
  const manifest: unknown = JSON.parse(FETCH_MANIFEST_TEXT);
  assert(isRecord(manifest) && isRecord(manifest["models"]), "fixture: manifest の形");
  const model = manifest["models"][String(manifest["defaultModel"])];
  assert(isRecord(model) && isRecord(model["quants"]), "fixture: 既定モデルの quants");
  const quant = Object.values(model["quants"])[0];
  assert(isRecord(quant), "fixture: quant 席");
  delete quant[testCase.field];
  if (testCase.hasValue) quant[testCase.field] = testCase.value;
  return JSON.stringify(manifest);
};

describe("quant の表示欄の受理集合（共有表 quant-presentation.json）", () => {
  it("表は label と description の両方の欄を覆う", () => {
    assertEquals([...new Set(cases.map((entry) => entry.field))].sort(), [
      "description",
      "label",
    ]);
  });

  for (const testCase of cases) {
    if (testCase.accepted) {
      it(`${testCase.name}: 受理し、書いた値をそのまま返す`, () => {
        const manifest = parseManifest(manifestWith(testCase));
        const quant = Object.values(manifest.models[manifest.defaultModel].quants)[0];
        const read = testCase.field === "label" ? quant.label : quant.description;
        assertEquals(read, testCase.hasValue ? testCase.value : undefined);
      });
    } else {
      it(`${testCase.name}: ManifestFormatError で拒否する`, () => {
        assertThrows(() => parseManifest(manifestWith(testCase)), ManifestFormatError);
      });
    }
  }
});
