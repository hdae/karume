import { readTextFile } from "@karume/test-support/fs";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { ManifestFormatError, parseManifest } from "../mod.ts";
import { FETCH_MANIFEST_TEXT } from "./helpers/fixture.ts";

// The hub's parser is the source of truth for the accepted values of a quant's display fields
// (label / description — ADR 0075) (ruling of 2026-10-11). The table
// tests/fixtures/quant-presentation.json is also run by Python's `karume dist`
// (tools/exporter/tests/test_dist.py) from the same single file — changing only one side turns one
// of the two red. This file only runs every row of the table and creates no second definition on
// the TS side.

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
    await readTextFile(new URL("./fixtures/quant-presentation.json", import.meta.url)),
  );
  if (!isRecord(raw) || !Array.isArray(raw["cases"])) throw new Error("fixture: cases missing");
  const cases = raw["cases"].map((entry: unknown): PresentationCase => {
    if (
      !isRecord(entry) ||
      typeof entry["name"] !== "string" ||
      typeof entry["field"] !== "string" ||
      typeof entry["accepted"] !== "boolean"
    ) {
      throw new Error(`fixture: malformed case ${JSON.stringify(entry)}`);
    }
    return {
      name: entry["name"],
      field: entry["field"],
      hasValue: Object.hasOwn(entry, "value"),
      value: entry["value"],
      accepted: entry["accepted"],
    };
  });
  if (cases.length === 0) throw new Error("fixture: no cases");
  return cases;
};

const cases = await loadCases();

/**
 * A valid manifest with one display field written (or removed) in the default model's first quant.
 */
const manifestWith = (testCase: PresentationCase): string => {
  const manifest: unknown = JSON.parse(FETCH_MANIFEST_TEXT);
  assert(isRecord(manifest) && isRecord(manifest["models"]), "fixture: manifest shape");
  const model = manifest["models"][String(manifest["defaultModel"])];
  assert(isRecord(model) && isRecord(model["quants"]), "fixture: quants of the default model");
  const quant = Object.values(model["quants"])[0];
  assert(isRecord(quant), "fixture: quant entry");
  delete quant[testCase.field];
  if (testCase.hasValue) quant[testCase.field] = testCase.value;
  return JSON.stringify(manifest);
};

describe("accepted values of a quant's display fields (shared table quant-presentation.json)", () => {
  it("the table covers both the label and the description field", () => {
    assertEquals([...new Set(cases.map((entry) => entry.field))].sort(), ["description", "label"]);
  });

  for (const testCase of cases) {
    if (testCase.accepted) {
      it(`${testCase.name}: accepts it and returns the written value unchanged`, () => {
        const manifest = parseManifest(manifestWith(testCase));
        const quant = Object.values(manifest.models[manifest.defaultModel].quants)[0];
        const read = testCase.field === "label" ? quant.label : quant.description;
        assertEquals(read, testCase.hasValue ? testCase.value : undefined);
      });
    } else {
      it(`${testCase.name}: rejects it with ManifestFormatError`, () => {
        assertThrows(() => parseManifest(manifestWith(testCase)), ManifestFormatError);
      });
    }
  }
});
