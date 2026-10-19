/**
 * fixture-notes: a change to a note string (`_doc` / `why` / `note`) is counted; any other change
 * is a structural difference; a note that still holds Japanese is reported.
 */

import { assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { compareNotes } from "../fixture-notes.ts";

const fixture = (why: string, value: number): string =>
  JSON.stringify({ _doc: ["注記", "second"], cases: [{ id: "a", why, value, text: "日本語" }] });

describe("compareNotes", () => {
  it("counts changed notes and reports the ones that still hold Japanese", () => {
    const report = compareNotes(fixture("理由", 1), fixture("the reason", 1));
    assertEquals(report.changed, 1);
    assertEquals(report.japaneseLeft, ["$._doc[0]: 注記"]);
    assertEquals(report.structuralDifference, undefined);
  });

  it("leaves data strings outside the note keys alone", () => {
    const report = compareNotes(fixture("理由", 1), fixture("理由", 1));
    assertEquals(report.changed, 0);
    assertEquals(report.japaneseLeft.length, 2);
  });

  it("reports a changed value, data string or key order as a structural difference", () => {
    assertEquals(
      compareNotes(fixture("x", 1), fixture("x", 2)).structuralDifference,
      "$.cases[0].value",
    );
    const dataChanged = fixture("x", 1).replace('"text":"日本語"', '"text":"Japanese"');
    assertEquals(
      compareNotes(fixture("x", 1), dataChanged).structuralDifference,
      "$.cases[0].text",
    );
    const reordered = JSON.stringify({ cases: [], _doc: [] });
    assertEquals(
      compareNotes(JSON.stringify({ _doc: [], cases: [] }), reordered).structuralDifference,
      "$ (keys)",
    );
  });
});
