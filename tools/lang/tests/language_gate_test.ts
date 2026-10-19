/**
 * language-gate: a file with Japanese is red unless an allowance covers it; an exact allowance wins
 * over a prefix and the longest prefix wins; a budget caps the lines; an allowance that covers
 * nothing is stale.
 */

import { assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { type Allowance, allowanceOf } from "../language-allow-list.ts";
import { describeReport, gateReport, isGreen } from "../language-gate.ts";

const LIST: readonly Allowance[] = [
  { path: "docs/research/", why: "snapshots" },
  { path: "packages/models/tests/", why: "tests", until: "0.15.x" },
  { path: "packages/models/tests/fixtures/", why: "data" },
  { path: "packages/models/tests/fixtures/notes.json", why: "one note", maxLines: 2 },
];

describe("allowanceOf", () => {
  it("prefers an exact path, then the longest prefix", () => {
    assertEquals(allowanceOf("packages/models/tests/fixtures/notes.json", LIST)?.why, "one note");
    assertEquals(allowanceOf("packages/models/tests/fixtures/vocab.json", LIST)?.why, "data");
    assertEquals(allowanceOf("packages/models/tests/a_test.ts", LIST)?.why, "tests");
    assertEquals(allowanceOf("docs/research/2026-01-01-x.md", LIST)?.why, "snapshots");
    assertEquals(allowanceOf("packages/runtime/src/ops.ts", LIST), undefined);
  });
});

describe("gateReport", () => {
  it("is green when every file is covered, within budget, and every allowance is used", () => {
    const report = gateReport(
      [
        { file: "docs/research/x.md", lines: 10 },
        { file: "packages/models/tests/a_test.ts", lines: 3 },
        { file: "packages/models/tests/fixtures/vocab.json", lines: 100 },
        { file: "packages/models/tests/fixtures/notes.json", lines: 2 },
      ],
      LIST,
    );
    assertEquals(isGreen(report), true);
    assertEquals(describeReport(report), []);
  });

  it("reports an uncovered file, a file over its budget, and a stale allowance", () => {
    const report = gateReport(
      [
        { file: "packages/runtime/src/ops.ts", lines: 1 },
        { file: "packages/models/tests/fixtures/notes.json", lines: 3 },
        { file: "packages/models/tests/a_test.ts", lines: 3 },
        { file: "packages/models/tests/fixtures/vocab.json", lines: 1 },
      ],
      LIST,
    );
    assertEquals(isGreen(report), false);
    assertEquals(
      report.uncovered.map((entry) => entry.file),
      ["packages/runtime/src/ops.ts"],
    );
    assertEquals(
      report.overBudget.map((entry) => [entry.file, entry.lines, entry.maxLines]),
      [["packages/models/tests/fixtures/notes.json", 3, 2]],
    );
    assertEquals(
      report.stale.map((allowance) => allowance.path),
      ["docs/research/"],
    );
    assertEquals(describeReport(report).length, 3);
  });
});
