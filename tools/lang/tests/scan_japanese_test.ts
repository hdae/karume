/**
 * scan-japanese: lines with Japanese text are counted per file and summed per area; binary files
 * are skipped.
 */

import { assertEquals } from "@std/assert";
import { afterAll, beforeAll, describe, it } from "vite-plus/test";
import { writeFileSync } from "node:fs";
import { makeTempDir, mkdir, remove, writeTextFile } from "@karume/test-support/fs";
import { areaOf, countJapaneseLines, scanFiles, totalsByArea } from "../scan-japanese.ts";

describe("countJapaneseLines", () => {
  it("counts a line once however many Japanese characters it has", () => {
    assertEquals(countJapaneseLines("abc\nあい\n漢字 x 漢字\n、\nkatakana カナ\n"), 4);
    assertEquals(countJapaneseLines("only ascii\n"), 0);
  });
});

describe("areaOf", () => {
  it("maps paths to reporting areas", () => {
    const cases: readonly (readonly [string, string])[] = [
      ["docs/decisions/0001-a.md", "docs/decisions"],
      ["docs/research/x.md", "docs/research"],
      ["docs/GLOSSARY.ja.md", "docs/GLOSSARY.ja.md"],
      ["docs/backlog.md", "docs (other)"],
      ["packages/runtime/src/a.ts", "packages/runtime/src"],
      ["packages/runtime/tests/fixtures/x.json", "packages/runtime/tests/fixtures"],
      ["packages/runtime/tests/a_test.ts", "packages/runtime/tests"],
      ["packages/e2e/README.md", "packages/e2e"],
      ["tools/exporter/src/a.py", "tools/exporter"],
      ["tools/exporter/tests/t.py", "tools/exporter/tests"],
      ["tools/export-recipes/wan/x.py", "tools/export-recipes"],
      ["tools/lang/scan-japanese.ts", "tools/lang"],
      ["CLAUDE.md", "root"],
      [".claude/ACTIVE_DESIGN.md", ".claude"],
    ];
    for (const [file, area] of cases) assertEquals(areaOf(file), area, file);
  });
});

describe("scanFiles / totalsByArea", () => {
  let root = "";

  beforeAll(async () => {
    root = await makeTempDir();
    await mkdir(`${root}/docs/decisions`, { recursive: true });
    await mkdir(`${root}/packages/runtime/src`, { recursive: true });
    await writeTextFile(`${root}/docs/decisions/0001-a.md`, "# 題\n\n本文\n\nplain\n");
    await writeTextFile(`${root}/docs/decisions/0002-b.md`, "# English only\n");
    await writeTextFile(`${root}/packages/runtime/src/a.ts`, "// 注記\nexport const a = 1;\n");
    writeFileSync(
      `${root}/packages/runtime/src/blob.bin`,
      Buffer.from([0x41, 0x00, 0xe3, 0x81, 0x82]),
    );
  });

  afterAll(async () => {
    await remove(root, { recursive: true });
  });

  it("lists only files with Japanese text and skips binaries", () => {
    const entries = scanFiles(root, [
      "docs/decisions/0001-a.md",
      "docs/decisions/0002-b.md",
      "packages/runtime/src/a.ts",
      "packages/runtime/src/blob.bin",
      "packages/runtime/src/missing.ts",
    ]);
    assertEquals(entries, [
      { file: "docs/decisions/0001-a.md", lines: 2 },
      { file: "packages/runtime/src/a.ts", lines: 1 },
    ]);
    assertEquals(totalsByArea(entries), [
      { area: "docs/decisions", files: 1, lines: 2 },
      { area: "packages/runtime/src", files: 1, lines: 1 },
    ]);
  });
});
