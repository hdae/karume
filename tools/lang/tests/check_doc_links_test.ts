/**
 * check-doc-links: relative links and `DECIDED:` pointers resolve to files and headings; the rest is
 * reported with its reason.
 */

import { assertEquals } from "@std/assert";
import { afterAll, beforeAll, describe, it } from "vite-plus/test";
import { makeTempDir, mkdir, remove, writeTextFile } from "@karume/test-support/fs";
import {
  checkLinks,
  decidedPointers,
  headingSlugs,
  linkTargets,
  slugOf,
} from "../check-doc-links.ts";

const MARKER = ["DECIDED", ":"].join("");

describe("slugOf / headingSlugs", () => {
  it("follows GitHub's rule, including Japanese headings and duplicates", () => {
    assertEquals(slugOf("Second heading"), "second-heading");
    assertEquals(slugOf("追記（2026-10-18）— deferred"), "追記2026-10-18-deferred");
    assertEquals(slugOf("`code` and *emphasis*"), "code-and-emphasis");
    const slugs = headingSlugs(
      "# Top\n\n## Same\n\ntext\n\n## Same\n\n```\n# not a heading\n```\n",
    );
    assertEquals([...slugs], ["top", "same", "same-1"]);
  });
});

describe("linkTargets / decidedPointers", () => {
  it("collects inline links and reference definitions outside fences", () => {
    const targets = linkTargets(
      'see [a](b.md) and ![i](img.png "t")\n[ref]: c.md#x\n```\n[no](d.md)\n```\n',
    );
    assertEquals(
      targets.map((t) => t.target),
      ["b.md", "img.png", "c.md#x"],
    );
  });

  it("extracts decision-record paths and ADR numbers from DECIDED: lines only", () => {
    // The marker is assembled so that the repository-wide check does not read this test as a pointer.
    const source = `// ${MARKER} docs/decisions/0042-plan.md#追記 and ADR 0058 決定 2\n// see ADR 0001 (not a pointer)\n`;
    assertEquals(
      decidedPointers(source).map((t) => t.target),
      ["docs/decisions/0042-plan.md#追記", "adr:0058"],
    );
  });
});

describe("checkLinks", () => {
  let root = "";

  beforeAll(async () => {
    root = await makeTempDir();
    await mkdir(`${root}/docs/decisions`, { recursive: true });
    await mkdir(`${root}/src`, { recursive: true });
    await writeTextFile(
      `${root}/docs/decisions/0001-first.md`,
      "# First\n\n## Second heading\n\n## 追記（2026-10-18）\n",
    );
    await writeTextFile(
      `${root}/docs/a.md`,
      [
        "# A",
        "[ok](decisions/0001-first.md)",
        "[ok anchor](decisions/0001-first.md#second-heading)",
        "[ok jp anchor](decisions/0001-first.md#追記2026-10-18)",
        "[self](#a)",
        "[web](https://example.com/x)",
        "[asset](../outputs/x.png)",
        "[gone](decisions/missing.md)",
        "[bad anchor](decisions/0001-first.md#nope)",
      ].join("\n"),
    );
    await writeTextFile(
      `${root}/src/x.ts`,
      `// ${MARKER} ADR 0001\n// ${MARKER} docs/decisions/0001-first.md#first\n// ${MARKER} ADR 0999\n`,
    );
  });

  afterAll(async () => {
    await remove(root, { recursive: true });
  });

  it("reports the missing file, the missing heading and the unknown ADR, and skips local assets", () => {
    const report = checkLinks(root, ["docs/a.md"], ["src/x.ts"]);
    assertEquals(report.skipped, 1);
    assertEquals(
      report.broken.map((b) => `${b.file}:${b.line} ${b.reason} ${b.target}`),
      [
        "docs/a.md:8 missing file decisions/missing.md",
        "docs/a.md:9 missing heading decisions/0001-first.md#nope",
        "src/x.ts:3 unknown ADR adr:0999",
      ],
    );
    // 5 Markdown links checked (web skipped by scheme, asset skipped) + 3 pointers.
    assertEquals(report.checked, 9);
  });
});
