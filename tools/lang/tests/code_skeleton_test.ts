/**
 * code-skeleton: a comment-only change keeps the skeleton; any change to code, strings or WGSL text
 * changes it.
 */

import { assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { commentSpansInTemplate, skeletonOf } from "../code-skeleton.ts";

const BT = "`";

describe("skeletonOf", () => {
  it("is the same when only comments and whitespace differ", () => {
    const before = "// 日本語の注記\nexport const x = 1; /* 補足 */\n";
    const after = "// English note\nexport   const x = 1;\n";
    assertEquals(skeletonOf("a.ts", before), skeletonOf("a.ts", after));
  });

  it("is the same when only comments inside a template literal differ (WGSL)", () => {
    const wgsl = (note: string): string =>
      `export const w = ${BT}// ${note}\nfn f() -> u32 { /* ${note} */ return 1u; }\n${BT};\n`;
    assertEquals(skeletonOf("w.ts", wgsl("行の注記")), skeletonOf("w.ts", wgsl("line note")));
  });

  it("handles template literals with substitutions", () => {
    const source = (note: string): string =>
      `const k = 4;\nexport const w = ${BT}// ${note}\nconst N = \${k}u; // ${note}\n${BT};\n`;
    assertEquals(skeletonOf("w.ts", source("注")), skeletonOf("w.ts", source("note")));
  });

  it("differs when the code changes", () => {
    assertNotEquals(
      skeletonOf("a.ts", "export const x = 1;\n"),
      skeletonOf("a.ts", "export const x = 2;\n"),
    );
  });

  it("differs when a string literal changes (an error message is not a comment)", () => {
    const before = 'throw new Error("日本語の文言");\n';
    const after = 'throw new Error("English message");\n';
    assertNotEquals(skeletonOf("e.ts", before), skeletonOf("e.ts", after));
  });

  it("differs when template text that is not a comment changes", () => {
    const before = `export const w = ${BT}let x = 1u; // same\n${BT};\n`;
    const after = `export const w = ${BT}let x = 2u; // same\n${BT};\n`;
    assertNotEquals(skeletonOf("w.ts", before), skeletonOf("w.ts", after));
  });

  it("keeps a URL inside a template literal (// after a colon is not a comment)", () => {
    const before = `export const u = ${BT}https://example.com/a${BT};\n`;
    const after = `export const u = ${BT}https://example.com/b${BT};\n`;
    assertNotEquals(skeletonOf("u.ts", before), skeletonOf("u.ts", after));
  });

  it("throws when the source does not parse", () => {
    assertThrows(() => skeletonOf("bad.ts", "const = ;\n"), Error, "cannot parse bad.ts");
  });
});

describe("commentSpansInTemplate", () => {
  it("finds line and block comments with the given offset", () => {
    const text = "a // b\nc /* d */ e";
    assertEquals(commentSpansInTemplate(text, 0), [
      [2, 6],
      [9, 16],
    ]);
    assertEquals(commentSpansInTemplate(text, 10), [
      [12, 16],
      [19, 26],
    ]);
  });

  it("runs an unterminated block comment to the end of the text", () => {
    assertEquals(commentSpansInTemplate("x /* open", 0), [[2, 9]]);
  });
});
