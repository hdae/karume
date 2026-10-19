/**
 * code-skeleton: a comment-only change keeps the skeleton; any change to code, strings or WGSL text
 * changes it. In strings mode the literals are lifted out and listed, and a changed literal that
 * held no Japanese is a code change.
 */

import { assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { analyze, commentSpansInTemplate, literalChanges, skeletonOf } from "../code-skeleton.ts";

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

  // A WGSL comment that an interpolation interrupts ("// tile (${rows} rows × ${cols} cols)") is
  // still a comment after the interpolation; the interpolated expressions stay code.
  it("treats the rest of a // comment after an interpolation as a comment", () => {
    const source = (note: string, rows: string): string =>
      `const rows = 4;\nexport const w = ${BT}// ${note} (\${${rows}} rows) ${note}\nfn f() {}\n${BT};\n`;
    assertEquals(
      skeletonOf("w.ts", source("注記", "rows")),
      skeletonOf("w.ts", source("note", "rows")),
    );
    assertNotEquals(
      skeletonOf("w.ts", source("note", "rows")),
      skeletonOf("w.ts", source("note", "rows + 1")),
    );
  });

  it("treats the rest of a block comment after an interpolation as a comment", () => {
    const source = (note: string): string =>
      `const k = 4;\nexport const w = ${BT}/* ${note} \${k} ${note} */ fn f() {}\n${BT};\n`;
    assertEquals(skeletonOf("w.ts", source("注")), skeletonOf("w.ts", source("note")));
  });
});

describe("analyze in strings mode", () => {
  const strings = (source: string) => analyze("s.ts", source, "strings");

  it("lifts string and template literals out of the skeleton, in source order, with WGSL comments removed", () => {
    const source = `throw new Error("日本語の文言"); const w = ${BT}let x = 1u; // 注記\n${BT}; const k = 'key';\n`;
    const result = strings(source);
    assertEquals(result.literals, ["日本語の文言", "let x = 1u;", "key"]);
    assertEquals(result.text, `throw new Error(\u{1});const w = ${BT}\u{1}${BT};const k = \u{1};`);
  });

  it("keeps the interpolated expressions of a template in the skeleton and marks them in the cell", () => {
    const source = (note: string) => `const m = ${BT}${note} \${count} 個${BT};\n`;
    const before = strings(source("合計"));
    const after = strings(source("total"));
    assertEquals(before.text, after.text);
    assertEquals(before.literals, ["合計 ${} 個"]);
    assertEquals(
      strings(`const m = ${BT}合計 \${count + 1} 個${BT};\n`).text !== before.text,
      true,
    );
  });

  it("ignores a formatter wrap (whitespace next to punctuation, a trailing comma) in strings mode", () => {
    const before = 'f(\n  a,\n  "x",\n);\nconst o = { k: 1, };\n';
    const after = 'f(a, "x");\nconst o = { k: 1 };\n';
    assertEquals(strings(before).text, strings(after).text);
    assertNotEquals(skeletonOf("t.ts", before), skeletonOf("t.ts", after));
    assertNotEquals(strings("f(a, b);\n").text, strings("f(a + b);\n").text);
  });

  it("lists the literals that changed and whether the old text held Japanese", () => {
    const before = strings('f("日本語", "same", "english");\n');
    const after = strings('f("Japanese", "same", "English");\n');
    assertEquals(literalChanges(before, after), [
      { index: 0, before: "日本語", after: "Japanese", japaneseBefore: true },
      { index: 2, before: "english", after: "English", japaneseBefore: false },
    ]);
  });

  it("treats a WGSL comment edit as no literal change and a WGSL token edit as a change without Japanese", () => {
    const wgsl = (comment: string, code: string) =>
      strings(`export const w = ${BT}// ${comment}\nfn f() { ${code} }\n${BT};\n`);
    assertEquals(literalChanges(wgsl("注記", "return;"), wgsl("note", "return;")), []);
    const [change] = literalChanges(wgsl("note", "return;"), wgsl("note", "return 1u;"));
    assertEquals(change.japaneseBefore, false);
  });

  it("refuses to align literals when the skeletons differ", () => {
    assertThrows(
      () => literalChanges(strings('f("a");\n'), strings('g("a");\n')),
      Error,
      "the skeletons differ",
    );
    assertThrows(
      () => literalChanges(strings('f("a");\n'), strings('f("a", "b");\n')),
      Error,
      "the skeletons differ",
    );
  });
});

describe("commentSpansInTemplate", () => {
  it("finds line and block comments with the given offset", () => {
    const text = "a // b\nc /* d */ e";
    assertEquals(commentSpansInTemplate(text, 0).spans, [
      [2, 6],
      [9, 16],
    ]);
    assertEquals(commentSpansInTemplate(text, 10).spans, [
      [12, 16],
      [19, 26],
    ]);
  });

  it("runs an unterminated block comment to the end of the text and reports the carry", () => {
    assertEquals(commentSpansInTemplate("x /* open", 0), { spans: [[2, 9]], carry: "block" });
    assertEquals(commentSpansInTemplate("// to the end", 0).carry, "line");
    assertEquals(commentSpansInTemplate(" still comment\ncode", 0, "line"), {
      spans: [[0, 14]],
      carry: "none",
    });
  });
});
