/**
 * Tests of the repository lint plugin's rule `karume/typed-array-buffer` (synthesised sources are
 * written into a temporary directory at paths such as `packages/<pkg>/src/…` and run through the real
 * `vp lint` (Oxlint) with a temporary `vite.config.ts` that registers the plugin — no GPU needed).
 *
 * What is checked:
 *
 * 1. **Reported** — in the package sources, a TypedArray in a type position without its type
 *    argument (parameter, return type, type argument, array, type alias, field, `satisfies`,
 *    qualified name `globalThis.X`, interface `extends`, class `implements`), and all 12 names
 * 2. **Not reported** — types that write `<ArrayBuffer>` / `<ArrayBufferLike>`, and value positions
 *    (`new` / `from` / `instanceof` / `typeof` type queries), comments and strings. Also, the forms
 *    ADR 0002 records as missed (`InstanceType<typeof X>` / `ReturnType<typeof X.from>`) are still
 *    missed (catches a drift between the record and the code)
 * 3. **Out of scope** — `packages/<pkg>/tests/`, `tools/` and `examples/` are not reported even
 *    without a type argument
 *
 * NOTE: written as `describe` (context) + `it` (behaviour), following the repository's convention.
 */

import { execFile } from "node:child_process";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { assert, assertEquals } from "@std/assert";
import { makeTempDir, mkdir, remove, writeTextFile } from "@karume/test-support/fs";
import { describe, it } from "vite-plus/test";
import { isTargetFile } from "../oxlint-plugin.ts";

const RULE = "karume/typed-array-buffer";
/** The rule's spelling in Oxlint's JSON output (`<plugin>(<rule>)`). */
const RULE_CODE = "karume(typed-array-buffer)";
const IN_SCOPE = "packages/runtime/src/sample.ts";

const PLUGIN_PATH = fileURLToPath(new URL("../oxlint-plugin.ts", import.meta.url));
/** This workspace's Vite+ CLI (runs the `vp` version the repository pins, independent of PATH). */
const VP_BIN = join(
  dirname(createRequire(import.meta.url).resolve("vite-plus/package.json")),
  "bin",
  "vp",
);

/** The fields of the `vp lint -f json` output read here. `offset` / `length` are in UTF-8 bytes. */
type LintDiagnostic = {
  readonly message: string;
  readonly code: string;
  readonly filename: string;
  readonly labels: readonly {
    readonly span: { readonly offset: number; readonly length: number };
  }[];
};
type LintOutput = { readonly diagnostics: readonly LintDiagnostic[] };

/** A config with only this rule enabled (built-in rules off — the reports come from this rule alone). */
const viteConfig = (): string =>
  `export default {
  lint: {
    plugins: [],
    categories: { correctness: "off" },
    jsPlugins: [{ name: "karume", specifier: ${JSON.stringify(PLUGIN_PATH)} }],
    rules: { ${JSON.stringify(RULE)}: "error" },
  },
};
`;

/**
 * Writes `files` (path relative to the temporary directory → source), runs them through `vp lint` and
 * returns this rule's diagnostics. `vp lint` exits with code 1 when it reports, so 0 and 1 are both
 * normal and the JSON is read.
 */
const lint = async (files: Readonly<Record<string, string>>): Promise<LintDiagnostic[]> => {
  const root = await makeTempDir({ prefix: "karume-lint-" });
  try {
    await writeTextFile(
      join(root, "package.json"),
      '{ "name": "fixture", "private": true, "type": "module" }\n',
    );
    await writeTextFile(join(root, "vite.config.ts"), viteConfig());
    for (const [path, source] of Object.entries(files)) {
      await mkdir(dirname(join(root, path)), { recursive: true });
      await writeTextFile(join(root, path), source);
    }
    const args = [VP_BIN, "lint", "-f", "json", ...Object.keys(files)];
    let stdout: string;
    try {
      ({ stdout } = await promisify(execFile)(process.execPath, args, { cwd: root }));
    } catch (error) {
      if (
        !(
          error instanceof Error &&
          "code" in error &&
          error.code === 1 &&
          "stdout" in error &&
          typeof error.stdout === "string"
        )
      ) {
        const stderr = error instanceof Error && "stderr" in error ? String(error.stderr) : "";
        throw new Error(`vp lint failed for a reason other than a report: ${stderr}`, {
          cause: error,
        });
      }
      stdout = error.stdout;
    }
    const output: LintOutput = JSON.parse(stdout);
    return output.diagnostics.filter((diagnostic) => diagnostic.code === RULE_CODE);
  } finally {
    await remove(root, { recursive: true });
  }
};

/** The original spelling each diagnostic points at (the text its span covers), per file in source order. */
const reportedByFile = async (
  files: Readonly<Record<string, string>>,
): Promise<Map<string, string[]>> => {
  const result = new Map<string, string[]>(Object.keys(files).map((path) => [path, []]));
  const diagnostics = await lint(files);
  const spans = diagnostics
    .flatMap((diagnostic) =>
      diagnostic.labels.map((label) => ({ filename: diagnostic.filename, ...label.span })),
    )
    .sort((a, b) => a.offset - b.offset);
  for (const { filename, offset, length } of spans) {
    const bytes = new TextEncoder().encode(files[filename]);
    const spelled = new TextDecoder().decode(bytes.subarray(offset, offset + length));
    const list = result.get(filename);
    assert(list !== undefined, `report on an unexpected file: ${filename}`);
    list.push(spelled);
  }
  return result;
};

/** The original spelling of the places the rule reported (the text each span covers). */
const reported = async (source: string, filename = IN_SCOPE): Promise<string[]> =>
  (await reportedByFile({ [filename]: source })).get(filename) ?? [];

describe("reports a TypedArray without its type argument in a type position", () => {
  it("catches parameters, return types, type arguments, arrays, type aliases, fields and satisfies", async () => {
    const source = [
      "export const f = (a: Float32Array): Uint8Array => new Uint8Array(a.length);",
      "export const g = async (): Promise<Int32Array> => new Int32Array(1);",
      "export const h: readonly Float64Array[] = [];",
      "export type Alias = Uint16Array;",
      "export type Holder = { readonly data: BigInt64Array };",
      "export const s = new Uint32Array(1) satisfies Uint32Array;",
    ].join("\n");
    assertEquals(await reported(source), [
      "Float32Array",
      "Uint8Array",
      "Int32Array",
      "Float64Array",
      "Uint16Array",
      "BigInt64Array",
      "Uint32Array",
    ]);
  });

  it("catches a qualified name globalThis.X by its rightmost name", async () => {
    const source = [
      "export type A = globalThis.Float32Array;",
      "export const f = (bytes: globalThis.Uint8Array): number => bytes.length;",
    ].join("\n");
    assertEquals(await reported(source), ["globalThis.Float32Array", "globalThis.Uint8Array"]);
  });

  it("also catches interface extends and class implements", async () => {
    const source = [
      "export interface Samples extends Float32Array {}",
      "export interface Bytes extends globalThis.Uint8Array {}",
      "export abstract class Lanes implements Int16Array {}",
    ].join("\n");
    assertEquals(await reported(source), ["Float32Array", "globalThis.Uint8Array", "Int16Array"]);
  });

  it("covers all 12 TypedArray names", async () => {
    const names = [
      "Int8Array",
      "Uint8Array",
      "Uint8ClampedArray",
      "Int16Array",
      "Uint16Array",
      "Int32Array",
      "Uint32Array",
      "Float16Array",
      "Float32Array",
      "Float64Array",
      "BigInt64Array",
      "BigUint64Array",
    ];
    const source = names.map((name, index) => `export type T${index} = ${name};`).join("\n");
    assertEquals(await reported(source), names);
  });

  it("the report message names the type and shows the explicit <ArrayBufferLike> way for a boundary", async () => {
    const [diagnostic] = await lint({ [IN_SCOPE]: "export type T = Float32Array;" });
    assert(diagnostic !== undefined, "no report");
    assertEquals(diagnostic.code, RULE_CODE);
    assert(diagnostic.message.includes("Float32Array has no type argument"), diagnostic.message);
    assert(diagnostic.message.includes("Float32Array<ArrayBufferLike>"), diagnostic.message);
  });
});

describe("does not report types with a type argument or value positions", () => {
  it("lets explicit <ArrayBuffer> and <ArrayBufferLike> pass", async () => {
    const source = [
      "export const f = (a: Float32Array<ArrayBuffer>): Uint8Array<ArrayBuffer> =>",
      "  new Uint8Array(a.length);",
      "export const g = (bytes: Uint8Array<ArrayBufferLike>): number => bytes.length;",
      "export type Holder = { readonly data: Int32Array<ArrayBuffer>[] };",
      "export type Qualified = globalThis.Float32Array<ArrayBuffer>;",
      "export interface Samples extends Float32Array<ArrayBuffer> {}",
      "export abstract class Lanes implements Int16Array<ArrayBufferLike> {}",
    ].join("\n");
    assertEquals(await reported(source), []);
  });

  it("new / from / instanceof / typeof type queries, comments and strings are out of scope", async () => {
    const source = [
      "// Float32Array in a comment is not a type position",
      "export const a = new Float32Array(4);",
      "export const b = Uint8Array.from([1, 2]);",
      "export const c = (value: unknown): boolean => value instanceof Int16Array;",
      "export const d: typeof Float64Array = Float64Array;",
      'export const e = "Uint32Array";',
      "export const f = Float32Array.BYTES_PER_ELEMENT;",
    ].join("\n");
    assertEquals(await reported(source), []);
  });

  it("misses the typeof forms that do not write the TypedArray name as a type reference (recorded in ADR 0002)", async () => {
    // Syntax alone does not tell what the type is (that it becomes the default ArrayBufferLike). When
    // the rule catches them, update the "Misses" item of ADR 0002 and this case together.
    const source = [
      "export type I = InstanceType<typeof Float32Array>;",
      "export type R = ReturnType<typeof Uint8Array.from>;",
    ].join("\n");
    assertEquals(await reported(source), []);
  });
});

describe("the scope is packages/<pkg>/src/ only", () => {
  const bare = "export const f = (a: Float32Array): number => a.length;";

  it("the src of the three packages is in scope", async () => {
    const filenames = ["runtime", "hub", "models"].map((pkg) => `packages/${pkg}/src/deep/file.ts`);
    const result = await reportedByFile(
      Object.fromEntries(filenames.map((filename) => [filename, bare])),
    );
    for (const filename of filenames) {
      assertEquals(result.get(filename), ["Float32Array"], filename);
    }
  });

  it("tests, tools and examples are not reported even without a type argument", async () => {
    const filenames = [
      "packages/runtime/tests/sample_test.ts",
      "packages/models/tests/helpers/sample.ts",
      "packages/hub/mod.ts",
      "tools/lint/sample.ts",
      "examples/wan/main.ts",
    ];
    const result = await reportedByFile(
      Object.fromEntries(filenames.map((filename) => [filename, bare])),
    );
    for (const filename of filenames) {
      assertEquals(result.get(filename), [], filename);
    }
  });

  it("gives the same result with Windows separators (\\)", () => {
    assert(isTargetFile("C:\\repo\\packages\\runtime\\src\\gpu\\submit.ts"));
    assert(!isTargetFile("C:\\repo\\packages\\runtime\\tests\\gpu_test.ts"));
  });
});
