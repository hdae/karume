/**
 * リポ内 lint プラグインの規則 `karume/typed-array-buffer` の検証（合成したソースを一時ディレクトリに
 * `packages/<pkg>/src/…` などの配置で書き、プラグインを登録した一時の `vite.config.ts` で実物の
 * `vp lint`（Oxlint）に通す — GPU は要らない）。
 *
 * 見る点:
 *
 * 1. **報告する** — packages の本体で、型の位置の TypedArray が型引数を省いた形（引数・戻り型・
 *    型引数・配列・型エイリアス・欄・`satisfies`・修飾名 `globalThis.X`・interface の `extends`・
 *    class の `implements`）と、12 種の名前の全て
 * 2. **報告しない** — `<ArrayBuffer>` / `<ArrayBufferLike>` を書いた型と、値の位置
 *    （`new` / `from` / `instanceof` / `typeof` の型クエリ）・コメント・文字列。加えて、取りこぼすと
 *    ADR 0002 に明記した形（`InstanceType<typeof X>` / `ReturnType<typeof X.from>`）が今も
 *    取りこぼされること（明記と現物の食い違いを落とす）
 * 3. **対象の外** — `packages/<pkg>/tests/`・`tools/`・`examples/` は型引数が無くても報告しない
 *
 * NOTE: リポの慣習に合わせて `describe`（文脈）+ `it`（振る舞い）で書く。
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
/** Oxlint の JSON 出力での規則の綴り（`<plugin>(<rule>)`）。 */
const RULE_CODE = "karume(typed-array-buffer)";
const IN_SCOPE = "packages/runtime/src/sample.ts";

const PLUGIN_PATH = fileURLToPath(new URL("../oxlint-plugin.ts", import.meta.url));
/** この workspace の Vite+ の CLI（PATH に依らず、リポが固定した版の `vp` を走らせる）。 */
const VP_BIN = join(
  dirname(createRequire(import.meta.url).resolve("vite-plus/package.json")),
  "bin",
  "vp",
);

/** `vp lint -f json` の出力のうち、ここで読む欄。`offset` / `length` は UTF-8 のバイト単位。 */
type LintDiagnostic = {
  readonly message: string;
  readonly code: string;
  readonly filename: string;
  readonly labels: readonly {
    readonly span: { readonly offset: number; readonly length: number };
  }[];
};
type LintOutput = { readonly diagnostics: readonly LintDiagnostic[] };

/** この規則だけを有効にした設定（組み込み規則は切る — 報告の数を規則の判定だけにする）。 */
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
 * `files`（一時ディレクトリからの相対の道 → ソース）を書き、`vp lint` に通してこの規則の診断を返す。
 * 報告があると `vp lint` は終了コード 1 で終わるので、0 と 1 は正常として JSON を読む。
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
        throw new Error(`vp lint が報告以外の理由で失敗した: ${stderr}`, { cause: error });
      }
      stdout = error.stdout;
    }
    const output: LintOutput = JSON.parse(stdout);
    return output.diagnostics.filter((diagnostic) => diagnostic.code === RULE_CODE);
  } finally {
    await remove(root, { recursive: true });
  }
};

/** 診断が指す元の綴り（報告の範囲が指す文字列）を、ファイルごとにソースの順で返す。 */
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
    assert(list !== undefined, `想定外のファイルへの報告: ${filename}`);
    list.push(spelled);
  }
  return result;
};

/** 規則が報告した箇所の元の綴り（報告の範囲が指す文字列）。 */
const reported = async (source: string, filename = IN_SCOPE): Promise<string[]> =>
  (await reportedByFile({ [filename]: source })).get(filename) ?? [];

describe("型の位置で型引数を省いた TypedArray を報告する", () => {
  it("引数・戻り型・型引数・配列・型エイリアス・欄・satisfies の全てで拾う", async () => {
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

  it("修飾名 globalThis.X は右端の名前で拾う", async () => {
    const source = [
      "export type A = globalThis.Float32Array;",
      "export const f = (bytes: globalThis.Uint8Array): number => bytes.length;",
    ].join("\n");
    assertEquals(await reported(source), ["globalThis.Float32Array", "globalThis.Uint8Array"]);
  });

  it("interface の extends と class の implements でも拾う", async () => {
    const source = [
      "export interface Samples extends Float32Array {}",
      "export interface Bytes extends globalThis.Uint8Array {}",
      "export abstract class Lanes implements Int16Array {}",
    ].join("\n");
    assertEquals(await reported(source), ["Float32Array", "globalThis.Uint8Array", "Int16Array"]);
  });

  it("12 種の TypedArray の名前を全て対象にする", async () => {
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

  it("報告の文言は名前と、境界なら <ArrayBufferLike> を明示する道を示す", async () => {
    const [diagnostic] = await lint({ [IN_SCOPE]: "export type T = Float32Array;" });
    assert(diagnostic !== undefined, "報告が無い");
    assertEquals(diagnostic.code, RULE_CODE);
    assert(diagnostic.message.includes("Float32Array の型引数が無い"), diagnostic.message);
    assert(diagnostic.message.includes("Float32Array<ArrayBufferLike>"), diagnostic.message);
  });
});

describe("型引数を書いた型と、値の位置は報告しない", () => {
  it("<ArrayBuffer> と <ArrayBufferLike> の明示は通す", async () => {
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

  it("new / from / instanceof / typeof の型クエリ・コメント・文字列は対象外", async () => {
    const source = [
      "// Float32Array をコメントに書いても型の位置ではない",
      "export const a = new Float32Array(4);",
      "export const b = Uint8Array.from([1, 2]);",
      "export const c = (value: unknown): boolean => value instanceof Int16Array;",
      "export const d: typeof Float64Array = Float64Array;",
      'export const e = "Uint32Array";',
      "export const f = Float32Array.BYTES_PER_ELEMENT;",
    ].join("\n");
    assertEquals(await reported(source), []);
  });

  it("TypedArray の名前を型参照に書かない typeof 経由の形は取りこぼす（ADR 0002 に明記）", async () => {
    // 構文だけでは型の中身（既定の ArrayBufferLike になること）が分からない。拾えるように
    // なったら ADR 0002 の「取りこぼす形」とこのケースを一緒に直す。
    const source = [
      "export type I = InstanceType<typeof Float32Array>;",
      "export type R = ReturnType<typeof Uint8Array.from>;",
    ].join("\n");
    assertEquals(await reported(source), []);
  });
});

describe("対象は packages/<pkg>/src/ の下だけ", () => {
  const bare = "export const f = (a: Float32Array): number => a.length;";

  it("packages の 3 パッケージの src は対象", async () => {
    const filenames = ["runtime", "hub", "models"].map((pkg) => `packages/${pkg}/src/deep/file.ts`);
    const result = await reportedByFile(
      Object.fromEntries(filenames.map((filename) => [filename, bare])),
    );
    for (const filename of filenames) {
      assertEquals(result.get(filename), ["Float32Array"], filename);
    }
  });

  it("tests・tools・examples は型引数が無くても報告しない", async () => {
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

  it("Windows の区切り（\\）でも同じ判定になる", () => {
    assert(isTargetFile("C:\\repo\\packages\\runtime\\src\\gpu\\submit.ts"));
    assert(!isTargetFile("C:\\repo\\packages\\runtime\\tests\\gpu_test.ts"));
  });
});
