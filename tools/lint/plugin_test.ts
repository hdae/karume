/**
 * リポ内 lint プラグインの規則 `karume/typed-array-buffer` の検証（合成したソースを
 * `Deno.lint.runPlugin` に通す — GPU もファイルも要らない）。
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
 * NOTE: リポの慣習に合わせて `Deno.test`（文脈）+ `t.step`（振る舞い）で書く。
 */

import { assert, assertEquals } from "@std/assert";
import plugin, { isTargetFile } from "./plugin.ts";

const RULE = "karume/typed-array-buffer";
const IN_SCOPE = "/repo/packages/runtime/src/sample.ts";

/** 規則が報告した箇所の元の綴り（報告の範囲が指す文字列）。 */
const reported = (source: string, filename = IN_SCOPE): string[] =>
  Deno.lint.runPlugin(plugin, filename, source)
    .filter((diagnostic) => diagnostic.id === RULE)
    .map((diagnostic) => source.slice(diagnostic.range[0], diagnostic.range[1]));

Deno.test("型の位置で型引数を省いた TypedArray を報告する", async (t) => {
  await t.step("引数・戻り型・型引数・配列・型エイリアス・欄・satisfies の全てで拾う", () => {
    const source = [
      "export const f = (a: Float32Array): Uint8Array => new Uint8Array(a.length);",
      "export const g = async (): Promise<Int32Array> => new Int32Array(1);",
      "export const h: readonly Float64Array[] = [];",
      "export type Alias = Uint16Array;",
      "export type Holder = { readonly data: BigInt64Array };",
      "export const s = new Uint32Array(1) satisfies Uint32Array;",
    ].join("\n");
    assertEquals(reported(source), [
      "Float32Array",
      "Uint8Array",
      "Int32Array",
      "Float64Array",
      "Uint16Array",
      "BigInt64Array",
      "Uint32Array",
    ]);
  });

  await t.step("修飾名 globalThis.X は右端の名前で拾う", () => {
    const source = [
      "export type A = globalThis.Float32Array;",
      "export const f = (bytes: globalThis.Uint8Array): number => bytes.length;",
    ].join("\n");
    assertEquals(reported(source), ["globalThis.Float32Array", "globalThis.Uint8Array"]);
  });

  await t.step("interface の extends と class の implements でも拾う", () => {
    const source = [
      "export interface Samples extends Float32Array {}",
      "export interface Bytes extends globalThis.Uint8Array {}",
      "export abstract class Lanes implements Int16Array {}",
    ].join("\n");
    assertEquals(reported(source), ["Float32Array", "globalThis.Uint8Array", "Int16Array"]);
  });

  await t.step("12 種の TypedArray の名前を全て対象にする", () => {
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
    assertEquals(reported(source), names);
  });

  await t.step("報告の文言は名前と、境界なら <ArrayBufferLike> を明示する道を示す", () => {
    const [diagnostic] = Deno.lint.runPlugin(plugin, IN_SCOPE, "export type T = Float32Array;");
    assertEquals(diagnostic.id, RULE);
    assert(diagnostic.message.includes("Float32Array"), diagnostic.message);
    assert(diagnostic.hint?.includes("Float32Array<ArrayBufferLike>"), diagnostic.hint);
  });
});

Deno.test("型引数を書いた型と、値の位置は報告しない", async (t) => {
  await t.step("<ArrayBuffer> と <ArrayBufferLike> の明示は通す", () => {
    const source = [
      "export const f = (a: Float32Array<ArrayBuffer>): Uint8Array<ArrayBuffer> =>",
      "  new Uint8Array(a.length);",
      "export const g = (bytes: Uint8Array<ArrayBufferLike>): number => bytes.length;",
      "export type Holder = { readonly data: Int32Array<ArrayBuffer>[] };",
      "export type Qualified = globalThis.Float32Array<ArrayBuffer>;",
      "export interface Samples extends Float32Array<ArrayBuffer> {}",
      "export abstract class Lanes implements Int16Array<ArrayBufferLike> {}",
    ].join("\n");
    assertEquals(reported(source), []);
  });

  await t.step("new / from / instanceof / typeof の型クエリ・コメント・文字列は対象外", () => {
    const source = [
      "// Float32Array をコメントに書いても型の位置ではない",
      "export const a = new Float32Array(4);",
      "export const b = Uint8Array.from([1, 2]);",
      "export const c = (value: unknown): boolean => value instanceof Int16Array;",
      "export const d: typeof Float64Array = Float64Array;",
      'export const e = "Uint32Array";',
      "export const f = Float32Array.BYTES_PER_ELEMENT;",
    ].join("\n");
    assertEquals(reported(source), []);
  });

  await t.step(
    "TypedArray の名前を型参照に書かない typeof 経由の形は取りこぼす（ADR 0002 に明記）",
    () => {
      // 構文だけでは型の中身（既定の ArrayBufferLike になること）が分からない。拾えるように
      // なったら ADR 0002 の「取りこぼす形」とこのケースを一緒に直す。
      const source = [
        "export type I = InstanceType<typeof Float32Array>;",
        "export type R = ReturnType<typeof Uint8Array.from>;",
      ].join("\n");
      assertEquals(reported(source), []);
    },
  );
});

Deno.test("対象は packages/<pkg>/src/ の下だけ", async (t) => {
  const bare = "export const f = (a: Float32Array): number => a.length;";

  await t.step("packages の 3 パッケージの src は対象", () => {
    for (const pkg of ["runtime", "hub", "models"]) {
      assertEquals(reported(bare, `/repo/packages/${pkg}/src/deep/file.ts`), ["Float32Array"]);
    }
  });

  await t.step("tests・tools・examples は型引数が無くても報告しない", () => {
    for (
      const filename of [
        "/repo/packages/runtime/tests/sample_test.ts",
        "/repo/packages/models/tests/helpers/sample.ts",
        "/repo/packages/hub/mod.ts",
        "/repo/tools/lint/sample.ts",
        "/repo/examples/wan/main.ts",
      ]
    ) {
      assertEquals(reported(bare, filename), [], filename);
    }
  });

  await t.step("Windows の区切り（\\）でも同じ判定になる", () => {
    assert(isTargetFile("C:\\repo\\packages\\runtime\\src\\gpu\\submit.ts"));
    assert(!isTargetFile("C:\\repo\\packages\\runtime\\tests\\gpu_test.ts"));
  });
});
