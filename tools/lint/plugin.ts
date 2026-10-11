/**
 * リポ内の deno lint プラグイン（`deno.json` の `lint.plugins` に登録 — `deno lint` = verify の
 * lint 段で走る）。
 *
 * 規則 `karume/typed-array-buffer`: packages の本体（`packages/<pkg>/src/`）で、型の位置に書いた
 * TypedArray が型引数を省くことを禁じる（ADR 0002 の「TypedArray は `<ArrayBuffer>` 明示」）。
 * 省くと TS 5.7 以降は既定の `ArrayBufferLike`（`SharedArrayBuffer` を含む）になり、
 * `GPUQueue.writeBuffer` などへ渡す値の型が黙って緩む。意図して任意のバッファを受ける境界は
 * `<ArrayBufferLike>` と明示する（例外の一覧は ADR 0002 の改訂節）。
 *
 * 判定は AST の型参照（`TSTypeReference`）だけを見る。型注釈・型引数・戻り型・型エイリアス・
 * 欄の型・`satisfies` は全てこの節点なので拾い、値の位置（`new Float32Array(n)` や
 * `instanceof Float32Array`）・コメント・文字列は構文上そもそも型参照にならない。
 */

/** 型引数（バッファの型）を取る TypedArray の名前。 */
const TYPED_ARRAY_NAMES: ReadonlySet<string> = new Set([
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
]);

/** 対象 = packages の本体。tests / tools / examples は対象外（区切りは `/` と `\` の両方）。 */
const TARGET_PATH = /(?:^|[\\/])packages[\\/][^\\/]+[\\/]src[\\/]/;

export const isTargetFile = (filename: string): boolean => TARGET_PATH.test(filename);

const plugin: Deno.lint.Plugin = {
  name: "karume",
  rules: {
    "typed-array-buffer": {
      create(context) {
        if (!isTargetFile(context.filename)) return {};
        return {
          TSTypeReference(node) {
            if (node.typeName.type !== "Identifier") return;
            if (!TYPED_ARRAY_NAMES.has(node.typeName.name)) return;
            if (node.typeArguments !== undefined) return;
            context.report({
              node,
              message: `${node.typeName.name} の型引数が無い（既定は ArrayBufferLike に緩む）`,
              hint:
                `${node.typeName.name}<ArrayBuffer> と書く。意図して任意のバッファを受ける境界なら ` +
                `${node.typeName.name}<ArrayBufferLike> と明示し、ADR 0002 の例外表に足す。`,
            });
          },
        };
      },
    },
  },
};

export default plugin;
