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
 * 判定は AST の型参照（`TSTypeReference`）と、interface の `extends`・class の `implements`
 * （`TSInterfaceHeritage` / `TSClassImplements`）を見る。型注釈・型引数・戻り型・型エイリアス・
 * 欄の型・`satisfies` は全て型参照の節点なので拾い、値の位置（`new Float32Array(n)` や
 * `instanceof Float32Array`）・コメント・文字列は構文上そもそも型参照にならない。修飾名
 * （`globalThis.Float32Array`）は右端の名前で判定する。
 *
 * NOTE: TypedArray の名前を型参照として書かずに既定の `ArrayBufferLike` を作る形
 * （`InstanceType<typeof Float32Array>` / `ReturnType<typeof Uint8Array.from>`）は取りこぼす —
 * 構文だけでは型の中身が分からない（ADR 0002 の改訂節）。
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

/**
 * 型参照・heritage の名前の節点から右端の名前を取る（`Float32Array` / `globalThis.Float32Array`
 * のどちらも `Float32Array`）。名前で無い節点（`this`・計算した添字・呼び出し）は undefined。
 */
const rightmostName = (node: Deno.lint.Node): string | undefined => {
  switch (node.type) {
    case "Identifier":
      return node.name;
    case "TSQualifiedName":
      return node.right.name;
    case "MemberExpression":
      return !node.computed && node.property.type === "Identifier" ? node.property.name : undefined;
    default:
      return undefined;
  }
};

const plugin: Deno.lint.Plugin = {
  name: "karume",
  rules: {
    "typed-array-buffer": {
      create(context) {
        if (!isTargetFile(context.filename)) return {};
        const check = (
          node: Deno.lint.Node,
          nameNode: Deno.lint.Node,
          typeArguments: Deno.lint.TSTypeParameterInstantiation | undefined,
        ): void => {
          const name = rightmostName(nameNode);
          if (name === undefined || !TYPED_ARRAY_NAMES.has(name)) return;
          if (typeArguments !== undefined) return;
          context.report({
            node,
            message: `${name} の型引数が無い（既定は ArrayBufferLike に緩む）`,
            hint: `${name}<ArrayBuffer> と書く。意図して任意のバッファを受ける境界なら ` +
              `${name}<ArrayBufferLike> と明示し、ADR 0002 の例外表に足す。`,
          });
        };
        return {
          TSTypeReference: (node) => check(node, node.typeName, node.typeArguments),
          TSInterfaceHeritage: (node) => check(node, node.expression, node.typeArguments),
          TSClassImplements: (node) => check(node, node.expression, node.typeArguments),
        };
      },
    },
  },
};

export default plugin;
