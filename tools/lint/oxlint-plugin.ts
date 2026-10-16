/**
 * リポ内の Oxlint JS プラグイン（`vite.config.ts` の `lint.jsPlugins` に登録 — `vp lint` = verify の
 * lint 段で走る）。Node の型除去でそのまま読み込める TS で書く（型だけの import しか使わない）。
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

import type { ESTree, Plugin } from "vite-plus/lint/plugins";

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
const rightmostName = (node: ESTree.Node): string | undefined => {
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

/**
 * 報告の文言。Oxlint の診断には Deno lint の `hint` に当たる欄が無いので、直し方を同じ文言の
 * 後ろに続ける（`vp lint` の出力に直し方が出るように）。
 */
const reportMessage = (name: string): string =>
  `${name} の型引数が無い（既定は ArrayBufferLike に緩む）。` +
  `${name}<ArrayBuffer> と書く。意図して任意のバッファを受ける境界なら ` +
  `${name}<ArrayBufferLike> と明示し、ADR 0002 の例外表に足す。`;

const plugin: Plugin = {
  meta: { name: "karume" },
  rules: {
    "typed-array-buffer": {
      meta: {
        type: "problem",
        docs: {
          description:
            "packages/<pkg>/src/ の型の位置の TypedArray にバッファの型引数を求める（ADR 0002）",
        },
      },
      create(context) {
        // `physicalFilename` は前処理（Markdown の中のコード等）を挟んでも実ファイルの道を指す。
        if (!isTargetFile(context.physicalFilename)) return {};
        const check = (
          node: ESTree.Node,
          nameNode: ESTree.Node,
          typeArguments: ESTree.TSTypeParameterInstantiation | null,
        ): void => {
          const name = rightmostName(nameNode);
          if (name === undefined || !TYPED_ARRAY_NAMES.has(name)) return;
          if (typeArguments !== null) return;
          context.report({ node, message: reportMessage(name) });
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
