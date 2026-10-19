/**
 * The repository's Oxlint JS plugin (registered in `lint.jsPlugins` of `vite.config.ts` — it runs as
 * `vp lint` inside `vp check`). Written in TS that Node's type stripping loads as is (it uses
 * type-only imports only).
 *
 * Rule `karume/typed-array-buffer`: in the package sources (`packages/<pkg>/src/`), a TypedArray
 * written in a type position MUST NOT omit its type argument (ADR 0002: "TypedArrays name
 * `<ArrayBuffer>` explicitly"). Since TS 5.7 an omitted argument means the default
 * `ArrayBufferLike` (which includes `SharedArrayBuffer`), silently widening the type of values
 * passed to `GPUQueue.writeBuffer` and the like. A boundary that deliberately accepts any buffer
 * writes `<ArrayBufferLike>` explicitly (the exceptions are listed in the addendum of 2026-10-11 to
 * ADR 0002).
 *
 * The check looks at the AST's type references (`TSTypeReference`) and at interface `extends` and
 * class `implements` (`TSInterfaceHeritage` / `TSClassImplements`). Type annotations, type
 * arguments, return types, type aliases, field types and `satisfies` are all type-reference nodes,
 * so they are caught; value positions (`new Float32Array(n)` or `instanceof Float32Array`),
 * comments and strings are never type references by syntax. A qualified name
 * (`globalThis.Float32Array`) is judged by its rightmost name.
 *
 * NOTE: forms that produce the default `ArrayBufferLike` without writing the TypedArray name as a
 * type reference (`InstanceType<typeof Float32Array>` / `ReturnType<typeof Uint8Array.from>`) are
 * missed — syntax alone does not tell what the type is (the addendum of 2026-10-11 to ADR 0002).
 */

import type { ESTree, Plugin } from "vite-plus/lint/plugins";

/** The names of the TypedArrays that take a type argument (the buffer type). */
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

/** In scope = the package sources. tests / tools / examples are out of scope (separator `/` or `\`). */
const TARGET_PATH = /(?:^|[\\/])packages[\\/][^\\/]+[\\/]src[\\/]/;

export const isTargetFile = (filename: string): boolean => TARGET_PATH.test(filename);

/**
 * Takes the rightmost name from the name node of a type reference or heritage clause (both
 * `Float32Array` and `globalThis.Float32Array` give `Float32Array`). A node that is not a name
 * (`this`, a computed index, a call) gives undefined.
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
 * The report message. Oxlint diagnostics have no field like Deno lint's `hint`, so the fix follows
 * in the same message (so that the output of `vp lint` shows how to fix it).
 */
const reportMessage = (name: string): string =>
  `${name} has no type argument (the default widens to ArrayBufferLike). ` +
  `Write ${name}<ArrayBuffer>. At a boundary that deliberately accepts any buffer, ` +
  `write ${name}<ArrayBufferLike> explicitly and add it to the exception table of ADR 0002.`;

const plugin: Plugin = {
  meta: { name: "karume" },
  rules: {
    "typed-array-buffer": {
      meta: {
        type: "problem",
        docs: {
          description:
            "Requires a buffer type argument on TypedArrays in type positions under packages/<pkg>/src/ (ADR 0002)",
        },
      },
      create(context) {
        // `physicalFilename` names the real file's path even through preprocessing (code inside Markdown, etc.).
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
