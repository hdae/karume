// Implementation of the public-surface snapshot check (ADR
// [0008](../../../../docs/decisions/0008-public-api.md) addendum 2026-09-20). Shared by the
// `tests/public_surface_test.ts` of the 3 packages.
//
// Division of roles with the by-name check (`packages/models/tests/models_barrel_surface_test.ts`):
//
// - **By name = a declaration of intent** — a person writes "this spelling is exported / this one
//   is not". Spellings not written pass through (green whether exports missing from the list are
//   added or removed).
// - **Snapshot = detecting additions and removals** (this file) — it does not say what is right,
//   it only looks at the difference from last time. Type exports are collected too, so it also
//   pins surface that values cannot observe (re-exports that are only `export type`).
//
// Collection uses the synchronous API of TypeScript 7 (`typescript/unstable/sync`) to read the
// exports of each entry's module symbol. The `exports` of `package.json` is the source of truth for
// entries, so one more surface automatically means one more scan (the check keeps no second list of
// surfaces). The kind vocabulary carries over the vocabulary of the earlier collector (`deno doc`)
// (so the fixture is read as is — the classification rules are in `kindsOf`).
//
// MUST: a shape that cannot be read throws instead of being guessed — reading "could not collect"
// as "the surface is empty" would let this check hide, in green, a surface that disappeared whole.

import { fileURLToPath } from "node:url";
import { assert } from "@std/assert";
import { readTextFile, writeTextFile } from "@karume/test-support/fs";
import {
  isArrowFunction,
  isClassExpression,
  isFunctionExpression,
  isVariableDeclaration,
  SyntaxKind,
} from "typescript/unstable/ast";
import {
  API,
  type Checker,
  type Program,
  type Symbol as TsSymbol,
  SymbolFlags,
} from "typescript/unstable/sync";

/**
 * Format version of the fixture. Bump it when the shape changes, and fix the reader
 * (`parseSnapshot`) too.
 */
const SCHEMA = 1;

/**
 * Environment variable that requests a snapshot rewrite (unset = compare / `"write"` = rewrite).
 */
const MODE_ENV = "KARUME_SURFACE";

/** The procedure handed to the reader when the surface differs. */
const HOW_TO_UPDATE = `If the change is intended, run the same test with ${MODE_ENV}=write to update the snapshot, and write the difference into the CHANGELOG.`;

/**
 * One public symbol. `kind` uses the vocabulary of `deno doc` (`class` / `function` / `typeAlias` /
 * `variable` / `interface` / `enum` / `namespace`) — the fixture is written in this vocabulary.
 */
export type PublicSymbol = {
  readonly name: string;
  readonly kind: string;
};

/** Shape of the fixture. The keys of `entries` are the spellings of `exports` in `package.json`. */
type SurfaceSnapshot = {
  readonly schema: number;
  readonly entries: Readonly<Record<string, readonly PublicSymbol[]>>;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const record = (value: unknown, what: string): Record<string, unknown> => {
  if (!isRecord(value)) throw new Error(`public-surface check: ${what} is not an object`);
  return value;
};

const text = (value: unknown, what: string): string => {
  if (typeof value !== "string") throw new Error(`public-surface check: ${what} is not a string`);
  return value;
};

const list = (value: unknown, what: string): readonly unknown[] => {
  if (!Array.isArray(value)) throw new Error(`public-surface check: ${what} is not an array`);
  return value;
};

/**
 * Locale-independent lexicographic order (so the fixture's order does not vary with the
 * environment).
 */
const compare = (left: string, right: string): number => (left < right ? -1 : left > right ? 1 : 0);

const bySymbol = (left: PublicSymbol, right: PublicSymbol): number =>
  left.name === right.name ? compare(left.kind, right.kind) : compare(left.name, right.name);

/**
 * "Key → symbol" index for the set operations. A key folds the pair of `name` and `kind`, and the
 * delimiter `::` appears neither in TS identifiers nor in the kind vocabulary. Keys are **never
 * parsed back** — the difference is displayed from the symbols the index holds.
 */
const index = (symbols: readonly PublicSymbol[]): ReadonlyMap<string, PublicSymbol> =>
  new Map(symbols.map((symbol) => [`${symbol.name}::${symbol.kind}`, symbol]));

/** One line of the difference. */
const labelOf = (symbol: PublicSymbol): string => `${symbol.name} (${symbol.kind})`;

/**
 * Normalizes the `exports` of `package.json` to "entry spelling → path relative to the package
 * root".
 *
 * All 3 packages use the map form today, each value a single string. The single-string form
 * (`"exports": "./mod.ts"`) has no key, so it gets `"."` as its spelling, the same as the default
 * entry of the map form (so the fixture keys are not split between two forms). Conditional exports
 * (an object value) throw unread — which condition's entry counts as the surface is not for this
 * check to decide.
 */
const exportEntries = (exports: unknown): ReadonlyMap<string, string> => {
  if (typeof exports === "string") return new Map([[".", exports]]);
  const entries = new Map<string, string>();
  for (const [spec, target] of Object.entries(record(exports, "exports of package.json"))) {
    entries.set(spec, text(target, `exports["${spec}"]`));
  }
  if (entries.size === 0)
    throw new Error("public-surface check: the exports of package.json are empty");
  return entries;
};

/** Symbol flags → kind. A symbol that has several (declaration merging) yields several kinds. */
const KIND_BY_FLAG: readonly (readonly [SymbolFlags, string])[] = [
  [SymbolFlags.Class, "class"],
  [SymbolFlags.Function, "function"],
  [SymbolFlags.Variable, "variable"],
  [SymbolFlags.TypeAlias, "typeAlias"],
  [SymbolFlags.Interface, "interface"],
  [SymbolFlags.Enum, "enum"],
  [SymbolFlags.Module, "namespace"],
];

/**
 * Kind of one variable declaration. `deno doc` counts it as `function` when the initializer is
 * **directly** an arrow function / function expression, `class` for a class expression, and
 * `variable` otherwise (including forms wrapped in parentheses, `as` or `satisfies`) (measured with
 * `deno doc --json` of deno 2.x, 2026-10-16). The fixture is written in that vocabulary, so the
 * same rule applies here.
 */
const variableKind = (declaration: TsSymbol["declarations"][number], name: string): string => {
  const node = declaration.resolve();
  if (node === undefined || !isVariableDeclaration(node)) {
    throw new Error(
      `public-surface check: cannot resolve the variable declaration of ${name} to the AST`,
    );
  }
  const initializer = node.initializer;
  if (initializer === undefined) return "variable";
  if (isArrowFunction(initializer) || isFunctionExpression(initializer)) return "function";
  if (isClassExpression(initializer)) return "class";
  return "variable";
};

/**
 * List of kinds of one exported symbol (for an alias, the resolved target). Overloads have one set
 * of flags, so they fold into one entry. Type-only re-exports (`export type { X }`) are counted by
 * the target's kind too (`deno doc` did not distinguish them either).
 */
const kindsOf = (symbol: TsSymbol, name: string): readonly string[] => {
  const kinds: string[] = [];
  for (const [flag, kind] of KIND_BY_FLAG) {
    if ((symbol.flags & flag) === 0) continue;
    if (flag !== SymbolFlags.Variable) {
      kinds.push(kind);
      continue;
    }
    for (const declaration of symbol.declarations) {
      if (declaration.kind !== SyntaxKind.VariableDeclaration) continue;
      kinds.push(variableKind(declaration, name));
    }
  }
  // A shape outside the vocabulary (reaching here is unexpected) is not read as "not on the
  // surface".
  if (kinds.length === 0) {
    throw new Error(
      `public-surface check: cannot determine the kind of ${name} (flags: ${symbol.flags})`,
    );
  }
  return kinds;
};

/** Collects the symbols one entry exports, from the type checker. */
const readSurface = (checker: Checker, program: Program, entry: string): PublicSymbol[] => {
  const sourceFile = program.getSourceFile(entry);
  if (sourceFile === undefined) {
    throw new Error(`public-surface check: ${entry} is not in the program of tsconfig.json`);
  }
  const moduleSymbol = checker.getSymbolAtLocation(sourceFile);
  if (moduleSymbol === undefined) {
    throw new Error(
      `public-surface check: ${entry} has no module symbol (cannot be read as a module)`,
    );
  }
  const collected: PublicSymbol[] = [];
  for (const exported of checker.getExportsOfModule(moduleSymbol)) {
    // The name is the export-side spelling (`b` for `export { a as b }`); the kind comes from the
    // target the alias resolves to.
    const name = exported.name;
    const target =
      (exported.flags & SymbolFlags.Alias) === 0 ? exported : checker.getAliasedSymbol(exported);
    if (checker.isUnknownSymbol(target)) {
      throw new Error(`public-surface check: ${entry}: cannot resolve the alias ${name}`);
    }
    for (const kind of kindsOf(target, name)) collected.push({ name, kind });
  }
  // Duplicates of the same kind (several variable declarations of one name) fold into one when
  // indexed.
  return [...index(collected).values()].sort(bySymbol);
};

/**
 * Collects the public surface of every entry of a package. Starts one tsgo process and reads on the
 * program of the workspace root's `tsconfig.json` (so `@karume/*` specifiers resolve to the
 * workspace's own sources).
 */
const collectSurfaces = (
  entries: ReadonlyMap<string, string>,
  packageRoot: URL,
  workspaceRoot: URL,
): Map<string, readonly PublicSymbol[]> => {
  const api = new API({ cwd: fileURLToPath(workspaceRoot) });
  try {
    const configFile = fileURLToPath(new URL("tsconfig.json", workspaceRoot));
    const project = api.updateSnapshot({ openProjects: [configFile] }).getProject(configFile);
    if (project === undefined) {
      throw new Error(`public-surface check: cannot open the project of ${configFile}`);
    }
    const actual = new Map<string, readonly PublicSymbol[]>();
    for (const [spec, target] of entries) {
      const entry = fileURLToPath(new URL(target, packageRoot));
      actual.set(spec, readSurface(project.checker, project.program, entry));
    }
    return actual;
  } finally {
    api.close();
  }
};

/** Reads the fixture, checks its shape and returns it. */
const parseSnapshot = (source: string, file: URL): SurfaceSnapshot => {
  const parsed = record(JSON.parse(source), `contents of ${file.href}`);
  if (parsed.schema !== SCHEMA) {
    throw new Error(
      `public-surface check: the snapshot schema is not ${SCHEMA} (${String(parsed.schema)})`,
    );
  }
  const entries: Record<string, readonly PublicSymbol[]> = {};
  for (const [spec, value] of Object.entries(record(parsed.entries, "entries of the snapshot"))) {
    entries[spec] = list(value, `entries["${spec}"]`).map((item) => {
      const symbol = record(item, `element of entries["${spec}"]`);
      return {
        name: text(symbol.name, `name of entries["${spec}"]`),
        kind: text(symbol.kind, `kind of entries["${spec}"]`),
      };
    });
  }
  return { schema: SCHEMA, entries };
};

/** Output form of the fixture (2 spaces, entries in spelling order, trailing newline). */
const serialize = (entries: ReadonlyMap<string, readonly PublicSymbol[]>): string => {
  const ordered: Record<string, readonly PublicSymbol[]> = {};
  // Order by entry spelling (so reordering `package.json` does not show up as a fixture diff).
  for (const spec of [...entries.keys()].sort(compare)) ordered[spec] = entries.get(spec) ?? [];
  return `${JSON.stringify({ schema: SCHEMA, entries: ordered }, null, 2)}\n`;
};

/**
 * Compares the public surface of one package with the snapshot (rewrites it with
 * `KARUME_SURFACE=write`).
 *
 * @param packageRoot The package root (where `package.json` lives). The fixture is
 *   `<packageRoot>/tests/fixtures/public-surface.json`.
 */
export const assertPublicSurface = async (packageRoot: URL): Promise<void> => {
  const mode = process.env[MODE_ENV];
  if (mode !== undefined && mode !== "write") {
    throw new Error(`public-surface check: ${MODE_ENV} accepts only unset or "write" (${mode})`);
  }
  const config = record(
    JSON.parse(await readTextFile(new URL("package.json", packageRoot))),
    `${packageRoot.href}package.json`,
  );
  // Two levels above `packages/<pkg>/` is the workspace root (where the root `tsconfig.json`
  // lives).
  const workspaceRoot = new URL("../../", packageRoot);
  const actual = collectSurfaces(exportEntries(config.exports), packageRoot, workspaceRoot);

  const snapshotFile = new URL("tests/fixtures/public-surface.json", packageRoot);
  if (mode === "write") {
    await writeTextFile(snapshotFile, serialize(actual));
    return;
  }

  const expected = parseSnapshot(await readTextFile(snapshotFile), snapshotFile).entries;
  const specs = [...new Set([...Object.keys(expected), ...actual.keys()])].sort(compare);
  const diff: string[] = [];
  // Additions and removals of a surface (entry) itself are reported **independently** of the symbol
  // diff. Absence on either side folds into an empty array via the `?? []` below, so adding or
  // removing a surface that exports no symbol (a module with only `export {}`) never shows up in
  // the symbol diff.
  for (const spec of specs) {
    const present = actual.has(spec);
    if (present === Object.hasOwn(expected, spec)) continue;
    diff.push(`  ${present ? "+" : "-"} surface ${spec}`);
  }
  for (const spec of specs) {
    const before = index(expected[spec] ?? []);
    const after = index(actual.get(spec) ?? []);
    const added = [...after].filter(([key]) => !before.has(key)).map(([, symbol]) => symbol);
    const removed = [...before].filter(([key]) => !after.has(key)).map(([, symbol]) => symbol);
    if (added.length === 0 && removed.length === 0) continue;
    diff.push(`  ${spec}:`);
    for (const symbol of added.sort(bySymbol)) diff.push(`    + ${labelOf(symbol)}`);
    for (const symbol of removed.sort(bySymbol)) diff.push(`    - ${labelOf(symbol)}`);
  }
  assert(
    diff.length === 0,
    `the public surface differs from the snapshot (+ = added / - = removed):\n${diff.join("\n")}\n` +
      `snapshot: ${snapshotFile.href}\n${HOW_TO_UPDATE}`,
  );
};
