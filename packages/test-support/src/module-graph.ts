/**
 * Static module graph of TypeScript sources, read with oxc-parser — the Node replacement for the
 * graph that `deno info --json` used to give the dependency-graph gate.
 *
 * An edge is every specifier a module names statically: `import` / `import type` / side-effect
 * imports, `export … from` (type-only included), `import()` with a literal argument, the type-level
 * `import("…")` (`TSImportType`), `import x = require("…")`, and `/// <reference path|types="…" />`.
 * Type-only edges count: the consumer's type-checker resolves them even though they vanish at run
 * time. A dynamic `import()` whose argument is not a literal (or a template with substitutions)
 * names no fixed module and is not an edge — the same boundary `deno info` had.
 *
 * MUST: a source that does not parse, or a file that cannot be read, throws. Reading "could not
 * parse" as "no edges" would turn a gate built on this graph green while it sees nothing.
 */

import { readFile } from "node:fs/promises";
import { parseSync } from "oxc-parser";

/** One edge: the module it starts in, the specifier as written, and where it resolved to. */
export type ModuleEdge = {
  readonly from: string;
  readonly written: string;
  readonly resolved: string;
};

/**
 * Resolves a non-relative specifier (bare name, `node:` / `npm:` / … scheme). Returning a `file:`
 * URL makes the walker follow it; any other string is recorded as the edge's target and not followed.
 */
export type ResolveBare = (from: string, written: string) => string;

/** Reads a module's source text by its `file:` URL. */
export type ReadSource = (url: string) => Promise<string>;

const isRelative = (specifier: string): boolean =>
  specifier.startsWith("./") || specifier.startsWith("../") || specifier.startsWith("/");

type AstNode = { readonly type: string } & Readonly<Record<string, unknown>>;

const isNode = (value: unknown): value is AstNode =>
  typeof value === "object" && value !== null && "type" in value && typeof value.type === "string";

/** The value of a string literal node, or of a template literal without substitutions. */
const literalOf = (node: unknown): string | undefined => {
  if (!isNode(node)) return undefined;
  if (node.type === "Literal" && typeof node.value === "string") return node.value;
  if (node.type === "TemplateLiteral" && Array.isArray(node.expressions) && node.expressions.length === 0) {
    const quasi: unknown = Array.isArray(node.quasis) ? node.quasis[0] : undefined;
    const value = isNode(quasi) ? quasi.value : undefined;
    if (typeof value === "object" && value !== null && "cooked" in value && typeof value.cooked === "string") {
      return value.cooked;
    }
  }
  return undefined;
};

/** Visits every AST node (objects carrying a string `type`) below `root`. */
const visit = (root: unknown, onNode: (node: AstNode) => void): void => {
  const pending: unknown[] = [root];
  while (pending.length > 0) {
    const value = pending.pop();
    if (Array.isArray(value)) {
      pending.push(...value);
      continue;
    }
    if (typeof value !== "object" || value === null) continue;
    if (isNode(value)) onNode(value);
    pending.push(...Object.values(value));
  }
};

const REFERENCE = /^\/\s*<reference\s+(path|types)\s*=\s*(["'])([^"']+)\2/;

/**
 * Every specifier `source` names statically, in source order per kind (duplicates kept). `file`
 * picks the parser's language by its extension and labels errors.
 */
export const importSpecifiersOf = (file: string, source: string): string[] => {
  const result = parseSync(file, source);
  if (result.errors.length > 0) {
    throw new Error(
      `module graph: cannot parse ${file}: ${result.errors.map((error) => error.message).join("; ")}`,
    );
  }
  const specifiers: string[] = [];
  for (const statement of result.module.staticImports) specifiers.push(statement.moduleRequest.value);
  for (const statement of result.module.staticExports) {
    // `export * from` / `export { a, b } from` carry the same request on every entry of the statement.
    const requests = new Set<string>();
    for (const entry of statement.entries) {
      if (entry.moduleRequest !== null) requests.add(entry.moduleRequest.value);
    }
    specifiers.push(...requests);
  }
  // Dynamic and type-level imports come from the AST: it carries the literal's decoded value, which
  // `module.dynamicImports` only gives as a source span.
  visit(result.program, (node) => {
    let literal: string | undefined;
    if (node.type === "ImportExpression" || node.type === "TSImportType") literal = literalOf(node.source);
    else if (node.type === "TSExternalModuleReference") literal = literalOf(node.expression);
    if (literal !== undefined) specifiers.push(literal);
  });
  for (const comment of result.comments) {
    if (comment.type !== "Line") continue;
    const reference = REFERENCE.exec(comment.value);
    if (reference === null) continue;
    // `path=` is a file relative to this module even without `./`; `types=` names a package.
    const [, kind, , target] = reference;
    if (target === undefined) continue;
    specifiers.push(kind === "path" && !isRelative(target) ? `./${target}` : target);
  }
  return specifiers;
};

const readSourceFromDisk: ReadSource = (url) => readFile(new URL(url), "utf8");

/**
 * Walks the graph from `entries` (`file:` URLs) and returns every edge of every reachable module.
 * Relative specifiers resolve against the importing module as written (no extension probing — the
 * sources carry `.ts`); everything else goes through `resolveBare`. Each module is read once.
 */
export const walkModuleGraph = async (
  entries: readonly string[],
  resolveBare: ResolveBare,
  readSource: ReadSource = readSourceFromDisk,
): Promise<ModuleEdge[]> => {
  const seen = new Set<string>(entries);
  const pending = [...entries];
  const edges: ModuleEdge[] = [];
  while (pending.length > 0) {
    const from = pending.shift();
    if (from === undefined) break;
    let source: string;
    try {
      source = await readSource(from);
    } catch (error) {
      throw new Error(`module graph: cannot read ${from}`, { cause: error });
    }
    for (const written of importSpecifiersOf(new URL(from).pathname, source)) {
      const resolved = isRelative(written) ? new URL(written, from).href : resolveBare(from, written);
      edges.push({ from, written, resolved });
      if (resolved.startsWith("file:") && !seen.has(resolved)) {
        seen.add(resolved);
        pending.push(resolved);
      }
    }
  }
  return edges;
};
