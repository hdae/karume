// The package dependency-graph check. No GPU and no real assets are needed (it only reads the
// sources with oxc-parser and follows the edges).
//
// The invariant it protects is CLAUDE.md's "runtime dependencies are Web standard APIs only
// (dependency packages are limited to ones built only on Web standard APIs)". Type checking cannot
// protect it — in a workspace, a package that a member's `package.json` `dependencies` does not
// declare still resolves locally if it is in devDependencies or the workspace root (the type check
// is green even if a source under `packages/*` writes the bare specifier `import "jpeg-js"`). Only
// a published package's `dependencies` travel with it, so the breakage surfaces only on the user's
// machine.
//
// Members are derived from `packages` in `pnpm-workspace.yaml` and each `packages/*/package.json`
// (`private` members = test scaffolding that is not published are not subject to the check; they
// still count as members as edge targets). What it looks at is every edge of the module graph
// (static edges that `@karume/test-support/module-graph` takes from the sources with oxc-parser)
// followed from each member's public entries (every value of its `package.json` `exports`). Only
// these three are allowed:
//
// - **Files inside the same member's root** (a relative import, or a self-reference by the
//   member's own name).
// - **Files that are public entries of another member**, pulled in with a bare `@karume/*`
//   specifier that the importing member's `dependencies` declares. A relative import that crosses
//   roots (`../../hub/src/x.ts`) fails even when it points at an entry, and so does an import of a
//   file that belongs to no member (the repository root, `examples/`, `tools/` and so on) — all of
//   them point outside the published package, so they cannot resolve on the user's machine.
// - **External packages declared by the `dependencies` of the `package.json` of the member the
//   edge comes from** (matched by name. Versions are not checked — a `catalog:` declaration only
//   leaves the version to the workspace catalog and is a declaration all the same. devDependencies
//   do not count). Which member's declaration allows it is decided by where the importing file
//   lives — models pulling in `@hdae/fetch-cache` through hub is allowed by hub's declaration, but
//   a models file writing it directly needs a declaration in models.
//
// `node:` built-in modules are not Web standard APIs, so they fail. The only exceptions are the
// files listed in `NODE_BUILTINS_ALLOWED_FROM` below (hub's `./node` subpath = the entry that Node
// users choose explicitly). A specifier with a scheme such as `npm:` / `jsr:` fails as soon as one
// appears in the sources (a spelling that does not resolve when the published package is loaded by
// Node or a bundler). The allow list is not copied into the test; it is derived from each
// `package.json`.
//
// NOTE: the rule of the Deno version of this check that "an edge from one external package to
// another external package (a transitive dependency) fails" is not here. Following the contents of
// external packages would require walking node_modules, but this check reads only the workspace
// sources, so that rule cannot be checked.
//
// The check looks only at static edges (literal import / import type / `export … from` / literal
// dynamic import / `import("…")` in type positions / `import x = require("…")` / `/// <reference
// path|types>`). The target of `new Worker(new URL(...))`, non-literal dynamic imports and the
// target of `import.meta.resolve` are not in the graph, so they are outside this check.
//
// MUST: a form that cannot be read throws instead of being guessed — reading "an edge that could
// not be read" as "no external edge" would make this check hide violations as green (sources whose
// syntax cannot be parsed, unreadable files, declarations or exports of an unknown form, and
// `@karume/*` subpaths that are not entries go red by throwing, not as violations).

import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { readDir, readTextFile } from "@karume/test-support/fs";
import { type ModuleEdge, walkModuleGraph } from "@karume/test-support/module-graph";
import { describe, it } from "vite-plus/test";

/** The repository root (where `pnpm-workspace.yaml` lives). */
const REPO_ROOT = new URL("../../../", import.meta.url);

/**
 * Files that may import `node:` built-in modules (the explicit allow list).
 *
 * hub's `./node` subpath is the entry that reads distributions through the Node file system, and
 * Node users import it explicitly (it cannot be reached from `mod.ts`). It is the one documented
 * carve-out from the invariant — the reason is at the top of `packages/hub/node.ts`.
 */
const NODE_BUILTINS_ALLOWED_FROM: ReadonlySet<string> = new Set([
  new URL("packages/hub/node.ts", REPO_ROOT).href,
]);

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const record = (value: unknown, what: string): Record<string, unknown> => {
  if (!isRecord(value)) throw new Error(`dependency graph check: ${what} is not an object`);
  return value;
};

const text = (value: unknown, what: string): string => {
  if (typeof value !== "string") throw new Error(`dependency graph check: ${what} is not a string`);
  return value;
};

/** One workspace member. */
type Member = {
  readonly name: string;
  /** The member's root (a `file:` URL with a trailing `/`). */
  readonly root: string;
  /** A `private: true` member (not published). Not subject to the check. */
  readonly private: boolean;
  /** Public entries (`exports` subpath → absolute URL). */
  readonly exports: ReadonlyMap<string, string>;
  /** Public entries (every value of `exports` as an absolute URL). */
  readonly entries: readonly string[];
  /** Names of the packages `dependencies` declares (workspace members and external ones alike). */
  readonly declared: ReadonlySet<string>;
};

/**
 * The package name of a specifier (`@scope/name/sub` → `@scope/name`, `name/sub` → `name`) and its
 * subpath.
 */
const packageOf = (specifier: string): { readonly name: string; readonly subpath: string } => {
  const parts = specifier.split("/");
  const length = specifier.startsWith("@") ? 2 : 1;
  const head = parts.slice(0, length);
  if (head.length < length || head.some((part) => part === "")) {
    throw new Error(`dependency graph check: cannot read the package specifier: ${specifier}`);
  }
  const rest = parts.slice(length);
  return { name: head.join("/"), subpath: rest.length === 0 ? "." : `./${rest.join("/")}` };
};

/**
 * The scheme of a specifier (`node:` / `npm:` / `jsr:` / `https:` …). `undefined` when there is
 * none.
 */
const schemeOf = (specifier: string): string | undefined =>
  /^([a-z][a-z0-9+.-]*):/i.exec(specifier)?.[1]?.toLowerCase();

const isRelative = (specifier: string): boolean =>
  specifier.startsWith("./") || specifier.startsWith("../") || specifier.startsWith("/");

/**
 * Derives the list of declarations (names) from a `package.json`'s `dependencies`. `workspace:` is
 * a member and `catalog:` is an external declaration that leaves its version to the workspace
 * catalog; both are counted by name.
 *
 * MUST: a local path declaration (`file:` / `link:` / `portal:` / a relative or absolute path)
 * throws instead of being guessed — in a published package it points into the user's machine, so it
 * is not a form this check may allow.
 */
const declaredPackagesOf = (dependencies: Readonly<Record<string, string>>): Set<string> => {
  const declared = new Set<string>();
  for (const [name, value] of Object.entries(dependencies)) {
    if (/^(file|link|portal):/.test(value) || /^(\.{1,2})?\//.test(value)) {
      throw new Error(
        `dependency graph check: a local path declaration (dependencies["${name}"] = ${value})`,
      );
    }
    declared.add(name);
  }
  return declared;
};

/**
 * Reads the `packages` list of `pnpm-workspace.yaml` (block-style `- glob` items).
 *
 * MUST: it does not read the whole YAML — a form it cannot read (flow style, `!` exclusions, globs
 * other than a single trailing `/*` level) throws instead of being guessed.
 */
const workspaceGlobsOf = (yaml: string): string[] => {
  const lines = yaml.split("\n");
  const start = lines.findIndex((line) => /^packages:\s*$/.test(line));
  if (start < 0)
    throw new Error("dependency graph check: pnpm-workspace.yaml has no packages list");
  const globs: string[] = [];
  for (const line of lines.slice(start + 1)) {
    if (/^\s*(#.*)?$/.test(line)) continue;
    if (!/^\s/.test(line)) break;
    const glob = /^\s+-\s+(["']?)([^"'#\s]+)\1\s*(#.*)?$/.exec(line)?.[2];
    if (glob === undefined || !/^[^!*]+(\/\*)?$/.test(glob)) {
      throw new Error(
        `dependency graph check: cannot read an item of packages in pnpm-workspace.yaml: ${line}`,
      );
    }
    globs.push(glob);
  }
  if (globs.length === 0) {
    throw new Error("dependency graph check: packages in pnpm-workspace.yaml is empty");
  }
  return globs;
};

/**
 * Expands one glob into member roots (with a trailing `/`; directories without a `package.json` do
 * not count — the same as pnpm).
 */
const memberRootsOf = async (glob: string): Promise<URL[]> => {
  if (!glob.endsWith("/*")) return [new URL(`${glob.replace(/\/+$/, "")}/`, REPO_ROOT)];
  const parent = new URL(glob.slice(0, -1), REPO_ROOT);
  const roots: URL[] = [];
  for await (const entry of readDir(parent)) {
    if (!entry.isDirectory) continue;
    const root = new URL(`${entry.name}/`, parent);
    let hasManifest = false;
    for await (const child of readDir(root)) {
      if (child.name === "package.json" && child.isFile) hasManifest = true;
    }
    if (hasManifest) roots.push(root);
  }
  return roots;
};

const readMember = async (memberRoot: URL): Promise<Member> => {
  const config = record(
    JSON.parse(await readTextFile(new URL("package.json", memberRoot))),
    `${memberRoot.href}package.json`,
  );
  const name = text(config.name, `name in ${memberRoot.href}package.json`);
  const isPrivate = config.private === true;
  // Conditional exports (`{ "import": …, "types": … }`) make text throw — a form that cannot be
  // read is not guessed. A private member without exports (a test-only home) has zero entries — it
  // is not subject to the check, and an edge from another member into it fails as "not a public
  // entry".
  const declaredExports: [string, unknown][] =
    typeof config.exports === "string"
      ? [[".", config.exports]]
      : config.exports === undefined && isPrivate
        ? []
        : Object.entries(record(config.exports, `exports of ${name}`));
  if (declaredExports.length === 0 && !isPrivate) {
    throw new Error(`dependency graph check: the exports of ${name} are empty`);
  }
  const exports = new Map(
    declaredExports.map(([subpath, entry]) => [
      subpath,
      new URL(text(entry, `${name} exports["${subpath}"]`), memberRoot).href,
    ]),
  );
  const dependencies = Object.fromEntries(
    Object.entries(record(config.dependencies ?? {}, `dependencies of ${name}`)).map(
      ([key, value]) => [key, text(value, `${name} dependencies["${key}"]`)],
    ),
  );
  return {
    name,
    root: memberRoot.href,
    private: isPrivate,
    exports,
    entries: [...exports.values()],
    declared: declaredPackagesOf(dependencies),
  };
};

const readMembers = async (): Promise<Member[]> => {
  const globs = workspaceGlobsOf(await readTextFile(new URL("pnpm-workspace.yaml", REPO_ROOT)));
  const roots = (await Promise.all(globs.map(memberRootsOf))).flat();
  const members = await Promise.all(roots.map(readMember));
  return members.sort((a, b) => a.name.localeCompare(b.name));
};

/** One graph edge (the source module → the written specifier → the resolved target). */
type Edge = ModuleEdge;

/**
 * The verdict. `external` is the number of allowed external edges (to tell apart a check that,
 * after a change of form, can read no external edge at all).
 */
type Verdict = {
  readonly violations: readonly string[];
  readonly external: number;
};

/**
 * The member surface the verdict needs (`root` is left out so that unit tests of the verdict can
 * pass rootless models).
 */
type Owner = Pick<Member, "name" | "entries" | "declared">;

/**
 * Judges the edges one by one. `ownerOf` returns the member a file belongs to (`undefined` for a
 * file that belongs to no member).
 */
const judgeEdges = (
  edges: readonly Edge[],
  ownerOf: (file: string) => Owner | undefined,
): Verdict => {
  const violations: string[] = [];
  let external = 0;
  for (const { from, written, resolved } of edges) {
    const label = `${from} → "${written}"（${resolved}）`;
    const source = ownerOf(from);
    const sourceName = source?.name ?? "a file that belongs to no member";
    const scheme = schemeOf(written);
    if (scheme === "node") {
      if (!NODE_BUILTINS_ALLOWED_FROM.has(from)) {
        violations.push(`node: built-in module (not a Web standard API): ${label}`);
      }
      continue;
    }
    if (scheme !== undefined) {
      violations.push(`${scheme}: specifier (forbidden whether declared or not): ${label}`);
      continue;
    }
    if (resolved.startsWith("file:")) {
      const target = ownerOf(resolved);
      if (target === undefined) {
        violations.push(`import of a file that belongs to no member: ${label}`);
      } else if (target.name === source?.name) {
        // Inside the same member (a relative import, or a self-reference by its own name).
      } else if (!target.entries.includes(resolved)) {
        violations.push(
          `cross-member import of a file that is not a public entry of ${target.name}: ${label}`,
        );
      } else if (isRelative(written)) {
        violations.push(
          `cross-member relative import (an entry of ${target.name} is reached through the "${target.name}" ` +
            `specifier): ${label}`,
        );
      } else if (source === undefined || !source.declared.has(target.name)) {
        violations.push(
          `dependency on a member not declared in the dependencies of the package.json of ${sourceName}: ` +
            label,
        );
      }
      continue;
    }
    if (source !== undefined && source.declared.has(packageOf(written).name)) {
      external++;
      continue;
    }
    violations.push(
      `external dependency not declared in the dependencies of the package.json of ${sourceName}: ${label}`,
    );
  }
  return { violations, external };
};

/**
 * The resolved target of a non-relative specifier. A specifier with a member's name
 * (`@karume/hub/node` and so on) resolves to that member's `exports` entry file; anything else
 * (external, with a scheme) is returned as written (not followed).
 *
 * MUST: a subpath missing from the member's `exports` throws instead of being guessed (a form that
 * Node's resolution would fail on too).
 */
const resolverOf =
  (members: readonly Member[]) =>
  (from: string, written: string): string => {
    if (schemeOf(written) !== undefined) return written;
    const { name, subpath } = packageOf(written);
    const member = members.find((candidate) => candidate.name === name);
    if (member === undefined) return written;
    const entry = member.exports.get(subpath);
    if (entry === undefined) {
      throw new Error(
        `dependency graph check: an import that cannot be resolved: ${from} → ${written}` +
          ` (${name} has no ${subpath} in its exports)`,
      );
    }
    return entry;
  };

/** Judges the graph of every entry of one member. */
const judgeMember = async (member: Member, members: readonly Member[]): Promise<Verdict> => {
  const ownerOf = (file: string): Member | undefined =>
    members.find((candidate) => file.startsWith(candidate.root));
  const edges = await walkModuleGraph(member.entries, resolverOf(members));
  return judgeEdges(edges, ownerOf);
};

const members = await readMembers();

describe("package dependency-graph check", () => {
  for (const member of members.filter((candidate) => !candidate.private)) {
    it(`the external dependencies reached from the public entries of ${member.name} are only those its source package declares`, async () => {
      const { violations } = await judgeMember(member, members);
      assertEquals(
        violations,
        [],
        "in the workspace undeclared dependencies still resolve locally, so the type check passes, " +
          "but there are dependencies that do not resolve in the published package (or are not closed over Web standard APIs alone)",
      );
    });
  }

  it("control: counts hub's declared @hdae/fetch-cache edge as an allowed external edge", async () => {
    // If a change in how edges are taken left no external edge readable, the check above would go
    // silently green. hub really uses `@hdae/fetch-cache`, so this checks that the edge is read.
    const hub = members.find((member) => member.name === "@karume/hub");
    assert(hub !== undefined, "the workspace has no @karume/hub");
    const { external } = await judgeMember(hub, members);
    assert(
      external > 0,
      "hub's graph has no allowed external edge at all (the source edges are not being read)",
    );
  });
});

describe("verdicts of the dependency-graph check", () => {
  const owner = {
    name: "@karume/example",
    entries: ["file:///repo/packages/example/mod.ts"],
    declared: declaredPackagesOf({ "@x/y": "catalog:", "@karume/other": "workspace:*" }),
  };
  const other = {
    name: "@karume/other",
    entries: ["file:///repo/packages/other/mod.ts"],
    declared: new Set<string>(),
  };
  const ownerOf = (path: string) =>
    path.startsWith("file:///repo/packages/example/")
      ? owner
      : path.startsWith("file:///repo/packages/other/")
        ? other
        : undefined;
  const file = "file:///repo/packages/example/src/a.ts";

  it("allows files inside the same member and public entries of other members", () => {
    const verdict = judgeEdges(
      [
        { from: file, written: "./b.ts", resolved: "file:///repo/packages/example/src/b.ts" },
        { from: file, written: "@karume/other", resolved: "file:///repo/packages/other/mod.ts" },
      ],
      ownerOf,
    );
    assertEquals(verdict, { violations: [], external: 0 });
  });

  it("rejects a root-crossing relative import of a file that is not another member's entry", () => {
    const { violations } = judgeEdges(
      [
        {
          from: file,
          written: "../../other/src/x.ts",
          resolved: "file:///repo/packages/other/src/x.ts",
        },
      ],
      ownerOf,
    );
    assertEquals(violations.length, 1);
    assert(violations[0]?.includes("not a public entry of @karume/other"), violations[0]);
  });

  it("rejects an import of a workspace file that belongs to no member", () => {
    const { violations } = judgeEdges(
      [{ from: file, written: "../../../shared.ts", resolved: "file:///repo/shared.ts" }],
      ownerOf,
    );
    assertEquals(violations.length, 1);
    assert(violations[0]?.includes("belongs to no member"), violations[0]);
  });

  it("a local path declaration (file: / link:) is not put on the allow list and throws", () => {
    assertEquals(
      declaredPackagesOf({ "@karume/other": "workspace:*", "@x/y": "catalog:", z: "^1.2.0" }),
      new Set(["@karume/other", "@x/y", "z"]),
    );
    assertThrows(() => declaredPackagesOf({ util: "file:../util" }), Error, "local path");
    assertThrows(() => declaredPackagesOf({ util: "link:../util" }), Error, "local path");
  });

  it("allows a declared external package even through a subpath", () => {
    const verdict = judgeEdges(
      [{ from: file, written: "@x/y/sub", resolved: "@x/y/sub" }],
      () => owner,
    );
    assertEquals(verdict, { violations: [], external: 1 });
  });

  it("rejects an undeclared external package and a dependency on an undeclared member", () => {
    const { violations } = judgeEdges(
      [
        { from: file, written: "jpeg-js", resolved: "jpeg-js" },
        {
          from: "file:///repo/packages/other/mod.ts",
          written: "@karume/example",
          resolved: "file:///repo/packages/example/mod.ts",
        },
      ],
      ownerOf,
    );
    assertEquals(violations.length, 2);
    assert(violations[0]?.includes("external dependency not declared"), violations[0]);
    assert(violations[1]?.includes("dependency on a member not declared"), violations[1]);
  });

  it("rejects npm: / jsr: specifiers whether declared or not", () => {
    const { violations } = judgeEdges(
      [
        { from: file, written: "npm:jpeg-js@^0.4.4", resolved: "npm:jpeg-js@^0.4.4" },
        { from: file, written: "jsr:@x/y@^1.2.0", resolved: "jsr:@x/y@^1.2.0" },
      ],
      () => owner,
    );
    assertEquals(violations.length, 2);
    assert(violations[0]?.includes("npm: specifier"), violations[0]);
    assert(violations[1]?.includes("jsr: specifier"), violations[1]);
  });

  it("allows node: only from the allow-listed file (hub's node.ts) and rejects it from any other file", () => {
    const hubNode = new URL("packages/hub/node.ts", REPO_ROOT).href;
    const { violations } = judgeEdges(
      [
        { from: hubNode, written: "node:fs/promises", resolved: "node:fs/promises" },
        { from: file, written: "node:fs", resolved: "node:fs" },
      ],
      () => owner,
    );
    assertEquals(violations.length, 1);
    assert(violations[0]?.includes(`${file} → "node:fs"`), violations[0]);
  });

  describe("edges taken from the sources (type-only edges / imports in type positions / dynamic imports)", () => {
    // Edges are taken from the sources with oxc-parser. File reads are replaced with a stub.
    const impl = "file:///repo/packages/example/src/impl.ts";
    const graphOf = (sources: Readonly<Record<string, string>>) =>
      walkModuleGraph([impl], resolverOf([]), (url) => {
        const source = sources[url];
        return source === undefined
          ? Promise.reject(new Error(`not found: ${url}`))
          : Promise.resolve(source);
      });

    it("rejects an edge that points at an npm type package", async () => {
      const edges = await graphOf({ [impl]: `import type { X } from "npm:@types/x";\n` });
      const { violations } = judgeEdges(edges, ownerOf);
      assertEquals(violations.length, 1);
      assert(violations[0]?.includes("npm:@types/x"), violations[0]);
    });

    it("reads an edge that points at a file inside the same member and allows it", async () => {
      const edges = await graphOf({
        [impl]:
          `export type { T } from "./types.ts";\n` +
          `export const lazy = () => import("./lazy.ts");\n`,
        "file:///repo/packages/example/src/types.ts": "export type T = number;\n",
        "file:///repo/packages/example/src/lazy.ts": "export {};\n",
      });
      assertEquals(edges, [
        {
          from: impl,
          written: "./types.ts",
          resolved: "file:///repo/packages/example/src/types.ts",
        },
        { from: impl, written: "./lazy.ts", resolved: "file:///repo/packages/example/src/lazy.ts" },
      ]);
      assertEquals(judgeEdges(edges, ownerOf), { violations: [], external: 0 });
    });

    it("throws when the syntax cannot be parsed", async () => {
      await assertRejects(
        () => graphOf({ [impl]: "import { from './x.ts'\n" }),
        Error,
        "cannot parse",
      );
    });

    it("rejects an edge that points at a type file belonging to no member", async () => {
      // An `import("…")` in a type position that points outside the members.
      const edges = await graphOf({
        [impl]: `export type U = import("../../../outside/types.ts").U;\n`,
        "file:///repo/outside/types.ts": "export type U = string;\n",
      });
      const { violations } = judgeEdges(edges, ownerOf);
      assertEquals(violations.length, 1);
      assert(violations[0]?.includes("belongs to no member"), violations[0]);
    });

    it("throws instead of guessing when the target file of an edge cannot be read", async () => {
      await assertRejects(
        () => graphOf({ [impl]: `export { y } from "./missing.ts";\n` }),
        Error,
        "cannot read file:///repo/packages/example/src/missing.ts",
      );
    });
  });
});
