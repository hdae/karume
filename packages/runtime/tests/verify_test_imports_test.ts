// The test import rules of ADR 0129 (open point 4, settled by the addendum of 2026-10-29). The
// private test packages may reach into the published packages, never the other way round:
//
// - **No runtime, hub or models test reaches `packages/e2e`.** The e2e package holds the
//   real-weight tests that run only in the family lanes; a published package's test that imports
//   one of its helpers drags local assets and a GPU into the core lane.
// - **Only the runtime GPU tests (`packages/runtime/tests/gpu_*_test.ts`, through their helpers)
//   and the playground reach `@karume/kernel-checks`.** The kernel identity checks drive runtime
//   internals; ADR 0129 names exactly those two consumers.
//
// `dependency_graph_test.ts` cannot see either rule: it walks the published entry points only, and
// tests are not entry points. This gate walks the static module graph
// (`@karume/test-support/module-graph`) from every root of the workspace — every `*_test.ts` and
// every source file outside a `tests/` directory (a file under `tests/` that is not a test is a
// helper or a fixture, reached through the tests that import it) — and fails on a root that
// reaches a forbidden file through any chain of imports, naming the chain. Roots are the tracked
// and the untracked-but-not-ignored files (`git ls-files --cached --others --exclude-standard`),
// so a new test file is checked before it is added. The playground and `packages/kernel-checks`
// themselves are not roots of the kernel-checks rule (they are its allowed side).
//
// MUST: an unresolvable `@karume/*` specifier throws, like the dependency graph gate — a chain
// that the walk cannot follow would otherwise read as "reaches nothing".

import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { assert, assertEquals, assertRejects, assertStringIncludes } from "@std/assert";
import {
  type ModuleEdge,
  type ReadSource,
  walkModuleGraph,
} from "@karume/test-support/module-graph";
import { describe, it, test } from "vite-plus/test";

const REPO_ROOT = new URL("../../../", import.meta.url);

/** A workspace member as the resolver sees it: its directory and its `exports` (subpath → file). */
type Member = {
  readonly directory: string;
  readonly exports: ReadonlyMap<string, string>;
};

/** The inputs of one judgement: repository-relative paths, resolved against `root`. */
type Workspace = {
  readonly root: URL;
  /** Every TypeScript file of the workspace (repository-relative). */
  readonly files: readonly string[];
  /** `@karume/*` package name → member. */
  readonly members: ReadonlyMap<string, Member>;
  readonly readSource: ReadSource;
};

type Verdict = {
  /** One sentence per root that breaks the e2e rule (the chain from the root to the import). */
  readonly e2e: readonly string[];
  /** One sentence per root that breaks the kernel-checks rule. */
  readonly kernelChecks: readonly string[];
  /** Allowed roots that reach `@karume/kernel-checks` (a control: zero means the walk saw nothing). */
  readonly kernelChecksConsumers: number;
};

const TYPESCRIPT = /\.(ts|tsx|mts|cts)$/;

const packageNameOf = (specifier: string): string =>
  specifier
    .split("/")
    .slice(0, specifier.startsWith("@") ? 2 : 1)
    .join("/");

const resolverOf =
  (members: ReadonlyMap<string, Member>) =>
  (from: string, written: string): string => {
    if (!written.startsWith("@karume/")) return written;
    const name = packageNameOf(written);
    const member = members.get(name);
    if (member === undefined) {
      throw new Error(`test import gate: ${from} imports "${written}", not a workspace member`);
    }
    // A member without `exports` (the e2e package, the tools) has no importable entry; the
    // specifier is kept as the edge's target and judged by its package name.
    if (member.exports.size === 0) return written;
    const subpath = `.${written.slice(name.length)}`;
    const entry = member.exports.get(subpath);
    if (entry === undefined) {
      throw new Error(
        `test import gate: ${from} imports "${written}", which ${name} does not export`,
      );
    }
    return entry;
  };

const isTestFile = (path: string): boolean => /_test\.(ts|tsx|mts|cts)$/.test(path);
const isUnderTests = (path: string): boolean => /(^|\/)tests\//.test(path);

/**
 * The roots that break a rule: every root that reaches an edge `forbidden` accepts, through any
 * chain, and that `allowed` does not accept. The chain is reconstructed from the next hop each
 * tainted module records (breadth-first from the forbidden edges, so it is a shortest chain).
 */
const reachingRoots = (
  edges: readonly ModuleEdge[],
  roots: readonly string[],
  forbidden: (edge: ModuleEdge) => boolean,
): Map<string, string[]> => {
  const importers = new Map<string, string[]>();
  for (const edge of edges) {
    const list = importers.get(edge.resolved) ?? [];
    list.push(edge.from);
    importers.set(edge.resolved, list);
  }
  /**
   * Module → the next step of its chain: another module, or the end of the chain (the forbidden
   * file, or the specifier as written when it resolves to no file).
   */
  const next = new Map<string, { readonly module?: string; readonly end?: string }>();
  const queue: string[] = [];
  for (const edge of edges) {
    if (forbidden(edge) && !next.has(edge.from)) {
      const end = edge.resolved.startsWith("file:") ? edge.resolved : `"${edge.written}"`;
      next.set(edge.from, { end });
      queue.push(edge.from);
    }
  }
  while (queue.length > 0) {
    const module = queue.shift();
    if (module === undefined) break;
    for (const importer of importers.get(module) ?? []) {
      if (next.has(importer)) continue;
      next.set(importer, { module });
      queue.push(importer);
    }
  }
  const chains = new Map<string, string[]>();
  for (const root of roots) {
    if (!next.has(root)) continue;
    const chain: string[] = [];
    let step: string | undefined = root;
    while (step !== undefined) {
      chain.push(step);
      const hop = next.get(step);
      if (hop?.end !== undefined) chain.push(hop.end);
      step = hop?.module;
    }
    chains.set(root, chain);
  }
  return chains;
};

const judgeTestImports = async (workspace: Workspace): Promise<Verdict> => {
  const { root, files, members, readSource } = workspace;
  const href = (path: string): string => new URL(path, root).href;
  const relative = (url: string): string =>
    url.startsWith(root.href) ? url.slice(root.href.length) : url;
  const describeChain = (chain: readonly string[]): string => chain.map(relative).join(" → ");

  const roots = files.filter((path) => isTestFile(path) || !isUnderTests(path));
  const edges = await walkModuleGraph(roots.map(href), resolverOf(members), readSource);

  const e2eDirectory = href("packages/e2e/");
  const publishedTest = /^packages\/(runtime|hub|models)\/tests\/.*_test\.(ts|tsx|mts|cts)$/;
  const e2eRoots = roots.filter((path) => publishedTest.test(path)).map(href);
  const e2e = [
    ...reachingRoots(
      edges,
      e2eRoots,
      (edge) =>
        edge.resolved.startsWith(e2eDirectory) || packageNameOf(edge.written) === "@karume/e2e",
    ).values(),
  ].map((chain) => `${describeChain(chain)} — a published package's test reaches packages/e2e`);

  const kernelChecksDirectory = href("packages/kernel-checks/");
  const allowedSide = /^(apps\/playground|packages\/kernel-checks)\//;
  const gpuTest = /^packages\/runtime\/tests\/gpu_[^/]*_test\.ts$/;
  const kernelRoots = roots.filter((path) => !allowedSide.test(path));
  const kernelChains = reachingRoots(
    edges,
    kernelRoots.map(href),
    (edge) =>
      edge.resolved.startsWith(kernelChecksDirectory) ||
      packageNameOf(edge.written) === "@karume/kernel-checks",
  );
  const kernelChecks: string[] = [];
  let kernelChecksConsumers = 0;
  for (const [rootUrl, chain] of kernelChains) {
    if (gpuTest.test(relative(rootUrl))) {
      kernelChecksConsumers += 1;
      continue;
    }
    kernelChecks.push(
      `${describeChain(chain)} — only the runtime GPU tests and the playground may reach @karume/kernel-checks`,
    );
  }
  return { e2e, kernelChecks, kernelChecksConsumers };
};

/** Reads a module for the walk: a non-TypeScript import (`.json`, `.wgsl?raw`) has no edges. */
const readFromDisk: ReadSource = async (url) => {
  const file = new URL(url);
  file.search = "";
  return TYPESCRIPT.test(file.pathname) ? readFileSync(file, "utf8") : "";
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const repositoryWorkspace = (): Workspace => {
  const listed = execFileSync(
    "git",
    [
      "-C",
      fileURLToPath(REPO_ROOT),
      "ls-files",
      "-z",
      "--cached",
      "--others",
      "--exclude-standard",
      "--",
      "packages",
      "tools",
      "apps",
    ],
    { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 },
  )
    .split("\0")
    .filter((path) => path.length > 0);
  // A tracked file deleted in the working tree is listed by `--cached` but has nothing to read.
  const present = [...new Set(listed)].filter((path) => existsSync(new URL(path, REPO_ROOT)));
  const members = new Map<string, Member>();
  for (const manifest of present.filter((path) =>
    /^(packages|tools|apps)\/[^/]+\/package\.json$/.test(path),
  )) {
    const config: unknown = JSON.parse(readFileSync(new URL(manifest, REPO_ROOT), "utf8"));
    if (!isRecord(config) || typeof config["name"] !== "string") {
      throw new Error(`test import gate: ${manifest} has no name`);
    }
    const directory = new URL("./", new URL(manifest, REPO_ROOT));
    const declared = config["exports"] ?? {};
    const entries = typeof declared === "string" ? { ".": declared } : declared;
    if (!isRecord(entries)) throw new Error(`test import gate: ${manifest} exports is not a map`);
    const exports = new Map<string, string>();
    for (const [subpath, target] of Object.entries(entries)) {
      if (typeof target !== "string") {
        throw new Error(`test import gate: ${manifest} exports['${subpath}'] is not a file path`);
      }
      exports.set(subpath, new URL(target, directory).href);
    }
    members.set(config["name"], { directory: directory.href, exports });
  }
  return {
    root: REPO_ROOT,
    files: present.filter((path) => TYPESCRIPT.test(path)),
    members,
    readSource: readFromDisk,
  };
};

test("no runtime, hub or models test reaches packages/e2e, and only the runtime GPU tests and the playground reach @karume/kernel-checks (ADR 0129)", async () => {
  const verdict = await judgeTestImports(repositoryWorkspace());
  assertEquals(verdict.e2e, [], "move the shared code out of packages/e2e (ADR 0129 open point 4)");
  assertEquals(verdict.kernelChecks, [], "ADR 0129 open point 4 names the two consumers");
  assert(
    verdict.kernelChecksConsumers > 0,
    "no runtime GPU test reaches @karume/kernel-checks — the walk no longer sees the imports it gates",
  );
});

describe("test import gate (fabricated workspaces)", () => {
  const root = new URL("file:///workspace/");
  const at = (path: string): string => new URL(path, root).href;
  const members = new Map<string, Member>([
    [
      "@karume/runtime",
      {
        directory: at("packages/runtime/"),
        exports: new Map([[".", at("packages/runtime/mod.ts")]]),
      },
    ],
    [
      "@karume/kernel-checks",
      {
        directory: at("packages/kernel-checks/"),
        exports: new Map([[".", at("packages/kernel-checks/mod.ts")]]),
      },
    ],
    ["@karume/e2e", { directory: at("packages/e2e/"), exports: new Map() }],
  ]);
  const base: Readonly<Record<string, string>> = {
    "packages/runtime/mod.ts": "export const run = 1;",
    "packages/kernel-checks/mod.ts": `import { run } from "@karume/runtime"; export const check = run;`,
    "packages/e2e/tests/helpers/assets.ts": "export const assets = 1;",
    "packages/runtime/tests/helpers/kernel-check-assert.ts": `export { check } from "@karume/kernel-checks";`,
    "packages/runtime/tests/gpu_subgroup_test.ts": `import { check } from "./helpers/kernel-check-assert.ts";`,
    "apps/playground/src/page.tsx": `import { check } from "@karume/kernel-checks";`,
    "packages/hub/tests/fetch_test.ts": `import "vite-plus/test";`,
  };

  const judge = (overrides: Readonly<Record<string, string>> = {}): Promise<Verdict> => {
    const sources = { ...base, ...overrides };
    return judgeTestImports({
      root,
      files: Object.keys(sources),
      members,
      readSource: async (url) => {
        const path = url.slice(root.href.length);
        const source = sources[path];
        if (source === undefined) throw new Error(`no such file ${path}`);
        return source;
      },
    });
  };

  it("passes the allowed shape: a runtime GPU test through its helper, and the playground", async () => {
    assertEquals(await judge(), { e2e: [], kernelChecks: [], kernelChecksConsumers: 1 });
  });

  it("flags a hub test that imports an e2e helper by relative path", async () => {
    const verdict = await judge({
      "packages/hub/tests/fetch_test.ts": `import { assets } from "../../e2e/tests/helpers/assets.ts";`,
    });
    assertEquals(verdict.e2e.length, 1, verdict.e2e.join("\n"));
    assertStringIncludes(
      verdict.e2e[0]!,
      "packages/hub/tests/fetch_test.ts → packages/e2e/tests/helpers/assets.ts",
    );
  });

  it("flags a models test that reaches @karume/e2e through a helper, naming the chain", async () => {
    const verdict = await judge({
      "packages/models/tests/helpers/shared.ts": `export * from "@karume/e2e";`,
      "packages/models/tests/pipeline_test.ts": `import "./helpers/shared.ts";`,
    });
    assertEquals(verdict.e2e, [
      `packages/models/tests/pipeline_test.ts → packages/models/tests/helpers/shared.ts → "@karume/e2e" — a published package's test reaches packages/e2e`,
    ]);
  });

  it("does not apply the e2e rule to the e2e tests themselves", async () => {
    const verdict = await judge({
      "packages/e2e/tests/e2e_anima_test.ts": `import "./helpers/assets.ts";`,
    });
    assertEquals(verdict.e2e, []);
  });

  it("flags a models test that imports @karume/kernel-checks", async () => {
    const verdict = await judge({
      "packages/models/tests/gemv_test.ts": `import { check } from "@karume/kernel-checks";`,
    });
    assertEquals(verdict.kernelChecks.length, 1, verdict.kernelChecks.join("\n"));
    assertStringIncludes(
      verdict.kernelChecks[0]!,
      "packages/models/tests/gemv_test.ts → packages/kernel-checks/mod.ts",
    );
  });

  it("flags a runtime test that is not a GPU test, even through the GPU tests' helper", async () => {
    const verdict = await judge({
      "packages/runtime/tests/linear_rows_test.ts": `import "./helpers/kernel-check-assert.ts";`,
    });
    assertEquals(verdict.kernelChecks.length, 1, verdict.kernelChecks.join("\n"));
    assertStringIncludes(verdict.kernelChecks[0]!, "packages/runtime/tests/linear_rows_test.ts →");
  });

  it("flags an e2e test and a published source file that reach kernel-checks", async () => {
    const verdict = await judge({
      "packages/e2e/tests/e2e_gemv_test.ts": `import "../../kernel-checks/mod.ts";`,
      "packages/runtime/src/check.ts": `import "@karume/kernel-checks";`,
    });
    assertEquals(verdict.kernelChecks.length, 2, verdict.kernelChecks.join("\n"));
  });

  it("throws on a @karume subpath the member does not export instead of reading it as no edge", async () => {
    await assertRejects(
      () => judge({ "packages/hub/tests/fetch_test.ts": `import "@karume/runtime/internal";` }),
      Error,
      "does not export",
    );
  });
});
