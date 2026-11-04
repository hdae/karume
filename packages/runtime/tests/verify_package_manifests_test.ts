// The package manifest gate: each published package's `deno.json` and `package.json` agree on
// `name`, `version` and `exports` (ADR 0127 — the lockstep bump of docs/release-runbook.md).
//
// The two manifests describe the same package to two consumers. JSR publishes from `deno.json`
// (`deno publish` at the repository root walks the root `deno.json` `workspace`), while the pnpm
// workspace, the type check, the dependency graph gate (`dependency_graph_test.ts` reads the
// `package.json` entry points) and `tools/published-smoke` (the expected version comes from
// `packages/models/package.json`) read `package.json`. A subpath added to one and not the other is
// published but untested, or tested but unpublished; a version bumped in one only makes the smoke
// check expect a version JSR never got. Nothing else compares the two, so this test does.
//
// The published set is the root `deno.json` `workspace` — the same list `deno publish` reads, so a
// new package is covered the day it becomes publishable. It runs in the core lane like
// `verify_lanes_test.ts` (no GPU, no assets; it reads seven small JSON files).
//
// MUST: an unreadable shape throws instead of counting as agreement (a missing file, an `exports`
// with conditions or a non-string target) — reading "could not compare" as "equal" hides a drift.

import { pathToFileURL } from "node:url";
import { assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import {
  makeTempDirSync,
  mkdirSync,
  readTextFileSync,
  removeSync,
  writeTextFileSync,
} from "@karume/test-support/fs";
import { describe, it, test } from "vite-plus/test";

const REPO_ROOT = new URL("../../../", import.meta.url);

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const readJson = (url: URL): Record<string, unknown> => {
  const value: unknown = JSON.parse(readTextFileSync(url));
  if (!isRecord(value)) throw new Error(`package manifest gate: ${url.pathname} is not an object`);
  return value;
};

/**
 * `exports` as a subpath → file map (a bare string is the `"."` entry in both formats). Anything
 * else — conditions, arrays, non-string targets — is a shape this gate does not compare, so it
 * throws.
 */
const exportMap = (value: unknown, where: string): Readonly<Record<string, string>> => {
  if (typeof value === "string") return { ".": value };
  if (!isRecord(value)) throw new Error(`package manifest gate: ${where} exports is not a map`);
  const map: Record<string, string> = {};
  for (const [subpath, target] of Object.entries(value)) {
    if (typeof target !== "string") {
      throw new Error(`package manifest gate: ${where} exports['${subpath}'] is not a file path`);
    }
    map[subpath] = target;
  }
  return map;
};

/** The subpaths whose entry differs between the two maps (missing on one side, or another file). */
const differingSubpaths = (
  left: Readonly<Record<string, string>>,
  right: Readonly<Record<string, string>>,
): string[] =>
  [...new Set([...Object.keys(left), ...Object.keys(right)])]
    .filter(
      (key) => !Object.hasOwn(left, key) || !Object.hasOwn(right, key) || left[key] !== right[key],
    )
    .sort();

/**
 * Every disagreement between the `deno.json` and the `package.json` of the packages the root
 * `deno.json` publishes (one sentence each, naming the member and the field; empty when they agree).
 */
const packageManifestMismatches = (root: URL): string[] => {
  const workspace = readJson(new URL("deno.json", root))["workspace"];
  if (!Array.isArray(workspace) || workspace.length === 0) {
    throw new Error("package manifest gate: the root deno.json has no workspace list");
  }
  const mismatches: string[] = [];
  for (const member of workspace) {
    if (typeof member !== "string") {
      throw new Error("package manifest gate: a root deno.json workspace entry is not a path");
    }
    const directory = new URL(member.replace(/\/?$/, "/"), root);
    const deno = readJson(new URL("deno.json", directory));
    const pkg = readJson(new URL("package.json", directory));
    for (const field of ["name", "version"] as const) {
      if (deno[field] !== pkg[field]) {
        mismatches.push(
          `${member}: ${field} is ${JSON.stringify(deno[field])} in deno.json but ` +
            `${JSON.stringify(pkg[field])} in package.json`,
        );
      }
    }
    const denoExports = exportMap(deno["exports"], `${member}/deno.json`);
    const pkgExports = exportMap(pkg["exports"], `${member}/package.json`);
    for (const subpath of differingSubpaths(denoExports, pkgExports)) {
      mismatches.push(
        `${member}: exports['${subpath}'] is ${JSON.stringify(denoExports[subpath])} in deno.json ` +
          `but ${JSON.stringify(pkgExports[subpath])} in package.json`,
      );
    }
  }
  return mismatches;
};

test("every published package's deno.json and package.json agree on name, version and exports", () => {
  assertEquals(
    packageManifestMismatches(REPO_ROOT),
    [],
    "JSR publishes deno.json and the workspace reads package.json: bump and edit both " +
      "(docs/release-runbook.md, the lockstep bump commit).",
  );
});

describe("package manifest gate (fabricated workspaces)", () => {
  const manifest = { name: "@karume/pkg", version: "1.2.3", exports: { ".": "./mod.ts" } };

  /** Writes a workspace with one member `packages/pkg` and returns the mismatches it reports. */
  const mismatchesOf = (
    deno: Readonly<Record<string, unknown>>,
    pkg: Readonly<Record<string, unknown>>,
  ): string[] => {
    const dir = makeTempDirSync({ prefix: "karume-manifest-gate-" });
    try {
      const root = pathToFileURL(`${dir}/`);
      writeTextFileSync(
        new URL("deno.json", root),
        JSON.stringify({ workspace: ["./packages/pkg"] }),
      );
      mkdirSync(new URL("packages/pkg/", root), { recursive: true });
      writeTextFileSync(new URL("packages/pkg/deno.json", root), JSON.stringify(deno));
      writeTextFileSync(new URL("packages/pkg/package.json", root), JSON.stringify(pkg));
      return packageManifestMismatches(root);
    } finally {
      removeSync(dir, { recursive: true });
    }
  };

  it("reports nothing when the two manifests agree (exports in another key order too)", () => {
    const two = { ".": "./mod.ts", "./node": "./node.ts" };
    const reversed = { "./node": "./node.ts", ".": "./mod.ts" };
    assertEquals(
      mismatchesOf({ ...manifest, exports: two }, { ...manifest, exports: reversed }),
      [],
    );
  });

  it("treats a bare string export as the '.' entry", () => {
    assertEquals(mismatchesOf({ ...manifest, exports: "./mod.ts" }, manifest), []);
  });

  it("reports a version bumped in deno.json only, naming the member and both values", () => {
    const [mismatch, ...rest] = mismatchesOf({ ...manifest, version: "1.2.4" }, manifest);
    assertEquals(rest, []);
    assertStringIncludes(mismatch!, "./packages/pkg: version");
    assertStringIncludes(mismatch!, `"1.2.4" in deno.json`);
    assertStringIncludes(mismatch!, `"1.2.3" in package.json`);
  });

  it("reports a version missing from package.json", () => {
    const unversioned = { name: manifest.name, exports: manifest.exports };
    assertEquals(mismatchesOf(manifest, unversioned).length, 1);
  });

  it("reports a name that differs", () => {
    const [mismatch] = mismatchesOf(manifest, { ...manifest, name: "@karume/other" });
    assertStringIncludes(mismatch!, "./packages/pkg: name");
  });

  it("reports a subpath published by deno.json but missing from package.json", () => {
    const deno = { ...manifest, exports: { ".": "./mod.ts", "./node": "./node.ts" } };
    const [mismatch, ...rest] = mismatchesOf(deno, manifest);
    assertEquals(rest, []);
    assertStringIncludes(mismatch!, "./packages/pkg: exports['./node']");
    assertStringIncludes(mismatch!, `"./node.ts" in deno.json but undefined in package.json`);
  });

  it("reports a subpath that points at another file", () => {
    const pkg = { ...manifest, exports: { ".": "./index.ts" } };
    assertEquals(mismatchesOf(manifest, pkg).length, 1);
  });

  it("throws on exports with conditions instead of counting them as equal", () => {
    const conditional = { ...manifest, exports: { ".": { import: "./mod.ts" } } };
    assertThrows(() => mismatchesOf(manifest, conditional), Error, "is not a file path");
  });

  it("throws when a published member has no package.json", () => {
    const dir = makeTempDirSync({ prefix: "karume-manifest-gate-" });
    try {
      const root = pathToFileURL(`${dir}/`);
      writeTextFileSync(new URL("deno.json", root), JSON.stringify({ workspace: ["./pkg"] }));
      mkdirSync(new URL("pkg/", root));
      writeTextFileSync(new URL("pkg/deno.json", root), JSON.stringify(manifest));
      assertThrows(() => packageManifestMismatches(root));
    } finally {
      removeSync(dir, { recursive: true });
    }
  });
});
