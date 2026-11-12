// The coverage check of the test lanes (ADR 0005 addendum 2026-09-20).
//
// A lane (`LANES` of `packages/test-support/src/lanes.ts` — run as a Vitest project with
// `vp test --project <lane>`) is an execution unit of the split full verify (`vp test`). The only
// danger of the split is a **coverage gap** — a new test that lands in no lane, never runs in a
// lane run, and still looks "green". This is the check that does not let that pass silently; it
// looks at the following 4 things and at the Vitest config passing `--expose-gc` to the workers
// (needed by the relaxation of the tests' GPU acquisition — EXPOSE_GC_FLAG below):
//
// - **Coverage**: core ∪ all lanes = every `*_test.ts` of the repository (under `packages/*/tests`
//   and `tools/*/tests`).
// - **No overlap**: core and the family lanes are disjoint (the same heavy e2e is not paid twice).
//   Overlap between family lanes is allowed — files that need both the regular and the QAT
//   distribution exist.
// - **Not empty**: each lane collects at least one file. Vitest fails a misspelled lane with "No
//   test files found", but only when that lane is run.
// - **Spelling**: test names are `*_test.ts` only. Vitest's `include` is `*_test.ts`, so the
//   spellings `*.test.ts` / `test.ts` look green **without ever being collected**. This is stopped
//   on the rejecting side rather than by widening the population.
//
// Two more checks guard the lane table itself. Coverage and disjointness still hold when a family
// row is dropped from `FAMILY_LANES` (its files fall back into core), so core may hold no
// end-to-end test but the GPU-free gates of `CORE_E2E_TESTS`, and a family lane that claims no
// end-to-end test is named in `HOST_ONLY_FAMILY_LANES` and must exist.
//
// The source of truth is `LANES` itself (the projects of `vite.config.ts` are built from the same
// table with `laneProjects()`). This check expands the globs itself (`*` does not cross `/`, `**`
// with a separator is zero or more directories).

import { readDirSync } from "@karume/test-support/fs";
import {
  ALL_TESTS,
  defaultLanes,
  laneProjects,
  LANES,
  type Lane,
  LOW_PRIORITY_FAMILIES,
} from "@karume/test-support/lanes";
import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import viteConfig from "../../../vite.config.ts";

/** The repository root (lane globs are relative to the repository). */
const REPO_ROOT = new URL("../../../", import.meta.url);

/**
 * Turns a glob into a regular expression. Only `*` (does not cross `/`), `**` with a directory
 * separator (zero or more directories) and `{a,b}` (alternation) are accepted.
 *
 * MUST: anything else (`?` / `[…]` / `**` without a separator) is not accepted. Failing with "a
 * spelling this check cannot read was used" is cheaper than staying green while disagreeing with
 * Vitest's expansion.
 */
const globToRegExp = (pattern: string): RegExp => {
  if (/\?|\[|\]/.test(pattern) || /\*\*(?!\/)/.test(pattern)) {
    throw new Error(`lane parsing: unsupported glob syntax: ${pattern}`);
  }
  let source = "";
  let rest = pattern;
  while (rest !== "") {
    if (rest.startsWith("**/")) {
      source += "(?:.*/)?";
      rest = rest.slice(3);
    } else if (rest.startsWith("*")) {
      source += "[^/]*";
      rest = rest.slice(1);
    } else if (rest.startsWith("{")) {
      const close = rest.indexOf("}");
      if (close < 0) throw new Error(`lane parsing: unclosed {: ${pattern}`);
      source += `(?:${rest
        .slice(1, close)
        .split(",")
        .map((part) => part.replace(/[.+^$()|\\]/g, "\\$&"))
        .join("|")})`;
      rest = rest.slice(close + 1);
    } else {
      source += rest[0]!.replace(/[.+^$()|\\]/g, "\\$&");
      rest = rest.slice(1);
    }
  }
  return new RegExp(`^${source}$`);
};

/**
 * Lists the files under the repository-relative `dir` recursively (does not descend into
 * `node_modules` or VCS directories).
 */
const walkFiles = (dir: string): string[] => {
  const found: string[] = [];
  const visit = (relative: string): void => {
    for (const entry of readDirSync(new URL(relative, REPO_ROOT))) {
      const child = `${relative}${entry.name}`;
      if (entry.isDirectory) {
        if (entry.name === "node_modules" || entry.name === ".git") continue;
        visit(`${child}/`);
      } else if (entry.isFile) {
        found.push(child);
      }
    }
  };
  visit(dir === "." ? "" : `${dir}/`);
  return found;
};

/** Every file in the test directories (under `packages/<pkg>/tests` and `tools/<tool>/tests`). */
const TEST_DIRS = ["packages", "tools"]
  .flatMap((parent) =>
    readDirSync(new URL(`${parent}/`, REPO_ROOT))
      .filter((entry) => entry.isDirectory)
      .map((entry) => `${parent}/${entry.name}/tests`),
  )
  .filter((dir) => {
    try {
      readDirSync(new URL(`${dir}/`, REPO_ROOT));
      return true;
    } catch {
      return false;
    }
  });
const ALL_FILES = TEST_DIRS.flatMap(walkFiles);

/** The lanes' population = `*_test.ts` only (the same spelling as Vitest's `include`). */
const ALL_TEST_FILES = ALL_FILES.filter((file) => globToRegExp(ALL_TESTS).test(file));

/**
 * Spellings that Vitest / Deno may treat as tests (`test.<ext>` / `*.test.<ext>` / `*_test.<ext>`).
 */
const TEST_EXTENSIONS = ["ts", "tsx", "mts", "js", "mjs", "jsx"] as const;
const looksLikeTestFile = (name: string): boolean =>
  TEST_EXTENSIONS.some(
    (extension) =>
      name === `test.${extension}` ||
      name.endsWith(`.test.${extension}`) ||
      name.endsWith(`_test.${extension}`),
  );
const basename = (file: string): string => file.slice(file.lastIndexOf("/") + 1);

/** The set of files a lane actually collects. */
const filesOf = (lane: Lane): Set<string> => {
  const includes = lane.include.map(globToRegExp);
  const excludes = lane.exclude.map(globToRegExp);
  return new Set(
    ALL_TEST_FILES.filter(
      (file) =>
        includes.some((matches) => matches.test(file)) &&
        !excludes.some((matches) => matches.test(file)),
    ),
  );
};

const [coreLane, ...familyLanes] = LANES;
if (coreLane === undefined || coreLane.name !== "core") {
  throw new Error("lane parsing: the first entry of LANES is not core");
}
const core = filesOf(coreLane);
const lanes = familyLanes.map((lane) => ({ name: lane.name, files: filesOf(lane) }));

const sorted = (files: Iterable<string>): string[] => [...files].sort();

test("lane check: core ∪ all lanes covers every test of the repository", () => {
  const covered = new Set(core);
  for (const lane of lanes) for (const file of lane.files) covered.add(file);
  const uncovered = ALL_TEST_FILES.filter((file) => !covered.has(file));
  assert(
    uncovered.length === 0,
    "some tests are in no lane (they never run in a lane run): " +
      `${uncovered.sort().join(", ")}. ` +
      "Fix core's exclude in lanes.ts or the matching models:<family>.",
  );
});

test("lane check: core and the family lanes are disjoint", () => {
  for (const lane of lanes) {
    const overlap = sorted(lane.files).filter((file) => core.has(file));
    assertEquals(overlap, [], `${lane.name} overlaps core (the heavy e2e is paid twice)`);
  }
});

test("lane check: each lane collects at least one file", () => {
  assert(lanes.length > 0, "lanes.ts has no models:<family> at all");
  const empty = lanes.filter((lane) => lane.files.size === 0).map((lane) => lane.name);
  assertEquals(empty, [], "some lanes collect nothing (Vitest fails with No test files found)");
  assert(core.size > 0, "core collects nothing");
});

test("lane check: tests are spelled *_test.ts only", () => {
  const strays = ALL_FILES.filter(
    (file) => looksLikeTestFile(basename(file)) && !file.endsWith("_test.ts"),
  ).sort();
  assertEquals(
    strays,
    [],
    "some files look like tests but are spelled outside Vitest's include (*_test.ts)" +
      " (they look green without being collected). Rename them to <name>_test.ts.",
  );
});

const E2E_DIR = "packages/e2e/tests/";

/**
 * The end-to-end tests that belong in core: gates over pure functions and manifests that need
 * neither a GPU nor real weights, and three GPU checks on small synthetic graphs that need no model
 * asset (the external fusion rules — the candidate and the models package's adopted rules — against
 * their unfused plans, which live in the e2e package because the runtime's tests import nothing
 * outside the runtime; they SKIP explicitly
 * without an adapter). Every other file of `packages/e2e/tests` runs real weights and
 * belongs to a family lane, so a dropped family row or a new e2e file whose name misses its
 * family's token turns the gate red instead of moving real-weight runs into core.
 */
const CORE_E2E_TESTS = [
  "packages/e2e/tests/e2e_ab_gate_test.ts",
  "packages/e2e/tests/e2e_assets_gate_test.ts",
  "packages/e2e/tests/e2e_ew_chain_rule_test.ts",
  "packages/e2e/tests/e2e_external_fusion_rule_test.ts",
  "packages/e2e/tests/e2e_fusion_candidates_report_test.ts",
  "packages/e2e/tests/e2e_session_options_mirror_test.ts",
  "packages/e2e/tests/e2e_vae_rms_silu_rule_test.ts",
  "packages/e2e/tests/ew_chain_rule_plan_test.ts",
  "packages/e2e/tests/fusion_candidates_test.ts",
  "packages/e2e/tests/fusion_counts_context_test.ts",
  "packages/e2e/tests/vae_rms_silu_rule_plan_test.ts",
];

/**
 * Family lanes that claim no end-to-end test (their real-asset tests are host-side ones in
 * `packages/models/tests`): dropping such a row moves no e2e file into core, so the lane is named
 * here and must exist in `LANES`.
 */
const HOST_ONLY_FAMILY_LANES = ["models:umt5"];

test("lane gate: core holds no end-to-end test but the gates that need no model asset", () => {
  assertEquals(
    sorted(core).filter((file) => file.startsWith(E2E_DIR)),
    CORE_E2E_TESTS,
    "core collects an end-to-end test that is not one of the gates without model assets (a family row was dropped from " +
      "FAMILY_LANES, or the file name misses its family's token): restore the lane or rename the file",
  );
});

test("lane gate: every family lane claims an end-to-end test or is a named host-only lane", () => {
  const names = new Set(lanes.map((lane) => lane.name));
  for (const name of HOST_ONLY_FAMILY_LANES) {
    assert(names.has(name), `${name} is missing from LANES (its tests fall back into core)`);
  }
  const withoutE2e = lanes
    .filter((lane) => ![...lane.files].some((file) => file.startsWith(E2E_DIR)))
    .map((lane) => lane.name);
  assertEquals(
    sorted(withoutE2e),
    sorted(HOST_ONLY_FAMILY_LANES),
    "a family lane claims no end-to-end test: name it in HOST_ONLY_FAMILY_LANES so that dropping " +
      "its row is caught",
  );
});

const EXPOSE_GC_FLAG = "--expose-gc";

test("lane check: the Vitest config builds projects from LANES and passes --expose-gc to the workers", () => {
  // A run without the flag has no `gc`, and the acquisition relaxation silently stops working (it
  // does not turn red).
  const config: unknown = viteConfig;
  assert(
    typeof config === "object" && config !== null && "test" in config,
    "vite.config.ts has no test config",
  );
  const testConfig: unknown = config.test;
  assert(
    typeof testConfig === "object" && testConfig !== null,
    "the test of vite.config.ts is not an object",
  );
  const execArgv: unknown = "execArgv" in testConfig ? testConfig.execArgv : undefined;
  assert(
    Array.isArray(execArgv) && execArgv.includes(EXPOSE_GC_FLAG),
    `${EXPOSE_GC_FLAG} is not passed to the workers`,
  );
  const projects: unknown = "projects" in testConfig ? testConfig.projects : undefined;
  assertEquals(projects, laneProjects(), "the projects of vite.config.ts disagree with LANES");
  assertEquals(
    laneProjects().map((project) => project.test.name),
    defaultLanes().map((lane) => lane.name),
  );
});

test("lane check: the families not run by default are in the lane table and enter projects only with KARUME_ALL_FAMILIES=1", () => {
  const lowPriority = LOW_PRIORITY_FAMILIES.map((family) => `models:${family}`);
  for (const name of lowPriority) {
    assert(
      LANES.some((lane) => lane.name === name),
      `${name} is not in LANES (core would pick it up)`,
    );
  }
  const without = laneProjects({}).map((project) => project.test.name);
  const withAll = laneProjects({ KARUME_ALL_FAMILIES: "1" }).map((project) => project.test.name);
  assertEquals(
    without.filter((name) => lowPriority.includes(name)),
    [],
  );
  assertEquals(
    withAll,
    LANES.map((lane) => lane.name),
  );
  assertEquals(without.length + lowPriority.length, withAll.length);
});

test("lane parsing: globs expand only *, **/ and {a,b}", () => {
  const matches = globToRegExp("packages/*/tests/**/*_test.ts");
  assert(matches.test("packages/hub/tests/fetch_test.ts"));
  assert(matches.test("packages/runtime/tests/helpers/nested_test.ts"));
  assert(!matches.test("packages/hub/src/fetch_test.ts"));
  assert(!matches.test("packages/hub/tests/fetch.test.ts"));
  assert(
    globToRegExp("packages/{models,e2e}/tests/*anima*_test.ts").test(
      "packages/e2e/tests/e2e_anima_test.ts",
    ),
  );
  assert(
    !globToRegExp("packages/models/tests/*anima*_test.ts").test(
      "packages/e2e/tests/e2e_anima_test.ts",
    ),
  );
  for (const bad of [
    "packages/**",
    "packages/?/tests/*_test.ts",
    "packages/[ab]/tests/*_test.ts",
  ]) {
    let threw = false;
    try {
      globToRegExp(bad);
    } catch {
      threw = true;
    }
    assert(threw, `an unsupported glob was accepted: ${bad}`);
  }
});
