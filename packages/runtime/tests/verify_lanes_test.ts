// テストレーンの被覆の門（ADR 0005 追記 2026-09-20）。
//
// レーン（`packages/test-support/src/lanes.ts` の `LANES` — Vitest の projects として
// `vp test --project <lane>` で回す）は、フル verify（`vp test`）を分割した実行単位である。
// 分割の唯一の危険は**被覆漏れ** —— 新しいテストを足したのにどのレーンにも入らず、レーン実行では
// 一度も走らないまま「緑」に見える形。ここはそれを無音で通さない門で、次の 4 つと、Vitest の設定が
// `--expose-gc` を worker に渡していること（テストの GPU の取得口の緩和が要る — 下の
// EXPOSE_GC_FLAG）を見る:
//
// - **被覆**: core ∪ 全レーン = リポの全 `*_test.ts`（`packages/*/tests` と `tools/*/tests` 配下）。
// - **重複なし**: core と系列レーンは互いに素（同じ重い e2e を 2 度払わない）。系列レーン
//   どうしの重複は許す —— 通常配布形と QAT 配布形の両方を要るファイルが実在する。
// - **空でない**: 各レーンは 1 本以上。綴りを間違えたレーンは Vitest が "No test files found" で
//   落ちるが、それはそのレーンを回したときにしか分からない。
// - **綴り**: テストの名前は `*_test.ts` だけ。Vitest の `include` は `*_test.ts` なので、
//   `*.test.ts` / `test.ts` の綴りは**収集されないまま**緑に見える。母集団を広げるのではなく
//   拒否側で止める。
//
// Two more checks guard the lane table itself. Coverage and disjointness still hold when a family
// row is dropped from `FAMILY_LANES` (its files fall back into core), so core may hold no
// end-to-end test but the GPU-free gates of `CORE_E2E_TESTS`, and a family lane that claims no
// end-to-end test is named in `HOST_ONLY_FAMILY_LANES` and must exist.
//
// 真実源は `LANES` そのもの（`vite.config.ts` の projects は `laneProjects()` で同じ表から作る）。
// glob の展開はこの門が自前で行う（`*` は `/` を跨がず、区切りを伴う `**` は 0 個以上のディレクトリ）。

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

/** リポジトリ根（レーンの glob はリポ相対）。 */
const REPO_ROOT = new URL("../../../", import.meta.url);

/**
 * glob を正規表現へ。受けるのは `*`（`/` を跨がない）・ディレクトリ区切りを伴う `**`
 * （0 個以上のディレクトリ）・`{a,b}`（選択）だけ。
 *
 * MUST: それ以外（`?` / `[…]` / 区切りを伴わない `**`）は受けない。Vitest 側の展開と食い違ったまま緑になる
 * より、「この門が読めない綴りを使った」と落ちる方が安い。
 */
const globToRegExp = (pattern: string): RegExp => {
  if (/\?|\[|\]/.test(pattern) || /\*\*(?!\/)/.test(pattern)) {
    throw new Error(`レーン解析: 未対応の glob 構文である: ${pattern}`);
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
      if (close < 0) throw new Error(`レーン解析: 閉じていない { である: ${pattern}`);
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

/** リポ相対 `dir` 以下のファイルを再帰列挙する（`node_modules` と VCS は降りない）。 */
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

/** テストの置き場の全ファイル（`packages/<pkg>/tests` と `tools/<tool>/tests` 配下）。 */
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

/** レーンの母集団 = `*_test.ts` だけ（Vitest の `include` と同じ綴り）。 */
const ALL_TEST_FILES = ALL_FILES.filter((file) => globToRegExp(ALL_TESTS).test(file));

/** Vitest / Deno がテストとして扱いうる綴り（`test.<ext>` / `*.test.<ext>` / `*_test.<ext>`）。 */
const TEST_EXTENSIONS = ["ts", "tsx", "mts", "js", "mjs", "jsx"] as const;
const looksLikeTestFile = (name: string): boolean =>
  TEST_EXTENSIONS.some(
    (extension) =>
      name === `test.${extension}` ||
      name.endsWith(`.test.${extension}`) ||
      name.endsWith(`_test.${extension}`),
  );
const basename = (file: string): string => file.slice(file.lastIndexOf("/") + 1);

/** レーンが実際に収集するファイル集合。 */
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
  throw new Error("レーン解析: LANES の先頭が core でない");
}
const core = filesOf(coreLane);
const lanes = familyLanes.map((lane) => ({ name: lane.name, files: filesOf(lane) }));

const sorted = (files: Iterable<string>): string[] => [...files].sort();

test("レーンの門: core ∪ 全レーンがリポの全テストを覆う", () => {
  const covered = new Set(core);
  for (const lane of lanes) for (const file of lane.files) covered.add(file);
  const uncovered = ALL_TEST_FILES.filter((file) => !covered.has(file));
  assert(
    uncovered.length === 0,
    "どのレーンにも入らないテストがある（レーン実行では一度も走らない）: " +
      `${uncovered.sort().join(", ")}。` +
      "lanes.ts の core の exclude か、該当する models:<系列> を直すこと。",
  );
});

test("レーンの門: core と系列レーンは互いに素", () => {
  for (const lane of lanes) {
    const overlap = sorted(lane.files).filter((file) => core.has(file));
    assertEquals(overlap, [], `${lane.name} が core と重複している（重い e2e を 2 度払う）`);
  }
});

test("レーンの門: 各レーンは 1 本以上を収集する", () => {
  assert(lanes.length > 0, "lanes.ts に models:<系列> が 1 本も無い");
  const empty = lanes.filter((lane) => lane.files.size === 0).map((lane) => lane.name);
  assertEquals(empty, [], "何も収集しないレーンがある（Vitest が No test files found で落ちる）");
  assert(core.size > 0, "core が何も収集しない");
});

test("レーンの門: テストの綴りは *_test.ts だけ", () => {
  const strays = ALL_FILES.filter(
    (file) => looksLikeTestFile(basename(file)) && !file.endsWith("_test.ts"),
  ).sort();
  assertEquals(
    strays,
    [],
    "テストに見えるのに Vitest の include（*_test.ts）に入らない綴りのファイルがある" +
      "（収集されないまま緑に見える）。ファイル名を <name>_test.ts へ揃えること。",
  );
});

const E2E_DIR = "packages/e2e/tests/";

/**
 * The end-to-end tests that belong in core: gates over pure functions and manifests that need
 * neither a GPU nor real weights. Every other file of `packages/e2e/tests` runs real weights and
 * belongs to a family lane, so a dropped family row or a new e2e file whose name misses its
 * family's token turns the gate red instead of moving real-weight runs into core.
 */
const CORE_E2E_TESTS = [
  "packages/e2e/tests/e2e_ab_gate_test.ts",
  "packages/e2e/tests/e2e_assets_gate_test.ts",
  "packages/e2e/tests/e2e_session_options_mirror_test.ts",
];

/**
 * Family lanes that claim no end-to-end test (their real-asset tests are host-side ones in
 * `packages/models/tests`): dropping such a row moves no e2e file into core, so the lane is named
 * here and must exist in `LANES`.
 */
const HOST_ONLY_FAMILY_LANES = ["models:umt5"];

test("lane gate: core holds no end-to-end test but the GPU-free gates", () => {
  assertEquals(
    sorted(core).filter((file) => file.startsWith(E2E_DIR)),
    CORE_E2E_TESTS,
    "core collects an end-to-end test that is not a GPU-free gate (a family row was dropped from " +
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

test("レーンの門: Vitest の設定は projects を LANES から作り、worker に --expose-gc を渡す", () => {
  // フラグの無い走行では `gc` が生えず、取得口の緩和が黙って効かなくなる（赤にはならない）。
  const config: unknown = viteConfig;
  assert(
    typeof config === "object" && config !== null && "test" in config,
    "vite.config.ts が test 設定を持たない",
  );
  const testConfig: unknown = config.test;
  assert(
    typeof testConfig === "object" && testConfig !== null,
    "vite.config.ts の test が object でない",
  );
  const execArgv: unknown = "execArgv" in testConfig ? testConfig.execArgv : undefined;
  assert(
    Array.isArray(execArgv) && execArgv.includes(EXPOSE_GC_FLAG),
    `${EXPOSE_GC_FLAG} を worker に渡していない`,
  );
  const projects: unknown = "projects" in testConfig ? testConfig.projects : undefined;
  assertEquals(projects, laneProjects(), "vite.config.ts の projects が LANES と食い違う");
  assertEquals(
    laneProjects().map((project) => project.test.name),
    defaultLanes().map((lane) => lane.name),
  );
});

test("レーンの門: 既定で走らせないファミリーはレーン表にあり、KARUME_ALL_FAMILIES=1 のときだけ projects に入る", () => {
  const lowPriority = LOW_PRIORITY_FAMILIES.map((family) => `models:${family}`);
  for (const name of lowPriority) {
    assert(
      LANES.some((lane) => lane.name === name),
      `${name} が LANES に無い（core が拾ってしまう）`,
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

test("レーン解析: glob は * と **/ と {a,b} だけを展開する", () => {
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
    assert(threw, `未対応の glob を受けてしまった: ${bad}`);
  }
});
