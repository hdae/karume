// テストレーンの被覆の門（ADR 0005 追記 2026-09-20）。
//
// `deno.json` の `test:core` と `test:models:*` は、フル verify（`deno test -A`）を分割した
// 実行単位である。分割の唯一の危険は**被覆漏れ** —— 新しいテストを足したのにどのレーンにも
// 入らず、レーン実行では一度も走らないまま「緑」に見える形。ここはそれを無音で通さない門で、
// 3 つだけを見る:
//
// - **被覆**: core ∪ 全レーン = リポの全 `*_test.ts`（`deno.json` の `exclude` 根を除く）。
// - **重複なし**: core と系列レーンは互いに素（同じ重い e2e を 2 度払わない）。系列レーン
//   どうしの重複は許す —— 通常配布形と QAT 配布形の両方を要るファイルが実在する。
// - **空でない**: 各レーンは 1 本以上。綴りを間違えたレーンは `deno test` が
//   "No test modules found" で落ちるが、それはそのレーンを回したときにしか分からない。
//
// 真実源は `deno.json` の task 文字列そのもの（レーンの定義を二重に持たない）。task 文字列の
// 解析は `deno test -A <対象…> [--ignore=<glob,…>]` という形に依存するので、その形から外れた
// 綴りは推測せず throw する（MUST: 読めない形を「対象ゼロ」と読み替えると、この門は
// レーンが壊れたことを緑で隠す）。

import { assert, assertEquals } from "@std/assert";

/** リポジトリ根（`deno.json` の置き場 = task 内の相対パスの基準）。 */
const REPO_ROOT = new URL("../../../", import.meta.url);

type DenoConfig = {
  readonly exclude?: readonly string[];
  readonly tasks?: Readonly<Record<string, string>>;
};

/** レーン 1 本の定義（task 文字列から取り出した対象と除外）。 */
type Lane = {
  readonly task: string;
  readonly targets: readonly string[];
  readonly ignores: readonly string[];
};

const config: DenoConfig = JSON.parse(
  await Deno.readTextFile(new URL("deno.json", REPO_ROOT)),
);

const tasks = config.tasks ?? {};

/** `exclude` の根（末尾 `/` を落とした先頭セグメント）+ VCS ディレクトリ。 */
const excludedRoots = new Set<string>([
  ".git",
  ...(config.exclude ?? []).map((entry) => entry.replace(/\/+$/, "")),
]);

/**
 * glob を正規表現へ。受けるのは `*`（`/` を跨がない）だけ。
 *
 * MUST: `**` / `?` / `[…]` / `{…}` は受けない。deno 側の展開と食い違ったまま緑になるより、
 * 「この門が読めない綴りを使った」と落ちる方が安い。
 */
const globToRegExp = (pattern: string): RegExp => {
  if (/\*\*|\?|\[|\]|\{|\}/.test(pattern)) {
    throw new Error(`レーン解析: 未対応の glob 構文である: ${pattern}`);
  }
  const source = pattern
    .split("*")
    .map((literal) => literal.replace(/[.+^$()|\\]/g, "\\$&"))
    .join("[^/]*");
  return new RegExp(`^${source}$`);
};

/** リポ相対 `dir` 以下の `*_test.ts` を再帰収集する（`exclude` 根と VCS は降りない）。 */
const walkTests = (dir: string): string[] => {
  const found: string[] = [];
  const visit = (relative: string): void => {
    for (const entry of Deno.readDirSync(new URL(relative, REPO_ROOT))) {
      const child = `${relative}${entry.name}`;
      if (entry.isDirectory) {
        if (excludedRoots.has(child)) continue;
        visit(`${child}/`);
      } else if (entry.isFile && entry.name.endsWith("_test.ts")) {
        found.push(child);
      }
    }
  };
  visit(dir === "." ? "" : `${dir}/`);
  return found;
};

/** task の対象 1 つ（ディレクトリ・glob・単一ファイル）をリポ相対パスの一覧へ展開する。 */
const expandTarget = (target: string): string[] => {
  if (target.includes("*")) {
    const cut = target.lastIndexOf("/");
    if (cut < 0) throw new Error(`レーン解析: ディレクトリを伴わない glob である: ${target}`);
    const dir = target.slice(0, cut);
    if (dir.includes("*")) {
      throw new Error(`レーン解析: ディレクトリ側に glob を含む: ${target}`);
    }
    const matches = globToRegExp(target.slice(cut + 1));
    return [...Deno.readDirSync(new URL(`${dir}/`, REPO_ROOT))]
      .filter((entry) => entry.isFile && matches.test(entry.name))
      .map((entry) => `${dir}/${entry.name}`);
  }
  const stat = Deno.statSync(new URL(target, REPO_ROOT));
  if (stat.isDirectory) return walkTests(target);
  if (!target.endsWith("_test.ts")) {
    throw new Error(`レーン解析: テストファイルでない対象である: ${target}`);
  }
  return [target];
};

/** task 文字列を空白で割る（`'…'` で括った glob は 1 語として保つ）。 */
const tokenize = (command: string): string[] => {
  const tokens: string[] = [];
  let current = "";
  let open = false;
  let started = false;
  for (const char of command) {
    if (char === '"' || char === "\\") {
      throw new Error(`レーン解析: 未対応の引用符である: ${command}`);
    }
    if (char === "'") {
      open = !open;
      started = true;
      continue;
    }
    if (!open && /\s/.test(char)) {
      if (started) tokens.push(current);
      current = "";
      started = false;
      continue;
    }
    current += char;
    started = true;
  }
  if (open) throw new Error(`レーン解析: 閉じていない引用符である: ${command}`);
  if (started) tokens.push(current);
  return tokens;
};

/** `deno test -A <対象…> [--ignore=<glob,…>]` を対象と除外に分解する。 */
const parseLane = (task: string, command: string): Lane => {
  const tokens = tokenize(command);
  if (tokens[0] !== "deno" || tokens[1] !== "test") {
    throw new Error(`レーン解析: ${task} が \`deno test\` で始まっていない: ${command}`);
  }
  const targets: string[] = [];
  const ignores: string[] = [];
  for (const token of tokens.slice(2)) {
    if (token === "-A") continue;
    if (token.startsWith("--ignore=")) {
      ignores.push(...token.slice("--ignore=".length).split(",").filter((glob) => glob !== ""));
      continue;
    }
    if (token.startsWith("-")) {
      throw new Error(`レーン解析: ${task} が未対応のフラグを持つ: ${token}`);
    }
    targets.push(token);
  }
  if (targets.length === 0) throw new Error(`レーン解析: ${task} に対象が無い: ${command}`);
  return { task, targets, ignores };
};

/** レーンが実際に収集するファイル集合。 */
const filesOf = (lane: Lane): Set<string> => {
  const ignored = lane.ignores.map(globToRegExp);
  const files = new Set<string>();
  for (const target of lane.targets) {
    for (const file of expandTarget(target)) {
      if (!ignored.some((pattern) => pattern.test(file))) files.add(file);
    }
  }
  return files;
};

const laneTaskNames = Object.keys(tasks).filter((name) => name.startsWith("test:models:")).sort();
const coreCommand = tasks["test:core"];
if (coreCommand === undefined) throw new Error("レーン解析: deno.json に test:core が無い");

const core = filesOf(parseLane("test:core", coreCommand));
const lanes = laneTaskNames.map((name) => {
  const command = tasks[name];
  return { name, files: filesOf(parseLane(name, command)) };
});

const sorted = (files: Iterable<string>): string[] => [...files].sort();

Deno.test("レーンの門: core ∪ 全レーンがリポの全テストを覆う", () => {
  const covered = new Set(core);
  for (const lane of lanes) for (const file of lane.files) covered.add(file);
  const all = walkTests(".");
  const uncovered = all.filter((file) => !covered.has(file));
  assert(
    uncovered.length === 0,
    "どのレーンにも入らないテストがある（レーン実行では一度も走らない）: " +
      `${uncovered.sort().join(", ")}。` +
      "deno.json の test:core の --ignore か、該当する test:models:<系列> を直すこと。",
  );
  // 逆向き —— レーンが実在しないファイルを指していないか（改名の取りこぼし）。
  const known = new Set(all);
  const missing = sorted(covered).filter((file) => !known.has(file));
  assertEquals(missing, [], "レーンが実在しないテストを指している");
});

Deno.test("レーンの門: core と系列レーンは互いに素", () => {
  for (const lane of lanes) {
    const overlap = sorted(lane.files).filter((file) => core.has(file));
    assertEquals(
      overlap,
      [],
      `${lane.name} が test:core と重複している（重い e2e を 2 度払う）`,
    );
  }
});

Deno.test("レーンの門: 各レーンは 1 本以上を収集する", () => {
  assert(lanes.length > 0, "deno.json に test:models:<系列> が 1 本も無い");
  const empty = lanes.filter((lane) => lane.files.size === 0).map((lane) => lane.name);
  assertEquals(
    empty,
    [],
    "何も収集しないレーンがある（deno test が No test modules found で落ちる）",
  );
});
