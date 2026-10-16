// パッケージの依存グラフの門。GPU も実資産も要らない（ソースを oxc-parser で読んで辺を辿るだけ）。
//
// 守る不変条件は CLAUDE.md の「ランタイム依存は Web 標準 API のみ（依存パッケージも Web 標準 API
// のみで構成されたものに限る）」。型検査ではこれを守れない —— ワークスペースでは、メンバーの
// `package.json` の `dependencies` が宣言していないパッケージでも、devDependencies やワークスペースの
// 根に入っていれば手元の解決は通る（`packages/*` のソースが裸の指定子 `import "jpeg-js"` を書いても
// 型検査は緑）。公開したパッケージに付いていくのはそのパッケージの `dependencies` だけなので、
// 壊れるのは利用者の手元になってから。
//
// メンバーは `pnpm-workspace.yaml` の `packages` と各 `packages/*/package.json` から導く（`private`
// のメンバー = 公開しないテスト用の足場は門の対象にしない。辺の行き先としてはメンバーに数える）。
// 見るのは各メンバーの公開の入口（その `package.json` の `exports` の全ての値）から辿ったモジュール
// グラフ（`@karume/test-support/module-graph` が oxc-parser でソースから採る静的な辺）の全ての辺。
// 許すのは次の 3 つだけ:
//
// - **同じメンバーの根の内側のファイル**（相対 import、または自分自身の名前での自己参照）。
// - **他のメンバーの公開の入口のファイル**を、`@karume/*` の裸の指定子で、かつ import を書いた
//   メンバーの `dependencies` が宣言したうえで引くもの。根をまたぐ相対 import（`../../hub/src/x.ts`）
//   は入口を指していても落とし、どのメンバーにも属さないファイル（リポジトリ根・`examples/`・`tools/`
//   など）への import も落とす —— どれも公開したパッケージの外を指すので、利用者の手元では解決
//   できない。
// - **辺の出どころのメンバー自身の `package.json` の `dependencies` が宣言した外部のパッケージ**
//   （名前の一致。版は見ない —— `catalog:` の宣言は版をワークスペースの catalog に預けるだけで、
//   宣言したことに変わりはない。devDependencies は数えない）。どのメンバーの宣言で許すかは、import を
//   書いたファイルの置き場で決める —— models が hub 経由で `@hdae/fetch-cache` を引くのは hub の
//   宣言で許すが、models のファイルが直に書けば models の宣言が要る。
//
// `node:` の組み込みモジュールは Web 標準 API でないので落とす。例外は下の
// `NODE_BUILTINS_ALLOWED_FROM` に書いたファイルだけ（hub の `./node` サブパス = Node の利用者が
// 明示的に選ぶ入口）。`npm:` / `jsr:` などスキームつきの指定子はソースに 1 つでも出たら落とす
// （公開したパッケージを Node やバンドラで読むと解決できない綴り）。許可の一覧はテストに書き
// 写さず、各 `package.json` から導く。
//
// NOTE: Deno 版の門にあった「外部のパッケージが別の外部のパッケージを引く辺（推移的な依存）を
// 落とす」規則はここに無い。外部のパッケージの中身を辿るには node_modules を歩く必要があるが、この門は
// ワークスペースのソースだけを読むので、その規則は検査できない。
//
// 門が見るのは静的な辺（リテラルの import / import type / `export … from` / リテラルの動的
// import / 型の位置の `import("…")` / `import x = require("…")` / `/// <reference path|types>`）に
// 限る。`new Worker(new URL(...))` の先・リテラルでない動的 import・`import.meta.resolve` の先は
// グラフに載らないので、この門の外である。
//
// MUST: 読めない形は推測せず throw する —— 「読めなかった辺」を「外部の辺が無い」と読み替えると、
// この門は違反を緑で隠す（構文を読めないソース・読めないファイル・知らない形の宣言や exports・
// 入口に無い `@karume/*` のサブパスは、違反としてではなく throw で赤にする）。

import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { readDir, readTextFile } from "@karume/test-support/fs";
import { type ModuleEdge, walkModuleGraph } from "@karume/test-support/module-graph";
import { describe, it } from "@karume/test-support/test";

/** リポジトリ根（`pnpm-workspace.yaml` の置き場）。 */
const REPO_ROOT = new URL("../../../", import.meta.url);

/**
 * `node:` の組み込みモジュールを引いてよいファイル（明示の許可の一覧）。
 *
 * hub の `./node` サブパスは Node のファイルシステムで配布形を読む入口で、Node の利用者が明示的に
 * import する（`mod.ts` からは辿れない）。不変条件の、文書化した唯一の切り出しである —— 理由は
 * `packages/hub/node.ts` の冒頭。
 */
const NODE_BUILTINS_ALLOWED_FROM: ReadonlySet<string> = new Set([
  new URL("packages/hub/node.ts", REPO_ROOT).href,
]);

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const record = (value: unknown, what: string): Record<string, unknown> => {
  if (!isRecord(value)) throw new Error(`依存グラフの門: ${what} がオブジェクトでない`);
  return value;
};

const text = (value: unknown, what: string): string => {
  if (typeof value !== "string") throw new Error(`依存グラフの門: ${what} が文字列でない`);
  return value;
};

/** ワークスペースのメンバー 1 つ。 */
type Member = {
  readonly name: string;
  /** メンバーの根（末尾 `/` 付きの `file:` URL）。 */
  readonly root: string;
  /** `private: true`（公開しない）のメンバー。門の対象にしない。 */
  readonly private: boolean;
  /** 公開の入口（`exports` のサブパス → 絶対 URL）。 */
  readonly exports: ReadonlyMap<string, string>;
  /** 公開の入口（`exports` の全ての値を絶対 URL にしたもの）。 */
  readonly entries: readonly string[];
  /** `dependencies` が宣言したパッケージの名前（ワークスペースのメンバーも外部も）。 */
  readonly declared: ReadonlySet<string>;
};

/** 指定子のパッケージ名（`@scope/name/sub` → `@scope/name`、`name/sub` → `name`）とサブパス。 */
const packageOf = (specifier: string): { readonly name: string; readonly subpath: string } => {
  const parts = specifier.split("/");
  const length = specifier.startsWith("@") ? 2 : 1;
  const head = parts.slice(0, length);
  if (head.length < length || head.some((part) => part === "")) {
    throw new Error(`依存グラフの門: パッケージの指定子を読めない: ${specifier}`);
  }
  const rest = parts.slice(length);
  return { name: head.join("/"), subpath: rest.length === 0 ? "." : `./${rest.join("/")}` };
};

/** 指定子のスキーム（`node:` / `npm:` / `jsr:` / `https:` …）。スキームが無ければ `undefined`。 */
const schemeOf = (specifier: string): string | undefined =>
  /^([a-z][a-z0-9+.-]*):/i.exec(specifier)?.[1]?.toLowerCase();

const isRelative = (specifier: string): boolean =>
  specifier.startsWith("./") || specifier.startsWith("../") || specifier.startsWith("/");

/**
 * `package.json` の `dependencies` から宣言の一覧（名前）を導く。`workspace:` はメンバー、
 * `catalog:` は版をワークスペースの catalog に預けた外部の宣言で、どちらも名前で数える。
 *
 * MUST: ローカルのパスの宣言（`file:` / `link:` / `portal:` / 相対・絶対パス）は推測せず throw
 * する —— 公開したパッケージでは利用者の手元を指すので、この門が許してよい形でない。
 */
const declaredPackagesOf = (dependencies: Readonly<Record<string, string>>): Set<string> => {
  const declared = new Set<string>();
  for (const [name, value] of Object.entries(dependencies)) {
    if (/^(file|link|portal):/.test(value) || /^(\.{1,2})?\//.test(value)) {
      throw new Error(
        `依存グラフの門: ローカルのパスの宣言である（dependencies["${name}"] = ${value}）`,
      );
    }
    declared.add(name);
  }
  return declared;
};

/**
 * `pnpm-workspace.yaml` の `packages` の並び（ブロック形式の `- glob`）を読む。
 *
 * MUST: YAML の全体は読まない —— 読めない形（流れ形式・`!` の除外・末尾の `/*` 1 段以外の glob）は
 * 推測せず throw する。
 */
const workspaceGlobsOf = (yaml: string): string[] => {
  const lines = yaml.split("\n");
  const start = lines.findIndex((line) => /^packages:\s*$/.test(line));
  if (start < 0) throw new Error("依存グラフの門: pnpm-workspace.yaml に packages の並びが無い");
  const globs: string[] = [];
  for (const line of lines.slice(start + 1)) {
    if (/^\s*(#.*)?$/.test(line)) continue;
    if (!/^\s/.test(line)) break;
    const glob = /^\s+-\s+(["']?)([^"'#\s]+)\1\s*(#.*)?$/.exec(line)?.[2];
    if (glob === undefined || !/^[^!*]+(\/\*)?$/.test(glob)) {
      throw new Error(`依存グラフの門: pnpm-workspace.yaml の packages の項目を読めない: ${line}`);
    }
    globs.push(glob);
  }
  if (globs.length === 0) {
    throw new Error("依存グラフの門: pnpm-workspace.yaml の packages が空である");
  }
  return globs;
};

/** glob 1 つをメンバーの根（末尾 `/` 付き）へ展開する（`package.json` の無いディレクトリは数えない — pnpm と同じ）。 */
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
  const name = text(config.name, `${memberRoot.href}package.json の name`);
  const isPrivate = config.private === true;
  // 条件つきの exports（`{ "import": …, "types": … }`）は text が throw する —— 読めない形は推測しない。
  // exports の無い private のメンバー（テストだけの置き場）は入口 0 本 —— 門の対象でなく、他の
  // メンバーからそこへ入る辺は「公開の入口でない」で落ちる。
  const declaredExports: [string, unknown][] =
    typeof config.exports === "string"
      ? [[".", config.exports]]
      : config.exports === undefined && isPrivate
        ? []
        : Object.entries(record(config.exports, `${name} の exports`));
  if (declaredExports.length === 0 && !isPrivate) {
    throw new Error(`依存グラフの門: ${name} の exports が空である`);
  }
  const exports = new Map(
    declaredExports.map(([subpath, entry]) => [
      subpath,
      new URL(text(entry, `${name} の exports["${subpath}"]`), memberRoot).href,
    ]),
  );
  const dependencies = Object.fromEntries(
    Object.entries(record(config.dependencies ?? {}, `${name} の dependencies`)).map(
      ([key, value]) => [key, text(value, `${name} の dependencies["${key}"]`)],
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

/** グラフの辺 1 本（出どころのモジュール → 書かれた指定子 → 解決先）。 */
type Edge = ModuleEdge;

/** 判定の結果。`external` は許した外部の辺の数（形が変わって外部の辺を 1 本も読めない門を見分ける）。 */
type Verdict = {
  readonly violations: readonly string[];
  readonly external: number;
};

/** 判定に要るメンバーの面（判定の単体テストが根の無い模型を渡せるよう `root` は含めない）。 */
type Owner = Pick<Member, "name" | "entries" | "declared">;

/**
 * 辺を 1 本ずつ判じる。`ownerOf` はファイルが属するメンバーを返す（どのメンバーにも属さない
 * ファイルは `undefined`）。
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
    const sourceName = source?.name ?? "どのメンバーにも属さないファイル";
    const scheme = schemeOf(written);
    if (scheme === "node") {
      if (!NODE_BUILTINS_ALLOWED_FROM.has(from)) {
        violations.push(`node: の組み込みモジュールである（Web 標準 API でない）: ${label}`);
      }
      continue;
    }
    if (scheme !== undefined) {
      violations.push(`${scheme}: の指定子である（宣言の有無によらず禁止）: ${label}`);
      continue;
    }
    if (resolved.startsWith("file:")) {
      const target = ownerOf(resolved);
      if (target === undefined) {
        violations.push(`どのメンバーにも属さないファイルへの import である: ${label}`);
      } else if (target.name === source?.name) {
        // 同じメンバーの内側（相対 import か、自分自身の名前での自己参照）。
      } else if (!target.entries.includes(resolved)) {
        violations.push(
          `${target.name} の公開の入口でないファイルへの、メンバーをまたぐ import である: ${label}`,
        );
      } else if (isRelative(written)) {
        violations.push(
          `メンバーをまたぐ相対 import である（${target.name} の入口へは "${target.name}" の` +
            `指定子で入る）: ${label}`,
        );
      } else if (source === undefined || !source.declared.has(target.name)) {
        violations.push(
          `${sourceName} の package.json の dependencies が宣言していないメンバーへの依存である: ` +
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
      `${sourceName} の package.json の dependencies が宣言していない外部の依存である: ${label}`,
    );
  }
  return { violations, external };
};

/**
 * 相対でない指定子の解決先。メンバーの名前の指定子（`@karume/hub/node` など）はそのメンバーの
 * `exports` の入口のファイルへ、それ以外（外部・スキームつき）は書かれたままを返す（辿らない）。
 *
 * MUST: メンバーの `exports` に無いサブパスは推測せず throw する（Node の解決でも失敗する形）。
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
        `依存グラフの門: 解決できない import がある: ${from} → ${written}` +
          `（${name} の exports に ${subpath} が無い）`,
      );
    }
    return entry;
  };

/** メンバー 1 つの全ての入口のグラフを判じる。 */
const judgeMember = async (member: Member, members: readonly Member[]): Promise<Verdict> => {
  const ownerOf = (file: string): Member | undefined =>
    members.find((candidate) => file.startsWith(candidate.root));
  const edges = await walkModuleGraph(member.entries, resolverOf(members));
  return judgeEdges(edges, ownerOf);
};

const members = await readMembers();

describe("パッケージの依存グラフの門", () => {
  for (const member of members.filter((candidate) => !candidate.private)) {
    it(`${member.name} の公開の入口から辿った外部の依存は、出どころのパッケージが宣言したものだけである`, async () => {
      const { violations } = await judgeMember(member, members);
      assertEquals(
        violations,
        [],
        "ワークスペースでは宣言していない依存も手元で解決できるので型検査は通るが、" +
          "公開したパッケージでは解決できない（または Web 標準 API だけで閉じない）依存がある",
      );
    });
  }

  it("対照: hub が宣言した @hdae/fetch-cache の辺を許した外部の辺として数える", async () => {
    // 辺の採り方が変わって外部の辺を 1 本も読めなくなると、上の門は無音で緑になる。hub は
    // `@hdae/fetch-cache` を実際に使うので、その辺を読めていることをここで見る。
    const hub = members.find((member) => member.name === "@karume/hub");
    assert(hub !== undefined, "ワークスペースに @karume/hub が無い");
    const { external } = await judgeMember(hub, members);
    assert(external > 0, "hub のグラフに許した外部の辺が 1 本も無い（ソースの辺を読めていない）");
  });
});

describe("依存グラフの門の判定", () => {
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

  it("同じメンバーの内側のファイルと、他のメンバーの公開の入口は許す", () => {
    const verdict = judgeEdges(
      [
        { from: file, written: "./b.ts", resolved: "file:///repo/packages/example/src/b.ts" },
        { from: file, written: "@karume/other", resolved: "file:///repo/packages/other/mod.ts" },
      ],
      ownerOf,
    );
    assertEquals(verdict, { violations: [], external: 0 });
  });

  it("他のメンバーの入口でないファイルへの、根をまたぐ相対 import は落とす", () => {
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
    assert(violations[0]?.includes("@karume/other の公開の入口でない"), violations[0]);
  });

  it("どのメンバーにも属さないワークスペース内のファイルへの import は落とす", () => {
    const { violations } = judgeEdges(
      [{ from: file, written: "../../../shared.ts", resolved: "file:///repo/shared.ts" }],
      ownerOf,
    );
    assertEquals(violations.length, 1);
    assert(violations[0]?.includes("どのメンバーにも属さない"), violations[0]);
  });

  it("ローカルのパスの宣言（file: / link:）は許可の一覧に入れず throw する", () => {
    assertEquals(
      declaredPackagesOf({ "@karume/other": "workspace:*", "@x/y": "catalog:", z: "^1.2.0" }),
      new Set(["@karume/other", "@x/y", "z"]),
    );
    assertThrows(() => declaredPackagesOf({ util: "file:../util" }), Error, "ローカルのパス");
    assertThrows(() => declaredPackagesOf({ util: "link:../util" }), Error, "ローカルのパス");
  });

  it("宣言した外部のパッケージはサブパスでも許す", () => {
    const verdict = judgeEdges(
      [{ from: file, written: "@x/y/sub", resolved: "@x/y/sub" }],
      () => owner,
    );
    assertEquals(verdict, { violations: [], external: 1 });
  });

  it("宣言していない外部のパッケージと、宣言していないメンバーへの依存は落とす", () => {
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
    assert(violations[0]?.includes("宣言していない外部の依存"), violations[0]);
    assert(violations[1]?.includes("宣言していないメンバーへの依存"), violations[1]);
  });

  it("npm: / jsr: の指定子は宣言の有無によらず落とす", () => {
    const { violations } = judgeEdges(
      [
        { from: file, written: "npm:jpeg-js@^0.4.4", resolved: "npm:jpeg-js@^0.4.4" },
        { from: file, written: "jsr:@x/y@^1.2.0", resolved: "jsr:@x/y@^1.2.0" },
      ],
      () => owner,
    );
    assertEquals(violations.length, 2);
    assert(violations[0]?.includes("npm: の指定子"), violations[0]);
    assert(violations[1]?.includes("jsr: の指定子"), violations[1]);
  });

  it("node: は許可の一覧のファイル（hub の node.ts）からだけ許し、ほかのファイルからは落とす", () => {
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

  describe("ソースから採る辺（型だけの辺 / 型の位置の import / 動的 import）", () => {
    // 辺は oxc-parser でソースから採る。ファイルの読み出しは模型に差し替える。
    const impl = "file:///repo/packages/example/src/impl.ts";
    const graphOf = (sources: Readonly<Record<string, string>>) =>
      walkModuleGraph([impl], resolverOf([]), (url) => {
        const source = sources[url];
        return source === undefined
          ? Promise.reject(new Error(`not found: ${url}`))
          : Promise.resolve(source);
      });

    it("npm の型パッケージを指していれば落とす", async () => {
      const edges = await graphOf({ [impl]: `import type { X } from "npm:@types/x";\n` });
      const { violations } = judgeEdges(edges, ownerOf);
      assertEquals(violations.length, 1);
      assert(violations[0]?.includes("npm:@types/x"), violations[0]);
    });

    it("同じメンバーの内側のファイルを指していれば辺として読んだうえで許す", async () => {
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

    it("構文を読めなければ throw する", async () => {
      await assertRejects(
        () => graphOf({ [impl]: "import { from './x.ts'\n" }),
        Error,
        "cannot parse",
      );
    });

    it("どのメンバーにも属さない型ファイルを指していれば落とす", async () => {
      // 型の位置の `import("…")` がメンバーの外を指す形。
      const edges = await graphOf({
        [impl]: `export type U = import("../../../outside/types.ts").U;\n`,
        "file:///repo/outside/types.ts": "export type U = string;\n",
      });
      const { violations } = judgeEdges(edges, ownerOf);
      assertEquals(violations.length, 1);
      assert(violations[0]?.includes("どのメンバーにも属さない"), violations[0]);
    });

    it("辺の先のファイルを読めなければ推測せず throw する", async () => {
      await assertRejects(
        () => graphOf({ [impl]: `export { y } from "./missing.ts";\n` }),
        Error,
        "cannot read file:///repo/packages/example/src/missing.ts",
      );
    });
  });
});
