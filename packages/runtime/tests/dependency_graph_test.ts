// パッケージの依存グラフの門。GPU も実資産も要らない（`deno info` を回すだけ）。
//
// 守る不変条件は CLAUDE.md の「ランタイム依存は Web 標準 API のみ（依存パッケージも Web 標準 API
// のみで構成されたものに限る）」。型検査ではこれを守れない —— Deno の workspace ではルートの
// `deno.json` の `imports`（examples 専用の npm 依存や `@std/*`）がメンバーにも効くので、
// `packages/*` のソースが裸の指定子（`import "jpeg-js"`）を書いても `deno check` は通る。公開した
// パッケージにはルートの import map が付いていかないので、壊れるのは利用者の手元になってから。
//
// 見るのは各パッケージの公開の入口（その package の `deno.json` の `exports` の全ての値）から
// 辿ったモジュールグラフ（`deno info --json`）の、外部の指定子（ワークスペース内のファイル以外）。
// 許すのは次の 3 つだけ:
//
// - **同じメンバーの根の内側のファイル**。
// - **他のメンバーの公開の入口のファイル**（`@karume/*` はワークスペースの解決でここへ着く）。
//   他のメンバーの入口でないファイル（`../../hub/src/x.ts` のような根をまたぐ相対 import）と、
//   どのメンバーにも属さないファイル（リポジトリ根・`examples/`・`tools/` など）は落とす —— どちらも
//   公開したパッケージの外を指すので、利用者の手元では解決できない。
// - **辺の出どころのパッケージ自身の `deno.json` が `imports` で宣言した jsr パッケージ**（名前と
//   版の範囲の一致）。どのパッケージの宣言で許すかは、import を書いたファイルの置き場で決める ——
//   models が hub 経由で `@hdae/fetch-cache` を引くのは hub の宣言で許すが、models のファイルが
//   直に書けば models の宣言が要る。
//
// 外部のパッケージが別の外部のパッケージを引く辺（推移的な依存）は落とす（誰も宣言していない
// 依存がグラフに入る形）。`npm:` は宣言の有無によらず 1 つでも出たら落とす。許可の一覧はテストに
// 書き写さず、各 package の `deno.json` から導く。
//
// 門が見るのは `deno info` の静的グラフ（リテラルの import / import type / 型参照 / リテラルの
// 動的 import）に限る。`new Worker(new URL(...))` の先・リテラルでない動的 import・
// `import.meta.resolve` の先はグラフに載らないので、この門の外である。
//
// `deno info` は `--frozen-lockfile` で回す（lockfile との食い違いは書き換えずに落とす）。外部の
// jsr パッケージ（hub の `@hdae/fetch-cache`）は DENO_DIR のキャッシュから読む —— キャッシュが
// 冷えた機ではネットワークに出る（hub のテストの型検査と同じ条件）。取れなければ deno info の失敗か
// モジュールの error で赤になる（依存の違反としてではなく、その旨の文言で）。
//
// MUST: 読めない形は推測せず throw する —— 「読めなかった辺」を「外部の辺が無い」と読み替えると、
// この門は違反を緑で隠す。`deno info --json` の形（2.9.6 実測）: 辺の解決先は
// `dependencies[].code.specifier` / `.type.specifier` に入り、import map 経由の jsr は
// `jsr:@scope/name@range`（サブパスは `jsr:/@scope/name@range/sub` と先頭に `/` が付く）、npm は
// `npm:name@range`、jsr パッケージの中身のモジュールは `https://jsr.io/@scope/name/<版>/…`。
// トップレベルの `npmPackages` / `packages` はグラフでなく lockfile 全体の一覧なので使わない。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";

/** リポジトリ根（ワークスペースの `deno.json` の置き場）。 */
const REPO_ROOT = new URL("../../../", import.meta.url);

/** 当てにしている `deno info --json` の出力形式版。 */
const INFO_VERSION = 1;

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

const list = (value: unknown, what: string): readonly unknown[] => {
  if (!Array.isArray(value)) throw new Error(`依存グラフの門: ${what} が配列でない`);
  return value;
};

/** ワークスペースのメンバー 1 つ。 */
type Member = {
  readonly name: string;
  /** メンバーの根（末尾 `/` 付きの `file:` URL）。 */
  readonly root: string;
  /** 公開の入口（`exports` の全ての値を絶対 URL にしたもの）。 */
  readonly entries: readonly string[];
  /** `imports` が宣言した jsr パッケージ（`名前@範囲`）。 */
  readonly allowed: ReadonlySet<string>;
};

/** jsr の指定子を `名前@範囲` へ畳む（サブパスは落とす）。jsr でなければ `undefined`。 */
const jsrPackageOf = (specifier: string): string | undefined => {
  const matched = /^jsr:\/?(@[^/@]+\/[^/@]+)(?:@([^/]+))?(?:\/.*)?$/.exec(specifier);
  if (matched === null) {
    if (specifier.startsWith("jsr:")) {
      throw new Error(`依存グラフの門: jsr の指定子を読めない: ${specifier}`);
    }
    return undefined;
  }
  return `${matched[1]}@${matched[2] ?? ""}`;
};

/** `https://jsr.io/@scope/name/<版>/…` のモジュールが属するパッケージの根。jsr.io 以外は `undefined`。 */
const jsrModuleRoot = (specifier: string): string | undefined =>
  /^https:\/\/jsr\.io\/@[^/]+\/[^/]+\/[^/]+\//.exec(specifier)?.[0];

/**
 * `deno.json` の `imports` の値から許可の一覧を導く。
 *
 * 相対パス（`./` `../`）の宣言はローカルの別名なので許可の一覧に入れない —— その辺は `file:` へ
 * 解決され、ファイルの辺の規則（同じメンバーの内側か、他のメンバーの入口か）で判じる。
 *
 * MUST: それ以外の jsr でない宣言は推測せず throw する —— npm はこの門の禁則そのもので、https 等は
 * 「同じ依存か」を判じる規則をまだ持たない（読めない宣言を黙って許すと門が穴になる）。
 */
const allowedPackagesOf = (imports: Readonly<Record<string, string>>): Set<string> => {
  const allowed = new Set<string>();
  for (const [key, value] of Object.entries(imports)) {
    if (value.startsWith("./") || value.startsWith("../")) continue;
    const identity = jsrPackageOf(value);
    if (identity === undefined) {
      throw new Error(`依存グラフの門: jsr 以外の宣言である（imports["${key}"] = ${value}）`);
    }
    allowed.add(identity);
  }
  return allowed;
};

const readMembers = async (): Promise<Member[]> => {
  const root = record(
    JSON.parse(await Deno.readTextFile(new URL("deno.json", REPO_ROOT))),
    "ルートの deno.json",
  );
  return await Promise.all(
    list(root.workspace, "ルートの deno.json の workspace").map(async (path, index) => {
      const memberRoot = new URL(
        `${text(path, `workspace[${index}]`).replace(/\/+$/, "")}/`,
        REPO_ROOT,
      );
      const config = record(
        JSON.parse(await Deno.readTextFile(new URL("deno.json", memberRoot))),
        `${memberRoot.href}deno.json`,
      );
      const name = text(config.name, `${memberRoot.href}deno.json の name`);
      const exports = typeof config.exports === "string"
        ? [config.exports]
        : Object.values(record(config.exports, `${name} の exports`));
      if (exports.length === 0) throw new Error(`依存グラフの門: ${name} の exports が空である`);
      const imports = Object.fromEntries(
        Object.entries(record(config.imports ?? {}, `${name} の imports`)).map((
          [key, value],
        ) => [key, text(value, `${name} の imports["${key}"]`)]),
      );
      return {
        name,
        root: memberRoot.href,
        entries: exports.map((entry, at) =>
          new URL(text(entry, `${name} の exports[${at}]`), memberRoot).href
        ),
        allowed: allowedPackagesOf(imports),
      };
    }),
  );
};

/** グラフの辺 1 本（出どころのモジュール → 書かれた指定子 → 解決先）。 */
type Edge = {
  readonly from: string;
  readonly written: string;
  readonly resolved: string;
};

/** `deno info --json` の出力から辺を全て取り出す（code と type の解決先が割れれば両方）。 */
const edgesOf = (info: unknown): Edge[] => {
  const parsed = record(info, "deno info の出力");
  if (parsed.version !== INFO_VERSION) {
    throw new Error(
      `依存グラフの門: deno info の出力形式版が ${String(parsed.version)} である` +
        `（当てにしているのは ${INFO_VERSION}）`,
    );
  }
  const edges: Edge[] = [];
  for (const [at, value] of list(parsed.modules, "modules").entries()) {
    const module = record(value, `modules[${at}]`);
    const from = text(module.specifier, `modules[${at}].specifier`);
    if (module.error !== undefined) {
      throw new Error(`依存グラフの門: モジュールを読めない: ${from}: ${String(module.error)}`);
    }
    for (
      const [index, dependency] of list(module.dependencies ?? [], `${from} の dependencies`)
        .entries()
    ) {
      const edge = record(dependency, `${from} の dependencies[${index}]`);
      const written = text(edge.specifier, `${from} の dependencies[${index}].specifier`);
      const targets = new Set<string>();
      for (const side of ["code", "type"] as const) {
        if (edge[side] === undefined) continue;
        const target = record(edge[side], `${from} → ${written} の ${side}`);
        if (target.error !== undefined) {
          throw new Error(
            `依存グラフの門: 解決できない import がある: ${from} → ${written}: ${
              String(target.error)
            }`,
          );
        }
        targets.add(text(target.specifier, `${from} → ${written} の ${side}.specifier`));
      }
      if (targets.size === 0) {
        throw new Error(`依存グラフの門: 解決先の無い辺である: ${from} → ${written}`);
      }
      for (const resolved of targets) edges.push({ from, written, resolved });
    }
  }
  return edges;
};

/** 判定の結果。`external` は許した外部の辺の数（形が変わって外部の辺を 1 本も読めない門を見分ける）。 */
type Verdict = {
  readonly violations: readonly string[];
  readonly external: number;
};

/** 判定に要るメンバーの面（判定の単体テストが根の無い模型を渡せるよう `root` は含めない）。 */
type Owner = Pick<Member, "name" | "entries" | "allowed">;

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
    if (resolved.startsWith("npm:")) {
      violations.push(`npm の依存である（宣言の有無によらず禁止）: ${label}`);
      continue;
    }
    if (from.startsWith("file:")) {
      if (resolved.startsWith("file:")) {
        const source = ownerOf(from);
        const target = ownerOf(resolved);
        if (target === undefined) {
          violations.push(`どのメンバーにも属さないファイルへの import である: ${label}`);
        } else if (target.name !== source?.name && !target.entries.includes(resolved)) {
          violations.push(
            `${target.name} の公開の入口でないファイルへの、メンバーをまたぐ import である: ${label}`,
          );
        }
        continue;
      }
      const owner = ownerOf(from);
      const identity = jsrPackageOf(resolved);
      if (owner !== undefined && identity !== undefined && owner.allowed.has(identity)) {
        external++;
        continue;
      }
      violations.push(
        `${owner?.name ?? "どのメンバーにも属さないファイル"} の deno.json の imports が` +
          `宣言していない外部の依存である: ${label}`,
      );
      continue;
    }
    // 外部のモジュールから出る辺 —— 同じ jsr パッケージの中に閉じていれば許す。
    const root = jsrModuleRoot(from);
    if (root !== undefined && resolved.startsWith(root)) continue;
    violations.push(`外部のパッケージが別の外部の依存を引いている（推移的な依存）: ${label}`);
  }
  return { violations, external };
};

/** 入口 1 つのモジュールグラフの辺を `deno info --json` で採る。 */
const readEdges = async (entry: string): Promise<Edge[]> => {
  // cwd はワークスペース根（`@karume/*` をワークスペース内の実体へ解決させるため）。
  const info = await new Deno.Command(Deno.execPath(), {
    args: ["info", "--json", "--frozen-lockfile", entry],
    cwd: REPO_ROOT,
    stdout: "piped",
    stderr: "piped",
  }).output();
  const decoder = new TextDecoder();
  if (!info.success) {
    throw new Error(
      `依存グラフの門: deno info が失敗した（${entry}）:\n${decoder.decode(info.stderr)}`,
    );
  }
  return edgesOf(JSON.parse(decoder.decode(info.stdout)));
};

/** メンバー 1 つの全ての入口のグラフを判じる。 */
const judgeMember = async (member: Member, members: readonly Member[]): Promise<Verdict> => {
  const ownerOf = (file: string): Member | undefined =>
    members.find((candidate) => file.startsWith(candidate.root));
  const edges = (await Promise.all(member.entries.map(readEdges))).flat();
  return judgeEdges(edges, ownerOf);
};

const members = await readMembers();

describe("パッケージの依存グラフの門", () => {
  for (const member of members) {
    it(`${member.name} の公開の入口から辿った外部の依存は、出どころのパッケージが宣言した jsr だけである`, async () => {
      const { violations } = await judgeMember(member, members);
      assertEquals(
        violations,
        [],
        "ルートの deno.json の imports はワークスペースのメンバーにも効くので deno check は通るが、" +
          "公開したパッケージでは解決できない（または Web 標準 API だけで閉じない）依存がある",
      );
    });
  }

  it("対照: hub が宣言した @hdae/fetch-cache の辺を許した外部の辺として数える", async () => {
    // 出力の形が変わって外部の辺を 1 本も読めなくなると、上の門は無音で緑になる。hub は
    // `@hdae/fetch-cache` を実際に使うので、その辺を読めていることをここで見る。
    const hub = members.find((member) => member.name === "@karume/hub");
    assert(hub !== undefined, "ワークスペースに @karume/hub が無い");
    const { external } = await judgeMember(hub, members);
    assert(
      external > 0,
      "hub のグラフに許した外部の辺が 1 本も無い（deno info の出力を読めていない）",
    );
  });
});

describe("依存グラフの門の判定", () => {
  const owner = {
    name: "@karume/example",
    entries: ["file:///repo/packages/example/mod.ts"],
    allowed: allowedPackagesOf({ "@x/y": "jsr:@x/y@^1.2.0" }),
  };
  const other = {
    name: "@karume/other",
    entries: ["file:///repo/packages/other/mod.ts"],
    allowed: new Set<string>(),
  };
  const ownerOf = (path: string) =>
    path.startsWith("file:///repo/packages/example/")
      ? owner
      : path.startsWith("file:///repo/packages/other/")
      ? other
      : undefined;
  const file = "file:///repo/packages/example/src/a.ts";

  it("同じメンバーの内側のファイルと、他のメンバーの公開の入口は許す", () => {
    const verdict = judgeEdges([
      { from: file, written: "./b.ts", resolved: "file:///repo/packages/example/src/b.ts" },
      { from: file, written: "@karume/other", resolved: "file:///repo/packages/other/mod.ts" },
    ], ownerOf);
    assertEquals(verdict, { violations: [], external: 0 });
  });

  it("他のメンバーの入口でないファイルへの、根をまたぐ相対 import は落とす", () => {
    const { violations } = judgeEdges([{
      from: file,
      written: "../../other/src/x.ts",
      resolved: "file:///repo/packages/other/src/x.ts",
    }], ownerOf);
    assertEquals(violations.length, 1);
    assert(violations[0].includes("@karume/other の公開の入口でない"), violations[0]);
  });

  it("どのメンバーにも属さないワークスペース内のファイルへの import は落とす", () => {
    const { violations } = judgeEdges(
      [{ from: file, written: "../../../shared.ts", resolved: "file:///repo/shared.ts" }],
      ownerOf,
    );
    assertEquals(violations.length, 1);
    assert(violations[0].includes("どのメンバーにも属さない"), violations[0]);
  });

  it("相対パスの宣言はローカルの別名として許可の一覧に入れない", () => {
    assertEquals(
      allowedPackagesOf({ "#util": "./src/util.ts", "@x/y": "jsr:@x/y@^1.2.0" }),
      new Set(["@x/y@^1.2.0"]),
    );
  });

  it("宣言した jsr はサブパス（jsr:/ の綴り）でも許す", () => {
    const verdict = judgeEdges(
      [{ from: file, written: "@x/y/sub", resolved: "jsr:/@x/y@^1.2.0/sub" }],
      () => owner,
    );
    assertEquals(verdict, { violations: [], external: 1 });
  });

  it("版の範囲が宣言と違う jsr は落とす", () => {
    const { violations } = judgeEdges(
      [{ from: file, written: "jsr:@x/y@^2", resolved: "jsr:@x/y@^2" }],
      () => owner,
    );
    assertEquals(violations.length, 1);
  });

  it("npm は出どころによらず落とし、npm の宣言は許可の一覧に入れない", () => {
    const { violations } = judgeEdges(
      [{ from: file, written: "jpeg-js", resolved: "npm:jpeg-js@^0.4.4" }],
      () => owner,
    );
    assertEquals(violations.length, 1);
    assertThrows(() => allowedPackagesOf({ "jpeg-js": "npm:jpeg-js@^0.4.4" }), Error, "jsr 以外");
  });

  it("jsr パッケージの中の辺は同じパッケージに閉じていれば許し、別のパッケージへ出れば落とす", () => {
    const inner = "https://jsr.io/@x/y/1.2.3/mod.ts";
    const { violations } = judgeEdges([
      { from: inner, written: "./a.ts", resolved: "https://jsr.io/@x/y/1.2.3/a.ts" },
      { from: inner, written: "jsr:@x/z@^1", resolved: "jsr:@x/z@^1" },
    ], () => owner);
    assertEquals(violations.length, 1);
    assert(violations[0].includes("jsr:@x/z@^1"), violations[0]);
  });
});
