// 公開面のスナップショット門の実体（ADR [0008](../../../../docs/decisions/0008-public-api.md)
// 追記 2026-09-20）。3 パッケージの `tests/public_surface_test.ts` が共通で使う。
//
// 名指しの門（`packages/models/tests/models_barrel_surface_test.ts`）との役割分担:
//
// - **名指し = 意図の宣言** — 「この綴りは出す / この綴りは出さない」を人が書く。書いていない
//   綴りは素通りする（リストに無い export が増えても減っても緑）。
// - **スナップショット = 増減の検出**（ここ）— 何が正しいかは言わず、前回との差だけを見る。
//   型 export も含めて採るので、値では観測できない面（`export type` だけの再輸出）も縛れる。
//
// 採り方は TypeScript 7 の同期 API（`typescript/unstable/sync`）で entry の module symbol の
// exports を引く。`package.json` の `exports` が entry の真実源で、面が 1 つ増えれば走査も
// 自動で増える（門の側に面の一覧を二重に持たない）。kind の語彙は以前の採り方（`deno doc`）の
// 語彙を引き継ぐ（fixture をそのまま読むため — 分類規則は `kindsOf` に書く）。
//
// MUST: 読めない形は推測せず throw する — 「採れなかった」を「面が空」と読み替えると、この門は
// 面が丸ごと消えたことを緑で隠す。

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

/** fixture の形式版。形を変えるときは上げて、読み手（`parseSnapshot`）も直す。 */
const SCHEMA = 1;

/** スナップショットの書き直しを指示する環境変数（未設定 = 比較 / `"write"` = 書き直し）。 */
const MODE_ENV = "KARUME_SURFACE";

/** 面が食い違ったときに読み手へ渡す手順。 */
const HOW_TO_UPDATE = `意図した変更なら ${MODE_ENV}=write で同じテストを回してスナップショットを更新し、差分を CHANGELOG に書くこと。`;

/**
 * 公開シンボル 1 つ。`kind` は `deno doc` の語彙（`class` / `function` / `typeAlias` /
 * `variable` / `interface` / `enum` / `namespace`）— fixture はこの語彙で書かれている。
 */
export type PublicSymbol = {
  readonly name: string;
  readonly kind: string;
};

/** fixture の形。`entries` のキーは `package.json` の `exports` の綴り。 */
type SurfaceSnapshot = {
  readonly schema: number;
  readonly entries: Readonly<Record<string, readonly PublicSymbol[]>>;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const record = (value: unknown, what: string): Record<string, unknown> => {
  if (!isRecord(value)) throw new Error(`公開面の門: ${what} がオブジェクトでない`);
  return value;
};

const text = (value: unknown, what: string): string => {
  if (typeof value !== "string") throw new Error(`公開面の門: ${what} が文字列でない`);
  return value;
};

const list = (value: unknown, what: string): readonly unknown[] => {
  if (!Array.isArray(value)) throw new Error(`公開面の門: ${what} が配列でない`);
  return value;
};

/** ロケール非依存の辞書順（fixture の並びを実行環境で揺らさない）。 */
const compare = (left: string, right: string): number => (left < right ? -1 : left > right ? 1 : 0);

const bySymbol = (left: PublicSymbol, right: PublicSymbol): number =>
  left.name === right.name ? compare(left.kind, right.kind) : compare(left.name, right.name);

/**
 * 集合演算に使う「キー → シンボル」の索引。キーは `name` と `kind` の対を畳んだもので、区切りの
 * `::` は TS の識別子にも kind の語彙にも現れない。キーは**読み戻さない** —— 差分の表示は
 * 索引が持つシンボルそのものから組む。
 */
const index = (symbols: readonly PublicSymbol[]): ReadonlyMap<string, PublicSymbol> =>
  new Map(symbols.map((symbol) => [`${symbol.name}::${symbol.kind}`, symbol]));

/** 差分の 1 行。 */
const labelOf = (symbol: PublicSymbol): string => `${symbol.name} (${symbol.kind})`;

/**
 * `package.json` の `exports` を「entry の綴り → パッケージ根からの相対パス」へ正規化する。
 *
 * 今の 3 パッケージはどれもマップ形で、値は文字列 1 本。文字列 1 本の形（`"exports": "./mod.ts"`）
 * はキーを持たないので、マップ形の既定面と同じ `"."` を綴りとして与える（fixture のキーを 2 つの
 * 形で割らない）。条件付き exports（値がオブジェクト）は読まずに throw する — どの条件の entry を
 * 面と見なすかはこの門では決めない。
 */
const exportEntries = (exports: unknown): ReadonlyMap<string, string> => {
  if (typeof exports === "string") return new Map([[".", exports]]);
  const entries = new Map<string, string>();
  for (const [spec, target] of Object.entries(record(exports, "package.json の exports"))) {
    entries.set(spec, text(target, `exports["${spec}"]`));
  }
  if (entries.size === 0) throw new Error("公開面の門: package.json の exports が空である");
  return entries;
};

/** symbol の flags → kind。1 つの symbol が複数を併せ持つ（宣言の併合）なら kind も複数出す。 */
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
 * 変数宣言 1 つの kind。`deno doc` は初期化子が**直に** arrow 関数 / function 式なら
 * `function`、class 式なら `class` と数え、それ以外（括弧・`as`・`satisfies` で包んだ形を
 * 含む）は `variable` と数える（deno 2 系の `deno doc --json` で実測 2026-10-16）。fixture は
 * その語彙で書かれているので同じ規則で分ける。
 */
const variableKind = (declaration: TsSymbol["declarations"][number], name: string): string => {
  const node = declaration.resolve();
  if (node === undefined || !isVariableDeclaration(node)) {
    throw new Error(`公開面の門: ${name} の変数宣言を AST へ解決できない`);
  }
  const initializer = node.initializer;
  if (initializer === undefined) return "variable";
  if (isArrowFunction(initializer) || isFunctionExpression(initializer)) return "function";
  if (isClassExpression(initializer)) return "class";
  return "variable";
};

/**
 * export された symbol（別名なら解決済みの実体）1 つの kind の一覧。overload は flags が 1 つ
 * なので 1 件に畳まれる。型だけの再輸出（`export type { X }`）も実体の kind で数える
 * （`deno doc` も区別しなかった）。
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
  // 語彙に無い形（ここまで来るのは想定外）を「面に無い」と読み替えない。
  if (kinds.length === 0) {
    throw new Error(`公開面の門: ${name} の kind を決められない（flags: ${symbol.flags}）`);
  }
  return kinds;
};

/** entry 1 つが export するシンボルを型検査器から採る。 */
const readSurface = (checker: Checker, program: Program, entry: string): PublicSymbol[] => {
  const sourceFile = program.getSourceFile(entry);
  if (sourceFile === undefined) {
    throw new Error(`公開面の門: ${entry} が tsconfig.json の program に入っていない`);
  }
  const moduleSymbol = checker.getSymbolAtLocation(sourceFile);
  if (moduleSymbol === undefined) {
    throw new Error(`公開面の門: ${entry} の module symbol が無い（module として読めない）`);
  }
  const collected: PublicSymbol[] = [];
  for (const exported of checker.getExportsOfModule(moduleSymbol)) {
    // 名前は export 側の綴り（`export { a as b }` は `b`）、kind は別名を辿った実体から採る。
    const name = exported.name;
    const target =
      (exported.flags & SymbolFlags.Alias) === 0 ? exported : checker.getAliasedSymbol(exported);
    if (checker.isUnknownSymbol(target)) {
      throw new Error(`公開面の門: ${entry} の ${name} の別名を解決できない`);
    }
    for (const kind of kindsOf(target, name)) collected.push({ name, kind });
  }
  // 同じ kind が重なる形（同名の変数宣言が複数）は索引に入れる過程で 1 つへ畳まれる。
  return [...index(collected).values()].sort(bySymbol);
};

/**
 * パッケージの全 entry の公開面を採る。tsgo のプロセスを 1 つ起こし、ワークスペース根の
 * `tsconfig.json` の program 上で読む（`@karume/*` の指定子をワークスペース内の実体へ
 * 解決させるため）。
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
      throw new Error(`公開面の門: ${configFile} の project を開けない`);
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

/** fixture を読み、形を検査して返す。 */
const parseSnapshot = (source: string, file: URL): SurfaceSnapshot => {
  const parsed = record(JSON.parse(source), `${file.href} の中身`);
  if (parsed.schema !== SCHEMA) {
    throw new Error(
      `公開面の門: スナップショットの schema が ${SCHEMA} でない（${String(parsed.schema)}）`,
    );
  }
  const entries: Record<string, readonly PublicSymbol[]> = {};
  for (const [spec, value] of Object.entries(
    record(parsed.entries, "スナップショットの entries"),
  )) {
    entries[spec] = list(value, `entries["${spec}"]`).map((item) => {
      const symbol = record(item, `entries["${spec}"] の要素`);
      return {
        name: text(symbol.name, `entries["${spec}"] の name`),
        kind: text(symbol.kind, `entries["${spec}"] の kind`),
      };
    });
  }
  return { schema: SCHEMA, entries };
};

/** fixture の書き出し形（2 スペース・entry の綴り順・末尾改行）。 */
const serialize = (entries: ReadonlyMap<string, readonly PublicSymbol[]>): string => {
  const ordered: Record<string, readonly PublicSymbol[]> = {};
  // entry の綴りで並べる（`package.json` の並べ替えが fixture の差分に化けないようにする）。
  for (const spec of [...entries.keys()].sort(compare)) ordered[spec] = entries.get(spec) ?? [];
  return `${JSON.stringify({ schema: SCHEMA, entries: ordered }, null, 2)}\n`;
};

/**
 * パッケージ 1 つの公開面をスナップショットと突き合わせる（`KARUME_SURFACE=write` なら書き直す）。
 *
 * @param packageRoot パッケージ根（`package.json` の置き場）。fixture は
 *   `<packageRoot>/tests/fixtures/public-surface.json`。
 */
export const assertPublicSurface = async (packageRoot: URL): Promise<void> => {
  const mode = process.env[MODE_ENV];
  if (mode !== undefined && mode !== "write") {
    throw new Error(`公開面の門: ${MODE_ENV} が受けるのは未設定か "write" だけである（${mode}）`);
  }
  const config = record(
    JSON.parse(await readTextFile(new URL("package.json", packageRoot))),
    `${packageRoot.href}package.json`,
  );
  // `packages/<pkg>/` の 2 つ上がワークスペース根（root `tsconfig.json` の置き場）。
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
  // 面（entry）そのものの増減は symbol 差分と**独立に**出す。両側の不在は下の `?? []` で
  // 空配列へ畳まれるので、シンボルを 1 つも出さない面（`export {}` だけのモジュール）の
  // 追加も削除も symbol 差分には一度も現れない。
  for (const spec of specs) {
    const present = actual.has(spec);
    if (present === Object.hasOwn(expected, spec)) continue;
    diff.push(`  ${present ? "+" : "-"} 公開面 ${spec}`);
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
    `公開面がスナップショットと食い違う（+ = 増えた / - = 消えた）:\n${diff.join("\n")}\n` +
      `スナップショット: ${snapshotFile.href}\n${HOW_TO_UPDATE}`,
  );
};
