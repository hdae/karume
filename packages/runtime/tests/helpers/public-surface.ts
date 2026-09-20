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
// 採り方は `deno doc --json <entry>` の出力。`deno.json` の `exports` が entry の真実源で、
// 面が 1 つ増えれば走査も自動で増える（門の側に面の一覧を二重に持たない）。
//
// MUST: 読めない形は推測せず throw する — 「採れなかった」を「面が空」と読み替えると、この門は
// 面が丸ごと消えたことを緑で隠す。

import { assert } from "@std/assert";

/** fixture の形式版。形を変えるときは上げて、読み手（`parseSnapshot`）も直す。 */
const SCHEMA = 1;

/** 当てにしている `deno doc --json` の出力形式版。 */
const DOC_VERSION = 2;

/** スナップショットの書き直しを指示する環境変数（未設定 = 比較 / `"write"` = 書き直し）。 */
const MODE_ENV = "KARUME_SURFACE";

/** 面が食い違ったときに読み手へ渡す手順。 */
const HOW_TO_UPDATE =
  `意図した変更なら ${MODE_ENV}=write で同じテストを回してスナップショットを更新し、差分を CHANGELOG に書くこと。`;

/**
 * 公開シンボル 1 つ。`kind` は `deno doc` の語彙をそのまま使う（`class` / `function` /
 * `typeAlias` / `variable` …）— 独自の分類を挟むと、deno 側が語彙を増やしたときに黙って
 * 丸められる。
 */
export type PublicSymbol = {
  readonly name: string;
  readonly kind: string;
};

/** fixture の形。`entries` のキーは `deno.json` の `exports` の綴り。 */
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
const compare = (left: string, right: string): number => left < right ? -1 : left > right ? 1 : 0;

const bySymbol = (left: PublicSymbol, right: PublicSymbol): number =>
  left.name === right.name ? compare(left.kind, right.kind) : compare(left.name, right.name);

/**
 * 集合演算に使う「キー → シンボル」の索引。キーは `name` と `kind` の対を畳んだもので、区切りの
 * `::` は TS の識別子にも `deno doc` の kind にも現れない。キーは**読み戻さない** —— 差分の表示は
 * 索引が持つシンボルそのものから組む。
 */
const index = (symbols: readonly PublicSymbol[]): ReadonlyMap<string, PublicSymbol> =>
  new Map(symbols.map((symbol) => [`${symbol.name}::${symbol.kind}`, symbol]));

/** 差分の 1 行。 */
const labelOf = (symbol: PublicSymbol): string => `${symbol.name} (${symbol.kind})`;

/**
 * `deno.json` の `exports` を「entry の綴り → パッケージ根からの相対パス」へ正規化する。
 *
 * 文字列 1 本の形（`"exports": "./mod.ts"` — runtime）はキーを持たないので、マップ形の既定面と
 * 同じ `"."` を綴りとして与える（fixture のキーを 2 つの形で割らない）。
 */
const exportEntries = (exports: unknown): ReadonlyMap<string, string> => {
  if (typeof exports === "string") return new Map([[".", exports]]);
  const entries = new Map<string, string>();
  for (const [spec, target] of Object.entries(record(exports, "deno.json の exports"))) {
    entries.set(spec, text(target, `exports["${spec}"]`));
  }
  if (entries.size === 0) throw new Error("公開面の門: deno.json の exports が空である");
  return entries;
};

/**
 * entry 1 つが export するシンボルを `deno doc --json` から採る。
 *
 * `cwd` はワークスペース根（`@karume/*` のバレ指定子をワークスペース内の実体へ解決させるため —
 * パッケージ側の `imports` だけで解決すると JSR の公開版を見に行く）。
 */
const readSurface = async (entry: URL, workspaceRoot: URL): Promise<readonly PublicSymbol[]> => {
  const doc = await new Deno.Command(Deno.execPath(), {
    args: ["doc", "--json", entry.href],
    cwd: workspaceRoot,
    stdout: "piped",
    stderr: "piped",
  }).output();
  const decoder = new TextDecoder();
  if (!doc.success) {
    throw new Error(
      `公開面の門: deno doc が失敗した（${entry.href}）:\n${decoder.decode(doc.stderr)}`,
    );
  }
  const parsed = record(JSON.parse(decoder.decode(doc.stdout)), "deno doc の出力");
  if (parsed.version !== DOC_VERSION) {
    throw new Error(
      `公開面の門: deno doc の出力が version ${DOC_VERSION} でない（${String(parsed.version)}）` +
        " — 採り方を見直すこと",
    );
  }
  const nodes = record(parsed.nodes, "deno doc の nodes");
  const node = nodes[entry.href];
  if (node === undefined) {
    throw new Error(
      `公開面の門: ${entry.href} の節が deno doc の出力に無い（節: ${
        Object.keys(nodes).join(", ")
      }）`,
    );
  }
  // module doc は `symbols` の**兄弟**（`module_doc`）なので、`symbols` だけ読めば除外できる。
  const symbols = list(record(node, "deno doc の節").symbols, "節の symbols");
  const collected: PublicSymbol[] = [];
  for (const rawSymbol of symbols) {
    const symbol = record(rawSymbol, "symbols の要素");
    const name = text(symbol.name, "シンボルの name");
    for (const rawDeclaration of list(symbol.declarations, `${name} の declarations`)) {
      const declaration = record(rawDeclaration, `${name} の declarations の要素`);
      // `--private` を付けていないので export 以外は出ない。出たら採り方の前提が崩れている。
      const declarationKind = text(declaration.declarationKind, `${name} の declarationKind`);
      if (declarationKind !== "export") {
        throw new Error(
          `公開面の門: ${name} の declarationKind が export でない（${declarationKind}）`,
        );
      }
      collected.push({ name, kind: text(declaration.kind, `${name} の kind`) });
    }
  }
  // overload は同じ名前で宣言が複数並ぶ（kind は同じ）。索引に入れる過程で 1 つへ畳まれる。
  return [...index(collected).values()].sort(bySymbol);
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
  for (
    const [spec, value] of Object.entries(record(parsed.entries, "スナップショットの entries"))
  ) {
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
  // entry の綴りで並べる（`deno.json` の並べ替えが fixture の差分に化けないようにする）。
  for (const spec of [...entries.keys()].sort(compare)) ordered[spec] = entries.get(spec) ?? [];
  return `${JSON.stringify({ schema: SCHEMA, entries: ordered }, null, 2)}\n`;
};

/**
 * パッケージ 1 つの公開面をスナップショットと突き合わせる（`KARUME_SURFACE=write` なら書き直す）。
 *
 * @param packageRoot パッケージ根（`deno.json` の置き場）。fixture は
 *   `<packageRoot>/tests/fixtures/public-surface.json`。
 */
export const assertPublicSurface = async (packageRoot: URL): Promise<void> => {
  const mode = Deno.env.get(MODE_ENV);
  if (mode !== undefined && mode !== "write") {
    throw new Error(`公開面の門: ${MODE_ENV} が受けるのは未設定か "write" だけである（${mode}）`);
  }
  const config = record(
    JSON.parse(await Deno.readTextFile(new URL("deno.json", packageRoot))),
    `${packageRoot.href}deno.json`,
  );
  // `packages/<pkg>/` の 2 つ上がワークスペース根（root `deno.json` の置き場）。
  const workspaceRoot = new URL("../../", packageRoot);
  const actual = new Map<string, readonly PublicSymbol[]>();
  for (const [spec, target] of exportEntries(config.exports)) {
    actual.set(spec, await readSurface(new URL(target, packageRoot), workspaceRoot));
  }

  const snapshotFile = new URL("tests/fixtures/public-surface.json", packageRoot);
  if (mode === "write") {
    await Deno.writeTextFile(snapshotFile, serialize(actual));
    return;
  }

  const expected = parseSnapshot(await Deno.readTextFile(snapshotFile), snapshotFile).entries;
  const specs = [...new Set([...Object.keys(expected), ...actual.keys()])].sort(compare);
  const diff: string[] = [];
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
