/**
 * verify-diff — 複数の機で採った検証結果を並べ、割れた所を 1 コマンドで出す読み取り専用の道具。
 *
 *     deno run -A tools/verify-diff/main.ts
 *     deno run -A tools/verify-diff/main.ts --family anima --family golden
 *     deno run -A tools/verify-diff/main.ts --date 2026-09-21
 *     deno run -A tools/verify-diff/main.ts --root ~/verify-from-another-box --json
 *
 * 読むのは `outputs/verify/<環境キー>/<日付>_<系列>/results.json`（ADR 0106 決定 5 の席）だけで、
 * 何も書かない。2 台目の結果は**手で** `outputs/verify/<環境キー>/` へコピーする前提である
 * （追跡外の結果を追跡下へ持ち込まないための席の分割なので、同期機構は置かない。別の置き場に
 * まとめてあるなら `--root` で指す）。
 *
 * 環境ごとに採るのは**最新の日付**の席 1 本（`--date` を付けるとその日付だけ）。
 *
 * 終了コードは**差異があっても 0** — これは門ではなく、割れた場所を指す道具である。sha256 の
 * クロスデバイス一致はそもそも保証しない（docs/limitations.md）ので、実物の sha 違いも差異として
 * 挙げるだけ。1 で落ちるのは読めない・スキーマが違うときだけで、黙って「差異なし」は出さない。
 */

import {
  buildDiff,
  type Cell,
  type Difference,
  type DiffReport,
  type FamilyMatrix,
  type LoadedResults,
  type Measurement,
  parseResultsDocument,
} from "./diff.ts";

/** 既定の根（`outputs/verify/`）— results.ts と同じく import.meta.url から解決する。 */
const DEFAULT_ROOT = new URL("../../outputs/verify/", import.meta.url);

/** 値を取るオプション。 */
const OPTIONS: ReadonlySet<string> = new Set(["root", "family", "date"]);
/** 値を取らないオプション（opbench の `--キー 値` の対に 1 形だけ足す）。 */
const FLAGS: ReadonlySet<string> = new Set(["json"]);

const USAGE = `使い方: deno run -A tools/verify-diff/main.ts [オプション]
  --root <dir>         結果の根（既定 = このリポジトリの outputs/verify/）
  --family <name>      この系列だけ（繰り返し可・既定 = 根にある全系列）
  --date YYYY-MM-DD    この日付の席だけ（既定 = 環境ごとに最新の日付）
  --json               行列と警告を JSON で出す（既定 = Markdown）`;

export type Args = {
  readonly values: ReadonlyMap<string, readonly string[]>;
  readonly flags: ReadonlySet<string>;
};

/**
 * `--キー 値` の並びを読む（opbench の流儀 — 対の崩れ・値の書き忘れ・未知のキーで落ちる）。
 *
 * 値を取らない `--json` だけは 1 語で消費する。対を数えるだけの走査だと、旗 1 つで以降の
 * 対が丸ごと 1 語ずれて別の意味に読めてしまうため、旗は既知集合で先に振り分ける。
 */
export const parseArgs = (argv: readonly string[]): Args => {
  const values = new Map<string, string[]>();
  const flags = new Set<string>();
  for (let at = 0; at < argv.length;) {
    const key = argv[at];
    if (!key.startsWith("--")) throw new Error(`引数 ${key} が '--キー 値' の対になっていない`);
    const name = key.slice(2);
    if (FLAGS.has(name)) {
      flags.add(name);
      at += 1;
      continue;
    }
    // 未知のオプションを黙って捨てない（`--families` の打ち間違いが全系列に静かに落ちる）。
    if (!OPTIONS.has(name)) throw new Error(`未知のオプション --${name}。\n${USAGE}`);
    const value = argv[at + 1];
    if (value === undefined) throw new Error(`引数 ${key} が '--キー 値' の対になっていない`);
    if (value.startsWith("--")) {
      throw new Error(`引数 ${key} の値が無い（'${value}' はオプション）`);
    }
    values.set(name, [...(values.get(name) ?? []), value]);
    at += 2;
  }
  return { values, flags };
};

/** 1 度しか指定できないキーを読む。 */
export const single = (args: Args, name: string): string | undefined => {
  const values = args.values.get(name);
  if (values === undefined) return undefined;
  if (values.length > 1) throw new Error(`--${name} は 1 度しか指定できない`);
  return values[0];
};

/** 席のディレクトリ名（`<日付>_<系列>`）。系列名は文書側が正本なので、ここでは日付だけ採る。 */
const SEAT = /^(\d{4}-\d{2}-\d{2})_.+$/;

const DATE = /^\d{4}-\d{2}-\d{2}$/;

/**
 * OS path をディレクトリ URL にする（tools/_shared/assets.ts は IR 解析ごと引き込むので写す）。
 *
 * MUST: 区切りごとに `encodeURIComponent` してから組む（{@link childUrl} と同じ流儀）。path を
 * そのまま `new URL` へ渡すと `#` / `?` / `%` が URL 構文として解かれ、実体 `x%41y` を渡したのに
 * `xAy` を黙って読む（別の場所の結果を、打った綴りの根として表示する）形になる。
 */
const directoryUrl = (path: string): URL => {
  const absolute = path.startsWith("/") ? path : `${Deno.cwd()}/${path}`;
  const encoded = absolute.split("/").map(encodeURIComponent).join("/");
  return new URL(`file://${encoded.endsWith("/") ? encoded : `${encoded}/`}`);
};

/** 子ディレクトリの URL（名前に `#` / `?` が入っても別の場所を指さないようにする）。 */
const childUrl = (parent: URL, name: string): URL =>
  new URL(`${encodeURIComponent(name)}/`, parent);

/**
 * 根を歩いて `results.json` を全部読む。
 *
 * 席らしい名前（`<日付>_<系列>`）のディレクトリに `results.json` が無ければ throw する
 * （席は作られた時点で「走行中」の `results.json` を持つので、無いのは壊れた席である）。
 * 席の綴りでないディレクトリとファイルは席ではないので黙って飛ばす。
 */
const loadRoot = async (root: URL, label: string): Promise<LoadedResults[]> => {
  const loaded: LoadedResults[] = [];
  let environments: Deno.DirEntry[];
  try {
    environments = await Array.fromAsync(Deno.readDir(root));
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) {
      throw new Error(`結果の根が無い: ${label}（--root で置き場を指す）`);
    }
    throw cause;
  }
  for (const environment of environments) {
    if (!environment.isDirectory) continue;
    const environmentDir = childUrl(root, environment.name);
    for (const seat of await Array.fromAsync(Deno.readDir(environmentDir))) {
      const matched = SEAT.exec(seat.name);
      if (!seat.isDirectory || matched === null) continue;
      const path = `${label}/${environment.name}/${seat.name}/results.json`;
      const file = new URL("results.json", childUrl(environmentDir, seat.name));
      let text: string;
      try {
        text = await Deno.readTextFile(file);
      } catch (cause) {
        if (cause instanceof Deno.errors.NotFound) {
          throw new Error(`${path}: 席に results.json が無い（壊れた席）`);
        }
        throw cause;
      }
      let parsed: unknown;
      try {
        parsed = JSON.parse(text);
      } catch (cause) {
        const reason = cause instanceof Error ? cause.message : String(cause);
        throw new Error(`${path}: JSON として読めない（${reason}）`);
      }
      loaded.push({ path, date: matched[1], document: parseResultsDocument(parsed, path) });
    }
  }
  return loaded;
};

/** 表の升に入れる文字列（`|` は表を壊すので逃がす）。 */
const md = (text: string): string => text.replaceAll("|", "\\|");

/** 帯と実測は桁が離れるので、指数と固定を桁で切り替える。 */
const num = (value: number | null): string => {
  if (value === null) return "null";
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude < 1e-3 || magnitude >= 1e4) return value.toExponential(2);
  return String(Number(value.toPrecision(4)));
};

const shortSha = (sha: string): string => sha.slice(0, 7);

const describeDifference = (difference: Difference): string => {
  const values = Object.entries(difference.values)
    .map(([environment, value]) =>
      `${environment}=${difference.kind === "actual" ? shortSha(value) : value}`
    )
    .join(" / ");
  if (difference.kind === "status") {
    return `\`${difference.caseId}\`: status が割れている — ${values}`;
  }
  if (difference.kind === "actual") {
    // クロスデバイスのビット同一は非保証（ADR 0106）なので、割れていても赤にはしない。
    return `\`${difference.caseId}\`: 実物の sha が違う（クロスデバイスでは非保証・差異として挙げるだけ） — ${values}`;
  }
  return `\`${difference.caseId}\`: ${
    (difference.missing ?? []).join(" / ")
  } に結果が無い — ${values}`;
};

/** 許容差の実測（帯に対する比は rtol=0 のときだけ導ける — rtol が乗ると帯が要素ごとに動く）。 */
const measurementRows = (
  environments: readonly string[],
  cells: Readonly<Record<string, Cell>>,
): string[] => {
  const rows: string[] = [];
  const outputs = new Set<string>();
  for (const environment of environments) {
    const cell = Object.hasOwn(cells, environment) ? cells[environment] : undefined;
    for (const measurement of cell?.measurements ?? []) outputs.add(measurement.output);
  }
  for (const output of [...outputs].sort()) {
    for (const environment of environments) {
      const cell = Object.hasOwn(cells, environment) ? cells[environment] : undefined;
      const found: Measurement | undefined = cell?.measurements?.find((one) =>
        one.output === output
      );
      if (found === undefined) continue;
      const { tolerance, maxAbs } = found;
      const margin = tolerance.rtol === 0 && maxAbs !== null && tolerance.atol !== 0
        ? num(maxAbs / tolerance.atol)
        : "-";
      rows.push(
        `| ${md(output)} | ${md(environment)} | ${num(maxAbs)} | ${num(found.maxRel)} | ` +
          `atol=${num(tolerance.atol)} rtol=${num(tolerance.rtol)} | ${margin} | ${found.stage} |`,
      );
    }
  }
  return rows;
};

const renderFamily = (matrix: FamilyMatrix): string[] => {
  const lines: string[] = [`## ${matrix.family}`, ""];
  for (const one of matrix.selected) {
    const checkout = one.checkout === null
      ? "checkout 無し"
      : `${shortSha(one.checkout.sha)}${one.checkout.dirty ? "・dirty" : ""}`;
    lines.push(`- ${one.environment}: ${one.date}（${checkout}）`);
  }
  if (matrix.absent.length > 0) {
    lines.push(`- この系列の結果が無い環境: ${matrix.absent.join(" / ")}`);
  }
  lines.push("");
  lines.push(`| ケース | ${matrix.environments.map(md).join(" | ")} |`);
  lines.push(`| --- | ${matrix.environments.map(() => "---").join(" | ")} |`);
  for (const row of matrix.rows) {
    const cells = matrix.environments.map((environment) =>
      Object.hasOwn(row.cells, environment) ? row.cells[environment].status : "—"
    );
    lines.push(`| ${md(row.id)} | ${cells.join(" | ")} |`);
  }
  lines.push("");
  if (matrix.warnings.length > 0) {
    lines.push("警告:", "");
    for (const warning of matrix.warnings) lines.push(`- ${warning}`);
    lines.push("");
  }
  if (matrix.differences.length === 0) {
    lines.push("差異なし", "");
  } else {
    lines.push("差異:", "");
    for (const difference of matrix.differences) lines.push(`- ${describeDifference(difference)}`);
    lines.push("");
  }
  for (const row of matrix.rows) {
    const rows = measurementRows(matrix.environments, row.cells);
    if (rows.length === 0) continue;
    lines.push(`### ${row.id} の実測`, "");
    lines.push("| 出力 | 環境 | maxAbs | maxRel | 帯 | maxAbs/atol | 段 |");
    lines.push("| --- | --- | --- | --- | --- | --- | --- |");
    lines.push(...rows, "");
  }
  return lines;
};

const renderMarkdown = (report: DiffReport, label: string): string => {
  const lines: string[] = ["# verify-diff", "", `根: ${label}`, ""];
  for (const warning of report.warnings) lines.push(`- 警告: ${warning}`);
  if (report.warnings.length > 0) lines.push("");
  if (report.families.length === 0) {
    lines.push("突き合わせる結果が 1 本も無い（根に席が無いか、絞り込みに当たっていない）", "");
  }
  for (const matrix of report.families) lines.push(...renderFamily(matrix));
  return lines.join("\n");
};

const run = async (argv: readonly string[]): Promise<void> => {
  const args = parseArgs(argv);
  const rootArg = single(args, "root");
  const root = rootArg === undefined ? DEFAULT_ROOT : directoryUrl(rootArg);
  // 文面に出す根は人が打った綴りのまま（既定のときだけ URL から素の path へ戻す）。
  const label = (rootArg ?? decodeURIComponent(DEFAULT_ROOT.pathname)).replace(/\/+$/, "");
  const date = single(args, "date");
  if (date !== undefined && !DATE.test(date)) {
    throw new Error(`--date は YYYY-MM-DD の形（'${date}'）`);
  }
  const families = args.values.get("family");
  const report = buildDiff(await loadRoot(root, label), {
    ...(families === undefined ? {} : { families }),
    ...(date === undefined ? {} : { date }),
  });
  console.log(
    args.flags.has("json") ? JSON.stringify(report, undefined, 2) : renderMarkdown(report, label),
  );
};

if (import.meta.main) {
  try {
    await run(Deno.args);
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    Deno.exit(1);
  }
}
