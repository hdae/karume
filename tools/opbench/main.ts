/**
 * opbench — OP マイクロベンチ基盤の CLI（1 段目 = 静的 census）。
 *
 *     deno run -A tools/opbench/main.ts census \
 *         --source models/karume-gemma4-e2b --out outputs/bench/karume-gemma4-e2b/2026-09-03_op-census
 *     deno run -A tools/opbench/main.ts census --source outputs/series/birefnet-hr-1024 \
 *         --out outputs/bench/birefnet-hr-1024/2026-09-03_op-census
 *     deno run -A tools/opbench/main.ts census --source models/karume-gemma4-e2b \
 *         --scenario long=M:1,C:8192 --out <dir>
 *
 * 1 実行 = 1 資産 = `census.jsonl` 1 本 + `summary.json` 1 本（先例 = tools/ram-peak/measure.ts の
 * 「1 構成 = 1 プロセス」）。GPU も重みバイトも使わない — 読むのは先頭 shard の safetensors
 * ヘッダだけ。
 */

import { readIrGraph, resolveAsset } from "./assets.ts";
import {
  censusComponent,
  type CensusRow,
  type CensusSummary,
  summarizeScenario,
} from "./census.ts";
import { assertBindingKeys, defaultScenarios, parseScenario, type Scenario } from "./scenario.ts";

/** `--key value` の並びを読む（同じキーの繰り返しは全て残す）。 */
const parseArgs = (argv: readonly string[]): ReadonlyMap<string, readonly string[]> => {
  const args = new Map<string, string[]>();
  for (let at = 0; at < argv.length; at += 2) {
    const [key, value] = [argv[at], argv[at + 1]];
    if (!key.startsWith("--") || value === undefined) {
      throw new Error(`引数 ${key} が '--キー 値' の対になっていない`);
    }
    // 値の書き忘れ（`--source --out`）は偶数個で終わると対として通り、`--out` を資産根として
    // 読みに行く NotFound になる。理由が打ち間違いだと分かる位置で落とす（値に `--` 始まりを
    // 取るオプションは 1 つも無い）。
    if (value.startsWith("--")) {
      throw new Error(`引数 ${key} の値が無い（'${value}' はオプション）`);
    }
    const name = key.slice(2);
    args.set(name, [...(args.get(name) ?? []), value]);
  }
  return args;
};

const single = (
  args: ReadonlyMap<string, readonly string[]>,
  name: string,
): string | undefined => {
  const values = args.get(name);
  if (values === undefined) return undefined;
  if (values.length > 1) throw new Error(`--${name} は 1 度しか指定できない`);
  return values[0];
};

/** 相対 path を cwd 基準のディレクトリ URL にする（末尾 `/` を必ず付ける）。 */
const directoryUrl = (path: string): URL =>
  new URL(path.endsWith("/") ? path : `${path}/`, `file://${Deno.cwd()}/`);

const runCensus = async (args: ReadonlyMap<string, readonly string[]>): Promise<void> => {
  const source = single(args, "source");
  if (source === undefined) throw new Error("--source <配布形 or 系列ディレクトリ> は必須");
  const out = single(args, "out");
  if (out === undefined) throw new Error("--out <ディレクトリ> は必須");

  const root = directoryUrl(source);
  const asset = await resolveAsset(
    root,
    single(args, "model"),
    single(args, "quant"),
    single(args, "family"),
  );
  const explicit = args.get("scenario");
  const scenarios: readonly Scenario[] = explicit === undefined
    ? defaultScenarios(asset.family)
    : explicit.map(parseScenario);
  const componentNames = asset.components.map((target) => target.component);
  for (const scenario of scenarios) assertBindingKeys(scenario, componentNames);

  // グラフは先頭 shard のヘッダにしか無いので、コンポーネントごとに 1 度だけ読む。
  const graphs = await Promise.all(
    asset.components.map(async (target) => ({
      target,
      graph: await readIrGraph(target.graphShard),
    })),
  );

  const rows: CensusRow[] = [];
  const summaries = scenarios.map((scenario) => {
    const scoped: CensusRow[] = [];
    const unused: string[] = [];
    const hits: Record<string, number> = {};
    for (const { target, graph } of graphs) {
      const census = censusComponent(graph, target, asset, scenario);
      scoped.push(...census.rows);
      unused.push(...census.unusedBindings.map((sym) => `${target.component}:${sym}`));
      for (const [rule, count] of Object.entries(census.fusionHits)) {
        hits[rule] = (hits[rule] ?? 0) + count;
      }
    }
    rows.push(...scoped);
    return summarizeScenario(scenario, scoped, unused, hits);
  });

  const summary: CensusSummary = {
    generated_at: new Date().toISOString(),
    source,
    family: asset.family,
    model: asset.model,
    quant: asset.quant,
    scenarios: summaries,
  };

  const outDir = directoryUrl(out);
  await Deno.mkdir(outDir, { recursive: true });
  await Deno.writeTextFile(
    new URL("census.jsonl", outDir),
    rows.map((row) => JSON.stringify(row)).join("\n") + "\n",
  );
  await Deno.writeTextFile(
    new URL("summary.json", outDir),
    JSON.stringify(summary, null, 2) + "\n",
  );
  console.log(JSON.stringify({
    source,
    family: asset.family,
    model: asset.model,
    quant: asset.quant,
    components: componentNames.length,
    scenarios: summaries.map((entry) => ({
      scenario: entry.scenario,
      nodes: entry.node_count,
      shapes: entry.weights.length,
      unusedBindings: entry.unused_bindings,
    })),
    out,
  }));
};

const USAGE = `使い方: deno run -A tools/opbench/main.ts census --source <dir> --out <dir>
  --source <dir>       配布形（karume.json あり）か outputs/series の系列ディレクトリ
  --out <dir>          census.jsonl / summary.json の書き出し先
  --model <name>       配布形の model（既定 = manifest の defaultModel）
  --quant <name>       配布形の quant（既定 = defaultQuant）/ 系列は格納 dtype グループ名
  --family <name>      家族名の明示（既定 = manifest の pipeline / ディレクトリ名から推定）
  --scenario <名前>=<記号>:<値>[,…]   記号次元の束縛（繰り返し可・既定は家族ごとの表）`;

if (import.meta.main) {
  const [subcommand, ...rest] = Deno.args;
  if (subcommand !== "census") {
    console.error(USAGE);
    Deno.exit(subcommand === undefined ? 1 : 2);
  }
  await runCensus(parseArgs(rest));
}
