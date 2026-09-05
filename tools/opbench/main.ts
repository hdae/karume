/**
 * opbench — OP マイクロベンチ基盤の CLI（1 段目 = 静的 census）。
 *
 *     deno run -A tools/opbench/main.ts census \
 *         --source models/karume-gemma4 --out outputs/bench/karume-gemma4/2026-09-03_op-census
 *     deno run -A tools/opbench/main.ts census --source outputs/series/birefnet-hr-1024 \
 *         --out outputs/bench/birefnet-hr-1024/2026-09-03_op-census
 *     deno run -A tools/opbench/main.ts census --source models/karume-gemma4 \
 *         --scenario long=M:1,C:8192 --out <dir>
 *
 * 1 実行 = 1 資産 = `census.jsonl` 1 本 + `summary.json` 1 本（先例 = tools/ram-peak/measure.ts の
 * 「1 構成 = 1 プロセス」）。GPU も重みバイトも使わない — 読むのは先頭 shard の safetensors
 * ヘッダだけ。
 */

import { directoryUrl, externalPath, readIrGraph, resolveAsset } from "../_shared/assets.ts";
import { acquireGpu } from "../../packages/runtime/mod.ts";
import { ROUNDS } from "./bench.ts";
import {
  buildCensusSummary,
  censusComponent,
  type CensusRow,
  type CensusSummary,
  summarizeScenario,
} from "./census.ts";
import {
  buildSingleSummary,
  type CaseFailure,
  createHeater,
  measureCase,
  type RunSingleOptions,
  selectCases,
  sessionOptionsOf,
  type SingleRecord,
  type SingleSummary,
} from "./single.ts";
import { compareWithCensus, driveOnce, type RunRecord } from "./graph.ts";
import { defaultVenv, runTorchBench, summarizeTorch, type TorchRecord } from "./torch.ts";
import {
  assertBindingKeys,
  assertPlainBindingKeys,
  defaultScenarios,
  parseScenarios,
  type Scenario,
} from "../_shared/scenario.ts";

/** `--key value` の並びを読む（同じキーの繰り返しは全て残す。門は main_test.ts が見る）。 */
export const parseArgs = (
  argv: readonly string[],
  known: ReadonlySet<string>,
): ReadonlyMap<string, readonly string[]> => {
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
  // 未知のオプションを黙って捨てない（`--scenarios` の打ち間違いが既定シナリオへ静かに落ちる）。
  for (const name of args.keys()) {
    if (!known.has(name)) throw new Error(`未知のオプション --${name}。\n${USAGE}`);
  }
  return args;
};

export const CENSUS_OPTIONS: ReadonlySet<string> = new Set([
  "source",
  "out",
  "model",
  "quant",
  "family",
  "scenario",
]);
const GRAPH_OPTIONS: ReadonlySet<string> = new Set([
  "source",
  "family",
  "out",
  "mode",
  "census",
  "scenario",
  "runs",
  "single",
  "model",
  "quant",
  "new-tokens",
  "steps",
  "size",
  "prompt",
]);
const TORCH_OPTIONS: ReadonlySet<string> = new Set([
  "single",
  "out",
  "venv",
  "rounds",
  "compile",
  "limit",
  "op",
]);
const SINGLE_OPTIONS: ReadonlySet<string> = new Set([
  "census",
  "out",
  "mode",
  "op",
  "scenario",
  "component",
  "limit",
  "session",
  "rounds",
]);

const positiveInteger = (text: string, where: string): number => {
  const value = Number(text);
  if (!Number.isInteger(value) || value <= 0) throw new Error(`${where} は正の整数（'${text}'）`);
  return value;
};

export const single = (
  args: ReadonlyMap<string, readonly string[]>,
  name: string,
): string | undefined => {
  const values = args.get(name);
  if (values === undefined) return undefined;
  if (values.length > 1) throw new Error(`--${name} は 1 度しか指定できない`);
  return values[0];
};

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
    : parseScenarios(explicit);
  const componentNames = asset.components.map((target) => target.component);
  for (const scenario of scenarios) assertBindingKeys(scenario, componentNames);

  // グラフは先頭 shard のヘッダにしか無いので、コンポーネントごとに 1 度だけ読む。
  const graphs = await Promise.all(
    asset.components.map(async (target) => ({
      target,
      graph: await readIrGraph(target.graphShard),
    })),
  );
  // 修飾なしキーの誤綴りはグラフを読むまで判らない（記号の宣言集合は IR にしか無い）。
  const irs = graphs.map(({ graph }) => graph);
  for (const scenario of scenarios) assertPlainBindingKeys(scenario, irs);

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

  const summary = buildCensusSummary(source, asset, summaries);

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

const USAGE = `使い方: deno run -A tools/opbench/main.ts <census|single> …
  census --source <dir> --out <dir>
    --source <dir>       配布形（karume.json あり）か outputs/series の系列ディレクトリ
    --out <dir>          census.jsonl / summary.json の書き出し先
    --model <name>       配布形の model（既定 = manifest の defaultModel）
    --quant <name>       配布形の quant（既定 = defaultQuant）/ 系列は格納 dtype グループ名
    --family <name>      家族名の明示（既定 = manifest の pipeline / ディレクトリ名から推定）
    --scenario <名前>=<記号>:<値>[,…]   記号次元の束縛（繰り返し可・既定は家族ごとの表）
  single --census <dir> --out <dir>
    --census <dir>       census の出力ディレクトリ（summary.json を読む）
    --out <dir>          single.jsonl / summary.json の書き出し先
    --mode timing|wall   計測モード（既定 = timestamp-query があれば timing、無ければ wall）
    --op <name>          この op だけ（繰り返し可）
    --scenario <name>    このシナリオだけ
    --component <name>   このコンポーネントだけ
    --limit <n>          加重（count × 出力要素）の降順で先頭 n 件だけ
    --session <knob>=<value>   実行変種の上書き（linearCompute / attentionCompute / attentionScoreStorage・繰り返し可）
    --rounds <n>         代表値（min）を採る反復回数（既定 ${ROUNDS}）
  graph --source <dir> --family <gemma4|anima> --out <dir> [--census <dir> --scenario <name>]
    --source <dir>       配布形（karume.json あり）— pipeline の fromPretrained で読む
    --family <name>      gemma4（chat 1 ターン）か anima（1 枚）
    --out <dir>          graph.jsonl（run ごと）/ summary.json の書き出し先
    --mode timing|wall   timing = op 別 GPU 時間（既定・timestamp-query が要る）/ wall = 計測無効で壁だけ
    --census <dir>       突合する census の出力（--scenario と組で・省略時は突合しない）
    --scenario <name>    census のシナリオ（gemma4 = decode / prefill・anima = 1024px）
    --runs <prefix>      突合に使う run の label 接頭辞（既定 gemma4 = decode / anima = transformer）
    --single <dir>       single の出力（op 別の single / graph 比を出す）
    --model / --quant    配布形の選択（既定 = manifest）
    --new-tokens <n>     gemma4 の生成 token 数（既定 8）/ --steps <n> --size <px> anima の step と辺（既定 2 / 1024・step は 2 以上）
    --prompt <text>      入力文（既定あり）
  torch --single <dir> --out <dir> [--venv <dir>] [--compile true] [--rounds <n>] [--limit <n>] [--op <name>]
    --single <dir>       single の出力（single.jsonl の各行を torch eager で組んで測る）
    --out <dir>          torch.jsonl / summary.json / comparison.json の書き出し先
    --venv <dir>         CUDA venv（既定 = KARUME_CUDA_VENV か ~/workspace/karume-cuda-venv）
    --compile true       torch.compile（Inductor）の f16 列も測る
    --rounds / --limit / --op   single と同じ意味`;

const timestampQueryAvailable = async (): Promise<boolean> => {
  const gpu: GPU | undefined = navigator.gpu;
  if (gpu === undefined) return false;
  const adapter = await gpu.requestAdapter();
  return adapter !== null && adapter.features.has("timestamp-query");
};

/** summary.json を読む（unknown 境界 — 欄の有無だけ見て fail loudly）。 */
const readCensusSummary = async (dir: URL): Promise<CensusSummary> => {
  const path = new URL("summary.json", dir);
  const parsed: unknown = JSON.parse(await Deno.readTextFile(path));
  if (
    typeof parsed !== "object" || parsed === null || !("scenarios" in parsed) ||
    !Array.isArray((parsed as { scenarios: unknown }).scenarios) || !("session" in parsed)
  ) {
    throw new Error(
      `${path.pathname}: opbench census の summary.json でない（scenarios / session が無い）`,
    );
  }
  return parsed as CensusSummary;
};

const runSingle = async (args: ReadonlyMap<string, readonly string[]>): Promise<void> => {
  const census = single(args, "census");
  if (census === undefined) throw new Error("--census <census の出力ディレクトリ> は必須");
  const out = single(args, "out");
  if (out === undefined) throw new Error("--out <ディレクトリ> は必須");
  const modeArg = single(args, "mode");
  if (modeArg !== undefined && modeArg !== "timing" && modeArg !== "wall") {
    throw new Error(`--mode は timing か wall（'${modeArg}'）`);
  }
  const mode: "timing" | "wall" = modeArg ??
    ((await timestampQueryAvailable()) ? "timing" : "wall");
  const limit = single(args, "limit");
  const rounds = single(args, "rounds");
  const ops = args.get("op");
  const overrides: Record<string, string> = {};
  for (const text of args.get("session") ?? []) {
    const split = text.indexOf("=");
    if (split <= 0) throw new Error(`--session は <knob>=<value> の形（'${text}'）`);
    overrides[text.slice(0, split)] = text.slice(split + 1);
  }

  const summary = await readCensusSummary(directoryUrl(census));
  const selection = selectCases(summary, {
    ...(ops === undefined ? {} : { ops: new Set(ops) }),
    ...(single(args, "scenario") === undefined ? {} : { scenario: single(args, "scenario") }),
    ...(single(args, "component") === undefined ? {} : { component: single(args, "component") }),
    ...(limit === undefined ? {} : { limit: positiveInteger(limit, "--limit") }),
  });
  const sessionOptions = sessionOptionsOf(summary.session, overrides);

  // MUST: timing は device 作成時にしか要求できない（1 構成 = 1 プロセス — モードごとに起動する）。
  const gpu = await acquireGpu({ gpuTiming: mode === "timing" });
  const heater = await createHeater(gpu);
  try {
    const options: RunSingleOptions = {
      gpu,
      session: sessionOptions,
      mode,
      heater,
      ...(rounds === undefined ? {} : { rounds: positiveInteger(rounds, "--rounds") }),
    };
    const records: SingleRecord[] = [];
    const failed: CaseFailure[] = [];
    for (const [index, testCase] of selection.cases.entries()) {
      const label = `[${
        index + 1
      }/${selection.cases.length}] ${testCase.row.component} ${testCase.row.op} ${
        JSON.stringify(testCase.row.in_shapes)
      }`;
      try {
        const record = await measureCase(testCase, options);
        records.push(record);
        console.log(
          `${label} reps=${record.reps} ${
            record.mode === "timing"
              ? `${
                record.ns_per_node_min?.toFixed(0)
              } ns/node (${record.dispatches_per_node} dispatch) × ${record.count} = ${
                record.weighted_ms?.toFixed(3)
              } ms [${record.keys.join(" ")}]`
              : `${record.wall_ms_per_rep_min?.toFixed(4)} ms/rep`
          }`,
        );
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        failed.push({
          scenario: testCase.scenario,
          component: testCase.row.component,
          op: testCase.row.op,
          in_shapes: testCase.row.in_shapes,
          message,
        });
        console.error(`${label} FAILED: ${message}`);
      }
    }
    const outDir = directoryUrl(out);
    await Deno.mkdir(outDir, { recursive: true });
    await Deno.writeTextFile(
      new URL("single.jsonl", outDir),
      records.map((record) => JSON.stringify(record)).join("\n") + "\n",
    );
    const singleSummary = buildSingleSummary(
      census,
      summary,
      gpu,
      options,
      records,
      selection.excluded,
      failed,
    );
    await Deno.writeTextFile(
      new URL("summary.json", outDir),
      JSON.stringify(singleSummary, null, 2) + "\n",
    );
    console.log(JSON.stringify({
      census,
      mode,
      measured: records.length,
      failed: failed.length,
      excluded: selection.excluded,
      weighted_ms_by_op_storage: singleSummary.weighted_ms_by_op_storage,
      out,
    }));
  } finally {
    await heater.dispose();
    gpu.destroy();
  }
};

const readSingleSummary = async (dir: URL): Promise<SingleSummary> => {
  const path = new URL("summary.json", dir);
  const parsed: unknown = JSON.parse(await Deno.readTextFile(path));
  if (
    typeof parsed !== "object" || parsed === null || !("weighted_ms_by_op_storage" in parsed)
  ) {
    throw new Error(`${path.pathname}: opbench single の summary.json でない`);
  }
  return parsed as SingleSummary;
};

const runGraph = async (args: ReadonlyMap<string, readonly string[]>): Promise<void> => {
  const source = single(args, "source");
  if (source === undefined) throw new Error("--source <配布形ディレクトリ> は必須");
  const family = single(args, "family");
  if (family !== "gemma4" && family !== "anima") {
    throw new Error(`--family は gemma4 か anima（'${family}'）— 他家族は未対応`);
  }
  const out = single(args, "out");
  if (out === undefined) throw new Error("--out <ディレクトリ> は必須");
  const modeArg = single(args, "mode");
  if (modeArg !== undefined && modeArg !== "timing" && modeArg !== "wall") {
    throw new Error(`--mode は timing か wall（'${modeArg}'）`);
  }
  const mode: "timing" | "wall" = modeArg ?? "timing";
  const censusDir = single(args, "census");
  const scenario = single(args, "scenario");
  if ((censusDir === undefined) !== (scenario === undefined)) {
    throw new Error("--census と --scenario は組で渡す");
  }
  const runsPrefix = single(args, "runs") ?? (family === "gemma4" ? "decode" : "transformer");
  const singleDir = single(args, "single");
  const newTokens = single(args, "new-tokens");
  const steps = single(args, "steps");
  const size = single(args, "size");

  const gpu = await acquireGpu({ gpuTiming: mode === "timing" });
  let result: Awaited<ReturnType<typeof driveOnce>>;
  try {
    result = await driveOnce({
      gpu,
      source,
      family,
      ...(single(args, "model") === undefined ? {} : { model: single(args, "model") }),
      ...(single(args, "quant") === undefined ? {} : { quant: single(args, "quant") }),
      ...(newTokens === undefined ? {} : { newTokens: positiveInteger(newTokens, "--new-tokens") }),
      ...(steps === undefined ? {} : { steps: positiveInteger(steps, "--steps") }),
      ...(size === undefined ? {} : { size: positiveInteger(size, "--size") }),
      ...(single(args, "prompt") === undefined ? {} : { prompt: single(args, "prompt") }),
    });
  } finally {
    gpu.destroy();
  }
  const compared: RunRecord[] = result.records.filter((record) =>
    record.label.startsWith(runsPrefix)
  );
  const comparison = censusDir === undefined || scenario === undefined ? null : compareWithCensus(
    compared,
    await readCensusSummary(directoryUrl(censusDir)),
    scenario,
    singleDir === undefined ? undefined : await readSingleSummary(directoryUrl(singleDir)),
  );
  const outDir = directoryUrl(out);
  await Deno.mkdir(outDir, { recursive: true });
  await Deno.writeTextFile(
    new URL("graph.jsonl", outDir),
    result.records.map((record) => JSON.stringify(record)).join("\n") + "\n",
  );
  const summary = {
    generated_at: new Date().toISOString(),
    source,
    family,
    mode,
    rig: { ...gpu.adapterInfo, deno: Deno.version.deno },
    runs: result.records.length,
    compared_runs: compared.length,
    runs_prefix: runsPrefix,
    load_ms: result.load_ms,
    wall_ms: result.wall_ms,
    gpu_ms_total: result.records.reduce((total, record) => total + (record.total_ns ?? 0), 0) / 1e6,
    dispatch_total: result.records.reduce((total, record) => total + record.dispatch_count, 0),
    comparison,
  };
  await Deno.writeTextFile(
    new URL("summary.json", outDir),
    JSON.stringify(summary, null, 2) + "\n",
  );
  console.log(JSON.stringify({
    family,
    mode,
    runs: summary.runs,
    compared_runs: summary.compared_runs,
    wall_ms: Math.round(summary.wall_ms),
    gpu_ms_total: Math.round(summary.gpu_ms_total * 100) / 100,
    dispatch_total: summary.dispatch_total,
    unmapped_keys: comparison?.unmapped_keys ?? null,
    top: comparison?.rows.slice(0, 8).map((row) => ({
      op: row.op,
      census: row.census_nodes,
      dispatches: row.measured_dispatches,
      ms: Math.round(row.measured_ms * 1000) / 1000,
      single_over_graph: row.single_over_graph === null
        ? null
        : Math.round(row.single_over_graph * 100) / 100,
    })) ?? null,
    out,
  }));
};

/** torch の summary.json の `errors`（列名 → 失敗件数と代表文言）。 */
type ColumnErrors = Readonly<Record<string, { readonly count: number; readonly example: string }>>;

const isColumnErrors = (value: unknown): value is ColumnErrors =>
  typeof value === "object" && value !== null &&
  Object.values(value as Record<string, unknown>).every((entry) =>
    typeof entry === "object" && entry !== null &&
    typeof (entry as { count?: unknown }).count === "number" &&
    typeof (entry as { example?: unknown }).example === "string"
  );

const runTorch = async (args: ReadonlyMap<string, readonly string[]>): Promise<void> => {
  const singleDir = single(args, "single");
  if (singleDir === undefined) throw new Error("--single <single の出力ディレクトリ> は必須");
  const out = single(args, "out");
  if (out === undefined) throw new Error("--out <ディレクトリ> は必須");
  const compileArg = single(args, "compile");
  if (compileArg !== undefined && compileArg !== "true" && compileArg !== "false") {
    throw new Error(`--compile は true か false（'${compileArg}'）`);
  }
  const rounds = single(args, "rounds");
  const limit = single(args, "limit");
  const options = {
    venv: single(args, "venv") ?? defaultVenv(),
    // python へ渡す 2 本は URL を経由しない（percent encode が子の実 path へ漏れる — 綴りの
    // 理由は externalPath の doc）。Deno 側の読み書きは従来どおり URL で行う。
    single: `${externalPath(singleDir)}/single.jsonl`,
    out: externalPath(out),
    compile: compileArg === "true",
    ...(rounds === undefined ? {} : { rounds: positiveInteger(rounds, "--rounds") }),
    ...(limit === undefined ? {} : { limit: positiveInteger(limit, "--limit") }),
    ...(args.get("op") === undefined ? {} : { ops: args.get("op") }),
  };
  await runTorchBench(options);
  const outDir = directoryUrl(out);
  const records: TorchRecord[] = (await Deno.readTextFile(new URL("torch.jsonl", outDir)))
    .split("\n").filter((line) => line.length > 0).map((line) => JSON.parse(line) as TorchRecord);
  const torchSummary: unknown = JSON.parse(
    await Deno.readTextFile(new URL("summary.json", outDir)),
  );
  const columns = (torchSummary as { columns?: unknown }).columns;
  if (!Array.isArray(columns) || !columns.every((column) => typeof column === "string")) {
    throw new Error("torch の summary.json に columns が無い");
  }
  // 列ごとの失敗件数を突合表の頭へ写す。これが無いと、ある列が全 case 失敗しても
  // median_ratio=null / weighted_ms=0 としか出ず「測れなかった」と「速かった」が区別できない。
  const columnErrors = (torchSummary as { errors?: unknown }).errors;
  if (!isColumnErrors(columnErrors)) {
    throw new Error("torch の summary.json に errors（列名 → 件数と代表文言）が無い");
  }
  const comparison = summarizeTorch(records, columns);
  await Deno.writeTextFile(
    new URL("comparison.json", outDir),
    JSON.stringify(
      {
        generated_at: new Date().toISOString(),
        single: singleDir,
        columns,
        column_errors: columnErrors,
        ops: comparison,
      },
      null,
      2,
    ) + "\n",
  );
  console.log(JSON.stringify({
    single: singleDir,
    measured: records.length,
    columns,
    top: comparison.slice(0, 10).map((row) => ({
      op: row.op,
      cases: row.cases,
      karume_ms: Math.round(row.weighted_ms.karume * 1000) / 1000,
      ...Object.fromEntries(columns.map((column) => [
        column,
        {
          ms: Math.round((row.weighted_ms[column] ?? 0) * 1000) / 1000,
          ratio: row.median_ratio[column] === null
            ? null
            : Math.round((row.median_ratio[column] ?? 0) * 100) / 100,
        },
      ])),
    })),
    out,
  }));
};

if (import.meta.main) {
  const [subcommand, ...rest] = Deno.args;
  if (subcommand === "torch") {
    await runTorch(parseArgs(rest, TORCH_OPTIONS));
  } else if (subcommand === "census") {
    await runCensus(parseArgs(rest, CENSUS_OPTIONS));
  } else if (subcommand === "single") {
    await runSingle(parseArgs(rest, SINGLE_OPTIONS));
  } else if (subcommand === "graph") {
    await runGraph(parseArgs(rest, GRAPH_OPTIONS));
  } else {
    console.error(USAGE);
    Deno.exit(subcommand === undefined ? 1 : 2);
  }
}
