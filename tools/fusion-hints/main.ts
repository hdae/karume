/**
 * 融合候補ヒントの列挙（Fusion 半自動発見 1 段目・GPU 不要）。
 *
 *     # 配布形（karume.json のある根 — 既定モデルの全コンポーネント）
 *     deno run -A tools/fusion-hints/main.ts enumerate --source models/karume-anima \
 *         --out outputs/bench/karume-anima/2026-09-03_fusion-hints
 *     # 系列出力（先頭 shard を木から探す）・記号は明示束縛
 *     deno run -A tools/fusion-hints/main.ts enumerate --source outputs/series/gemma4-e2b-decode \
 *         --bind M=1 --bind C=640 --max-window 9
 *
 * 出力は `<out>/candidates.jsonl`（1 行 1 レコード）と `<out>/candidates.md`（上位表）。
 * `--out` を省くと Markdown を標準出力へ書く。
 */

import { bindGraphSymbols, discoverGraphs, readIrGraph } from "./assets.ts";
import {
  aggregate,
  enumerateGraph,
  type GraphReport,
  type SourceReport,
  toJsonl,
  toMarkdown,
} from "./enumerate.ts";

const USAGE =
  "使い方: deno run -A tools/fusion-hints/main.ts enumerate --source <配布形 or 系列ディレクトリ>" +
  " [--out <dir>] [--max-window 9] [--top 10] [--model <名前>] [--bind <記号>=<値>]" +
  " [--default-symbol <値>] [--no-fusion]";

type Options = {
  readonly source: string;
  readonly out?: string;
  readonly model?: string;
  readonly maxWindow: number;
  readonly top: number;
  readonly fused: boolean;
  readonly binds: Readonly<Record<string, number>>;
  readonly defaultSymbol?: number;
};

const parseArgs = (argv: readonly string[]): Options => {
  if (argv[0] !== "enumerate") throw new Error(`未知のサブコマンド '${argv[0] ?? ""}'。${USAGE}`);
  const values = new Map<string, string>();
  const binds: Record<string, number> = {};
  let fused = true;
  for (let at = 1; at < argv.length;) {
    const key = argv[at];
    if (!key.startsWith("--")) throw new Error(`引数 '${key}' がオプションでない。${USAGE}`);
    if (key === "--no-fusion") {
      fused = false;
      at += 1;
      continue;
    }
    const value = argv[at + 1];
    if (value === undefined) throw new Error(`引数 ${key} が対でない。${USAGE}`);
    if (key === "--bind") {
      const [symbol, extent] = value.split("=");
      if (extent === undefined) throw new Error(`--bind は <記号>=<値> の形で渡す（'${value}'）`);
      binds[symbol] = Number(extent);
      at += 2;
      continue;
    }
    values.set(key.slice(2), value);
    at += 2;
  }
  const source = values.get("source");
  if (source === undefined) throw new Error(`--source は必須。${USAGE}`);
  const defaultSymbol = values.get("default-symbol");
  return {
    source,
    out: values.get("out"),
    model: values.get("model"),
    maxWindow: Number(values.get("max-window") ?? "9"),
    top: Number(values.get("top") ?? "10"),
    fused,
    binds,
    defaultSymbol: defaultSymbol === undefined ? undefined : Number(defaultSymbol),
  };
};

const directoryUrl = (path: string): URL =>
  new URL(path.endsWith("/") ? path : `${path}/`, `file://${Deno.cwd()}/`);

const options = parseArgs(Deno.args);
const root = directoryUrl(options.source);
const sources = await discoverGraphs(root, options.model);
if (sources.length === 0) throw new Error(`${options.source} に先頭 shard が 1 本も無い`);

const graphs: GraphReport[] = [];
const skipped: { name: string; reason: string }[] = [];
for (const source of sources) {
  let report: GraphReport;
  try {
    const graph = await readIrGraph(source.url);
    const bound = bindGraphSymbols(graph, options.binds, options.defaultSymbol);
    const candidates = enumerateGraph(graph, bound, {
      maxWindow: options.maxWindow,
      fused: options.fused,
    });
    report = {
      graph: source.name,
      path: source.url.pathname.slice(root.pathname.length),
      nodeCount: candidates.nodeCount,
      symbols: bound.symbols,
      counts: candidates.counts,
      windowCount: candidates.windows.length,
      rows: aggregate(candidates.windows),
    };
  } catch (cause) {
    // 読めない 1 本で全体を止めない。ただし黙って落とさず、理由を報告に残す
    // （IR を載せない資産・束縛の足りないシンボルはここに出る）。
    const reason = cause instanceof Error ? cause.message : String(cause);
    skipped.push({ name: source.name, reason });
    console.error(`[fusion-hints] ${source.name}: ${reason}`);
    continue;
  }
  graphs.push(report);
}

const report: SourceReport = {
  source: options.source,
  maxWindow: options.maxWindow,
  fused: options.fused,
  graphs,
  skipped,
};
const markdown = toMarkdown(report, options.top);
if (options.out === undefined) {
  console.log(markdown);
} else {
  const out = directoryUrl(options.out);
  await Deno.mkdir(out, { recursive: true });
  await Deno.writeTextFile(new URL("candidates.jsonl", out), toJsonl(report));
  await Deno.writeTextFile(new URL("candidates.md", out), markdown);
  console.log(
    `${graphs.length} グラフ / 候補窓 ${
      graphs.reduce((sum, graph) => sum + graph.windowCount, 0)
    } 本 → ${options.out}`,
  );
}
