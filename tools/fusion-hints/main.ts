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

/**
 * 数値引数は正の安全整数だけ受ける（打ち間違いを fail loudly）。
 *
 * 素の `Number` で受けると壊れ方が黙る: `--top abc` は NaN → `slice(0, NaN)` で候補表が 0 行の
 * まま「候補なし」とも書かれない Markdown が出るし、`--bind M=-1` は負の次元で計画が進む
 * （`Number.isSafeInteger(-1)` は真なので下流の安全整数検査も通り抜ける）。
 */
const positiveInteger = (raw: string, where: string): number => {
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${where} の値 '${raw}' が正の整数でない。${USAGE}`);
  }
  return value;
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
    // 値の書き忘れ（`--source --out dir`）を「次のオプション名を値として受ける」形で飲まない。
    // 値に `--` 始まりを取るオプションは 1 つも無い。
    if (value.startsWith("--")) {
      throw new Error(`引数 ${key} の値が無い（'${value}' はオプション）。${USAGE}`);
    }
    if (key === "--bind") {
      // 最初の `=` で割る（`--bind M=1=2` のような綴りを黙って捨てない）。
      const split = value.indexOf("=");
      if (split <= 0) throw new Error(`--bind は <記号>=<値> の形で渡す（'${value}'）`);
      const symbol = value.slice(0, split);
      binds[symbol] = positiveInteger(value.slice(split + 1), `--bind ${symbol}`);
      at += 2;
      continue;
    }
    values.set(key.slice(2), value);
    at += 2;
  }
  const source = values.get("source");
  if (source === undefined) throw new Error(`--source は必須。${USAGE}`);
  const maxWindow = values.get("max-window");
  const top = values.get("top");
  const defaultSymbol = values.get("default-symbol");
  return {
    source,
    out: values.get("out"),
    model: values.get("model"),
    maxWindow: maxWindow === undefined ? 9 : positiveInteger(maxWindow, "--max-window"),
    top: top === undefined ? 10 : positiveInteger(top, "--top"),
    fused,
    binds,
    defaultSymbol: defaultSymbol === undefined
      ? undefined
      : positiveInteger(defaultSymbol, "--default-symbol"),
  };
};

const directoryUrl = (path: string): URL =>
  new URL(path.endsWith("/") ? path : `${path}/`, `file://${Deno.cwd()}/`);

// MUST: CLI の本体は `import.meta.main` の内側だけで走らせる（横断の不変条件「全モジュール
// 副作用ゼロ = import 時実行の禁止」— 型や関数をここから import した瞬間に引数解析と
// ディレクトリ書き出しが走るのを防ぐ）。
if (import.meta.main) {
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
}
