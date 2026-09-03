/**
 * 融合候補の列挙と集計（GPU 不要・純関数）。
 *
 * 判定そのものは持たない — `planFusions` と同じ適格条件を掛ける
 * `enumerateUnfusedWindows`（packages/runtime/src/runtime/fusion.ts）を呼ぶだけで、ここは
 * 「どのグラフをどう束縛して計画するか」と「出てきた窓をどう数えるか」だけを持つ。
 */

import type { IrGraph } from "../../packages/runtime/src/format/ir.ts";
import {
  aliasesInput,
  enumerateUnfusedWindows,
  type ExecStep,
  type FusionCounts,
  type FusionLimits,
  planFusions,
  type UnfusedWindow,
} from "../../packages/runtime/src/runtime/fusion.ts";
import { bindSymbols, countUses, planGraph } from "../../packages/runtime/src/runtime/plan.ts";
import type { BoundShapes } from "./assets.ts";

/**
 * 判定に使う device の上限。**WebGPU core 既定**（128MiB / 65535）を固定で使う — 行ブロック
 * 枚数は候補の本数に効かないので、機の実測値を持ち込まないことで表が機に依らない。
 */
export const DEFAULT_LIMITS: FusionLimits = {
  maxStorageBufferBindingSize: 128 * 1024 * 1024,
  maxComputeWorkgroupsPerDimension: 65535,
};

export type EnumerateOptions = {
  /** 連続窓の最大長。 */
  readonly maxWindow: number;
  /**
   * `false` = **融合を切った**計画に掛ける（既知のヒット数との答え合わせ用）。既定の `true` は
   * 現行計画 = 掴めなかった鎖だけが候補に出る。
   */
  readonly fused?: boolean;
  readonly limits?: FusionLimits;
};

export type GraphCandidates = {
  readonly nodeCount: number;
  /** 現行計画（融合あり）のルール別ヒット数 — 候補表の読み合わせ相手。 */
  readonly counts: FusionCounts;
  readonly windows: readonly UnfusedWindow[];
};

/** グラフ 1 本を計画して候補窓を列挙する。 */
export const enumerateGraph = (
  graph: IrGraph,
  bound: BoundShapes,
  options: EnumerateOptions,
): GraphCandidates => {
  const bindings = bindSymbols(graph, bound.inputShapes);
  const { nodes } = planGraph(graph, bindings, bound.stateShapes);
  const context = {
    useCounts: countUses(graph),
    outputNames: new Set(graph.outputs),
    limits: options.limits ?? DEFAULT_LIMITS,
  };
  const plan = planFusions(nodes, context);
  // 融合を切った計画は「全ノードが素のステップ」— planFusions の別名化判定
  // （`aliasesInput`）だけは共有する（ステップの形を 2 通りに割らない）。
  const steps: readonly ExecStep[] = options.fused === false
    ? nodes.map((node) => ({ kind: "node", plan: node, aliasesInput: aliasesInput(node) }))
    : plan.steps;
  return {
    nodeCount: nodes.length,
    counts: plan.counts,
    windows: enumerateUnfusedWindows(steps, context, options.maxWindow),
  };
};

/** 同じ op 名列の候補をまとめた 1 行。 */
export type CandidateRow = {
  /** 畳む鎖の op 名列。 */
  readonly ops: readonly string[];
  readonly count: number;
  /**
   * そのうち**極大**な出現の本数（同じ先頭ノードから、これより長い窓が受理されなかったもの）。
   * n-gram 列挙は長い鎖の接頭辞も別の候補として出すので、読む順はここが大きい行から。
   */
  readonly maximal: number;
  /** 出現した連続窓の幅（昇順）。`ops.length` より大きい幅は窓内 passthrough を含む。 */
  readonly windowSizes: readonly number[];
  readonly example: { readonly nodeIndex: number; readonly outputName: string };
};

const keyOf = (ops: readonly string[]): string => ops.join(",");

/**
 * 候補窓を op 名列ごとに数える。並びは**極大の多い順** → 本数の多い順 → 鎖の長い順 →
 * op 名列の**コードポイント順**（表の並びを機に依らせない — `localeCompare` は `,` のような
 * 可変重み文字の扱いが既定ロケール依存で、同じ資産で採った表の diff が環境で割れる）。
 * 極大を先頭に置くのは、n-gram 列挙が長い鎖の接頭辞も同じ本数で出すため — 本数順だと
 * 接頭辞が上位を埋めて、読むべき最長の鎖が沈む。
 */
export const aggregate = (windows: readonly UnfusedWindow[]): readonly CandidateRow[] => {
  // 同じ先頭ノードから出た最長の窓幅（極大の判定 = n-gram の接頭辞を数え落とさずに区別する）。
  const longest = new Map<number, number>();
  for (const window of windows) {
    longest.set(window.nodeIndex, Math.max(longest.get(window.nodeIndex) ?? 0, window.windowSize));
  }
  type Bucket = {
    ops: readonly string[];
    count: number;
    maximal: number;
    sizes: Set<number>;
    example: { nodeIndex: number; outputName: string };
  };
  const buckets = new Map<string, Bucket>();
  for (const window of windows) {
    const key = keyOf(window.ops);
    const bucket = buckets.get(key) ?? {
      ops: window.ops,
      count: 0,
      maximal: 0,
      sizes: new Set<number>(),
      example: { nodeIndex: window.nodeIndex, outputName: window.outputName },
    };
    bucket.count += 1;
    if (longest.get(window.nodeIndex) === window.windowSize) bucket.maximal += 1;
    bucket.sizes.add(window.windowSize);
    buckets.set(key, bucket);
  }
  return [...buckets.values()]
    .map((bucket): CandidateRow => ({
      ops: bucket.ops,
      count: bucket.count,
      maximal: bucket.maximal,
      windowSizes: [...bucket.sizes].sort((a, b) => a - b),
      example: bucket.example,
    }))
    .sort((a, b) => {
      const left = keyOf(a.ops);
      const right = keyOf(b.ops);
      return b.maximal - a.maximal || b.count - a.count || b.ops.length - a.ops.length ||
        (left < right ? -1 : left > right ? 1 : 0);
    });
};

/** グラフ 1 本ぶんの報告（jsonl / Markdown の素）。 */
export type GraphReport = {
  readonly graph: string;
  readonly path: string;
  readonly nodeCount: number;
  readonly symbols: Readonly<Record<string, number>>;
  readonly counts: FusionCounts;
  readonly windowCount: number;
  readonly rows: readonly CandidateRow[];
};

export type SourceReport = {
  readonly source: string;
  readonly maxWindow: number;
  readonly fused: boolean;
  readonly graphs: readonly GraphReport[];
  /** 読めなかった / IR を持たないファイル（黙って落とさない）。 */
  readonly skipped: readonly { readonly name: string; readonly reason: string }[];
};

/** 1 行 1 レコードの jsonl（`kind` で graph 行と candidate 行を分ける）。 */
export const toJsonl = (report: SourceReport): string => {
  const lines: string[] = [];
  for (const graph of report.graphs) {
    lines.push(JSON.stringify({
      kind: "graph",
      source: report.source,
      fused: report.fused,
      maxWindow: report.maxWindow,
      graph: graph.graph,
      path: graph.path,
      nodeCount: graph.nodeCount,
      symbols: graph.symbols,
      counts: graph.counts,
      windowCount: graph.windowCount,
    }));
    for (const row of graph.rows) {
      lines.push(JSON.stringify({
        kind: "candidate",
        source: report.source,
        fused: report.fused,
        graph: graph.graph,
        ops: row.ops,
        count: row.count,
        maximal: row.maximal,
        windowSizes: row.windowSizes,
        example: row.example,
      }));
    }
  }
  for (const skip of report.skipped) {
    lines.push(JSON.stringify({ kind: "skipped", source: report.source, ...skip }));
  }
  return `${lines.join("\n")}\n`;
};

const hitSummary = (counts: FusionCounts): string => {
  const hits = Object.entries(counts).filter(([, value]) => value > 0);
  return hits.length === 0 ? "なし" : hits.map(([name, value]) => `${name} ${value}`).join(" / ");
};

/** 候補表（上位 `top` 行 / グラフ）。 */
export const toMarkdown = (report: SourceReport, top: number): string => {
  const out: string[] = [
    `# fusion hints — ${report.source}`,
    "",
    `- 計画: ${report.fused ? "現行（融合あり）" : "融合を切った計画"}`,
    `- 窓の最大長: ${report.maxWindow}`,
    "- 候補であって設計ではない（共通の適格条件を通しただけ・ADR 0040 決定 2 の受理集合は" +
    "広げていない）",
    "",
  ];
  for (const graph of report.graphs) {
    const symbols = Object.entries(graph.symbols).map(([name, value]) => `${name}=${value}`);
    out.push(
      `## ${graph.graph}`,
      "",
      `- ノード ${graph.nodeCount} / 束縛 ${symbols.length === 0 ? "なし" : symbols.join(" ")}`,
      `- 現行計画のヒット: ${hitSummary(graph.counts)}`,
      `- 候補窓 ${graph.windowCount} 本 / 相異なる op 名列 ${graph.rows.length} 種`,
      "",
    );
    if (graph.rows.length === 0) {
      out.push("候補なし。", "");
      continue;
    }
    out.push(
      "| 鎖の op 名列 | 本数 | うち極大 | 窓幅 | 例（node / 出力） |",
      "| --- | ---: | ---: | --- | --- |",
    );
    for (const row of graph.rows.slice(0, top)) {
      out.push(
        `| \`${keyOf(row.ops)}\` | ${row.count} | ${row.maximal} | ${
          row.windowSizes.join(",")
        } | ${row.example.nodeIndex} / \`${row.example.outputName}\` |`,
      );
    }
    if (graph.rows.length > top) {
      out.push(`| …（残り ${graph.rows.length - top} 種は jsonl） | | | | |`);
    }
    out.push("");
  }
  if (report.skipped.length > 0) {
    out.push("## 読めなかったファイル", "");
    for (const skip of report.skipped) out.push(`- \`${skip.name}\`: ${skip.reason}`);
    out.push("");
  }
  return out.join("\n");
};
