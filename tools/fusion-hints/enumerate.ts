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
import { CORE_LIMITS } from "../_shared/assets.ts";
import { resolveComponentBindings, type Scenario } from "../_shared/scenario.ts";
import { bindGraphSymbols, type BoundShapes } from "./binding.ts";

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
  readonly node_count: number;
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
    limits: options.limits ?? CORE_LIMITS,
  };
  const plan = planFusions(nodes, context);
  // 融合を切った計画は「全ノードが素のステップ」— planFusions の別名化判定
  // （`aliasesInput`）だけは共有する（ステップの形を 2 通りに割らない）。
  const steps: readonly ExecStep[] = options.fused === false
    ? nodes.map((node) => ({ kind: "node", plan: node, aliasesInput: aliasesInput(node) }))
    : plan.steps;
  return {
    node_count: nodes.length,
    counts: plan.counts,
    windows: enumerateUnfusedWindows(steps, context, options.maxWindow),
  };
};

/**
 * 同じ op 名列の候補をまとめた 1 行。
 *
 * 欄の綴りは jsonl へそのまま出るので**全て snake_case**（census の行と混ぜて読むため — 未公開
 * ツールなので旧綴りの互換シムは置かない）。
 */
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
  readonly window_sizes: readonly number[];
  readonly example: { readonly node_index: number; readonly output_name: string };
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
    example: { node_index: number; output_name: string };
  };
  const buckets = new Map<string, Bucket>();
  for (const window of windows) {
    const key = keyOf(window.ops);
    const bucket = buckets.get(key) ?? {
      ops: window.ops,
      count: 0,
      maximal: 0,
      sizes: new Set<number>(),
      example: { node_index: window.nodeIndex, output_name: window.outputName },
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
      window_sizes: [...bucket.sizes].sort((a, b) => a - b),
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
  /** 表の見出し（`<model>/<component>`）。 */
  readonly graph: string;
  /** opbench の census 行と同じ component 綴り。 */
  readonly component: string;
  readonly path: string;
  readonly node_count: number;
  /** このグラフの記号に実際に入った値（`bindings` を `graph.symbols` へ絞ったもの）。 */
  readonly symbols: Readonly<Record<string, number>>;
  readonly counts: FusionCounts;
  readonly window_count: number;
  readonly rows: readonly CandidateRow[];
};

/** 列挙にかける 1 コンポーネント（IR は先頭 shard のヘッダから読んだもの）。 */
export type GraphInput = {
  readonly component: string;
  readonly graph: string;
  readonly path: string;
  readonly ir: IrGraph;
};

/**
 * シナリオ 1 本ぶんの報告。欄の綴り（`scenario` / `bindings` / `binding_source` / `provenance`）は
 * opbench の `summary.json` の `scenarios[]` と同じ — 同じ資産の census と候補表を機械で
 * 突き合わせるのはこの 4 欄。
 */
export type ScenarioReport = {
  readonly scenario: string;
  /** シナリオが宣言した束縛そのもの（`<component>.SYM` 込み）。 */
  readonly bindings: Readonly<Record<string, number>>;
  readonly binding_source: Scenario["source"];
  readonly provenance: string;
  readonly graphs: readonly GraphReport[];
};

export type SourceReport = {
  readonly source: string;
  readonly family: string;
  readonly model: string;
  readonly quant: string;
  readonly max_window: number;
  readonly fused: boolean;
  readonly scenarios: readonly ScenarioReport[];
  /** 読めなかった / IR を持たないファイル（黙って落とさない）。 */
  readonly skipped: readonly { readonly name: string; readonly reason: string }[];
};

/**
 * シナリオ 1 本を全コンポーネントに掛ける。
 *
 * 束縛は `resolveComponentBindings` が `<component>.SYM` を解いたものだけを渡す — 同じ記号名が
 * コンポーネントごとに別の意味を持つ資産（irodori の `T` = backbone のトークン数 /
 * codec_encoder のフレーム数）で、家族共通の値が混ざらないようにする。誤綴りと未束縛を
 * 打ち切るのもその中（判定は census と同じ 1 箇所）。
 */
export const reportScenario = (
  scenario: Scenario,
  graphs: readonly GraphInput[],
  options: EnumerateOptions,
): ScenarioReport => ({
  scenario: scenario.name,
  bindings: scenario.bindings,
  binding_source: scenario.source,
  provenance: scenario.provenance,
  graphs: graphs.map((entry): GraphReport => {
    const { symbols } = resolveComponentBindings(scenario, entry.component, entry.ir);
    const bound = bindGraphSymbols(entry.ir, symbols);
    const candidates = enumerateGraph(entry.ir, bound, options);
    return {
      graph: entry.graph,
      component: entry.component,
      path: entry.path,
      node_count: candidates.node_count,
      symbols: bound.symbols,
      counts: candidates.counts,
      window_count: candidates.windows.length,
      rows: aggregate(candidates.windows),
    };
  }),
});

/** 1 行 1 レコードの jsonl（`kind` で graph 行と candidate 行を分ける）。 */
export const toJsonl = (report: SourceReport): string => {
  const lines: string[] = [];
  const identity = {
    source: report.source,
    family: report.family,
    model: report.model,
    quant: report.quant,
    fused: report.fused,
  };
  for (const scenario of report.scenarios) {
    for (const graph of scenario.graphs) {
      lines.push(JSON.stringify({
        kind: "graph",
        ...identity,
        max_window: report.max_window,
        scenario: scenario.scenario,
        bindings: scenario.bindings,
        binding_source: scenario.binding_source,
        graph: graph.graph,
        component: graph.component,
        path: graph.path,
        node_count: graph.node_count,
        symbols: graph.symbols,
        counts: graph.counts,
        window_count: graph.window_count,
      }));
      for (const row of graph.rows) {
        lines.push(JSON.stringify({
          kind: "candidate",
          ...identity,
          scenario: scenario.scenario,
          graph: graph.graph,
          component: graph.component,
          ops: row.ops,
          count: row.count,
          maximal: row.maximal,
          window_sizes: row.window_sizes,
          example: row.example,
        }));
      }
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
    `- 資産: ${report.family} / model ${report.model} / quant ${report.quant}`,
    `- 計画: ${report.fused ? "現行（融合あり）" : "融合を切った計画"}`,
    `- 窓の最大長: ${report.max_window}`,
    "- 候補であって設計ではない（共通の適格条件を通しただけ・ADR 0040 決定 2 の受理集合は" +
    "広げていない）",
    "",
  ];
  for (const scenario of report.scenarios) {
    const bindings = Object.entries(scenario.bindings).map(([name, value]) => `${name}=${value}`);
    out.push(
      `## シナリオ ${scenario.scenario}（束縛 ${
        bindings.length === 0 ? "なし" : bindings.join(" ")
      }）`,
      "",
      `- 束縛の出どころ: ${scenario.binding_source} — ${scenario.provenance}`,
      "",
    );
    for (const graph of scenario.graphs) {
      const symbols = Object.entries(graph.symbols).map(([name, value]) => `${name}=${value}`);
      out.push(
        `### ${graph.graph}`,
        "",
        `- ノード ${graph.node_count} / 束縛 ${symbols.length === 0 ? "なし" : symbols.join(" ")}`,
        `- 現行計画のヒット: ${hitSummary(graph.counts)}`,
        `- 候補窓 ${graph.window_count} 本 / 相異なる op 名列 ${graph.rows.length} 種`,
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
            row.window_sizes.join(",")
          } | ${row.example.node_index} / \`${row.example.output_name}\` |`,
        );
      }
      if (graph.rows.length > top) {
        out.push(`| …（残り ${graph.rows.length - top} 種は jsonl） | | | | |`);
      }
      out.push("");
    }
  }
  if (report.skipped.length > 0) {
    out.push("## 読めなかったファイル", "");
    for (const skip of report.skipped) out.push(`- \`${skip.name}\`: ${skip.reason}`);
    out.push("");
  }
  return out.join("\n");
};
