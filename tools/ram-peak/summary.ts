/**
 * `results.jsonl` → **中央値の表**（`matrix.ts` の人が読む側）。
 *
 * 集計をランナー本体から分けてあるのは、検収③の読み方を決めている 2 つの規則
 * （digest を「全体 (payload)」の 2 数で出すこと・warm の payload が 0 でない行に印を付けること）
 * を、実 GPU を回さずに門に掛けるためである。
 */

import type { MeasureReport, MeasureState } from "./measure.ts";

const MIB = 1024 * 1024;

/** 中央値（偶数本は中央 2 つの平均・小数第 1 位で丸める）。 */
export const median = (values: readonly number[]): number => {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = sorted.length >> 1;
  const value = sorted.length % 2 === 1
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
  return Math.round(value * 10) / 10;
};

/** 表の 1 行（取得元 1 通りぶんの中央値）。 */
export type SummaryRow = {
  readonly state: MeasureState;
  readonly runs: number;
  readonly vmHwmMiB: number;
  readonly rssMaxMiB: number;
  readonly externalLoadMiB: number;
  readonly externalRunMiB: number;
  readonly loadMs: number;
  readonly runMs: number;
  readonly digestCalls: number;
  readonly digestPayloadCalls: number;
  readonly cachePuts: number;
  readonly cachePutMiB: number;
  /**
   * warm なのに payload 側の digest が残っている（検収③の要注意行）。
   *
   * MUST: 判定は `digest.calls` ではなく **payload 側**で行う — descriptor 2 文書の突合は
   * cold / warm どちらでも必ず掛かる（container-v1 §7 の①）ので、全体の回数で印を付けると
   * 全ての warm 行が要注意になって印の意味が消える。
   */
  readonly flagged: boolean;
};

/** 取得元ごとに中央値の行を作る（1 本も無い取得元は行を作らない）。 */
export const summarize = (
  results: readonly MeasureReport[],
  states: readonly MeasureState[],
): readonly SummaryRow[] => {
  const rows: SummaryRow[] = [];
  for (const state of states) {
    const runs = results.filter((report) => report.state === state);
    if (runs.length === 0) continue;
    const payload = median(runs.map((report) => report.digest.total.payloadCalls));
    rows.push({
      state,
      runs: runs.length,
      // VmHWM を持たない OS では標本化の最大値で代替する（列の意味は「高水位標」のまま）。
      vmHwmMiB: median(runs.map((report) => report.vmHwmMiB ?? report.peaks.total.rssMaxMiB)),
      rssMaxMiB: median(runs.map((report) => report.peaks.total.rssMaxMiB)),
      externalLoadMiB: median(runs.map((report) => report.peaks.load.externalMaxMiB)),
      externalRunMiB: median(runs.map((report) => report.peaks.run.externalMaxMiB)),
      loadMs: median(runs.map((report) => report.loadMs)),
      runMs: median(runs.map((report) => report.runMs)),
      digestCalls: median(runs.map((report) => report.digest.total.calls)),
      digestPayloadCalls: payload,
      cachePuts: median(runs.map((report) => report.cache?.puts ?? 0)),
      cachePutMiB: median(runs.map((report) => (report.cache?.putBytes ?? 0) / MIB)),
      flagged: state === "warm" && payload > 0,
    });
  }
  return rows;
};

/** 表の見出し（`summary.md` の冒頭に載る条件）。 */
export type SummaryContext = {
  readonly label: string;
  readonly source: string;
  readonly repeat: number;
  readonly states: readonly MeasureState[];
  readonly resultsPath: string;
};

const mibText = (bytes: number): string => (bytes / MIB).toFixed(1);

/** 中央値の表（markdown）。`results` が空なら落とす（欠けた表を作らない）。 */
export const renderSummary = (
  results: readonly MeasureReport[],
  context: SummaryContext,
): string => {
  const first = results[0];
  if (first === undefined) throw new Error("ram-peak: 集計する run が 1 本も無い");
  const rows = summarize(results, context.states);
  const estimate = [
    `- 宣言からの見積り: 合計 ${mibText(first.fetch.totalBytes)} MiB ` +
    `/ 最大 part ${mibText(first.fetch.maxPartBytes)} MiB ` +
    `/ 最大 block ${mibText(first.fetch.maxBlockBytes)} MiB`,
    `  （part 0 ${(first.fetch.descriptorPartBytes / 1024).toFixed(1)} KiB ` +
    `/ const ${mibText(first.fetch.constPartBytes)} MiB ` +
    `/ 重み ${mibText(first.fetch.weightPartBytes)} MiB ` +
    `/ 資産 part ${mibText(first.fetch.assetPartBytes)} MiB ` +
    `/ 容器外の資産 ${mibText(first.fetch.manifestAssetBytes)} MiB）`,
  ];
  return [
    `# RAM ピーク — ${context.label}`,
    "",
    `- 配布形: \`${context.source}\` / model \`${first.model ?? "（既定）"}\`` +
    ` / quant \`${first.quant ?? "（既定）"}\``,
    `- 反復: ${context.repeat} 回 × ${context.states.join(" → ")}（1 構成 1 プロセス）`,
    `- 生の記録: \`${context.resultsPath}\`（1 行 1 run）`,
    ...estimate,
    "",
    "| 取得元 | n | VmHWM MiB | rss 最大 MiB | external 最大 load MiB |" +
    " external 最大 run MiB | load ms | run ms | digest 全体 (payload) | cache 書込 本 (MiB) |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ...rows.map((row) =>
      `| ${row.flagged ? "⚠ " : ""}${row.state} | ${row.runs} | ${row.vmHwmMiB}` +
      ` | ${row.rssMaxMiB} | ${row.externalLoadMiB} | ${row.externalRunMiB}` +
      ` | ${row.loadMs} | ${row.runMs} | ${row.digestCalls} (${row.digestPayloadCalls})` +
      ` | ${row.cachePuts} (${row.cachePutMiB}) |`
    ),
    "",
    "凡例:",
    "",
    "- `digest 全体 (payload)`: `crypto.subtle.digest` の回数と、そのうち descriptor 2 文書の" +
    "突合を除いたぶん。descriptor の突合は cold / warm どちらでも必ず掛かる" +
    "（container-v1 §7 の①）ので、検収③が 0 を要求するのは括弧の中だけである。",
    "- `⚠`: warm なのに payload 側の digest が残っている行（検収③の要注意 — 判定は人が読む）。",
    "- 取得層（`@hdae/fetch-cache`）が cold の part に掛ける逐次 sha256 は**純 TS 実装**であり、" +
    "`crypto.subtle.digest` を通らないのでこの計数には現れない。cold / warm の対比は " +
    "`cache 書込` 列（warm は 0 本）と併せて読むこと。",
    "- `持越し scale`（piece 分割で part を跨いで持つ companion scale のバイト数）の欄は" +
    "**runtime の診断に席が無い**ため出していない（`results.jsonl` の `missingFromRuntime`）。",
    "",
  ].join("\n");
};
