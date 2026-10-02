/**
 * Anima のタブの「A/B（既定 vs 表）」の要約（区間 A = `default` を注入・区間 B = 適用中の幾何プロファイル）。
 *
 * 純関数だけを置く（DOM は呼び手 — `anima-tab.ts` が描画する）。見たいのは 2 点:
 *
 * 1. 表は幾何（タイルの担当割り）だけを変えるので PNG のビットは動かない（ADR 0115）— 区間の中・区間の間で
 *    sha256 が一致するか。
 * 2. 段ごとの時間がどう変わったか — 常駐 DiT は区間ごとに GPU を取り直して組み直すので、各区間の 1 回目は
 *    DiT の読み込みを含む。比べるのは 2 回目以降の中央値（N = 1 なら 1 回目しか無いのでそれ）。
 */
import type { AnimaRunComponent } from "../../../packages/models/anima.ts";
import type { Row } from "../../anima-residency/record.ts";

/** 要約に並べる段（pipeline の run の順）。 */
export const AB_STAGES: readonly AnimaRunComponent[] = [
  "text_encoder",
  "text_conditioner",
  "transformer",
  "vae_decoder",
];

/** 区間の中の PNG sha256 の一致（`distinct` = 出た sha の先頭 12 桁・重複を除き最初に出た順）。 */
export type ShaAgreement = { readonly match: boolean; readonly distinct: readonly string[] };

export type IntervalSummary =
  | { readonly kind: "ok"; readonly count: number; readonly sha: ShaAgreement }
  /** 区間の中に失敗行がある（バッチは失敗行で止まるので、最初の失敗行の `error.name`）。 */
  | { readonly kind: "failed"; readonly errorName: string }
  /** 回していない（区間 A が失敗したら区間 B は回さない）。 */
  | { readonly kind: "skipped" };

/** 両区間の代表値（区間ごとの中央値）と B ÷ A。値が無ければ欄が無い（表示は「—」）。 */
export type AbMetric = { readonly a?: number; readonly b?: number; readonly ratio?: number };

export type AbStageMetric = {
  readonly component: AnimaRunComponent;
  /** 段の壁時計（ms・同じ段が 2 回出た行はその合計）。 */
  readonly wallMs: AbMetric;
  /** 段の GPU 時間（ns・`stages[].gpu.totalNs` の合計）。 */
  readonly gpuNs: AbMetric;
};

export type AbComparison = {
  /** 区間 A と B の全ての PNG が同じ sha256 か。 */
  readonly shaMatch: boolean;
  /** 全体の壁時計（`wallMs`）。 */
  readonly wallMs: AbMetric;
  readonly stages: readonly AbStageMetric[];
};

export type AbSummary = {
  readonly a: IntervalSummary;
  readonly b: IntervalSummary;
  /** 両区間とも成功したときだけ（失敗した区間の数字は比べ物にならないので出さない）。 */
  readonly comparison?: AbComparison;
};

const SHA_PREFIX = 12;

/** 昇順に並べた中央値（偶数個なら真ん中 2 つの平均）。 */
const median = (values: readonly number[]): number => {
  const sorted = [...values].sort((x, y) => x - y);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};

/** 代表に使う行 = 2 回目以降（1 回目は区間の頭で DiT を読み込む）。1 行しか無ければその行。 */
const representative = (rows: readonly Row[]): readonly Row[] =>
  rows.length === 1 ? rows : rows.slice(1);

/** 行ごとの値の中央値（1 行でも値が無ければ無し — 欠けた行を黙って外すと何の中央値か変わる）。 */
const medianOf = (
  rows: readonly Row[],
  pick: (row: Row) => number | undefined,
): number | undefined => {
  const values = representative(rows).map(pick);
  return values.every((value) => value !== undefined) ? median(values) : undefined;
};

/** 行の中のその段の記録の合計（記録が無い・どれかが値を持たなければ無し）。 */
const stageSum = (
  row: Row,
  component: AnimaRunComponent,
  value: (stage: Row["stages"][number]) => number | undefined,
): number | undefined => {
  const values = row.stages.filter((stage) => stage.component === component).map(value);
  if (values.length === 0) return undefined;
  let sum = 0;
  for (const entry of values) {
    if (entry === undefined) return undefined;
    sum += entry;
  }
  return sum;
};

const metric = (a: number | undefined, b: number | undefined): AbMetric => ({
  ...(a === undefined ? {} : { a }),
  ...(b === undefined ? {} : { b }),
  ...(a === undefined || b === undefined || a === 0 ? {} : { ratio: b / a }),
});

const shaOf = (row: Row): string => {
  // 失敗行でない行は必ず PNG を持つ（generate の成功 = PNG の sha まで取れた）
  if (row.pngSha256 === undefined) {
    throw Error(`行 ${row.index} が失敗でないのに PNG sha256 を持たない`);
  }
  return row.pngSha256;
};

const distinctPrefixes = (rows: readonly Row[]): string[] => [
  ...new Set(rows.map((row) => shaOf(row).slice(0, SHA_PREFIX))),
];

const summarizeInterval = (rows: readonly Row[]): IntervalSummary => {
  if (rows.length === 0) return { kind: "skipped" };
  const failed = rows.find((row) => row.error !== undefined);
  if (failed?.error !== undefined) return { kind: "failed", errorName: failed.error.name };
  const shas = new Set(rows.map(shaOf));
  return {
    kind: "ok",
    count: rows.length,
    sha: { match: shas.size === 1, distinct: distinctPrefixes(rows) },
  };
};

/**
 * 区間 A（`default` の注入）と区間 B（適用中の選択）の行から要約を作る。区間 A の行は 1 行以上要る
 * （区間 B の空 = 回していない）。
 */
export const summarizeAb = (rowsA: readonly Row[], rowsB: readonly Row[]): AbSummary => {
  if (rowsA.length === 0) throw Error("A/B の要約: 区間 A の行が無い");
  const a = summarizeInterval(rowsA);
  const b = summarizeInterval(rowsB);
  if (a.kind !== "ok" || b.kind !== "ok") return { a, b };
  const both = (pick: (row: Row) => number | undefined): AbMetric =>
    metric(medianOf(rowsA, pick), medianOf(rowsB, pick));
  return {
    a,
    b,
    comparison: {
      shaMatch: new Set([...rowsA, ...rowsB].map(shaOf)).size === 1,
      wallMs: both((row) => row.wallMs),
      stages: AB_STAGES.map((component) => ({
        component,
        wallMs: both((row) =>
          stageSum(
            row,
            component,
            (stage) => stage.endMs === undefined ? undefined : stage.endMs - stage.startMs,
          )
        ),
        gpuNs: both((row) => stageSum(row, component, (stage) => stage.gpu?.totalNs)),
      })),
    },
  };
};

const DASH = "—";

const formatMs = (value: number | undefined): string =>
  value === undefined ? DASH : Math.round(value).toLocaleString("en-US");

const formatRatio = (value: number | undefined): string =>
  value === undefined ? DASH : `×${value.toFixed(3)}`;

const intervalCell = (interval: IntervalSummary): string => {
  switch (interval.kind) {
    case "ok":
      return interval.sha.match
        ? `一致（${interval.count} 枚 · ${interval.sha.distinct[0]}）`
        : `不一致（${interval.sha.distinct.join(" / ")}）`;
    case "failed":
      return `失敗 — ${interval.errorName}`;
    case "skipped":
      return "回していない";
  }
};

/** 要約の表の 1 行（項目・区間 A・区間 B・B ÷ A）。 */
export type AbTableRow = readonly [string, string, string, string];

/** 要約の表の中身（文字列だけ — DOM に組むのは呼び手）。 */
export const abTableRows = (summary: AbSummary): AbTableRow[] => {
  const shaRow: AbTableRow = [
    "PNG sha256（区間の中）",
    intervalCell(summary.a),
    intervalCell(summary.b),
    summary.comparison === undefined
      ? DASH
      : summary.comparison.shaMatch
      ? "A と B が一致"
      : "A と B が不一致",
  ];
  const comparison = summary.comparison;
  if (comparison === undefined) return [shaRow];
  const metricRow = (title: string, value: AbMetric, scale = 1): AbTableRow => [
    title,
    formatMs(value.a === undefined ? undefined : value.a * scale),
    formatMs(value.b === undefined ? undefined : value.b * scale),
    formatRatio(value.ratio),
  ];
  return [
    shaRow,
    metricRow("全体の壁時計 (ms)", comparison.wallMs),
    ...comparison.stages.flatMap(({ component, wallMs, gpuNs }) => [
      metricRow(`${component} 壁時計 (ms)`, wallMs),
      metricRow(`${component} GPU 時間 (ms)`, gpuNs, 1e-6),
    ]),
  ];
};

/** sha の判定（区間 A の中・区間 B の中・A と B の間）。 */
const shaVerdicts = (summary: AbSummary, comparison: AbComparison): string => {
  const { a, b } = summary;
  return `sha: 区間 A ${a.kind === "ok" && a.sha.match ? "一致" : "不一致"} · 区間 B ${
    b.kind === "ok" && b.sha.match ? "一致" : "不一致"
  } · A と B ${comparison.shaMatch ? "一致" : "不一致"}`;
};

const transformerRatios = (comparison: AbComparison): string => {
  const transformer = comparison.stages.find((stage) => stage.component === "transformer");
  return `transformer 壁時計 ${formatRatio(transformer?.wallMs.ratio)} / GPU ${
    formatRatio(transformer?.gpuNs.ratio)
  }`;
};

/** 比べられなかった A/B の句（どの区間がどうなったか）。 */
const incompletePhrase = ({ a, b }: AbSummary): string =>
  `区間 A: ${intervalCell(a)} · 区間 B: ${intervalCell(b)}`;

/** quant 1 つ分の A/B の要約。 */
export type AbQuantSummary = { readonly quant: string; readonly summary: AbSummary };

/** A/B で既定の quant と対にする、量子化していない席。 */
const AB_RAW_QUANT = "f16";

/**
 * A/B で回す quant の並び: 配布形の既定 quant → `f16`（既定が `f16` ならその 1 つ）。select の選択は見ない —
 * 欲しいのは既定の席と量子化していない席の対だけで、選択肢の全部（6 席）を回すと無駄に長い。
 */
export const abQuantPlan = (
  options: readonly string[],
  defaultQuant: string,
): readonly string[] => {
  const quants = defaultQuant === AB_RAW_QUANT ? [defaultQuant] : [defaultQuant, AB_RAW_QUANT];
  // 選択肢に無い quant を回すと manifest に無い quant で GPU を取りに行く — 押した時点で止める
  for (const quant of quants) {
    if (!options.includes(quant)) throw Error(`A/B: quant ${quant} が選択肢に無い`);
  }
  return quants;
};

/**
 * quant ごとの A/B を状態行の 1 行に並べる（`<quant>: <句> / <quant>: <句>`）。全ての quant で比べられたら
 * 「完了」、1 つでも比べられなければ「未完」。quant が 1 つなら全体の B ÷ A も含め、2 つ以上なら行が長くなるので
 * sha の判定と transformer の B ÷ A だけ。
 */
export const abQuantsStatusLine = (results: readonly AbQuantSummary[]): string => {
  if (results.length === 0) throw Error("A/B の状態行: quant の要約が無い");
  const complete = results.every(({ summary }) => summary.comparison !== undefined);
  const phrases = results.map(({ quant, summary }) => {
    const { comparison } = summary;
    if (comparison === undefined) return `${quant}: ${incompletePhrase(summary)}`;
    const overall = results.length === 1 ? `全体 ${formatRatio(comparison.wallMs.ratio)} · ` : "";
    return `${quant}: ${shaVerdicts(summary, comparison)} · B ÷ A: ${overall}${
      transformerRatios(comparison)
    }`;
  });
  return `A/B ${complete ? "完了" : "未完"} — ${phrases.join(" / ")}`;
};
