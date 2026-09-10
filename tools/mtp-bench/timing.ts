/**
 * `--gpu-timing` の op 別 GPU 時間の器と集計（**純関数だけ** — GPU も I/O も時計も触らない）。
 *
 * 畳む軸が **mode × kind** の 2 段なのは、`kind` だけで畳むと `auto` のゲートが落とした
 * plain step と `W1` プローブ（どちらも decode 形）が `plain` モードの decode に埋もれるからで
 * ある。この表の用途は「ゲートを付けた走行と付けない走行で、同じカーネルが 1 run 何 ms だったか」
 * を比べることで、1 段の表ではその比較ができない（`gpu.decode` が 3 モードの M=1 run の混合に
 * なる）。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 */

import type { GpuTimingStats } from "../../packages/runtime/mod.ts";
import { BENCH_MODES, type BenchMode, RUN_KINDS, type RunKind } from "./summary.ts";

/** 1 つの (mode, kind) に書く op（パイプラインキー）の本数。 */
const TIMING_TOP = 12;

/** GPU 内訳の器（1 つの (mode, kind) ぶん・暖機を除く全ターンの合算）。 */
export type TimingTally = {
  runs: number;
  totalNs: number;
  dispatchCount: number;
  clampedNegativeSamples: number;
  readonly keys: Map<string, { ns: number; dispatchCount: number }>;
};

/** 走行 1 本ぶんの器（{@link recordRunTiming} が積み、{@link summarizeTiming} が畳む）。 */
export type TimingTallies = {
  readonly [M in BenchMode]: { readonly [K in RunKind]: TimingTally };
};

export const emptyTimingTallies = (): TimingTallies => {
  const one = (): TimingTally => ({
    runs: 0,
    totalNs: 0,
    dispatchCount: 0,
    clampedNegativeSamples: 0,
    keys: new Map(),
  });
  const byKind = (): { readonly [K in RunKind]: TimingTally } => ({
    prefill: one(),
    decode: one(),
    draft: one(),
    verify: one(),
  });
  return { plain: byKind(), always: byKind(), auto: byKind() };
};

/**
 * run 1 本の内訳を、その run が出た (mode, kind) の器へ積む。
 *
 * 暖機（`measured === false`）が落ちる口は**ここ 1 箇所**である。立ち上げ（WGSL の解析・params
 * の生成）を含む run が定常の内訳に混ざると、op 別の ms/run が初回の費用ぶん過大になる。
 */
export const recordRunTiming = (
  tallies: TimingTallies,
  at: { readonly mode: BenchMode; readonly kind: RunKind; readonly measured: boolean },
  stats: GpuTimingStats,
): void => {
  if (!at.measured) return;
  const bucket = tallies[at.mode][at.kind];
  bucket.runs += 1;
  bucket.totalNs += stats.totalNs;
  bucket.dispatchCount += stats.dispatchCount;
  bucket.clampedNegativeSamples += stats.clampedNegativeSamples;
  for (const entry of stats.entries) {
    const key = bucket.keys.get(entry.key) ?? { ns: 0, dispatchCount: 0 };
    key.ns += entry.ns;
    key.dispatchCount += entry.dispatchCount;
    bucket.keys.set(entry.key, key);
  }
};

/** 内訳 1 行（パイプラインキー 1 本ぶん・分母は同じ (mode, kind) の `runs`）。 */
export type TimingKeySummary = {
  readonly key: string;
  readonly msPerRun: number;
  readonly dispatchesPerRun: number;
};

/** 1 つの (mode, kind) の内訳（`clampedNegativeSamples` 以外の分母は全て `runs`）。 */
export type TimingSummary = {
  readonly runs: number;
  readonly msPerRun: number;
  readonly dispatchesPerRun: number;
  readonly clampedNegativeSamples: number;
  /** GPU 時間の降順（同値はキーの辞書順）に上から {@link TIMING_TOP} 本。 */
  readonly keys: readonly TimingKeySummary[];
};

/**
 * JSON の `gpu`（mode → kind → 内訳）。
 *
 * run が 0 本の (mode, kind) は**欄ごと無い**（`always` に decode は無い・`plain` に draft /
 * verify は無い）。0 を書くと「測ったら 0 ms だった」と読めてしまう（`RunSummary.msPerRun` と
 * 同じ原則）。run が 1 本も無かった mode も同じ理由で欄ごと落ちる（空の `{}` は「測った」と
 * 読めてしまう）。
 */
export type GpuBreakdown = {
  readonly [M in BenchMode]?: { readonly [K in RunKind]?: TimingSummary };
};

const summarizeOne = (bucket: TimingTally): TimingSummary => ({
  runs: bucket.runs,
  msPerRun: bucket.totalNs / 1e6 / bucket.runs,
  dispatchesPerRun: bucket.dispatchCount / bucket.runs,
  clampedNegativeSamples: bucket.clampedNegativeSamples,
  keys: [...bucket.keys.entries()]
    .sort(([leftKey, left], [rightKey, right]) =>
      right.ns - left.ns || (leftKey < rightKey ? -1 : leftKey > rightKey ? 1 : 0)
    )
    .slice(0, TIMING_TOP)
    .map(([key, total]) => ({
      key,
      msPerRun: total.ns / 1e6 / bucket.runs,
      dispatchesPerRun: total.dispatchCount / bucket.runs,
    })),
});

/** 器 → JSON の `gpu`。欄の並びは {@link BENCH_MODES} × {@link RUN_KINDS}。 */
export const summarizeTiming = (tallies: TimingTallies): GpuBreakdown => {
  const breakdown: { [M in BenchMode]?: { [K in RunKind]?: TimingSummary } } = {};
  for (const mode of BENCH_MODES) {
    let byKind: { [K in RunKind]?: TimingSummary } | undefined;
    for (const kind of RUN_KINDS) {
      const bucket = tallies[mode][kind];
      if (bucket.runs === 0) continue;
      byKind ??= {};
      byKind[kind] = summarizeOne(bucket);
    }
    if (byKind !== undefined) breakdown[mode] = byKind;
  }
  return breakdown;
};
