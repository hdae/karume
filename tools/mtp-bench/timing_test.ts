/**
 * mode × kind の GPU 内訳集計の単体検証（純関数だけ・GPU 不要）。
 *
 * 見る 5 点:
 *
 * 1. **mode ごとに kind が分かれる** — `auto` の decode（ゲートの plain step / `W1` プローブ）が
 *    `plain` の decode と別の欄に立つ（1 段で畳んでいた頃はここが混ざっていた）
 * 2. **run が 0 本の (mode, kind) は欄ごと無い** — `Object.hasOwn` が false（1 本も無い mode は
 *    mode の欄ごと無い）
 * 3. **分母は runs** — bucket の `msPerRun` は `totalNs` から・key 別も同じ分母
 * 4. **keys の並び** — GPU 時間の降順（同値はキーの辞書順）・上から 12 本で切る
 * 5. **暖機の run は積まれない** — フォールト注入（暖機に巨大な内訳を流しても欄が立たない）
 *
 * NOTE: リポの慣習に合わせて `Deno.test`（文脈）+ `t.step`（振る舞い）で書く。
 */

import { assert, assertAlmostEquals, assertEquals, assertFalse } from "@std/assert";
import type { GpuTimingEntry, GpuTimingStats } from "../../packages/runtime/mod.ts";
import type { BenchMode, RunKind } from "./summary.ts";
import {
  emptyTimingTallies,
  type GpuBreakdown,
  recordRunTiming,
  summarizeTiming,
  type TimingSummary,
} from "./timing.ts";

const entry = (key: string, ns: number, dispatchCount: number): GpuTimingEntry => ({
  key,
  ns,
  dispatchCount,
  workgroupCount: dispatchCount * 8,
});

/**
 * 合成の内訳 1 通。
 *
 * `totalNs` を entries の総和と**わざと別の数**にしてあるのは、bucket の `msPerRun` が
 * `totalNs` から出ていることを確かめるためである（一致させると entries を足し直す実装でも
 * 緑になり、取り違えが値に出ない）。`dispatchCount` も同じ理由で別に与える。
 */
const stats = (
  over: {
    readonly entries: readonly GpuTimingEntry[];
    readonly totalNs: number;
    readonly dispatchCount: number;
    readonly clampedNegativeSamples?: number;
  },
): GpuTimingStats => ({
  entries: over.entries,
  totalNs: over.totalNs,
  dispatchCount: over.dispatchCount,
  clampedNegativeSamples: over.clampedNegativeSamples ?? 0,
});

/** 1 ms = 1e6 ns（内訳は ns で積み、表は ms で読む）。 */
const MS = 1e6;

/**
 * 合成の走行 1 本ぶん — plain: decode 2 本 / always: draft 1 本 + verify 1 本 /
 * auto: decode 1 本 + verify 1 本。
 *
 * 6 本すべてに違う数を置くのは、(mode, kind) を取り違えた実装が値で分かるようにするためである。
 */
const measuredRuns = (): readonly {
  readonly mode: BenchMode;
  readonly kind: RunKind;
  readonly stats: GpuTimingStats;
}[] => [
  {
    mode: "plain",
    kind: "decode",
    stats: stats({
      entries: [entry("linear/w4", 6 * MS, 40), entry("softmax", 2 * MS, 4)],
      totalNs: 10 * MS,
      dispatchCount: 50,
    }),
  },
  {
    mode: "plain",
    kind: "decode",
    stats: stats({
      entries: [entry("linear/w4", 8 * MS, 40), entry("rmsnorm", 4 * MS, 6)],
      totalNs: 14 * MS,
      dispatchCount: 54,
    }),
  },
  {
    mode: "always",
    kind: "draft",
    stats: stats({
      entries: [entry("linear/w4", 3 * MS, 12)],
      totalNs: 4 * MS,
      dispatchCount: 15,
      clampedNegativeSamples: 2,
    }),
  },
  {
    mode: "always",
    kind: "verify",
    stats: stats({
      entries: [entry("linear/w4", 20 * MS, 40), entry("attention/m4", 5 * MS, 8)],
      totalNs: 30 * MS,
      dispatchCount: 60,
    }),
  },
  {
    mode: "auto",
    kind: "decode",
    stats: stats({
      entries: [entry("linear/w4", 7 * MS, 40)],
      totalNs: 11 * MS,
      dispatchCount: 52,
    }),
  },
  {
    mode: "auto",
    kind: "verify",
    stats: stats({
      entries: [entry("linear/w4", 21 * MS, 40)],
      totalNs: 33 * MS,
      dispatchCount: 61,
    }),
  },
];

const breakdownOf = (
  runs: readonly {
    readonly mode: BenchMode;
    readonly kind: RunKind;
    readonly measured?: boolean;
    readonly stats: GpuTimingStats;
  }[],
): GpuBreakdown => {
  const tallies = emptyTimingTallies();
  for (const run of runs) {
    recordRunTiming(tallies, {
      mode: run.mode,
      kind: run.kind,
      measured: run.measured ?? true,
    }, run.stats);
  }
  return summarizeTiming(tallies);
};

/** 欄が在ることを型の上でも確かめる（無ければテストをその場で落とす）。 */
const seatOf = (breakdown: GpuBreakdown, mode: BenchMode, kind: RunKind): TimingSummary => {
  const byKind = breakdown[mode];
  assert(byKind !== undefined, `mode ${mode} の欄が無い`);
  const one = byKind[kind];
  assert(one !== undefined, `${mode}/${kind} の欄が無い`);
  return one;
};

Deno.test("GPU 内訳は mode × kind に割れる", async (t) => {
  const breakdown = breakdownOf(measuredRuns());

  await t.step("mode の欄は plain / always / auto の順で、走った mode だけ立つ", () => {
    assertEquals(Object.keys(breakdown), ["plain", "always", "auto"]);
  });

  await t.step("kind の欄はその mode で走った種別だけ（並びは prefill → verify の順）", () => {
    assertEquals(Object.keys(breakdown.plain ?? {}), ["decode"]);
    assertEquals(Object.keys(breakdown.always ?? {}), ["draft", "verify"]);
    assertEquals(Object.keys(breakdown.auto ?? {}), ["decode", "verify"]);
  });

  await t.step("run が 0 本の (mode, kind) は欄ごと無い", () => {
    assertFalse(Object.hasOwn(breakdown.plain ?? {}, "draft"));
    assertFalse(Object.hasOwn(breakdown.plain ?? {}, "verify"));
    assertFalse(Object.hasOwn(breakdown.always ?? {}, "decode"));
    assertFalse(Object.hasOwn(breakdown.plain ?? {}, "prefill"));
  });

  await t.step("auto の decode は plain の decode に混ざらない", () => {
    // 混ざっていれば plain/decode は 3 本（10 + 14 + 11 ns → 11.67 ms/run）になる。
    assertEquals(seatOf(breakdown, "plain", "decode").runs, 2);
    assertAlmostEquals(seatOf(breakdown, "plain", "decode").msPerRun, 12);
    assertEquals(seatOf(breakdown, "auto", "decode").runs, 1);
    assertAlmostEquals(seatOf(breakdown, "auto", "decode").msPerRun, 11);
  });

  await t.step("分母は runs（bucket は totalNs / dispatchCount の合計から）", () => {
    const plainDecode = seatOf(breakdown, "plain", "decode");
    // (10 + 14) / 2 ms・(50 + 54) / 2 dispatch。
    assertAlmostEquals(plainDecode.msPerRun, 12);
    assertAlmostEquals(plainDecode.dispatchesPerRun, 52);
    const autoVerify = seatOf(breakdown, "auto", "verify");
    assertAlmostEquals(autoVerify.msPerRun, 33);
    assertAlmostEquals(autoVerify.dispatchesPerRun, 61);
  });

  await t.step("key 別も同じ分母（runs）で、丸めずそのまま出る", () => {
    // linear/w4 は 6 + 8 = 14 ms・40 + 40 = 80 dispatch を 2 本で割る。片方の run にしか出ない
    // key（rmsnorm / softmax）も分母は同じ 2 本（1 本で割ると 4 ms / 2 ms になる）。
    assertEquals(seatOf(breakdown, "plain", "decode").keys, [
      { key: "linear/w4", msPerRun: 7, dispatchesPerRun: 40 },
      { key: "rmsnorm", msPerRun: 2, dispatchesPerRun: 3 },
      { key: "softmax", msPerRun: 1, dispatchesPerRun: 2 },
    ]);
  });

  await t.step("clampedNegativeSamples は合計をそのまま持つ（畳んで消さない）", () => {
    assertEquals(seatOf(breakdown, "always", "draft").clampedNegativeSamples, 2);
    assertEquals(seatOf(breakdown, "always", "verify").clampedNegativeSamples, 0);
  });
});

Deno.test("keys の並びは GPU 時間の降順（同値はキーの辞書順）で上から 12 本", async (t) => {
  // 同値（3 ms）を 2 本置いて辞書順を確かめる。ns が大きい順に 14 本 → 上 12 本だけ残る。
  const entries: readonly GpuTimingEntry[] = [
    entry("zzz/last", 3 * MS, 1),
    entry("aaa/tie", 3 * MS, 1),
    ...Array.from({ length: 12 }, (_unused, at) => entry(`op/${at}`, (100 - at) * MS, 1)),
  ];
  const breakdown = breakdownOf([
    {
      mode: "always",
      kind: "verify",
      stats: stats({ entries, totalNs: 1000 * MS, dispatchCount: 14 }),
    },
  ]);
  const keys = breakdown.always?.verify?.keys ?? [];

  await t.step("降順に並ぶ", () => {
    assertEquals(keys.map((one) => one.key), Array.from({ length: 12 }, (_u, at) => `op/${at}`));
  });

  await t.step("12 本で切る（同値の 2 本は落ちる）", () => {
    assertEquals(keys.length, 12);
    assertFalse(keys.some((one) => one.key === "aaa/tie" || one.key === "zzz/last"));
  });

  await t.step("同値どうしはキーの辞書順（上位を 10 本に減らして境界を見る）", () => {
    const short = breakdownOf([
      {
        mode: "always",
        kind: "verify",
        stats: stats({
          entries: entries.filter((one) => one.ns <= 3 * MS || one.ns >= 99 * MS),
          totalNs: 1000 * MS,
          dispatchCount: 4,
        }),
      },
    ]);
    assertEquals((short.always?.verify?.keys ?? []).map((one) => one.key), [
      "op/0",
      "op/1",
      "aaa/tie",
      "zzz/last",
    ]);
  });
});

Deno.test("暖機の run は内訳に積まれない（フォールト注入）", async (t) => {
  const warmup = stats({
    entries: [entry("linear/w4", 6000 * MS, 40000)],
    totalNs: 9000 * MS,
    dispatchCount: 50000,
  });

  await t.step("暖機だけの (mode, kind) は欄ごと立たない", () => {
    const breakdown = breakdownOf([
      { mode: "plain", kind: "prefill", measured: false, stats: warmup },
      { mode: "always", kind: "verify", measured: false, stats: warmup },
    ]);
    assertEquals(breakdown, {});
  });

  await t.step("暖機を混ぜても定常の ms/run が動かない", () => {
    const runs = measuredRuns();
    const withWarmup = breakdownOf([
      { mode: "plain", kind: "decode", measured: false, stats: warmup },
      ...runs,
      { mode: "auto", kind: "verify", measured: false, stats: warmup },
    ]);
    assertEquals(withWarmup, breakdownOf(runs));
  });
});
