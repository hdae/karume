/**
 * 局面別バケット集計の単体検証（純関数だけ・GPU 不要）。
 *
 * 見る 5 点:
 *
 * 1. **ゲートの居ないターン**（`plain` / `always`）は形がそのまま局面になる
 * 2. **ゲート付きのターン**の 6 局面 — 観測に混ぜない run / `W1` プローブ / 定常の投機 /
 *    探索バースト / plain の定常 / plain から戻った後の投機
 * 3. **`firstExitRun`** — 最初に `switches` が 1 になった run の通し番号（prefill と draft は数えない）
 * 4. **`cold` / `rest`** — 先頭 8 run とその後ろ（局面別とは独立した 2 つ目の分割）
 * 5. **簿記の破れ** — `wallMs` の無い観測は落ちる・`delivered` の無い verify は `accepted + 1`
 *
 * NOTE: リポの慣習に合わせて `Deno.test`（文脈）+ `t.step`（振る舞い）で書く。
 */

import { assertEquals, assertThrows } from "@std/assert";
import type { GenerationGateTrace, GenerationRunPhase } from "../../packages/models/gemma.ts";
import { type TraceBucket, traceOf, type TurnTrace } from "./trace.ts";

const prefill = (chunk = 1, chunks = 1): GenerationRunPhase => ({ kind: "prefill", chunk, chunks });
const draft = (cycle: number): GenerationRunPhase => ({ kind: "draft", cycle });

const verify = (
  over: {
    readonly cycle?: number;
    readonly accepted?: number;
    readonly delivered?: number;
    readonly wallMs?: number;
    readonly gate?: GenerationGateTrace;
  },
): GenerationRunPhase => ({
  kind: "verify",
  cycle: over.cycle ?? 1,
  rows: 4,
  accepted: over.accepted ?? 3,
  ...(over.delivered === undefined ? {} : { delivered: over.delivered }),
  ...(over.wallMs === undefined ? {} : { wallMs: over.wallMs }),
  ...(over.gate === undefined ? {} : { gate: over.gate }),
});

const decode = (
  step: number,
  wallMs: number | undefined,
  gate?: GenerationGateTrace,
): GenerationRunPhase => ({
  kind: "decode",
  step,
  ...(wallMs === undefined ? {} : { wallMs }),
  ...(gate === undefined ? {} : { gate }),
});

const gateOf = (
  mode: GenerationGateTrace["mode"],
  switches: number,
  measured: boolean,
): GenerationGateTrace => ({ mode, switches, measured });

const EMPTY: TraceBucket = { runs: 0, ms: 0, delivered: 0 };
const bucket = (runs: number, ms: number, delivered: number): TraceBucket => ({
  runs,
  ms,
  delivered,
});

/** 局面別 6 つの合計（= 生成相の run 全部）— `cold + rest` と一致するのが表の整合である。 */
const placesTotal = (trace: TurnTrace): TraceBucket => {
  const places = [
    trace.speculate,
    trace.speculateAfterReturn,
    trace.burst,
    trace.w1Probe,
    trace.plain,
    trace.unmeasured,
  ];
  return {
    runs: places.reduce((sum, one) => sum + one.runs, 0),
    ms: places.reduce((sum, one) => sum + one.ms, 0),
    delivered: places.reduce((sum, one) => sum + one.delivered, 0),
  };
};

const agesTotal = (trace: TurnTrace): TraceBucket => ({
  runs: trace.cold.runs + trace.rest.runs,
  ms: trace.cold.ms + trace.rest.ms,
  delivered: trace.cold.delivered + trace.rest.delivered,
});

Deno.test("ゲートの居ないターンは run の形がそのまま局面になる", async (t) => {
  await t.step("always（verify だけ）は全部 speculate・draft と prefill は数えない", () => {
    const trace = traceOf([
      prefill(),
      draft(1),
      verify({ cycle: 1, accepted: 3, delivered: 4, wallMs: 100 }),
      draft(2),
      verify({ cycle: 2, accepted: 1, delivered: 2, wallMs: 80 }),
    ]);
    assertEquals(trace.speculate, bucket(2, 180, 6));
    assertEquals(trace.plain, EMPTY);
    assertEquals(trace.burst, EMPTY);
    assertEquals(trace.w1Probe, EMPTY);
    assertEquals(trace.unmeasured, EMPTY);
    assertEquals(trace.speculateAfterReturn, EMPTY);
    // 先頭 8 run までは cold（この 2 本はどちらも cold）。
    assertEquals(trace.cold, bucket(2, 180, 6));
    assertEquals(trace.rest, EMPTY);
    assertEquals(trace.firstExitRun, undefined);
  });

  await t.step("plain（decode だけ）は全部 plain・1 run が 1 token", () => {
    const trace = traceOf([prefill(), decode(1, 10), decode(2, 11), decode(3, 12)]);
    assertEquals(trace.plain, bucket(3, 33, 3));
    assertEquals(trace.speculate, EMPTY);
    assertEquals(trace.w1Probe, EMPTY);
    assertEquals(trace.cold, bucket(3, 33, 3));
  });

  await t.step("複数 chunk の prefill も数えない（生成相の run だけ）", () => {
    const trace = traceOf([prefill(1, 3), prefill(2, 3), prefill(3, 3), decode(1, 10)]);
    assertEquals(trace.plain, bucket(1, 10, 1));
    assertEquals(agesTotal(trace), bucket(1, 10, 1));
  });
});

/**
 * ゲート付きのターン 1 本（T16 の走行を土台にした合成列 — 局面の網羅が目的。モードの反転は
 * 実装と同じく `switches` の増加と同時にしか起きない形にしてある）。
 *
 * 順に: ①観測に混ぜない先頭 cycle ②`W1` プローブ ③定常の投機 ④その観測で plain へ倒れた cycle
 * ⑤⑥plain の定常 ⑦戻った後の投機 ⑧予算末尾の強制 plain（混ぜない）⑨3 度目の切替の後・cold の外の plain。
 */
const gatedPhases = (): readonly GenerationRunPhase[] => [
  prefill(),
  draft(1),
  verify({ cycle: 1, accepted: 3, delivered: 4, wallMs: 100, gate: gateOf("speculate", 0, false) }),
  decode(1, 10, gateOf("speculate", 0, true)),
  draft(2),
  verify({ cycle: 2, accepted: 3, delivered: 4, wallMs: 100, gate: gateOf("speculate", 0, true) }),
  draft(3),
  verify({ cycle: 3, accepted: 3, delivered: 4, wallMs: 100, gate: gateOf("plain", 1, true) }),
  decode(2, 10, gateOf("plain", 1, true)),
  decode(3, 10, gateOf("plain", 1, true)),
  draft(4),
  verify({ cycle: 4, accepted: 2, delivered: 3, wallMs: 90, gate: gateOf("speculate", 2, true) }),
  verify({ cycle: 5, accepted: 0, delivered: 1, wallMs: 1, gate: gateOf("speculate", 2, false) }),
  decode(4, 5, gateOf("plain", 3, true)),
];

Deno.test("ゲート付きのターンは kind × mode で 6 局面に分かれる", async (t) => {
  const trace = traceOf(gatedPhases());

  await t.step("混ぜない run（先頭 cycle と予算末尾の強制 plain）は unmeasured", () => {
    assertEquals(trace.unmeasured, bucket(2, 101, 5));
  });

  await t.step("speculate 中の decode は W1 プローブ・plain 中の decode は plain の定常", () => {
    assertEquals(trace.w1Probe, bucket(1, 10, 1));
    // plain 3 本（cold の内 2 本 + 外 1 本）。
    assertEquals(trace.plain, bucket(3, 25, 3));
  });

  await t.step("定常 speculate の verify は切替回数で定常と復帰に分かれる", () => {
    assertEquals(trace.speculate, bucket(1, 100, 4));
    assertEquals(trace.speculateAfterReturn, bucket(1, 90, 3));
  });

  await t.step("定常 plain の verify は探索バースト（倒れた観測そのものも 1 本入る）", () => {
    assertEquals(trace.burst, bucket(1, 100, 4));
  });

  await t.step("局面別の合計と cold + rest が一致する（表が時間を落としていない）", () => {
    assertEquals(placesTotal(trace), bucket(9, 426, 20));
    assertEquals(agesTotal(trace), bucket(9, 426, 20));
  });

  await t.step("cold は先頭 8 run・rest はその後ろ", () => {
    assertEquals(trace.cold, bucket(8, 421, 19));
    assertEquals(trace.rest, bucket(1, 5, 1));
  });

  await t.step(
    "firstExitRun は switches が 1 になった run の通し番号（prefill / draft を除く）",
    () => {
      // ①verify ②decode ③verify ④verify ← ここで 1 になる。draft 3 本を数えると 6 になる。
      assertEquals(trace.firstExitRun, 4);
    },
  );

  await t.step("切替が起きなければ firstExitRun は欄ごと無い", () => {
    const steady = traceOf([
      prefill(),
      draft(1),
      verify({ cycle: 1, delivered: 4, wallMs: 100, gate: gateOf("speculate", 0, false) }),
      draft(2),
      verify({ cycle: 2, delivered: 4, wallMs: 100, gate: gateOf("speculate", 0, true) }),
    ]);
    assertEquals(Object.hasOwn(steady, "firstExitRun"), false);
    assertEquals(steady.speculate, bucket(1, 100, 4));
    assertEquals(steady.unmeasured, bucket(1, 100, 4));
  });
});

Deno.test("簿記の破れ", async (t) => {
  await t.step("wallMs の無い観測は落ちる（黙って 0 として数えない）", () => {
    assertThrows(
      () => traceOf([prefill(), decode(1, 10), decode(2, undefined)]),
      Error,
      "decode run 2: 観測に wallMs が無い",
    );
    assertThrows(
      () => traceOf([prefill(), draft(1), verify({ delivered: 4 })]),
      Error,
      "verify run 1: 観測に wallMs が無い",
    );
  });

  await t.step("delivered の無い verify は accepted + 1 で補う", () => {
    // 停止 token で列挙を打ち切った cycle だけ 1 だけ過大になる近似（欄がある走行では使わない）。
    const trace = traceOf([prefill(), draft(1), verify({ accepted: 2, wallMs: 50 })]);
    assertEquals(trace.speculate, bucket(1, 50, 3));
  });
});
