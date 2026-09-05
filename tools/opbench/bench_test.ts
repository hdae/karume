// 計測規約の純関数（GPU 不要）: 反復数の校正と、クロックを張り付かせる空回しの打ち切り条件。
//
// `pinClocks` は fake Heater（決まった ms を返すだけ）で回すので、実 GPU も timestamp も要らない。

import { assertEquals } from "@std/assert";
import {
  calibrateReps,
  type Heater,
  MAX_REPS,
  pinClocks,
  TARGET_PASS_MS,
  WARMUP_MIN_RUNS,
} from "./bench.ts";

/** 何回呼ばれたかを数える fake（`ms` は呼び出し回数から決める）。 */
const fakeHeater = (ms: (call: number) => number): Heater & { readonly calls: () => number } => {
  let calls = 0;
  return {
    run: () => {
      calls += 1;
      return Promise.resolve(ms(calls));
    },
    calls: () => calls,
  };
};

Deno.test("calibrateReps: 推定が測れない値（0 / NaN / 負）なら上限まで積む", () => {
  assertEquals(calibrateReps(0), MAX_REPS);
  assertEquals(calibrateReps(Number.NaN), MAX_REPS);
  assertEquals(calibrateReps(-1), MAX_REPS);
});

Deno.test("calibrateReps: 1 dispatch で目標長ちょうどなら 1 本", () => {
  assertEquals(calibrateReps(TARGET_PASS_MS * 1e6), 1);
});

Deno.test("calibrateReps: 速すぎる dispatch は上限で打ち切る", () => {
  assertEquals(calibrateReps(1e3), MAX_REPS);
});

Deno.test("pinClocks: 安定した filler は累計 500ms（= WARMUP_NS）に届くまで回る", async () => {
  const heater = fakeHeater(() => 10);
  const { runs, ms } = await pinClocks(heater);
  // 10ms × 50 回 = 500ms でちょうど下限に届く（回数下限 WARMUP_MIN_RUNS も満たしている）。
  assertEquals(heater.calls(), 50);
  assertEquals(ms, 500);
  // NOTE: `runs` は break した時点のループ添字なので、呼び出し回数より 1 小さい。
  assertEquals(runs, heater.calls() - 1);
  assertEquals(runs >= WARMUP_MIN_RUNS, true);
});

Deno.test("pinClocks: 毎回遅くなる filler は安定と見なさず 64 回で打ち切る", async () => {
  const heater = fakeHeater((call) => 2 ** call);
  const { runs } = await pinClocks(heater);
  assertEquals(heater.calls(), 64);
  assertEquals(runs, 64);
});
