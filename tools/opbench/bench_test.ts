// 計測規約の純関数（GPU 不要）: 反復数の校正と、クロックを張り付かせる空回しの打ち切り条件。
//
// `pinClocks` は fake Heater（決まった ms を返すだけ）で回すので、実 GPU も timestamp も要らない。

import { assert, assertEquals, assertRejects } from "@std/assert";
import {
  calibrateReps,
  type Heater,
  MAX_REPS,
  measureWall,
  pinClocks,
  TARGET_PASS_MS,
  timingWallWarning,
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

Deno.test("timingWallWarning: ns が正しい run（目標長 + フェンスの床）は警告しない", () => {
  const targetNs = TARGET_PASS_MS * 1e6;
  assertEquals(timingWallWarning(targetNs, targetNs + 11e6), undefined);
  // 床が支配する短い run は、比が小さくても判定しない（壁時計が目標長に届いていない）。
  assertEquals(timingWallWarning(1e6, 12e6), undefined);
});

Deno.test("timingWallWarning: raw tick のままの ns（B570 の period 52.08）は壁時計と矛盾するので警告する", () => {
  // 校正は過小な ns を見て反復を 52 倍積むので、実時間は目標長の 52 倍になる。
  const realNs = TARGET_PASS_MS * 1e6 * 52.08;
  const warning = timingWallWarning(realNs / 52.08, realNs + 11e6);
  assert(warning !== undefined && warning.includes("ns に換算されていない疑い"), warning);
});

Deno.test("timingWallWarning: GPU 実時間が壁時計を超えたら警告する（pass は run の内側で走る）", () => {
  const warning = timingWallWarning(120e6, 100e6);
  assert(warning !== undefined && warning.includes("壁時計を超えた"), warning);
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

/**
 * device 単位で排他の区間ロックを真似る fake（本物は runtime の `GpuContext.beginBatch`）。
 * 2 本目の `beginBatch` は 1 本目の `finish` まで決着しない。
 */
const fakeExclusiveGpu = () => {
  let held: Promise<void> = Promise.resolve();
  let finishes = 0;
  return {
    beginBatch: async () => {
      await held;
      const lock = Promise.withResolvers<void>();
      held = lock.promise;
      return {
        finish: () => {
          finishes += 1;
          lock.resolve();
          return Promise.resolve();
        },
      };
    },
    finishes: () => finishes,
  };
};

Deno.test("measureWall: enqueue が reject しても batch を閉じ、次の beginBatch が決着する", async () => {
  const gpu = fakeExclusiveGpu();
  const failure = new Error("host 側の enqueue 失敗");
  let calls = 0;
  const session = {
    enqueue: () => {
      calls += 1;
      return calls === 2 ? Promise.reject(failure) : Promise.resolve();
    },
  };
  const rejected = await assertRejects(() => measureWall(gpu, session, {}, 3, 1));
  assertEquals(rejected, failure);
  assertEquals(gpu.finishes() >= 1, true);
  // 区間ロックが返っていなければ beginBatch は決着せず、macrotask 1 つ後の番兵が先に勝つ。
  const next = await Promise.race([
    gpu.beginBatch().then(() => "opened"),
    new Promise<string>((resolve) => setTimeout(() => resolve("hung"), 0)),
  ]);
  assertEquals(next, "opened");
});
