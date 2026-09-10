/**
 * 要約の計算の単体検証（純関数だけ・GPU 不要）。
 *
 * 見る 12 点:
 *
 * 1. **ローテーションの台本** — 暖機 3 本の後に P S S P P A A P / P A A P P S S P が交互に `rounds` 回
 * 2. **分母の 2 本**（`main.ts` の進捗行も呼ぶ口）— `tokensAfterFirst` / `tokensPerCycle`
 *    （正本は `delivered / cycles`）
 * 3. **中央値と派生量** — `msPerToken` / `hostMsPerToken`（3 モード）/ `tokensPerCycle` /
 *    `cycleMs` / `hostMsPerCycle`（always だけ）/ `plainSteps` / `switches`（auto だけ）/
 *    `speedup` / `speedupAuto` / run 別の本数と 1 本あたりの壁
 * 4. **実効 k** — 受理数ヒストグラムの長さ − 1・ターン間で食い違えば落ちる
 * 5. **暖機は要約に入らない** — フォールト注入（暖機の値を 1000 倍にしても中央値が動かない）
 * 6. **token 列の突合** — always / auto それぞれの一致・不一致の最初の添字・接頭辞
 * 7. **局面別内訳の中央値**（`trace.ts` のバケット）— バケットごとに 3 欄それぞれの中央値・
 *    `firstExitRun` は起きたターンだけの中央値
 * 8. **warm（`--warm`）の要約** — 3 モードに揃う自ターン番号までしか入らない・列一致は番号ごと
 * 9. **warm の容量の門** — 走る前に溢れを落とす式（`warmPeakPositions` / `assertWarmCapacity`）
 * 10. **warm の前置** — 前ターンの停止から次のターンの前置を決める（打ち切り → 閉じ札 1 個・
 *     閉じ札で閉じた → 前置なし・閉じ札以外の停止 token → 落ちる）
 * 11. **warm の context 長の一致** — 自ターン番号ごとに 3 モードが揃っているか（cold では欄ごと無い）
 * 12. **warm の追記列の門** — 台本の自ターン数が発話の本数を超えないこと（`assertWarmFollowUps`）
 *
 * NOTE: リポの慣習に合わせて `Deno.test`（文脈）+ `t.step`（振る舞い）で書く。
 */

import { assert, assertAlmostEquals, assertEquals, assertThrows } from "@std/assert";
import type { GenerationSpeculation } from "../../packages/models/gemma.ts";
import {
  assertWarmCapacity,
  assertWarmFollowUps,
  median,
  summarizeTurns,
  tokensAfterFirst,
  tokensPerCycle,
  type TurnPlan,
  turnPlan,
  type TurnRecord,
  warmPeakPositions,
  warmTurnPrefix,
} from "./summary.ts";
import type { TraceBucket, TurnTrace } from "./trace.ts";

/** 合成ターン: 生成 10 token 相当（`tokens - 1 = 10`）で、run 壁だけを与える。 */
const IDS: readonly number[] = [11, 22, 33];

const bucket = (runs: number, ms: number, delivered: number): TraceBucket => ({
  runs,
  ms,
  delivered,
});

/**
 * 合成の局面別内訳（`scale` 倍したもの）。
 *
 * 要約が見るのはバケットごとの中央値だけなので、バケットを取り違えた実装が値で分かるように
 * 6 局面すべてに違う数を置く。`scale` は「ターンごとに違う値」を作るための倍率である。
 */
const traceFixture = (scale = 1): TurnTrace => ({
  speculate: bucket(4 * scale, 40 * scale, 16 * scale),
  speculateAfterReturn: bucket(1 * scale, 11 * scale, 3 * scale),
  burst: bucket(2 * scale, 22 * scale, 5 * scale),
  w1Probe: bucket(3 * scale, 33 * scale, 3 * scale),
  plain: bucket(5 * scale, 55 * scale, 5 * scale),
  unmeasured: bucket(6 * scale, 66 * scale, 7 * scale),
  cold: bucket(8 * scale, 88 * scale, 20 * scale),
  rest: bucket(13 * scale, 139 * scale, 19 * scale),
});

const plainTurn = (
  round: number,
  generationMs: number,
  decodeWallMs: number,
  over: Partial<TurnRecord> = {},
): TurnRecord => ({
  mode: "plain",
  round,
  warmup: false,
  // cold の要約は自ターン番号も context 長も読まない（warm のケースは `over` で明示する）。
  ownIndex: 1,
  contextTokens: 0,
  tokens: 11,
  stopReason: "max-tokens",
  ids: IDS,
  text: "",
  turnMs: generationMs + 100,
  firstTokenMs: 100,
  generationMs,
  runs: {
    prefill: { count: 1, wallMs: 50 },
    decode: { count: 10, wallMs: decodeWallMs },
    draft: { count: 0, wallMs: 0 },
    verify: { count: 0, wallMs: 0 },
  },
  trace: traceFixture(),
  ...over,
});

/** 合成ターン（常時投機）: 5 cycle・確定 11 token（`tokensPerCycle` = 2.2）。 */
const alwaysTurn = (
  round: number,
  generationMs: number,
  runWallMs: { draft: number; verify: number },
  over: Partial<TurnRecord> = {},
): TurnRecord => ({
  mode: "always",
  round,
  warmup: false,
  ownIndex: 1,
  contextTokens: 0,
  tokens: 11,
  stopReason: "max-tokens",
  ids: IDS,
  text: "",
  turnMs: generationMs + 100,
  firstTokenMs: 100,
  generationMs,
  runs: {
    prefill: { count: 1, wallMs: 50 },
    decode: { count: 0, wallMs: 0 },
    draft: { count: 5, wallMs: runWallMs.draft },
    verify: { count: 5, wallMs: runWallMs.verify },
  },
  trace: traceFixture(),
  speculation: {
    cycles: 5,
    draftRuns: 5,
    drafted: 15,
    accepted: 6,
    delivered: 11,
    acceptedHistogram: [1, 2, 2],
  },
  ...over,
});

/**
 * 合成ターン（ゲート付き）: 4 cycle で 9 token 確定（`tokensPerCycle` = 2.25）+ plain step 2 本。
 *
 * plain step は観測席では `decode` として上がる（`Gemma4RunPhase` に枝を足していない）ので、
 * run 壁も decode の欄に積まれる。
 */
const autoTurn = (
  round: number,
  generationMs: number,
  runWallMs: { draft: number; verify: number; decode: number },
  over: Partial<TurnRecord> = {},
): TurnRecord => ({
  mode: "auto",
  round,
  warmup: false,
  ownIndex: 1,
  contextTokens: 0,
  tokens: 11,
  stopReason: "max-tokens",
  ids: IDS,
  text: "",
  turnMs: generationMs + 100,
  firstTokenMs: 100,
  generationMs,
  runs: {
    prefill: { count: 1, wallMs: 50 },
    decode: { count: 2, wallMs: runWallMs.decode },
    draft: { count: 4, wallMs: runWallMs.draft },
    verify: { count: 4, wallMs: runWallMs.verify },
  },
  trace: traceFixture(),
  speculation: {
    cycles: 4,
    draftRuns: 4,
    drafted: 12,
    accepted: 5,
    delivered: 9,
    acceptedHistogram: [1, 2, 1],
    plainSteps: 2,
    switches: 1,
  },
  ...over,
});

/**
 * 3 ターンずつの素の測定。
 *
 * plain 10/12/14 → 中央値 12・always 5/6/7 → 6（`speedup` 2）・auto 7/8/9 → 8
 * （`speedupAuto` 1.5）ms/tok。
 */
const measuredTurns = (): TurnRecord[] => [
  plainTurn(1, 100, 80),
  plainTurn(1, 120, 100),
  plainTurn(2, 140, 120),
  alwaysTurn(1, 50, { draft: 10, verify: 30 }),
  alwaysTurn(1, 60, { draft: 10, verify: 30 }),
  alwaysTurn(2, 70, { draft: 10, verify: 30 }),
  autoTurn(1, 70, { draft: 10, verify: 24, decode: 6 }),
  autoTurn(1, 80, { draft: 10, verify: 24, decode: 6 }),
  autoTurn(2, 90, { draft: 10, verify: 24, decode: 6 }),
];

const withoutMode = (turns: readonly TurnRecord[], mode: TurnRecord["mode"]): TurnRecord[] =>
  turns.filter((turn) => turn.mode !== mode);

/**
 * warm の合成記録の `ids`（自ターン番号ごとに違う列）。
 *
 * warm では各ターンが会話の続きなので、同じモードの中でも列は番号ごとに違う。番号を取り違えた
 * 突合は「別の番号どうし」を比べることになり、そこで割れる形にしてある。
 */
const warmIds = (ownIndex: number): readonly number[] => [11, 22 + ownIndex, 33];

/**
 * warm の合成記録 — plain 6 本 / always 3 本 / auto 3 本（自ターン番号 1..n）。
 *
 * plain の 4〜6 本目（揃わない番号）は ms/token が 10 倍になっている: 要約に混ざれば中央値が
 * 12 → 100 に動くので、揃える実装だけが緑になる（フォールト注入）。
 */
const warmTurns = (): TurnRecord[] => [
  ...[100, 120, 140, 1000, 1200, 1400].map((generationMs, at) =>
    plainTurn(1, generationMs, generationMs * 0.8, {
      ownIndex: at + 1,
      contextTokens: 1000 * (at + 1),
      ids: warmIds(at + 1),
    })
  ),
  ...[50, 60, 70].map((generationMs, at) =>
    alwaysTurn(1, generationMs, { draft: 10, verify: 30 }, {
      ownIndex: at + 1,
      contextTokens: 1000 * (at + 1),
      ids: warmIds(at + 1),
    })
  ),
  ...[70, 80, 90].map((generationMs, at) =>
    autoTurn(1, generationMs, { draft: 10, verify: 24, decode: 6 }, {
      ownIndex: at + 1,
      contextTokens: 1000 * (at + 1),
      ids: warmIds(at + 1),
    })
  ),
];

Deno.test("ローテーションの台本", async (t) => {
  await t.step(
    "暖機 3 本の後に P S S P P A A P（奇数 round）と P A A P P S S P（偶数 round）を交互に",
    () => {
      const plans = turnPlan(2);
      assertEquals(plans.length, 3 + 16);
      assertEquals(
        plans.map((plan) => `${plan.mode}#${plan.round}${plan.warmup ? "w" : ""}`),
        [
          "plain#0w",
          "always#0w",
          "auto#0w",
          "plain#1",
          "always#1",
          "always#1",
          "plain#1",
          "plain#1",
          "auto#1",
          "auto#1",
          "plain#1",
          "plain#2",
          "auto#2",
          "auto#2",
          "plain#2",
          "plain#2",
          "always#2",
          "always#2",
          "plain#2",
        ],
      );
    },
  );

  await t.step(
    "2 つの投機モードは round を跨いで前席と後席を同じ回数ずつ取る（漂流の偏りを消す）",
    () => {
      const seat = (mode: string, round: number): number =>
        (turnPlan(4).findIndex((plan) => plan.round === round && plan.mode === mode) - 3) % 8;
      // 奇数 round は always が前席（添字 1）・偶数 round は auto が前席。4 round で 2 回ずつ。
      assertEquals([1, 2, 3, 4].map((round) => seat("always", round)), [1, 5, 1, 5]);
      assertEquals([1, 2, 3, 4].map((round) => seat("auto", round)), [5, 1, 5, 1]);
    },
  );

  await t.step("既定の 3 round は 27 ターン・2 round は 19 ターン", () => {
    assertEquals(turnPlan(3).length, 27);
    assertEquals(turnPlan(2).length, 19);
  });

  await t.step("2 つの投機モードを挟む plain は同じ本数（1 round に各 2 本）", () => {
    const measured = turnPlan(1).filter((plan) => !plan.warmup).map((plan) => plan.mode);
    assertEquals(measured.filter((mode) => mode === "plain").length, 4);
    assertEquals(measured.filter((mode) => mode === "always").length, 2);
    assertEquals(measured.filter((mode) => mode === "auto").length, 2);
  });

  await t.step("暖機は round 0 で warmup が立ち、以後は立たない", () => {
    const plans: readonly TurnPlan[] = turnPlan(1);
    assertEquals(plans.filter((plan) => plan.warmup).map((plan) => plan.round), [0, 0, 0]);
    assert(plans.slice(3).every((plan) => !plan.warmup && plan.round === 1));
  });

  await t.step("rounds が 1 未満・非整数なら落ちる", () => {
    assertThrows(() => turnPlan(0), Error, "1 以上の整数");
    assertThrows(() => turnPlan(1.5), Error, "1 以上の整数");
  });
});

Deno.test("中央値", async (t) => {
  await t.step("昇順に並べた中央（偶数本は上側）を補間せずに返す", () => {
    assertEquals(median([3, 1, 2]), 2);
    assertEquals(median([4, 1, 3, 2]), 3);
  });

  await t.step("空の列は落ちる", () => {
    assertThrows(() => median([]), Error, "空の列");
  });
});

Deno.test("分母の 2 本（進捗行と要約が共有する式）", async (t) => {
  /** 5 cycle・確定 11 token の勘定（`alwaysTurn` と同じ形 — 単体で呼ぶために独立に組む）。 */
  const tally = (over: Partial<GenerationSpeculation> = {}): GenerationSpeculation => ({
    cycles: 5,
    draftRuns: 5,
    drafted: 15,
    accepted: 6,
    delivered: 11,
    acceptedHistogram: [1, 2, 2],
    ...over,
  });

  await t.step("tokensAfterFirst は tokens − 1", () => {
    assertEquals(tokensAfterFirst(plainTurn(1, 100, 80)), 10);
  });

  await t.step("1 token で終わったターンは落ちる（分母 1 に丸めない）", () => {
    assertThrows(
      () => tokensAfterFirst(plainTurn(1, 100, 80, { tokens: 1 })),
      Error,
      "ms/token",
    );
  });

  await t.step("tokensPerCycle は delivered / cycles（accepted からは作らない）", () => {
    assertAlmostEquals(tokensPerCycle(tally()), 2.2);
    // 停止 token で打ち切った cycle があるターン: `(accepted + cycles) / cycles` は 2.2 と
    // 出るが、実際に確定したのは 10 token = 2.0 である。
    assertAlmostEquals(tokensPerCycle(tally({ delivered: 10 })), 2);
    assertAlmostEquals(
      tokensPerCycle(tally({ accepted: 0, delivered: 5, acceptedHistogram: [5, 0, 0] })),
      1,
    );
  });

  await t.step("cycles 0 では割れずに落ちる", () => {
    assertThrows(
      () => tokensPerCycle(tally({ cycles: 0, accepted: 0, delivered: 0 })),
      Error,
      "cycles 0",
    );
  });
});

Deno.test("要約の派生量", async (t) => {
  const summary = summarizeTurns(measuredTurns());

  await t.step("msPerToken は generationMs / (tokens − 1) の中央値（3 モード）", () => {
    assertAlmostEquals(summary.plain.msPerToken, 12);
    assertAlmostEquals(summary.always.msPerToken, 6);
    assertAlmostEquals(summary.auto.msPerToken, 8);
    assertEquals(summary.plain.turns, 3);
    assertEquals(summary.always.turns, 3);
    assertEquals(summary.auto.turns, 3);
  });

  await t.step("speedup は plain / always・speedupAuto は plain / auto", () => {
    assertAlmostEquals(summary.speedup, 2);
    assertAlmostEquals(summary.speedupAuto, 1.5);
  });

  await t.step("hostMsPerToken は生成相の run 壁を引いた残り（prefill は引かない）", () => {
    // plain: (100−80)/10 = 2・(120−100)/10 = 2・(140−120)/10 = 2
    assertAlmostEquals(summary.plain.hostMsPerToken, 2);
    // always: (50−40)/10 = 1・(60−40)/10 = 2・(70−40)/10 = 3 → 中央値 2
    assertAlmostEquals(summary.always.hostMsPerToken, 2);
    // auto: (70−40)/10 = 3・(80−40)/10 = 4・(90−40)/10 = 5 → 中央値 4
    assertAlmostEquals(summary.auto.hostMsPerToken, 4);
  });

  await t.step("prefill の壁 50 ms は hostMsPerToken に効かない（フォールト注入）", () => {
    const heavyPrefill = measuredTurns().map((turn) => ({
      ...turn,
      runs: { ...turn.runs, prefill: { count: 1, wallMs: 5000 } },
    }));
    assertAlmostEquals(summarizeTurns(heavyPrefill).plain.hostMsPerToken, 2);
    assertAlmostEquals(summarizeTurns(heavyPrefill).auto.hostMsPerToken, 4);
  });

  await t.step("plain には cycle 由来の欄もゲートの欄も出さない", () => {
    assertEquals(summary.plain.tokensPerCycle, undefined);
    assertEquals(summary.plain.k, undefined);
    assertEquals(summary.plain.cycleMs, undefined);
    assertEquals(summary.plain.plainSteps, undefined);
  });

  await t.step("always は cycle 基準の 3 つと受理ヒストグラムの合計を出す", () => {
    assertAlmostEquals(summary.always.tokensPerCycle ?? Number.NaN, 2.2);
    // 50/5 = 10・60/5 = 12・70/5 = 14 → 中央値 12
    assertAlmostEquals(summary.always.cycleMs ?? Number.NaN, 12);
    // (50−40)/5 = 2・(60−40)/5 = 4・(70−40)/5 = 6 → 中央値 4
    assertAlmostEquals(summary.always.hostMsPerCycle ?? Number.NaN, 4);
    assertEquals(summary.always.acceptedHistogram, [3, 6, 6]);
    assertEquals(summary.always.plainSteps, undefined);
    assertEquals(summary.always.switches, undefined);
  });

  await t.step("auto はゲートの 2 欄と投機 cycle だけの tok/cycle を出す", () => {
    assertEquals(summary.auto.plainSteps, 2);
    assertEquals(summary.auto.switches, 1);
    // 9 / 4（plain step の 2 token は cycle の勘定に入らない）
    assertAlmostEquals(summary.auto.tokensPerCycle ?? Number.NaN, 2.25);
    assertEquals(summary.auto.acceptedHistogram, [3, 6, 3]);
  });

  await t.step("auto には cycle を分母にした壁を出さない（plain step が混ざる）", () => {
    assertEquals(summary.auto.cycleMs, undefined);
    assertEquals(summary.auto.hostMsPerCycle, undefined);
  });

  await t.step("run 別は本数の中央値と 1 本あたりの壁（本数 0 の種別は欄ごと無い）", () => {
    assertEquals(summary.plain.runs.prefill, { countPerTurn: 1, msPerRun: 50 });
    assertEquals(summary.plain.runs.decode.countPerTurn, 10);
    assertAlmostEquals(summary.plain.runs.decode.msPerRun ?? Number.NaN, 10);
    assertEquals(summary.plain.runs.draft, { countPerTurn: 0 });
    assertEquals(summary.always.runs.draft.countPerTurn, 5);
    assertAlmostEquals(summary.always.runs.verify.msPerRun ?? Number.NaN, 6);
    assertEquals(summary.always.runs.decode, { countPerTurn: 0 });
    // auto の plain step は decode として上がる（2 本 · 6 ms → 3 ms/run）。
    assertEquals(summary.auto.runs.decode.countPerTurn, 2);
    assertAlmostEquals(summary.auto.runs.decode.msPerRun ?? Number.NaN, 3);
  });
});

Deno.test("実効 k は受理数ヒストグラムの長さから出す", async (t) => {
  /** 3 段の drafter の勘定（長さ `k+1` = 4）— cycle 数と確定数は素の合成ターンと同じに保つ。 */
  const threeStep: GenerationSpeculation = {
    cycles: 5,
    draftRuns: 5,
    drafted: 15,
    accepted: 6,
    delivered: 11,
    acceptedHistogram: [1, 2, 1, 1],
  };
  const withThreeStep = (): TurnRecord[] =>
    measuredTurns().map((turn) =>
      turn.mode === "always" ? { ...turn, speculation: threeStep } : turn
    );

  await t.step("長さ 4 なら k=3（合成の既定は長さ 3 で k=2）", () => {
    assertEquals(summarizeTurns(measuredTurns()).always.k, 2);
    assertEquals(summarizeTurns(withThreeStep()).always.k, 3);
  });

  await t.step("auto も同じ口から出す（合成は長さ 3 で k=2）", () => {
    assertEquals(summarizeTurns(measuredTurns()).auto.k, 2);
  });

  await t.step("投機ターンの間で長さが食い違えば落ちる", () => {
    const turns = measuredTurns();
    const [first, ...rest] = turns.filter((turn) => turn.mode === "always");
    assertThrows(
      () =>
        summarizeTurns([
          ...withoutMode(turns, "always"),
          { ...first, speculation: threeStep },
          ...rest,
        ]),
      Error,
      "実効 k が食い違う",
    );
  });
});

Deno.test("暖機は要約に入らない", async (t) => {
  await t.step("暖機の壁を 1000 倍にしても中央値も倍率も動かない（フォールト注入）", () => {
    const baseline = summarizeTurns(measuredTurns());
    const withWarmup = summarizeTurns([
      plainTurn(0, 120000, 100000, { warmup: true, round: 0 }),
      alwaysTurn(0, 60000, { draft: 10000, verify: 30000 }, { warmup: true, round: 0 }),
      autoTurn(0, 80000, { draft: 10000, verify: 24000, decode: 6000 }, {
        warmup: true,
        round: 0,
      }),
      ...measuredTurns(),
    ]);
    assertEquals(withWarmup.plain.msPerToken, baseline.plain.msPerToken);
    assertEquals(withWarmup.always.msPerToken, baseline.always.msPerToken);
    assertEquals(withWarmup.auto.msPerToken, baseline.auto.msPerToken);
    assertEquals(withWarmup.speedup, baseline.speedup);
    assertEquals(withWarmup.speedupAuto, baseline.speedupAuto);
    assertEquals(withWarmup.plain.turns, 3);
    assertEquals(withWarmup.auto.turns, 3);
    assertEquals(withWarmup.always.acceptedHistogram, [3, 6, 6]);
  });

  await t.step("暖機しか無ければ落ちる（暖機だけの要約を出さない）", () => {
    assertThrows(
      () =>
        summarizeTurns([
          plainTurn(0, 100, 80, { warmup: true, round: 0 }),
          alwaysTurn(0, 50, { draft: 10, verify: 30 }, { warmup: true, round: 0 }),
          autoTurn(0, 70, { draft: 10, verify: 24, decode: 6 }, { warmup: true, round: 0 }),
        ]),
      Error,
      "ターンが 1 本も無い",
    );
  });
});

Deno.test("token 列の突合", async (t) => {
  await t.step("全ターン一致なら identical / identicalAuto が立ち添字は出ない", () => {
    const { identity } = summarizeTurns(measuredTurns());
    assertEquals(identity, {
      plainConsistent: true,
      alwaysConsistent: true,
      autoConsistent: true,
      identical: true,
      identicalAuto: true,
    });
  });

  await t.step("always の食い違いは identical 側の最初の添字で報せる", () => {
    const turns = measuredTurns();
    const { identity } = summarizeTurns([
      ...withoutMode(turns, "always"),
      alwaysTurn(1, 50, { draft: 10, verify: 30 }, { ids: [11, 99, 33] }),
    ]);
    assertEquals(identity.identical, false);
    assertEquals(identity.firstDivergence, 1);
    // auto は動いていないので、そちらの突合は立ったまま。
    assertEquals(identity.identicalAuto, true);
    assertEquals(identity.firstDivergenceAuto, undefined);
  });

  await t.step("auto の食い違いは identicalAuto 側だけを落とす", () => {
    const turns = measuredTurns();
    const { identity } = summarizeTurns([
      ...withoutMode(turns, "auto"),
      autoTurn(1, 70, { draft: 10, verify: 24, decode: 6 }, { ids: [11, 22, 44] }),
    ]);
    assertEquals(identity.identical, true);
    assertEquals(identity.identicalAuto, false);
    assertEquals(identity.firstDivergenceAuto, 2);
  });

  await t.step("片方が接頭辞なら短い側の長さを報せる", () => {
    const turns = measuredTurns();
    const { identity } = summarizeTurns([
      ...withoutMode(turns, "always"),
      alwaysTurn(1, 50, { draft: 10, verify: 30 }, { ids: [11, 22] }),
    ]);
    assertEquals(identity.identical, false);
    assertEquals(identity.firstDivergence, 2);
  });

  await t.step("同じモードの中で列が動いたら consistent が落ちる", () => {
    const turns = measuredTurns();
    const { identity } = summarizeTurns([
      plainTurn(1, 100, 80),
      plainTurn(1, 120, 100, { ids: [11, 22, 44] }),
      ...withoutMode(turns, "plain"),
    ]);
    assertEquals(identity.plainConsistent, false);
    assertEquals(identity.alwaysConsistent, true);
    assertEquals(identity.autoConsistent, true);
    // 最初の 1 本ずつは一致しているので `identical` は立つ — 2 つの検査は別物である。
    assertEquals(identity.identical, true);
  });
});

Deno.test("簿記の破れは落とす", async (t) => {
  await t.step("投機のターンに speculation が無ければ落ちる", () => {
    const turns = measuredTurns();
    const { speculation: _tally, ...withoutTally } = alwaysTurn(1, 50, { draft: 10, verify: 30 });
    assertThrows(
      () => summarizeTurns([...withoutMode(turns, "always"), withoutTally]),
      Error,
      "speculation",
    );
  });

  await t.step("auto のターンに plainSteps / switches が無ければ落ちる", () => {
    const turns = measuredTurns();
    /** 片方の欄だけを落とした勘定（`"always"` の勘定が auto の席に紛れた形）。 */
    const gateless = (missing: "plainSteps" | "switches"): TurnRecord[] => {
      const tally: GenerationSpeculation = {
        cycles: 4,
        draftRuns: 4,
        drafted: 12,
        accepted: 5,
        delivered: 9,
        acceptedHistogram: [1, 2, 1],
        ...(missing === "plainSteps" ? { switches: 1 } : { plainSteps: 2 }),
      };
      return [
        ...withoutMode(turns, "auto"),
        autoTurn(1, 70, { draft: 10, verify: 24, decode: 6 }, { speculation: tally }),
      ];
    };
    assertThrows(() => summarizeTurns(gateless("plainSteps")), Error, "plainSteps");
    assertThrows(() => summarizeTurns(gateless("switches")), Error, "switches");
  });

  await t.step("token が 2 個未満のターンは ms/token を出さずに落ちる", () => {
    const turns = measuredTurns();
    assertThrows(
      () => summarizeTurns([plainTurn(1, 100, 80, { tokens: 1 }), ...turns.slice(3)]),
      Error,
      "ms/token",
    );
  });
});

Deno.test("局面別内訳の要約", async (t) => {
  /** auto の 3 ターンだけ内訳をずらす（1 / 2 / 3 倍 → 中央値は 2 倍のターン）。 */
  const scaled = (): TurnRecord[] => [
    ...withoutMode(measuredTurns(), "auto"),
    autoTurn(1, 70, { draft: 10, verify: 24, decode: 6 }, { trace: traceFixture(1) }),
    autoTurn(1, 80, { draft: 10, verify: 24, decode: 6 }, {
      trace: { ...traceFixture(2), firstExitRun: 3 },
    }),
    autoTurn(2, 90, { draft: 10, verify: 24, decode: 6 }, {
      trace: { ...traceFixture(3), firstExitRun: 9 },
    }),
  ];

  await t.step("バケットごとに runs / ms / delivered それぞれの中央値を採る", () => {
    const { trace } = summarizeTurns(scaled()).auto;
    assertEquals(trace.speculate, bucket(8, 80, 32));
    assertEquals(trace.speculateAfterReturn, bucket(2, 22, 6));
    assertEquals(trace.burst, bucket(4, 44, 10));
    assertEquals(trace.w1Probe, bucket(6, 66, 6));
    assertEquals(trace.plain, bucket(10, 110, 10));
    assertEquals(trace.unmeasured, bucket(12, 132, 14));
    assertEquals(trace.cold, bucket(16, 176, 40));
    assertEquals(trace.rest, bucket(26, 278, 38));
  });

  await t.step("firstExitRun は切替が起きたターンだけの中央値（偶数本は上側）", () => {
    // 3 ターンのうち切替は 2 本（3 と 9）→ 上側の 9。起きなかったターンを 0 で埋める実装なら 3。
    assertEquals(summarizeTurns(scaled()).auto.trace.firstExitRun, 9);
  });

  await t.step("切替が 1 度も起きなければ欄ごと無い", () => {
    const summary = summarizeTurns(measuredTurns());
    assertEquals(Object.hasOwn(summary.auto.trace, "firstExitRun"), false);
    assertEquals(Object.hasOwn(summary.plain.trace, "firstExitRun"), false);
  });

  await t.step("内訳は全モードに出る（plain / always も同じ欄を持つ）", () => {
    const summary = summarizeTurns(measuredTurns());
    assertEquals(summary.plain.trace.plain, bucket(5, 55, 5));
    assertEquals(summary.always.trace.speculate, bucket(4, 40, 16));
  });

  await t.step("暖機の内訳は要約に入らない（フォールト注入）", () => {
    const baseline = summarizeTurns(measuredTurns()).auto.trace;
    const withWarmup = summarizeTurns([
      autoTurn(0, 80, { draft: 10, verify: 24, decode: 6 }, {
        warmup: true,
        round: 0,
        trace: traceFixture(1000),
      }),
      ...measuredTurns(),
    ]).auto.trace;
    assertEquals(withWarmup, baseline);
  });
});

Deno.test("warm の要約（自ターン番号で揃える）", async (t) => {
  await t.step(
    "3 モードに揃う番号までしか入らない（plain 6 / always 3 / auto 3 → 各 3 本）",
    () => {
      const summary = summarizeTurns(warmTurns(), { warm: true });
      assertEquals(summary.ownTurnLimit, 3);
      assertEquals([summary.plain.turns, summary.always.turns, summary.auto.turns], [3, 3, 3]);
    },
  );

  await t.step("揃わない番号のターンは中央値を動かさない（フォールト注入）", () => {
    const summary = summarizeTurns(warmTurns(), { warm: true });
    // plain の 1〜3 本目は 10 / 12 / 14 ms/tok。4〜6 本目（100 / 120 / 140）が混ざれば 100 になる。
    assertAlmostEquals(summary.plain.msPerToken, 12);
    assertAlmostEquals(summary.always.msPerToken, 6);
    assertAlmostEquals(summary.auto.msPerToken, 8);
    assertAlmostEquals(summary.speedup, 2);
    assertAlmostEquals(summary.speedupAuto, 1.5);
  });

  await t.step("cold（warm なし）は同じ記録の全ターンを入れる", () => {
    const summary = summarizeTurns(warmTurns());
    assertEquals(Object.hasOwn(summary, "ownTurnLimit"), false);
    assertEquals(summary.plain.turns, 6);
    // 6 本の中央値は上側 = 100 ms/tok（揃える処理が cold へ漏れていないことの裏返し）。
    assertAlmostEquals(summary.plain.msPerToken, 100);
  });

  await t.step("暖機は warm でも要約に入らないが、番号の勘定には入る", () => {
    // 各モードの 1 本目を暖機にすると、揃う番号は 3 のまま・要約に入るのは 2 本ずつになる。
    const turns = warmTurns().map((turn) =>
      turn.ownIndex === 1 ? { ...turn, warmup: true, round: 0 } : turn
    );
    const summary = summarizeTurns(turns, { warm: true });
    assertEquals(summary.ownTurnLimit, 3);
    assertEquals([summary.plain.turns, summary.always.turns, summary.auto.turns], [2, 2, 2]);
    // 残るのは 2 / 3 本目 = 12 / 14 → 上側の 14。
    assertAlmostEquals(summary.plain.msPerToken, 14);
  });

  await t.step("ターンが 1 本も無いモードがあれば落ちる", () => {
    assertThrows(
      () => summarizeTurns(withoutMode(warmTurns(), "auto"), { warm: true }),
      Error,
      "auto",
    );
  });
});

Deno.test("warm の token 列の突合（自ターン番号ごと）", async (t) => {
  /** 指定した番号の always のターンの列を割る。 */
  const brokenAlways = (ownIndex: number, ids: readonly number[]): TurnRecord[] =>
    warmTurns().map((turn) =>
      turn.mode === "always" && turn.ownIndex === ownIndex ? { ...turn, ids } : turn
    );

  await t.step(
    "全番号一致なら identical / identicalAuto だけが立つ（モード内の再現性は問わない）",
    () => {
      const { identity } = summarizeTurns(warmTurns(), { warm: true });
      // warm はターンごとに会話が伸びるので、`plainConsistent` 等は**欄ごと無い**
      // （`false` を書くとビット同一性の破れに読める）。`contextAligned` は warm だけの欄で、
      // この合成記録は番号ごとに context が揃っている（欄の検証は下の Deno.test）。
      assertEquals(identity, { identical: true, identicalAuto: true, contextAligned: true });
    },
  );

  await t.step("2 本目で割れたら番号と位置の両方を報せる", () => {
    const { identity } = summarizeTurns(brokenAlways(2, [11, 24, 99]), { warm: true });
    assertEquals(identity.identical, false);
    assertEquals(identity.firstDivergenceTurn, 2);
    assertEquals(identity.firstDivergence, 2);
    // auto は動いていないので、そちらの突合は立ったまま。
    assertEquals(identity.identicalAuto, true);
    assertEquals(identity.firstDivergenceAuto, undefined);
  });

  await t.step("複数の番号で割れたら最初の番号を報せる", () => {
    const turns = brokenAlways(3, [11, 25, 77]).map((turn) =>
      turn.mode === "always" && turn.ownIndex === 2 ? { ...turn, ids: [11, 24, 99] } : turn
    );
    const { identity } = summarizeTurns(turns, { warm: true });
    assertEquals(identity.firstDivergenceTurn, 2);
  });

  await t.step("番号を無視して並び順で比べる実装は落ちる（plain だけ番号を入れ替える）", () => {
    // plain の 1 本目と 2 本目を入れ替えても、番号で対を作る限り突合は立つ。
    const turns = warmTurns();
    const plain = turns.filter((turn) => turn.mode === "plain");
    const { identity } = summarizeTurns(
      [plain[1], plain[0], ...plain.slice(2), ...withoutMode(turns, "plain")],
      { warm: true },
    );
    assertEquals(identity.identical, true);
    assertEquals(identity.identicalAuto, true);
  });
});

Deno.test("warm の容量の門", async (t) => {
  /** 2 round の台本（plain 9 本・always / auto 5 本ずつ）。 */
  const plans = turnPlan(2);

  await t.step("ピークは最後のターンの発行後（prompt + 追記 ×(N−1) + 生成 ×N − 1）", () => {
    assertEquals(
      warmPeakPositions({ turns: 1, promptTokens: 33, turnTokens: 20, newTokens: 200 }),
      232,
    );
    assertEquals(
      warmPeakPositions({ turns: 3, promptTokens: 33, turnTokens: 20, newTokens: 200 }),
      672,
    );
  });

  await t.step("自ターン数が 1 以上の整数でなければ落ちる", () => {
    assertThrows(
      () => warmPeakPositions({ turns: 0, promptTokens: 33, turnTokens: 20, newTokens: 200 }),
      Error,
      "1 以上の整数",
    );
  });

  await t.step("長文脈（prompt 4,800）は自ターン 9 本で capacity 8192 に入らない", () => {
    assertThrows(
      () =>
        assertWarmCapacity({
          plans,
          capacity: 8192,
          promptTokens: 4800,
          turnTokens: 4800,
          newTokens: 200,
        }),
      Error,
      "plain は自ターン 9 本",
    );
  });

  await t.step("自由文（prompt 33）は既定の容量で通る", () => {
    assertWarmCapacity({
      plans,
      capacity: 8192,
      promptTokens: 33,
      turnTokens: 33,
      newTokens: 200,
    });
  });

  await t.step("境界はちょうど capacity まで通る", () => {
    const needed = warmPeakPositions({
      turns: 9,
      promptTokens: 33,
      turnTokens: 33,
      newTokens: 200,
    });
    const request = { plans, promptTokens: 33, turnTokens: 33, newTokens: 200 };
    assertWarmCapacity({ ...request, capacity: needed });
    assertThrows(() => assertWarmCapacity({ ...request, capacity: needed - 1 }), Error, "plain");
  });

  await t.step("台本に居ないモードがあれば落ちる", () => {
    assertThrows(
      () =>
        assertWarmCapacity({
          plans: plans.filter((plan) => plan.mode !== "auto"),
          capacity: 8192,
          promptTokens: 33,
          turnTokens: 33,
          newTokens: 200,
        }),
      Error,
      "auto",
    );
  });

  await t.step("追記の長さが混ざるときは最長で見る（短い方で見ると溢れを見逃す）", () => {
    // 追記が 10 token と 24 token の 2 本ある走行（`main.ts` が渡すのは最長 + 閉じ札 1 個）。
    const request = { plans, promptTokens: 33, newTokens: 200 };
    const capacity = warmPeakPositions({ ...request, turns: 9, turnTokens: 24 + 1 }) - 1;
    assertThrows(
      () => assertWarmCapacity({ ...request, capacity, turnTokens: 24 + 1 }),
      Error,
      "plain は自ターン 9 本",
    );
    // 短い方で見ると同じ容量が通ってしまう — 最長を渡すことが門の前提である。
    assertWarmCapacity({ ...request, capacity, turnTokens: 10 + 1 });
  });
});

Deno.test("warm の追記列の門（発話が尽きる台本を落とす）", async (t) => {
  await t.step("既定の rounds 3（plain 13 本）は 12 本で通り、11 本では落ちる", () => {
    // 回せる自ターンは 1 + 本数（1 本目は会話全体を流すので追記を使わない）。
    assertWarmFollowUps({ plans: turnPlan(3), followUps: 12 });
    assertThrows(() => assertWarmFollowUps({ plans: turnPlan(3), followUps: 11 }), Error, "plain");
  });

  await t.step("rounds 4（plain 17 本）は 12 本では落ちる（本数を名指す）", () => {
    const request = { plans: turnPlan(4), followUps: 12 };
    assertThrows(() => assertWarmFollowUps(request), Error, "plain は自ターン 17 本");
    assertThrows(() => assertWarmFollowUps(request), Error, "追記の発話 12 本");
  });

  await t.step("本数が 1 以上の整数でなければ落ちる", () => {
    assertThrows(
      () => assertWarmFollowUps({ plans: turnPlan(1), followUps: 0 }),
      Error,
      "1 本以上要る",
    );
    assertThrows(
      () => assertWarmFollowUps({ plans: turnPlan(1), followUps: 1.5 }),
      Error,
      "1 本以上要る",
    );
  });

  await t.step("台本に居ないモードがあれば落ちる", () => {
    assertThrows(
      () =>
        assertWarmFollowUps({
          plans: turnPlan(2).filter((plan) => plan.mode !== "auto"),
          followUps: 12,
        }),
      Error,
      "auto",
    );
  });
});

Deno.test("warm の前置（前ターンの停止から決める）", async (t) => {
  /** 閉じ札の id は 106 として、それ以外の停止 token と混ざらない数を使う。 */
  const request = { mode: "plain", endOfTurnId: 106 } as const;

  await t.step("閉じ札で閉じたターンの後は前置しない（frontier がその id）", () => {
    assertEquals(warmTurnPrefix({ ...request, prior: { reason: "eos", token: 106 } }), []);
    // 要求が足した停止 token の枝（`stop-token`）でも、閉じ札なら同じ扱いである。
    assertEquals(warmTurnPrefix({ ...request, prior: { reason: "stop-token", token: 106 } }), []);
  });

  await t.step("--new-tokens で打ち切ったターンの後は閉じ札を 1 個前置する", () => {
    assertEquals(warmTurnPrefix({ ...request, prior: { reason: "max-tokens" } }), [106]);
  });

  await t.step("消費側が閉じたターンの後も打ち切りと同じ（model turn が開いたまま）", () => {
    assertEquals(warmTurnPrefix({ ...request, prior: { reason: "closed" } }), [106]);
    assertEquals(warmTurnPrefix({ ...request, prior: { reason: "aborted" } }), [106]);
  });

  await t.step("閉じ札以外の停止 token で閉じたターンの後は落ちる（token id を名乗る）", () => {
    // `<eos>` で止まったターン。前置すると `本文 <eos> <turn|> 差分` を KV に積むことになる。
    assertThrows(
      () => warmTurnPrefix({ ...request, prior: { reason: "eos", token: 1 } }),
      Error,
      "token id 1（eos）",
    );
    assertThrows(
      () => warmTurnPrefix({ ...request, prior: { reason: "stop-token", token: 262144 } }),
      Error,
      "token id 262144（stop-token）",
    );
  });

  await t.step("落ちるときはモードも名乗る（3 本の会話のどれが壊れたか）", () => {
    assertThrows(
      () =>
        warmTurnPrefix({
          mode: "auto",
          endOfTurnId: 106,
          prior: { reason: "eos", token: 1 },
        }),
      Error,
      "mode auto",
    );
  });
});

Deno.test("warm の context 長の一致（自ターン番号ごと）", async (t) => {
  /** 指定した番号の auto のターンだけ context 長をずらす（`warmTurns` の既定は 1000 × 番号）。 */
  const shiftedAuto = (ownIndex: number, contextTokens: number): TurnRecord[] =>
    warmTurns().map((turn) =>
      turn.mode === "auto" && turn.ownIndex === ownIndex ? { ...turn, contextTokens } : turn
    );

  await t.step("番号ごとに 3 モードが揃えば contextAligned が立ち、番号の欄は出ない", () => {
    const { identity } = summarizeTurns(warmTurns(), { warm: true });
    assertEquals(identity.contextAligned, true);
    assertEquals(Object.hasOwn(identity, "firstContextMismatchTurn"), false);
  });

  await t.step("auto の 2 本目だけずれたら落ちずに false と番号 2 を報せる", () => {
    const { identity } = summarizeTurns(shiftedAuto(2, 2100), { warm: true });
    assertEquals(identity.contextAligned, false);
    assertEquals(identity.firstContextMismatchTurn, 2);
    // 列そのものは動いていないので、ビット同一性の突合は立ったまま（別の検査である）。
    assertEquals(identity.identical, true);
    assertEquals(identity.identicalAuto, true);
  });

  await t.step("ずれた番号が複数なら最初の番号を報せる", () => {
    const turns = shiftedAuto(3, 3100).map((turn) =>
      turn.mode === "auto" && turn.ownIndex === 2 ? { ...turn, contextTokens: 2100 } : turn
    );
    assertEquals(summarizeTurns(turns, { warm: true }).identity.firstContextMismatchTurn, 2);
  });

  await t.step("要約に入らない番号のずれは見ない（plain の 4 本目は上限の外）", () => {
    const turns = warmTurns().map((turn) =>
      turn.mode === "plain" && turn.ownIndex === 4 ? { ...turn, contextTokens: 42 } : turn
    );
    assertEquals(summarizeTurns(turns, { warm: true }).identity.contextAligned, true);
  });

  await t.step("cold では欄ごと無い（毎ターン新しい sequence なので問いが立たない）", () => {
    const { identity } = summarizeTurns(warmTurns());
    assertEquals(Object.hasOwn(identity, "contextAligned"), false);
    assertEquals(Object.hasOwn(identity, "firstContextMismatchTurn"), false);
  });
});
