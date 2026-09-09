/**
 * 要約の計算の単体検証（純関数だけ・GPU 不要）。
 *
 * 見る 6 点:
 *
 * 1. **ローテーションの台本** — 暖機 3 本の後に P S S P P A A P が `rounds` 回
 * 2. **分母の 2 本**（`main.ts` の進捗行も呼ぶ口）— `tokensAfterFirst` / `tokensPerCycle`
 *    （正本は `delivered / cycles`）
 * 3. **中央値と派生量** — `msPerToken` / `hostMsPerToken`（3 モード）/ `tokensPerCycle` /
 *    `cycleMs` / `hostMsPerCycle`（always だけ）/ `plainSteps` / `switches`（auto だけ）/
 *    `speedup` / `speedupAuto` / run 別の本数と 1 本あたりの壁
 * 4. **実効 k** — 受理数ヒストグラムの長さ − 1・ターン間で食い違えば落ちる
 * 5. **暖機は要約に入らない** — フォールト注入（暖機の値を 1000 倍にしても中央値が動かない）
 * 6. **token 列の突合** — always / auto それぞれの一致・不一致の最初の添字・接頭辞
 *
 * NOTE: リポの慣習に合わせて `Deno.test`（文脈）+ `t.step`（振る舞い）で書く。
 */

import { assert, assertAlmostEquals, assertEquals, assertThrows } from "@std/assert";
import type { GenerationSpeculation } from "../../packages/models/gemma.ts";
import {
  median,
  summarizeTurns,
  tokensAfterFirst,
  tokensPerCycle,
  type TurnPlan,
  turnPlan,
  type TurnRecord,
} from "./summary.ts";

/** 合成ターン: 生成 10 token 相当（`tokens - 1 = 10`）で、run 壁だけを与える。 */
const IDS: readonly number[] = [11, 22, 33];

const plainTurn = (
  round: number,
  generationMs: number,
  decodeWallMs: number,
  over: Partial<TurnRecord> = {},
): TurnRecord => ({
  mode: "plain",
  round,
  warmup: false,
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

Deno.test("ローテーションの台本", async (t) => {
  await t.step("暖機 3 本の後に P S S P P A A P を rounds 回", () => {
    const plans = turnPlan(1);
    assertEquals(plans.length, 3 + 8);
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
      ],
    );
  });

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
