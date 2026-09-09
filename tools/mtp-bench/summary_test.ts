/**
 * 要約の計算の単体検証（純関数だけ・GPU 不要）。
 *
 * 見る 6 点:
 *
 * 1. **ABBA の台本** — 暖機 2 本の後に plain / spec / spec / plain が `rounds` 回
 * 2. **分母の 2 本**（`main.ts` の進捗行も呼ぶ口）— `tokensAfterFirst` / `tokensPerCycle`
 * 3. **中央値と派生量** — `msPerToken` / `hostMsPerToken` / `tokensPerCycle` / `cycleMs` /
 *    `hostMsPerCycle` / `speedup` / run 別の本数と 1 本あたりの壁
 * 4. **実効 k** — 受理数ヒストグラムの長さ − 1・ターン間で食い違えば落ちる
 * 5. **暖機は要約に入らない** — フォールト注入（暖機の値を 1000 倍にしても中央値が動かない）
 * 6. **token 列の突合** — 一致 / 不一致の最初の添字 / 接頭辞
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

/** 合成ターン（投機）: 5 cycle・受理 6 個 = 11 token（`tokensPerCycle` = 2.2）。 */
const specTurn = (
  round: number,
  generationMs: number,
  runWallMs: { draft: number; verify: number },
  over: Partial<TurnRecord> = {},
): TurnRecord => ({
  mode: "speculative",
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
    acceptedHistogram: [1, 2, 2],
  },
  ...over,
});

/** 3 ターンずつの素の測定（plain 10/12/14 ms/tok・spec 5/6/7 ms/tok → 中央値 12 と 6）。 */
const measuredTurns = (): TurnRecord[] => [
  plainTurn(1, 100, 80),
  plainTurn(1, 120, 100),
  plainTurn(2, 140, 120),
  specTurn(1, 50, { draft: 10, verify: 30 }),
  specTurn(1, 60, { draft: 10, verify: 30 }),
  specTurn(2, 70, { draft: 10, verify: 30 }),
];

Deno.test("ABBA の台本", async (t) => {
  await t.step("暖機 2 本の後に plain / spec / spec / plain を rounds 回", () => {
    const plans = turnPlan(2);
    assertEquals(plans.length, 2 + 4 * 2);
    assertEquals(
      plans.map((plan) => `${plan.mode}#${plan.round}${plan.warmup ? "w" : ""}`),
      [
        "plain#0w",
        "speculative#0w",
        "plain#1",
        "speculative#1",
        "speculative#1",
        "plain#1",
        "plain#2",
        "speculative#2",
        "speculative#2",
        "plain#2",
      ],
    );
  });

  await t.step("暖機は round 0 で warmup が立ち、以後は立たない", () => {
    const plans: readonly TurnPlan[] = turnPlan(1);
    assertEquals(plans.filter((plan) => plan.warmup).map((plan) => plan.round), [0, 0]);
    assert(plans.slice(2).every((plan) => !plan.warmup && plan.round === 1));
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
  /** 5 cycle・受理 6 個の勘定（`specTurn` と同じ形 — 単体で呼ぶために独立に組む）。 */
  const tally = (over: Partial<GenerationSpeculation> = {}): GenerationSpeculation => ({
    cycles: 5,
    draftRuns: 5,
    drafted: 15,
    accepted: 6,
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

  await t.step("tokensPerCycle は (accepted + cycles) / cycles（受理ゼロなら 1）", () => {
    assertAlmostEquals(tokensPerCycle(tally()), 2.2);
    assertAlmostEquals(tokensPerCycle(tally({ accepted: 0, acceptedHistogram: [5, 0, 0] })), 1);
  });

  await t.step("cycles 0 では割れずに落ちる", () => {
    assertThrows(
      () => tokensPerCycle(tally({ cycles: 0, accepted: 0, acceptedHistogram: [0, 0, 0] })),
      Error,
      "cycles 0",
    );
  });
});

Deno.test("要約の派生量", async (t) => {
  const summary = summarizeTurns(measuredTurns());

  await t.step("msPerToken は generationMs / (tokens − 1) の中央値", () => {
    assertAlmostEquals(summary.plain.msPerToken, 12);
    assertAlmostEquals(summary.speculative.msPerToken, 6);
    assertEquals(summary.plain.turns, 3);
    assertEquals(summary.speculative.turns, 3);
  });

  await t.step("speedup は plain / speculative の ms/token 比", () => {
    assertAlmostEquals(summary.speedup, 2);
  });

  await t.step("plain の hostMsPerToken は decode run 壁を引いた残り", () => {
    // (100−80)/10 = 2・(120−100)/10 = 2・(140−120)/10 = 2
    assertAlmostEquals(summary.plain.hostMsPerToken ?? Number.NaN, 2);
    assertEquals(summary.plain.tokensPerCycle, undefined);
  });

  await t.step("speculative は cycle 基準の 3 つと受理ヒストグラムの合計を出す", () => {
    assertAlmostEquals(summary.speculative.tokensPerCycle ?? Number.NaN, 2.2);
    // 50/5 = 10・60/5 = 12・70/5 = 14 → 中央値 12
    assertAlmostEquals(summary.speculative.cycleMs ?? Number.NaN, 12);
    // (50−40)/5 = 2・(60−40)/5 = 4・(70−40)/5 = 6 → 中央値 4
    assertAlmostEquals(summary.speculative.hostMsPerCycle ?? Number.NaN, 4);
    assertEquals(summary.speculative.acceptedHistogram, [3, 6, 6]);
    assertEquals(summary.speculative.hostMsPerToken, undefined);
  });

  await t.step("run 別は本数の中央値と 1 本あたりの壁（本数 0 の種別は欄ごと無い）", () => {
    assertEquals(summary.plain.runs.prefill, { countPerTurn: 1, msPerRun: 50 });
    assertEquals(summary.plain.runs.decode.countPerTurn, 10);
    assertAlmostEquals(summary.plain.runs.decode.msPerRun ?? Number.NaN, 10);
    assertEquals(summary.plain.runs.draft, { countPerTurn: 0 });
    assertEquals(summary.speculative.runs.draft.countPerTurn, 5);
    assertAlmostEquals(summary.speculative.runs.verify.msPerRun ?? Number.NaN, 6);
    assertEquals(summary.speculative.runs.decode, { countPerTurn: 0 });
  });
});

Deno.test("実効 k は受理数ヒストグラムの長さから出す", async (t) => {
  /** 3 段の drafter の勘定（長さ `k+1` = 4）— cycle 数と受理数は素の合成ターンと同じに保つ。 */
  const threeStep: GenerationSpeculation = {
    cycles: 5,
    draftRuns: 5,
    drafted: 15,
    accepted: 6,
    acceptedHistogram: [1, 2, 1, 1],
  };
  const withThreeStep = (): TurnRecord[] =>
    measuredTurns().map((turn) =>
      turn.mode === "speculative" ? { ...turn, speculation: threeStep } : turn
    );

  await t.step("長さ 4 なら k=3（合成の既定は長さ 3 で k=2）", () => {
    assertEquals(summarizeTurns(measuredTurns()).speculative.k, 2);
    assertEquals(summarizeTurns(withThreeStep()).speculative.k, 3);
  });

  await t.step("plain には k を出さない（投機の勘定が無い）", () => {
    assertEquals(summarizeTurns(measuredTurns()).plain.k, undefined);
  });

  await t.step("投機ターンの間で長さが食い違えば落ちる", () => {
    const turns = measuredTurns();
    const [first, ...rest] = turns.filter((turn) => turn.mode === "speculative");
    assertThrows(
      () =>
        summarizeTurns([
          ...turns.filter((turn) => turn.mode === "plain"),
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
      specTurn(0, 60000, { draft: 10000, verify: 30000 }, { warmup: true, round: 0 }),
      ...measuredTurns(),
    ]);
    assertEquals(withWarmup.plain.msPerToken, baseline.plain.msPerToken);
    assertEquals(withWarmup.speculative.msPerToken, baseline.speculative.msPerToken);
    assertEquals(withWarmup.speedup, baseline.speedup);
    assertEquals(withWarmup.plain.turns, 3);
    assertEquals(withWarmup.speculative.acceptedHistogram, [3, 6, 6]);
  });

  await t.step("暖機しか無ければ落ちる（暖機だけの要約を出さない）", () => {
    assertThrows(
      () =>
        summarizeTurns([
          plainTurn(0, 100, 80, { warmup: true, round: 0 }),
          specTurn(0, 50, { draft: 10, verify: 30 }, { warmup: true, round: 0 }),
        ]),
      Error,
      "ターンが 1 本も無い",
    );
  });
});

Deno.test("token 列の突合", async (t) => {
  await t.step("全ターン一致なら identical で firstDivergence は無い", () => {
    const { identity } = summarizeTurns(measuredTurns());
    assertEquals(identity, {
      plainConsistent: true,
      speculativeConsistent: true,
      identical: true,
    });
  });

  await t.step("食い違いは最初の添字で報せる", () => {
    const turns = measuredTurns();
    const { identity } = summarizeTurns([
      ...turns.filter((turn) => turn.mode === "plain"),
      specTurn(1, 50, { draft: 10, verify: 30 }, { ids: [11, 99, 33] }),
    ]);
    assertEquals(identity.identical, false);
    assertEquals(identity.firstDivergence, 1);
  });

  await t.step("片方が接頭辞なら短い側の長さを報せる", () => {
    const turns = measuredTurns();
    const { identity } = summarizeTurns([
      ...turns.filter((turn) => turn.mode === "plain"),
      specTurn(1, 50, { draft: 10, verify: 30 }, { ids: [11, 22] }),
    ]);
    assertEquals(identity.identical, false);
    assertEquals(identity.firstDivergence, 2);
  });

  await t.step("同じモードの中で列が動いたら consistent が落ちる", () => {
    const turns = measuredTurns();
    const { identity } = summarizeTurns([
      plainTurn(1, 100, 80),
      plainTurn(1, 120, 100, { ids: [11, 22, 44] }),
      ...turns.filter((turn) => turn.mode !== "plain"),
    ]);
    assertEquals(identity.plainConsistent, false);
    assertEquals(identity.speculativeConsistent, true);
    // 最初の 1 本ずつは一致しているので `identical` は立つ — 2 つの検査は別物である。
    assertEquals(identity.identical, true);
  });
});

Deno.test("簿記の破れは落とす", async (t) => {
  await t.step("投機のターンに speculation が無ければ落ちる", () => {
    const turns = measuredTurns();
    const { speculation: _tally, ...withoutTally } = specTurn(1, 50, { draft: 10, verify: 30 });
    assertThrows(
      () =>
        summarizeTurns([
          ...turns.filter((turn) => turn.mode === "plain"),
          withoutTally,
        ]),
      Error,
      "speculation",
    );
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
