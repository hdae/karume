import { judgeSessionAb, type SessionAbJudgement } from "@karume/runtime/tune";
import { describe, expect, it } from "vite-plus/test";

import { armRow, judgementView } from "../src/models/session-ab/view.ts";

import { abRecord, DIFFERING, RULE, steady } from "./helpers/session-ab-records.ts";

const judged = (record: ReturnType<typeof abRecord>): SessionAbJudgement => {
  const [judgement] = judgeSessionAb(record, { minGain: 0.01 });
  if (judgement === undefined) throw new Error("no judgement");
  return judgement;
};

describe("what the session A/B page shows of a judgement", () => {
  it("adopt: the label, the default badge and the numbers behind it", () => {
    const judgement = judged(abRecord({ base: steady(100), rule: steady(90) }));
    expect(judgement.verdict).toBe("adopt");
    const view = judgementView(judgement);
    expect([view.arm, view.base, view.label, view.tone]).toEqual([
      "rule",
      "base",
      "Adopt",
      "default",
    ]);
    expect(Object.fromEntries(view.numbers)).toMatchObject({
      "Identical to the base": "yes",
      "Median paired difference (rule − base)": "−10.000 ms",
      Gain: "10.00%",
      "Paired differences per round": "−10.000 ms, −10.000 ms, −10.000 ms",
    });
    expect(view.reason).toBe(judgement.reason);
  });

  it("reject when slower: the destructive badge, a positive difference and the judge's reason", () => {
    const judgement = judged(abRecord({ base: steady(100), rule: steady(105) }));
    const view = judgementView(judgement);
    expect([view.label, view.tone]).toEqual(["Reject", "destructive"]);
    expect(Object.fromEntries(view.numbers)["Gain"]).toBe("-5.00%");
    expect(view.reason).toMatch(/^slower than the base/);
  });

  it("reject when the tokens differ, whatever the times: identity is shown as no", () => {
    const judgement = judged(
      abRecord({ base: steady(100), rule: steady(90), ruleIdentity: DIFFERING }),
    );
    const view = judgementView(judgement);
    expect([view.label, Object.fromEntries(view.numbers)["Identical to the base"]]).toEqual([
      "Reject",
      "no",
    ]);
    expect(view.reason).toMatch(/^identity is required/);
  });

  it("inconclusive inside the noise floor: the secondary badge", () => {
    const judgement = judged(
      abRecord({
        base: [
          [100, 104],
          [100, 104],
          [100, 104],
        ],
        rule: [
          [101, 103],
          [100, 104],
          [101, 103],
        ],
      }),
    );
    expect([judgementView(judgement).label, judgementView(judgement).tone]).toEqual([
      "Inconclusive",
      "secondary",
    ]);
  });
});

describe("what the session A/B page shows of an arm", () => {
  it("an identical arm with counts: its runs, the rule's hits (0 when its counter is absent) and dispatches", () => {
    const [base, rule] = abRecord({ base: steady(100), rule: steady(90) }).arms;
    expect(armRow(base, RULE)).toEqual({
      name: "base",
      identity: "identical (20 runs)",
      identical: true,
      hits: "0",
      dispatches: "1294",
      rounds: ["100.0 ms / 100.1 ms", "100.1 ms / 100.0 ms", "100.0 ms / 100.1 ms"],
    });
    expect([armRow(rule, RULE).hits, armRow(rule, RULE).dispatches]).toEqual(["60", "1234"]);
  });

  it("a differing arm: how many words of which run, and the first", () => {
    const [, rule] = abRecord({
      base: steady(100),
      rule: steady(90),
      ruleIdentity: DIFFERING,
    }).arms;
    expect(armRow(rule, RULE)).toMatchObject({
      identical: false,
      identity:
        "3 of 96 words differ (round 2, visit 2, run 2; first at tokens[41] (word): expected 100, got 200)",
    });
  });

  it("an arm without counts shows dashes, not zeros", () => {
    const [base] = abRecord({ base: steady(100), rule: steady(90), noCounts: true }).arms;
    expect([armRow(base, RULE).hits, armRow(base, RULE).dispatches]).toEqual(["—", "—"]);
  });
});
