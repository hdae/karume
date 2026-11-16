/** Check outcomes for the kernel-check entry's tests: a clean part, a passed check, a full run. */

import type { CheckPart } from "@karume/kernel-checks";

import {
  type CheckOutcome,
  type CheckOutcomes,
  KERNEL_CHECK_IDS,
} from "../../src/kernel-check/verdict.ts";

/** A part whose every case matched word for word and whose one requirement was met. */
export const cleanPart = (
  id: string,
  labels: readonly string[] = ["x"],
  group?: string,
): CheckPart => ({
  id,
  title: id,
  cases: labels.map((label) => ({
    label,
    ...(group === undefined ? {} : { group }),
    criteria: [{ name: "words (u32)", words: 16, failures: 0 }],
  })),
  requirements: [{ id: "census", name: "word classes met", met: true, detail: "{}" }],
});

/** A check that ran its one clean part, with the case count it exports. */
export const passed: CheckOutcome = {
  kind: "done",
  seconds: 1,
  report: { parts: [cleanPart("a")], expectedCases: { a: 1 } },
};

/** Every check passed except the ones given. */
export const allPassedExcept = (overrides: CheckOutcomes): CheckOutcomes => ({
  ...Object.fromEntries(KERNEL_CHECK_IDS.map((id) => [id, passed])),
  ...overrides,
});
