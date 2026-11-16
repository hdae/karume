import type { CheckPart } from "@karume/kernel-checks";
import { describe, expect, it } from "vite-plus/test";

import kernelCheck from "../src/entries/kernel-check.ts";
import { parseParameters } from "../src/entry.ts";
import { entryVerdict, kernelCheckResult, summaryLines } from "../src/kernel-check/summary.ts";
import {
  type CheckOutcome,
  type CheckOutcomes,
  KERNEL_CHECK_IDS,
} from "../src/kernel-check/verdict.ts";

const cleanPart = (id: string): CheckPart => ({
  id,
  title: id,
  cases: [{ label: "x", criteria: [{ name: "words (u32)", words: 16, failures: 0 }] }],
  requirements: [],
});

const passed: CheckOutcome = {
  kind: "done",
  seconds: 1,
  report: { parts: [cleanPart("a")], expectedCases: { a: 1 } },
};

const failed: CheckOutcome = {
  kind: "done",
  seconds: 2,
  report: {
    parts: [
      cleanPart("gemv-planes-clean"),
      {
        id: "gemv-planes",
        title: "planes",
        cases: [
          {
            label: "K=1536 M=8",
            criteria: [
              {
                name: "f64 band",
                words: 64,
                failures: 2,
                failure: { index: 7, expected: 0x3f800000, actual: 0xbf800000, at: "worst" },
              },
            ],
          },
        ],
        requirements: [],
      },
    ],
    expectedCases: { "gemv-planes-clean": 1, "gemv-planes": 1 },
  },
};

/** Every check passed except the ones given. */
const allPassedExcept = (overrides: CheckOutcomes): CheckOutcomes => ({
  ...Object.fromEntries(KERNEL_CHECK_IDS.map((id) => [id, passed])),
  ...overrides,
});

describe("the kernel-check entry", () => {
  it("declares one choices parameter over the twelve check ids, all by default", () => {
    expect(kernelCheck.id).toBe("kernel-check");
    expect(kernelCheck.parameters).toHaveLength(1);
    expect(parseParameters(kernelCheck.parameters, {})).toEqual({ checks: [...KERNEL_CHECK_IDS] });
    expect(parseParameters(kernelCheck.parameters, { checks: "gemv-integer" })).toEqual({
      checks: ["gemv-integer"],
    });
  });
});

describe("entryVerdict", () => {
  it("maps pass and fail through, and an incomplete run or one that settled nothing to incomplete", () => {
    expect(entryVerdict("pass")).toBe("pass");
    expect(entryVerdict("fail")).toBe("fail");
    expect(entryVerdict("incomplete")).toBe("incomplete");
    expect(entryVerdict("none")).toBe("incomplete");
  });
});

describe("summaryLines", () => {
  it("is one PASS line with the counts when every check passed", () => {
    expect(summaryLines(allPassedExcept({}))).toEqual([
      "PASS — checks: 12 passed, 0 failed, 0 incomplete, 0 not available, 0 error, 0 not run",
    ]);
  });

  it("adds a line per check that is not a pass: the first failing word in hex, not available, error, not run", () => {
    const outcomes: CheckOutcomes = {
      ...allPassedExcept({
        "gemv-planes": failed,
        "gemv-subgroup32": { kind: "not-available", reason: "no subgroups" },
        "state-kv16": { kind: "error", message: "GPUValidationError: bad shader", seconds: 0.5 },
      }),
      "attention-online-tiled": undefined,
    };
    expect(summaryLines(outcomes)).toEqual([
      "FAIL — checks: 8 passed, 1 failed, 0 incomplete, 1 not available, 1 error, 1 not run",
      "gemv-subgroup32: not available",
      "gemv-planes: FAIL — gemv-planes: worst failure in K=1536 M=8 — f64 band — word 7: expected 0x3f800000, actual 0xbf800000",
      "state-kv16: ERROR",
      "attention-online-tiled: not run",
    ]);
  });

  it("names the verdict outside the K-95 classes next to the check that has fold classes", () => {
    const withClasses: CheckOutcome = {
      kind: "done",
      seconds: 1,
      report: {
        parts: [cleanPart("state-qk-fold")],
        expectedCases: { "state-qk-fold": 1 },
        mutants: [],
        mutantMatrix: [],
        expectedMutants: 1,
        foldClasses: [],
      },
    };
    const [, line] = summaryLines(allPassedExcept({ "state-attention-rewrites": withClasses }));
    expect(line).toBe("state-attention-rewrites: FAIL (outside the K-95 classes: FAIL)");
  });

  it("reads a run with no outcome at all as not run everywhere", () => {
    const lines = summaryLines({});
    expect(lines[0]).toBe(
      "not run — checks: 0 passed, 0 failed, 0 incomplete, 0 not available, 0 error, 12 not run",
    );
    expect(lines).toHaveLength(13);
  });
});

describe("kernelCheckResult", () => {
  it("carries the contract verdict, the summary and the record body with every check", () => {
    const result = kernelCheckResult(["subgroups"], { "gemv-integer": passed });
    expect(result.verdict).toBe("incomplete");
    expect(result.summary[0]).toContain("INCOMPLETE");
    expect(result.data).toMatchObject({ adapterFeatures: ["subgroups"], verdict: "incomplete" });
    expect(kernelCheckResult([], allPassedExcept({})).verdict).toBe("pass");
    expect(kernelCheckResult([], {}).verdict).toBe("incomplete");
  });
});
