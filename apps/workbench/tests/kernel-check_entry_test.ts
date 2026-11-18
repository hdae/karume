import { describe, expect, it } from "vite-plus/test";

import kernelCheck from "../src/entries/kernel-check.ts";
import { parseParameters } from "../src/entry.ts";
import { entryVerdict, kernelCheckResult, summaryLines } from "../src/kernel-check/summary.ts";
import {
  type CheckOutcome,
  type CheckOutcomes,
  KERNEL_CHECK_IDS,
} from "../src/kernel-check/verdict.ts";

import { allPassedExcept, cleanPart, passed } from "./helpers/kernel-check-outcomes.ts";

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

describe("the kernel-check entry", () => {
  it("declares one choices parameter over the thirteen check ids, all by default", () => {
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
      "PASS — checks: 13 passed, 0 failed, 0 incomplete, 0 not available, 0 error, 0 not run",
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
      "FAIL — checks: 9 passed, 1 failed, 0 incomplete, 1 not available, 1 error, 1 not run",
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
      "not run — checks: 0 passed, 0 failed, 0 incomplete, 0 not available, 0 error, 13 not run",
    );
    expect(lines).toHaveLength(14);
  });
});

describe("kernelCheckResult", () => {
  it("carries the contract verdict, the summary and the record body with every check", () => {
    const result = kernelCheckResult(["subgroups"], { "gemv-integer": passed });
    expect(result.verdict).toBe("incomplete");
    expect(result.summary[0]).toContain("INCOMPLETE");
    expect(result.data).toMatchObject({ adapterFeatures: ["subgroups"], verdict: "incomplete" });
  });
});
