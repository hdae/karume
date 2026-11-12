import type { CheckPart, FoldClass, MutantMatrixRow, MutantResult } from "@karume/kernel-checks";
import { describe, expect, it } from "vite-plus/test";

import {
  type CheckOutcome,
  countMismatches,
  foldClassRows,
  matrixRowPasses,
  notRunChecks,
  overallVerdict,
  productVerdictOf,
  subgroupShortfall,
  summarizePart,
  verdictOf,
  withoutOutcomes,
} from "../src/models/kernel-check/verdict.ts";

const cleanPart = (id: string, labels: readonly string[], group?: string): CheckPart => ({
  id,
  title: id,
  cases: labels.map((label) => ({
    label,
    ...(group === undefined ? {} : { group }),
    criteria: [{ name: "words (u32)", words: 16, failures: 0 }],
  })),
  requirements: [{ id: "census", name: "word classes met", met: true, detail: "{}" }],
});

const failingPart: CheckPart = {
  id: "state-qk-fold",
  title: "folded vs ①'",
  cases: [
    {
      label: "e4b-full P=0 M=1",
      group: "e4b-full M=1",
      criteria: [
        { name: "block 0", words: 8, failures: 0 },
        {
          name: "block 1",
          words: 8,
          failures: 3,
          failure: { index: 5, expected: 0x3f800000, actual: 0x3f800001, at: "first" },
        },
      ],
    },
    {
      label: "e4b-full P=17 M=1",
      group: "e4b-full M=1",
      criteria: [
        {
          name: "block 0",
          words: 8,
          failures: 2,
          failure: { index: 0, expected: 0, actual: 1, at: "first" },
        },
      ],
    },
    {
      label: "e2b-full P=0 M=1",
      group: "e2b-full M=1",
      criteria: [{ name: "block 0", words: 8, failures: 0 }],
    },
  ],
  requirements: [{ id: "census", name: "word classes met", met: true, detail: "{}" }],
};

const mutant = (differingWords: number): MutantResult => ({
  label: "swap",
  kernel: "pv_par",
  case: "e2b-full P=4095",
  differingWords,
  words: 8192,
});

const foldClass = (shape: string, productFoldHeads: number, use: FoldClass["use"]): FoldClass => ({
  group: `${shape} M=1`,
  shape,
  kvRepeat: use === "k95" ? 4 : 8,
  depth: 512,
  chunkRows: 1,
  checkedFoldHeads: use === "k95" ? 4 : 8,
  productFoldHeads,
  use,
});

const matrixRow = (overrides: Partial<MutantMatrixRow>): MutantMatrixRow => ({
  label: "swap",
  kernel: "pv_par",
  shape: "e2b-full",
  actsWhen: "live >= 241",
  cases: 84,
  acting: 28,
  actingMatched: [],
  idle: 42,
  idleMismatched: [],
  contraction: 14,
  contractionMismatched: 0,
  ...overrides,
});

describe("summarizePart", () => {
  it("counts the failing cases and words and keeps the first failure in case order", () => {
    const summary = summarizePart(failingPart);
    expect(summary.cases).toBe(3);
    expect(summary.failingCases).toBe(2);
    expect(summary.mismatchingWords).toBe(5);
    expect(summary.firstFailure).toEqual({
      case: "e4b-full P=0 M=1",
      criterion: "block 1",
      failure: { index: 5, expected: 0x3f800000, actual: 0x3f800001, at: "first" },
    });
  });

  it("has no first failure for a clean part", () => {
    const summary = summarizePart(cleanPart("gemv", ["a", "b"]));
    expect(summary.failingCases).toBe(0);
    expect(summary.mismatchingWords).toBe(0);
    expect(summary).not.toHaveProperty("firstFailure");
  });
});

describe("verdictOf", () => {
  it("passes a check whose cases, requirements and mutants are all clean", () => {
    expect(
      verdictOf({
        kind: "done",
        seconds: 1,
        report: {
          parts: [cleanPart("a", ["x"])],
          expectedCases: { a: 1 },
          mutants: [mutant(12)],
          mutantMatrix: [matrixRow({})],
          expectedMutants: 1,
        },
      }),
    ).toBe("pass");
  });

  it("fails on one mismatching word", () => {
    expect(
      verdictOf({
        kind: "done",
        seconds: 1,
        report: { parts: [failingPart], expectedCases: { "state-qk-fold": 3 } },
      }),
    ).toBe("fail");
  });

  it("fails on an unmet requirement even when every word matched", () => {
    const part = cleanPart("a", ["x"]);
    expect(
      verdictOf({
        kind: "done",
        seconds: 1,
        report: {
          parts: [
            {
              ...part,
              requirements: [{ id: "full-write", name: "poison left", met: false, detail: "3" }],
            },
          ],
          expectedCases: { a: 1 },
        },
      }),
    ).toBe("fail");
  });

  it("fails when a mutant matched word for word (the comparison would prove nothing)", () => {
    expect(
      verdictOf({
        kind: "done",
        seconds: 1,
        report: {
          parts: [cleanPart("a", ["x"])],
          expectedCases: { a: 1 },
          mutants: [mutant(12), mutant(0)],
          mutantMatrix: [matrixRow({}), matrixRow({})],
          expectedMutants: 2,
        },
      }),
    ).toBe("fail");
  });

  it("fails a part that ran fewer cases than the check exports, though every word it compared matched", () => {
    const report = {
      parts: [cleanPart("gemv-subgroup32", [])],
      expectedCases: { "gemv-subgroup32": 180 },
    };
    expect(verdictOf({ kind: "done", seconds: 1, report })).toBe("fail");
    expect(countMismatches(report)).toEqual(["gemv-subgroup32: 0 cases, expected 180"]);
  });

  it("fails a report without the parts the check exports counts for, or with a part it does not", () => {
    const report = {
      parts: [cleanPart("rms-rows", ["x"]), cleanPart("rms-extra", ["y"])],
      expectedCases: { "rms-rows": 1, "rms-fusion": 12 },
    };
    expect(verdictOf({ kind: "done", seconds: 1, report })).toBe("fail");
    expect(verdictOf({ kind: "done", seconds: 1, report: { ...report, parts: [] } })).toBe("fail");
    expect(countMismatches(report)).toEqual([
      "rms-extra: not a part of this check",
      "rms-fusion: missing, expected 12 cases",
    ]);
  });

  it("fails when the mutants or the mutant matrix rows fall short of the exported count", () => {
    const report = {
      parts: [cleanPart("a", ["x"])],
      expectedCases: { a: 1 },
      mutants: [mutant(12)],
      mutantMatrix: [],
      expectedMutants: 5,
    };
    expect(verdictOf({ kind: "done", seconds: 1, report })).toBe("fail");
    expect(countMismatches(report)).toEqual([
      "mutants: 1, expected 5",
      "mutant matrix rows: 0, expected 5",
    ]);
  });

  it("keeps a check that could not run or threw apart from a pass", () => {
    expect(verdictOf({ kind: "not-available", reason: "no subgroups" })).toBe("not-available");
    expect(verdictOf({ kind: "error", message: "validation", seconds: 0 })).toBe("error");
  });
});

const passed: CheckOutcome = {
  kind: "done",
  seconds: 1,
  report: { parts: [cleanPart("a", ["x"])], expectedCases: { a: 1 } },
};
const failed: CheckOutcome = {
  kind: "done",
  seconds: 1,
  report: { parts: [failingPart], expectedCases: { "state-qk-fold": 3 } },
};

describe("overallVerdict", () => {
  it("is PASS only when every check of the page ran and passed", () => {
    expect(
      overallVerdict({
        "gemv-subgroup32": passed,
        "rms-subgroup32": passed,
        "rms-norm-prefetch": passed,
        "state-attention-rewrites": passed,
        "gemv-integer": passed,
        "gemv-planes": passed,
        "state-kv16": passed,
        "score-f16-store": passed,
        "state-bidirectional": passed,
        "expert-linear": passed,
        "fusion-twins": passed,
        "attention-online-tiled": passed,
      }),
    ).toBe("pass");
    expect(overallVerdict({})).toBe("none");
  });

  it("is not a pass when only some checks were run, even if each of them passed", () => {
    expect(overallVerdict({ "state-attention-rewrites": passed })).toBe("incomplete");
    expect(notRunChecks({ "state-attention-rewrites": passed })).toEqual([
      "gemv-subgroup32",
      "rms-subgroup32",
      "rms-norm-prefetch",
      "gemv-integer",
      "gemv-planes",
      "state-kv16",
      "score-f16-store",
      "state-bidirectional",
      "expert-linear",
      "fusion-twins",
      "attention-online-tiled",
    ]);
  });

  it("is not a pass when a check was not available", () => {
    expect(
      overallVerdict({
        "gemv-subgroup32": { kind: "not-available", reason: "no subgroups" },
        "rms-subgroup32": passed,
        "rms-norm-prefetch": passed,
        "state-attention-rewrites": passed,
        "gemv-integer": passed,
        "gemv-planes": passed,
        "state-kv16": passed,
        "score-f16-store": passed,
        "state-bidirectional": passed,
        "expert-linear": passed,
        "fusion-twins": passed,
        "attention-online-tiled": passed,
      }),
    ).toBe("incomplete");
  });

  it("fails on any failure or error, whatever else is missing", () => {
    expect(
      overallVerdict({
        "gemv-subgroup32": { kind: "not-available", reason: "no subgroups" },
        "state-attention-rewrites": failed,
      }),
    ).toBe("fail");
    expect(
      overallVerdict({ "rms-subgroup32": { kind: "error", message: "validation", seconds: 0 } }),
    ).toBe("fail");
  });
});

describe("a part the device cannot run (notRun)", () => {
  const notRun: CheckPart = {
    id: "fusion-sg32",
    title: "online sg32",
    cases: [],
    requirements: [],
    notRun: "the online attention takes the wg form on this device",
  };
  const report = {
    parts: [cleanPart("fusion-a", ["x"]), notRun],
    expectedCases: { "fusion-a": 1, "fusion-sg32": 4 },
  };

  it("is no count mismatch, and the summary keeps its reason", () => {
    expect(countMismatches(report)).toEqual([]);
    expect(summarizePart(notRun).notRun).toBe(
      "the online attention takes the wg form on this device",
    );
    expect(summarizePart(cleanPart("fusion-a", ["x"]))).not.toHaveProperty("notRun");
  });

  it("makes a check whose other parts pass incomplete, never a pass", () => {
    expect(verdictOf({ kind: "done", seconds: 1, report })).toBe("incomplete");
  });

  it("still fails the check on a failure in a part that ran", () => {
    expect(
      verdictOf({
        kind: "done",
        seconds: 1,
        report: {
          parts: [failingPart, notRun],
          expectedCases: { "state-qk-fold": 3, "fusion-sg32": 4 },
        },
      }),
    ).toBe("fail");
  });

  it("keeps the page from PASS when every other check passed", () => {
    expect(
      overallVerdict({
        "gemv-subgroup32": passed,
        "rms-subgroup32": passed,
        "rms-norm-prefetch": passed,
        "state-attention-rewrites": passed,
        "gemv-integer": passed,
        "gemv-planes": passed,
        "state-kv16": passed,
        "score-f16-store": passed,
        "state-bidirectional": passed,
        "expert-linear": passed,
        "fusion-twins": { kind: "done", seconds: 1, report },
        "attention-online-tiled": passed,
      }),
    ).toBe("incomplete");
  });
});

describe("withoutOutcomes", () => {
  const everyCheckPassed = {
    "gemv-subgroup32": passed,
    "rms-subgroup32": passed,
    "rms-norm-prefetch": passed,
    "state-attention-rewrites": passed,
    "gemv-integer": passed,
    "gemv-planes": passed,
    "state-kv16": passed,
    "score-f16-store": passed,
    "state-bidirectional": passed,
    "expert-linear": passed,
    "fusion-twins": passed,
    "attention-online-tiled": passed,
  };

  it("makes a check about to re-run read as not run, so cancelling it leaves INCOMPLETE, not its earlier PASS", () => {
    expect(overallVerdict(everyCheckPassed)).toBe("pass");
    const rerunning = withoutOutcomes(everyCheckPassed, ["rms-subgroup32"]);
    expect(overallVerdict(rerunning)).toBe("incomplete");
    expect(notRunChecks(rerunning)).toEqual(["rms-subgroup32"]);
  });

  it("keeps the outcomes of the checks the run leaves alone", () => {
    expect(
      withoutOutcomes({ ...everyCheckPassed, "gemv-subgroup32": failed }, ["rms-subgroup32"]),
    ).toEqual({
      "gemv-subgroup32": failed,
      "rms-norm-prefetch": passed,
      "state-attention-rewrites": passed,
      "gemv-integer": passed,
      "gemv-planes": passed,
      "state-kv16": passed,
      "score-f16-store": passed,
      "state-bidirectional": passed,
      "expert-linear": passed,
      "fusion-twins": passed,
      "attention-online-tiled": passed,
    });
  });
});

describe("matrixRowPasses", () => {
  it("passes when every acting case mismatched and no idle case did, whatever the contraction-only cases did", () => {
    expect(matrixRowPasses(matrixRow({}))).toBe(true);
    expect(matrixRowPasses(matrixRow({ contractionMismatched: 14 }))).toBe(true);
  });

  it("fails when an acting case matched (the comparison was vacuous there) or no case acted", () => {
    expect(matrixRowPasses(matrixRow({ actingMatched: ["e2b-full P=4095 M=1"] }))).toBe(false);
    expect(matrixRowPasses(matrixRow({ acting: 0 }))).toBe(false);
  });

  it("fails when an idle case mismatched (the classification is wrong)", () => {
    expect(matrixRowPasses(matrixRow({ idleMismatched: ["e2b-full P=0 M=1"] }))).toBe(false);
  });

  it("fails the check through verdictOf", () => {
    expect(
      verdictOf({
        kind: "done",
        seconds: 1,
        report: {
          parts: [cleanPart("a", ["x"])],
          expectedCases: { a: 1 },
          mutants: [mutant(12)],
          mutantMatrix: [matrixRow({ actingMatched: ["c"] })],
          expectedMutants: 1,
        },
      }),
    ).toBe("fail");
  });
});

describe("productVerdictOf", () => {
  const withClasses = (
    parts: readonly CheckPart[],
    mutants: readonly MutantResult[] = [],
  ): CheckOutcome => ({
    kind: "done",
    seconds: 1,
    report: {
      parts,
      expectedCases: Object.fromEntries(parts.map(({ id, cases }) => [id, cases.length])),
      mutants,
      expectedMutants: mutants.length,
      foldClasses: [foldClass("e2b-full", 8, "product"), foldClass("e4b-full", 1, "k95")],
    },
  });

  it("passes when the failures are confined to the classes kept for K-95", () => {
    const onlyK95: CheckPart = {
      ...failingPart,
      cases: failingPart.cases.filter(({ group }) => group === "e4b-full M=1"),
    };
    expect(verdictOf(withClasses([onlyK95]))).toBe("fail");
    expect(productVerdictOf(withClasses([onlyK95]))).toBe("pass");
  });

  it("fails on a failure in a class the product folds, in a part without classes or in a mutant", () => {
    const inProduct: CheckPart = {
      ...failingPart,
      cases: failingPart.cases.map((testCase) => ({ ...testCase, group: "e2b-full M=1" })),
    };
    expect(productVerdictOf(withClasses([inProduct]))).toBe("fail");
    const ungrouped: CheckPart = {
      ...failingPart,
      cases: failingPart.cases.map(({ label, criteria }) => ({ label, criteria })),
    };
    expect(productVerdictOf(withClasses([ungrouped]))).toBe("fail");
    expect(productVerdictOf(withClasses([cleanPart("a", ["x"])], [mutant(0)]))).toBe("fail");
  });

  it("fails on a case count that differs from the check's, counted before the K-95 classes are left out", () => {
    const onlyK95: CheckPart = {
      ...failingPart,
      cases: failingPart.cases.filter(({ group }) => group === "e4b-full M=1"),
    };
    const outcome = withClasses([onlyK95]);
    if (outcome.kind !== "done") throw new Error("expected a done outcome");
    expect(productVerdictOf(outcome)).toBe("pass");
    expect(
      productVerdictOf({
        ...outcome,
        report: { ...outcome.report, expectedCases: { "state-qk-fold": 252 } },
      }),
    ).toBe("fail");
  });

  it("is absent for a check without fold classes", () => {
    expect(productVerdictOf(passed)).toBeUndefined();
    expect(productVerdictOf({ kind: "not-available", reason: "no subgroups" })).toBeUndefined();
  });
});

describe("subgroupShortfall", () => {
  it("names the adapter features and the WGSL feature that are missing", () => {
    expect(subgroupShortfall(new Set(["subgroups"]), new Set())).toEqual([
      "subgroup-size-control",
      "WGSL subgroup_id",
    ]);
    expect(
      subgroupShortfall(new Set(["subgroups", "subgroup-size-control"]), new Set(["subgroup_id"])),
    ).toEqual([]);
  });
});

describe("foldClassRows", () => {
  it("labels the classes the product folds, the K-95 class and the rest, with their failing cases", () => {
    const rows = foldClassRows({
      parts: [failingPart],
      expectedCases: { "state-qk-fold": 3 },
      foldClasses: [
        foldClass("e2b-full", 8, "product"),
        foldClass("e4b-full", 1, "k95"),
        foldClass("minicpm5-full", 1, "product"),
      ],
    });
    expect(rows.map(({ status, cases, failingCases }) => [status, cases, failingCases])).toEqual([
      ["in use", 1, 0],
      ["kept for K-95", 2, 2],
      ["not selected", 0, 0],
    ]);
  });
});
