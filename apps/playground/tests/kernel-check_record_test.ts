import type { CheckPart } from "@karume/kernel-checks";
import { describe, expect, it } from "vite-plus/test";

import type { RecordEnvironment } from "../src/lib/environment.ts";
import {
  KERNEL_CHECK_RECORD_FORMAT,
  kernelCheckRecord,
} from "../src/models/kernel-check/verdict.ts";

const ENVIRONMENT: RecordEnvironment = {
  recordedAt: "2026-10-28T10:00:00.000Z",
  userAgent: "test",
  crossOriginIsolated: true,
  adapter: { vendor: "nvidia", architecture: "blackwell", device: "", description: "d3d12" },
  device: {
    features: ["shader-f16"],
    wgslLanguageFeatures: [],
    subgroupLanesLinear: false,
    gpuTimingEnabled: false,
    limits: {
      maxComputeWorkgroupStorageSize: 32768,
      maxStorageBufferBindingSize: 2147483644,
      maxBufferSize: 2147483644,
    },
  },
};

const PART: CheckPart = {
  id: "state-pv-unroll",
  title: "Unrolled ③' vs v1",
  cases: [
    { label: "a", criteria: [{ name: "O words", words: 4, failures: 0 }] },
    {
      label: "b",
      criteria: [
        {
          name: "O words",
          words: 4,
          failures: 1,
          failure: { index: 2, expected: 1, actual: 2, at: "first" },
        },
      ],
    },
  ],
  requirements: [{ id: "full-write", name: "no poison left", met: true, detail: "0 poison words" }],
};

describe("kernelCheckRecord", () => {
  const record = kernelCheckRecord(ENVIRONMENT, ["subgroups", "shader-f16"], {
    "state-attention-rewrites": {
      kind: "done",
      seconds: 74.5,
      report: {
        parts: [PART],
        expectedCases: { "state-pv-unroll": 2 },
        mutants: [{ label: "swap", kernel: "pv_par", case: "c", differingWords: 9, words: 16 }],
        expectedMutants: 1,
        foldClasses: [],
      },
    },
    "gemv-subgroup32": { kind: "not-available", reason: "no subgroups" },
  });

  it("starts with the format, the page and the environment, and sorts the adapter features", () => {
    expect(record.format).toBe(KERNEL_CHECK_RECORD_FORMAT);
    expect(record.format).toBe("karume-playground-kernel-check/1");
    expect(record.page).toBe("kernel-check");
    expect(record.adapter).toEqual(ENVIRONMENT.adapter);
    expect(record.adapterFeatures).toEqual(["shader-f16", "subgroups"]);
  });

  it("lists every check in check order: an unavailable one with its reason, a not-run one as such", () => {
    expect(record.checks.map(({ id, verdict }) => [id, verdict])).toEqual([
      ["gemv-subgroup32", "not-available"],
      ["rms-subgroup32", "not-run"],
      ["state-attention-rewrites", "fail"],
      ["gemv-integer", "not-run"],
      ["gemv-planes", "not-run"],
      ["state-kv16", "not-run"],
      ["score-f16-store", "not-run"],
      ["state-bidirectional", "not-run"],
      ["expert-linear", "not-run"],
      ["fusion-twins", "not-run"],
    ]);
    expect(record.checks[1]).toEqual({ id: "rms-subgroup32", verdict: "not-run" });
    expect(record.checks[0]).toEqual({
      id: "gemv-subgroup32",
      verdict: "not-available",
      reason: "no subgroups",
    });
    expect(record.verdict).toBe("fail");
  });

  it("keeps every case with its criteria, the part summary, the mutants and the fold classes", () => {
    const check = record.checks[2];
    if (check?.verdict !== "fail") throw new Error("expected the failing check");
    const [part] = check.parts;
    expect(part?.results).toEqual(PART.cases);
    expect(part?.requirements).toEqual(PART.requirements);
    expect(part?.failingCases).toBe(1);
    expect(part?.mismatchingWords).toBe(1);
    expect(part?.firstFailure?.case).toBe("b");
    expect(check.mutants).toHaveLength(1);
    expect(check.foldClasses).toEqual([]);
    expect(check.productVerdict).toBe("fail");
    expect(check.seconds).toBe(74.5);
  });

  it("names the counts that differ from what the check exports (here the missing mutant matrix)", () => {
    const check = record.checks[2];
    if (check?.verdict !== "fail") throw new Error("expected the failing check");
    expect(check.countMismatches).toEqual(["mutant matrix rows: 0, expected 1"]);
  });

  it("survives JSON (the saved file is the record as is)", () => {
    expect(JSON.parse(JSON.stringify(record))).toEqual(record);
  });
});

describe("kernelCheckRecord of a check with a part the device could not run", () => {
  const record = kernelCheckRecord(ENVIRONMENT, [], {
    "fusion-twins": {
      kind: "done",
      seconds: 3,
      report: {
        parts: [
          { ...PART, id: "fusion-a", cases: [PART.cases[0]] },
          {
            id: "fusion-sg32",
            title: "online sg32",
            cases: [],
            requirements: [],
            notRun: "the online attention takes the wg form on this device",
          },
        ],
        expectedCases: { "fusion-a": 1, "fusion-sg32": 4 },
      },
    },
  });

  it("records the check as incomplete with the part's reason, and the page as incomplete", () => {
    const check = record.checks.find(({ id }) => id === "fusion-twins");
    if (check?.verdict !== "incomplete") throw new Error("expected the incomplete check");
    expect(check.countMismatches).toEqual([]);
    expect(check.parts.map(({ id, cases, notRun }) => [id, cases, notRun])).toEqual([
      ["fusion-a", 1, undefined],
      ["fusion-sg32", 0, "the online attention takes the wg form on this device"],
    ]);
    expect(record.verdict).toBe("incomplete");
  });
});
