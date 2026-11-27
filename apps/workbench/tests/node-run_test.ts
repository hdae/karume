import { describe, expect, it } from "vite-plus/test";
import { ParameterError } from "@karume/sweep-app-support/parameters";
import { buildRecord, type EntryResult, type Settlement } from "@karume/sweep-app-support/record";

import { type CommandEnding, exitCodeOf, parseCommandLine, UsageError } from "../node-run.ts";
import type { StepState } from "../src/batch-run.ts";
import type { Entry } from "../src/entry.ts";

const ENTRY: Entry = {
  id: "probe",
  title: "Probe",
  description: "A test entry.",
  parameters: [
    { kind: "integer", name: "rounds", label: "Rounds", min: 1, max: 9, default: 3 },
    { kind: "text", name: "arms", label: "Arms", default: "" },
  ],
  run: () => Promise.resolve({ summary: [], data: null }),
};
const ENTRIES: readonly Entry[] = [ENTRY];

const thrown = (args: readonly string[]): unknown => {
  try {
    parseCommandLine(args, ENTRIES);
  } catch (error) {
    return error;
  }
  throw new Error(`${args.join(" ")} parsed`);
};

describe("parseCommandLine", () => {
  it.each([[["--list"]], [["probe", "--list"]], [["--list", "no-such-entry"]]])(
    "lists on %j",
    (args) => {
      expect(parseCommandLine(args, ENTRIES)).toEqual({ kind: "list" });
    },
  );

  it("runs the named entry with the typed values of --name=value (the first = splits)", () => {
    expect(parseCommandLine(["probe", "--rounds=4", "--arms=a=b,c"], ENTRIES)).toEqual({
      kind: "run",
      entry: ENTRY,
      parameters: { rounds: 4, arms: "a=b,c" },
    });
    expect(parseCommandLine(["probe"], ENTRIES)).toEqual({
      kind: "run",
      entry: ENTRY,
      parameters: { rounds: 3, arms: "" },
    });
  });

  it.each([
    [[], UsageError, "Name an entry"],
    [["--rounds=4"], UsageError, "Name an entry"],
    [["probe", "rounds=4"], UsageError, "rounds=4 is not --name=value"],
    [["no-such-entry"], UsageError, "No entry no-such-entry; the entries are probe"],
    [["probe", "--nope=1"], ParameterError, "Parameter nope"],
    [["probe", "--rounds=10"], ParameterError, "Parameter rounds"],
  ])("refuses %j", (args, kind, message) => {
    const error = thrown(args);
    expect(error).toBeInstanceOf(kind);
    expect(error).toHaveProperty("message", expect.stringContaining(message));
  });
});

const RESULT: EntryResult = { summary: [], data: null };

const recorded = (settlement: Settlement): CommandEnding => ({
  kind: "recorded",
  record: buildRecord({
    entry: { id: "probe", parameters: {} },
    endedAt: new Date("2026-11-16T09:30:12.345Z"),
    milliseconds: 1,
    environment: {
      host: "node",
      userAgent: "test",
      adapter: { vendor: "", architecture: "", device: "", description: "" },
      features: [],
      limits: {},
      wgslLanguageFeatures: [],
    },
    checkout: { commit: "0123abcd", dirty: false, runtimeVersion: "0.22.0" },
    settlement,
  }),
});

describe("exitCodeOf", () => {
  it.each<[string, CommandEnding, number]>([
    [
      "done, verdict pass",
      recorded({ kind: "resolved", result: { ...RESULT, verdict: "pass" }, aborted: false }),
      0,
    ],
    [
      "done, verdict incomplete",
      recorded({ kind: "resolved", result: { ...RESULT, verdict: "incomplete" }, aborted: false }),
      0,
    ],
    ["done, no verdict", recorded({ kind: "resolved", result: RESULT, aborted: false }), 0],
    [
      "done, verdict fail",
      recorded({ kind: "resolved", result: { ...RESULT, verdict: "fail" }, aborted: false }),
      1,
    ],
    ["error", recorded({ kind: "rejected", error: new Error("boom"), aborted: false }), 1],
    [
      "cancelled with a passing result",
      recorded({ kind: "resolved", result: { ...RESULT, verdict: "pass" }, aborted: true }),
      1,
    ],
    ["cancelled while throwing", recorded({ kind: "rejected", error: "x", aborted: true }), 1],
    ["no entry named", { kind: "threw", error: thrown([]) }, 2],
    ["an unknown entry", { kind: "threw", error: thrown(["no-such-entry"]) }, 2],
    ["an argument that is not --name=value", { kind: "threw", error: thrown(["probe", "x"]) }, 2],
    ["a parameter that does not fit", { kind: "threw", error: thrown(["probe", "--rounds=0"]) }, 2],
    ["a failure around the run", { kind: "threw", error: new Error("no adapter") }, 1],
  ])("%s exits %i", (_case, ending, code) => {
    expect(exitCodeOf(ending)).toBe(code);
  });
});

describe("parseCommandLine with --batch", () => {
  it("runs the batch file the path names, its id the file name without .json", () => {
    expect(
      parseCommandLine(["--batch=apps/workbench/batches/synthetic-quick.json"], ENTRIES),
    ).toEqual({
      kind: "batch",
      path: "apps/workbench/batches/synthetic-quick.json",
      id: "synthetic-quick",
    });
    expect(parseCommandLine(["--batch=../a.json"], ENTRIES)).toEqual({
      kind: "batch",
      path: "../a.json",
      id: "a",
    });
  });

  it("leaves --list as it is", () => {
    expect(parseCommandLine(["--batch=a.json", "--list"], ENTRIES)).toEqual({ kind: "list" });
  });

  it.each([
    [["probe", "--batch=a.json"], "runs a batch file alone"],
    [["--batch=a.json", "probe"], "runs a batch file alone"],
    [["--batch=a.json", "--rounds=4"], "runs a batch file alone"],
    [["--batch=a.json", "--batch=b.json"], "runs a batch file alone"],
    [["--batch"], "Name the batch file"],
    [["--batch="], "Name the batch file"],
    [["--batch=batches/quick.txt"], "batches/quick.txt: a batch file is named <id>.json"],
  ])("refuses %j as a usage error", (args, message) => {
    const error = thrown(args);
    expect(error).toBeInstanceOf(UsageError);
    expect(error).toHaveProperty("message", expect.stringContaining(message));
    expect(exitCodeOf({ kind: "threw", error })).toBe(2);
  });
});

const ranState = (settlement: Settlement): StepState<unknown> => {
  const ending = recorded(settlement);
  if (ending.kind !== "recorded") throw new Error("expected a record");
  return { kind: "ran", run: { record: ending.record, text: "", saved: "path" } };
};

const PASSED = ranState({
  kind: "resolved",
  result: { ...RESULT, verdict: "pass" },
  aborted: false,
});

describe("exitCodeOf a batch", () => {
  it.each<[string, readonly StepState<unknown>[], number]>([
    [
      "every step done, verdict pass, incomplete or none",
      [
        PASSED,
        ranState({
          kind: "resolved",
          result: { ...RESULT, verdict: "incomplete" },
          aborted: false,
        }),
        ranState({ kind: "resolved", result: RESULT, aborted: false }),
      ],
      0,
    ],
    [
      "a step with verdict fail",
      [
        PASSED,
        ranState({ kind: "resolved", result: { ...RESULT, verdict: "fail" }, aborted: false }),
      ],
      1,
    ],
    [
      "a step that threw",
      [ranState({ kind: "rejected", error: new Error("boom"), aborted: false }), PASSED],
      1,
    ],
    [
      "a cancelled step and the steps after it",
      [PASSED, ranState({ kind: "rejected", error: "x", aborted: true }), { kind: "not-run" }],
      1,
    ],
    [
      "a step that failed around its run",
      [PASSED, { kind: "host-failed", error: new Error("disk full") }, { kind: "not-run" }],
      1,
    ],
  ])("%s exits %i", (_case, steps, code) => {
    expect(exitCodeOf({ kind: "batch", steps })).toBe(code);
  });
});
