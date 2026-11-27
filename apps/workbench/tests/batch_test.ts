import { describe, expect, it } from "vite-plus/test";
import { parseParameters } from "@karume/sweep-app-support/parameters";
import { buildRecord, recordText } from "@karume/sweep-app-support/record";

import {
  BATCH_FORMAT,
  type BatchFile,
  type BatchResolution,
  listBatches,
  type ResolvedBatch,
  resolveBatch,
} from "../src/batch.ts";
import type { Entry } from "../src/entry.ts";

const PROBE: Entry = {
  id: "probe",
  title: "Probe",
  description: "A test entry.",
  parameters: [
    { kind: "boolean", name: "warm", label: "Warm up", default: false },
    { kind: "integer", name: "rounds", label: "Rounds", min: 1, max: 9, default: 3 },
    { kind: "number", name: "gain", label: "Gain", min: 0, max: 1, default: 0.5 },
    { kind: "text", name: "arms", label: "Arms", default: "" },
    { kind: "choice", name: "set", label: "Set", options: ["quick", "quick+"], default: "quick" },
    {
      kind: "choices",
      name: "checks",
      label: "Checks",
      options: ["a", "b", "c"],
      default: ["a", "b", "c"],
    },
  ],
  run: () => Promise.resolve({ summary: [], data: null }),
};

const PLAIN: Entry = {
  id: "plain",
  title: "Plain",
  description: "A test entry without parameters.",
  parameters: [],
  run: () => Promise.resolve({ summary: [], data: null }),
};

const ENTRIES: readonly Entry[] = [PROBE, PLAIN];

/** A batch file's text: the format and a title around `fields`, unless `fields` replaces them. */
const fileText = (fields: Record<string, unknown>): string =>
  JSON.stringify({ format: BATCH_FORMAT, title: "A test batch", ...fields });

const resolved = (resolution: BatchResolution): ResolvedBatch => {
  if (resolution.kind !== "resolved") {
    throw new Error(`expected a resolved batch, got ${resolution.problems.join("; ")}`);
  }
  return resolution;
};

const problemsOf = (resolution: BatchResolution): readonly string[] => {
  if (resolution.kind !== "problems") throw new Error("expected problems, the batch resolved");
  return resolution.problems;
};

const stepsProblems = (steps: readonly unknown[]): readonly string[] =>
  problemsOf(resolveBatch("test", fileText({ steps }), ENTRIES));

describe("resolveBatch", () => {
  it("resolves a valid file to each step's entry and typed values, the defaults filled in for what a step does not write", () => {
    const batch = resolved(
      resolveBatch(
        "session-ab-three",
        fileText({
          description: "One paragraph.",
          steps: [
            { id: "probe" },
            {
              id: "probe",
              parameters: {
                warm: true,
                rounds: 7,
                gain: 0.25,
                arms: "e2b:split",
                set: "quick+",
                checks: ["c", "a"],
              },
            },
            { id: "plain", parameters: {} },
          ],
        }),
        ENTRIES,
      ),
    );
    expect(batch).toMatchObject({
      id: "session-ab-three",
      title: "A test batch",
      description: "One paragraph.",
    });
    expect(batch.steps.map(({ number, entry }) => [number, entry])).toEqual([
      [1, PROBE],
      [2, PROBE],
      [3, PLAIN],
    ]);
    expect(batch.steps[0]?.parameters).toEqual(parseParameters(PROBE.parameters, {}));
    expect(batch.steps[0]?.given).toEqual({});
    expect(batch.steps[1]?.parameters).toEqual({
      warm: true,
      rounds: 7,
      gain: 0.25,
      arms: "e2b:split",
      set: "quick+",
      checks: ["a", "c"],
    });
    expect(batch.steps[1]?.given).toEqual({
      warm: "true",
      rounds: "7",
      gain: "0.25",
      arms: "e2b:split",
      set: "quick+",
      checks: "c,a",
    });
    expect(batch.steps[2]?.parameters).toEqual({});
  });

  it('reads a value the way the URL and the command line do: 3 and "3" alike, a list joined with commas', () => {
    const batch = resolved(
      resolveBatch(
        "test",
        fileText({
          steps: [
            { id: "probe", parameters: { rounds: 3, checks: ["b"] } },
            { id: "probe", parameters: { rounds: "3", checks: "b" } },
            { id: "probe", parameters: { checks: [], arms: ["x", "y"] } },
          ],
        }),
        ENTRIES,
      ),
    );
    expect(batch.steps[0]?.parameters).toEqual(batch.steps[1]?.parameters);
    expect(batch.steps[2]?.parameters["checks"]).toEqual([]);
    expect(batch.steps[2]?.parameters["arms"]).toBe("x,y");
  });

  it("leaves the description out when the file has none", () => {
    expect(
      resolved(resolveBatch("test", fileText({ steps: [{ id: "plain" }] }), ENTRIES)),
    ).not.toHaveProperty("description");
  });

  it("takes the entry object of a saved record as a step, back to the same typed values", () => {
    const parameters = parseParameters(PROBE.parameters, {
      warm: "true",
      rounds: "9",
      gain: "0.125",
      arms: "a,b c",
      set: "quick+",
      checks: "",
    });
    const text = recordText(
      buildRecord({
        entry: { id: "probe", parameters },
        endedAt: new Date("2026-11-16T09:30:12.345Z"),
        milliseconds: 1,
        environment: {
          host: "browser",
          userAgent: "test",
          adapter: { vendor: "", architecture: "", device: "", description: "" },
          features: [],
          limits: {},
          wgslLanguageFeatures: [],
        },
        checkout: { commit: "0123abcd", dirty: false, runtimeVersion: "0.22.0" },
        settlement: { kind: "resolved", result: { summary: [], data: null }, aborted: false },
      }),
    );
    const saved: unknown = JSON.parse(text);
    const entry = typeof saved === "object" && saved !== null && "entry" in saved && saved.entry;
    const batch = resolved(resolveBatch("test", fileText({ steps: [entry] }), ENTRIES));
    expect(batch.steps[0]?.entry).toBe(PROBE);
    expect(batch.steps[0]?.parameters).toEqual(parameters);
  });

  it.each<[string, unknown, string]>([
    ["an id no entry has", { id: "nope" }, "step 1 (nope): no registered entry has this id"],
    ["a missing id", { parameters: {} }, "step 1: id is missing"],
    ["an id that is not a string", { id: 7 }, "step 1: id is not a string"],
    ["a step that is not an object", "probe", "step 1: a step is an object"],
    ["a step that is a list", ["probe"], "step 1: a step is an object"],
    [
      "a key a step does not have",
      { id: "probe", repeat: 2 },
      'step 1 (probe): the key "repeat" is not part of a step',
    ],
    [
      "parameters that are not an object",
      { id: "probe", parameters: ["rounds=3"] },
      "step 1 (probe): parameters is not an object",
    ],
    [
      "parameters that are null",
      { id: "probe", parameters: null },
      "step 1 (probe): parameters is not an object",
    ],
    [
      "a parameter the entry does not declare",
      { id: "probe", parameters: { rouds: 3 } },
      "step 1 (probe): Parameter rouds: this entry has no such parameter",
    ],
    [
      "a value off the bounds",
      { id: "probe", parameters: { rounds: 10 } },
      'step 1 (probe): Parameter rounds: expected an integer from 1 to 9, got "10"',
    ],
    [
      "a boolean for an integer",
      { id: "probe", parameters: { rounds: true } },
      'step 1 (probe): Parameter rounds: expected an integer from 1 to 9, got "true"',
    ],
    [
      "an option the choice does not have",
      { id: "probe", parameters: { set: "full" } },
      'step 1 (probe): Parameter set: expected one of quick, quick+, got "full"',
    ],
    [
      "a null value",
      { id: "probe", parameters: { rounds: null } },
      "step 1 (probe): parameter rounds is not a boolean, a number, a string or a list of strings",
    ],
    [
      "an object value",
      { id: "probe", parameters: { arms: { a: 1 } } },
      "step 1 (probe): parameter arms is not a boolean, a number, a string or a list of strings",
    ],
    [
      "a list of numbers",
      { id: "probe", parameters: { checks: [1, 2] } },
      "step 1 (probe): parameter checks is not a boolean, a number, a string or a list of strings",
    ],
  ])("reports %s, naming the step by its number", (_case, step, problem) => {
    const problems = stepsProblems([step]);
    expect(problems).toHaveLength(1);
    expect(problems[0]).toContain(problem);
  });

  it("reports every problem of every step, not only the first, each naming its step", () => {
    expect(
      stepsProblems([
        { id: "probe" },
        { id: "probe", parameters: { rounds: 0, set: "full", nope: true } },
        { id: "plain" },
        { id: "missing", parameters: { rounds: null } },
      ]),
    ).toEqual([
      'step 2 (probe): Parameter rounds: expected an integer from 1 to 9, got "0"',
      'step 2 (probe): Parameter set: expected one of quick, quick+, got "full"',
      "step 2 (probe): Parameter nope: this entry has no such parameter",
      "step 4 (missing): no registered entry has this id",
      "step 4 (missing): parameter rounds is not a boolean, a number, a string or a list of strings",
    ]);
  });

  it("reports a key the format does not name, at the top as in a step", () => {
    expect(
      problemsOf(
        resolveBatch(
          "test",
          fileText({ steps: [{ id: "plain", expect: "pass" }], repeat: 2, loop: true }),
          ENTRIES,
        ),
      ),
    ).toEqual([
      'the key "repeat" is not part of the format',
      'the key "loop" is not part of the format',
      'step 1 (plain): the key "expect" is not part of a step',
    ]);
  });

  it("reports a missing or wrong format, title, description and steps", () => {
    expect(problemsOf(resolveBatch("test", JSON.stringify({}), ENTRIES))).toEqual([
      `format is missing (it is "${BATCH_FORMAT}")`,
      "title is missing",
      "steps is missing",
    ]);
    expect(
      problemsOf(
        resolveBatch(
          "test",
          JSON.stringify({
            format: "karume-workbench-batch/2",
            title: 5,
            description: ["x"],
            steps: { id: "plain" },
          }),
          ENTRIES,
        ),
      ),
    ).toEqual([
      `format is "karume-workbench-batch/2", not "${BATCH_FORMAT}"`,
      "title is not a string",
      "description is not a string",
      "steps is not a list",
    ]);
    expect(problemsOf(resolveBatch("test", fileText({ title: " ", steps: [] }), ENTRIES))).toEqual([
      "title is empty",
      "steps is empty: a batch has at least one step",
    ]);
  });

  it("reports text that is not JSON, and JSON that is not an object", () => {
    const [notJson] = problemsOf(resolveBatch("test", "{ steps: [", ENTRIES));
    expect(notJson).toMatch(/^the file is not JSON \(/);
    expect(problemsOf(resolveBatch("test", "[]", ENTRIES))).toEqual([
      "the file is not a JSON object",
    ]);
    expect(problemsOf(resolveBatch("test", "null", ENTRIES))).toEqual([
      "the file is not a JSON object",
    ]);
  });

  it("reports a file name that is not kebab-case with the rest", () => {
    expect(
      problemsOf(resolveBatch("Session_AB", fileText({ steps: [{ id: "nope" }] }), ENTRIES)),
    ).toEqual([
      "the file name Session_AB.json is not kebab-case (lower-case letters and digits, words joined by hyphens)",
      "step 1 (nope): no registered entry has this id",
    ]);
  });

  it("keeps the title and the description of a batch with problems, the id standing in for a title it cannot read", () => {
    const named = resolveBatch(
      "named",
      fileText({ description: "Why.", steps: [{ id: "nope" }] }),
      ENTRIES,
    );
    expect(named).toMatchObject({
      id: "named",
      title: "A test batch",
      description: "Why.",
      kind: "problems",
      unknownEntries: ["nope"],
    });
    expect(resolveBatch("untitled", "{", ENTRIES)).toMatchObject({
      id: "untitled",
      title: "untitled",
    });
  });
});

const VALID = fileText({ steps: [{ id: "plain" }] });

const listed = (
  tracked: readonly BatchFile[],
  untracked: readonly BatchFile[],
  production = false,
) =>
  listBatches({ tracked, untracked, entries: ENTRIES, production }).map((batch) => ({
    id: batch.id,
    source: batch.source,
    kind: batch.kind,
    problems: batch.kind === "problems" ? batch.problems : [],
  }));

describe("listBatches", () => {
  it("lists the tracked batches, then the untracked ones, each resolved", () => {
    expect(
      listed(
        [
          { id: "a", text: VALID },
          { id: "b", text: VALID },
        ],
        [{ id: "c", text: VALID }],
      ),
    ).toEqual([
      { id: "a", source: "tracked", kind: "resolved", problems: [] },
      { id: "b", source: "tracked", kind: "resolved", problems: [] },
      { id: "c", source: "untracked", kind: "resolved", problems: [] },
    ]);
  });

  it("lists an untracked batch with a tracked batch's id as not runnable, saying so first", () => {
    expect(
      listed(
        [{ id: "same", text: VALID }],
        [{ id: "same", text: fileText({ steps: [{ id: "nope" }] }) }],
      ),
    ).toEqual([
      { id: "same", source: "tracked", kind: "resolved", problems: [] },
      {
        id: "same",
        source: "untracked",
        kind: "problems",
        problems: [
          "the tracked batch apps/workbench/batches/same.json has the same id; rename this file",
          "step 1 (nope): no registered entry has this id",
        ],
      },
    ]);
  });

  it("lists a file that is not kebab-case and a file the source could not give, with their problems", () => {
    expect(
      listed(
        [],
        [
          { id: "Not_Kebab", text: VALID },
          { id: "huge", problem: "the file is larger than 1048576 bytes, so it was not read" },
        ],
      ),
    ).toEqual([
      {
        id: "Not_Kebab",
        source: "untracked",
        kind: "problems",
        problems: [
          "the file name Not_Kebab.json is not kebab-case (lower-case letters and digits, words joined by hyphens)",
        ],
      },
      {
        id: "huge",
        source: "untracked",
        kind: "problems",
        problems: ["the file is larger than 1048576 bytes, so it was not read"],
      },
    ]);
  });

  it("lists in a production build only the tracked batches whose every step names an entry the build holds", () => {
    const tracked: readonly BatchFile[] = [
      { id: "runs", text: VALID },
      { id: "needs-local", text: fileText({ steps: [{ id: "plain" }, { id: "session-ab" }] }) },
      { id: "bad-value", text: fileText({ steps: [{ id: "probe", parameters: { rounds: 0 } }] }) },
    ];
    expect(listed(tracked, [{ id: "local", text: VALID }], true)).toEqual([
      { id: "runs", source: "tracked", kind: "resolved", problems: [] },
      {
        id: "bad-value",
        source: "tracked",
        kind: "problems",
        problems: ['step 1 (probe): Parameter rounds: expected an integer from 1 to 9, got "0"'],
      },
    ]);
    expect(listed(tracked, [], false).map(({ id }) => id)).toEqual([
      "runs",
      "needs-local",
      "bad-value",
    ]);
  });
});
