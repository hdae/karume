import { describe, expect, it } from "vite-plus/test";
import type {
  Checkout,
  EntryResult,
  RecordEnvironment,
  WorkbenchRecord,
} from "@karume/sweep-app-support/record";

import { BATCH_FORMAT, type ResolvedBatch, resolveBatch } from "../src/batch.ts";
import {
  type BatchEvent,
  type BatchRun,
  batchSummary,
  runBatch,
  type StepState,
} from "../src/batch-run.ts";
import type { Entry, RunContext } from "../src/entry.ts";
import type { RunHost } from "../src/execute.ts";

const ENVIRONMENT: RecordEnvironment = {
  host: "node",
  userAgent: "test",
  backend: { name: "vulkan", source: "dawn-default" },
  adapter: { vendor: "nvidia", architecture: "ampere", device: "rtx", description: "" },
  features: [],
  limits: {},
  wgslLanguageFeatures: [],
};

const CHECKOUT: Checkout = { commit: "0123abcd", dirty: false, runtimeVersion: "0.22.0" };

/** What the fake entries did, in order. */
let ran: string[];

const entryOf = (id: string, run: (context: RunContext) => Promise<EntryResult>): Entry => ({
  id,
  title: id,
  description: "A test entry.",
  parameters: [{ kind: "integer", name: "rounds", label: "Rounds", min: 1, max: 9, default: 3 }],
  run: (context) => {
    ran.push(`${id}:${String(context.parameters["rounds"])}`);
    return run(context);
  },
});

const PASS = entryOf("passes", async ({ progress }) => {
  progress("half way");
  return { verdict: "pass", summary: ["PASS"], data: null };
});
const FAIL = entryOf("fails", () => Promise.resolve({ verdict: "fail", summary: [], data: null }));
const THROW = entryOf("throws", () => Promise.reject(new Error("refused")));

/** A Cancel while the step runs: the controller of the batch run is aborted from inside it. */
const cancelling = (controller: AbortController): Entry =>
  entryOf("cancels", async ({ signal }) => {
    controller.abort();
    signal.throwIfAborted();
    return { summary: [], data: null };
  });

const batchOf = (entries: readonly Entry[], steps: readonly unknown[]): ResolvedBatch => {
  const resolution = resolveBatch(
    "test-batch",
    JSON.stringify({ format: BATCH_FORMAT, title: "Test", steps }),
    entries,
  );
  if (resolution.kind !== "resolved") throw new Error(resolution.problems.join("; "));
  return resolution;
};

type Host = RunHost<string> & { readonly saved: WorkbenchRecord[] };

/** A host that saves into `saved`; `failReadAt` makes the n-th environment read (1-based) reject. */
const hostOf = (failReadAt?: number): Host => {
  const saved: WorkbenchRecord[] = [];
  let reads = 0;
  return {
    saved,
    readEnvironment: () => {
      reads += 1;
      return reads === failReadAt
        ? Promise.reject(new Error("WebGPU returned no adapter"))
        : Promise.resolve(ENVIRONMENT);
    },
    readCheckout: () => Promise.resolve(CHECKOUT),
    gpuOptions: {},
    openDistribution: () => Promise.reject(new Error("the entry reads no distribution")),
    openInput: () => Promise.reject(new Error("the entry reads no input file")),
    save: (record) => {
      saved.push(record);
      return Promise.resolve(`outputs/workbench/node-rtx/${record.entry.id}_${saved.length}.json`);
    },
  };
};

const run = (
  batch: ResolvedBatch,
  options: {
    readonly host?: Host;
    readonly selected?: readonly number[];
    readonly controller?: AbortController;
    readonly events?: BatchEvent<string>[];
  } = {},
): Promise<BatchRun<string>> => {
  ran = [];
  return runBatch({
    batch,
    selected: options.selected ?? batch.steps.map(({ number }) => number),
    signal: (options.controller ?? new AbortController()).signal,
    host: options.host ?? hostOf(),
    onEvent: (event) => options.events?.push(event),
  });
};

/** A step's state in a few words: `done/pass`, `error`, `not-run`, … */
const brief = (state: StepState<string>): string =>
  state.kind === "ran"
    ? `${state.run.record.outcome}${state.run.record.verdict === undefined ? "" : `/${state.run.record.verdict}`}`
    : state.kind;

const ENTRIES: readonly Entry[] = [PASS, FAIL, THROW];

describe("runBatch", () => {
  it("runs the steps in file order with their own values, and saves each step's record", async () => {
    const host = hostOf();
    const result = await run(
      batchOf(ENTRIES, [
        { id: "fails", parameters: { rounds: 5 } },
        { id: "passes" },
        { id: "fails", parameters: { rounds: "2" } },
      ]),
      { host },
    );
    expect(ran).toEqual(["fails:5", "passes:3", "fails:2"]);
    expect(host.saved.map(({ entry }) => entry)).toEqual([
      { id: "fails", parameters: { rounds: 5 } },
      { id: "passes", parameters: { rounds: 3 } },
      { id: "fails", parameters: { rounds: 2 } },
    ]);
    const states = result.steps.map(({ state }) => state);
    expect(states.map(brief)).toEqual(["done/fail", "done/pass", "done/fail"]);
    expect(states[1]?.kind === "ran" && states[1].run.saved).toBe(
      "outputs/workbench/node-rtx/passes_2.json",
    );
  });

  it("goes on after a step that throws and after a step whose verdict is fail", async () => {
    const result = await run(
      batchOf(ENTRIES, [{ id: "throws" }, { id: "fails" }, { id: "passes" }]),
    );
    expect(ran).toEqual(["throws:3", "fails:3", "passes:3"]);
    expect(result.steps.map(({ state }) => brief(state))).toEqual([
      "error",
      "done/fail",
      "done/pass",
    ]);
  });

  it("stops after a Cancel: the cancelled step's record is kept and saved, the steps after it do not run", async () => {
    const controller = new AbortController();
    const host = hostOf();
    const result = await run(
      batchOf(
        [...ENTRIES, cancelling(controller)],
        [{ id: "passes" }, { id: "cancels" }, { id: "passes" }, { id: "throws" }],
      ),
      { host, controller },
    );
    expect(ran).toEqual(["passes:3", "cancels:3"]);
    expect(result.steps.map(({ state }) => brief(state))).toEqual([
      "done/pass",
      "cancelled",
      "not-run",
      "not-run",
    ]);
    expect(host.saved.map(({ outcome }) => outcome)).toEqual(["done", "cancelled"]);
  });

  it("runs nothing when the signal is aborted before the first step", async () => {
    const controller = new AbortController();
    controller.abort();
    const result = await run(batchOf(ENTRIES, [{ id: "passes" }, { id: "passes" }]), {
      controller,
    });
    expect(ran).toEqual([]);
    expect(result.steps.map(({ state }) => brief(state))).toEqual(["not-run", "not-run"]);
  });

  it("does not run an unselected step, and keeps the file's numbering in the records", async () => {
    const host = hostOf();
    const result = await run(
      batchOf(ENTRIES, [
        { id: "passes", parameters: { rounds: 1 } },
        { id: "passes", parameters: { rounds: 2 } },
        { id: "passes", parameters: { rounds: 3 } },
        { id: "passes", parameters: { rounds: 4 } },
      ]),
      { host, selected: [4, 2] },
    );
    expect(ran).toEqual(["passes:2", "passes:4"]);
    expect(result.steps.map(({ step, state }) => [step.number, brief(state)])).toEqual([
      [1, "not-selected"],
      [2, "done/pass"],
      [3, "not-selected"],
      [4, "done/pass"],
    ]);
    expect(host.saved.map(({ batch }) => [batch?.step, batch?.steps])).toEqual([
      [2, 4],
      [4, 4],
    ]);
  });

  it("gives every record the batch field: the batch's id, one startedAt for the run, the step's number and the file's step count", async () => {
    const host = hostOf();
    const before = Date.now();
    const result = await run(
      batchOf(ENTRIES, [{ id: "passes" }, { id: "throws" }, { id: "fails" }]),
      {
        host,
      },
    );
    expect(host.saved.map(({ batch }) => batch)).toEqual([
      { id: "test-batch", startedAt: result.startedAt, step: 1, steps: 3 },
      { id: "test-batch", startedAt: result.startedAt, step: 2, steps: 3 },
      { id: "test-batch", startedAt: result.startedAt, step: 3, steps: 3 },
    ]);
    expect(new Date(result.startedAt).toISOString()).toBe(result.startedAt);
    expect(Date.parse(result.startedAt)).toBeGreaterThanOrEqual(before);
  });

  it("stops the batch when a step's environment cannot be read: that step does not run, nor do the rest", async () => {
    const host = hostOf(2);
    const result = await run(
      batchOf(ENTRIES, [{ id: "passes" }, { id: "fails" }, { id: "passes" }]),
      { host },
    );
    expect(ran).toEqual(["passes:3"]);
    expect(result.steps.map(({ state }) => brief(state))).toEqual([
      "done/pass",
      "host-failed",
      "not-run",
    ]);
    const failed = result.steps[1]?.state;
    expect(failed?.kind === "host-failed" && failed.error).toEqual(
      new Error("WebGPU returned no adapter"),
    );
    expect(host.saved).toHaveLength(1);
  });

  it("reports a start and progress for each step it runs, and one settled event per step in file order", async () => {
    const events: BatchEvent<string>[] = [];
    await run(batchOf(ENTRIES, [{ id: "passes" }, { id: "throws" }, { id: "passes" }]), {
      events,
      selected: [1, 2],
    });
    expect(
      events.map((event) =>
        event.kind === "progress"
          ? `${event.step.number} progress ${event.text}`
          : `${event.step.number} ${event.kind}${event.kind === "settled" ? ` ${brief(event.state)}` : ""}`,
      ),
    ).toEqual([
      "1 started",
      "1 progress half way",
      "1 settled done/pass",
      "2 started",
      "2 settled error",
      "3 settled not-selected",
    ]);
  });
});

describe("batchSummary", () => {
  it("counts what happened to every step, the steps that did not run included", async () => {
    const controller = new AbortController();
    const result = await run(
      batchOf(
        [...ENTRIES, cancelling(controller)],
        [
          { id: "passes" },
          { id: "fails" },
          { id: "throws" },
          { id: "passes" },
          { id: "cancels" },
          { id: "passes" },
        ],
      ),
      { controller, selected: [1, 2, 3, 5, 6] },
    );
    expect(batchSummary("test-batch", result)).toBe(
      "batch test-batch: 6 steps — 2 done (1 with verdict fail), 1 error, 1 cancelled, 1 did not run, 1 not selected",
    );
  });

  it("names a host failure, and a lone step in the singular", async () => {
    const result = await run(batchOf(ENTRIES, [{ id: "passes" }]), { host: hostOf(1) });
    expect(batchSummary("one", result)).toBe("batch one: 1 step — 1 failed around the run");
  });
});
