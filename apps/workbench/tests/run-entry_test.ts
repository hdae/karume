import { describe, expect, it } from "vite-plus/test";
import {
  type Checkout,
  type RecordEnvironment,
  recordText,
  type WorkbenchRecord,
} from "@karume/sweep-app-support/record";

import type { Entry, GpuOptions, OpenDistribution, OpenInput, RunContext } from "../src/entry.ts";
import { type RunHost, runEntry } from "../src/execute.ts";

const ENVIRONMENT: RecordEnvironment = {
  host: "node",
  userAgent: "test",
  backend: { name: "vulkan", source: "dawn-default" },
  adapter: { vendor: "nvidia", architecture: "ampere", device: "rtx", description: "" },
  features: [],
  limits: {},
  wgslLanguageFeatures: [],
};

const CHECKOUT: Checkout = { commit: "0123abcd", dirty: true, runtimeVersion: "0.22.0" };

const GPU_OPTIONS: GpuOptions = {};

const OPEN_DISTRIBUTION: OpenDistribution = () =>
  Promise.reject(new Error("the entry reads no distribution"));

const OPEN_INPUT: OpenInput = () => Promise.reject(new Error("the entry reads no input file"));

const entryOf = (run: Entry["run"]): Entry => ({
  id: "probe",
  title: "Probe",
  description: "A test entry.",
  parameters: [{ kind: "integer", name: "rounds", label: "Rounds", min: 1, max: 9, default: 3 }],
  run,
});

const hostOf = (save: RunHost<string>["save"]): RunHost<string> => ({
  readEnvironment: () => Promise.resolve(ENVIRONMENT),
  readCheckout: () => Promise.resolve(CHECKOUT),
  gpuOptions: GPU_OPTIONS,
  openDistribution: OPEN_DISTRIBUTION,
  openInput: OPEN_INPUT,
  save,
});

describe("runEntry", () => {
  it("runs the entry with the host's GPU options, distributions, input files and backend, and saves the record's text", async () => {
    const contexts: RunContext[] = [];
    const saves: { record: WorkbenchRecord; text: string }[] = [];
    const progress: string[] = [];
    const run = await runEntry(
      hostOf((record, text) => {
        saves.push({ record, text });
        return Promise.resolve("outputs/workbench/node-rtx/probe.json");
      }),
      {
        entry: entryOf(async (context) => {
          contexts.push(context);
          context.progress("half way");
          return { verdict: "pass", summary: ["PASS"], data: { n: 1 } };
        }),
        parameters: { rounds: 4 },
        signal: new AbortController().signal,
        progress: (line) => progress.push(line),
      },
    );
    expect(contexts).toHaveLength(1);
    expect(contexts[0]).toMatchObject({ parameters: { rounds: 4 }, backend: "vulkan" });
    expect(contexts[0]?.gpuOptions).toBe(GPU_OPTIONS);
    expect(contexts[0]?.openDistribution).toBe(OPEN_DISTRIBUTION);
    expect(contexts[0]?.openInput).toBe(OPEN_INPUT);
    expect(progress).toEqual(["half way"]);
    expect(run.record).toMatchObject({
      outcome: "done",
      verdict: "pass",
      entry: { id: "probe", parameters: { rounds: 4 } },
      environment: ENVIRONMENT,
      checkout: CHECKOUT,
    });
    expect(run.text).toBe(recordText(run.record));
    expect(saves).toEqual([{ record: run.record, text: run.text }]);
    expect(run.saved).toBe("outputs/workbench/node-rtx/probe.json");
  });

  it("records a throw of the entry and still saves it", async () => {
    let saved = 0;
    const run = await runEntry(
      hostOf(() => {
        saved += 1;
        return Promise.resolve("path");
      }),
      {
        entry: entryOf(() => Promise.reject(new Error("refused"))),
        parameters: { rounds: 3 },
        signal: new AbortController().signal,
        progress: () => {},
      },
    );
    expect(run.record.outcome).toBe("error");
    expect(saved).toBe(1);
  });

  it("rejects with what the host's save throws", async () => {
    await expect(
      runEntry(
        hostOf(() => Promise.reject(new Error("disk full"))),
        {
          entry: entryOf(() => Promise.resolve({ summary: [], data: null })),
          parameters: { rounds: 3 },
          signal: new AbortController().signal,
          progress: () => {},
        },
      ),
    ).rejects.toThrow("disk full");
  });
});
