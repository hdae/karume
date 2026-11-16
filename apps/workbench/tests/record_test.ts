import { describe, expect, it } from "vite-plus/test";

import type { Entry, EntryResult } from "../src/entry.ts";
import { dawnBackend } from "../src/environment.ts";
import { executeEntry } from "../src/execute.ts";
import {
  buildRecord,
  type Checkout,
  RECORD_FORMAT,
  type RecordEnvironment,
  recordText,
  type Settlement,
  utcStamp,
} from "../src/record.ts";

const ENVIRONMENT: RecordEnvironment = {
  host: "browser",
  userAgent: "test",
  crossOriginIsolated: true,
  adapter: { vendor: "nvidia", architecture: "blackwell", device: "", description: "" },
  features: ["shader-f16", "subgroups"],
  limits: { maxBufferSize: 2147483644 },
  wgslLanguageFeatures: ["subgroup_id"],
};

const CHECKOUT: Checkout = { commit: "0123abcd", dirty: false, runtimeVersion: "0.22.0" };

const RESULT: EntryResult = { verdict: "pass", summary: ["PASS"], data: { checks: 12 } };

const record = (settlement: Settlement, environment: RecordEnvironment = ENVIRONMENT) =>
  buildRecord({
    entry: { id: "kernel-check", parameters: { checks: ["gemv-integer"] } },
    endedAt: new Date("2026-11-16T09:30:12.345Z"),
    milliseconds: 1234.5678,
    environment,
    checkout: CHECKOUT,
    settlement,
  });

describe("buildRecord", () => {
  it("wraps a run that resolved in the envelope, the entry's verdict, summary and data after it", () => {
    const done = record({ kind: "resolved", result: RESULT, aborted: false });
    expect(done).toEqual({
      format: RECORD_FORMAT,
      entry: { id: "kernel-check", parameters: { checks: ["gemv-integer"] } },
      recordedAt: "2026-11-16T09:30:12.345Z",
      seconds: 1.235,
      outcome: "done",
      environment: ENVIRONMENT,
      checkout: CHECKOUT,
      verdict: "pass",
      summary: ["PASS"],
      data: { checks: 12 },
    });
    expect(Object.keys(done)).toEqual([
      "format",
      "entry",
      "recordedAt",
      "seconds",
      "outcome",
      "environment",
      "checkout",
      "verdict",
      "summary",
      "data",
    ]);
    expect(RECORD_FORMAT).toBe("karume-workbench/1");
  });

  it("records a run that threw as error with its stack, and no result", () => {
    const failed = record({ kind: "rejected", error: new Error("device lost"), aborted: false });
    expect(failed.outcome).toBe("error");
    expect(failed.error).toContain("Error: device lost");
    expect(failed).not.toHaveProperty("verdict");
    expect(failed).not.toHaveProperty("summary");
    expect(failed).not.toHaveProperty("data");
    expect(record({ kind: "rejected", error: "plain", aborted: false }).error).toBe("plain");
  });

  it("records a run that settled after the abort as cancelled, keeping what it returned", () => {
    const thrown = record({ kind: "rejected", error: new Error("aborted"), aborted: true });
    expect(thrown.outcome).toBe("cancelled");
    expect(thrown).not.toHaveProperty("error");
    expect(thrown).not.toHaveProperty("summary");
    const partial = record({
      kind: "resolved",
      result: { ...RESULT, verdict: "incomplete" },
      aborted: true,
    });
    expect(partial.outcome).toBe("cancelled");
    expect(partial.verdict).toBe("incomplete");
    expect(partial.data).toEqual({ checks: 12 });
  });

  it("leaves the backend out when none was declared, and keeps a declared one", () => {
    expect(
      record({ kind: "resolved", result: RESULT, aborted: false }).environment,
    ).not.toHaveProperty("backend");
    const declared = record(
      { kind: "resolved", result: RESULT, aborted: false },
      { ...ENVIRONMENT, backend: { name: "d3d12", source: "declared" } },
    );
    expect(declared.environment.backend).toEqual({ name: "d3d12", source: "declared" });
  });

  it("leaves the verdict out for an entry that only measures", () => {
    const measured = record({
      kind: "resolved",
      result: { summary: ["3 cases"], data: [] },
      aborted: false,
    });
    expect(measured).not.toHaveProperty("verdict");
    expect(measured.summary).toEqual(["3 cases"]);
  });
});

describe("recordText", () => {
  it("is the record as JSON with one-space indentation and a final newline, and parses back to it", () => {
    const done = record({ kind: "resolved", result: RESULT, aborted: false });
    const text = recordText(done);
    expect(text).toBe(`${JSON.stringify(done, null, 1)}\n`);
    expect(JSON.parse(text)).toEqual(done);
  });
});

describe("utcStamp", () => {
  it("is the UTC time to the second with dashes for colons", () => {
    expect(utcStamp(new Date("2026-11-16T09:30:12.999Z"))).toBe("2026-11-16T09-30-12");
  });
});

describe("executeEntry", () => {
  const entryOf = (run: Entry["run"]): Entry => ({
    id: "probe",
    title: "Probe",
    description: "A test entry.",
    parameters: [],
    run,
  });
  const execute = (entry: Entry, controller = new AbortController()) =>
    executeEntry({
      entry,
      context: {
        parameters: {},
        gpuOptions: {},
        signal: controller.signal,
        progress: () => {},
        openDistribution: () => Promise.reject(new Error("the entry reads no distribution")),
      },
      environment: ENVIRONMENT,
      checkout: CHECKOUT,
    });

  it("passes the context through and records the result as done", async () => {
    const seen: string[] = [];
    const done = await execute(
      entryOf(async ({ progress }) => {
        progress("half way");
        seen.push("ran");
        return RESULT;
      }),
    );
    expect(seen).toEqual(["ran"]);
    expect(done.outcome).toBe("done");
    expect(done.entry).toEqual({ id: "probe", parameters: {} });
    expect(done.seconds).toBeGreaterThanOrEqual(0);
  });

  it("hands the entry the backend the record names, and none when the record has none", async () => {
    const seen: (string | undefined)[] = [];
    const entry = entryOf(async ({ backend }) => {
      seen.push(backend);
      return RESULT;
    });
    await executeEntry({
      entry,
      context: {
        parameters: {},
        gpuOptions: {},
        signal: new AbortController().signal,
        progress: () => {},
        openDistribution: () => Promise.reject(new Error("the entry reads no distribution")),
      },
      environment: { ...ENVIRONMENT, backend: { name: "d3d12", source: "declared" } },
      checkout: CHECKOUT,
    });
    const { backend: _dropped, ...withoutBackend } = ENVIRONMENT;
    await executeEntry({
      entry,
      context: {
        parameters: {},
        gpuOptions: {},
        signal: new AbortController().signal,
        progress: () => {},
        openDistribution: () => Promise.reject(new Error("the entry reads no distribution")),
      },
      environment: withoutBackend,
      checkout: CHECKOUT,
    });
    expect(seen).toEqual(["d3d12", undefined]);
  });

  it("records a throw as error and a run cancelled midway as cancelled", async () => {
    expect((await execute(entryOf(() => Promise.reject(new Error("boom"))))).outcome).toBe("error");
    const controller = new AbortController();
    const cancelled = await execute(
      entryOf(async ({ signal }) => {
        controller.abort();
        signal.throwIfAborted();
        return RESULT;
      }),
      controller,
    );
    expect(cancelled.outcome).toBe("cancelled");
  });
});

describe("dawnBackend", () => {
  it("takes a backend= flag, else Dawn's default for the platform", () => {
    expect(dawnBackend(["disable-dawn-features=x", "backend=opengl"], "linux")).toEqual({
      name: "opengl",
      source: "dawn-flag",
    });
    expect(dawnBackend([], "linux")).toEqual({ name: "vulkan", source: "dawn-default" });
    expect(dawnBackend([], "darwin")).toEqual({ name: "metal", source: "dawn-default" });
    expect(dawnBackend([], "win32")).toEqual({ name: "d3d12", source: "dawn-default" });
    expect(() => dawnBackend([], "aix")).toThrow("aix");
  });
});
