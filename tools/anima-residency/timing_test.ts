import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import {
  aggregateRuns,
  looksQuantized,
  type RunDiagnostics,
  snapshotRun,
  topEntries,
} from "./timing.ts";
import { createGenerateRecorder, geometryProfilesOf, type StageRunDiagnostics } from "./record.ts";

/** 計測ありの run 1 回ぶんの診断（`workgroupCount` は集計に載らない欄として混ぜておく）。 */
const timedRun = (
  entries: readonly { key: string; ns: number; dispatchCount: number }[],
  clampedNegativeSamples = 0,
): RunDiagnostics => ({
  lastRunTiming: {
    entries: entries.map((entry) => ({ ...entry, workgroupCount: 7 })),
    totalNs: entries.reduce((sum, entry) => sum + entry.ns, 0),
    dispatchCount: entries.reduce((sum, entry) => sum + entry.dispatchCount, 0),
    clampedNegativeSamples,
  },
  lastRunPipelines: entries.map(({ key, dispatchCount }) => ({ key, dispatchCount })),
});

const untimedRun = (
  pipelines: readonly { key: string; dispatchCount: number }[],
): RunDiagnostics => ({
  lastRunTiming: undefined,
  lastRunPipelines: pipelines,
});

/** 記録器へ渡す診断（Session が名乗る幾何プロファイルの id を添える）。 */
const onDevice = (
  diagnostics: RunDiagnostics,
  geometryProfile = "default",
): StageRunDiagnostics => ({
  ...diagnostics,
  geometryProfile,
});

describe("aggregateRuns", () => {
  it("sums GPU time and dispatch counts per key across the runs of one stage", () => {
    const profile = aggregateRuns([
      snapshotRun(timedRun([{ key: "linear", ns: 300, dispatchCount: 2 }, {
        key: "attn",
        ns: 100,
        dispatchCount: 1,
      }], 1)),
      snapshotRun(timedRun([{ key: "attn", ns: 250, dispatchCount: 1 }, {
        key: "norm",
        ns: 50,
        dispatchCount: 4,
      }], 2)),
    ]);
    assertEquals(profile.gpu, {
      totalNs: 700,
      runs: 2,
      clampedNegativeSamples: 3,
      entries: [
        { key: "attn", ns: 350, dispatchCount: 2 },
        { key: "linear", ns: 300, dispatchCount: 2 },
        { key: "norm", ns: 50, dispatchCount: 4 },
      ],
    });
    assertEquals(profile.pipelines, [
      { key: "attn", dispatchCount: 2 },
      { key: "linear", dispatchCount: 2 },
      { key: "norm", dispatchCount: 4 },
    ]);
  });

  it("orders keys with equal time by name so the export is stable", () => {
    const profile = aggregateRuns([
      snapshotRun(timedRun([{ key: "b", ns: 10, dispatchCount: 1 }, {
        key: "a",
        ns: 10,
        dispatchCount: 1,
      }])),
    ]);
    assertEquals(profile.gpu?.entries.map((entry) => entry.key), ["a", "b"]);
  });

  it("keeps the dispatch counts but leaves GPU time out when the device does not measure", () => {
    const profile = aggregateRuns([
      snapshotRun(untimedRun([{ key: "vae", dispatchCount: 3 }])),
      snapshotRun(untimedRun([{ key: "vae", dispatchCount: 3 }])),
    ]);
    assertEquals(profile, { pipelines: [{ key: "vae", dispatchCount: 6 }] });
  });

  it("rejects a stage that mixes measured and unmeasured runs", () => {
    assertThrows(
      () =>
        aggregateRuns([
          snapshotRun(timedRun([{ key: "a", ns: 1, dispatchCount: 1 }])),
          snapshotRun(untimedRun([{ key: "a", dispatchCount: 1 }])),
        ]),
      Error,
      "混ざった",
    );
  });

  it("rejects a run whose planned dispatch table is missing", () => {
    assertThrows(
      () => snapshotRun({ lastRunTiming: undefined, lastRunPipelines: undefined }),
      Error,
      "lastRunPipelines",
    );
  });

  it("does not keep references into the diagnostics it copied", () => {
    const entry = { key: "a", ns: 5, dispatchCount: 1, workgroupCount: 1 };
    const pipeline = { key: "a", dispatchCount: 1 };
    const sample = snapshotRun({
      lastRunTiming: { entries: [entry], totalNs: 5, dispatchCount: 1, clampedNegativeSamples: 0 },
      lastRunPipelines: [pipeline],
    });
    entry.ns = 999;
    pipeline.dispatchCount = 999;
    assertEquals(sample.timing?.entries[0].ns, 5);
    assertEquals(sample.pipelines[0].dispatchCount, 1);
  });
});

describe("topEntries / looksQuantized", () => {
  it("returns the leading keys with their share of the stage", () => {
    const profile = aggregateRuns([
      snapshotRun(timedRun([
        { key: "a", ns: 600, dispatchCount: 1 },
        { key: "b", ns: 300, dispatchCount: 1 },
        { key: "c", ns: 100, dispatchCount: 1 },
      ])),
    ]);
    if (profile.gpu === undefined) throw new Error("gpu が無い");
    assertEquals(topEntries(profile.gpu, 2), [
      { key: "a", ns: 600, dispatchCount: 1, share: 0.6 },
      { key: "b", ns: 300, dispatchCount: 1, share: 0.3 },
    ]);
  });

  it("flags stage totals that are all multiples of Chrome's 100 µs quantum", () => {
    const quantized = aggregateRuns([
      snapshotRun(timedRun([{ key: "a", ns: 200_000, dispatchCount: 3 }, {
        key: "b",
        ns: 0,
        dispatchCount: 1,
      }])),
    ]).gpu;
    const exact = aggregateRuns([
      snapshotRun(timedRun([{ key: "a", ns: 200_001, dispatchCount: 3 }])),
    ]).gpu;
    if (quantized === undefined || exact === undefined) throw new Error("gpu が無い");
    assertEquals(looksQuantized(quantized), true);
    assertEquals(looksQuantized(exact), false);
  });
});

describe("createGenerateRecorder", () => {
  it("attributes each run to the open stage of the same component and times stages from restart", () => {
    let clock = 1_000;
    const recorder = createGenerateRecorder(() => clock);
    clock = 5_000;
    recorder.restart();
    recorder.onEvent({ kind: "stage", component: "text_encoder", at: "start" });
    clock += 10;
    recorder.onRun("text_encoder", onDevice(timedRun([{ key: "te", ns: 7, dispatchCount: 1 }])));
    recorder.onEvent({ kind: "stage", component: "text_encoder", at: "end" });
    recorder.onEvent({ kind: "stage", component: "transformer", at: "start" });
    clock += 5;
    recorder.onEvent({
      kind: "residency",
      component: "transformer",
      action: "retained",
      reason: "request",
    });
    recorder.onRun("transformer", onDevice(timedRun([{ key: "dit", ns: 40, dispatchCount: 2 }])));
    recorder.onRun("transformer", onDevice(timedRun([{ key: "dit", ns: 60, dispatchCount: 2 }])));
    clock += 20;
    recorder.onEvent({ kind: "stage", component: "transformer", at: "end" });
    const { stages, residency } = recorder.finish();
    assertEquals(recorder.elapsedMs(), 35);
    assertEquals(
      stages.map(({ component, startMs, endMs, gpu }) => ({
        component,
        startMs,
        endMs,
        totalNs: gpu?.totalNs,
        runs: gpu?.runs,
      })),
      [
        { component: "text_encoder", startMs: 0, endMs: 10, totalNs: 7, runs: 1 },
        { component: "transformer", startMs: 10, endMs: 35, totalNs: 100, runs: 2 },
      ],
    );
    assertEquals(residency, [
      { atMs: 15, action: "retained", reason: "request", position: "transformer の途中" },
    ]);
  });

  it("fails loudly when a run finishes outside any open stage of its component", () => {
    const recorder = createGenerateRecorder(() => 0);
    recorder.onEvent({ kind: "stage", component: "text_encoder", at: "start" });
    assertThrows(
      () =>
        recorder.onRun(
          "vae_decoder",
          onDevice(untimedRun([{ key: "vae", dispatchCount: 1 }])),
        ),
      Error,
      "段の外",
    );
  });

  it("records the geometry profile each stage's session reported, keeping stages that differ apart", () => {
    const recorder = createGenerateRecorder(() => 0);
    recorder.onEvent({ kind: "stage", component: "text_encoder", at: "start" });
    recorder.onRun(
      "text_encoder",
      onDevice(untimedRun([{ key: "te", dispatchCount: 1 }]), "apple-metal-3"),
    );
    recorder.onEvent({ kind: "stage", component: "text_encoder", at: "end" });
    recorder.onEvent({ kind: "stage", component: "transformer", at: "start" });
    recorder.onRun(
      "transformer",
      onDevice(untimedRun([{ key: "dit", dispatchCount: 2 }]), "nvidia-blackwell"),
    );
    recorder.onEvent({ kind: "stage", component: "transformer", at: "end" });
    // run が 1 回も終わらない段（VAE の段が最初の run を終える前に落ちた形）には id が無い。
    recorder.onEvent({ kind: "stage", component: "vae_decoder", at: "start" });
    const { stages } = recorder.finish();
    assertEquals(
      stages.map(({ component, geometryProfile }) => ({ component, geometryProfile })),
      [
        { component: "text_encoder", geometryProfile: "apple-metal-3" },
        { component: "transformer", geometryProfile: "nvidia-blackwell" },
        { component: "vae_decoder", geometryProfile: undefined },
      ],
    );
    assertEquals(Object.hasOwn(stages[2], "geometryProfile"), false);
    assertEquals(geometryProfilesOf(stages), ["apple-metal-3", "nvidia-blackwell"]);
  });

  it("fails loudly when runs of one stage name different geometry profiles", () => {
    const recorder = createGenerateRecorder(() => 0);
    recorder.onEvent({ kind: "stage", component: "transformer", at: "start" });
    recorder.onRun("transformer", onDevice(untimedRun([{ key: "dit", dispatchCount: 1 }])));
    assertThrows(
      () =>
        recorder.onRun(
          "transformer",
          onDevice(untimedRun([{ key: "dit", dispatchCount: 1 }]), "apple-metal-3"),
        ),
      Error,
      "幾何プロファイルが割れた",
    );
  });
});
