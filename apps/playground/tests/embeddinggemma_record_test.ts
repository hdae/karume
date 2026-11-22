import { parseManifest } from "@karume/hub";
import { describe, expect, it } from "vite-plus/test";

import type { RecordEnvironment } from "../src/lib/environment.ts";
import { NO_SUBMITS } from "../src/lib/record.ts";
import {
  buildStatus,
  firstValuesText,
  rowLabel,
  rowOutcomeText,
  similarityText,
  summaryRows,
} from "../src/models/embeddinggemma/format.ts";
import {
  assembleRecord,
  type DiagnosticsReport,
  type MeasuredRow,
  modelDiagnostics,
  RECORD_FORMAT,
} from "../src/models/embeddinggemma/record.ts";
import { effectiveSelection } from "../src/models/embeddinggemma/request.ts";

const hex = (character: string): string => character.repeat(64);

/** A `karume/6` manifest shaped like karume-embeddinggemma-2: model `text`, preset `bf16`. */
const manifest = parseManifest(
  JSON.stringify({
    format: "karume/6",
    generator: "test",
    defaultModel: "text",
    models: {
      text: {
        pipeline: "embeddinggemma/1",
        assets: {},
        weights: {
          model: {
            bf16: {
              container: {
                descriptor: {
                  graph: { length: 8, sha256: hex("d") },
                  model: { length: 8, sha256: hex("e") },
                },
                parts: [
                  { path: "text/model/model.bf16-00001-of-00002.krm", size: 40, sha256: hex("a") },
                  { path: "text/model/model.bf16-00002-of-00002.krm", size: 8, sha256: hex("b") },
                ],
              },
            },
          },
        },
        quants: {
          bf16: { weights: { model: "bf16" }, session: {} },
          other: { weights: { model: "bf16" }, session: {} },
        },
        defaultQuant: "bf16",
        pipelineConfig: {},
      },
    },
  }),
);

describe("effectiveSelection", () => {
  it("starts from the manifest's default model and preset, both marked default", () => {
    expect(effectiveSelection(manifest, undefined, undefined)).toEqual({
      model: "text",
      quant: "bf16",
      modelIsDefault: true,
      quantIsDefault: true,
    });
  });

  it("keeps a preset the model has and falls back for a model or preset it does not have", () => {
    expect(effectiveSelection(manifest, "text", "other")).toMatchObject({
      quant: "other",
      quantIsDefault: false,
    });
    expect(effectiveSelection(manifest, "image", "f16")).toMatchObject({
      model: "text",
      quant: "bf16",
    });
  });
});

const buildStats = {
  partCount: 5,
  supplyWaitMs: 900.04,
  decodeMs: 0,
  bufferCreateMs: 2.26,
  writeBufferIssueMs: 30,
  uploadFenceMs: 300.5,
  uploadedBytes: 571_000_000,
};

/** One report; `runsSoFar` = the Session's runs including this one (2 submits, 300 dispatches each). */
const report = (timingNs: number | undefined, runsSoFar = 1): DiagnosticsReport => ({
  weights: { allocatedBytes: 2 ** 30 },
  submit: { submitCount: 2 * runsSoFar, dispatchCount: 300 * runsSoFar },
  geometryProfile: "nvidia-ampere",
  buildStats,
  lastRunPipelines: [{ key: "linear:v1:bf16", dispatchCount: 200 }],
  lastRunTiming:
    timingNs === undefined
      ? undefined
      : {
          entries: [
            { key: "linear:v1:bf16", ns: timingNs, dispatchCount: 200, workgroupCount: 900 },
          ],
          totalNs: timingNs,
          dispatchCount: 200,
          clampedNegativeSamples: 0,
        },
});

describe("modelDiagnostics", () => {
  it("counts one run per embedded row and sums the GPU time of the timed ones", () => {
    const model = modelDiagnostics([report(2_000_000, 1), report(3_000_000, 2)], NO_SUBMITS);
    expect(model?.runs).toBe(2);
    expect(model?.timing?.totalMs).toBe(5);
    expect(model?.build.supplyWaitMs).toBe(900);
  });

  it("has nothing to say when every row was refused (no Session run)", () => {
    expect(modelDiagnostics([], NO_SUBMITS)).toBeUndefined();
  });

  it("counts only this run's submits and dispatches on a Session kept from earlier runs", () => {
    expect(
      modelDiagnostics([report(undefined, 3)], { submitCount: 4, dispatchCount: 600 }),
    ).toMatchObject({
      submitCount: 2,
      dispatchCount: 300,
      sessionSubmitCount: 6,
      sessionDispatchCount: 900,
    });
  });
});

const environment: RecordEnvironment = {
  recordedAt: "2026-11-21T00:00:00.000Z",
  userAgent: "test",
  crossOriginIsolated: true,
  adapter: { vendor: "nvidia", architecture: "ampere", device: "", description: "Vulkan" },
  device: {
    features: ["subgroups"],
    wgslLanguageFeatures: [],
    subgroupLanesLinear: true,
    gpuTimingEnabled: false,
    limits: {
      maxComputeWorkgroupStorageSize: 32768,
      maxStorageBufferBindingSize: 2 ** 31,
      maxBufferSize: 2 ** 31,
    },
  },
};

const embedded = (
  text: string,
  prompt: string | undefined,
  values: readonly number[],
  sha: string,
  tokens: number | undefined = 7,
): MeasuredRow => ({
  text,
  prompt,
  tokens,
  embedMs: 15.2345,
  outcome: { kind: "embedding", values: new Float32Array(values), sha256: hex(sha) },
});

const refused: MeasuredRow = {
  text: "an image <|image|> in text",
  prompt: "query",
  tokens: 16,
  embedMs: 0.4,
  outcome: {
    kind: "refused",
    error: {
      name: "ModelInputError",
      message: "embeddinggemma.Pipeline: the text holds the image placeholder token",
    },
  },
};

const assemble = (rows: readonly MeasuredRow[], buildMs: number | undefined) =>
  assembleRecord({
    environment,
    directory: "models/karume-embeddinggemma-2",
    manifestSha256: hex("f"),
    source: { kind: "direct" },
    selection: effectiveSelection(manifest, undefined, undefined),
    presetLabel: "Original weights (bf16)",
    dimension: 2,
    rows,
    buildMs,
    wallMs: 1200,
    reports: rows
      .filter(({ outcome }) => outcome.kind === "embedding")
      .map((_, index) => report(undefined, index + 1)),
    countersBefore: NO_SUBMITS,
  });

describe("assembleRecord", () => {
  const rows = [
    embedded("how do I reset my password", "query", [1, 0], "1"),
    refused,
    embedded("", undefined, [Math.SQRT1_2, Math.SQRT1_2], "2", 2),
  ];

  it("records every row as sent, the prompt as a name or null, and the width of the run", () => {
    const record = assemble(rows, 1100);
    expect(record.format).toBe(RECORD_FORMAT);
    expect(record.page).toBe("embeddinggemma");
    expect(record.source).toEqual({ kind: "direct" });
    expect(record.preset).toEqual({
      quant: "bf16",
      label: "Original weights (bf16)",
      isDefault: true,
    });
    expect(record.request).toEqual({
      dimension: 2,
      rows: [
        { text: "how do I reset my password", prompt: "query", tokens: 7 },
        { text: "an image <|image|> in text", prompt: "query", tokens: 16 },
        { text: "", prompt: null, tokens: 2 },
      ],
    });
  });

  it("records a row the pipeline could not tokenize (an undeclared prompt) with a null count", () => {
    const undeclared: MeasuredRow = {
      text: "a",
      prompt: "Query",
      tokens: undefined,
      embedMs: 0.1,
      outcome: {
        kind: "refused",
        error: {
          name: "ModelInputError",
          message: 'embeddinggemma.Pipeline: prompt "Query" is not one of this model\'s prompts',
        },
      },
    };
    expect(assemble([undeclared], undefined).request.rows).toEqual([
      { text: "a", prompt: "Query", tokens: null },
    ]);
  });

  it("keeps each row's vector statistics and sha256, or the refusal as it was raised", () => {
    const record = assemble(rows, 1100);
    const [first, second, third] = record.output.rows;
    expect(first).toMatchObject({ kind: "embedding", dimension: 2, l2Norm: 1, sha256: hex("1") });
    expect(second).toEqual({
      kind: "refused",
      error: {
        name: "ModelInputError",
        message: "embeddinggemma.Pipeline: the text holds the image placeholder token",
      },
      embedSeconds: 0,
    });
    expect(third).toMatchObject({ kind: "embedding", sha256: hex("2"), embedSeconds: 0.015 });
  });

  it("puts the similarity of every embedded pair in the matrix and none for the refused row", () => {
    expect(assemble(rows, 1100).output.similarity).toEqual([
      [1, null, 0.707107],
      [null, null, null],
      [0.707107, null, 1],
    ]);
  });

  it("folds one diagnostics report per embedded row and times the build", () => {
    const record = assemble(rows, 1100);
    expect(record.diagnostics.model?.runs).toBe(2);
    expect(record.timings).toEqual({ pipelineReused: false, buildSeconds: 1.1, wallSeconds: 1.2 });
  });

  it("marks a reused pipeline and has no Session record when every row was refused", () => {
    const record = assemble([refused], undefined);
    expect(record.timings.pipelineReused).toBe(true);
    expect(record.timings.buildSeconds).toBeNull();
    expect(record.diagnostics).toEqual({});
    expect(record.output.similarity).toEqual([[null]]);
  });
});

describe("what the result card shows", () => {
  const record = assemble(
    [embedded("a", "query", [0.6, 0.8], "1"), refused, embedded("b", undefined, [0, 1], "2")],
    undefined,
  );

  it("numbers the rows from 1 and shows a missing similarity as a dash", () => {
    expect(rowLabel(0)).toBe("Row 1");
    expect(similarityText(0.8)).toBe("0.8000");
    expect(similarityText(null)).toBe("—");
  });

  it("sums a row up as its width, norm and seconds, or as refused", () => {
    const [first, second] = record.output.rows;
    if (first === undefined || second === undefined) throw new Error("two rows expected");
    expect(rowOutcomeText(first)).toBe("2 values · norm 1 · 0.015 s");
    expect(firstValuesText(first)).toBe("0.6000, 0.8000");
    expect(rowOutcomeText(second)).toBe("refused");
    expect(firstValuesText(second)).toBe("");
  });

  it("says how many rows were embedded and when the build cost comes from an earlier run", () => {
    const rows = new Map(summaryRows(record));
    expect(rows.get("Rows")).toBe("3 (2 embedded)");
    expect(rows.get("Dimension")).toBe("2");
    expect(rows.get("Build s")).toMatch(/reused/);
    expect(rows.has("Model build (earlier run)")).toBe(true);
  });
});

describe("buildStatus", () => {
  const MIB = 2 ** 20;

  it("keeps Cancel offered while the weights are fetched", () => {
    const status = buildStatus({
      phase: "downloading",
      path: "text/model/model.bf16-00003-of-00005.krm",
      fileLoaded: 50 * MIB,
      fileTotal: 224 * MIB,
      loaded: 100 * MIB,
      total: 555 * MIB,
    });
    expect(status.stage).toBe("fetching the weights");
    expect(status.note).toBe(
      "downloading text/model/model.bf16-00003-of-00005.krm (50 / 224 MiB) · total 100 / 555 MiB",
    );
    expect(status.cancellable).toBeUndefined();
  });

  it("marks the Session build that follows the last file as not cancellable", () => {
    const status = buildStatus({
      phase: "complete",
      path: "text/tokenizer/tokenizer.json",
      fileLoaded: 9 * MIB,
      fileTotal: 9 * MIB,
      loaded: 555 * MIB,
      total: 555 * MIB,
    });
    expect(status.stage).toBe("building the Session");
    expect(status.cancellable).toBe(false);
  });
});
