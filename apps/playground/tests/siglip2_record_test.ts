import { parseManifest } from "@karume/hub";
import { describe, expect, it } from "vite-plus/test";

import type { PipelineSource } from "../src/lib/distributions.ts";
import type { RecordEnvironment } from "../src/lib/environment.ts";
import type { LoadedImage } from "../src/lib/image.ts";
import { NO_SUBMITS } from "../src/lib/record.ts";
import {
  buildStatus,
  comparisonText,
  embeddingRows,
  summaryRows,
} from "../src/models/siglip2/format.ts";
import {
  assembleRecord,
  type DiagnosticsReport,
  type MeasuredImage,
  RECORD_FORMAT,
  visionDiagnostics,
} from "../src/models/siglip2/record.ts";
import {
  canReuse,
  effectiveSelection,
  type PipelineKey,
  modelFacts,
  modelFactsText,
  requestImages,
  runBlocker,
} from "../src/models/siglip2/request.ts";

const hex = (character: string): string => character.repeat(64);

/** name, input size, embedding width, presets. */
const MODELS: readonly (readonly [string, number, number, readonly string[]])[] = [
  ["base", 224, 768, ["f32"]],
  ["so400m", 384, 1152, ["f32", "f16"]],
];

/** A `karume/5` manifest shaped like the SigLIP 2 distribution: base (default) and so400m. */
const manifest = parseManifest(
  JSON.stringify({
    format: "karume/5",
    generator: "test",
    defaultModel: "base",
    models: Object.fromEntries(
      MODELS.map(([name, size, hidden, quants]) => [
        name,
        {
          pipeline: "siglip2/1",
          assets: {},
          weights: {
            vision: {
              f32: {
                container: {
                  descriptor: {
                    graph: { length: 8, sha256: hex("d") },
                    model: { length: 8, sha256: hex("e") },
                  },
                  parts: [
                    { path: `${name}/vision-00001-of-00002.krm`, size: 40, sha256: hex("a") },
                    { path: `${name}/vision-00002-of-00002.krm`, size: 8, sha256: hex("b") },
                  ],
                },
              },
            },
          },
          quants: Object.fromEntries(
            quants.map((quant) => [quant, { weights: { vision: "f32" }, session: {} }]),
          ),
          defaultQuant: name === "so400m" ? "f16" : "f32",
          pipelineConfig: {
            imageWidth: size,
            imageHeight: size,
            imageMean: [0.5, 0.5, 0.5],
            imageStd: [0.5, 0.5, 0.5],
            hiddenDim: hidden,
            interpolation: "bilinear",
          },
        },
      ]),
    ),
  }),
);

const image = (name: string): LoadedImage => ({
  name,
  rgba: { data: new Uint8Array(16), width: 2, height: 2 },
  rgb: { data: new Uint8Array(12), width: 2, height: 2 },
});

describe("effectiveSelection", () => {
  it("starts from the manifest's default model and its default preset, both marked default", () => {
    expect(effectiveSelection(manifest, undefined, undefined)).toEqual({
      model: "base",
      quant: "f32",
      modelIsDefault: true,
      quantIsDefault: true,
    });
  });

  it("follows a chosen model to its own default preset", () => {
    expect(effectiveSelection(manifest, "so400m", undefined)).toMatchObject({
      model: "so400m",
      quant: "f16",
      modelIsDefault: false,
    });
  });

  it("keeps a chosen preset the model has and drops one it does not have", () => {
    expect(effectiveSelection(manifest, "so400m", "f32")).toMatchObject({
      quant: "f32",
      quantIsDefault: false,
    });
    expect(effectiveSelection(manifest, "base", "f16")).toMatchObject({
      quant: "f32",
      quantIsDefault: true,
    });
  });

  it("falls back to the default model for a name the manifest does not have", () => {
    expect(effectiveSelection(manifest, "large", undefined).model).toBe("base");
  });
});

describe("modelFacts", () => {
  it("reads the input size and embedding width the model declares", () => {
    const facts = modelFacts(manifest, "so400m");
    expect(facts).toEqual({ imageWidth: 384, imageHeight: 384, hiddenDim: 1152 });
    expect(modelFactsText(facts)).toBe("384×384 input, 1152-dim embedding");
  });

  it("leaves out what a config does not declare as a positive integer", () => {
    expect(modelFacts(manifest, "missing")).toEqual({});
    expect(modelFactsText({ hiddenDim: 768 })).toBe("768-dim embedding");
  });
});

describe("requestImages and runBlocker", () => {
  const ready = { gpuReady: true, manifestReady: true, running: false };

  it("orders A before B and leaves out an empty slot", () => {
    expect(requestImages(image("a.png"), image("b.png")).map(({ slot }) => slot)).toEqual([
      "a",
      "b",
    ]);
    expect(requestImages(undefined, image("b.png")).map(({ slot }) => slot)).toEqual(["b"]);
  });

  it("can run with image A alone", () => {
    expect(runBlocker({ ...ready, images: requestImages(image("a.png"), undefined) })).toBe(
      undefined,
    );
  });

  it("needs image A even when B is there", () => {
    expect(runBlocker({ ...ready, images: requestImages(undefined, image("b.png")) })).toBe(
      "Choose image A.",
    );
  });

  it("names the GPU, the manifest or a running run first", () => {
    const images = requestImages(image("a.png"), undefined);
    expect(runBlocker({ ...ready, gpuReady: false, images })).toBe("The GPU is not ready.");
    expect(runBlocker({ ...ready, manifestReady: false, images })).toBe(
      "The manifest is not loaded.",
    );
    expect(runBlocker({ ...ready, running: true, images })).toBe("A run is in progress.");
  });
});

const buildStats = {
  partCount: 4,
  supplyWaitMs: 1200.04,
  decodeMs: 0,
  bufferCreateMs: 2.26,
  writeBufferIssueMs: 30,
  uploadFenceMs: 400.56,
  uploadedBytes: 371_000_000,
};

/** One report; `runsSoFar` = the Session's runs including this one (3 submits, 120 dispatches each). */
const report = (timingNs: number | undefined, runsSoFar = 1): DiagnosticsReport => ({
  weights: { allocatedBytes: 2 ** 29 },
  submit: { submitCount: 3 * runsSoFar, dispatchCount: 120 * runsSoFar },
  geometryProfile: "nvidia-ampere",
  buildStats,
  lastRunPipelines: [{ key: "linear:v1:f32", dispatchCount: 96 }],
  lastRunTiming:
    timingNs === undefined
      ? undefined
      : {
          entries: [{ key: "linear:v1:f32", ns: timingNs, dispatchCount: 96, workgroupCount: 960 }],
          totalNs: timingNs,
          dispatchCount: 96,
          clampedNegativeSamples: 0,
        },
});

describe("visionDiagnostics", () => {
  it("counts one run per embedding and keeps the Session's build cost", () => {
    const vision = visionDiagnostics([report(undefined, 1), report(undefined, 2)], NO_SUBMITS);
    expect(vision?.runs).toBe(2);
    expect(vision?.build.supplyWaitMs).toBe(1200);
    expect(vision?.timing).toBeUndefined();
  });

  it("sums the GPU time of the timed runs", () => {
    const vision = visionDiagnostics([report(2_000_000, 1), report(3_000_000, 2)], NO_SUBMITS);
    expect(vision?.timing?.runs).toBe(2);
    expect(vision?.timing?.totalMs).toBe(5);
  });

  it("has nothing to say without a report", () => {
    expect(visionDiagnostics([], NO_SUBMITS)).toBeUndefined();
  });

  it("counts only this run's submits and dispatches on a Session kept from earlier runs", () => {
    // An earlier run left the Session at 3 submits / 120 dispatches; this run's one embed adds as many.
    const reused = visionDiagnostics([report(undefined, 2)], {
      submitCount: 3,
      dispatchCount: 120,
    });
    const fresh = visionDiagnostics([report(undefined, 1)], NO_SUBMITS);
    expect(reused).toMatchObject({
      submitCount: 3,
      dispatchCount: 120,
      sessionSubmitCount: 6,
      sessionDispatchCount: 240,
    });
    expect(reused?.submitCount).toBe(fresh?.submitCount);
    expect(reused?.dispatchCount).toBe(fresh?.dispatchCount);
  });

  it("refuses counters that went down (a baseline from another Session)", () => {
    expect(() =>
      visionDiagnostics([report(undefined, 1)], { submitCount: 30, dispatchCount: 1200 }),
    ).toThrow(/went down/);
  });
});

const environment: RecordEnvironment = {
  recordedAt: "2026-10-28T00:00:00.000Z",
  userAgent: "test",
  crossOriginIsolated: true,
  adapter: { vendor: "nvidia", architecture: "blackwell", device: "", description: "D3D12" },
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

const measured = (slot: "a" | "b", values: readonly number[]): MeasuredImage => ({
  slot,
  name: `${slot}.png`,
  width: 640,
  height: 480,
  rgbSha256: hex(slot),
  embedding: new Float32Array(values),
  embeddingSha256: hex(slot === "a" ? "1" : "2"),
  embedMs: slot === "a" ? 812.3456 : 41.2,
});

const assemble = (images: readonly MeasuredImage[], buildMs: number | undefined) =>
  assembleRecord({
    environment,
    directory: "models/karume-siglip2",
    manifestSha256: hex("f"),
    source: {
      kind: "cached",
      ref: { repo: "hdae/karume-siglip2", revision: "0".repeat(40), hubUrl: "http://x/hf" },
      cacheGeneration: 0,
    },
    selection: effectiveSelection(manifest, undefined, undefined),
    facts: modelFacts(manifest, "base"),
    presetLabel: "f32",
    images,
    buildMs,
    wallMs: 2500,
    reports: images.map((_, index) => report(undefined, index + 1)),
    countersBefore: NO_SUBMITS,
  });

describe("assembleRecord", () => {
  it("records two images, their embeddings with sha256 and the cosine of the pair", () => {
    const record = assemble([measured("a", [1, 0, 0]), measured("b", [1, 1, 0])], 1500);
    expect(record.format).toBe(RECORD_FORMAT);
    expect(record.page).toBe("siglip2");
    expect(record.source).toEqual({
      kind: "cached",
      repo: "hdae/karume-siglip2",
      revision: "0".repeat(40),
      hubUrl: "http://x/hf",
    });
    expect(record.model).toEqual({
      name: "base",
      isDefault: true,
      imageWidth: 224,
      imageHeight: 224,
      hiddenDim: 768,
    });
    expect(record.request.images.map(({ slot, rgbSha256 }) => [slot, rgbSha256])).toEqual([
      ["a", hex("a")],
      ["b", hex("b")],
    ]);
    expect(record.output.embeddings.map(({ sha256 }) => sha256)).toEqual([hex("1"), hex("2")]);
    expect(record.output.comparison).toEqual({ cosineSimilarity: 0.707107 });
    expect(record.timings).toEqual({
      pipelineReused: false,
      buildSeconds: 1.5,
      embedSeconds: [0.812, 0.041],
      wallSeconds: 2.5,
    });
    expect(record.diagnostics.vision).toMatchObject({
      runs: 2,
      submitCount: 6,
      dispatchCount: 240,
    });
  });

  it("writes non-finite leading values so the saved JSON tells them apart", () => {
    const record = assemble([measured("a", [Number.NaN, Number.NEGATIVE_INFINITY, 1])], 10);
    const saved: unknown = JSON.parse(JSON.stringify(record));
    expect(saved).toMatchObject({
      output: { embeddings: [{ nonFinite: 2, first: ["NaN", "-Infinity", 1] }] },
    });
    expect(embeddingRows(record).find(({ label }) => label === "First values")?.values).toEqual([
      "NaN, -Infinity, 1.0000",
    ]);
  });

  it("has no comparison for one image and marks a reused pipeline", () => {
    const record = assemble([measured("a", [1, 2])], undefined);
    expect(record.output.comparison).toBeUndefined();
    expect(record.timings.pipelineReused).toBe(true);
    expect(record.timings.buildSeconds).toBeNull();
    expect(comparisonText(record)).toBeUndefined();
  });

  it("says why there is no cosine when an embedding has non-finite values", () => {
    const record = assemble([measured("a", [1, Number.NaN]), measured("b", [1, 1])], 10);
    expect(record.output.comparison).toMatchObject({ cosineSimilarity: null });
    expect(comparisonText(record)).toMatch(/not finite/);
  });
});

describe("result rows", () => {
  it("puts one column per image in the embeddings table", () => {
    const record = assemble([measured("a", [3, 4]), measured("b", [0, 1])], 10);
    const rows = embeddingRows(record);
    expect(rows.find(({ label }) => label === "L2 norm")?.values).toEqual(["5.000000", "1.000000"]);
    expect(rows.find(({ label }) => label === "Input")?.values).toEqual([
      "a.png · 640×480",
      "b.png · 640×480",
    ]);
    expect(rows.every(({ values }) => values.length === 2)).toBe(true);
  });

  it("says when the build cost comes from an earlier run", () => {
    const rows = new Map(summaryRows(assemble([measured("a", [1])], undefined)));
    expect(rows.get("Build s")).toMatch(/reused/);
    expect(rows.has("Vision build (earlier run)")).toBe(true);
    expect(rows.get("Embed Image A s")).toBe("0.812");
  });
});

describe("canReuse", () => {
  const gpu = { name: "gpu" };
  const source: Extract<PipelineSource, { kind: "cached" }> = {
    kind: "cached",
    ref: { repo: "hdae/karume-siglip2", revision: "0".repeat(40), hubUrl: "http://x/hf" },
    cacheGeneration: 0,
  };
  const direct: PipelineSource = { kind: "direct" };
  const held: PipelineKey<typeof gpu, PipelineSource> = {
    gpu,
    source,
    model: "base",
    quant: "f32",
  };

  it("reuses the held pipeline when the GPU, source, model and preset all match", () => {
    expect(canReuse(held, { gpu, source, model: "base", quant: "f32" })).toBe(true);
  });

  it("builds a new one when any of the four differs", () => {
    expect(canReuse(held, { gpu: { name: "gpu" }, source, model: "base", quant: "f32" })).toBe(
      false,
    );
    expect(canReuse(held, { gpu, source: direct, model: "base", quant: "f32" })).toBe(false);
    expect(canReuse(held, { gpu, source, model: "so400m", quant: "f32" })).toBe(false);
    expect(canReuse(held, { gpu, source, model: "base", quant: "f16" })).toBe(false);
  });

  it("treats a new source object as a new source, even with the same content", () => {
    // A cleared cache bumps the generation; even an equal-looking new object rebuilds (fresh readers).
    expect(
      canReuse(held, {
        gpu,
        source: { ...source, cacheGeneration: 1 },
        model: "base",
        quant: "f32",
      }),
    ).toBe(false);
    expect(canReuse(held, { gpu, source: { ...source }, model: "base", quant: "f32" })).toBe(false);
  });

  it("builds when nothing is held", () => {
    expect(canReuse(undefined, { gpu, source, model: "base", quant: "f32" })).toBe(false);
  });
});

describe("buildStatus", () => {
  const MIB = 2 ** 20;

  it("names the file's own bytes and the load's total apart while fetching", () => {
    const status = buildStatus({
      phase: "downloading",
      path: "base/vision-00002-of-00002.krm",
      fileLoaded: 50 * MIB,
      fileTotal: 200 * MIB,
      loaded: 250 * MIB,
      total: 372 * MIB,
    });
    expect(status.stage).toBe("fetching the weights");
    expect(status.note).toBe(
      "downloading base/vision-00002-of-00002.krm (50 / 200 MiB) · total 250 / 372 MiB",
    );
    // The fetch takes the signal, so Cancel stays offered.
    expect(status.cancellable).toBeUndefined();
  });

  it("says that the Session build follows once every file is fetched, not cancellable", () => {
    const status = buildStatus({
      phase: "complete",
      path: "base/vision-00002-of-00002.krm",
      fileLoaded: 200 * MIB,
      fileTotal: 200 * MIB,
      loaded: 372 * MIB,
      total: 372 * MIB,
    });
    expect(status.stage).toBe("building the Session");
    expect(status.note).toBe("uploading the weights (372 MiB fetched)");
    expect(status.cancellable).toBe(false);
    expect(status.cancelNote).toMatch(/Cancel is unavailable until it ends/);
  });
});
