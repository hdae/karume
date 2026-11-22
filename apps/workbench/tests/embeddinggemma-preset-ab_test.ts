import type { SessionDiagnostics } from "@karume/runtime";
import { judgeSessionAb, runSessionAb } from "@karume/runtime/tune";
import { describe, expect, it } from "vite-plus/test";

import {
  EMBEDDING_OUTPUT,
  type EmbeddingPipeline,
  presetArm,
  probePreset,
} from "../src/embeddinggemma-preset-ab/arm.ts";
import {
  cosine,
  medianRoundMs,
  minimumMs,
  PRESET_AB_FORMAT,
  presetAbData,
  presetAbRequest,
  presetAbResult,
} from "../src/embeddinggemma-preset-ab/data.ts";
import { type PresetAbDevice, runPresetAb } from "../src/embeddinggemma-preset-ab/run.ts";
import {
  cutText,
  GOLDEN_CASES,
  type GoldenCase,
  LONGEST_CASE_TOKENS,
  sourceCase,
} from "../src/embeddinggemma-preset-ab/text.ts";
import presetAb from "../src/entries/local/embeddinggemma-preset-ab.ts";
import { ParameterError, parseParameters } from "../src/entry.ts";

import {
  hex,
  manifestOf,
  manifestText,
  memoryDistributions,
} from "./helpers/session-ab-manifests.ts";
import { spanRunner } from "./helpers/session-ab-runner.ts";

/** The mirror's shape: one model, `bf16` (the default) and `i8`. */
const MANIFEST_TEXT = manifestText({
  pipeline: "embeddinggemma/1",
  models: ["text"],
  component: "model",
  presets: { bf16: {}, i8: {} },
  weights: { bf16: { model: "bf16" }, i8: { model: "i8" } },
  defaultPreset: "bf16",
});
const MANIFEST = manifestOf(MANIFEST_TEXT);

const VALUES = { base: "", compared: "i8", tokens: 512, rounds: 9, minGain: 0.01 };

/** A tokenizer stand-in: `<bos>`, one id per word (a run of non-spaces), `<eos>`, and two for the prompt. */
const wordCount =
  (prompt: string | undefined) =>
  (text: string): number =>
    (prompt === undefined ? 0 : 2) + 2 + (text.match(/\S+/g)?.length ?? 0);

const CASE: GoldenCase = {
  name: "doc",
  prompt: "document",
  text: "one two three four five six seven eight",
  tokens: 12,
};

/** A vector of `width` words, each `value`. */
const vector = (value: number, width = 4): Float32Array<ArrayBuffer> =>
  new Float32Array(width).fill(value);

/** A pipeline stand-in that returns `vectors` in turn (the last one again once they run out). */
const fakePipeline = (
  vectors: readonly Float32Array<ArrayBuffer>[],
  log: string[] = [],
): EmbeddingPipeline => {
  let calls = 0;
  return {
    embed: (text, options) => {
      log.push(`embed ${text} ${options?.prompt ?? "-"}`);
      const next = vectors[Math.min(calls, vectors.length - 1)];
      calls += 1;
      return Promise.resolve(next);
    },
    tokenize: () => Int32Array.of(1, 2, 3),
    dispose: () => {
      log.push("dispose");
      return Promise.resolve();
    },
  };
};

/** The diagnostics fields a probe reads; every other field is left out of the stand-in. */
const diagnosticsOf = (run: number): SessionDiagnostics =>
  ({
    buildStats: {
      partCount: 4,
      supplyWaitMs: 400,
      decodeMs: 0,
      bufferCreateMs: 16,
      writeBufferIssueMs: 150,
      uploadedBytes: 2 ** 29,
      uploadFenceMs: 20,
    },
    weights: {
      allocCount: 400,
      reuseCount: 0,
      allocatedBytes: 2 ** 29,
      transientBytes: 0,
      peakTransientBytes: 0,
    },
    storage: { residentCompressedBytes: 2 ** 28, hostExpandedBytes: 0 },
    planBacking: {
      residentBytes: run === 1 ? 0 : 2 ** 26,
      inputBytes: 0,
      retainedCount: run - 1,
      buildCount: run - 1,
    },
    lastRunPipelines: [
      { key: "a", dispatchCount: 10 },
      { key: "b", dispatchCount: 5 },
    ],
    geometryProfile: "default",
  }) as unknown as SessionDiagnostics;

const CUT = { case: "doc", prompt: "document", text: "one two", codePoints: 7, tokens: 6 };

describe("the entry's declaration", () => {
  it("compares the default preset with i8 at 512 tokens over 9 rounds unless told otherwise", () => {
    expect(parseParameters(presetAb.parameters, {})).toEqual({
      base: "",
      compared: "i8",
      tokens: 512,
      rounds: 9,
      minGain: 0.01,
    });
  });

  it("refuses a length past the longest golden case before anything runs", () => {
    expect(() =>
      parseParameters(presetAb.parameters, { tokens: String(LONGEST_CASE_TOKENS + 1) }),
    ).toThrow(ParameterError);
  });
});

describe("the text", () => {
  it("is cut from the shortest golden case that is long enough", () => {
    expect(sourceCase(GOLDEN_CASES, 16).name).toBe("query-en");
    expect(sourceCase(GOLDEN_CASES, 512).name).toBe("document-609");
    expect(sourceCase(GOLDEN_CASES, 2048).name).toBe("document-2048");
    expect(sourceCase(GOLDEN_CASES, 8192).name).toBe("document-8192");
    expect(() => sourceCase(GOLDEN_CASES, LONGEST_CASE_TOKENS + 1)).toThrow(/No golden case/);
  });

  it("is the shortest prefix whose ids are exactly the asked length", () => {
    const cut = cutText(CASE, 7, wordCount("document"));
    expect(cut).toEqual({
      case: "doc",
      prompt: "document",
      text: "one two t",
      codePoints: 9,
      tokens: 7,
    });
    expect(wordCount("document")(cut.text)).toBe(7);
  });

  it("reaches the case's own length on a prefix of its text", () => {
    const cut = cutText(CASE, 12, wordCount("document"));
    expect(cut.text).toBe("one two three four five six seven e");
    expect(CASE.text.startsWith(cut.text)).toBe(true);
  });

  it("keeps a case without a prompt without one", () => {
    const bare = { name: "bare", text: "a b c", tokens: 5 };
    expect(cutText(bare, 4, wordCount(undefined))).toEqual({
      case: "bare",
      text: "a b",
      codePoints: 3,
      tokens: 4,
    });
  });

  it("cuts in whole code points", () => {
    const wide = { name: "wide", text: "𝒜𝒜 𝒜", tokens: 4 };
    // One id per code point, `<bos>` and `<eos>`.
    const perCodePoint = (text: string): number => Array.from(text).length + 2;
    expect(cutText(wide, 3, perCodePoint)).toMatchObject({ text: "𝒜", codePoints: 1 });
  });

  it("fails loudly when no prefix has the length", () => {
    // The prompt, <bos> and <eos> are 4 ids before any word.
    expect(() => cutText(CASE, 3, wordCount("document"))).toThrow(/No prefix .* exactly 3 tokens/);
    // Every step is two ids: an odd length is stepped over.
    const twoPerWord = (text: string): number => 2 + 2 * (text.match(/\S+/g)?.length ?? 0);
    expect(() => cutText(CASE, 5, twoPerWord)).toThrow(/No prefix/);
    expect(() => cutText(CASE, 13, wordCount("document"))).toThrow(/fewer than the 13/);
  });
});

describe("the request", () => {
  it("names the default model and resolves an empty base to its default preset", () => {
    expect(presetAbRequest(MANIFEST, VALUES)).toEqual({
      mount: "embeddinggemma-2",
      model: "text",
      base: "bf16",
      compared: "i8",
      tokens: 512,
      rounds: 9,
      minGain: 0.01,
    });
  });

  it("refuses a preset the model does not have, naming the parameter and the presets", () => {
    expect(() => presetAbRequest(MANIFEST, { ...VALUES, compared: "i4" })).toThrow(
      "Parameter compared: 'i4' is not a preset of model 'text'; its presets are bf16, i8",
    );
    expect(() => presetAbRequest(MANIFEST, { ...VALUES, base: "f32" })).toThrow(
      /^Parameter base: 'f32'/,
    );
  });

  it("refuses an empty or a repeated compared preset", () => {
    expect(() => presetAbRequest(MANIFEST, { ...VALUES, compared: "" })).toThrow(
      /Parameter compared: names no preset/,
    );
    expect(() => presetAbRequest(MANIFEST, { ...VALUES, compared: "bf16" })).toThrow(
      "Parameter compared: 'bf16' is the base preset too",
    );
  });
});

describe("an arm", () => {
  it("runs one embed of the text with its prompt inside the runner's timed span, and nothing else", async () => {
    const runner = spanRunner();
    const log: string[] = [];
    const inSpan: boolean[] = [];
    const pipelineOf = (value: number): EmbeddingPipeline => {
      const inner = fakePipeline([vector(value)], log);
      return {
        ...inner,
        embed: (text, options) => {
          inSpan.push(runner.inSpan());
          return inner.embed(text, options);
        },
      };
    };
    const signal = new AbortController().signal;
    const arms = [
      presetArm({ name: "bf16", text: CUT, build: () => Promise.resolve(pipelineOf(1)), signal }),
      presetArm({ name: "i8", text: CUT, build: () => Promise.resolve(pipelineOf(2)), signal }),
    ];
    const record = await runner.run(arms, 3, false);
    // Warm-up and three timed rounds, two visits of each arm per round.
    expect(log.filter((line) => line === "embed one two document")).toHaveLength(16);
    expect(inSpan.slice(4).every(Boolean)).toBe(true);
    expect(log.filter((line) => line === "dispose")).toHaveLength(2);
    expect(arms.map((arm) => arm.facts().runs)).toEqual([8, 8]);
    // The runner's identity holds against the base's first vector, which the other preset differs from.
    expect(record.arms[1].identity.worst.failures).toBe(4);
    expect(arms.map((arm) => arm.facts().differingRuns)).toEqual([0, 0]);
  });

  it("counts the runs whose vector differs from the arm's own first", async () => {
    const arm = presetArm({
      name: "i8",
      text: CUT,
      build: () => Promise.resolve(fakePipeline([vector(1), vector(1), vector(3)])),
      signal: new AbortController().signal,
      clock: (() => {
        let tick = 0;
        return () => (tick += 5);
      })(),
    });
    await arm.open();
    for (let run = 0; run < 3; run += 1) {
      await arm.run();
      await arm.report?.();
    }
    const facts = arm.facts();
    expect(facts).toMatchObject({ openMs: [5], runs: 3, differingRuns: 1 });
    expect(facts.first).toEqual(vector(1));
  });

  it("stops at the report after a cancel, outside the timed span", async () => {
    const controller = new AbortController();
    const arm = presetArm({
      name: "bf16",
      text: CUT,
      build: () => Promise.resolve(fakePipeline([vector(1)])),
      signal: controller.signal,
    });
    await arm.open();
    await arm.run();
    controller.abort(new Error("cancelled"));
    await expect(arm.report?.()).rejects.toThrow("cancelled");
  });
});

describe("a probe", () => {
  it("times the load and two embeds, reads the second run's diagnostics and disposes the pipeline", async () => {
    const log: string[] = [];
    let tick = 0;
    const result = await probePreset({
      preset: "i8",
      build: (sink) => {
        tick += 100;
        const inner = fakePipeline([vector(1)], log);
        let runs = 0;
        return Promise.resolve({
          ...inner,
          embed: async (text, options) => {
            runs += 1;
            tick += runs === 1 ? 30 : 10;
            const out = await inner.embed(text, options);
            sink(diagnosticsOf(runs));
            return out;
          },
        });
      },
      textOf: () => CUT,
      clock: () => tick,
    });
    expect(result.text).toBe(CUT);
    expect(result.probe).toMatchObject({
      preset: "i8",
      loadMs: 100,
      firstRunMs: 30,
      secondRunMs: 10,
      planBacking: { residentBytes: 2 ** 26 },
      dispatches: 15,
      geometryProfile: "default",
    });
    expect(log).toEqual(["embed one two document", "embed one two document", "dispose"]);
  });

  it("disposes the pipeline when the text cannot be cut", async () => {
    const log: string[] = [];
    await expect(
      probePreset({
        preset: "bf16",
        build: () => Promise.resolve(fakePipeline([vector(1)], log)),
        textOf: () => {
          throw new Error("no prefix");
        },
      }),
    ).rejects.toThrow("no prefix");
    expect(log).toEqual(["dispose"]);
  });
});

describe("the data and the summary", () => {
  const request = presetAbRequest(MANIFEST, { ...VALUES, rounds: 3 });
  const facts = (first: Float32Array<ArrayBuffer>) => ({
    openMs: [900],
    runs: 8,
    differingRuns: 0,
    first,
  });
  // Not parallel, so a pairing of an arm's vector with itself (cosine 1) would show.
  const BASE_FIRST = Float32Array.of(3, 4, 0, 0);
  const COMPARED_FIRST = Float32Array.of(4, 3, 0, 0);
  const probe = (preset: string) => ({
    preset,
    loadMs: 1000,
    firstRunMs: 100,
    secondRunMs: 50,
    build: diagnosticsOf(2).buildStats,
    weights: diagnosticsOf(2).weights,
    storage: diagnosticsOf(2).storage,
    planBacking: diagnosticsOf(2).planBacking,
    dispatches: 853,
    geometryProfile: "default",
  });
  /**
   * The real runner over two constant arms; the clock gives each timed run its time from
   * `TIMES` (per round: bf16, i8, i8, bf16) and every warm-up run 0 ms.
   */
  const TIMES = [10, 30, 31, 12, 11, 29, 30, 12, 10, 31, 30, 11];
  const recordOf = () => {
    const arms = ["bf16", "i8"].map((name) => ({
      name,
      open: () => Promise.resolve(),
      run: () =>
        Promise.resolve({ outputs: { [EMBEDDING_OUTPUT]: vector(name === "bf16" ? 1 : 2) } }),
      close: () => Promise.resolve(),
    }));
    // Two reads per run (its start and its end); the warm-up round's four runs come first.
    let now = 0;
    let reads = 0;
    const clock = (): number => {
      reads += 1;
      if (reads > 8 && reads % 2 === 0) now += TIMES[(reads - 10) / 2];
      return now;
    };
    return runSessionAb({
      arms,
      rounds: 3,
      environment: {
        adapter: { vendor: "test", architecture: "", device: "", description: "" },
        userAgent: "test",
        backend: "other",
        date: "2026-11-22T00:00:00.000Z",
      },
      clock,
    });
  };

  it("keeps the record, the judgement without identity, each arm's median and the cosine", async () => {
    const record = await recordOf();
    const judgements = judgeSessionAb(record, { minGain: 0.01, requireIdentity: false });
    const data = presetAbData({
      request,
      distribution: {
        directory: "models/karume-embeddinggemma-2",
        manifestSha256: hex("f"),
        read: "directory",
      },
      text: CUT,
      probes: [probe("bf16"), probe("i8")],
      arms: [facts(BASE_FIRST), facts(COMPARED_FIRST)],
      record,
      judgements,
    });
    expect(data.format).toBe(PRESET_AB_FORMAT);
    expect(data.text).toEqual({ case: "doc", prompt: "document", codePoints: 7, tokens: 6 });
    expect(data.arms.map(({ medianMs }) => medianMs)).toEqual(
      record.arms.map(({ timesMs }) => medianRoundMs(timesMs)),
    );
    expect(data.arms[0].medianMs).toBe(judgements[0].baseMedianMs);
    expect(data.arms.map(({ minMs }) => minMs)).toEqual([10, 29]);
    expect(data.cosine).toBe(24 / 25);
    expect(data.judgement).toBe(judgements[0]);
    const { summary, verdict } = presetAbResult(data);
    expect(verdict).toBeUndefined();
    expect(summary[0]).toBe(
      "512 tokens: a prefix of doc (7 code points, prompt document), 3 rounds",
    );
    expect(summary[1]).toMatch(
      /^Median embed .*: bf16 .* ms · i8 .* ms; fastest visit bf16 10\.0 ms · i8 29\.0 ms$/,
    );
    expect(summary[2]).toMatch(/^i8 against bf16: /);
    expect(summary.at(-1)).toBe("Cosine of i8 against bf16: 0.9600000");
  });

  it("refuses a record of other arms or rounds, probes of other presets and an arm without a first vector", async () => {
    const record = await recordOf();
    const judgements = judgeSessionAb(record, { minGain: 0.01, requireIdentity: false });
    const input = {
      request,
      distribution: { directory: "d", manifestSha256: hex("f"), read: "directory" as const },
      text: CUT,
      probes: [probe("bf16"), probe("i8")] as const,
      arms: [facts(BASE_FIRST), facts(COMPARED_FIRST)] as const,
      record,
      judgements,
    };
    expect(() => presetAbData({ ...input, request: { ...request, rounds: 4 } })).toThrow(
      /is not the request's/,
    );
    expect(() => presetAbData({ ...input, probes: [probe("i8"), probe("bf16")] })).toThrow(
      /The probes \(i8, bf16\)/,
    );
    expect(() =>
      presetAbData({
        ...input,
        arms: [facts(BASE_FIRST), { openMs: [], runs: 0, differingRuns: 0 }],
      }),
    ).toThrow("The arm i8 has no first vector");
  });
});

describe("the median and the cosine", () => {
  it("keeps the fastest visit of any round", () => {
    expect(
      minimumMs([
        [12, 11],
        [30, 9],
      ]),
    ).toBe(9);
  });

  it("takes the median over the rounds of each round's mean", () => {
    expect(
      medianRoundMs([
        [10, 12],
        [30, 30],
        [20, 22],
      ]),
    ).toBe(21);
    expect(
      medianRoundMs([
        [10, 10],
        [20, 20],
      ]),
    ).toBe(15);
  });

  it("is 1 for one direction, 0 for orthogonal vectors, and refuses two widths", () => {
    expect(cosine(Float32Array.of(1, 2), Float32Array.of(2, 4))).toBeCloseTo(1, 12);
    expect(cosine(Float32Array.of(1, 0), Float32Array.of(0, 3))).toBe(0);
    expect(() => cosine(Float32Array.of(1), Float32Array.of(1, 2))).toThrow(/widths differ/);
  });
});

describe("the entry's run", () => {
  /** Each preset's vector: not parallel, so the cosine shows which first vector went where. */
  const VECTORS: Readonly<Record<string, Float32Array<ArrayBuffer>>> = {
    bf16: Float32Array.of(3, 4, 0, 0),
    i8: Float32Array.of(4, 3, 0, 0),
  };

  /**
   * A device whose pipelines tokenize one id per code point (plus `<bos>`, `<eos>` and two for a
   * prompt) and embed their preset's vector after `delayMs` (so the runner reads a time above 0),
   * sending the diagnostics of each run to the sink when there is one. `idOffset` shifts a preset's
   * ids. `log` gets the builds, the disposes and the device's destroy.
   */
  const fakeDevice = (
    log: string[],
    idOffset: Readonly<Record<string, number>> = {},
  ): PresetAbDevice => ({
    adapterInfo: { vendor: "test", architecture: "", device: "", description: "" },
    pipeline: (_ref, options) => {
      const { quant, onRunDiagnostics } = options;
      log.push(`build ${quant}${onRunDiagnostics === undefined ? "" : " with diagnostics"}`);
      let runs = 0;
      return Promise.resolve({
        tokenize: (text, tokenizeOptions) =>
          Int32Array.from(
            {
              length: (tokenizeOptions?.prompt === undefined ? 0 : 2) + 2 + Array.from(text).length,
            },
            (_, index) => index + (idOffset[quant] ?? 0),
          ),
        embed: async () => {
          runs += 1;
          await new Promise((resolve) => setTimeout(resolve, quant === "bf16" ? 2 : 1));
          onRunDiagnostics?.(diagnosticsOf(runs));
          return VECTORS[quant].slice();
        },
        dispose: () => {
          log.push(`dispose ${quant}`);
          return Promise.resolve();
        },
      });
    },
    destroy: () => {
      log.push("destroy");
    },
  });

  const contextOf = (progressLines: string[]) => ({
    parameters: parseParameters(presetAb.parameters, { tokens: "16", rounds: "3" }),
    gpuOptions: {},
    signal: new AbortController().signal,
    progress: (line: string) => {
      progressLines.push(line);
    },
    openDistribution: memoryDistributions({ "embeddinggemma-2": MANIFEST_TEXT }),
    openInput: () => Promise.reject(new Error("the entry reads no input file")),
    backend: "test",
  });

  it("probes the base, then the compared preset, times both arms and keeps the data base first", async () => {
    const log: string[] = [];
    const progressLines: string[] = [];
    const result = await runPresetAb(contextOf(progressLines), () =>
      Promise.resolve(fakeDevice(log)),
    );
    expect(result.verdict).toBeUndefined();
    expect(result.data).toMatchObject({
      format: PRESET_AB_FORMAT,
      request: {
        mount: "embeddinggemma-2",
        model: "text",
        base: "bf16",
        compared: "i8",
        tokens: 16,
      },
      distribution: { directory: "models/karume-embeddinggemma-2" },
      // query-en with its prompt: 2 ids for the prompt, <bos>, <eos> and 12 code points.
      text: { case: "query-en", prompt: "query", codePoints: 12, tokens: 16 },
      probes: [
        { preset: "bf16", dispatches: 15 },
        { preset: "i8", dispatches: 15 },
      ],
      arms: [
        { name: "bf16", runs: 8, differingRuns: 0 },
        { name: "i8", runs: 8, differingRuns: 0 },
      ],
      cosine: 24 / 25,
      record: { rounds: 3, arms: [{ name: "bf16" }, { name: "i8" }] },
      judgement: { base: "bf16", arm: "i8" },
    });
    expect(result.data).not.toHaveProperty("text.text");
    // The probes build, run and dispose one after the other; the arms are built once each.
    expect(log.slice(0, 4)).toEqual([
      "build bf16 with diagnostics",
      "dispose bf16",
      "build i8 with diagnostics",
      "dispose i8",
    ]);
    expect(log.slice(4, 6)).toEqual(["build bf16", "build i8"]);
    expect(log.slice(6, 8).toSorted()).toEqual(["dispose bf16", "dispose i8"]);
    expect(log.at(-1)).toBe("destroy");
    expect(log).toHaveLength(9);
    expect(progressLines.slice(0, 2)).toEqual([
      "probing bf16: loading, cutting query-en to 16 tokens, two embeds",
      "probing i8: loading, two embeds",
    ]);
  });

  it("refuses a compared preset whose pipeline reads the text as other ids, and releases the device", async () => {
    const log: string[] = [];
    await expect(
      runPresetAb(contextOf([]), () => Promise.resolve(fakeDevice(log, { i8: 1 }))),
    ).rejects.toThrow("The i8 pipeline reads the text as other token ids than the bf16 pipeline");
    expect(log).toEqual([
      "build bf16 with diagnostics",
      "dispose bf16",
      "build i8 with diagnostics",
      "dispose i8",
      "destroy",
    ]);
  });
});
