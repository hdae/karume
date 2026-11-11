import { readFileSync } from "node:fs";

import { EW_CHAIN_RULE_NAME } from "@karume/candidate-rules";
import { parseManifest } from "@karume/hub";
import { runSessionAb, type SessionAbArm, type SessionAbRecord } from "@karume/runtime/tune";
import { describe, expect, it } from "vite-plus/test";

import type { RecordEnvironment } from "../src/lib/environment.ts";
import {
  type DitInput,
  MOSS_DIT_WORK,
  mossAbModel,
  mossDitFeeds,
  sessionAbMossRequest,
  steppedArm,
} from "../src/models/session-ab/moss.ts";
import {
  buildSessionAbMossPageRecord,
  SESSION_AB_MOSS_PAGE_FORMAT,
} from "../src/models/session-ab/record.ts";
import { INITIAL_SESSION_AB_FORM, parseSessionAbForm } from "../src/models/session-ab/request.ts";
import type { ArmStep } from "../src/models/session-ab/run.ts";
import { recordedCounts } from "../src/models/session-ab/view.ts";

import { abRecord, steady } from "./helpers/session-ab-records.ts";

/** The DiT's inputs as its graph declares them, at small widths (the real ones are 128 / 3072 / 256). */
const DIT_INPUTS: readonly DitInput[] = [
  { name: "latent_tokens", dtype: "f32", shape: [1, "S", 4] },
  { name: "text_context", dtype: "f32", shape: [1, "S", 6] },
  { name: "is_image", dtype: "bool", shape: [1, "S", 1] },
  { name: "rope_cos", dtype: "f32", shape: [1, 1, "S", 2] },
  { name: "rope_sin", dtype: "f32", shape: [1, 1, "S", 2] },
  { name: "timesteps_proj", dtype: "f32", shape: [1, 8] },
];

const IMAGE = 6;
const TEXT = 3;

const wordsOf = (data: Float32Array | Int32Array | Uint32Array): readonly number[] => [
  ...new Uint32Array(data.buffer, data.byteOffset, data.length),
];

const settings = (form = {}) => {
  const parsed = parseSessionAbForm({ ...INITIAL_SESSION_AB_FORM, backend: "d3d12", ...form });
  if (!parsed.ok) throw new Error(parsed.errors.join(" "));
  return parsed.value;
};

const hex = (character: string): string => character.repeat(64);

/** A `karume/5` manifest with the MOSS-Image mirror's shape: one model, two presets of `dit`. */
const manifest = parseManifest(
  JSON.stringify({
    format: "karume/5",
    generator: "test",
    defaultModel: "sft",
    models: {
      sft: {
        pipeline: "moss-image/1",
        assets: {},
        weights: {
          dit: {
            i8: {
              container: {
                descriptor: {
                  graph: { length: 8, sha256: hex("d") },
                  model: { length: 8, sha256: hex("e") },
                },
                parts: [
                  { path: "sft/dit/model-00001-of-00002.krm", size: 40, sha256: hex("a") },
                  { path: "sft/dit/model-00002-of-00002.krm", size: 8, sha256: hex("b") },
                ],
              },
            },
          },
        },
        quants: {
          "f32+dit8": { weights: { dit: "i8" }, session: {} },
          "f32+dit8-a8-attn8-online": {
            weights: { dit: "i8" },
            session: {
              linearCompute: "a8",
              attentionCompute: "a8",
              attentionScoreStorage: "none",
            },
          },
        },
        defaultQuant: "f32+dit8-a8-attn8-online",
        pipelineConfig: {},
      },
    },
  }),
);

const MODEL = mossAbModel(manifest, "f32+dit8-a8-attn8-online");

const ENVIRONMENT: RecordEnvironment = {
  recordedAt: "2026-11-11T12:00:00.000Z",
  userAgent: "Mozilla/5.0 Chrome/150.0.0.0",
  crossOriginIsolated: true,
  adapter: { vendor: "nvidia", architecture: "blackwell", device: "", description: "" },
  device: {
    features: [],
    wgslLanguageFeatures: [],
    subgroupLanesLinear: false,
    gpuTimingEnabled: false,
    limits: {
      maxComputeWorkgroupStorageSize: 32768,
      maxStorageBufferBindingSize: 2 ** 31,
      maxBufferSize: 2 ** 31,
    },
  },
};

/** A runner record whose arms carry the counts a `sessionArm` reports. */
const countedRecord = (): SessionAbRecord => {
  const record = abRecord({ base: steady(1792), rule: steady(1620) });
  const [base, rule] = record.arms;
  return {
    ...record,
    arms: [
      { ...base, fusions: { silu: 1 }, dispatches: 2466 },
      { ...rule, fusions: { silu: 1, [EW_CHAIN_RULE_NAME]: 292 }, dispatches: 1880 },
    ],
  };
};

const pageRecordOf = (
  record: SessionAbRecord,
  request = sessionAbMossRequest(MODEL, settings({ rounds: "3" }), EW_CHAIN_RULE_NAME),
) =>
  buildSessionAbMossPageRecord({
    environment: ENVIRONMENT,
    versions: { runtime: "9.8.7", models: "6.5.4" },
    request,
    distribution: { directory: "models/karume-moss-image", manifestSha256: hex("f") },
    source: { kind: "direct" },
    container: { component: "dit", part0Sha256: hex("c") },
    ruleName: EW_CHAIN_RULE_NAME,
    record,
    judgements: [],
  });

describe("the session A/B page's MOSS-Image DiT target", () => {
  describe("the feeds", () => {
    const feeds = mossDitFeeds(DIT_INPUTS, IMAGE, TEXT, MOSS_DIT_WORK.seed);

    it("bind S to the image and text tokens, input by input", () => {
      expect(
        Object.fromEntries(Object.entries(feeds).map(([name, { shape }]) => [name, shape])),
      ).toEqual({
        latent_tokens: [1, 9, 4],
        text_context: [1, 9, 6],
        is_image: [1, 9, 1],
        rope_cos: [1, 1, 9, 2],
        rope_sin: [1, 1, 9, 2],
        timesteps_proj: [1, 8],
      });
    });

    it("mark the image tokens, which come first, in is_image", () => {
      expect(feeds.is_image.dtype).toBe("bool");
      expect([...feeds.is_image.data]).toEqual([1, 1, 1, 1, 1, 1, 0, 0, 0]);
    });

    it("draw every f32 value from [−1, 1), and the same words again from the same seed", () => {
      for (const [name, tensor] of Object.entries(feeds)) {
        if (tensor.dtype !== "f32") continue;
        expect(
          [...tensor.data].every((value) => value >= -1 && value < 1),
          name,
        ).toBe(true);
      }
      const again = mossDitFeeds(DIT_INPUTS, IMAGE, TEXT, MOSS_DIT_WORK.seed);
      const other = mossDitFeeds(DIT_INPUTS, IMAGE, TEXT, MOSS_DIT_WORK.seed + 1);
      for (const name of Object.keys(feeds)) {
        const [first, second, third] = [feeds[name], again[name], other[name]];
        expect(wordsOf(second.data), name).toEqual(wordsOf(first.data));
        if (first.dtype === "f32")
          expect(wordsOf(third.data), name).not.toEqual(wordsOf(first.data));
      }
    });

    it("refuse an input they were not made for: another symbol, another dtype, another bool input", () => {
      expect(() =>
        mossDitFeeds([{ name: "x", dtype: "f32", shape: [1, "T"] }], IMAGE, TEXT, 1),
      ).toThrow("input 'x' has symbol 'T'");
      expect(() =>
        mossDitFeeds([{ name: "ids", dtype: "i32", shape: ["S"] }], IMAGE, TEXT, 1),
      ).toThrow("no feed for input 'ids'");
      expect(() =>
        mossDitFeeds([{ name: "mask", dtype: "bool", shape: [1, "S", 1] }], IMAGE, TEXT, 1),
      ).toThrow("no feed for input 'mask'");
    });
  });

  describe("the request", () => {
    it("names the two arms, the knobs of the preset and the 1024 × 1024 sequence (S = 4,159)", () => {
      const request = sessionAbMossRequest(MODEL, settings(), EW_CHAIN_RULE_NAME);
      expect(request).toEqual({
        target: "moss-dit-forward",
        mount: "moss-image",
        component: "dit",
        model: "sft",
        quant: "f32+dit8-a8-attn8-online",
        session: { linearCompute: "a8", attentionCompute: "a8", attentionScoreStorage: "none" },
        rounds: 4,
        minGain: 0.01,
        openPerVisit: true,
        backend: "d3d12",
        imageTokens: 4096,
        textTokens: 63,
        seed: MOSS_DIT_WORK.seed,
        arms: [
          { name: "base", fusionRules: [] },
          { name: "rule", fusionRules: [EW_CHAIN_RULE_NAME] },
        ],
      });
      expect(request.imageTokens + request.textTokens).toBe(4159);
    });

    it("opens per visit whatever the form says: two DiT Sessions never live together", () => {
      const request = sessionAbMossRequest(
        MODEL,
        settings({ openPerVisit: false }),
        EW_CHAIN_RULE_NAME,
      );
      expect(request.openPerVisit).toBe(true);
    });

    it("takes the knobs from the manifest's preset, none for the reference preset, and refuses another", () => {
      expect(mossAbModel(manifest, "f32+dit8")).toEqual({
        model: "sft",
        quant: "f32+dit8",
        session: {},
      });
      expect(() => mossAbModel(manifest, "f16")).toThrow("no preset 'f16' of model 'sft'");
    });
  });

  describe("the arms' steps", () => {
    /** Runs the real runner over two fake arms wrapped by steppedArm, logging what happens when. */
    const observe = async (openPerVisit: boolean) => {
      const log: { readonly entry: string; readonly timed: boolean }[] = [];
      let clockOpen = false;
      let tick = 0;
      const fake = (name: string, dispatches: number): SessionAbArm => ({
        name,
        open: () => {
          log.push({ entry: `${name}:open`, timed: clockOpen });
          return Promise.resolve();
        },
        run: () => {
          log.push({ entry: `${name}:run`, timed: clockOpen });
          return Promise.resolve({ outputs: { y: Float32Array.of(1, 2) } });
        },
        report: () => Promise.resolve({ fusions: { silu: 1 }, dispatches }),
        close: () => {
          log.push({ entry: `${name}:close`, timed: clockOpen });
          return Promise.resolve();
        },
      });
      const onStep = (step: ArmStep): void => {
        log.push({ entry: `${step.arm}:step:${step.step}`, timed: clockOpen });
      };
      const record = await runSessionAb({
        arms: [steppedArm(fake("base", 10), onStep), steppedArm(fake("rule", 7), onStep)],
        rounds: 3,
        openPerVisit,
        environment: {
          adapter: { vendor: "test", architecture: "", device: "", description: "" },
          userAgent: "test",
          backend: "other",
          date: "2026-11-11T00:00:00.000Z",
        },
        clock: () => {
          clockOpen = !clockOpen;
          tick += 1;
          return tick;
        },
      });
      return { log, record };
    };

    for (const openPerVisit of [true, false]) {
      it(`are reported outside the timed spans, around each open, run and close (openPerVisit: ${openPerVisit})`, async () => {
        const { log } = await observe(openPerVisit);
        expect(log.filter(({ timed }) => timed).map(({ entry }) => entry.split(":")[1])).toEqual(
          log.filter(({ timed }) => timed).map(() => "run"),
        );
        expect(log.some(({ timed }) => timed)).toBe(true);
        const steps = log.map(({ entry }) => entry);
        // Every open is preceded by its step, every run followed by its `ran`, every close preceded.
        steps.forEach((entry, index) => {
          const [arm, what] = entry.split(":");
          if (what === "open") expect(steps[index - 1]).toBe(`${arm}:step:open`);
          if (what === "run") expect(steps[index + 1]).toBe(`${arm}:step:ran`);
          if (what === "close") expect(steps[index - 1]).toBe(`${arm}:step:close`);
        });
      });
    }

    it("pass the arm's counts through to the runner's record", async () => {
      const { record } = await observe(true);
      expect(
        record.arms.map(({ name, fusions, dispatches }) => ({ name, fusions, dispatches })),
      ).toEqual([
        { name: "base", fusions: { silu: 1 }, dispatches: 10 },
        { name: "rule", fusions: { silu: 1 }, dispatches: 7 },
      ]);
    });
  });

  describe("the record", () => {
    it("is an envelope of its own around the runtime's record, with the container and the rule's key prefix", () => {
      const record = countedRecord();
      const page = pageRecordOf(record);
      expect(page.format).toBe(SESSION_AB_MOSS_PAGE_FORMAT);
      expect(page.format).not.toBe("karume-playground-session-ab/2");
      expect([page.page, page.target]).toEqual(["session-ab", "moss-dit-forward"]);
      expect(page.container).toEqual({ component: "dit", part0Sha256: hex("c") });
      expect(page.rule).toEqual({
        name: EW_CHAIN_RULE_NAME,
        package: "@karume/candidate-rules",
        keyPrefix: "ext:candidate:",
      });
      expect(page.record).toBe(record);
      expect(JSON.parse(JSON.stringify(page))).toEqual(page);
    });

    it("refuses a runtime record of other arms or rounds, or whose arms reported no counts", () => {
      const record = countedRecord();
      expect(() => pageRecordOf({ ...record, rounds: 4 })).toThrow("is not the request's");
      expect(() => pageRecordOf({ ...record, arms: [...record.arms].reverse() })).toThrow(
        "is not the request's",
      );
      expect(() => pageRecordOf(abRecord({ base: steady(10), rule: steady(9) }))).toThrow(
        "reported no counts",
      );
    });

    it("gives each arm's row the counts the runner's record keeps, and none for an arm without them", () => {
      const [base, rule] = countedRecord().arms;
      expect(recordedCounts(base)).toEqual({ arm: "base", fusions: { silu: 1 }, dispatches: 2466 });
      expect(recordedCounts(rule)?.fusions[EW_CHAIN_RULE_NAME]).toBe(292);
      const [uncounted] = abRecord({ base: steady(10), rule: steady(9) }).arms;
      expect(recordedCounts(uncounted)).toBeUndefined();
    });
  });

  describe("its feeds' twin in the opt-in Node A/B", () => {
    /** The text between the twin markers, `export` aside. */
    const twinOf = (path: string): string => {
      const text = readFileSync(new URL(path, import.meta.url), "utf8");
      const begin = text.indexOf("// ---- The MOSS DiT feeds: twin begin");
      const endMarker = "// ---- The MOSS DiT feeds: twin end ----";
      const end = text.indexOf(endMarker);
      expect(begin, path).toBeGreaterThanOrEqual(0);
      expect(end, path).toBeGreaterThan(begin);
      return text.slice(begin, end + endMarker.length).replace(/^export /gm, "");
    };

    it("is the same text as the page's (a change to one is made to the other)", () => {
      const page = twinOf("../src/models/session-ab/moss.ts");
      expect(page).toContain("const mossDitFeeds = (");
      expect(twinOf("../../../packages/e2e/tests/e2e_moss_session_ab_test.ts")).toBe(page);
    });
  });
});
