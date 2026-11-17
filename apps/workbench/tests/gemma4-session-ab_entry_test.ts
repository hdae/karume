import { MountUnavailableError } from "@karume/local-distributions/browser";
import { GELU_MUL_RULE_NAME } from "@karume/candidate-rules";
import { FITTED_EXPERT_SLOTS, type ModelKey } from "@karume/gemma4-app-support/settings";
import { judgeSessionAb, type SessionAbRecord } from "@karume/runtime/tune";
import { describe, expect, it } from "vite-plus/test";

import gemmaSessionAb from "../src/entries/local/gemma4-session-ab.ts";
import { ParameterError, parseParameters } from "../src/entry.ts";
import { type SessionAbArmProbe, sessionAbGemmaData } from "../src/session-ab/data.ts";
import { gemmaRequest, openGemmaModel, parseModelText } from "../src/session-ab/gemma-target.ts";
import { SESSION_AB_PROMPT } from "../src/session-ab/gemma-request.ts";
import { sessionAbGemmaResult } from "../src/session-ab/summary.ts";

import {
  gemmaManifestText,
  hex,
  manifestOf,
  memoryDistributions,
} from "./helpers/session-ab-manifests.ts";
import { abRecord, DIFFERING, steady } from "./helpers/session-ab-records.ts";

const QAT = gemmaManifestText(["12b", "26b", "e2b"]);
const MOBILE = gemmaManifestText(["e2b", "e4b", "12b"]);
const REGULAR = gemmaManifestText(["e4b", "e2b"]);

const SIGNAL = new AbortController().signal;

/** The typed parameters of `given` (strings, as a URL or a command line gives them). */
const parameters = (given: Readonly<Record<string, string>> = {}) =>
  parseParameters(gemmaSessionAb.parameters, given);

const QAT_26B: ModelKey = { mount: "gemma4-qat", model: "26b" };

const requestOf = (given: Readonly<Record<string, string>>, key: ModelKey = QAT_26B) =>
  gemmaRequest(parameters(given), key, manifestOf(QAT), GELU_MUL_RULE_NAME);

describe("the gemma4-session-ab entry", () => {
  describe("its parameters", () => {
    it("default to the default model and preset, 96 tokens, 4 rounds, 1%, open per visit, probed, subgroups when the device has them, no cache, cached", () => {
      expect(parameters()).toEqual({
        model: "",
        preset: "",
        tokens: 96,
        rounds: 4,
        minGain: 0.01,
        openPerVisit: true,
        probeCounts: true,
        f16Kv: false,
        subgroups: true,
        expertCacheSlots: "",
        hostTier: "",
        depth: 0,
        source: "cached",
      });
    });

    it("take 2 to 512 tokens: a run of one token has no decode run to count", () => {
      for (const tokens of ["2", "512"]) expect(parameters({ tokens }).tokens).toBe(Number(tokens));
      for (const tokens of ["1", "513"]) {
        expect(() => parameters({ tokens }), tokens).toThrow(ParameterError);
      }
    });

    it("take a self-speculation depth from 0 (off) to 4", () => {
      expect(parameters({ depth: "4" }).depth).toBe(4);
      expect(() => parameters({ depth: "5" })).toThrow(ParameterError);
      expect(() => parameters({ depth: "-1" })).toThrow(ParameterError);
    });
  });

  describe("the model", () => {
    it("is read as <mount>/<model>, the empty text being the default", () => {
      expect(parseModelText("")).toBeUndefined();
      expect(parseModelText("gemma4-qat-mobile/e4b")).toEqual({
        mount: "gemma4-qat-mobile",
        model: "e4b",
      });
    });

    it("refuses text that is not <mount>/<model>, and a mount that is not a Gemma mount, naming the mounts", () => {
      for (const text of ["gemma4", "gemma4/", "/e2b"]) {
        expect(() => parseModelText(text), text).toThrow(
          `Parameter model: expected <mount>/<model> with a mount of gemma4, gemma4-qat, gemma4-qat-mobile (empty: the default), got "${text}"`,
        );
      }
      expect(() => parseModelText("anima/e2b")).toThrow(
        "Parameter model: 'anima' is not one of the mounts gemma4, gemma4-qat, gemma4-qat-mobile",
      );
    });

    it("defaults to the default model of the first mount whose mirror is present", async () => {
      const opened: string[] = [];
      const model = await openGemmaModel(
        undefined,
        "direct",
        memoryDistributions({ "gemma4-qat": QAT, "gemma4-qat-mobile": MOBILE }, opened),
        SIGNAL,
      );
      expect(model.key).toEqual({ mount: "gemma4-qat", model: "12b" });
      expect(model.opened.facts.directory).toBe("models/karume-gemma4-qat");
      expect(opened).toEqual(["gemma4:direct", "gemma4-qat:direct"]);
      const regular = await openGemmaModel(
        undefined,
        "cached",
        memoryDistributions({ gemma4: REGULAR, "gemma4-qat": QAT }),
        SIGNAL,
      );
      expect(regular.key).toEqual({ mount: "gemma4", model: "e4b" });
    });

    it("refuses the default when no mirror is present", async () => {
      await expect(
        openGemmaModel(undefined, "cached", memoryDistributions({}), SIGNAL),
      ).rejects.toThrow(
        "Parameter model: the default is the default model of the first of gemma4, gemma4-qat, gemma4-qat-mobile whose mirror is present, and none is",
      );
    });

    it("refuses a model its mount does not have, naming the models it has", async () => {
      await expect(
        openGemmaModel(
          { mount: "gemma4-qat", model: "31b" },
          "cached",
          memoryDistributions({ "gemma4-qat": QAT }),
          SIGNAL,
        ),
      ).rejects.toThrow(
        "Parameter model: gemma4-qat has no model '31b'; its models are 12b, 26b, e2b",
      );
    });

    it("refuses a mobile QAT model its entry point does not take, naming the ones it does", async () => {
      await expect(
        openGemmaModel(
          { mount: "gemma4-qat-mobile", model: "12b" },
          "cached",
          memoryDistributions({ "gemma4-qat-mobile": MOBILE }),
          SIGNAL,
        ),
      ).rejects.toThrow(
        "Parameter model: gemma4QatMobile.Pipeline does not accept '12b'; the models of gemma4-qat-mobile it accepts are e2b, e4b",
      );
    });

    it("leaves a named mount whose mirror is missing to openDistribution's error, which names the mount", async () => {
      await expect(
        openGemmaModel(QAT_26B, "cached", memoryDistributions({ gemma4: REGULAR }), SIGNAL),
      ).rejects.toThrow(MountUnavailableError);
    });
  });

  describe("the request", () => {
    it("is the model at its default preset with the settings, the fixed work and the two arms", () => {
      expect(requestOf({}, { mount: "gemma4-qat", model: "12b" })).toEqual({
        mount: "gemma4-qat",
        model: "12b",
        quant: "i4-fast",
        rounds: 4,
        tokens: 96,
        minGain: 0.01,
        openPerVisit: true,
        probeCounts: true,
        prompt: SESSION_AB_PROMPT,
        sampler: { temperature: 0, seed: 0 },
        capacity: 1024,
        arms: [
          { name: "base", fusionRules: [] },
          { name: "rule", fusionRules: [GELU_MUL_RULE_NAME] },
        ],
      });
    });

    it("takes the named preset, f16 KV and the settings as given", () => {
      expect(
        requestOf({
          preset: "i4",
          f16Kv: "true",
          tokens: "8",
          rounds: "3",
          minGain: "0",
          openPerVisit: "false",
          probeCounts: "false",
        }),
      ).toMatchObject({
        quant: "i4",
        fullStateStorage: "f16",
        tokens: 8,
        rounds: 3,
        minGain: 0,
        openPerVisit: false,
        probeCounts: false,
      });
    });

    it("takes the expert cache on gemma4-qat/26b: the slots, the host tier and the depth", () => {
      expect(
        requestOf({ expertCacheSlots: "64", hostTier: "all", depth: "2" }).expertCache,
      ).toEqual({
        slots: 64,
        hostBytes: Number.MAX_SAFE_INTEGER,
        depth: 2,
      });
      expect(requestOf({ expertCacheSlots: "fitted", hostTier: "12" }).expertCache).toEqual({
        slots: FITTED_EXPERT_SLOTS,
        hostBytes: 12 * 2 ** 30,
      });
      expect(Object.hasOwn(requestOf({}), "expertCache")).toBe(false);
    });

    it("refuses each expert cache option on another model", () => {
      const key: ModelKey = { mount: "gemma4-qat", model: "12b" };
      for (const [name, value] of [
        ["expertCacheSlots", "64"],
        ["hostTier", "all"],
        ["depth", "2"],
      ] as const) {
        expect(() => requestOf({ [name]: value }, key), name).toThrow(
          `Parameter ${name}: the expert cache is only for gemma4-qat/26b (a mixture-of-experts model), not gemma4-qat/12b`,
        );
      }
    });

    it("refuses a host tier or a depth without slots", () => {
      expect(() => requestOf({ hostTier: "all" })).toThrow(
        "Parameter hostTier: needs expertCacheSlots (it applies to the expert cache)",
      );
      expect(() => requestOf({ depth: "2" })).toThrow(
        "Parameter depth: needs expertCacheSlots (it applies to the expert cache)",
      );
    });

    it("refuses slots or a host tier that does not parse, with the forms they take", () => {
      expect(() => requestOf({ expertCacheSlots: "15" })).toThrow(
        /^Parameter expertCacheSlots: Expert cache slots must be an integer from 16 to 128/,
      );
      expect(() => requestOf({ expertCacheSlots: "64", hostTier: "0" })).toThrow(
        /^Parameter hostTier: Host tier must be a positive number of GiB/,
      );
    });
  });

  describe("its run", () => {
    it("refuses a depth without slots before it acquires a GPU", async () => {
      const opened: string[] = [];
      await expect(
        gemmaSessionAb.run({
          parameters: parameters({ model: "gemma4-qat/26b", depth: "2" }),
          gpuOptions: {},
          signal: SIGNAL,
          progress: () => undefined,
          openDistribution: memoryDistributions({ "gemma4-qat": QAT }, opened),
          openInput: () => Promise.reject(new Error("the entry reads no input file")),
        }),
      ).rejects.toThrow("Parameter depth: needs expertCacheSlots");
      expect(opened).toEqual(["gemma4-qat:cached"]);
    });
  });

  describe("its result", () => {
    const DISTRIBUTION = {
      directory: "models/karume-gemma4-qat",
      manifestSha256: hex("f"),
      read: "directory",
    } as const;

    const PROBES: readonly SessionAbArmProbe[] = [
      { name: "base", counts: { fusions: { rmsNormAdd: 30 }, dispatches: 1294, ruleKeys: [] } },
      {
        name: "rule",
        counts: {
          fusions: { rmsNormAdd: 30, [GELU_MUL_RULE_NAME]: 60 },
          dispatches: 1234,
          ruleKeys: ["ext:candidate:gelu_mul:v1:gelu-other:f32:wg128"],
        },
      },
    ];

    const resultOf = (
      record: SessionAbRecord,
      given: Readonly<Record<string, string>> = { rounds: "3" },
      /** "none" = the entry ran no probe. */
      probes: readonly SessionAbArmProbe[] | "none" = PROBES,
    ) => {
      const request = requestOf(given);
      // With the cache, each arm's pipeline resolved it (two routed layers here).
      const facts =
        request.expertCache === undefined
          ? {}
          : { expertCache: { slots: [64, 64], depth: request.expertCache.depth } };
      const data = sessionAbGemmaData({
        request,
        distribution: DISTRIBUTION,
        ruleName: GELU_MUL_RULE_NAME,
        arms: request.arms.map(({ name }) => ({ name, facts })),
        ...(probes === "none" ? {} : { probes }),
        record,
        judgements: judgeSessionAb(record, { minGain: request.minGain }),
      });
      return { data, result: sessionAbGemmaResult(data) };
    };

    it("passes on token identity, and keeps the data body as built", () => {
      const { data, result } = resultOf(abRecord({ base: steady(100), rule: steady(90) }));
      expect(result.verdict).toBe("pass");
      expect(result.data).toBe(data);
      expect(data.pipeline).toBe("gemma.Pipeline");
    });

    it("sums up the judgement, the probe's counts and identity, one line each", () => {
      const { result } = resultOf(abRecord({ base: steady(100), rule: steady(90) }));
      expect(result.summary).toHaveLength(3);
      expect(result.summary[0]).toMatch(/^rule against base: Adopt · gain 10\.00% · /);
      expect(result.summary[1]).toBe(
        `Probe counts (first decode run): ${GELU_MUL_RULE_NAME} hits base 0, rule 60; dispatches base 1294, rule 1234 (rule − base −60)`,
      );
      expect(result.summary[2]).toBe(
        "Identity against the base's first run: base identical (20 runs); rule identical (20 runs)",
      );
    });

    it("names the first verify run as the counted one under self-speculation", () => {
      const { result } = resultOf(abRecord({ base: steady(100), rule: steady(90) }), {
        rounds: "3",
        expertCacheSlots: "64",
        depth: "2",
      });
      expect(result.summary[1]).toMatch(/^Probe counts \(first verify run\): /);
    });

    it("has no counts line without the probe", () => {
      const { result } = resultOf(
        abRecord({ base: steady(100), rule: steady(90) }),
        { rounds: "3", probeCounts: "false" },
        "none",
      );
      expect(result.summary).toHaveLength(2);
      expect(result.summary[1]).toMatch(/^Identity against the base's first run: /);
    });

    it("fails when the rule arm's tokens differ", () => {
      const { result } = resultOf(
        abRecord({ base: steady(100), rule: steady(90), ruleIdentity: DIFFERING }),
      );
      expect(result.verdict).toBe("fail");
    });
  });
});
