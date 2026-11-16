import { EW_CHAIN_RULE_NAME } from "@karume/models/moss";
import { VAE_RMS_SILU_RULE_NAME } from "@karume/models/wan";
import { judgeSessionAb, type SessionAbRecord } from "@karume/runtime/tune";
import { describe, expect, it } from "vite-plus/test";

import sessionAb from "../src/entries/local/session-ab.ts";
import { ParameterError, parseParameters } from "../src/entry.ts";
import {
  COMPONENT_TARGETS,
  componentArmOptions,
  componentFeeds,
  componentRequest,
  targetMount,
} from "../src/session-ab/component-target.ts";
import { sessionAbMossData, sessionAbVaeData } from "../src/session-ab/data.ts";
import type { GraphInput } from "../src/session-ab/feeds.ts";
import { MOSS_DIT_WORK } from "../src/session-ab/moss.ts";
import { sessionAbComponentResult } from "../src/session-ab/summary.ts";
import { vaeArmOptions, vaeFeeds } from "../src/session-ab/vae.ts";

import {
  hex,
  manifestOf,
  manifestText,
  memoryDistributions,
  MOSS_MANIFEST,
  WAN_MANIFEST,
} from "./helpers/session-ab-manifests.ts";
import { abRecord, DIFFERING, SETTINGS, steady } from "./helpers/session-ab-records.ts";

const MOSS = manifestOf(MOSS_MANIFEST);
const WAN = manifestOf(WAN_MANIFEST);

/** The anima mirror's shape around the tile graph. */
const ANIMA = manifestText({
  pipeline: "anima/1",
  models: ["anima-turbo-v1.1"],
  component: "vae_decoder",
  presets: { f16: { convCompute: "a8" } },
  defaultPreset: "f16",
});

const DISTRIBUTION = {
  directory: "models/karume-wan2.2",
  manifestSha256: hex("f"),
  read: "cached",
} as const;

/** A runner record whose arms carry the counts each arm reported. */
const counted = (
  record: SessionAbRecord,
  base: readonly [Readonly<Record<string, number>>, number],
  rule: readonly [Readonly<Record<string, number>>, number],
): SessionAbRecord => {
  const [baseArm, ruleArm] = record.arms;
  return {
    ...record,
    arms: [
      { ...baseArm, fusions: base[0], dispatches: base[1] },
      { ...ruleArm, fusions: rule[0], dispatches: rule[1] },
    ],
  };
};

const vaeData = (record: SessionAbRecord) => {
  const request = componentRequest("wan22-vae-next", WAN, "", SETTINGS);
  if (request.target === "moss-dit-forward") throw new Error("not a VAE request");
  return sessionAbVaeData({
    request,
    distribution: DISTRIBUTION,
    container: { component: "vae_decoder_next", part0Sha256: hex("c") },
    record,
    judgements: judgeSessionAb(record, { minGain: request.minGain }),
  });
};

describe("the session-ab entry", () => {
  describe("its parameters", () => {
    it("default to the anima VAE, the manifest's preset, 4 rounds, a 1% minimum gain and the cached route", () => {
      expect(parseParameters(sessionAb.parameters, {})).toEqual({
        target: "anima-vae-decode",
        preset: "",
        rounds: 4,
        minGain: 0.01,
        source: "cached",
      });
    });
  });

  describe("each target", () => {
    it("reads its own mount", () => {
      expect(COMPONENT_TARGETS.map(targetMount)).toEqual(["moss-image", "wan22", "anima"]);
    });

    it("asks for the manifest's default model at the named preset, or the model's default preset when none is named", () => {
      for (const [target, manifest, preset, expected] of [
        [
          "moss-dit-forward",
          MOSS,
          "",
          {
            mount: "moss-image",
            component: "dit",
            model: "sft",
            quant: "f32+dit8-a8-attn8-online",
          },
        ],
        ["moss-dit-forward", MOSS, "f32+dit8", { quant: "f32+dit8", session: {} }],
        [
          "wan22-vae-next",
          WAN,
          "i8",
          { mount: "wan22", component: "vae_decoder_next", quant: "i8", sessionOptions: {} },
        ],
        [
          "anima-vae-decode",
          manifestOf(ANIMA),
          "",
          {
            mount: "anima",
            component: "vae_decoder",
            model: "anima-turbo-v1.1",
            quant: "f16",
            sessionOptions: {},
          },
        ],
      ] as const) {
        expect(
          componentRequest(target, manifest, preset, SETTINGS),
          `${target}@${preset}`,
        ).toMatchObject({ target, ...expected });
      }
    });

    it("gives the MOSS arms the preset's knobs, the rule arm models/ew_chain too, and the VAE arms what vaeArmOptions gives them", () => {
      const moss = componentArmOptions(componentRequest("moss-dit-forward", MOSS, "", SETTINGS));
      expect(moss.map(({ name }) => name)).toEqual(["base", "rule"]);
      expect(moss[0].options).toEqual({
        linearCompute: "a8",
        attentionCompute: "a8",
        attentionScoreStorage: "none",
      });
      expect(moss[1].options.attentionCompute).toBe("a8");
      expect(moss[1].options.fusionRules?.map(({ name }) => name)).toEqual([EW_CHAIN_RULE_NAME]);
      const wan = componentRequest("wan22-vae-next", WAN, "", SETTINGS);
      if (wan.target === "moss-dit-forward") throw new Error("not a VAE request");
      expect(componentArmOptions(wan)).toEqual(vaeArmOptions(wan));
    });

    it("feeds its graph with its own seeded feeds", () => {
      const moss = componentRequest("moss-dit-forward", MOSS, "", SETTINGS);
      const ditFeeds = componentFeeds(moss, [
        { name: "is_image", dtype: "bool", shape: [1, "S", 1] },
      ]);
      expect(ditFeeds.is_image.shape).toEqual([
        1,
        MOSS_DIT_WORK.imageTokens + MOSS_DIT_WORK.textTokens,
        1,
      ]);
      const wan = componentRequest("wan22-vae-next", WAN, "", SETTINGS);
      const inputs: readonly GraphInput[] = [{ name: "latent", dtype: "f32", shape: [3, 1, 2, 2] }];
      expect(componentFeeds(wan, inputs)).toEqual(vaeFeeds(inputs, wan.seed));
    });
  });

  describe("its run", () => {
    it("opens the target's mount the way it is asked and refuses an unknown preset before it acquires a GPU", async () => {
      const opened: string[] = [];
      await expect(
        sessionAb.run({
          parameters: parseParameters(sessionAb.parameters, {
            target: "wan22-vae-next",
            preset: "f16",
            source: "direct",
          }),
          gpuOptions: {},
          signal: new AbortController().signal,
          progress: () => undefined,
          openDistribution: memoryDistributions({ wan22: WAN_MANIFEST }, opened),
        }),
      ).rejects.toThrow(ParameterError);
      expect(opened).toEqual(["wan22:direct"]);
    });
  });

  describe("its result", () => {
    const record = counted(
      abRecord({ base: steady(100), rule: steady(90) }),
      [{ silu: 29 }, 457],
      [{ silu: 0, [VAE_RMS_SILU_RULE_NAME]: 30 }, 248],
    );

    it("passes on identity, and keeps the data body as built", () => {
      const data = vaeData(record);
      const result = sessionAbComponentResult(data);
      expect(result.verdict).toBe("pass");
      expect(result.data).toBe(data);
    });

    it("sums up the judgement, the counts per forward and identity, one line each", () => {
      const [judgement, counts, identity] = sessionAbComponentResult(vaeData(record)).summary;
      expect(judgement).toMatch(
        /^rule against base: Adopt · gain 10\.00% · interval of the median difference \(\d+\.\d% coverage\) −10\.000 ms to −10\.000 ms$/,
      );
      expect(counts).toBe(
        `Counts per forward: ${VAE_RMS_SILU_RULE_NAME} hits base 0, rule 30; dispatches base 457, rule 248 (rule − base −209)`,
      );
      expect(identity).toBe(
        "Identity against the base's first run: base identical (20 runs); rule identical (20 runs)",
      );
    });

    it("fails when an arm's outputs differ from the base's, whatever the times", () => {
      const differing = counted(
        abRecord({ base: steady(100), rule: steady(90), ruleIdentity: DIFFERING }),
        [{ silu: 29 }, 457],
        [{ silu: 0, [VAE_RMS_SILU_RULE_NAME]: 30 }, 248],
      );
      const result = sessionAbComponentResult(vaeData(differing));
      expect(result.verdict).toBe("fail");
      expect(result.summary.at(-1)).toBe(
        "Identity against the base's first run: base identical (20 runs); rule 3 of 96 words differ (round 2, visit 2, run 2; first at tokens[41] (word): expected 100, got 200)",
      );
    });

    it("passes when the rule is slower but identical: the judgement is a measurement, not the verdict", () => {
      const slower = counted(
        abRecord({ base: steady(100), rule: steady(105) }),
        [{ silu: 29 }, 457],
        [{ silu: 0, [VAE_RMS_SILU_RULE_NAME]: 30 }, 248],
      );
      const result = sessionAbComponentResult(vaeData(slower));
      expect(result.verdict).toBe("pass");
      expect(result.summary[0]).toMatch(/^rule against base: Reject · gain -5\.00% · /);
    });

    it("counts the MOSS-Image rule's hits from the runner's record", () => {
      const mossRecord = counted(
        abRecord({ base: steady(1792), rule: steady(1620) }),
        [{ silu: 1 }, 2466],
        [{ silu: 1, [EW_CHAIN_RULE_NAME]: 292 }, 1880],
      );
      const request = componentRequest("moss-dit-forward", MOSS, "", SETTINGS);
      if (request.target !== "moss-dit-forward") throw new Error("not a MOSS request");
      const result = sessionAbComponentResult(
        sessionAbMossData({
          request,
          distribution: { ...DISTRIBUTION, directory: "models/karume-moss-image" },
          container: { component: "dit", part0Sha256: hex("c") },
          record: mossRecord,
          judgements: judgeSessionAb(mossRecord, { minGain: 0.01 }),
        }),
      );
      expect(result.summary[1]).toBe(
        `Counts per forward: ${EW_CHAIN_RULE_NAME} hits base 0, rule 292; dispatches base 2466, rule 1880 (rule − base −586)`,
      );
    });
  });
});
