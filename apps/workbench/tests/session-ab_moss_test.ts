import { EW_CHAIN_RULE_NAME } from "@karume/models/moss";
import type { SessionAbRecord } from "@karume/runtime/tune";
import { describe, expect, it } from "vite-plus/test";

import { sessionAbMossData } from "../src/session-ab/data.ts";
import type { GraphInput } from "../src/session-ab/feeds.ts";
import {
  MOSS_DIT_WORK,
  mossAbModel,
  mossDitFeeds,
  sessionAbMossRequest,
} from "../src/session-ab/moss.ts";
import { presetOf } from "../src/session-ab/shared.ts";
import { recordedCounts } from "../src/session-ab/summary.ts";

import { hex, manifestOf, MOSS_MANIFEST } from "./helpers/session-ab-manifests.ts";
import { abRecord, SETTINGS, steady, wordsOf } from "./helpers/session-ab-records.ts";

/** The DiT's inputs as its graph declares them, at small widths (the real ones are 128 / 3072 / 256). */
const DIT_INPUTS: readonly GraphInput[] = [
  { name: "latent_tokens", dtype: "f32", shape: [1, "S", 4] },
  { name: "text_context", dtype: "f32", shape: [1, "S", 6] },
  { name: "is_image", dtype: "bool", shape: [1, "S", 1] },
  { name: "rope_cos", dtype: "f32", shape: [1, 1, "S", 2] },
  { name: "rope_sin", dtype: "f32", shape: [1, 1, "S", 2] },
  { name: "timesteps_proj", dtype: "f32", shape: [1, 8] },
];

const IMAGE = 6;
const TEXT = 3;

const manifest = manifestOf(MOSS_MANIFEST);

const MODEL = mossAbModel("sft", presetOf(manifest, "sft", "f32+dit8-a8-attn8-online"));

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

const dataOf = (
  record: SessionAbRecord,
  request = sessionAbMossRequest(MODEL, SETTINGS, EW_CHAIN_RULE_NAME),
) =>
  sessionAbMossData({
    request,
    distribution: {
      directory: "models/karume-moss-image",
      manifestSha256: hex("f"),
      read: "direct",
    },
    container: { component: "dit", part0Sha256: hex("c") },
    record,
    judgements: [],
  });

describe("the session A/B's MOSS-Image DiT target", () => {
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
      const request = sessionAbMossRequest(MODEL, SETTINGS, EW_CHAIN_RULE_NAME);
      expect(request).toEqual({
        target: "moss-dit-forward",
        mount: "moss-image",
        component: "dit",
        model: "sft",
        quant: "f32+dit8-a8-attn8-online",
        session: { linearCompute: "a8", attentionCompute: "a8", attentionScoreStorage: "none" },
        rounds: 3,
        minGain: 0.01,
        openPerVisit: true,
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

    it("takes the knobs from the manifest's preset, none for the reference preset", () => {
      expect(mossAbModel("sft", presetOf(manifest, "sft", "f32+dit8"))).toEqual({
        model: "sft",
        quant: "f32+dit8",
        session: {},
      });
    });
  });

  describe("the record", () => {
    it("holds the target, the container and the rule's key prefix around the runtime's record", () => {
      const record = countedRecord();
      const data = dataOf(record);
      expect(data.target).toBe("moss-dit-forward");
      expect(data.container).toEqual({ component: "dit", part0Sha256: hex("c") });
      expect(data.rule).toEqual({
        name: EW_CHAIN_RULE_NAME,
        package: "@karume/models",
        keyPrefix: "ext:models:",
      });
      expect(data.record).toBe(record);
    });

    it("reads each arm's counts from the runner's record, and none for an arm without them", () => {
      const [base, rule] = countedRecord().arms;
      expect(recordedCounts(base)).toEqual({ arm: "base", fusions: { silu: 1 }, dispatches: 2466 });
      expect(recordedCounts(rule)?.fusions[EW_CHAIN_RULE_NAME]).toBe(292);
      const [uncounted] = abRecord({ base: steady(10), rule: steady(9) }).arms;
      expect(recordedCounts(uncounted)).toBeUndefined();
    });
  });
});
