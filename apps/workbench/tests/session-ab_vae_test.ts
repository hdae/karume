import { VAE_RMS_SILU_RULE_NAME } from "@karume/models/wan";
import type { SessionAbArmRecord, SessionAbRecord } from "@karume/runtime/tune";
import { describe, expect, it } from "vite-plus/test";

import { sessionAbVaeData } from "../src/session-ab/data.ts";
import type { GraphInput } from "../src/session-ab/feeds.ts";
import { presetOf } from "../src/session-ab/shared.ts";
import {
  sessionAbVaeRequest,
  VAE_AB_TARGETS,
  vaeAbModel,
  vaeArmOptions,
  vaeFeeds,
  vaeSessionOptions,
} from "../src/session-ab/vae.ts";

import { hex, manifestOf, WAN_MANIFEST } from "./helpers/session-ab-manifests.ts";
import { abRecord, SETTINGS, steady, wordsOf } from "./helpers/session-ab-records.ts";

const RULE = VAE_RMS_SILU_RULE_NAME;

/** A Wan chunk graph's inputs at small widths (the real latent is [48, 1, 16, 16]). */
const CHUNK_INPUTS: readonly GraphInput[] = [
  { name: "latent", dtype: "f32", shape: [3, 1, 2, 2] },
  { name: "cache_00", dtype: "f32", shape: [3, 2, 2, 2] },
  { name: "cache_01", dtype: "f32", shape: [5, 2, 2, 2] },
];

const WAN = manifestOf(WAN_MANIFEST);

/** The Wan2.2 model at `preset` for the next chunk graph. */
const wanModel = (preset: string) =>
  vaeAbModel("wan22-vae-next", "ti2v-5b", presetOf(WAN, "ti2v-5b", preset));

const CONV8 = wanModel("i8-a8-attn8-online-conv8");

/** A two-arm runner record whose arms carry the counts a `sessionArm` reports. */
const countedRecord = (): SessionAbRecord => {
  const record = abRecord({ base: steady(584), rule: steady(564) });
  const [base, rule] = record.arms;
  const arm = (
    template: SessionAbArmRecord,
    name: string,
    fusions: Readonly<Record<string, number>>,
    dispatches: number,
  ): SessionAbArmRecord => ({ ...template, name, fusions, dispatches });
  return {
    ...record,
    arms: [arm(base, "base", { silu: 29 }, 457), arm(rule, "rule", { silu: 0, [RULE]: 30 }, 248)],
  };
};

const dataOf = (
  record: SessionAbRecord,
  request = sessionAbVaeRequest("wan22-vae-next", CONV8, SETTINGS),
) =>
  sessionAbVaeData({
    request,
    distribution: { directory: "models/karume-wan2.2", manifestSha256: hex("f"), read: "direct" },
    container: { component: "vae_decoder_next", part0Sha256: hex("c") },
    record,
    judgements: [],
  });

describe("the session A/B's VAE targets", () => {
  it("are the two VAE targets and nothing else", () => {
    expect(Object.keys(VAE_AB_TARGETS)).toEqual(["wan22-vae-next", "anima-vae-decode"]);
  });

  describe("the feeds", () => {
    const feeds = vaeFeeds(CHUNK_INPUTS, VAE_AB_TARGETS["wan22-vae-next"].seed);

    it("feed every input, the caches included, at its declared shape", () => {
      expect(
        Object.fromEntries(Object.entries(feeds).map(([name, { shape }]) => [name, shape])),
      ).toEqual({ latent: [3, 1, 2, 2], cache_00: [3, 2, 2, 2], cache_01: [5, 2, 2, 2] });
    });

    it("draw every value from [−1, 1), and the same words again from the same seed", () => {
      for (const [name, tensor] of Object.entries(feeds)) {
        expect(tensor.dtype, name).toBe("f32");
        expect(
          [...tensor.data].every((value) => value >= -1 && value < 1),
          name,
        ).toBe(true);
      }
      const again = vaeFeeds(CHUNK_INPUTS, VAE_AB_TARGETS["wan22-vae-next"].seed);
      const other = vaeFeeds(CHUNK_INPUTS, VAE_AB_TARGETS["wan22-vae-next"].seed + 1);
      for (const name of Object.keys(feeds)) {
        expect(wordsOf(again[name].data), name).toEqual(wordsOf(feeds[name].data));
        expect(wordsOf(other[name].data), name).not.toEqual(wordsOf(feeds[name].data));
      }
      // One generator across the inputs: the second input does not restart the sequence.
      expect(wordsOf(feeds.cache_00.data).slice(0, 12)).not.toEqual(
        wordsOf(feeds.latent.data).slice(0, 12),
      );
    });

    it("refuse an input they were not made for: a symbolic dimension or another dtype", () => {
      expect(() =>
        vaeFeeds([{ name: "latents", dtype: "f32", shape: [1, 16, "h", "w"] }], 1),
      ).toThrow("input 'latents' has symbol 'h'");
      expect(() => vaeFeeds([{ name: "mask", dtype: "bool", shape: [4] }], 1)).toThrow(
        "no feed for input 'mask' (bool)",
      );
    });
  });

  describe("the VAE Session's options", () => {
    it("are the conv knob alone on Wan2.2, as wanVaeSessionOptions gives the VAE", () => {
      expect(vaeSessionOptions("wan22-vae-next", CONV8.session)).toEqual({ convCompute: "a8" });
      expect(
        vaeSessionOptions("wan22-vae-next", wanModel("f16+dit8-a8-attn8-online").session),
      ).toEqual({});
      expect(vaeSessionOptions("wan22-vae-next", {})).toEqual({});
    });

    it("are none on anima, whose VAE stage takes none of the quant's session", () => {
      expect(vaeSessionOptions("anima-vae-decode", { convCompute: "a8" })).toEqual({});
      expect(vaeSessionOptions("anima-vae-decode", { linearCompute: "a8" })).toEqual({});
    });
  });

  describe("the model and the request", () => {
    it("take the preset's declared session and the component's storage, and refuse a preset without it", () => {
      expect(CONV8).toEqual({
        model: "ti2v-5b",
        quant: "i8-a8-attn8-online-conv8",
        session: {
          linearCompute: "a8",
          attentionCompute: "a8",
          attentionScoreStorage: "none",
          convCompute: "a8",
        },
        storage: "i8",
      });
      expect(wanModel("f16+dit8-a8-attn8-online").storage).toBe("f16");
      // The anima target's component is not in a Wan2.2 manifest.
      expect(() =>
        vaeAbModel("anima-vae-decode", "ti2v-5b", presetOf(WAN, "ti2v-5b", "i8")),
      ).toThrow("Preset 'i8' of model 'ti2v-5b' has no 'vae_decoder'");
    });

    it("names the two arms, the options every arm gets and the seed", () => {
      const request = sessionAbVaeRequest("wan22-vae-next", CONV8, SETTINGS);
      expect(request).toEqual({
        target: "wan22-vae-next",
        mount: "wan22",
        component: "vae_decoder_next",
        ...CONV8,
        rounds: 3,
        minGain: 0.01,
        openPerVisit: true,
        seed: 0x5eed,
        sessionOptions: { convCompute: "a8" },
        arms: [
          { name: "base", fusionRules: [] },
          { name: "rule", fusionRules: [RULE] },
        ],
      });
      const anima = sessionAbVaeRequest(
        "anima-vae-decode",
        { model: "anima-turbo-v1.1", quant: "f16", session: {}, storage: "f16" },
        SETTINGS,
      );
      expect([anima.mount, anima.component, anima.sessionOptions]).toEqual([
        "anima",
        "vae_decoder",
        {},
      ]);
    });

    it("gives each arm the request's options and the rule it names, the base none", () => {
      const arms = vaeArmOptions(sessionAbVaeRequest("wan22-vae-next", CONV8, SETTINGS));
      expect(arms.map(({ name }) => name)).toEqual(["base", "rule"]);
      expect(arms[0].options).toEqual({ convCompute: "a8" });
      const { options } = arms[1];
      expect(options.convCompute).toBe("a8");
      expect(options.fusionRules?.map(({ name }) => name)).toEqual([RULE]);
    });
  });

  describe("the record", () => {
    it("holds the target, the container and the rule around the runtime's record", () => {
      const record = countedRecord();
      const data = dataOf(record);
      expect(data.target).toBe("wan22-vae-next");
      expect(data.container).toEqual({ component: "vae_decoder_next", part0Sha256: hex("c") });
      expect(data.rules).toEqual([
        { name: RULE, package: "@karume/models", keyPrefix: "ext:models:" },
      ]);
      expect(data.record).toBe(record);
    });
  });
});
