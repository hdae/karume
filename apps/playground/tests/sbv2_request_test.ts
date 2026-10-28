import type { Utterance } from "@karume/models/sbv2";
import { describe, expect, it } from "vite-plus/test";

import type { Sbv2ModelConfig } from "../src/models/sbv2/config.ts";
import {
  buildRequest,
  defaultKnobs,
  formForModel,
  readingSegments,
  readingText,
  type Sbv2Form,
} from "../src/models/sbv2/request.ts";

const F1: Sbv2ModelConfig = {
  speakers: ["jvnv-F1-jp"],
  styles: ["Neutral", "Angry", "Happy"],
  maxTokens: 512,
  maxFrames: 4096,
  defaults: {
    speaker: "jvnv-F1-jp",
    style: "Neutral",
    styleWeight: 1,
    sdpRatio: 0.2,
    noiseScale: 0.6,
    noiseScaleW: 0.8,
    lengthScale: 1,
  },
};

const M1: Sbv2ModelConfig = {
  ...F1,
  speakers: ["jvnv-M1-jp"],
  styles: ["Neutral", "Sad"],
  defaults: { ...F1.defaults, speaker: "jvnv-M1-jp" },
};

const form = (patch: Partial<Sbv2Form> = {}): Sbv2Form => ({
  ...defaultKnobs(F1.defaults),
  text: "text",
  ...patch,
});

describe("buildRequest", () => {
  it("passes every option explicitly, the defaults included, with the seed as a number", () => {
    expect(buildRequest(form({ seed: " 42 " }))).toEqual({
      text: "text",
      options: {
        speaker: "jvnv-F1-jp",
        style: "Neutral",
        styleWeight: 1,
        sdpRatio: 0.2,
        noiseScale: 0.6,
        noiseScaleW: 0.8,
        lengthScale: 1,
        seed: 42,
      },
    });
  });

  it("refuses an empty or blank text", () => {
    expect(() => buildRequest(form({ text: " \n" }))).toThrow("The text is empty");
  });

  it("refuses a seed that is not a number", () => {
    expect(() => buildRequest(form({ seed: "abc" }))).toThrow("Seed must be a number (got 'abc')");
    expect(() => buildRequest(form({ seed: "" }))).toThrow("Seed must be a number");
  });

  it("leaves the range of the numbers to the pipeline (a negative seed or length scale passes)", () => {
    const { options } = buildRequest(form({ seed: "-1", lengthScale: 0 }));
    expect(options.seed).toBe(-1);
    expect(options.lengthScale).toBe(0);
  });
});

describe("formForModel", () => {
  it("takes the new model's speaker and keeps a style the model also has", () => {
    const next = formForModel(form({ style: "Neutral", styleWeight: 1.5 }), M1);
    expect(next.speaker).toBe("jvnv-M1-jp");
    expect(next.style).toBe("Neutral");
    expect(next.styleWeight).toBe(1.5);
  });

  it("falls back to the model's default style when it lacks the chosen one", () => {
    expect(formForModel(form({ style: "Happy" }), M1).style).toBe("Neutral");
  });
});

/** konnichiwa, ame!? — kana spelled in romaji (any string is a kana to the grouping). */
const utterance: Utterance = {
  leadingPunctuations: ["…"],
  moras: [
    { kana: "ko", consonant: "k", vowel: "o", tone: 0 },
    { kana: "n", vowel: "N", tone: 1 },
    { kana: "ni", consonant: "n", vowel: "i", tone: 1 },
    { kana: "chi", consonant: "ch", vowel: "i", tone: 1 },
    { kana: "wa", consonant: "w", vowel: "a", tone: 1, punctuations: [","] },
    { kana: "a", vowel: "a", tone: 1 },
    { kana: "me", consonant: "m", vowel: "e", tone: 0, punctuations: ["!", "?"] },
  ],
  words: [],
};

describe("readingSegments", () => {
  it("groups consecutive moras of one tone and puts each punctuation mark where it is read", () => {
    expect(readingSegments(utterance)).toEqual([
      { kind: "punctuation", text: "…" },
      { kind: "moras", kana: "ko", high: false },
      { kind: "moras", kana: "nnichiwa", high: true },
      { kind: "punctuation", text: "," },
      { kind: "moras", kana: "a", high: true },
      { kind: "moras", kana: "me", high: false },
      { kind: "punctuation", text: "!" },
      { kind: "punctuation", text: "?" },
    ]);
  });
});

describe("readingText", () => {
  it("brackets the high moras", () => {
    expect(readingText(utterance)).toBe("…ko[nnichiwa],[a]me!?");
  });
});
