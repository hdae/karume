/**
 * The pure part of the preset comparison: the manifest it reads, the cosines at a shorter width,
 * the cross-modal top-1 with a shared transcript, the time summary and the command line. No GPU,
 * no mirror.
 */

import { describe, expect, it } from "vite-plus/test";
import {
  agreement,
  CompareInputError,
  cosine,
  distinctTexts,
  parseCompareCommandLine,
  parseQualitySetManifest,
  retrieval,
  summarizeCosines,
  summarizeTimes,
} from "../compare.ts";
import { MANIFEST_FORMAT } from "../set.ts";

const vector = (...values: number[]): Float32Array<ArrayBuffer> => Float32Array.from(values);

const image = (file: string, caption: string) => ({
  file,
  caption,
  generator: "karume-anima/anima-turbo-v1.1",
  prompt: "p",
  seed: 1000,
});
const clip = (file: string, transcript: string, sampleRate = 16000) => ({
  file,
  transcript,
  generator: "sbv2-jvnv/F1",
  sampleRate,
});
const manifestText = (overrides: Record<string, unknown> = {}): string =>
  JSON.stringify({
    format: MANIFEST_FORMAT,
    images: [image("images/a.png", "A."), image("images/b.png", "B.")],
    audio: [clip("audio/F1-01.wav", "one"), clip("audio/F2-01.wav", "one")],
    ...overrides,
  });

describe("parseQualitySetManifest", () => {
  it("reads every image with its caption and every clip with its transcript and rate", () => {
    const manifest = parseQualitySetManifest(manifestText(), "m.json");
    expect(manifest.images.map(({ file, caption }) => [file, caption])).toEqual([
      ["images/a.png", "A."],
      ["images/b.png", "B."],
    ]);
    expect(manifest.audio.map(({ file, sampleRate }) => [file, sampleRate])).toEqual([
      ["audio/F1-01.wav", 16000],
      ["audio/F2-01.wav", 16000],
    ]);
  });

  it("refuses another format, fewer than two items, a missing caption or a rate that is no integer", () => {
    for (const overrides of [
      { format: "karume-embeddinggemma2-quality-set/2" },
      { images: [image("images/a.png", "A.")] },
      { images: [image("images/a.png", "A."), { ...image("images/b.png", "B."), caption: "" }] },
      { audio: [clip("audio/F1-01.wav", "one"), clip("audio/F2-01.wav", "one", 16000.5)] },
    ]) {
      expect(() => parseQualitySetManifest(manifestText(overrides), "m.json")).toThrow(
        CompareInputError,
      );
    }
  });
});

describe("distinctTexts", () => {
  it("keeps the first appearance of each text and maps every item to its index", () => {
    expect(distinctTexts(["one", "two", "one", "three", "two"])).toEqual({
      texts: ["one", "two", "three"],
      itemText: [0, 1, 0, 2, 1],
    });
  });
});

describe("cosine", () => {
  it("reads only the leading values at a shorter width, whatever their norm", () => {
    const left = vector(1, 0, 5, -3);
    const right = vector(2, 0, -7, 4);
    expect(cosine(left, right, 2)).toBeCloseTo(1, 15);
    expect(cosine(left, right, 4)).toBeLessThan(0);
    expect(cosine(vector(3, 4), vector(6, 8), 2)).toBeCloseTo(1, 15);
  });

  it("refuses a width past either vector", () => {
    expect(() => cosine(vector(1, 0), vector(1, 0, 0), 3)).toThrow(RangeError);
  });
});

describe("summarizeCosines", () => {
  it("gives the lowest value, the mean and the lowest's name", () => {
    expect(
      summarizeCosines([
        { name: "a", value: 0.9999 },
        { name: "b", value: 0.9995 },
        { name: "c", value: 0.9998 },
      ]),
    ).toEqual({ min: 0.9995, mean: (0.9999 + 0.9995 + 0.9998) / 3, worst: "b" });
  });
});

describe("retrieval", () => {
  // Three items, two distinct texts; items 0 and 2 share text 0.
  const items = [vector(1, 0), vector(0, 1), vector(0.9, 0.1)];
  const texts = [vector(1, 0.05), vector(0.05, 1)];
  const itemText = [0, 1, 0];

  it("hits an item whose own text ranks first, and a text whose first item is any item of it", () => {
    const result = retrieval(items, texts, itemText, 2);
    expect(result.itemToText).toEqual({ hits: 3, choices: [0, 1, 0] });
    // Text 0 picks item 0, text 1 picks item 1: both are items of the querying text.
    expect(result.textToItem).toEqual({ hits: 2, choices: [0, 1] });
  });

  it("misses when another text ranks first, and counts the choices two presets share", () => {
    const swapped = retrieval(items, [texts[1], texts[0]], itemText, 2);
    expect(swapped.itemToText.hits).toBe(0);
    expect(swapped.textToItem.hits).toBe(0);
    const reference = retrieval(items, texts, itemText, 2);
    expect(agreement(reference.itemToText.choices, swapped.itemToText.choices)).toBe(0);
    expect(agreement(reference.textToItem.choices, reference.textToItem.choices)).toBe(2);
  });

  it("ranks at the width it is given", () => {
    // At width 1 the second value, which alone separates item 1 from text 0, is not read.
    const narrow = retrieval([vector(1, 0), vector(1, 9)], [vector(1, 0), vector(1, 9)], [0, 1], 1);
    expect(narrow.itemToText.choices).toEqual([0, 0]);
  });
});

describe("summarizeTimes", () => {
  it("gives the mean, the median (the middle pair's mean for an even count), min and max", () => {
    expect(summarizeTimes([4, 1, 3])).toEqual({ mean: 8 / 3, median: 3, min: 1, max: 4 });
    expect(summarizeTimes([4, 1, 3, 2])).toEqual({ mean: 2.5, median: 2.5, min: 1, max: 4 });
  });
});

describe("parseCompareCommandLine", () => {
  it("reads one optional record directory and refuses options and a second directory", () => {
    expect(parseCompareCommandLine([])).toBeUndefined();
    expect(parseCompareCommandLine(["out"])).toBe("out");
    expect(() => parseCompareCommandLine(["--limit", "1"])).toThrow(CompareInputError);
    expect(() => parseCompareCommandLine(["a", "b"])).toThrow(CompareInputError);
  });
});
