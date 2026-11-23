/**
 * The plan of the quality set and the command line: the near-duplicates across the two image
 * generators, the sizes both generators accept, distinct files, what `--limit` keeps, and what the
 * command line refuses. No GPU, no mirror.
 */

import { describe, expect, it } from "vite-plus/test";
import { parseCommandLine, planSet, SENTENCES, SPEAKERS, SUBJECTS, UsageError } from "../set.ts";

describe("planSet", () => {
  it("draws the same subjects with the same seeds and sizes in both image generators", () => {
    const { images } = planSet();
    const anima = images.filter((image) => image.generator === "anima");
    const moss = images.filter((image) => image.generator === "moss");
    expect(anima).toHaveLength(16);
    expect(moss).toHaveLength(16);
    const facts = (image: (typeof images)[number]) => ({
      subject: image.subject.name,
      seed: image.item.seed,
      shape: image.subject.shape,
      caption: image.item.caption,
    });
    expect(moss.map(facts)).toEqual(anima.map(facts));
    expect(new Set(SUBJECTS.map((subject) => subject.name)).size).toBe(16);
  });

  it("sizes every image within what Anima and MOSS-Image accept, in more than one aspect", () => {
    for (const { shape } of SUBJECTS) {
      for (const side of [shape.width, shape.height]) {
        expect(side % 16).toBe(0);
        expect(side).toBeGreaterThanOrEqual(512);
        expect(side).toBeLessThanOrEqual(2048);
      }
    }
    const aspects = new Set(SUBJECTS.map(({ shape }) => Math.sign(shape.width - shape.height)));
    expect(aspects).toEqual(new Set([-1, 0, 1]));
  });

  it("speaks every sentence by every speaker into its own file", () => {
    const { images, audio } = planSet();
    expect(audio).toHaveLength(SPEAKERS.length * SENTENCES.length);
    const files = [...images, ...audio].map(({ item }) => item.file);
    expect(new Set(files).size).toBe(64);
  });

  it("keeps the first n items of each generator and each speaker under a limit", () => {
    const { images, audio } = planSet(1);
    expect(images.map(({ item }) => item.file)).toEqual([
      "images/anima-fox.png",
      "images/moss-fox.png",
    ]);
    expect(audio.map(({ item }) => item.file)).toEqual([
      "audio/F1-01.wav",
      "audio/F2-01.wav",
      "audio/M1-01.wav",
      "audio/M2-01.wav",
    ]);
  });
});

describe("parseCommandLine", () => {
  it("reads an output directory and a limit in either spelling", () => {
    expect(parseCommandLine([])).toEqual({ output: undefined, limit: undefined });
    expect(parseCommandLine(["out", "--limit", "2"])).toEqual({ output: "out", limit: 2 });
    expect(parseCommandLine(["--limit=3"])).toEqual({ output: undefined, limit: 3 });
  });

  it("refuses a limit that is not a positive integer, a second directory and unknown options", () => {
    for (const args of [["--limit"], ["--limit", "0"], ["--limit=1.5"], ["a", "b"], ["--all"]]) {
      expect(() => parseCommandLine(args)).toThrow(UsageError);
    }
  });
});
