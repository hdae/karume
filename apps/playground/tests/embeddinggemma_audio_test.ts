import { describe, expect, it } from "vite-plus/test";

import { audioSummary, mixToMono } from "../src/models/embeddinggemma/audio.ts";

describe("mixToMono", () => {
  it("keeps one channel as it is and averages several in f64, rounded to f32 once", () => {
    const mono = Float32Array.from([0.1, -0.2, 0.3]);
    expect([...mixToMono([mono])]).toEqual([...mono]);
    const left = Float32Array.from([0.1, 1, -1]);
    const right = Float32Array.from([0.2, 1, 1]);
    expect([...mixToMono([left, right])]).toEqual(
      [0, 1, 2].map((index) => Math.fround(((left[index] ?? 0) + (right[index] ?? 0)) / 2)),
    );
  });

  it("refuses no channel and channels of two lengths", () => {
    expect(() => mixToMono([])).toThrow(/no channel/);
    expect(() => mixToMono([new Float32Array(2), new Float32Array(3)])).toThrow(/lengths differ/);
  });
});

describe("audioSummary", () => {
  it("shows the name, the duration, the samples and the rate, and the channels mixed", () => {
    const audio = { data: new Float32Array(16100), sampleRate: 16000 };
    expect(audioSummary({ name: "click.wav", audio, channels: 1 })).toBe(
      "click.wav · 1.01 s · 16100 samples at 16000 Hz mono",
    );
    expect(audioSummary({ name: "click.wav", audio, channels: 2 })).toBe(
      "click.wav · 1.01 s · 16100 samples at 16000 Hz mono (mixed from 2 channels)",
    );
  });
});
