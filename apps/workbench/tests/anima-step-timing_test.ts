import { describe, expect, it } from "vite-plus/test";

import {
  ANIMA_MOUNT_MODEL,
  ANIMA_STEP_TIMING_FORMAT,
  animaStepTimingData,
  animaStepTimingSummary,
  CLOCK,
  generationTiming,
  median,
  presetTiming,
  stepStatistics,
  type TimedEvent,
  timingModel,
} from "../src/anima-step-timing/record.ts";
import stepTiming from "../src/entries/local/anima-step-timing.ts";
import { ParameterError, parseParameters } from "../src/entry.ts";

/** The callbacks of one generation: stages around `stepMs` (each step's length in ms). */
const timeline = (stepMs: readonly number[]): TimedEvent[] => {
  const events: TimedEvent[] = [];
  let ms = 0;
  const at = (event: TimedEvent["event"], advance = 0): void => {
    ms += advance;
    events.push({ event, ms });
  };
  at({ kind: "stage", component: "text_encoder", at: "start" }, 100);
  at({ kind: "stage", component: "text_encoder", at: "end" }, 2000);
  at({ kind: "stage", component: "text_conditioner", at: "start" });
  at({ kind: "stage", component: "text_conditioner", at: "end" }, 500);
  at({ kind: "stage", component: "transformer", at: "start" });
  stepMs.forEach((length, index) =>
    at({ kind: "denoise-step", step: index + 1, steps: stepMs.length }, length),
  );
  at({ kind: "stage", component: "transformer", at: "end" }, 300);
  at({ kind: "stage", component: "vae_decoder", at: "start" });
  at({ kind: "stage", component: "vae_decoder", at: "end" }, 4000);
  return events;
};

const steps = (seconds: readonly number[]) => ({ stepSeconds: seconds });

describe("the anima-step-timing parameters", () => {
  it("default to the 2.9B mount, one preset without and one with int8 attention, three generations at 1024²", () => {
    expect(parseParameters(stepTiming.parameters, {})).toEqual({
      mount: "anima-2.9b",
      model: "",
      presets: ["bf16+dit8-a8", "bf16+dit8-a8-attn8-s16"],
      generations: 3,
      width: 1024,
      height: 1024,
      source: "cached",
    });
  });

  it("keep the presets in the declared order, whatever order they are written in", () => {
    expect(
      parseParameters(stepTiming.parameters, { presets: "bf16+dit8-a8-attn8,bf16" })["presets"],
    ).toEqual(["bf16", "bf16+dit8-a8-attn8"]);
  });

  it("refuse a mount the entry does not read and a preset no Anima distribution declares", () => {
    expect(() => parseParameters(stepTiming.parameters, { mount: "wan22" })).toThrow(
      ParameterError,
    );
    expect(() => parseParameters(stepTiming.parameters, { presets: "i8" })).toThrow(ParameterError);
  });
});

describe("the model a run times", () => {
  it("is the one named", () => {
    expect(timingModel("anima", "anima-turbo-v1.1", "anima-turbo-v1.1")).toBe("anima-turbo-v1.1");
  });

  it("is the 20-step base on the anima mount when none is named, not its turbo default", () => {
    expect(timingModel("anima", "", "anima-turbo-v1.1")).toBe(ANIMA_MOUNT_MODEL);
  });

  it("is the manifest's default on the other mounts", () => {
    expect(timingModel("anima-2.9b", "", "anima-2.9b-preview-v1")).toBe("anima-2.9b-preview-v1");
  });
});

describe("one generation's timeline", () => {
  it("times step 1 from the transformer stage's start and every later step from the step before", () => {
    const timing = generationTiming(42, 0, 20_000, timeline([5200, 1600, 30_600, 1700]));
    expect(timing.stepSeconds).toEqual([5.2, 1.6, 30.6, 1.7]);
    expect(timing.totalSeconds).toBe(20);
    expect(timing.seed).toBe(42);
  });

  it("times each stage from its start to its end callback", () => {
    const timing = generationTiming(42, 0, 1, timeline([1000, 1000]));
    expect(timing.stages).toEqual([
      { component: "text_encoder", seconds: 2 },
      { component: "text_conditioner", seconds: 0.5 },
      { component: "transformer", seconds: 2.3 },
      { component: "vae_decoder", seconds: 4 },
    ]);
    expect(timing.residency).toEqual([]);
  });

  it("keeps a residency event with its time from the start of generate", () => {
    const events: TimedEvent[] = [
      ...timeline([1000, 1000]),
      { event: { kind: "residency", action: "evicted", reason: "headroom" }, ms: 9500 },
    ];
    expect(generationTiming(42, 500, 10_000, events).residency).toEqual([
      { action: "evicted", reason: "headroom", atSeconds: 9 },
    ]);
  });

  it("refuses a step out of order and a step before the transformer stage", () => {
    const skipped = timeline([1000, 1000, 1000]).filter(
      ({ event }) => !(event.kind === "denoise-step" && event.step === 2),
    );
    expect(() => generationTiming(42, 0, 1, skipped)).toThrow(/denoise step 3 arrived after 1/);
    expect(() =>
      generationTiming(42, 0, 1, [{ event: { kind: "denoise-step", step: 1, steps: 2 }, ms: 5 }]),
    ).toThrow(/before the transformer stage started/);
  });

  it("refuses a stage that ends without having started", () => {
    expect(() =>
      generationTiming(42, 0, 1, [
        { event: { kind: "stage", component: "vae_decoder", at: "end" }, ms: 5 },
      ]),
    ).toThrow(/vae_decoder ended without having started/);
  });
});

describe("the stalled steps of a preset", () => {
  it("are the steps over ×3 the median, a stall split over two steps counted twice", () => {
    // The second generation of the 2.9B default preset as it was seen: steps 5, 11 and 17 stall.
    const second = [4.6, 1.7, 1.8, 1.7, 30.8, 1.8, 1.8, 1.7, 1.7, 1.8, 30.6, 1.8, 1.7, 1.7, 1.8];
    const third = [4.6, 1.7, 1.8, 1.7, 18.5, 13.9, 1.7, 1.8];
    const statistics = stepStatistics([steps(second), steps(third)]);
    expect(statistics.medianStepSeconds).toBe(1.8);
    expect(statistics.stepsCounted).toBe(second.length - 1 + third.length - 1);
    expect(statistics.stalled.map(({ generation, step }) => [generation, step])).toEqual([
      [1, 5],
      [1, 11],
      [2, 5],
      [2, 6],
    ]);
    expect(statistics.stalled[0]).toEqual({
      generation: 1,
      step: 5,
      seconds: 30.8,
      ratio: 30.8 / 1.8,
    });
  });

  it("leave step 1 out, so building the DiT Session is no stall", () => {
    const statistics = stepStatistics([steps([9, 1, 1, 1])]);
    expect(statistics).toEqual({ medianStepSeconds: 1, stepsCounted: 3, stalled: [] });
  });

  it("do not count a step at exactly ×3 the median", () => {
    expect(stepStatistics([steps([1, 1, 1, 3])]).stalled).toEqual([]);
  });

  it("read the median of an even count as the mean of its two middle values", () => {
    expect(median([4, 1, 3, 2])).toBe(2.5);
    expect(() => median([])).toThrow();
  });
});

describe("the anima-step-timing record", () => {
  const generation = (seed: number, stepMs: readonly number[]) =>
    generationTiming(seed, 0, 60_000, timeline(stepMs));
  const data = animaStepTimingData({
    request: {
      mount: "anima-2.9b",
      model: "anima-2.9b-preview-v1",
      presets: ["bf16+dit8-a8", "bf16+dit8-a8-attn8-s16"],
      generations: 2,
      resolution: { width: 1024, height: 1024 },
      steps: 4,
      guidanceScale: 4,
      prompt: "p",
      seeds: [42, 43],
    },
    distribution: {
      directory: "models/karume-anima-2.9b",
      manifestSha256: "c".repeat(64),
      read: "directory",
    },
    presets: [
      presetTiming({
        preset: "bf16+dit8-a8",
        session: { linearCompute: "a8" },
        buildSeconds: 3,
        generations: [
          generation(42, [5000, 2400, 2400, 2400]),
          generation(43, [5000, 2400, 2400, 2500]),
        ],
      }),
      presetTiming({
        preset: "bf16+dit8-a8-attn8-s16",
        session: { linearCompute: "a8", attentionCompute: "a8" },
        buildSeconds: 3,
        generations: [
          generation(42, [5000, 1700, 30_600, 1700]),
          generation(43, [5000, 1700, 1700, 1700]),
        ],
      }),
    ],
  });

  it("carries the format, the clock, the stall rule and each preset's statistics", () => {
    expect(data.format).toBe(ANIMA_STEP_TIMING_FORMAT);
    expect(data.clock).toBe(CLOCK);
    expect(data.stallRule).toEqual({ factor: 3, firstStepLeftOut: true });
    expect(data.presets.map((preset) => preset.stalled.length)).toEqual([0, 1]);
    expect(data.presets[1].medianStepSeconds).toBe(1.7);
    expect(data.presets[1].session).toEqual({ linearCompute: "a8", attentionCompute: "a8" });
  });

  it("says per preset how many steps stalled, then lists every generation's steps", () => {
    expect(animaStepTimingSummary(data)).toEqual([
      "anima-2.9b / anima-2.9b-preview-v1 at 1024×1024, 4 steps, guidance 4: 2 presets × 2 generations; steps timed by the wall clock between onEvent callbacks, a stall is a step over ×3 the preset's median (step 1 left out)",
      "bf16+dit8-a8: median step 2.40 s over 6 steps, 0 stalled steps; generations 60.0 s, 60.0 s",
      "bf16+dit8-a8-attn8-s16: median step 1.70 s over 6 steps, 1 stalled step (generation 1 step 3 30.6 s ×18.0); generations 60.0 s, 60.0 s",
      "bf16+dit8-a8 #1 steps 5.0 2.4 2.4 2.4",
      "bf16+dit8-a8 #2 steps 5.0 2.4 2.4 2.5",
      "bf16+dit8-a8-attn8-s16 #1 steps 5.0 1.7 30.6 1.7",
      "bf16+dit8-a8-attn8-s16 #2 steps 5.0 1.7 1.7 1.7",
      "residency events: none",
    ]);
  });

  it("is plain JSON (it survives the record's text unchanged)", () => {
    expect(JSON.parse(JSON.stringify(data))).toEqual(data);
  });
});
