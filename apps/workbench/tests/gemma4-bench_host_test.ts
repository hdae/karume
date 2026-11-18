import { describe, expect, it } from "vite-plus/test";

import {
  type ChatMarks,
  type HostCall,
  type HostCapture,
  hostFigures,
  measureChat,
} from "../src/gemma4-bench/measure.ts";
import type { BenchSettings } from "../src/gemma4-bench/plan.ts";
import { markedSignal, type RunArm, settleDecodeBench } from "../src/gemma4-bench/run.ts";
import { hostSummaryText, summarizeHost, summarizeRuns } from "../src/gemma4-bench/summary.ts";
import { createGpuTap } from "../src/gemma4-bench/tap.ts";
import { fakeWebGpu } from "./helpers/fake-webgpu.ts";
import { fakeBenchDependencies } from "./helpers/gemma4-bench-fakes.ts";

const MAP_READ = 0x0001;
const COPY_DST = 0x0008;
const STORAGE = 0x0080;

/** A device acquired through a tap whose clock ticks 1 per reading, with its probe in the host mode. */
const hostDevice = async () => {
  const fake = fakeWebGpu();
  let clock = 0;
  const tap = createGpuTap(fake.gpu, () => (clock += 1));
  const adapter = await tap.gpu.requestAdapter();
  if (adapter === null) throw new Error("no adapter");
  const device = await adapter.requestDevice();
  const probe = tap.probe(device);
  if (probe === undefined) throw new Error("no probe");
  probe.mode = "host";
  probe.enabled = true;
  return { fake, device, probe };
};

/** One run as the product issues it: a queue write, one pass, a submit, then the read-back's map cycle. */
const runOnce = async (device: GPUDevice, readBytes: number) => {
  const staging = device.createBuffer({ size: readBytes, usage: MAP_READ | COPY_DST });
  device.queue.writeBuffer(staging, 0, new Uint8Array(4));
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.dispatchWorkgroups(1);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(MAP_READ);
  staging.getMappedRange();
  staging.unmap();
  return staging;
};

describe("the GPU tap in host mode", () => {
  it("marks a queue write, a new encoder, a submit and its return and a read-back buffer's map cycle, in call order on the tap's clock", async () => {
    const { device, probe } = await hostDevice();
    await runOnce(device, 64);
    expect(probe.marks()).toEqual([
      { kind: "write", at: 1 },
      { kind: "encoder", at: 2 },
      { kind: "submit", at: 3 },
      { kind: "submitted", at: 4 },
      { kind: "map", at: 5 },
      { kind: "mapped", at: 6 },
      { kind: "range", at: 7, bytes: 64 },
      { kind: "unmap", at: 8 },
    ]);
    expect(probe.submits()).toBe(1);
  });

  it("leaves the passes as the product encodes them: no timestamps and no pass record", async () => {
    const { fake, device, probe } = await hostDevice();
    await runOnce(device, 8);
    expect(fake.realPasses.map(({ descriptor }) => descriptor)).toEqual([undefined]);
    expect(fake.realPasses[0].calls).toEqual([["dispatch", 1, 1, 1], ["end"]]);
    expect(probe.passes()).toEqual([]);
  });

  it("marks the map cycle of read-back buffers only", async () => {
    const { device, probe } = await hostDevice();
    const storage = device.createBuffer({ size: 16, usage: STORAGE | COPY_DST });
    await storage.mapAsync(MAP_READ);
    storage.getMappedRange();
    storage.unmap();
    expect(probe.marks()).toEqual([]);
  });

  it("marks nothing in the other modes or while disabled, the calls still reaching the device", async () => {
    const { fake, device, probe } = await hostDevice();
    probe.mode = "passes";
    await runOnce(device, 8);
    probe.mode = "host";
    probe.enabled = false;
    await runOnce(device, 8);
    expect(probe.marks()).toEqual([]);
    expect(fake.devices[0].submitted).toEqual([1, 1]);
    expect(fake.devices[0].written).toEqual([0, 0]);
  });

  it("stops marking a read-back buffer made in the host mode once the mode is left", async () => {
    const { device, probe } = await hostDevice();
    const staging = await runOnce(device, 8);
    const marked = probe.marks().length;
    probe.enabled = false;
    await staging.mapAsync(MAP_READ);
    staging.getMappedRange();
    staging.unmap();
    expect(probe.marks()).toHaveLength(marked);
  });

  it("starts over after a reset", async () => {
    const { device, probe } = await hostDevice();
    await runOnce(device, 8);
    probe.reset();
    expect(probe.marks()).toEqual([]);
    device.queue.submit([device.createCommandEncoder().finish()]);
    expect(probe.marks().map(({ kind }) => kind)).toEqual(["encoder", "submit", "submitted"]);
  });
});

/**
 * A chat whose first token arrives at 100 ms and every later one 10 ms after the one before. Each
 * decode step, from the previous token at `t`: the signal is checked at `t + 1.5` (after the yield)
 * and at `t + 2` (before the run), the inputs are written at `t + 2.25`, two submits return at
 * `t + 3` and `t + 4`, the map is asked for at `t + 4.25` and settles at `t + 9`, the 1 MiB range is
 * read by `t + 9.25`, and the token is delivered at `t + 10`. The prefill makes calls of its own.
 */
const hostChat = (
  tokens: number,
  options: { readonly signal?: boolean; readonly stop?: boolean } = {},
) => {
  const calls: HostCall[] = [
    { kind: "write", at: 10 },
    { kind: "submit", at: 20 },
    { kind: "submitted", at: 21 },
    { kind: "map", at: 22 },
    { kind: "mapped", at: 90 },
    { kind: "range", at: 91, bytes: 1_048_576 },
    { kind: "unmap", at: 92 },
  ];
  const signalChecks: number[] = options.signal === false ? [] : [1, 5];
  // The steps after the first token; the last one draws the stop token when the chat stops on one.
  const steps = options.stop === true ? tokens : tokens - 1;
  for (let step = 0; step < steps; step += 1) {
    const t = 100 + step * 10;
    if (options.signal !== false) signalChecks.push(t + 1.5, t + 2);
    calls.push(
      { kind: "write", at: t + 2.25 },
      { kind: "encoder", at: t + 2.5 },
      { kind: "submit", at: t + 2.75 },
      { kind: "submitted", at: t + 3 },
      { kind: "encoder", at: t + 3.25 },
      { kind: "submit", at: t + 3.5 },
      { kind: "submitted", at: t + 4 },
      { kind: "map", at: t + 4.25 },
      { kind: "mapped", at: t + 9 },
      { kind: "range", at: t + 9.125, bytes: 1_048_576 },
      { kind: "unmap", at: t + 9.25 },
    );
  }
  const marks: ChatMarks = {
    sentAt: 0,
    prefillAt: [95],
    tokens: Array.from({ length: tokens }, (_unused, index) => ({
      at: 100 + index * 10,
      passes: 0,
      submits: 0,
    })),
    endAt: 100 + steps * 10 + (options.stop === true ? 0 : 0.5),
  };
  const host: HostCapture = { calls, signalChecks };
  return { marks, host };
};

describe("the host's segments of a decode token", () => {
  it("cuts a token's interval at the signal's first and last check, the first and the last submit's return, the map's settlement and the last unmap", () => {
    const { marks, host } = hostChat(5);
    const figures = hostFigures(host, marks, 4);
    expect(figures?.intervals).toBe(4);
    expect(figures?.meanMs).toEqual({
      between: 1.5,
      derive: 0.5,
      start: 1,
      encode: 1,
      wait: 5,
      read: 0.25,
      draw: 0.75,
    });
    expect(figures?.medianMs).toEqual(figures?.meanMs);
  });

  it("adds up to the token's interval", () => {
    const { marks, host } = hostChat(5);
    const mean = hostFigures(host, marks, 4)?.meanMs;
    expect(Object.values(mean ?? {}).reduce((sum, value) => sum + value, 0)).toBe(10);
  });

  it("puts the derived inputs in `between` for a chat without a signal (no check, no yield)", () => {
    const { marks, host } = hostChat(5, { signal: false });
    expect(hostFigures(host, marks, 4)?.meanMs).toEqual({
      between: 2.25,
      derive: 0,
      start: 0.75,
      encode: 1,
      wait: 5,
      read: 0.25,
      draw: 0.75,
    });
  });

  it("counts the run that drew a stop token as the last interval, up to the end of the chat", () => {
    // Four delivered tokens and a stop token: four decode runs, the last one's token not delivered.
    const { marks, host } = hostChat(4, { stop: true });
    const figures = hostFigures(host, marks, 4);
    expect(figures?.intervals).toBe(4);
    expect(figures?.meanMs.wait).toBe(5);
  });

  it("takes an interval without a submit — a token handed out of a batch already read back — as `between`", () => {
    const { marks, host } = hostChat(3);
    // The second step's calls are gone: its token came out of the first step's batch.
    const calls = host.calls.filter(({ at }) => at < 110 || at > 120);
    const figures = hostFigures({ calls, signalChecks: host.signalChecks }, marks, 2);
    expect(figures?.meanMs).toEqual({
      between: (1.5 + 10) / 2,
      derive: 0.25,
      start: 0.5,
      encode: 0.5,
      wait: 2.5,
      read: 0.125,
      draw: 0.375,
    });
  });

  it("takes the median over the tokens, so one slow wait does not move it", () => {
    const { marks, host } = hostChat(6);
    // The third step's map settles 100 ms late; its token and everything after move with it.
    const late = (at: number): number => (at >= 129 ? at + 100 : at);
    const calls = host.calls.map((call) => ({ ...call, at: late(call.at) }));
    const figures = hostFigures(
      { calls, signalChecks: host.signalChecks.map(late) },
      { ...marks, tokens: marks.tokens.map((token) => ({ ...token, at: late(token.at) })) },
      5,
    );
    expect(figures?.medianMs.wait).toBe(5);
    expect(figures?.meanMs.wait).toBe(25);
  });

  it("counts the calls and the mapped bytes per decode token, the prefill's left out", () => {
    const { marks, host } = hostChat(5);
    expect(hostFigures(host, marks, 4)).toMatchObject({
      submits: 2,
      queueWrites: 1,
      maps: 1,
      mappedBytes: 1_048_576,
    });
  });

  it("is absent for a chat without a decode token", () => {
    const { marks, host } = hostChat(1);
    expect(hostFigures(host, marks, 0)).toBeUndefined();
    expect(measureChat({ marks, promptTokens: 21, tokens: 1, host }).host).toBeUndefined();
  });

  it("is measured only for a chat that has the host's marks", () => {
    const { marks, host } = hostChat(5);
    expect(measureChat({ marks, promptTokens: 21, tokens: 5 }).host).toBeUndefined();
    expect(measureChat({ marks, promptTokens: 21, tokens: 5, host }).host?.intervals).toBe(4);
  });
});

describe("a marked signal", () => {
  it("notes every check and still throws the run's reason once the run is aborted", () => {
    const controller = new AbortController();
    let checks = 0;
    const marked = markedSignal(controller.signal, () => (checks += 1));
    marked.throwIfAborted();
    expect(checks).toBe(1);
    const reason = new Error("Cancelled by the user");
    controller.abort(reason);
    expect(marked.aborted).toBe(true);
    expect(() => marked.throwIfAborted()).toThrow(reason);
    expect(checks).toBe(2);
  });

  it("leaves the run's own signal as it was", () => {
    const controller = new AbortController();
    let checks = 0;
    markedSignal(controller.signal, () => (checks += 1));
    controller.signal.throwIfAborted();
    expect(checks).toBe(0);
  });
});

const arm = (patch: Partial<RunArm> = {}): RunArm => ({
  mount: "gemma4",
  model: "e2b",
  quant: "i4-fast",
  quantLabel: "int4 fast",
  isDefaultQuant: true,
  tap: true,
  sampler: { temperature: 0 },
  distribution: {
    ref: { repo: "test/e2b" },
    facts: { directory: "models/e2b", manifestSha256: "00", read: "directory" },
  },
  ...patch,
});

const SETTINGS: BenchSettings = {
  chats: 2,
  newTokens: 4,
  rounds: 1,
  sampler: "greedy",
  prompts: ["few-ids"],
};

const runBench = async (arms: readonly RunArm[], signal = new AbortController().signal) => {
  const { dependencies, probe, world } = fakeBenchDependencies();
  const modes: (string | undefined)[] = [];
  const build = dependencies.build;
  const outcome = await settleDecodeBench({
    arms,
    settings: SETTINGS,
    signal,
    dependencies: {
      ...dependencies,
      build: async (given, buildSignal) => {
        const pipeline = await build(given, buildSignal);
        return {
          ...pipeline,
          chat: (messages, options) => {
            modes.push(probe.mode);
            return pipeline.chat(messages, options);
          },
        };
      },
    },
    onProgress: () => {},
  });
  return { ...outcome, world, modes, signal };
};

describe("the bench loop", () => {
  describe("on an arm in the tap's host mode", () => {
    it("runs the probe in the host mode and measures the host's segments instead of the tap's figures", async () => {
      const { chats, modes, stop } = await runBench([arm({ hostMarks: true })]);
      expect(stop).toEqual({ kind: "completed" });
      expect(modes).toEqual(["host", "host"]);
      const [, warm] = chats;
      expect(warm.measurement.tap).toBeUndefined();
      // Three decode tokens after the first, the stop token's run included; one submit and one
      // 8-byte map each.
      expect(warm.measurement.host).toMatchObject({
        intervals: 3,
        submits: 1,
        maps: 1,
        mappedBytes: 8,
      });
    });

    it("sends the chat with a signal of its own that follows the run's, and cuts the segments at its checks", async () => {
      const controller = new AbortController();
      const { chats, world } = await runBench([arm({ hostMarks: true })], controller.signal);
      const sent = world.sentOptions[0].signal;
      expect(sent).toBeInstanceOf(AbortSignal);
      expect(sent).not.toBe(controller.signal);
      controller.abort(new Error("stop"));
      expect(sent?.aborted).toBe(true);
      // The fake checks the signal twice before each run, one clock tick apart.
      expect(chats[1].measurement.host?.meanMs.derive).toBe(1);
    });
  });

  describe("on an arm without the host mode", () => {
    it("sends the run's own signal and measures no host segment", async () => {
      const controller = new AbortController();
      const { chats, world, modes } = await runBench([arm()], controller.signal);
      expect(world.sentOptions[0].signal).toBe(controller.signal);
      expect(modes).toEqual(["passes", "passes"]);
      expect(chats[1].measurement.host).toBeUndefined();
    });
  });

  describe("on an arm sent without a signal", () => {
    it("passes no signal to the chat", async () => {
      const { world, stop } = await runBench([arm({ noSignal: true })]);
      expect(stop).toEqual({ kind: "completed" });
      for (const options of world.sentOptions) {
        expect(Object.keys(options).sort()).toEqual([
          "maxNewTokens",
          "onPrefill",
          "onToken",
          "sampler",
        ]);
      }
    });

    it("has no check to cut at, so the host's segments keep the derived inputs in `between`", async () => {
      const { chats } = await runBench([arm({ noSignal: true, hostMarks: true })]);
      expect(chats[1].measurement.host?.meanMs.derive).toBe(0);
    });
  });
});

describe("the summary of the host's segments", () => {
  /** A chat's figures with its mean wait set to `wait` and its per-token median wait 0.5 below. */
  const figures = (wait: number) => {
    const { marks, host } = hostChat(5);
    const measured = hostFigures(host, marks, 4);
    if (measured === undefined) throw new Error("no figures");
    return {
      ...measured,
      meanMs: { ...measured.meanMs, wait },
      medianMs: { ...measured.medianMs, wait: wait - 0.5 },
    };
  };

  it("spreads each segment over the warm chats: the chat's mean, and the chat's per-token median", () => {
    const summary = summarizeHost([figures(5), figures(6), figures(9)]);
    expect(summary?.chats).toBe(3);
    expect(summary?.meanMs.wait).toEqual({ count: 3, min: 5, median: 6, max: 9 });
    expect(summary?.medianMs.wait).toEqual({ count: 3, min: 4.5, median: 5.5, max: 8.5 });
    expect(summary?.meanMs.between).toEqual({ count: 3, min: 1.5, median: 1.5, max: 1.5 });
    expect(summary).toMatchObject({ submits: 2, queueWrites: 1, maps: 1, mappedBytes: 1_048_576 });
  });

  it("is absent for no chat, and for an arm × prompt whose warm chats have no host figures", () => {
    expect(summarizeHost([])).toBeUndefined();
    const { marks } = hostChat(5);
    const [entry] = summarizeRuns(
      [
        {
          arm: 0,
          prompt: "few-ids",
          cold: false,
          tokenIdsSha256: "same",
          measurement: measureChat({ marks, promptTokens: 21, tokens: 5 }),
        },
      ],
      1,
      ["few-ids"],
    );
    expect(entry.host).toBeUndefined();
  });

  it("reads as a table per arm and prompt: the segments, their sum, and the calls per token", () => {
    const { marks, host } = hostChat(5);
    const summary = summarizeRuns(
      [
        {
          arm: 0,
          prompt: "few-ids",
          cold: false,
          tokenIdsSha256: "same",
          measurement: measureChat({ marks, promptTokens: 21, tokens: 5, host }),
        },
      ],
      1,
      ["few-ids"],
    );
    expect(hostSummaryText(summary, [arm({ hostMarks: true })])).toEqual([
      "arm 1 Gemma 4 e2b i4-fast (host) · Story request · 1 warm chats · host segments, ms per decode token",
      "                      between   derive    start   encode     wait     read     draw      sum",
      "  mean (median)         1.500    0.500    1.000    1.000    5.000    0.250    0.750   10.000",
      "  mean (min)            1.500    0.500    1.000    1.000    5.000    0.250    0.750   10.000",
      "  per-token median      1.500    0.500    1.000    1.000    5.000    0.250    0.750   10.000",
      "  2.0 submits · 1.0 queue writes · 1.0 maps · 1,048,576 B mapped per token",
    ]);
  });
});
