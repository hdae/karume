import type { Manifest } from "@karume/hub";
import { describe, expect, it } from "vite-plus/test";

import {
  type ChatMarks,
  type KeyPerToken,
  measureChat,
  type PassTime,
  type TokenMark,
  UNLABELED_KEY,
} from "../src/models/gemma4-bench/measure.ts";
import { armLabel, resolveArm } from "../src/models/gemma4-bench/plan.ts";
import {
  splitSummaryText,
  summarizeRuns,
  summarizeSplit,
  type SummaryInput,
} from "../src/models/gemma4-bench/summary.ts";

const LINEAR = "linear_gemv_parallel:wi4g32:l32";
const NORM = "rms_norm:v1:f32:lastdim:wg256";
const EW = "ew:v3:mul:f32>f32:r3:wg256";

/** A split pass: one dispatch of `label`, `ms` of GPU time from `atMs` (timestamps in ns). */
const pass = (label: string | undefined, atMs: number, ms: number): PassTime => ({
  dispatches: 1,
  ...(label === undefined ? {} : { label }),
  beginNs: BigInt(Math.round(atMs * 1e6)),
  endNs: BigInt(Math.round((atMs + ms) * 1e6)),
});

/**
 * A split chat: a prefill pass (an embedding), the first token at 40 ms, then `steps` decode steps
 * 10 ms apart, each a 2 ms linear and two norms of 1 and 0.5 ms (the last step draws the stop token).
 */
const splitChat = (steps: number) => {
  const passes: PassTime[] = [pass("embedding:v1:f32:i32:wg256:wi8", 1, 5)];
  const tokens: TokenMark[] = [];
  for (let step = 0; step < steps; step += 1) {
    const at = 40 + step * 10;
    tokens.push({ at, passes: passes.length, submits: step + 1 });
    passes.push(pass(LINEAR, at + 1, 2), pass(NORM, at + 4, 1), pass(NORM, at + 6, 0.5));
  }
  const marks: ChatMarks = { sentAt: 0, prefillAt: [20], tokens, endAt: 40 + steps * 10 };
  return { marks, passes, submits: steps + 1 };
};

const measureSplit = (passes: readonly PassTime[], split = true) => {
  const chat = splitChat(4);
  return measureChat({
    marks: chat.marks,
    promptTokens: 21,
    tokens: 5,
    tap: { passes, submits: chat.submits, ...(split ? { split: true } : {}) },
  }).tap;
};

describe("a split chat's measurement", () => {
  it("breaks the decode window down per pipeline key, per decode token, GPU ms descending", () => {
    const figures = measureSplit(splitChat(4).passes);
    expect(figures?.perKey).toEqual([
      { key: LINEAR, dispatches: 1, gpuMs: 2 },
      { key: NORM, dispatches: 2, gpuMs: 1.5 },
    ]);
  });

  it("accounts for the whole decode window: its keys add up to the GPU ms per token", () => {
    const figures = measureSplit(splitChat(4).passes);
    const sum = (figures?.perKey ?? []).reduce((total, { gpuMs }) => total + (gpuMs ?? 0), 0);
    expect(sum).toBe(figures?.perToken?.gpuBusyMs);
  });

  it("leaves the GPU ms out of a key one of whose passes went untimed, and keeps its count", () => {
    const passes = splitChat(4).passes.map((recorded, index) =>
      index === 3 ? { dispatches: 1, label: NORM } : recorded,
    );
    expect(measureSplit(passes)?.perKey).toEqual([
      { key: LINEAR, dispatches: 1, gpuMs: 2 },
      { key: NORM, dispatches: 2 },
    ]);
  });

  it("gathers the passes without a pipeline label under one key", () => {
    const passes = splitChat(4).passes.map((recorded, index) =>
      index === 1 ? pass(undefined, 41, 2) : recorded,
    );
    expect(measureSplit(passes)?.perKey?.map(({ key }) => key)).toContain(UNLABELED_KEY);
  });

  it("has no breakdown when the tap did not run split", () => {
    expect(measureSplit(splitChat(4).passes, false)?.perKey).toBeUndefined();
  });
});

describe("the split summary", () => {
  const chats: readonly (readonly KeyPerToken[])[] = [
    [
      { key: LINEAR, dispatches: 1, gpuMs: 2 },
      { key: NORM, dispatches: 2, gpuMs: 1.5 },
    ],
    [
      { key: LINEAR, dispatches: 1, gpuMs: 4 },
      { key: NORM, dispatches: 2, gpuMs: 1.5 },
    ],
    [
      { key: LINEAR, dispatches: 1, gpuMs: 3 },
      { key: NORM, dispatches: 2, gpuMs: 1.5 },
      { key: EW, dispatches: 1, gpuMs: 0.3 },
    ],
  ];

  it("takes the median of every key and family over the chats, counting a missing key as 0", () => {
    const split = summarizeSplit(chats);
    expect(split?.keys).toEqual([
      { key: LINEAR, family: "linear", dispatches: 1, gpuMs: 3 },
      { key: NORM, family: "norm", dispatches: 2, gpuMs: 1.5 },
      { key: EW, family: "elementwise", dispatches: 0, gpuMs: 0 },
    ]);
    expect(split?.families).toEqual([
      { family: "linear", dispatches: 1, gpuMs: 3 },
      { family: "norm", dispatches: 2, gpuMs: 1.5 },
      { family: "elementwise", dispatches: 0, gpuMs: 0 },
    ]);
    // Per chat 3.5 / 5.5 / 4.8 ms.
    expect(split?.totalGpuMs).toBe(4.8);
    expect(split?.chats).toBe(3);
  });

  it("adds the keys of one family before taking the median", () => {
    const split = summarizeSplit([
      [
        { key: NORM, dispatches: 2, gpuMs: 1 },
        { key: "rms_norm_add:v2:x", dispatches: 1, gpuMs: 0.5 },
      ],
    ]);
    expect(split?.families).toEqual([{ family: "norm", dispatches: 3, gpuMs: 1.5 }]);
  });

  it("leaves a chat whose key went untimed out of that key's, its family's and the total's GPU ms", () => {
    const split = summarizeSplit([
      chats[0],
      [
        { key: LINEAR, dispatches: 1, gpuMs: 4 },
        { key: NORM, dispatches: 2 },
      ],
    ]);
    expect(split?.keys.find(({ key }) => key === NORM)).toEqual({
      key: NORM,
      family: "norm",
      dispatches: 2,
      gpuMs: 1.5,
    });
    expect(split?.keys.find(({ key }) => key === LINEAR)?.gpuMs).toBe(3);
    expect(split?.totalGpuMs).toBe(3.5);
  });

  it("is absent without a split chat", () => {
    expect(summarizeSplit([])).toBeUndefined();
  });

  /** A warm (or cold) chat of `arm` whose measurement carries `perKey` (or none). */
  const run = (arm: number, perKey?: readonly KeyPerToken[], cold = false): SummaryInput => ({
    arm,
    prompt: "few-ids",
    cold,
    tokenIdsSha256: "same",
    measurement: {
      timing: { totalSeconds: 1, decodeTokens: 127 },
      decodeTokens: 127,
      decodeMsPerToken: 10,
      ...(perKey === undefined
        ? {}
        : {
            tap: {
              prefill: { passes: 1, dispatches: 1, submits: 1, untimedPasses: 0 },
              decode: { passes: 1, dispatches: 1, submits: 1, untimedPasses: 0 },
              perKey,
            },
          }),
    },
  });

  it("breaks down each split arm over its warm chats only, and leaves the other arms without", () => {
    const entries = summarizeRuns(
      [
        run(0),
        run(0),
        run(1, [{ key: LINEAR, dispatches: 1, gpuMs: 90 }], true),
        run(1, chats[0]),
        run(1, chats[1]),
      ],
      2,
      ["few-ids"],
    );
    expect(entries[0].split).toBeUndefined();
    expect(entries[1].split?.chats).toBe(2);
    expect(entries[1].split?.keys[0]).toEqual({
      key: LINEAR,
      family: "linear",
      dispatches: 1,
      gpuMs: 3,
    });
  });

  it("writes the breakdown as text: per split arm and prompt, families then keys with their shares", () => {
    const entries = summarizeRuns([run(0, chats[0]), run(0, chats[2])], 1, ["few-ids"]);
    const text = splitSummaryText(entries, [
      { mount: "gemma4-qat", model: "e2b", quant: "i4", tap: true, split: true },
    ]);
    const lines = text.split("\n");
    expect(lines[0]).toBe(
      "arm 1 QAT e2b i4 (split) · Story request · 2 warm chats · GPU 4.150 ms/token (split)",
    );
    expect(lines).toContain(`     2.500  60.2%      1.0  linear`);
    expect(lines).toContain(`     1.500  36.1%      2.0  norm  ${NORM}`);
    expect(lines.findIndex((line) => line.endsWith("  linear"))).toBeLessThan(
      lines.findIndex((line) => line.endsWith(LINEAR)),
    );
  });

  it("writes nothing when no arm ran split", () => {
    expect(splitSummaryText(summarizeRuns([run(0)], 1, ["few-ids"]), [])).toBe("");
  });
});

describe("a split arm", () => {
  const manifest = {
    defaultModel: "e2b",
    models: {
      e2b: {
        defaultQuant: "i4",
        quants: { i4: { label: "int4" } },
        weights: {},
        pipelineConfig: {
          capacity: 4096,
          chunkLength: 768,
          maxChunkLength: 768,
          maxPosition: 32768,
          rope: {
            sliding_attention: { theta: 10000, headDim: 256, rotaryDim: 256 },
            full_attention: { theta: 1000000, headDim: 512, rotaryDim: 128 },
          },
        },
      },
    },
  } as unknown as Manifest;
  const draft = { id: 0, key: { mount: "gemma4-qat" as const, model: "e2b" }, split: true };

  it("is labelled as split", () => {
    expect(
      armLabel({ mount: "gemma4-qat", model: "e2b", quant: "i4", tap: true, split: true }),
    ).toBe("QAT e2b i4 (split)");
  });

  it("splits only with the tap on", () => {
    const on = resolveArm({ ...draft, tap: true }, { "gemma4-qat": manifest });
    const off = resolveArm({ ...draft, tap: false }, { "gemma4-qat": manifest });
    expect(on.ok && on.arm.split).toBe(true);
    expect(off.ok && off.arm.split).toBeUndefined();
    expect(off.ok && armLabel(off.arm)).toBe("QAT e2b i4 (tap off)");
  });
});
