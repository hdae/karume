import { describe, expect, it } from "vite-plus/test";

import { buildBenchData } from "../src/gemma4-bench/data.ts";
import type { RunArm } from "../src/gemma4-bench/run.ts";

const armOf = (options: Pick<RunArm, "session" | "pipeline" | "drafter"> = {}): RunArm => ({
  mount: "gemma4-qat-mobile",
  model: "e2b",
  quant: "i4-fast",
  quantLabel: "int4 fast",
  isDefaultQuant: true,
  tap: true,
  ...options,
  sampler: { temperature: 0 },
  distribution: {
    ref: { repo: "test/gemma4-qat-mobile" },
    facts: { directory: "qat", manifestSha256: "00", read: "direct" },
  },
});

describe("the bench's data", () => {
  it("records per arm whether it ran the integer GEMV and the Session and pipeline options it passed ({} when none)", () => {
    const data = buildBenchData({
      timestampQuery: true,
      complete: true,
      settings: { chats: 2, newTokens: 2, rounds: 1, sampler: "greedy", prompts: ["few-ids"] },
      arms: [
        armOf(),
        armOf({ session: { integerGemv: true } }),
        armOf({ pipeline: { pleResidency: "gpu", decodeAhead: 4 } }),
      ],
      visits: [],
      chats: [],
    });
    expect(
      data.arms.map(({ integerGemv, sessionOptions, pipelineOptions }) => ({
        integerGemv,
        sessionOptions,
        pipelineOptions,
      })),
    ).toEqual([
      { integerGemv: false, sessionOptions: {}, pipelineOptions: {} },
      { integerGemv: true, sessionOptions: { integerGemv: true }, pipelineOptions: {} },
      {
        integerGemv: false,
        sessionOptions: {},
        pipelineOptions: { pleResidency: "gpu", decodeAhead: 4 },
      },
    ]);
  });

  it("records per arm whether it decoded with the MTP drafter", () => {
    const data = buildBenchData({
      timestampQuery: true,
      complete: true,
      settings: { chats: 2, newTokens: 2, rounds: 1, sampler: "greedy", prompts: ["few-ids"] },
      arms: [armOf(), armOf({ drafter: true })],
      visits: [],
      chats: [],
    });
    expect(data.arms.map(({ drafter }) => drafter)).toEqual([false, true]);
  });

  it("records per arm the GEMV choices its devices ran with when the bench has them, and no such field otherwise", () => {
    const input = {
      timestampQuery: true,
      complete: true,
      settings: { chats: 2, newTokens: 2, rounds: 1, sampler: "greedy", prompts: ["few-ids"] },
      arms: [armOf(), armOf(), armOf()],
      visits: [],
      chats: [],
    } as const;
    const choices = {
      switches: ["general-sequential"],
      table: "nvidia-ampere",
      tableCarriesGemv: true,
      linearGemv: {
        sequential: { variant: { cols: 32, unroll: 4 }, shapes: [] },
        rowsThreadTarget: 16384,
        parallelShapes: [],
      },
    } as const;
    // The third arm's first visit never began
    const data = buildBenchData({
      ...input,
      gemvChoices: [{ ...choices, switches: [] }, choices, undefined],
    });
    expect(data.arms.map(({ gemvChoices }) => gemvChoices)).toEqual([
      { ...choices, switches: [] },
      choices,
      undefined,
    ]);
    expect(data.arms[2]).not.toHaveProperty("gemvChoices");
    for (const arm of buildBenchData(input).arms) expect(arm).not.toHaveProperty("gemvChoices");
  });
});
