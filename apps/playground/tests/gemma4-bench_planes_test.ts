import type { Manifest, SessionSpec } from "@karume/hub";
import { describe, expect, it } from "vite-plus/test";

import type { RecordEnvironment } from "../src/lib/environment.ts";
import { type ArmDraft, armLabel, resolveArm } from "../src/models/gemma4-bench/plan.ts";
import { buildBenchRecord } from "../src/models/gemma4-bench/record.ts";
import type { RunArm } from "../src/models/gemma4-bench/run.ts";

/** The QAT `i4-fast` declaration (the packed activations and what they need). */
const FAST: SessionSpec = {
  linearGemvReduce: "parallel",
  fuseRmsNormAdd: true,
  fuseLinearStaticQuantize: true,
  fuseGeluMulStaticQuantize: true,
  packedStaticQuantize: true,
};

/** A QAT manifest with `i4` (no session declaration) and `i4-fast` (the default). */
const manifest = {
  defaultModel: "e2b",
  models: {
    e2b: {
      defaultQuant: "i4-fast",
      quants: {
        i4: { label: "int4", weights: {}, session: {} },
        "i4-fast": { label: "int4 fast", weights: {}, session: FAST },
      },
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

const draft = (patch: Partial<ArmDraft> = {}): ArmDraft => ({
  id: 0,
  key: { mount: "gemma4-qat-mobile", model: "e2b" },
  tap: true,
  ...patch,
});

describe("the byte-plane GEMV switch", () => {
  describe("an arm", () => {
    it("passes linearGemvPlanes alone, whatever the preset declares", () => {
      for (const quant of ["i4-fast", "i4"]) {
        const arm = resolveArm(draft({ quant, linearGemvPlanes: true }), {
          "gemma4-qat-mobile": manifest,
        });
        expect(arm.ok && arm.arm.session).toEqual({ linearGemvPlanes: true });
      }
    });

    it("passes it next to what the integer switch passes when both are on", () => {
      const plain = resolveArm(draft({ quant: "i4", integerGemv: true, linearGemvPlanes: true }), {
        "gemma4-qat-mobile": manifest,
      });
      expect(plain.ok && plain.arm.session).toEqual({
        integerGemv: true,
        packedStaticQuantize: true,
        linearGemvReduce: "parallel",
        linearGemvPlanes: true,
      });
    });

    it("passes no Session options when both switches are off", () => {
      for (const linearGemvPlanes of [false, undefined]) {
        const off = resolveArm(draft({ linearGemvPlanes }), { "gemma4-qat-mobile": manifest });
        expect(off.ok && off.arm.session).toBeUndefined();
      }
    });

    it("is labelled as planes, after the integer mark and before the tap's own", () => {
      const arm = { mount: "gemma4-qat-mobile", model: "e2b", quant: "i4-fast" } as const;
      expect(armLabel({ ...arm, tap: true, session: { linearGemvPlanes: true } })).toBe(
        "mobile QAT e2b i4-fast (planes)",
      );
      expect(
        armLabel({
          ...arm,
          tap: false,
          session: { integerGemv: true, linearGemvPlanes: true },
        }),
      ).toBe("mobile QAT e2b i4-fast (integer) (planes) (tap off)");
      expect(
        armLabel({ ...arm, tap: true, split: true, session: { linearGemvPlanes: true } }),
      ).toBe("mobile QAT e2b i4-fast (planes) (split)");
    });
  });

  describe("the record", () => {
    const ENVIRONMENT: RecordEnvironment = {
      recordedAt: "2026-11-03T10:00:00.000Z",
      userAgent: "test",
      crossOriginIsolated: true,
      adapter: { vendor: "apple", architecture: "metal-3", device: "", description: "" },
      device: {
        features: [],
        wgslLanguageFeatures: [],
        subgroupLanesLinear: false,
        gpuTimingEnabled: false,
        limits: {
          maxComputeWorkgroupStorageSize: 32768,
          maxStorageBufferBindingSize: 2147483644,
          maxBufferSize: 2147483644,
        },
      },
    };
    const armOf = (session?: RunArm["session"]): RunArm => ({
      mount: "gemma4-qat-mobile",
      model: "e2b",
      quant: "i4-fast",
      quantLabel: "int4 fast",
      isDefaultQuant: true,
      tap: true,
      ...(session === undefined ? {} : { session }),
      sampler: { temperature: 0 },
    });

    it("records the option in the arm's Session options", () => {
      const record = buildBenchRecord({
        environment: ENVIRONMENT,
        timestampQuery: true,
        complete: true,
        settings: { chats: 2, newTokens: 2, rounds: 1, sampler: "greedy", prompts: ["few-ids"] },
        arms: [armOf(), armOf({ linearGemvPlanes: true })],
        distributions: { "gemma4-qat-mobile": { directory: "qat", manifestSha256: "00" } },
        source: { kind: "direct" },
        visits: [],
        chats: [],
      });
      expect(record.arms.map(({ sessionOptions }) => sessionOptions)).toEqual([
        {},
        { linearGemvPlanes: true },
      ]);
    });
  });
});
