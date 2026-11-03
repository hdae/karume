import type { Manifest, SessionSpec } from "@karume/hub";
import { describe, expect, it } from "vite-plus/test";

import type { RecordEnvironment } from "../src/lib/environment.ts";
import {
  type ArmDraft,
  armLabel,
  integerGemvSession,
  resolveArm,
} from "../src/models/gemma4-bench/plan.ts";
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
  key: { mount: "gemma4-qat", model: "e2b" },
  tap: true,
  ...patch,
});

describe("the integer GEMV switch", () => {
  describe("the explicit Session options it passes", () => {
    it("is integerGemv alone when the preset declares the packed activations", () => {
      expect(integerGemvSession(FAST)).toEqual({ integerGemv: true });
    });

    it("adds the packed activations and the parallel GEMV a preset does not declare", () => {
      expect(integerGemvSession({})).toEqual({
        integerGemv: true,
        packedStaticQuantize: true,
        linearGemvReduce: "parallel",
      });
      expect(integerGemvSession({ linearGemvReduce: "parallel" })).toEqual({
        integerGemv: true,
        packedStaticQuantize: true,
      });
      expect(
        integerGemvSession({ linearGemvReduce: "parallel", packedStaticQuantize: false }),
      ).toEqual({ integerGemv: true, packedStaticQuantize: true });
    });
  });

  describe("an arm", () => {
    it("carries the options for its preset when the switch is on, and none when it is off", () => {
      const fast = resolveArm(draft({ integerGemv: true }), { "gemma4-qat": manifest });
      expect(fast.ok && fast.arm.session).toEqual({ integerGemv: true });
      const plain = resolveArm(draft({ quant: "i4", integerGemv: true }), {
        "gemma4-qat": manifest,
      });
      expect(plain.ok && plain.arm.session).toEqual({
        integerGemv: true,
        packedStaticQuantize: true,
        linearGemvReduce: "parallel",
      });
      for (const integerGemv of [false, undefined]) {
        const off = resolveArm(draft({ integerGemv }), { "gemma4-qat": manifest });
        expect(off.ok && off.arm.session).toBeUndefined();
      }
    });

    it("is labelled as integer, before the tap's own mark", () => {
      const session = { integerGemv: true } as const;
      expect(
        armLabel({ mount: "gemma4-qat", model: "e2b", quant: "i4-fast", tap: true, session }),
      ).toBe("QAT e2b i4-fast (integer)");
      expect(
        armLabel({
          mount: "gemma4-qat",
          model: "e2b",
          quant: "i4-fast",
          tap: true,
          split: true,
          session,
        }),
      ).toBe("QAT e2b i4-fast (integer) (split)");
      expect(
        armLabel({ mount: "gemma4-qat", model: "e2b", quant: "i4", tap: false, session }),
      ).toBe("QAT e2b i4 (integer) (tap off)");
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
      mount: "gemma4-qat",
      model: "e2b",
      quant: "i4-fast",
      quantLabel: "int4 fast",
      isDefaultQuant: true,
      tap: true,
      ...(session === undefined ? {} : { session }),
      sampler: { temperature: 0 },
    });

    it("records per arm whether it ran the integer GEMV and the options it passed", () => {
      const record = buildBenchRecord({
        environment: ENVIRONMENT,
        timestampQuery: true,
        complete: true,
        settings: { chats: 2, newTokens: 2, rounds: 1, sampler: "greedy", prompts: ["few-ids"] },
        arms: [armOf(), armOf({ integerGemv: true })],
        distributions: { "gemma4-qat": { directory: "qat", manifestSha256: "00" } },
        source: { kind: "direct" },
        visits: [],
        chats: [],
      });
      expect(
        record.arms.map(({ integerGemv, sessionOptions }) => ({ integerGemv, sessionOptions })),
      ).toEqual([
        { integerGemv: false, sessionOptions: {} },
        { integerGemv: true, sessionOptions: { integerGemv: true } },
      ]);
    });
  });
});
