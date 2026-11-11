/**
 * The fusion context of a Session (src/runtime/session-fusion-context.ts) — the one mapping from the
 * facts a Session fixes to the context its plans are fused under, shared by the executor and the
 * fusion-count checks on the distribution graphs. No GPU.
 */

import { describe, it } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { FUSION_RULES } from "../src/runtime/fusion.ts";
import {
  type ResidentWeightStorage,
  type SessionFusionKnobs,
  sessionFusionContext,
} from "../src/runtime/session-fusion-context.ts";

/** The knobs of a Session built with no options (the defaults construction fills in). */
const DEFAULT_KNOBS: SessionFusionKnobs = {
  fuseRmsNormAdd: false,
  fuseLinearStaticQuantize: false,
  fuseGeluMulStaticQuantize: false,
  packedStaticQuantize: false,
  integerGemv: false,
  linearGemvPlanes: false,
  linearGemvReduce: "sequential",
  linearCompute: "f32",
  attentionCompute: "f32",
  attentionScoreStorage: "f32",
  rmsNormReduce: "workgroup",
  rowBlockSplit: undefined,
  attentionRowBlockMaxBytes: undefined,
};

const RESIDENT: ReadonlyMap<string, ResidentWeightStorage> = new Map<string, ResidentWeightStorage>(
  [
    ["w_i8", { storage: "i8" }],
    ["w_i4", { storage: "i4", groupSize: 32 }],
    ["w_bf16", { storage: "bf16" }],
    ["w_expert", { storage: "i4f16", groupSize: 32 }],
  ],
);

const contextOf = (knobs: Partial<SessionFusionKnobs>) =>
  sessionFusionContext({
    rules: FUSION_RULES,
    knobs: { ...DEFAULT_KNOBS, ...knobs },
    useCounts: new Map(),
    outputNames: new Set(),
    limits: {
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
      maxComputeWorkgroupsPerDimension: 65535,
    },
    residentWeights: RESIDENT,
  });

describe("sessionFusionContext", () => {
  it("projects the weight layouts only under the knobs whose decisions read them, every resident storage but i4f16", () => {
    assertEquals(contextOf({}).weightLayouts, undefined, "no reader");
    for (const [label, knobs] of [
      ["linear→SRQ", { fuseLinearStaticQuantize: true }],
      ["packed activations", { packedStaticQuantize: true }],
      ["linear prologue", { linearCompute: "a8" }],
    ] as const) {
      assertEquals(
        contextOf(knobs).weightLayouts,
        new Map([
          ["w_i8", { storage: "i8" }],
          ["w_i4", { storage: "i4", groupSize: 32 }],
          ["w_bf16", { storage: "bf16" }],
        ]),
        label,
      );
    }
  });

  it("carries the optional facts only when the Session has them", () => {
    const plain = contextOf({});
    assertEquals(
      ["rowBlockSplit", "attentionRowBlockMaxBytes", "attentionDevice"].filter((key) =>
        Object.hasOwn(plain, key),
      ),
      [],
    );
    const forced = contextOf({ rowBlockSplit: 3, attentionRowBlockMaxBytes: 4096 });
    assertEquals([forced.rowBlockSplit, forced.attentionRowBlockMaxBytes], [3, 4096]);
  });
});
