/**
 * The fusion context of a Session (src/runtime/session-fusion-context.ts) — the one mapping from the
 * facts a Session fixes to the context its plans are fused under, shared by the executor and the
 * fusion-count checks on the distribution graphs — and the knob and weight-fact derivations every
 * such caller shares. No GPU.
 */

import { describe, it } from "vite-plus/test";
import { assertEquals, assertThrows } from "@std/assert";
import { FUSION_RULES } from "../src/runtime/fusion.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import {
  plannedResidentWeights,
  type ResidentWeightStorage,
  type SessionFusionKnobs,
  sessionFusionContext,
  sessionFusionKnobs,
  sharedOrPooledWeights,
} from "../src/runtime/session-fusion-context.ts";
import { DISABLED_FUSION_RULES, ROW_BLOCK_SPLIT } from "../src/runtime/session-types.ts";
import type { WeightResidency } from "../src/runtime/weight-residency.ts";

/** A JavaScript caller's value outside the type. */
const outsideTheType = <T>(value: unknown): T => value as T;

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
  disabledFusionRules: undefined,
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
    sharedOrPooledWeights: new Set(["w_borrowed", "w_pooled"]),
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

  it("lists every weight not resident as f32 for the external rules, whatever the knobs: each stored form but f32, borrowed and pooled", () => {
    for (const knobs of [{}, { fuseLinearStaticQuantize: true }] as const) {
      assertEquals(
        contextOf(knobs).weightsNotF32,
        new Set(["w_i8", "w_i4", "w_bf16", "w_expert", "w_borrowed", "w_pooled"]),
      );
    }
  });

  it("carries the optional facts only when the Session has them", () => {
    const plain = contextOf({});
    assertEquals(
      ["rowBlockSplit", "attentionRowBlockMaxBytes", "attentionDevice", "disabledRules"].filter(
        (key) => Object.hasOwn(plain, key),
      ),
      [],
    );
    const forced = contextOf({ rowBlockSplit: 3, attentionRowBlockMaxBytes: 4096 });
    assertEquals([forced.rowBlockSplit, forced.attentionRowBlockMaxBytes], [3, 4096]);
  });

  it("carries the Session's disabled fusion rules to the pass", () => {
    assertEquals(contextOf({ disabledFusionRules: "all" }).disabledRules, "all");
    assertEquals(
      contextOf({ disabledFusionRules: new Set(["silu"]) }).disabledRules,
      new Set(["silu"]),
    );
  });
});

describe("sessionFusionKnobs", () => {
  it("fills the defaults Session construction fills when no option is given", () => {
    assertEquals(sessionFusionKnobs({}, FUSION_RULES), DEFAULT_KNOBS);
  });

  it("maps every option the fusion pass reads to the knob of the same name, the test-only symbols included", () => {
    assertEquals(
      sessionFusionKnobs(
        {
          fuseRmsNormAdd: true,
          fuseLinearStaticQuantize: true,
          fuseGeluMulStaticQuantize: true,
          packedStaticQuantize: true,
          integerGemv: true,
          linearGemvPlanes: true,
          linearGemvReduce: "parallel",
          linearCompute: "f32",
          attentionCompute: "a8",
          attentionScoreStorage: "none",
          rmsNormReduce: "subgroup32",
          attentionRowBlockMaxBytes: 4096,
          [ROW_BLOCK_SPLIT]: 2,
          [DISABLED_FUSION_RULES]: ["silu", "rope"],
        },
        FUSION_RULES,
      ),
      {
        fuseRmsNormAdd: true,
        fuseLinearStaticQuantize: true,
        fuseGeluMulStaticQuantize: true,
        packedStaticQuantize: true,
        integerGemv: true,
        linearGemvPlanes: true,
        linearGemvReduce: "parallel",
        linearCompute: "f32",
        attentionCompute: "a8",
        attentionScoreStorage: "none",
        rmsNormReduce: "subgroup32",
        rowBlockSplit: 2,
        disabledFusionRules: new Set(["silu", "rope"]),
        attentionRowBlockMaxBytes: 4096,
      },
    );
  });

  it("checks the disabled rule names against the rule list it is given, and refuses a value of another form", () => {
    assertEquals(
      sessionFusionKnobs({ [DISABLED_FUSION_RULES]: "all" }, FUSION_RULES).disabledFusionRules,
      "all",
    );
    assertThrows(
      () => sessionFusionKnobs({ [DISABLED_FUSION_RULES]: ["sliu"] }, FUSION_RULES),
      ExecutionError,
      "the disabled name(s) 'sliu' are not names of the rule list",
    );
    assertThrows(
      () =>
        sessionFusionKnobs(
          { [DISABLED_FUSION_RULES]: outsideTheType<readonly string[]>("silu") },
          FUSION_RULES,
        ),
      ExecutionError,
      'options[DISABLED_FUSION_RULES] must be "all" or an array of rule names',
    );
  });
});

describe("plannedResidentWeights / sharedOrPooledWeights", () => {
  const RESIDENCY: ReadonlyMap<string, WeightResidency> = new Map<string, WeightResidency>([
    ["w_raw", { seat: "raw", payloadBytes: 16 }],
    ["w_f16", { seat: "f16", payloadBytes: 8 }],
    ["w_bf16", { seat: "bf16", payloadBytes: 8 }],
    ["w_i8", { seat: "i8", payloadBytes: 4, scaleBytes: 4, rowAxis: 0 }],
    ["w_i2", { seat: "i2", payloadBytes: 4, scaleBytes: 4, rowAxis: 1 }],
    ["w_i4", { seat: "i4", payloadBytes: 4, scaleBytes: 4, groupSize: 64 }],
    ["w_expert", { seat: "i4f16", payloadBytes: 4, scaleBytes: 4, groupSize: 32 }],
    ["w_expanded", { seat: "expanded", payloadBytes: 8, expandedBytes: 16 }],
    ["w_borrowed", { seat: "shared", eligible: true, i4Eligible: false, i2Eligible: false }],
  ]);

  it("gives each compressed seat its storage (the group size from the plan) and leaves f32, expanded and borrowed weights out", () => {
    assertEquals(
      plannedResidentWeights(RESIDENCY),
      new Map<string, ResidentWeightStorage>([
        ["w_f16", { storage: "f16" }],
        ["w_bf16", { storage: "bf16" }],
        ["w_i8", { storage: "i8" }],
        ["w_i2", { storage: "i2" }],
        ["w_i4", { storage: "i4", groupSize: 64 }],
        ["w_expert", { storage: "i4f16", groupSize: 32 }],
      ]),
    );
  });

  it("lists the borrowed weights, and the pooled expert weights when there is a pool", () => {
    assertEquals(sharedOrPooledWeights(RESIDENCY, undefined), new Set(["w_borrowed"]));
    assertEquals(
      sharedOrPooledWeights(RESIDENCY, [
        { weights: [{ name: "w_expert" }, { name: "w_up" }] },
        { weights: [{ name: "w_down" }] },
      ]),
      new Set(["w_borrowed", "w_expert", "w_up", "w_down"]),
    );
  });
});
