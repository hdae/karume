/**
 * The fusion context of the memory estimate (src/runtime/estimate.ts): it plans under the facts a
 * Session with the same options derives — the rule list, the knobs and the weight facts, through the
 * shared functions of src/runtime/session-fusion-context.ts — and takes the Session options the
 * fusion pass reads, with the Session's checks. No GPU.
 */

import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { UNREAD_ATTENTION_DEVICE } from "../src/runtime/attention-route.ts";
import { type EstimateOptions, estimateFusionFacts } from "../src/runtime/estimate.ts";
import {
  defineExternalRule,
  type FusedBuild,
  type FusionLimits,
  type FusionMatch,
} from "../src/runtime/fusion-rule.ts";
import { sessionFusionRules } from "../src/runtime/fusion.ts";
import { ExecutionError, type NodePlan } from "../src/runtime/plan.ts";
import {
  plannedResidentWeights,
  sessionFusionKnobs,
  sharedOrPooledWeights,
} from "../src/runtime/session-fusion-context.ts";
import {
  DISABLED_FUSION_RULES,
  ROW_BLOCK_SPLIT,
  type SessionOptions,
} from "../src/runtime/session-types.ts";
import { CORE_TRANSIENT_LIMITS } from "../src/runtime/transient-plan.ts";
import type { WeightResidency } from "../src/runtime/weight-residency.ts";
import { decomposedAttentionGraph, openGraph } from "./helpers/estimate-graphs.ts";

/** A JavaScript caller's value outside the type. */
const outsideTheType = <T>(value: unknown): T => value as T;

/** A residency plan with a compressed, an f32 and a borrowed weight. */
const RESIDENCY: ReadonlyMap<string, WeightResidency> = new Map<string, WeightResidency>([
  ["w_i8", { seat: "i8", payloadBytes: 4, scaleBytes: 4, rowAxis: 0 }],
  ["w_i4", { seat: "i4", payloadBytes: 4, scaleBytes: 4, groupSize: 32 }],
  ["w_raw", { seat: "raw", payloadBytes: 16 }],
  ["w_borrowed", { seat: "shared", eligible: true, i4Eligible: false, i2Eligible: false }],
]);

const DEVICE = { subgroups: true, maxComputeWorkgroupStorageSize: 32768 } as const;

/**
 * An external rule tried at every `bmm` that never takes a window; `seen` collects the limits its
 * matcher was given (a test-only side effect — the plan is the same with or without it).
 */
const watchingRule = (seen: FusionLimits[] = []) =>
  defineExternalRule({
    name: "test/watch_bmm",
    heads: ["bmm"],
    contract: "E",
    internals: "private",
    match: (_nodes: readonly NodePlan[], _index: number, context): FusionMatch | undefined => {
      seen.push({ ...context.limits });
      return undefined;
    },
    build: (): FusedBuild => {
      throw new Error("never built: the matcher takes no window");
    },
  });

/** The facts a Session constructed with `options` plans under, from the shared functions. */
const sessionFacts = (options: SessionOptions) => {
  const rules = sessionFusionRules(options.fusionRules);
  return {
    rules,
    knobs: sessionFusionKnobs(options, rules),
    attentionDevice: DEVICE,
    residentWeights: plannedResidentWeights(RESIDENCY),
    sharedOrPooledWeights: sharedOrPooledWeights(RESIDENCY, undefined),
  };
};

/** One score tensor = 4×2048×4200×4 B, over the WebGPU core binding limit unless row-blocked. */
const LARGE_SCORES = { heads: 4, queries: 2048, keys: 4200, headDim: 8 } as const;
const CORE_LIMIT = CORE_TRANSIENT_LIMITS.maxStorageBufferBindingSize;
const SMALL_SCORES = { heads: 2, queries: 8, keys: 8, headDim: 4 } as const;

describe("estimateFusionFacts", () => {
  it("derives the facts a Session with the same options plans under", () => {
    const optionSets: readonly EstimateOptions[] = [
      {},
      { linearCompute: "a8" },
      { attentionCompute: "a8", attentionScoreStorage: "none", attentionRowBlockMaxBytes: 4096 },
      {
        linearGemvReduce: "parallel",
        packedStaticQuantize: true,
        integerGemv: true,
        fuseLinearStaticQuantize: true,
        fuseRmsNormAdd: true,
        fuseGeluMulStaticQuantize: true,
        linearGemvPlanes: true,
        rmsNormReduce: "subgroup32",
        fusionRules: [watchingRule()],
        [DISABLED_FUSION_RULES]: ["silu", "test/watch_bmm"],
        maxStorageBufferBindingSize: CORE_LIMIT,
        maxComputeWorkgroupsPerDimension: 65535,
      },
    ];
    for (const options of optionSets) {
      assertEquals(
        estimateFusionFacts(RESIDENCY, undefined, options, DEVICE),
        {
          ...sessionFacts(options),
          maxComputeWorkgroupsPerDimension:
            options.maxComputeWorkgroupsPerDimension ?? Number.MAX_SAFE_INTEGER,
        },
        JSON.stringify(options),
      );
    }
  });

  it("does not take the test-only row-block split from a Session's options object", () => {
    const options: SessionOptions = { attentionRowBlockMaxBytes: 4096, [ROW_BLOCK_SPLIT]: 2 };
    const session = sessionFacts(options);
    assertEquals(session.knobs.rowBlockSplit, 2, "the Session takes it");
    assertEquals(estimateFusionFacts(RESIDENCY, undefined, options, DEVICE), {
      ...session,
      knobs: { ...session.knobs, rowBlockSplit: undefined },
      maxComputeWorkgroupsPerDimension: Number.MAX_SAFE_INTEGER,
    });
  });

  it("lists the expert plan's pooled weights among those an external step may not bind", () => {
    const pooled = [{ weights: [{ name: "w_i4" }, { name: "w_gate" }] }];
    assertEquals(
      estimateFusionFacts(RESIDENCY, pooled, {}, UNREAD_ATTENTION_DEVICE).sharedOrPooledWeights,
      new Set(["w_borrowed", "w_i4", "w_gate"]),
    );
  });
});

describe("the estimate given the Session options the fusion pass reads", () => {
  it("needs both device limits for a non-empty fusionRules, and names them", () => {
    const model = openGraph(decomposedAttentionGraph(SMALL_SCORES));
    for (const limits of [
      {},
      { maxStorageBufferBindingSize: CORE_LIMIT },
      { maxComputeWorkgroupsPerDimension: 65535 },
    ]) {
      assertThrows(
        () => model.estimate({ fusionRules: [watchingRule()], ...limits }),
        ExecutionError,
        "options.fusionRules needs options.maxStorageBufferBindingSize and options.maxComputeWorkgroupsPerDimension",
        JSON.stringify(limits),
      );
    }
    assertEquals(model.estimate({ fusionRules: [] }), model.estimate(), "an empty list needs none");
  });

  it("plans with the external rules and hands their matchers the limits it was given", () => {
    const model = openGraph(decomposedAttentionGraph(SMALL_SCORES));
    const seen: FusionLimits[] = [];
    const limits = {
      maxStorageBufferBindingSize: CORE_LIMIT,
      maxComputeWorkgroupsPerDimension: 4096,
    };
    const report = model.estimate({ fusionRules: [watchingRule(seen)], ...limits });
    assert(seen.length > 0, "the matcher was tried");
    assertEquals(
      new Set(seen.map((limit) => JSON.stringify(limit))),
      new Set([JSON.stringify(limits)]),
    );
    // A rule that takes no window leaves the plan, and with it the numbers, as without it.
    assertEquals(report, model.estimate(limits));
  });

  it("refuses an external rule list the Session refuses, with the Session's message", () => {
    assertThrows(
      () =>
        openGraph(decomposedAttentionGraph(SMALL_SCORES)).estimate({
          fusionRules: [watchingRule(), watchingRule()],
          maxStorageBufferBindingSize: CORE_LIMIT,
          maxComputeWorkgroupsPerDimension: 65535,
        }),
      ExecutionError,
      "options.fusionRules: the rule 'test/watch_bmm' is given twice",
    );
  });

  it("plans with the disabled rules left out: row-block attention off counts the scores unfused", () => {
    const model = openGraph(decomposedAttentionGraph(LARGE_SCORES));
    const limits = { maxStorageBufferBindingSize: CORE_LIMIT };
    // Fused, the window is counted as one row block's temporaries and the estimate returns (the
    // bytes are pinned in estimate_test.ts).
    assertEquals(model.estimate(limits).scenarios.length, 1);
    for (const disabled of [["rowBlockAttention"], "all"] as const) {
      assertThrows(
        () => model.estimate({ ...limits, [DISABLED_FUSION_RULES]: disabled }),
        ExecutionError,
        "'masked' 137625600B",
        JSON.stringify(disabled),
      );
    }
    assertThrows(
      () => model.estimate({ ...limits, [DISABLED_FUSION_RULES]: ["rowBlockAtention"] }),
      ExecutionError,
      "the disabled name(s) 'rowBlockAtention' are not names of the rule list",
    );
  });

  it("checks the new fields with the Session's checks and messages", () => {
    const model = openGraph(decomposedAttentionGraph(SMALL_SCORES));
    for (const [options, message] of [
      [
        { fuseRmsNormAdd: outsideTheType<boolean>("yes") },
        "options.fuseRmsNormAdd must be a boolean",
      ],
      [{ integerGemv: true }, "integerGemv requires packedStaticQuantize: true"],
      [
        { packedStaticQuantize: true },
        "packedStaticQuantize supports only linearGemvReduce: parallel / linearCompute: f32",
      ],
      [
        { rmsNormReduce: outsideTheType<"workgroup">("warp") },
        'options.rmsNormReduce: unsupported value "warp"',
      ],
      [
        { maxComputeWorkgroupsPerDimension: 0 },
        "options.maxComputeWorkgroupsPerDimension 0 must be a positive safe integer",
      ],
    ] satisfies readonly (readonly [EstimateOptions, string])[]) {
      assertThrows(() => model.estimate(options), ExecutionError, message, message);
    }
  });
});
