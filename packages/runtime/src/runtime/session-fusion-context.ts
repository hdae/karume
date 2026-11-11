/**
 * The fusion context of a Session: the one mapping from the facts a Session fixes at construction to
 * the {@link FusionContext} its plans are fused under (`Session.#planSteps` in executor.ts).
 *
 * MUST: every caller that plans "what a Session would" builds its context here (the executor, and
 * the fusion-count checks on the distribution graphs in packages/e2e). Two hand-built copies drift
 * apart without an exception — a gate added to one and not the other only moves a hit count.
 */

import type { AttentionDeviceFacts } from "./attention-route.ts";
import type { FusionContext, FusionLimits, FusionRule, FusionWeightLayout } from "./fusion.ts";
import type { SessionState } from "./session-build.ts";

/**
 * The Session-fixed knobs the fusion pass reads: the fields of the Session state with the same
 * names, with the defaults Session construction filled in.
 */
export type SessionFusionKnobs = Pick<
  SessionState,
  | "fuseRmsNormAdd"
  | "fuseLinearStaticQuantize"
  | "fuseGeluMulStaticQuantize"
  | "packedStaticQuantize"
  | "integerGemv"
  | "linearGemvPlanes"
  | "linearGemvReduce"
  | "linearCompute"
  | "attentionCompute"
  | "attentionScoreStorage"
  | "rmsNormReduce"
  | "rowBlockSplit"
  | "disabledFusionRules"
  | "attentionRowBlockMaxBytes"
>;

/**
 * The storage of a resident weight without its GPU objects (the part of `ResidentWeight` in
 * weight-residency.ts the fusion decision reads). A weight that is not resident in compressed form
 * is absent and read as f32.
 */
export type ResidentWeightStorage =
  | { readonly storage: "f16" | "bf16" | "i8" | "i2" }
  | { readonly storage: "i4"; readonly groupSize: number }
  | { readonly storage: "i4f16"; readonly groupSize: number };

/** What {@link sessionFusionContext} maps to a {@link FusionContext}. */
export type SessionFusionFacts<Name extends string> = {
  /**
   * The rule list the Session plans with (`sessionFusionRules` in fusion.ts: the external rules it
   * was given, then the built-in `FUSION_RULES`).
   */
  readonly rules: readonly FusionRule<Name>[];
  readonly knobs: SessionFusionKnobs;
  /** Value name → consumer count in the graph (`countUses` in plan.ts). */
  readonly useCounts: ReadonlyMap<string, number>;
  readonly outputNames: ReadonlySet<string>;
  /** The device's granted limits the plan reads. */
  readonly limits: FusionLimits;
  /**
   * The device facts of the online attention route (a Session always has them). Omitted only by a
   * caller without a device; a decision that needs them then fails loudly in the pass.
   */
  readonly attentionDevice?: AttentionDeviceFacts;
  readonly residentWeights: ReadonlyMap<string, ResidentWeightStorage>;
  /**
   * The weights whose buffer is not this Session's own copy of the tensor: borrowed from another
   * Session (`SessionOptions.sharedWeights`, ADR 0096) or held in the expert cache's pool (ADR 0146 —
   * its buffer holds slots, not the tensor). With {@link residentWeights} they are the weights an
   * external rule's step may not bind (`FusionContext.weightsNotF32`).
   */
  readonly sharedOrPooledWeights: ReadonlySet<string>;
};

/**
 * The storage descriptions the pass reads, projected from the resident weights. An i4f16 weight
 * (expert_linear only — ADR 0145) feeds no fused linear, so it is absent.
 */
const fusionWeightLayouts = (
  weights: ReadonlyMap<string, ResidentWeightStorage>,
): ReadonlyMap<string, FusionWeightLayout> =>
  new Map(
    [...weights].flatMap(([name, weight]): [string, FusionWeightLayout][] => {
      if (weight.storage === "i4f16") return [];
      return [
        [
          name,
          weight.storage === "i4"
            ? { storage: weight.storage, groupSize: weight.groupSize }
            : { storage: weight.storage },
        ],
      ];
    }),
  );

/**
 * The {@link FusionContext} a Session's plans are fused under.
 *
 * The weight layouts are projected only when a decision reads them: the linear→SRQ rule
 * (`fuseLinearStaticQuantize`) and the packed-activation pairing (ADR 0105 — it looks at the storage
 * of the consuming linear) read them, and so does the linear prologue (perf-ledger F-4), which
 * decides the linear's i8a8 route from the same storage under `linearCompute: "a8"`.
 */
export const sessionFusionContext = <Name extends string>(
  facts: SessionFusionFacts<Name>,
): FusionContext<Name> => {
  const { knobs } = facts;
  const weightLayouts =
    knobs.fuseLinearStaticQuantize || knobs.packedStaticQuantize || knobs.linearCompute === "a8"
      ? fusionWeightLayouts(facts.residentWeights)
      : undefined;
  return {
    rules: facts.rules,
    ...(knobs.disabledFusionRules === undefined
      ? {}
      : { disabledRules: knobs.disabledFusionRules }),
    useCounts: facts.useCounts,
    fuseRmsNormAdd: knobs.fuseRmsNormAdd,
    fuseLinearStaticQuantize: knobs.fuseLinearStaticQuantize,
    fuseGeluMulStaticQuantize: knobs.fuseGeluMulStaticQuantize,
    packedStaticQuantize: knobs.packedStaticQuantize,
    integerGemv: knobs.integerGemv,
    linearGemvPlanes: knobs.linearGemvPlanes,
    linearGemvReduce: knobs.linearGemvReduce,
    linearCompute: knobs.linearCompute,
    attentionCompute: knobs.attentionCompute,
    // The output permute is absorbed only on the online route, decided by the same attentionRoute
    // buildAttention calls on these seats and device facts (perf-ledger F-3 A4).
    attentionScoreStorage: knobs.attentionScoreStorage,
    ...(facts.attentionDevice === undefined ? {} : { attentionDevice: facts.attentionDevice }),
    ...(weightLayouts === undefined ? {} : { weightLayouts }),
    // Every stored form but f32 (i4f16 included), whatever the knobs: only external rules read it.
    weightsNotF32: new Set([...facts.residentWeights.keys(), ...facts.sharedOrPooledWeights]),
    rmsNormReduce: knobs.rmsNormReduce,
    outputNames: facts.outputNames,
    limits: facts.limits,
    ...(knobs.rowBlockSplit === undefined ? {} : { rowBlockSplit: knobs.rowBlockSplit }),
    ...(knobs.attentionRowBlockMaxBytes === undefined
      ? {}
      : { attentionRowBlockMaxBytes: knobs.attentionRowBlockMaxBytes }),
  };
};
