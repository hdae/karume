/**
 * The fusion context of a Session: the one mapping from the facts a Session fixes at construction to
 * the {@link FusionContext} its plans are fused under (`Session.#planSteps` in executor.ts).
 *
 * MUST: every caller that plans "what a Session would" builds its context here (the executor, the
 * memory estimate, and the fusion-count checks on the distribution graphs in packages/e2e), and
 * derives its knobs ({@link sessionFusionKnobs}) and weight facts ({@link plannedResidentWeights},
 * {@link sharedOrPooledWeights}) through the functions below. Two hand-built copies drift apart
 * without an exception — a gate added to one and not the other only moves a hit count.
 *
 * Pure and device-free: no import here touches the GPU (the estimate imports this module).
 */

import type { AttentionDeviceFacts } from "./attention-route.ts";
import {
  assertDisabledRulesNamed,
  type FusionContext,
  type FusionLimits,
  type FusionRule,
  type FusionWeightLayout,
} from "./fusion.ts";
import { describeInputValue, ExecutionError } from "./plan.ts";
import type { SessionState } from "./session-build.ts";
import { executionKnobDefaults } from "./session-options-check.ts";
import { DISABLED_FUSION_RULES, ROW_BLOCK_SPLIT, type SessionOptions } from "./session-types.ts";
import type { WeightResidency } from "./weight-residency.ts";

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
 * The {@link SessionOptions} fields a Session's fusion context derives from: the knobs of
 * {@link SessionFusionKnobs} under their option names, the rule list (`fusionRules`) and the two
 * test-only symbols. A field the fusion pass reads is added here, so every caller that takes these
 * options (the memory estimate's `EstimateOptions` among them) sees it in its type.
 */
export type SessionFusionOptions = Pick<
  SessionOptions,
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
  | "attentionRowBlockMaxBytes"
  | "fusionRules"
  | typeof ROW_BLOCK_SPLIT
  | typeof DISABLED_FUSION_RULES
>;

/**
 * The fusion rules a Session leaves out (test only — `DISABLED_FUSION_RULES`) as the set its fusion
 * context carries, checked against the rule list the Session plans with (`rules` — the list
 * `Session.#planSteps` passes, external rules included) by the pass's own check, so a bad name fails
 * at construction instead of at the first run.
 */
export const sessionDisabledFusionRules = (
  disabled: SessionOptions[typeof DISABLED_FUSION_RULES],
  rules: readonly FusionRule[],
): "all" | ReadonlySet<string> | undefined => {
  if (disabled === undefined || disabled === "all") return disabled;
  const names: unknown = disabled;
  if (!Array.isArray(names) || !names.every((name) => typeof name === "string")) {
    throw new ExecutionError(
      `options[DISABLED_FUSION_RULES] must be "all" or an array of rule names (got ${describeInputValue(
        names,
      )})`,
    );
  }
  const set = new Set<string>(names);
  assertDisabledRulesNamed(rules, set);
  return set;
};

/**
 * The knobs a Session constructed with `options` fixes for its fusion pass, with the defaults
 * construction fills in (`executionKnobDefaults` — the one spelling of the defaults). `rules` is the
 * Session's rule list (`sessionFusionRules` in fusion.ts), against which the disabled names are
 * checked ({@link sessionDisabledFusionRules} — it throws `ExecutionError` on a bad value).
 *
 * MUST: the caller has run `sessionOptionsViolation` on `options` first (the knobs are read as
 * checked values; this adds no check of its own besides the disabled names).
 */
export const sessionFusionKnobs = (
  options: SessionFusionOptions,
  rules: readonly FusionRule[],
): SessionFusionKnobs => {
  const defaults = executionKnobDefaults(options);
  return {
    fuseRmsNormAdd: options.fuseRmsNormAdd ?? false,
    fuseLinearStaticQuantize: options.fuseLinearStaticQuantize ?? false,
    fuseGeluMulStaticQuantize: options.fuseGeluMulStaticQuantize ?? false,
    packedStaticQuantize: options.packedStaticQuantize ?? false,
    integerGemv: options.integerGemv ?? false,
    linearGemvPlanes: options.linearGemvPlanes ?? false,
    linearGemvReduce: defaults.linearGemvReduce,
    linearCompute: defaults.linearCompute,
    attentionCompute: defaults.attentionCompute,
    attentionScoreStorage: defaults.attentionScoreStorage,
    rmsNormReduce: defaults.rmsNormReduce,
    rowBlockSplit: options[ROW_BLOCK_SPLIT],
    disabledFusionRules: sessionDisabledFusionRules(options[DISABLED_FUSION_RULES], rules),
    attentionRowBlockMaxBytes: options.attentionRowBlockMaxBytes,
  };
};

/**
 * The storage of a resident weight without its GPU objects (the part of `ResidentWeight` in
 * weight-residency.ts the fusion decision reads). A weight that is not resident in compressed form
 * is absent and read as f32.
 */
export type ResidentWeightStorage =
  | { readonly storage: "f16" | "bf16" | "i8" | "i2" }
  | { readonly storage: "i4"; readonly groupSize: number }
  | { readonly storage: "i4f16"; readonly groupSize: number };

/**
 * The storage each initializer is resident in, from the residency plan alone (no device): the
 * classification Session construction applies when it uploads (the compressed seats keep their
 * storage, the group size comes with the plan — the planner copies it from the declaration, so it
 * is not derived twice). Uncompressed (`raw`) and expanded weights are f32 and absent.
 *
 * NOTE: a borrowed weight (`seat: "shared"`) is absent too: its storage is the lender's, which a
 * Session learns only from the lender's resident record at construction and a caller without the
 * lender cannot know. Such a caller reads it as f32 (a linear it feeds is counted unfolded — on the
 * large side); {@link sharedOrPooledWeights} still lists it, so no external rule binds it.
 * Session construction does not call this: its resident records carry GPU objects (the scale
 * buffers) made at upload, and a borrowed weight takes the lender's record.
 */
export const plannedResidentWeights = (
  residency: ReadonlyMap<string, WeightResidency>,
): ReadonlyMap<string, ResidentWeightStorage> =>
  new Map(
    [...residency].flatMap(([name, weight]): [string, ResidentWeightStorage][] => {
      switch (weight.seat) {
        case "i4":
        case "i4f16":
          return [[name, { storage: weight.seat, groupSize: weight.groupSize }]];
        case "f16":
        case "bf16":
        case "i8":
        case "i2":
          return [[name, { storage: weight.seat }]];
        default:
          return [];
      }
    }),
  );

/**
 * The weights whose buffer is not the Session's own copy of the tensor
 * ({@link SessionFusionFacts.sharedOrPooledWeights}): the borrowed ones of the residency plan, and
 * the expert weights held in the expert cache's pools (`pooled` — the layers of the expert plan or
 * of the pool built from it, ADR 0146; omitted without the cache).
 */
export const sharedOrPooledWeights = (
  residency: ReadonlyMap<string, WeightResidency>,
  pooled: readonly { readonly weights: readonly { readonly name: string }[] }[] | undefined,
): ReadonlySet<string> =>
  new Set([
    ...[...residency].flatMap(([name, weight]) => (weight.seat === "shared" ? [name] : [])),
    ...(pooled?.flatMap((layer) => layer.weights.map((weight) => weight.name)) ?? []),
  ]);

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
