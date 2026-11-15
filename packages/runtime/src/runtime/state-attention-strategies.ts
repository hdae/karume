/**
 * The strategies of the states-form attention (the attention that reads its keys and values from
 * resident KV slots — autoregressive generation) and their selection.
 *
 * The node runs three stages — ①QK writes the scores, ② takes the row statistics, ③PV writes the
 * output — and ① and ③ each have several kernels that compute the same thing: the sequential
 * reference form, a form that shares K or V row tiles, a form that splits the reduction across
 * threads, and for ① a form folded over the query heads of one kv plane; ② and ③ can also run as one
 * fused kernel. Which form a stage takes depends on the plan's row count, the layer's kind, the
 * Session's resolved `stateAttentionReduce` (from attention's numeric level) and the block plan.
 *
 * {@link resolveStateAttention} is the only place that decides it, and the only place that holds the
 * priority order between the forms. The recipe builder builds what it returns: key, WGSL, uniform and
 * workgroup count of a stage all follow from the one choice, so they cannot disagree (a key of one
 * form with the grid of another leaves tiles unwritten without an exception).
 *
 * {@link STATE_ATTENTION_STRATEGIES} states each form's requirement level and evidence. Nothing
 * selects by level yet.
 *
 * MUST: the inputs are values fixed when the plan is made (`chunkRows` comes from the declared shape).
 * Branching on a run's logical length would make one plan key point at different pipelines per run.
 *
 * NOTE: device-free (kernel modules for their eligibility functions only). Not part of the public
 * surface of mod.ts (ADR 0008). The readonly form (a drafter reading a lender's KV) always runs the
 * parallel forms at M = 1 and has no selection.
 */

import {
  assertStateQkFold,
  statePvTiledEligible,
  stateQkFoldHeads,
  stateQkParallelEligible,
  stateQkTiledEligible,
} from "../kernels/state-attention.ts";
import { ExecutionError } from "./plan.ts";
import type { StateAttentionReduce } from "./session-types.ts";
import type { StrategyInfo } from "./strategy-level.ts";

export type StateQkStrategyId = "qk-sequential" | "qk-tiled" | "qk-parallel" | "qk-folded";
export type StatePvStrategyId = "pv-sequential" | "pv-tiled" | "pv-parallel" | "stats-pv-fused";

/**
 * What each form is. The tiled forms are bit-identical to the sequential ones, so they are `exact`
 * and need no option. The parallel forms change the order of the reduction (`reorder`); the folded ①
 * and the fused ②③ give the parallel forms' bits and inherit their level.
 */
export const STATE_ATTENTION_STRATEGIES: Readonly<
  Record<StateQkStrategyId | StatePvStrategyId, StrategyInfo>
> = {
  "qk-sequential": { level: "exact", evidence: "reference", admittedBy: undefined },
  "qk-tiled": { level: "exact", evidence: "mirror", admittedBy: undefined },
  "qk-parallel": {
    level: "reorder",
    evidence: "reorder",
    admittedBy: 'numerics: attention at "reorder"',
  },
  "qk-folded": {
    level: "reorder",
    evidence: "mirror",
    admittedBy: 'numerics: attention at "reorder"',
  },
  "pv-sequential": { level: "exact", evidence: "reference", admittedBy: undefined },
  "pv-tiled": { level: "exact", evidence: "mirror", admittedBy: undefined },
  "pv-parallel": {
    level: "reorder",
    evidence: "reorder",
    admittedBy: 'numerics: attention at "reorder"',
  },
  "stats-pv-fused": {
    level: "reorder",
    evidence: "mirror",
    admittedBy: 'numerics: attention at "reorder"',
  },
};

/** Everything the selection of one states-form attention depends on. */
export type StateAttentionFacts = {
  /** The head of a refusal's message (`attention (states) [q] × [k]`). */
  readonly where: string;
  /** The plan's row count M (from the declared shape). */
  readonly chunkRows: number;
  readonly depth: number;
  /** Query heads per kv head (1 without GQA). */
  readonly kvRepeat: number;
  /** A sliding-window layer (a ring slot); otherwise a full layer. */
  readonly sliding: boolean;
  readonly reduce: StateAttentionReduce;
  /** The block plan fuses ② and ③ (`planStateAttention` — the form, the slot storage, the shape). */
  readonly fusedStatsPv: boolean;
  /** The generation declares bidirectional rows (image soft tokens — ADR 0144). */
  readonly bidirectionalRows: boolean;
};

export type StateQkChoice =
  | { readonly id: "qk-sequential" }
  | { readonly id: "qk-parallel" }
  /** One workgroup serves `heads` query heads of one kv plane. */
  | { readonly id: "qk-folded"; readonly heads: number }
  /** `bidirectional`: the kernel carries the predicate of `bidirectionalRows`. */
  | { readonly id: "qk-tiled"; readonly bidirectional: boolean };

export type StatePvChoice = { readonly id: StatePvStrategyId };

export type StateAttentionChoice = {
  readonly qk: StateQkChoice;
  readonly pv: StatePvChoice;
};

/** The outcome of the selection: the forms of the two stages, or the message of the refusal. */
export type StateAttentionResolution =
  | { readonly choice: StateAttentionChoice }
  | { readonly refusal: string };

/**
 * The form of ①QK.
 *
 * - The parallel form ①' only when the resolved form asks for it **and** the plan has at most 8
 *   rows (decode's M = 1 and speculative verify's M ≤ 8). ①' shortens the latency of one invocation's
 *   D-serial loop, the bottleneck only where the active invocations are "live columns × few rows"; in
 *   prefill it only stacks workgroups and barriers (measured 2026-09-06: decode ×1.55–1.72, prefill's
 *   ①QK GPU time 1.5–1.9× longer). The condition stays inside the measured range.
 * - ①' folded over the query heads of one kv plane where `stateQkFoldHeads` names a class (it was
 *   observed bit-identical to ①' on every backend checked, so it rides on ①''s selection without an
 *   option). MUST: `F | r` is checked before the folded form is chosen — r travels in the uniform
 *   and F in the key, and a workgroup whose heads span two kv planes reads another plane's K without
 *   an exception or a NaN.
 * - The tiled form ①ₜ for plans of at least 16 rows that did not take ①'. It is bit-identical to ①,
 *   so it needs no option. The two row conditions do not overlap (8 < 16).
 */
const qkChoice = (facts: StateAttentionFacts, bidirectional: boolean): StateQkChoice => {
  const { chunkRows, sliding, kvRepeat, depth, where } = facts;
  if (facts.reduce !== "sequential" && stateQkParallelEligible(chunkRows)) {
    const heads = stateQkFoldHeads(sliding, kvRepeat, depth, chunkRows);
    if (heads <= 1) return { id: "qk-parallel" };
    assertStateQkFold(heads, kvRepeat, where);
    return { id: "qk-folded", heads };
  }
  return stateQkTiledEligible(chunkRows)
    ? { id: "qk-tiled", bidirectional }
    : { id: "qk-sequential" };
};

/**
 * The form of ③PV.
 *
 * - The fused ②③ where the block plan chose it.
 * - The tiled form ③ₜ for plans of at least 16 rows — **before the resolved form**: a prefill plan
 *   under `"parallel"` gets ③ₜ (the reference path's values), not ③'. ③''s gain is confined to decode,
 *   while ③ₜ cuts traffic and is bit-identical to ③.
 * - The parallel form ③' when the resolved form asks for it; otherwise the sequential form.
 */
const pvChoice = ({ fusedStatsPv, chunkRows, reduce }: StateAttentionFacts): StatePvChoice => {
  if (fusedStatsPv) return { id: "stats-pv-fused" };
  if (statePvTiledEligible(chunkRows)) return { id: "pv-tiled" };
  return { id: reduce !== "sequential" ? "pv-parallel" : "pv-sequential" };
};

/**
 * The forms of ①QK and ③PV of one states-form attention, or why the plan is refused.
 *
 * A bidirectional block is as long as an image's soft tokens, far over ①ₜ's 16 rows, so only ①ₜ
 * carries the predicate of `bidirectionalRows`, and only on sliding layers (full layers stay causal).
 * A plan that asks for it and does not take ①ₜ is refused.
 */
export const resolveStateAttention = (facts: StateAttentionFacts): StateAttentionResolution => {
  const bidirectional = facts.sliding && facts.bidirectionalRows;
  const qk = qkChoice(facts, bidirectional);
  if (bidirectional && qk.id !== "qk-tiled") {
    return {
      refusal: `${facts.where}: generation.bidirectionalRows needs a plan of at least 16 rows (this one has ${facts.chunkRows})`,
    };
  }
  return { choice: { qk, pv: pvChoice(facts) } };
};

/** {@link resolveStateAttention} for a caller that fails on a refusal ({@link ExecutionError}). */
export const selectStateAttention = (facts: StateAttentionFacts): StateAttentionChoice => {
  const resolution = resolveStateAttention(facts);
  if ("refusal" in resolution) throw new ExecutionError(resolution.refusal);
  return resolution.choice;
};
