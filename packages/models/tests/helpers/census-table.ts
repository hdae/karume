/**
 * **The per-bundle census table** (ADR 0110 decision 5 ① / ADR 0058 decision 4 ③) — family × model
 * × effective SessionOptions × component × phase → the **expected count (exact)** per option.
 *
 * The predicates that tell options apart live in runtime's `tests/helpers/pipeline-census.ts`
 * (`SEAT_SIGNATURES`); this file holds only "how many variant keys and how many reference-variant
 * keys run in that phase of that bundle". The e2e tests check a bundle that has a row in this table
 * for the **exact** count through {@link phaseExpectations} → `assertPipelineCensus` (a bundle
 * without a row is checked as before by `assertSeatsApplied`, at least one / zero).
 *
 * ## Key
 *
 * A bundle's key is made by {@link bundleKey} from the **effective** SessionOptions (the result of
 * passing the quant preset's `session` from the manifest and the explicit overrides through the
 * family's merge `resolveSessionOptions` — ADR 0111). It is not looked up by the literal `session`
 * of the manifest because a value added outside the merge (as gemma's models default did before
 * stateAttentionReduce was promoted) would make the table key and the bundle that actually runs
 * diverge. Fields at their reference value are dropped from the key (an explicit
 * `linearGemvReduce: "sequential"` is the same bundle as leaving it unset).
 *
 * ## Where the values come from (as of 2026-09-26, all derived without a GPU)
 *
 * - **Values scattered across existing tests**: irodori DiT's i8a8 linears 317
 *   (`e2e_irodori_ab_test.ts` — after ADR 0114 split the graph, `dit` 245 + `dit_context` 72),
 *   gemma4 E2B decode's GEMVs 277 (`e2e_gemma4_greedy_test.ts`), RMS→add fusion 106 / linear→SRQ
 *   fusion 275 / packed SRQ 70 (`runtime/tests/e2e_*_fusion_counts_test.ts`).
 * - **Values derived mechanically from the distribution mirrors' plans**: read the container's
 *   part 0 (the graph declarations only) and, with the same approach as the GPU-free describe of
 *   `gpu_linear_gemv_parallel_test.ts`, pass each linear's storage (`planWeightResidency`) and
 *   shape through the same predicates as the recipe builder (`linearGemvParallelEligible` / i8a8's
 *   "i8 or i4 × k % 4 == 0" / f16 compute's "anything but i8 or i4") to get the counts. The counts
 *   set by fusion (RMS→add, linear→SRQ, packed) are the dispatch keys the fusion steps of
 *   `planFusions` hold at plan time, counted with `SEAT_SIGNATURES` (WebGPU core default limits —
 *   the same as `e2e_*_fusion_counts_test.ts`). States-form attention gives one each of ① and ③ per
 *   attention node, from the choice in `recipe-builders/attention.ts` (①' at M ≤ 8 / ③' at M < 16;
 *   readonly always ①'③') and a single row block at M ≤ 8. The gemma4-qat E4B rows were derived
 *   this way (after confirming that the same computation reproduces E2B's known values 275 / 2,
 *   106, 275, 70 + 70 and 35 layers, read at M 1 / 4 / 8 (prefill RMS→add at M 32 / 768)).
 *
 * ## What is not written (not derived — no row / phase / option is entered)
 *
 * - Prefill's parallel GEMV and states-form attention: the count depends on the physical chunk
 *   row count (the bucket). Prefill carries only the M-independent fusion (RMS→add).
 * - `stateAttentionReduce: "parallel-fused"`: whether the fused kernel is taken depends on the
 *   column limit of the full layers (= the capacity).
 * - `attentionCompute` / `attentionScoreStorage`: the number of row blocks of non-states-form
 *   attention depends on the device's `maxStorageBufferBindingSize` (the S of anima DiT at 1024px
 *   does not fit in one).
 *
 * NOTE: the values of phase `verify` assume speculation's verify row count M = k + 1 ≤ 8 (the
 * measured parallel GEMV table and the range of ①' cover M ≤ 8). Looking up this table with
 * speculation at k ≥ 8 gives zero parallel GEMVs and turns red.
 * NOTE: the census is the **planned** count (`SessionDiagnostics.lastRunPipelines`) and counts a
 * states-form dispatch with zero work as one. The states-form counts here are only for decode /
 * verify / draft at M ≤ 8, where there is one row block, so plan and issue do not differ.
 */

import type { SessionSpec } from "@karume/hub";
import type { SessionDiagnostics, SessionOptions } from "@karume/runtime";
import { linearKey } from "../../../runtime/src/kernels/linear.ts";
import { linearGemvKey, linearGemvParallelKey } from "../../../runtime/src/kernels/linear-gemv.ts";
import { statePvKey, stateQkKey } from "../../../runtime/src/kernels/state-attention.ts";
import {
  assertPipelineCensus,
  type CensusExpectation,
  type KeyPredicate,
  type NumericSeat,
  SEAT_REFERENCE,
  SEAT_SIGNATURES,
} from "../../../runtime/tests/helpers/pipeline-census.ts";
import { ANIMA_SESSION_POLICY } from "../../src/anima/pipeline.ts";
import { BIREFNET_SESSION_POLICY } from "../../src/birefnet/pipeline.ts";
import { DEPTH_ANYTHING_SESSION_POLICY } from "../../src/depth-anything/pipeline.ts";
import { GEMMA_SESSION_POLICY } from "../../src/gemma/session-options.ts";
import { IRODORI_SESSION_POLICY } from "../../src/irodori/admission.ts";
import { MOSS_SESSION_POLICY } from "../../src/moss/preset.ts";
import { SBV2_SESSION_POLICY } from "../../src/sbv2/pipeline.ts";
import {
  type FamilySessionPolicy,
  resolveSessionOptions,
  type SessionOverrides,
} from "../../src/session/options.ts";
import { SIGLIP2_SESSION_POLICY } from "../../src/siglip2/pipeline.ts";
import { UMT5_ENCODER_SESSION_POLICY } from "../../src/umt5/encoder.ts";
import { VOWEL_DETECTOR_SESSION_POLICY } from "../../src/vowel-detector/pipeline.ts";
import { WAN_SESSION_POLICY } from "../../src/wan/family.ts";

// --------------------------------------------------------------------------- Key
// ---------------------------------------------------------------------------

/**
 * Manifest pipeline name → the family's acceptance table (used to merge the effective
 * SessionOptions).
 *
 * MUST: {@link effectiveSessionOptions} fails on an unknown pipeline name — so that on the day a
 * new family's mirror is added, the census check does not silently leave that family out.
 */
export const SESSION_POLICIES: Readonly<Record<string, FamilySessionPolicy>> = {
  anima: ANIMA_SESSION_POLICY,
  birefnet: BIREFNET_SESSION_POLICY,
  "depth-anything": DEPTH_ANYTHING_SESSION_POLICY,
  gemma4: GEMMA_SESSION_POLICY,
  "gemma4-qat": GEMMA_SESSION_POLICY,
  irodori: IRODORI_SESSION_POLICY,
  "moss-image": MOSS_SESSION_POLICY,
  sbv2: SBV2_SESSION_POLICY,
  siglip2: SIGLIP2_SESSION_POLICY,
  "umt5-encoder": UMT5_ENCODER_SESSION_POLICY,
  "vowel-detector": VOWEL_DETECTOR_SESSION_POLICY,
  wan: WAN_SESSION_POLICY,
  // Wan2.2 TI2V shares 2.1's acceptance table (the common admission — `family.ts`).
  "wan-ti2v": WAN_SESSION_POLICY,
};

/**
 * The effective SessionOptions: the quant preset's declaration and the explicit overrides passed
 * through the **family's merge** (`resolveSessionOptions` — the one the entry points actually
 * call). The table keys are made from it.
 */
export const effectiveSessionOptions = (
  family: string,
  declared: SessionSpec,
  overrides: SessionOverrides = {},
  where = `census: ${family}`,
): SessionOptions => {
  if (!Object.hasOwn(SESSION_POLICIES, family)) {
    throw new Error(
      `${where}: pipeline '${family}' has no acceptance table in the census table (add it to SESSION_POLICIES)`,
    );
  }
  return resolveSessionOptions(SESSION_POLICIES[family], declared, overrides, where);
};

/**
 * The option names of the exhaustive table (`Object.keys` returns string[], so the type is taken
 * from the table again).
 */
const NUMERIC_SEATS = (Object.keys(SEAT_REFERENCE) as NumericSeat[]).toSorted();

/**
 * Effective SessionOptions → the bundle key (only the fields of the number-changing options that
 * hold a **non-reference value**, as `option=value` joined with `,` in option-name order). The
 * empty string when every option is at its reference value (= the reference bundle).
 *
 * NOTE: fields that do not change numbers (`submitPolicy` and so on — `NonNumericField` in
 * pipeline-census.ts) do not affect the census, so they are not in the key.
 */
export const bundleKey = (options: SessionOptions): string =>
  NUMERIC_SEATS.flatMap((seat) => {
    const value = options[seat];
    return value === undefined || value === SEAT_REFERENCE[seat] ? [] : [`${seat}=${value}`];
  }).join(",");

// --------------------------------------------------------------------------- Reference-variant
// predicates ---------------------------------------------------------------------------

/** The key's kernel family (the part before the first `:`). */
const familyOf = (key: string): string => {
  const colon = key.indexOf(":");
  return colon < 0 ? key : key.slice(0, colon);
};

/**
 * The kernel families that carry linear op dispatches (default GEMM skeleton, sequential GEMV (M=1
 * / row block), parallel GEMV).
 */
const LINEAR_OP_FAMILIES: ReadonlySet<string> = new Set(
  [linearKey("f32", true), linearGemvKey("i8"), linearGemvParallelKey("i8", undefined, 4)].map(
    familyOf,
  ),
);

/** Whether the key matches the predicate of any of the option's non-reference values. */
const anyVariant =
  (predicates: Readonly<Record<string, KeyPredicate>>): KeyPredicate =>
  (key) =>
    Object.values(predicates).some((predicate) => predicate(key));

/**
 * Every key of the sequential ①QK / ③PV (the states-form reference path) (the 4 combinations of
 * `(sliding, gqa)`).
 */
const STATE_SEQUENTIAL_KEYS: ReadonlySet<string> = new Set(
  [stateQkKey, statePvKey].flatMap((key) =>
    [false, true].flatMap((sliding) => [false, true].map((gqa) => key(sliding, gqa))),
  ),
);

/**
 * The options that have a reference-variant predicate (dispatches left on the option's reference
 * path although the option holds a non-reference value).
 *
 * Zero when fully eligible (ADR 0110 decision 5 ①). A bundle with a non-zero count records the
 * number of out-of-scope shapes that fall to the sequential path as is (a shape that leaves the
 * eligibility table moves it upwards — Fable review I8-02). The boolean fusion options have none,
 * because "nodes that were not fused" always remain by construction (a miss is visible from the
 * variant count alone).
 */
type ReferenceSeat = "linearCompute" | "linearGemvReduce" | "stateAttentionReduce";

const REFERENCE_SIGNATURES: { readonly [Seat in ReferenceSeat]: KeyPredicate } = {
  // Linear op dispatches that match none of linearCompute's variants (i8a8 / c16).
  linearCompute: (key) =>
    LINEAR_OP_FAMILIES.has(familyOf(key)) && !anyVariant(SEAT_SIGNATURES.linearCompute)(key),
  // Linear op dispatches that did not land in the parallel GEMV family (sequential GEMV, default
  // GEMM).
  linearGemvReduce: (key) =>
    LINEAR_OP_FAMILIES.has(familyOf(key)) && !anyVariant(SEAT_SIGNATURES.linearGemvReduce)(key),
  // The sequential ①QK / ③PV (the row-tile variants ①ₜ③ₜ are a default independent of the option,
  // so they are not included).
  stateAttentionReduce: (key) => STATE_SEQUENTIAL_KEYS.has(key),
};

// --------------------------------------------------------------------------- Table types
// ---------------------------------------------------------------------------

/** The expected counts of one option (per run, exact). */
type SeatCount<Seat extends NumericSeat> = Seat extends ReferenceSeat
  ? {
      /** The number of variant keys of the non-reference value the bundle selects. */
      readonly variant: number;
      /** The number of keys left on that option's reference path (0 when fully eligible). */
      readonly reference: number;
    }
  : { readonly variant: number };

/**
 * The expectations of one phase (only the bundle's non-reference options are written — the bundle's
 * `session` decides the value).
 *
 * NOTE: an option that is not written (not derived) has no count check in this phase. The e2e tests
 * keep checking it with `assertSeatsApplied`'s at least one / zero.
 */
export type PhaseCensus = { readonly [Seat in NumericSeat]?: SeatCount<Seat> };

/**
 * Per family, component → phase (the component names and phase spellings `onRunDiagnostics`
 * reports).
 */
type FamilyAxes = {
  readonly gemma4: GemmaAxes;
  readonly "gemma4-qat": GemmaAxes;
  readonly irodori: { readonly dit: "step"; readonly "dit-context": "run" };
  readonly anima: { readonly transformer: "step" };
  readonly sbv2: { readonly front: "run"; readonly voice: "run" };
  /** Phase `pass` is one DiT forward (one CFG step is two passes, conditional / unconditional). */
  readonly wan: { readonly transformer: "pass" };
  /**
   * Wan2.2 TI2V likewise (the shared modules' DiT stage reports the component name and phase — the
   * same spelling as 2.1).
   */
  readonly "wan-ti2v": { readonly transformer: "pass" };
  /** MOSS-Image: the DiT pair's two graphs; phase `pass` is one forward of a graph. */
  readonly "moss-image": { readonly dit: "pass"; readonly dit_skip22: "pass" };
};

/**
 * gemma's components (the product graph = target, speculation's drafter) and phases
 * (`GenerationRunPhase["kind"]`).
 */
type GemmaAxes = {
  readonly target: "prefill" | "decode" | "verify";
  readonly drafter: "draft";
};

/** The families that have rows in the table (manifest pipeline names). */
export type CensusFamily = keyof FamilyAxes;

type ComponentCensus<Family extends CensusFamily> = {
  readonly [Component in keyof FamilyAxes[Family]]?: {
    readonly [Phase in FamilyAxes[Family][Component] & string]?: PhaseCensus;
  };
};

/** One table row (family × models × bundle). */
type FamilyBundleRow<Family extends CensusFamily> = {
  readonly family: Family;
  /** The models with the same graph (the same counts) (keys of the manifest's models). */
  readonly models: readonly string[];
  /** The bundle (effective SessionOptions — the key is {@link bundleKey}). */
  readonly session: SessionOverrides;
  readonly census: ComponentCensus<Family>;
};

export type BundleCensusRow = {
  readonly [Family in CensusFamily]: FamilyBundleRow<Family>;
}[CensusFamily];

// --------------------------------------------------------------------------- Table
// ---------------------------------------------------------------------------

/**
 * Pipeline `gemma4` rows: the q4_0 QAT line's mirror (`models/karume-gemma4-qat`) — the line that
 * keeps the int4 presets once the regular line ships `bf16` alone. Re-derived 2026-11-04 from that
 * mirror's E2B / E4B `model` and `drafter` part 0 with the procedure of the file header, after the
 * same computation reproduced the regular i4 mirror's former rows (E2B 277 / 0, RMS→add 106,
 * 35 attention nodes, drafter 2 / 66 / 24 / 12; E4B 344 / 0, 127, 42). The values moved because the
 * line did: the q4_0 output head is int4, a shape the parallel GEMV table has no row for, where the
 * regular line's was int8.
 *
 * E2B (product graph): of the 277 linears, 276 (i4 g32) take a parallel GEMV row at M ≤ 8 and the
 * i4 lm_head 262144×1536 falls to the sequential GEMV.
 */
const GEMMA4_E2B_GEMV = { variant: 276, reference: 1 } as const;
/**
 * The E2B QAT drafter: of its 68 linears (all i4 g32) only `post_projection` 1536×256 (twice) is in
 * the parallel GEMV table — the other 66 (256×{1024,2048,3072}, {1024,2048}×256, lm_head 262144×256)
 * fall to the sequential GEMV (the table has no drafter shapes; the census pins today's count).
 */
const GEMMA4_E2B_DRAFTER_GEMV = { variant: 2, reference: 66 } as const;
/**
 * gemma4-qat E2B (product graph): of the 277 linears, 275 take the parallel GEMV (i2 60 + i4 145 +
 * i8 70). The other 2 fall to the sequential path — the i2 lm_head 262144×1536 (a shape the
 * measured table does not have) and the f32-stored per_layer_model_projection 8960×1536.
 */
const GEMMA4_QAT_E2B_GEMV = { variant: 275, reference: 2 } as const;
/** RMS→add fusion (independent of M — `e2e_*_fusion_counts_test.ts` pins it at M 1–64). */
const GEMMA4_E2B_RMS_ADD = { variant: 106 } as const;
const GEMMA4_E2B_DRAFTER_RMS_ADD = { variant: 24 } as const;
/** linear→SRQ fusion (all 275 that land in the parallel GEMV — M ≤ 8). */
const GEMMA4_QAT_E2B_LINEAR_SRQ = { variant: 275 } as const;
/**
 * gelu_tanh → mul → SRQ fold (perf-ledger K-48 step 1b): the per-layer-input gate of each of the 35
 * layers (M ≤ 8 — the measured form; a prefill plan keeps the plain chain).
 */
const GEMMA4_QAT_E2B_GELU_MUL_SRQ = { variant: 35 } as const;
/**
 * Packed int8 activations (ADR 0105): 70 standalone SRQs on the producer side (packed variant) + 70
 * packed variants of the parallel GEMV that reads them (whether linear→SRQ fusion is on changes the
 * consumer's kernel family, but both carry the packed fragment).
 */
const GEMMA4_QAT_E2B_PACKED = { variant: 140 } as const;
/**
 * States-form attention: the product graph's 35 layers × one each of ①' / ③' (M ≤ 8, one row
 * block).
 */
const GEMMA4_E2B_STATE_PARALLEL = { variant: 70, reference: 0 } as const;
/**
 * The drafter's 12 readonly attention nodes × one each of ①' / ③' (readonly is always the parallel
 * form, whatever the option).
 */
const GEMMA4_E2B_DRAFTER_STATE_PARALLEL = { variant: 24, reference: 0 } as const;

/**
 * 12B (product graph, 48 layers), read from a real run's `lastRunPipelines` at decode rather than
 * derived from the plan: all 329 linears, the int4 output head included, take the parallel GEMV.
 * `e2e_gemma4_12b_decode_ahead_test.ts` compares these with the run.
 */
const GEMMA4_12B_GEMV = { variant: 329, reference: 0 } as const;
const GEMMA4_12B_RMS_ADD = { variant: 96 } as const;
const GEMMA4_12B_STATE_PARALLEL = { variant: 96, reference: 0 } as const;

/**
 * 26B A4B (product graph, 30 layers — the mixture of experts), read from a real run of the mirror at
 * decode as 12B's: all 236 dense linears take the parallel GEMV (the 60 routed `expert_linear`
 * dispatches have no parallel form and are not GEMVs), RMS→add folds 90 times and states-form
 * attention runs ①' / ③' once each on the 30 layers. The run was under the exact expert cache
 * (ADR 0146), which changes where the routed experts live and none of these kernels. NOTE: no test
 * compares these rows with a run — the 12 GB development GPU cannot hold 26B resident.
 */
const GEMMA4_26B_GEMV = { variant: 236, reference: 0 } as const;
const GEMMA4_26B_RMS_ADD = { variant: 90 } as const;
const GEMMA4_26B_STATE_PARALLEL = { variant: 60, reference: 0 } as const;

/**
 * gemma4-qat E4B (product graph, 42 layers): of the 344 linears, 342 take the parallel GEMV — the
 * per-shape breakdown is pinned by the QAT E4B describe of runtime's
 * `gpu_linear_gemv_parallel_test.ts`. The other 2 fall to the sequential path for the same reason
 * as E2B (the i2 lm_head and the f32-stored per_layer_model_projection).
 */
const GEMMA4_QAT_E4B_GEMV = { variant: 342, reference: 2 } as const;
/**
 * RMS→add fusion (the same count in the decode plans at M 1–8 and the prefill plans at M 32 / 768).
 */
const GEMMA4_QAT_E4B_RMS_ADD = { variant: 127 } as const;
/** linear→SRQ fusion (all 342 that land in the parallel GEMV — M ≤ 8). */
const GEMMA4_QAT_E4B_LINEAR_SRQ = { variant: 342 } as const;
/** gelu_tanh → mul → SRQ fold: one per layer (42 — M ≤ 8, as E2B). */
const GEMMA4_QAT_E4B_GELU_MUL_SRQ = { variant: 42 } as const;
/**
 * Packed int8 activations: 84 standalone SRQs on the producer side (packed variant) + 84 packed
 * variants of the parallel GEMV that reads them (o 42 + down 42 — l32 of g2048 / g4096; broken down
 * the same way as E2B's 70 + 70).
 */
const GEMMA4_QAT_E4B_PACKED = { variant: 168 } as const;
/**
 * States-form attention: the product graph's 42 layers × one each of ①' / ③' (M ≤ 8, one row
 * block).
 */
const GEMMA4_QAT_E4B_STATE_PARALLEL = { variant: 84, reference: 0 } as const;

/**
 * gemma4 E4B (product graph, 42 layers — the q4_0 mirror, derived with the E2B rows above): of the
 * 344 linears, 343 (i4 g32) take a parallel GEMV row at M ≤ 8 and the i4 lm_head 262144×2560 falls
 * to the sequential GEMV; RMS→add folds 127 times (decode M 1 / 4 / 8 and prefill M 32 / 768 alike)
 * and states-form attention runs ①' / ③' once each on the 42 layers.
 */
const GEMMA4_E4B_GEMV = { variant: 343, reference: 1 } as const;
const GEMMA4_E4B_RMS_ADD = { variant: 127 } as const;
const GEMMA4_E4B_STATE_PARALLEL = { variant: 84, reference: 0 } as const;
/**
 * The E4B QAT drafter (the regular E4B shipped none, so these columns are new): of its 68 linears
 * (all i4 g32) only `post_projection` 2560×256 (twice) is in the parallel GEMV table — the other 66
 * (256×{1024,2048,5120}, {1024,2048}×256, lm_head 262144×256) fall to the sequential GEMV. RMS→add
 * folds 24 times and its 12 readonly attention nodes run ①' / ③' once each, as E2B's drafter.
 */
const GEMMA4_E4B_DRAFTER_GEMV = { variant: 2, reference: 66 } as const;
const GEMMA4_E4B_DRAFTER_RMS_ADD = { variant: 24 } as const;
const GEMMA4_E4B_DRAFTER_STATE_PARALLEL = { variant: 24, reference: 0 } as const;

const E2B = ["e2b"] as const;
const E4B = ["e4b"] as const;
const TWELVE_B = ["12b"] as const;
const TWENTY_SIX_B = ["26b"] as const;

/**
 * The irodori v4 DiT: 245 linears (of the 317 before the split, k ∈ {32, 192, 512, 768, 1280, 3680}
 * — all i8 × k % 4 == 0).
 *
 * The expected value changed (317 → 245 / `dit_context` 72) because the graph changed (ADR 0114 —
 * the 72 context K/V projections (12 blocks × text / speaker / caption × K / V) were split out into
 * `dit_context`). The count was not loosened. There is no check of the 317 total (pinning each
 * component exactly is enough, and a total can cancel out a mix-up on one side).
 */
const IRODORI_DIT_A8 = { variant: 245, reference: 0 } as const;
/**
 * The irodori v4 `dit_context`: the 72 linears of the context K/V projections (k ∈ {512, 768} — one
 * run per generation).
 */
const IRODORI_DIT_CONTEXT_A8 = { variant: 72, reference: 0 } as const;
/**
 * The irodori v4.1-small-mf DiT (the MeanFlow student — ADR 0150): the 245 of the RF `dit` plus the
 * 3 bias-free linears of `delta_cond_module` (k ∈ {512, 1280} — i8 × k % 4 == 0), counted from the
 * graph of the local mirror's `dit` (`linear` nodes of the i8 and i4 containers, 2026-11-14). Its
 * `dit_context` is the RF one's shape (72).
 */
const IRODORI_MEANFLOW_DIT_A8 = { variant: 248, reference: 0 } as const;
/** anima DiT: 454 linears (the 5 official variants and the 2 additional ones share the graph). */
const ANIMA_DIT_LINEARS = { variant: 454, reference: 0 } as const;
const ANIMA_MODELS = [
  "anima-turbo-v1.1",
  "anima-v1.0",
  "anima-aesthetic-v1.1",
  "anima-turbo-v1.0",
  "anima-aesthetic-v1.0",
  "anima-wai-v1.0",
  "anima-copycat-20260610",
] as const;
const SBV2_MODELS = ["F1", "F2", "M1", "M2"] as const;
/**
 * The Wan2.1 T2V 1.3B DiT: 307 linears (k ∈ {64, 256, 1536, 4096, 8960} — all i8 × k % 4 == 0,
 * including the M = 1 time MLP and `time_proj` — ADR 0120 decision 2, research §2.2).
 */
const WAN_DIT_LINEARS = { variant: 307, reference: 0 } as const;
/**
 * The Wan2.2 TI2V 5B DiT: 310 linear dispatches (307 i8 weights, the same layout as 2.1, but the
 * time MLP (2 linears and `time_proj`) runs twice at M = 1, which adds 3 nodes on the same weights
 * — ADR 0121 decision 2; 310 in the IR check of step 1. k ∈ {192, 256, 3072, 4096, 14336} — all k %
 * 4 == 0).
 */
const WAN22_DIT_LINEARS = { variant: 310, reference: 0 } as const;

/**
 * MOSS-Image's DiT pair: every linear of the graph takes i8a8 under `linearCompute: "a8"` — `dit`
 * has 258 (42 blocks × 6, `video_embedder`, the time MLP's two, the two modulations and
 * `final_linear_video` — the recipe's census of the graph) and `dit_skip22` 252 (41 × 6 + 6). The
 * exact i8a8 counts are what `packages/e2e/tests/e2e_moss_ab_test.ts` measures (`LINEAR_A8`,
 * 2026-10-28 on the RTX 3080 Ti), so none stays on the reference route.
 */
const MOSS_DIT_LINEARS = { variant: 258, reference: 0 } as const;
const MOSS_DIT_SKIP22_LINEARS = { variant: 252, reference: 0 } as const;

/**
 * The per-bundle census table.
 *
 * MUST: every quant preset of a distribution mirror with at least one number-changing non-reference
 * value needs a row here (`census_table_test.ts` fails without a GPU — it keeps bundles from
 * growing without a census). A row holds only what could be derived (phases not derived are not
 * written — see the file header).
 */
export const CENSUS_TABLE: readonly BundleCensusRow[] = [
  // --- gemma4 E2B (the q4_0 mirror, with its QAT drafter) -------------------
  {
    // The `i4-gemvpar` override preset of e2e_gemma4_quant_test / `i4-gemvpar` with ③ set back by
    // an explicit sequential.
    family: "gemma4",
    models: E2B,
    session: { linearGemvReduce: "parallel" },
    census: {
      target: {
        decode: { linearGemvReduce: GEMMA4_E2B_GEMV },
        verify: { linearGemvReduce: GEMMA4_E2B_GEMV },
      },
      drafter: { draft: { linearGemvReduce: GEMMA4_E2B_DRAFTER_GEMV } },
    },
  },
  {
    // The `i4-fast` override preset of e2e_gemma4_quant_test (the speculation check sets ③ to an
    // explicit sequential).
    family: "gemma4",
    models: E2B,
    session: { linearGemvReduce: "parallel", fuseRmsNormAdd: true },
    census: {
      target: {
        prefill: { fuseRmsNormAdd: GEMMA4_E2B_RMS_ADD },
        decode: { linearGemvReduce: GEMMA4_E2B_GEMV, fuseRmsNormAdd: GEMMA4_E2B_RMS_ADD },
        verify: { linearGemvReduce: GEMMA4_E2B_GEMV, fuseRmsNormAdd: GEMMA4_E2B_RMS_ADD },
      },
      drafter: {
        draft: {
          linearGemvReduce: GEMMA4_E2B_DRAFTER_GEMV,
          fuseRmsNormAdd: GEMMA4_E2B_DRAFTER_RMS_ADD,
        },
      },
    },
  },
  {
    // The distribution mirror's `i4-gemvpar`.
    family: "gemma4",
    models: E2B,
    session: { linearGemvReduce: "parallel", stateAttentionReduce: "parallel" },
    census: {
      target: {
        decode: {
          linearGemvReduce: GEMMA4_E2B_GEMV,
          stateAttentionReduce: GEMMA4_E2B_STATE_PARALLEL,
        },
        verify: {
          linearGemvReduce: GEMMA4_E2B_GEMV,
          stateAttentionReduce: GEMMA4_E2B_STATE_PARALLEL,
        },
      },
      drafter: {
        draft: {
          linearGemvReduce: GEMMA4_E2B_DRAFTER_GEMV,
          stateAttentionReduce: GEMMA4_E2B_DRAFTER_STATE_PARALLEL,
        },
      },
    },
  },
  {
    // The distribution mirror's `i4-fast` (③'s parallel-fused is not derived — file header).
    family: "gemma4",
    models: E2B,
    session: {
      linearGemvReduce: "parallel",
      fuseRmsNormAdd: true,
      stateAttentionReduce: "parallel-fused",
    },
    census: {
      target: {
        prefill: { fuseRmsNormAdd: GEMMA4_E2B_RMS_ADD },
        decode: { linearGemvReduce: GEMMA4_E2B_GEMV, fuseRmsNormAdd: GEMMA4_E2B_RMS_ADD },
        verify: { linearGemvReduce: GEMMA4_E2B_GEMV, fuseRmsNormAdd: GEMMA4_E2B_RMS_ADD },
      },
      drafter: {
        draft: {
          linearGemvReduce: GEMMA4_E2B_DRAFTER_GEMV,
          fuseRmsNormAdd: GEMMA4_E2B_DRAFTER_RMS_ADD,
        },
      },
    },
  },
  // --- gemma4 E4B (the q4_0 mirror, with its QAT drafter) -------------------
  {
    // The distribution mirror's `i4-gemvpar`.
    family: "gemma4",
    models: E4B,
    session: { linearGemvReduce: "parallel", stateAttentionReduce: "parallel" },
    census: {
      target: {
        decode: {
          linearGemvReduce: GEMMA4_E4B_GEMV,
          stateAttentionReduce: GEMMA4_E4B_STATE_PARALLEL,
        },
        verify: {
          linearGemvReduce: GEMMA4_E4B_GEMV,
          stateAttentionReduce: GEMMA4_E4B_STATE_PARALLEL,
        },
      },
      drafter: {
        draft: {
          linearGemvReduce: GEMMA4_E4B_DRAFTER_GEMV,
          stateAttentionReduce: GEMMA4_E4B_DRAFTER_STATE_PARALLEL,
        },
      },
    },
  },
  {
    // The distribution mirror's `i4-fast` (③'s parallel-fused is not derived — file header).
    family: "gemma4",
    models: E4B,
    session: {
      linearGemvReduce: "parallel",
      fuseRmsNormAdd: true,
      stateAttentionReduce: "parallel-fused",
    },
    census: {
      target: {
        prefill: { fuseRmsNormAdd: GEMMA4_E4B_RMS_ADD },
        decode: { linearGemvReduce: GEMMA4_E4B_GEMV, fuseRmsNormAdd: GEMMA4_E4B_RMS_ADD },
        verify: { linearGemvReduce: GEMMA4_E4B_GEMV, fuseRmsNormAdd: GEMMA4_E4B_RMS_ADD },
      },
      drafter: {
        draft: {
          linearGemvReduce: GEMMA4_E4B_DRAFTER_GEMV,
          fuseRmsNormAdd: GEMMA4_E4B_DRAFTER_RMS_ADD,
        },
      },
    },
  },
  // --- gemma4 12B (the q4_0 mirror; its drafter's rows are not derived) ------
  {
    // The distribution mirror's `i4-gemvpar`.
    family: "gemma4",
    models: TWELVE_B,
    session: { linearGemvReduce: "parallel", stateAttentionReduce: "parallel" },
    census: {
      target: {
        decode: {
          linearGemvReduce: GEMMA4_12B_GEMV,
          stateAttentionReduce: GEMMA4_12B_STATE_PARALLEL,
        },
      },
    },
  },
  {
    // The distribution mirror's `i4-fast` (③'s parallel-fused is not derived — file header).
    family: "gemma4",
    models: TWELVE_B,
    session: {
      linearGemvReduce: "parallel",
      fuseRmsNormAdd: true,
      stateAttentionReduce: "parallel-fused",
    },
    census: {
      target: {
        prefill: { fuseRmsNormAdd: GEMMA4_12B_RMS_ADD },
        decode: { linearGemvReduce: GEMMA4_12B_GEMV, fuseRmsNormAdd: GEMMA4_12B_RMS_ADD },
      },
    },
  },
  // --- gemma4 26B (the q4_0 mirror; it ships no drafter) ---------------------
  {
    // The distribution mirror's `i4-gemvpar`.
    family: "gemma4",
    models: TWENTY_SIX_B,
    session: { linearGemvReduce: "parallel", stateAttentionReduce: "parallel" },
    census: {
      target: {
        decode: {
          linearGemvReduce: GEMMA4_26B_GEMV,
          stateAttentionReduce: GEMMA4_26B_STATE_PARALLEL,
        },
      },
    },
  },
  {
    // The distribution mirror's `i4-fast` (③'s parallel-fused is not derived — file header).
    family: "gemma4",
    models: TWENTY_SIX_B,
    session: {
      linearGemvReduce: "parallel",
      fuseRmsNormAdd: true,
      stateAttentionReduce: "parallel-fused",
    },
    census: {
      target: {
        prefill: { fuseRmsNormAdd: GEMMA4_26B_RMS_ADD },
        decode: { linearGemvReduce: GEMMA4_26B_GEMV, fuseRmsNormAdd: GEMMA4_26B_RMS_ADD },
      },
    },
  },
  // --- gemma4-qat E2B (no drafter) -----------------------------------------
  {
    // The `i4-gemvpar` override preset of e2e_gemma4_quant_test.
    family: "gemma4-qat",
    models: E2B,
    session: { linearGemvReduce: "parallel" },
    census: {
      target: {
        decode: { linearGemvReduce: GEMMA4_QAT_E2B_GEMV },
        verify: { linearGemvReduce: GEMMA4_QAT_E2B_GEMV },
      },
    },
  },
  {
    // The distribution mirror's `i4-gemvpar`.
    family: "gemma4-qat",
    models: E2B,
    session: { linearGemvReduce: "parallel", stateAttentionReduce: "parallel" },
    census: {
      target: {
        decode: {
          linearGemvReduce: GEMMA4_QAT_E2B_GEMV,
          stateAttentionReduce: GEMMA4_E2B_STATE_PARALLEL,
        },
        verify: {
          linearGemvReduce: GEMMA4_QAT_E2B_GEMV,
          stateAttentionReduce: GEMMA4_E2B_STATE_PARALLEL,
        },
      },
    },
  },
  {
    // The distribution mirror's `i4-fast` (③'s parallel-fused is not derived — file header).
    family: "gemma4-qat",
    models: E2B,
    session: {
      linearGemvReduce: "parallel",
      fuseRmsNormAdd: true,
      fuseLinearStaticQuantize: true,
      fuseGeluMulStaticQuantize: true,
      packedStaticQuantize: true,
      stateAttentionReduce: "parallel-fused",
    },
    census: {
      target: {
        prefill: { fuseRmsNormAdd: GEMMA4_E2B_RMS_ADD },
        decode: {
          linearGemvReduce: GEMMA4_QAT_E2B_GEMV,
          fuseRmsNormAdd: GEMMA4_E2B_RMS_ADD,
          fuseLinearStaticQuantize: GEMMA4_QAT_E2B_LINEAR_SRQ,
          fuseGeluMulStaticQuantize: GEMMA4_QAT_E2B_GELU_MUL_SRQ,
          packedStaticQuantize: GEMMA4_QAT_E2B_PACKED,
        },
        verify: {
          linearGemvReduce: GEMMA4_QAT_E2B_GEMV,
          fuseRmsNormAdd: GEMMA4_E2B_RMS_ADD,
          fuseLinearStaticQuantize: GEMMA4_QAT_E2B_LINEAR_SRQ,
          fuseGeluMulStaticQuantize: GEMMA4_QAT_E2B_GELU_MUL_SRQ,
          packedStaticQuantize: GEMMA4_QAT_E2B_PACKED,
        },
      },
    },
  },
  {
    // The `i4-fast` override preset of e2e_gemma4_quant_test (default / explicit-fast).
    family: "gemma4-qat",
    models: E2B,
    session: {
      linearGemvReduce: "parallel",
      fuseRmsNormAdd: true,
      fuseLinearStaticQuantize: true,
      packedStaticQuantize: true,
    },
    census: {
      target: {
        prefill: { fuseRmsNormAdd: GEMMA4_E2B_RMS_ADD },
        decode: {
          linearGemvReduce: GEMMA4_QAT_E2B_GEMV,
          fuseRmsNormAdd: GEMMA4_E2B_RMS_ADD,
          fuseLinearStaticQuantize: GEMMA4_QAT_E2B_LINEAR_SRQ,
          packedStaticQuantize: GEMMA4_QAT_E2B_PACKED,
        },
      },
    },
  },
  {
    // disable-rms of e2e_gemma4_quant_test.
    family: "gemma4-qat",
    models: E2B,
    session: {
      linearGemvReduce: "parallel",
      fuseLinearStaticQuantize: true,
      packedStaticQuantize: true,
    },
    census: {
      target: {
        decode: {
          linearGemvReduce: GEMMA4_QAT_E2B_GEMV,
          fuseLinearStaticQuantize: GEMMA4_QAT_E2B_LINEAR_SRQ,
          packedStaticQuantize: GEMMA4_QAT_E2B_PACKED,
        },
      },
    },
  },
  {
    // disable-srq of e2e_gemma4_quant_test (the packed consumer is the packed variant of the plain
    // parallel GEMV).
    family: "gemma4-qat",
    models: E2B,
    session: { linearGemvReduce: "parallel", fuseRmsNormAdd: true, packedStaticQuantize: true },
    census: {
      target: {
        prefill: { fuseRmsNormAdd: GEMMA4_E2B_RMS_ADD },
        decode: {
          linearGemvReduce: GEMMA4_QAT_E2B_GEMV,
          fuseRmsNormAdd: GEMMA4_E2B_RMS_ADD,
          packedStaticQuantize: GEMMA4_QAT_E2B_PACKED,
        },
      },
    },
  },
  {
    // disable-both of e2e_gemma4_quant_test.
    family: "gemma4-qat",
    models: E2B,
    session: { linearGemvReduce: "parallel", packedStaticQuantize: true },
    census: {
      target: {
        decode: {
          linearGemvReduce: GEMMA4_QAT_E2B_GEMV,
          packedStaticQuantize: GEMMA4_QAT_E2B_PACKED,
        },
      },
    },
  },
  // --- gemma4-qat E4B (no drafter) -----------------------------------------
  {
    // The distribution mirror's `i4-gemvpar`.
    family: "gemma4-qat",
    models: E4B,
    session: { linearGemvReduce: "parallel", stateAttentionReduce: "parallel" },
    census: {
      target: {
        decode: {
          linearGemvReduce: GEMMA4_QAT_E4B_GEMV,
          stateAttentionReduce: GEMMA4_QAT_E4B_STATE_PARALLEL,
        },
        verify: {
          linearGemvReduce: GEMMA4_QAT_E4B_GEMV,
          stateAttentionReduce: GEMMA4_QAT_E4B_STATE_PARALLEL,
        },
      },
    },
  },
  {
    // The distribution mirror's `i4-fast` (③'s parallel-fused is not derived — file header).
    family: "gemma4-qat",
    models: E4B,
    session: {
      linearGemvReduce: "parallel",
      fuseRmsNormAdd: true,
      fuseLinearStaticQuantize: true,
      fuseGeluMulStaticQuantize: true,
      packedStaticQuantize: true,
      stateAttentionReduce: "parallel-fused",
    },
    census: {
      target: {
        prefill: { fuseRmsNormAdd: GEMMA4_QAT_E4B_RMS_ADD },
        decode: {
          linearGemvReduce: GEMMA4_QAT_E4B_GEMV,
          fuseRmsNormAdd: GEMMA4_QAT_E4B_RMS_ADD,
          fuseLinearStaticQuantize: GEMMA4_QAT_E4B_LINEAR_SRQ,
          fuseGeluMulStaticQuantize: GEMMA4_QAT_E4B_GELU_MUL_SRQ,
          packedStaticQuantize: GEMMA4_QAT_E4B_PACKED,
        },
        verify: {
          linearGemvReduce: GEMMA4_QAT_E4B_GEMV,
          fuseRmsNormAdd: GEMMA4_QAT_E4B_RMS_ADD,
          fuseLinearStaticQuantize: GEMMA4_QAT_E4B_LINEAR_SRQ,
          fuseGeluMulStaticQuantize: GEMMA4_QAT_E4B_GELU_MUL_SRQ,
          packedStaticQuantize: GEMMA4_QAT_E4B_PACKED,
        },
      },
    },
  },
  // --- irodori (the preset's session reaches only dit and dit_context) ------
  {
    family: "irodori",
    models: ["v4-small", "v4.1-small"],
    session: { linearCompute: "a8" },
    census: {
      dit: { step: { linearCompute: IRODORI_DIT_A8 } },
      "dit-context": { run: { linearCompute: IRODORI_DIT_CONTEXT_A8 } },
    },
  },
  {
    family: "irodori",
    models: ["v4.1-small-mf"],
    session: { linearCompute: "a8" },
    census: {
      dit: { step: { linearCompute: IRODORI_MEANFLOW_DIT_A8 } },
      "dit-context": { run: { linearCompute: IRODORI_DIT_CONTEXT_A8 } },
    },
  },
  // --- anima (the preset's session reaches only the DiT = transformer) ------
  {
    family: "anima",
    models: ANIMA_MODELS,
    session: { linearCompute: "a8" },
    census: { transformer: { step: { linearCompute: ANIMA_DIT_LINEARS } } },
  },
  {
    // attentionCompute is not derived (file header).
    family: "anima",
    models: ANIMA_MODELS,
    session: { linearCompute: "a8", attentionCompute: "a8" },
    census: { transformer: { step: { linearCompute: ANIMA_DIT_LINEARS } } },
  },
  {
    // attentionCompute / attentionScoreStorage are not derived (file header).
    family: "anima",
    models: ANIMA_MODELS,
    session: { linearCompute: "a8", attentionCompute: "a8", attentionScoreStorage: "f16" },
    census: { transformer: { step: { linearCompute: ANIMA_DIT_LINEARS } } },
  },
  {
    // Every f16-stored linear goes to the c16 GEMM skeleton (the M=1 GEMV computes in f32 only).
    // attentionCompute is not derived (file header).
    family: "anima",
    models: ANIMA_MODELS,
    session: { linearCompute: "f16", attentionCompute: "f16" },
    census: { transformer: { step: { linearCompute: ANIMA_DIT_LINEARS } } },
  },
  // --- sbv2 (the preset's session reaches only front / voice) ---------------
  {
    // front has 2 linears and voice 4 (the rest are conv1d, outside linearCompute's scope).
    family: "sbv2",
    models: SBV2_MODELS,
    session: { linearCompute: "a8" },
    census: {
      front: { run: { linearCompute: { variant: 2, reference: 0 } } },
      voice: { run: { linearCompute: { variant: 4, reference: 0 } } },
    },
  },
  // --- wan (the preset's session reaches only the DiT = transformer) --------
  {
    // The distribution mirror's practical preset `f16+dit8-a8-attn8-s16` (ADR 0120 decision 4).
    // attentionCompute / attentionScoreStorage are not derived (file header).
    family: "wan",
    models: ["t2v-1.3b"],
    session: { linearCompute: "a8", attentionCompute: "a8", attentionScoreStorage: "f16" },
    census: { transformer: { pass: { linearCompute: WAN_DIT_LINEARS } } },
  },
  {
    // The Wan2.2 distribution mirror's practical preset `f16+dit8-a8-attn8-s16` (ADR 0121 step 8a —
    // the same bundle as 2.1). attentionCompute / attentionScoreStorage are not derived, as in the
    // 2.1 row (file header).
    family: "wan-ti2v",
    models: ["ti2v-5b"],
    session: { linearCompute: "a8", attentionCompute: "a8", attentionScoreStorage: "f16" },
    census: { transformer: { pass: { linearCompute: WAN22_DIT_LINEARS } } },
  },
  {
    // Wan2.2's online-attention preset `f16+dit8-a8-attn8-online` (ADR 0132 decision 6): the same
    // linears as the s16 row (only the attention route differs). attentionCompute /
    // attentionScoreStorage stay underived as in the s16 row (top of the file); the e2e checks the
    // online route through `assertAttentionSeatsComplete` with the "none" seat.
    family: "wan-ti2v",
    models: ["ti2v-5b"],
    session: { linearCompute: "a8", attentionCompute: "a8", attentionScoreStorage: "none" },
    census: { transformer: { pass: { linearCompute: WAN22_DIT_LINEARS } } },
  },
  {
    // Wan2.2's conv8 preset `i8-a8-attn8-online-conv8` (ADR 0133 decision 6): the online bundle plus
    // the VAE decoder's conv knob. The transformer has no conv3d, so its passes carry exactly 0
    // conv3d w8a8 keys (the knob reaches the decoder Sessions — `e2e_wan_ti2v_vae_a8_test.ts` checks
    // them).
    family: "wan-ti2v",
    models: ["ti2v-5b"],
    session: {
      linearCompute: "a8",
      attentionCompute: "a8",
      attentionScoreStorage: "none",
      convCompute: "a8",
    },
    census: {
      transformer: { pass: { linearCompute: WAN22_DIT_LINEARS, convCompute: { variant: 0 } } },
    },
  },
  // --- moss-image (the preset's session reaches the DiT pair only — ADR 0138 decision 13) ------
  {
    // The distribution's practical preset `f32+dit8-a8-attn8-online` (ADR 0138 addendum of
    // 2026-11-01). attentionCompute / attentionScoreStorage stay underived as in Wan2.2's online row
    // (top of the file); the e2e A/B checks the online route through `assertAttentionSeatsComplete`
    // and its own exact online-dispatch count.
    family: "moss-image",
    models: ["sft"],
    session: { linearCompute: "a8", attentionCompute: "a8", attentionScoreStorage: "none" },
    census: {
      dit: { pass: { linearCompute: MOSS_DIT_LINEARS } },
      dit_skip22: { pass: { linearCompute: MOSS_DIT_SKIP22_LINEARS } },
    },
  },
];

// --------------------------------------------------------------------------- Lookup and checks
// ---------------------------------------------------------------------------

/**
 * The row of (family, model, effective SessionOptions). `undefined` when there is none (= the
 * census is at least one / zero).
 */
export const censusRowOf = (
  family: string,
  model: string,
  effective: SessionOptions,
): BundleCensusRow | undefined => {
  const key = bundleKey(effective);
  return CENSUS_TABLE.find(
    (row) => row.family === family && row.models.includes(model) && bundleKey(row.session) === key,
  );
};

/**
 * The row's phase table (to look up component and phase by string — the type is set by the row's
 * family).
 */
const phaseOf = (
  row: BundleCensusRow,
  component: string,
  phase: string,
): PhaseCensus | undefined => {
  const census: Readonly<Record<string, Readonly<Record<string, PhaseCensus>> | undefined>> =
    row.census;
  const phases = Object.hasOwn(census, component) ? census[component] : undefined;
  return phases !== undefined && Object.hasOwn(phases, phase) ? phases[phase] : undefined;
};

/**
 * The expectations of the row's (component, phase). `undefined` when the phase is not filled in.
 *
 * Per option it yields "exactly `variant` variant keys of the non-reference value" and, for options
 * with a reference-variant predicate, also "exactly `reference` keys left on that option's
 * reference path".
 */
export const phaseExpectations = (
  row: BundleCensusRow,
  component: string,
  phase: string,
): readonly CensusExpectation[] | undefined => {
  const census = phaseOf(row, component, phase);
  if (census === undefined) return undefined;
  return NUMERIC_SEATS.flatMap((seat): CensusExpectation[] => {
    const count: { readonly variant: number; readonly reference?: number } | undefined =
      census[seat];
    if (count === undefined) return [];
    const value = row.session[seat as keyof SessionOverrides];
    const signatures: Readonly<Record<string, KeyPredicate>> = SEAT_SIGNATURES[seat];
    const predicate = value === undefined ? undefined : signatures[String(value)];
    if (predicate === undefined) {
      throw new Error(
        `census table: ${row.family} [${bundleKey(row.session)}]: ${component}/${phase} holds a ` +
          `count for option ${seat}, which the bundle does not set (or sets to its reference value)`,
      );
    }
    const variant: CensusExpectation = {
      label: `variant of ${seat}=${String(value)}`,
      match: predicate,
      count: count.variant,
    };
    if (count.reference === undefined) return [variant];
    const reference = (REFERENCE_SIGNATURES as Readonly<Record<string, KeyPredicate>>)[seat];
    return [
      variant,
      {
        label: `dispatches left on the reference path of ${seat}`,
        match: reference,
        count: count.reference,
      },
    ];
  });
};

/**
 * Checks each phase's run censuses against the row's expectations **one run at a time** (the counts
 * are per run).
 *
 * Returns the number of runs checked. MUST: callers do not accept 0 — if a row exists but no phase
 * with expectations ever ran, not a single exact check was made.
 *
 * @param runs Phase → the `lastRunPipelines` of that phase's runs.
 * @param componentOf Phase → component (for gemma only draft is the drafter).
 */
export const assertRowCensus = (
  row: BundleCensusRow,
  runs: ReadonlyMap<string, readonly SessionDiagnostics["lastRunPipelines"][]>,
  componentOf: (phase: string) => string,
  where: string,
): number => {
  let checked = 0;
  for (const [phase, phaseRuns] of runs) {
    const component = componentOf(phase);
    const expected = phaseExpectations(row, component, phase);
    if (expected === undefined) continue;
    phaseRuns.forEach((census, index) => {
      assertPipelineCensus(census, expected, `${where} ${component}/${phase} #${index}`);
      checked++;
    });
  }
  return checked;
};

/**
 * gemma's phase → component (only speculation's draft is the drafter's Session; the rest is the
 * product graph).
 */
export const gemmaComponentOf = (phase: string): string =>
  phase === "draft" ? "drafter" : "target";
