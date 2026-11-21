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
 * A bundle is a `CensusSeats` (pipeline-census.ts): the approximation options and the forms the
 * levels of `numerics` resolve to, under the names the execution knobs carry (`linearGemvReduce`,
 * `stateAttentionReduce`, …). A row states its bundle directly; a run's bundle is `censusSeats` of
 * the **effective** SessionOptions (the result of passing the quant preset's `session` from the
 * manifest and the explicit overrides through the family's merge `resolveSessionOptions` — ADR
 * 0111). It is not looked up by the literal `session` of the manifest because a value added outside
 * the merge (as gemma's models default did before the reduction form was promoted) would make the
 * table key and the bundle that actually runs diverge. {@link bundleKey} drops the seats at their
 * reference value (an explicit `numerics: { ops: { linear: "exact" } }` is the same bundle as
 * leaving it unset).
 *
 * The built-in fusions (`FUSION_SIGNATURES` — RMS→add, linear→SRQ, gelu_tanh→mul→SRQ) apply in
 * every bundle wherever they match, so they are not seats: a phase writes their counts under the
 * rule name, and they hold in that bundle as in any other with the same linear form.
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
 * - `stateAttentionReduce: "parallel"` (the parallel reductions without the fold): only the private
 *   `REORDER_STRATEGIES` selects it, which no pipeline passes, so no bundle has it.
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
import { DEFAULT_GEOMETRY_PROFILE } from "../../../runtime/src/kernels/geometry-profile.ts";
import {
  defaultLinearGemvVariant,
  linearGemvKey,
  linearGemvParallelKey,
} from "../../../runtime/src/kernels/linear-gemv.ts";
import { statePvKey, stateQkKey } from "../../../runtime/src/kernels/state-attention.ts";
import {
  assertPipelineCensus,
  type CensusExpectation,
  type CensusSeats,
  censusSeats,
  FUSION_SIGNATURES,
  type KeyPredicate,
  type NumericSeat,
  SEAT_REFERENCE,
  SEAT_SIGNATURES,
} from "../../../runtime/tests/helpers/pipeline-census.ts";
import { ANIMA_SESSION_POLICY } from "../../src/anima/pipeline.ts";
import { BIREFNET_SESSION_POLICY } from "../../src/birefnet/pipeline.ts";
import { DEPTH_ANYTHING_SESSION_POLICY } from "../../src/depth-anything/pipeline.ts";
import { EMBEDDINGGEMMA_SESSION_POLICY } from "../../src/embeddinggemma/pipeline.ts";
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
  embeddinggemma: EMBEDDINGGEMMA_SESSION_POLICY,
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
 * A bundle → its key (only the seats that hold a **non-reference value**, as `seat=value` joined
 * with `,` in seat-name order). The empty string when every seat is at its reference value (= the
 * reference bundle). A run's bundle is `censusSeats(effective)`.
 *
 * NOTE: fields that do not change numbers (`submitPolicy` and so on — `NonNumericField` in
 * pipeline-census.ts) are not seats, so they are not in the key.
 */
export const bundleKey = (seats: CensusSeats): string =>
  NUMERIC_SEATS.flatMap((seat) => {
    const value = seats[seat];
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
  [
    linearKey("f32", true),
    linearGemvKey("i8", undefined, defaultLinearGemvVariant(DEFAULT_GEOMETRY_PROFILE.linearGemv)),
    linearGemvParallelKey("i8", undefined, 4),
  ].map(familyOf),
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
 * eligibility table moves it upwards — Fable review I8-02). The fusions and the boolean options have
 * none, because "nodes that were not fused" always remain by construction (a miss is visible from
 * the variant count alone).
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

/** A built-in fusion rule (`FUSION_SIGNATURES` — the fused kernel's key predicate). */
type FusionRule = keyof typeof FUSION_SIGNATURES;

/**
 * The expectations of one phase: the bundle's non-reference seats (the bundle's `seats` decide the
 * value) and the fused-kernel counts of the built-in fusions (by rule name — they apply in every
 * bundle).
 *
 * NOTE: a seat or fusion that is not written (not derived) has no count check in this phase. The
 * e2e tests keep checking a seat with `assertSeatsApplied`'s at least one / zero.
 */
export type PhaseCensus = { readonly [Seat in NumericSeat]?: SeatCount<Seat> } & {
  readonly [Rule in FusionRule]?: { readonly variant: number };
};

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

/** One table row (family × models × bundle, and the distribution where the family has two). */
type FamilyBundleRow<Family extends CensusFamily> = {
  readonly family: Family;
  /**
   * The distribution mirror the counts were derived from (its directory under `models/`), for a
   * family that two distributions share with the same model names: pipeline `gemma4` is both the
   * regular line (`karume-gemma4`, bf16) and the q4_0 QAT line (`karume-gemma4-qat`, int4), whose
   * presets resolve to the same bundles on different linears. Absent for a family with one
   * distribution. MUST: a family with one row that names it names it on every row — a row without
   * it would be read for either distribution ({@link censusRowOf} refuses a lookup without one).
   */
  readonly distribution?: string;
  /** The models with the same graph (the same counts) (keys of the manifest's models). */
  readonly models: readonly string[];
  /** The bundle (the seats of the effective SessionOptions — the key is {@link bundleKey}). */
  readonly seats: CensusSeats;
  readonly census: ComponentCensus<Family>;
};

export type BundleCensusRow = {
  readonly [Family in CensusFamily]: FamilyBundleRow<Family>;
}[CensusFamily];

// --------------------------------------------------------------------------- Table
// ---------------------------------------------------------------------------

/** The two distributions of pipeline `gemma4` (the rows' `distribution`). */
const GEMMA4_QAT_Q4_0_MIRROR = "karume-gemma4-qat";
const GEMMA4_REGULAR_MIRROR = "karume-gemma4";

/**
 * Pipeline `gemma4` rows of the regular line's mirror (`models/karume-gemma4`, bf16 — its preset
 * `bf16-fast` declares `level: "reorder"`). Derived from the mirror's E2B / E4B `model` and E2B
 * `drafter` part 0 with the procedure of the file header, on the parallel rows of the table the
 * development machine's RTX 3080 Ti selects (`nvidia-ampere` as derived 2026-11-21: the default
 * profile's 17 bf16 rows, its own bf16 `1536 × 2048` and `2560 × 256`, and int8 rows on five shapes
 * of the drafter), where the real-GPU runs compare them with the dispatched kernels
 * (`e2e_gemma4_bf16_parallel_gemv_plan_test.ts` compares the plan with these rows, and the default
 * profile's plan with that table's).
 *
 * E2B (product graph): of the 277 bf16 linears, 235 take a parallel GEMV row at M ≤ 8 — the 207 the
 * default profile's rows reach and the 28 `1536 × 2048` of `nvidia-ampere`'s own row; the other 42
 * (`12288 × 1536` 40, the output head 262144×1536 and `8960 × 1536`) have no row and keep the
 * sequential GEMV at M = 1 and the GEMM above it. Under the default profile `1536 × 2048` has no row
 * either (207 / 70).
 */
const GEMMA4_BF16_E2B_GEMV = { variant: 235, reference: 42 } as const;
/**
 * The regular E2B int8 drafter: of its 68 linears, 65 take a parallel GEMV row — the two
 * `post_projection` 1536×256 of the default profile and the 63 that `nvidia-ampere`'s own rows reach
 * (`2048 × 256` 27, `256 × 2048` 15, `1024 × 256` 9, `256 × 1024` 9 and `256 × 3072` 3); the
 * other 3 (the output head 262144×256) have none. Under the default profile only the two
 * `post_projection` have a row (2 / 66).
 */
const GEMMA4_BF16_E2B_DRAFTER_GEMV = { variant: 65, reference: 3 } as const;
/**
 * E4B (product graph): of the 344 bf16 linears, 258 take a parallel GEMV row at M ≤ 8 — the 216
 * the default profile's rows reach and the 42 `2560 × 256` of `nvidia-ampere`'s own row; the other
 * 86 (`10240 × 2560` 84, the output head 262144×2560 and `10752 × 2560`) have none. Under the
 * default profile `2560 × 256` has no row either (216 / 128).
 */
const GEMMA4_BF16_E4B_GEMV = { variant: 258, reference: 86 } as const;

/**
 * Pipeline `gemma4` rows: the q4_0 QAT line's mirror (`models/karume-gemma4-qat`) — the line that
 * keeps the int4 presets, where the regular line's storage is bf16. Re-derived 2026-11-04 from that
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
 * The E2B QAT drafter: of its 68 linears (all i4 g32), 56 take a parallel GEMV row on the table the
 * development machine's RTX 3080 Ti selects (`nvidia-ampere` as derived 2026-11-21) —
 * `post_projection` 1536×256 (twice, the model's per_layer_projection row), `pre_projection`
 * 256×3072 (3) and the sliding o 256×1024 (9) of the default profile, and 2048×256 (27) and 256×2048
 * (15) of that table's own rows. The other 12 (1024×256 9, lm_head 262144×256 3) fall to the
 * sequential GEMV. Under the default profile 256×2048 and 2048×256 have no row either (14 / 54: the
 * GEMV sweep's records of 2026-11-18 gave them none — docs/research/2026-11-18-gemma4-gemv-census.md
 * §4).
 */
const GEMMA4_E2B_DRAFTER_GEMV = { variant: 56, reference: 12 } as const;
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
 * 12B (product graph, 48 layers), read from a real run's `lastRunPipelines` at decode rather than
 * derived from the plan: all 329 linears, the int4 output head included, take the parallel GEMV.
 * `e2e_gemma4_12b_decode_ahead_test.ts` compares these with the run.
 */
const GEMMA4_12B_GEMV = { variant: 329, reference: 0 } as const;
const GEMMA4_12B_RMS_ADD = { variant: 96 } as const;

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
 * gemma4 E4B (product graph, 42 layers — the q4_0 mirror, derived with the E2B rows above): of the
 * 344 linears, 343 (i4 g32) take a parallel GEMV row at M ≤ 8 and the i4 lm_head 262144×2560 falls
 * to the sequential GEMV; RMS→add folds 127 times (decode M 1 / 4 / 8 and prefill M 32 / 768 alike)
 * and states-form attention runs ①' / ③' once each on the 42 layers.
 */
const GEMMA4_E4B_GEMV = { variant: 343, reference: 1 } as const;
const GEMMA4_E4B_RMS_ADD = { variant: 127 } as const;
/**
 * The E4B QAT drafter (the regular E4B shipped none, so these columns are new): of its 68 linears
 * (all i4 g32), 56 take a parallel GEMV row on `nvidia-ampere` (as the E2B drafter's) —
 * `post_projection` 2560×256 (twice) and the sliding o 256×1024 (9) of the default profile, and
 * 2048×256 (27), 256×2048 (15) and 256×5120 (3) of that table's own rows. The other 12 (1024×256 9,
 * lm_head 262144×256 3) fall to the sequential GEMV; under the default profile 11 / 57. RMS→add folds
 * 24 times and its 12 readonly attention nodes run ①' / ③' once each, as E2B's drafter.
 */
const GEMMA4_E4B_DRAFTER_GEMV = { variant: 56, reference: 12 } as const;
const GEMMA4_E4B_DRAFTER_RMS_ADD = { variant: 24 } as const;

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
/**
 * The irodori v4-large DiT (ADR 0151 — 24 blocks of width 2048, where v4 has 12): 485 linears, all
 * i8 × k % 4 == 0, counted from the graph of `dit` in the recipe's i8 series
 * (`outputs/series/irodori-v4-large-i8/`, whose bytes the distribution ships — `linear` nodes and
 * their weight residency, 2026-11-14; the same count gives v4-small's 245 / 72 from its own i8
 * series). Checked against a real dispatch on 2026-11-14 by the key census of
 * `e2e_irodori_ab_test.ts` on the mirror `models/karume-irodori-v4-large/` (`i8-a8` — `dit` 485 /
 * `dit_context` 144 i8a8 GEMMs per run, no f32 linear): the measurement equals the count.
 */
const IRODORI_V4_LARGE_DIT_A8 = { variant: 485, reference: 0 } as const;
/** The irodori v4-large `dit_context`: 144 linears (24 blocks × text / speaker / caption × K / V). */
const IRODORI_V4_LARGE_DIT_CONTEXT_A8 = { variant: 144, reference: 0 } as const;
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
  // --- gemma4 E2B / E4B (the regular bf16 mirror; E2B with its int8 drafter) -
  {
    // The distribution mirror's `bf16-fast` (it declares `level: "reorder"`; ③'s parallel-fused is
    // not derived — file header). RMS→add folds as many pairs as on the q4_0 graphs.
    family: "gemma4",
    distribution: GEMMA4_REGULAR_MIRROR,
    models: E2B,
    seats: { linearGemvReduce: "parallel", stateAttentionReduce: "parallel-fused" },
    census: {
      target: {
        prefill: { rmsNormAdd: GEMMA4_E2B_RMS_ADD },
        decode: { linearGemvReduce: GEMMA4_BF16_E2B_GEMV, rmsNormAdd: GEMMA4_E2B_RMS_ADD },
        verify: { linearGemvReduce: GEMMA4_BF16_E2B_GEMV, rmsNormAdd: GEMMA4_E2B_RMS_ADD },
      },
      drafter: {
        draft: {
          linearGemvReduce: GEMMA4_BF16_E2B_DRAFTER_GEMV,
          rmsNormAdd: GEMMA4_E2B_DRAFTER_RMS_ADD,
        },
      },
    },
  },
  {
    // The distribution mirror's `bf16-fast` (as E2B's; E4B ships no drafter).
    family: "gemma4",
    distribution: GEMMA4_REGULAR_MIRROR,
    models: E4B,
    seats: { linearGemvReduce: "parallel", stateAttentionReduce: "parallel-fused" },
    census: {
      target: {
        prefill: { rmsNormAdd: GEMMA4_E4B_RMS_ADD },
        decode: { linearGemvReduce: GEMMA4_BF16_E4B_GEMV, rmsNormAdd: GEMMA4_E4B_RMS_ADD },
        verify: { linearGemvReduce: GEMMA4_BF16_E4B_GEMV, rmsNormAdd: GEMMA4_E4B_RMS_ADD },
      },
    },
  },
  // --- gemma4 E2B (the q4_0 mirror, with its QAT drafter) -------------------
  {
    // Linear at `reorder`, attention at `exact`: the `i4-gemvpar` / `i4-fast` override presets of
    // e2e_gemma4_quant_test (its speculation check included) and the mirror's presets with attention
    // set back to `exact`.
    family: "gemma4",
    distribution: GEMMA4_QAT_Q4_0_MIRROR,
    models: E2B,
    seats: { linearGemvReduce: "parallel" },
    census: {
      target: {
        prefill: { rmsNormAdd: GEMMA4_E2B_RMS_ADD },
        decode: { linearGemvReduce: GEMMA4_E2B_GEMV, rmsNormAdd: GEMMA4_E2B_RMS_ADD },
        verify: { linearGemvReduce: GEMMA4_E2B_GEMV, rmsNormAdd: GEMMA4_E2B_RMS_ADD },
      },
      drafter: {
        draft: {
          linearGemvReduce: GEMMA4_E2B_DRAFTER_GEMV,
          rmsNormAdd: GEMMA4_E2B_DRAFTER_RMS_ADD,
        },
      },
    },
  },
  {
    // The distribution mirror's `i4-fast` (it declares `level: "reorder"`; ③'s parallel-fused is
    // not derived — file header).
    family: "gemma4",
    distribution: GEMMA4_QAT_Q4_0_MIRROR,
    models: E2B,
    seats: { linearGemvReduce: "parallel", stateAttentionReduce: "parallel-fused" },
    census: {
      target: {
        prefill: { rmsNormAdd: GEMMA4_E2B_RMS_ADD },
        decode: { linearGemvReduce: GEMMA4_E2B_GEMV, rmsNormAdd: GEMMA4_E2B_RMS_ADD },
        verify: { linearGemvReduce: GEMMA4_E2B_GEMV, rmsNormAdd: GEMMA4_E2B_RMS_ADD },
      },
      drafter: {
        draft: {
          linearGemvReduce: GEMMA4_E2B_DRAFTER_GEMV,
          rmsNormAdd: GEMMA4_E2B_DRAFTER_RMS_ADD,
        },
      },
    },
  },
  // --- gemma4 E4B (the q4_0 mirror, with its QAT drafter) -------------------
  {
    // The distribution mirror's `i4-fast` (③'s parallel-fused is not derived — file header).
    family: "gemma4",
    distribution: GEMMA4_QAT_Q4_0_MIRROR,
    models: E4B,
    seats: { linearGemvReduce: "parallel", stateAttentionReduce: "parallel-fused" },
    census: {
      target: {
        prefill: { rmsNormAdd: GEMMA4_E4B_RMS_ADD },
        decode: { linearGemvReduce: GEMMA4_E4B_GEMV, rmsNormAdd: GEMMA4_E4B_RMS_ADD },
        verify: { linearGemvReduce: GEMMA4_E4B_GEMV, rmsNormAdd: GEMMA4_E4B_RMS_ADD },
      },
      drafter: {
        draft: {
          linearGemvReduce: GEMMA4_E4B_DRAFTER_GEMV,
          rmsNormAdd: GEMMA4_E4B_DRAFTER_RMS_ADD,
        },
      },
    },
  },
  // --- gemma4 12B (the q4_0 mirror; its drafter's rows are not derived) ------
  {
    // The distribution mirror's `i4-fast` (③'s parallel-fused is not derived — file header).
    family: "gemma4",
    distribution: GEMMA4_QAT_Q4_0_MIRROR,
    models: TWELVE_B,
    seats: { linearGemvReduce: "parallel", stateAttentionReduce: "parallel-fused" },
    census: {
      target: {
        prefill: { rmsNormAdd: GEMMA4_12B_RMS_ADD },
        decode: { linearGemvReduce: GEMMA4_12B_GEMV, rmsNormAdd: GEMMA4_12B_RMS_ADD },
      },
    },
  },
  // --- gemma4 26B (the q4_0 mirror; it ships no drafter) ---------------------
  {
    // The distribution mirror's `i4-fast` (③'s parallel-fused is not derived — file header).
    family: "gemma4",
    distribution: GEMMA4_QAT_Q4_0_MIRROR,
    models: TWENTY_SIX_B,
    seats: { linearGemvReduce: "parallel", stateAttentionReduce: "parallel-fused" },
    census: {
      target: {
        prefill: { rmsNormAdd: GEMMA4_26B_RMS_ADD },
        decode: { linearGemvReduce: GEMMA4_26B_GEMV, rmsNormAdd: GEMMA4_26B_RMS_ADD },
      },
    },
  },
  // --- gemma4-qat E2B (no drafter) -----------------------------------------
  {
    // The `i4-gemvpar` override preset of e2e_gemma4_quant_test (linear at `reorder` only).
    family: "gemma4-qat",
    models: E2B,
    seats: { linearGemvReduce: "parallel" },
    census: {
      target: {
        prefill: { rmsNormAdd: GEMMA4_E2B_RMS_ADD },
        decode: {
          linearGemvReduce: GEMMA4_QAT_E2B_GEMV,
          rmsNormAdd: GEMMA4_E2B_RMS_ADD,
          linearStaticQuantize: GEMMA4_QAT_E2B_LINEAR_SRQ,
          geluMulStaticQuantize: GEMMA4_QAT_E2B_GELU_MUL_SRQ,
        },
        verify: {
          linearGemvReduce: GEMMA4_QAT_E2B_GEMV,
          rmsNormAdd: GEMMA4_E2B_RMS_ADD,
          linearStaticQuantize: GEMMA4_QAT_E2B_LINEAR_SRQ,
          geluMulStaticQuantize: GEMMA4_QAT_E2B_GELU_MUL_SRQ,
        },
      },
    },
  },
  {
    // The distribution mirror's `i4-gemvpar` (`level: "reorder"`; ③'s parallel-fused is not
    // derived — file header).
    family: "gemma4-qat",
    models: E2B,
    seats: { linearGemvReduce: "parallel", stateAttentionReduce: "parallel-fused" },
    census: {
      target: {
        prefill: { rmsNormAdd: GEMMA4_E2B_RMS_ADD },
        decode: {
          linearGemvReduce: GEMMA4_QAT_E2B_GEMV,
          rmsNormAdd: GEMMA4_E2B_RMS_ADD,
          linearStaticQuantize: GEMMA4_QAT_E2B_LINEAR_SRQ,
          geluMulStaticQuantize: GEMMA4_QAT_E2B_GELU_MUL_SRQ,
        },
        verify: {
          linearGemvReduce: GEMMA4_QAT_E2B_GEMV,
          rmsNormAdd: GEMMA4_E2B_RMS_ADD,
          linearStaticQuantize: GEMMA4_QAT_E2B_LINEAR_SRQ,
          geluMulStaticQuantize: GEMMA4_QAT_E2B_GELU_MUL_SRQ,
        },
      },
    },
  },
  {
    // The distribution mirror's `i4-fast` (`level: "reorder"` with the packed activations; ③'s
    // parallel-fused is not derived — file header).
    family: "gemma4-qat",
    models: E2B,
    seats: {
      linearGemvReduce: "parallel",
      packedStaticQuantize: true,
      stateAttentionReduce: "parallel-fused",
    },
    census: {
      target: {
        prefill: { rmsNormAdd: GEMMA4_E2B_RMS_ADD },
        decode: {
          linearGemvReduce: GEMMA4_QAT_E2B_GEMV,
          rmsNormAdd: GEMMA4_E2B_RMS_ADD,
          linearStaticQuantize: GEMMA4_QAT_E2B_LINEAR_SRQ,
          geluMulStaticQuantize: GEMMA4_QAT_E2B_GELU_MUL_SRQ,
          packedStaticQuantize: GEMMA4_QAT_E2B_PACKED,
        },
        verify: {
          linearGemvReduce: GEMMA4_QAT_E2B_GEMV,
          rmsNormAdd: GEMMA4_E2B_RMS_ADD,
          linearStaticQuantize: GEMMA4_QAT_E2B_LINEAR_SRQ,
          geluMulStaticQuantize: GEMMA4_QAT_E2B_GELU_MUL_SRQ,
          packedStaticQuantize: GEMMA4_QAT_E2B_PACKED,
        },
      },
    },
  },
  {
    // The `i4-fast` override preset of e2e_gemma4_quant_test (linear at `reorder` with the packed
    // activations — its default / explicit-fast / override-attention modes).
    family: "gemma4-qat",
    models: E2B,
    seats: { linearGemvReduce: "parallel", packedStaticQuantize: true },
    census: {
      target: {
        prefill: { rmsNormAdd: GEMMA4_E2B_RMS_ADD },
        decode: {
          linearGemvReduce: GEMMA4_QAT_E2B_GEMV,
          rmsNormAdd: GEMMA4_E2B_RMS_ADD,
          linearStaticQuantize: GEMMA4_QAT_E2B_LINEAR_SRQ,
          geluMulStaticQuantize: GEMMA4_QAT_E2B_GELU_MUL_SRQ,
          packedStaticQuantize: GEMMA4_QAT_E2B_PACKED,
        },
      },
    },
  },
  // --- gemma4-qat E4B (no drafter) -----------------------------------------
  {
    // The distribution mirror's `i4-gemvpar` (`level: "reorder"`; ③'s parallel-fused is not
    // derived — file header).
    family: "gemma4-qat",
    models: E4B,
    seats: { linearGemvReduce: "parallel", stateAttentionReduce: "parallel-fused" },
    census: {
      target: {
        prefill: { rmsNormAdd: GEMMA4_QAT_E4B_RMS_ADD },
        decode: {
          linearGemvReduce: GEMMA4_QAT_E4B_GEMV,
          rmsNormAdd: GEMMA4_QAT_E4B_RMS_ADD,
          linearStaticQuantize: GEMMA4_QAT_E4B_LINEAR_SRQ,
          geluMulStaticQuantize: GEMMA4_QAT_E4B_GELU_MUL_SRQ,
        },
        verify: {
          linearGemvReduce: GEMMA4_QAT_E4B_GEMV,
          rmsNormAdd: GEMMA4_QAT_E4B_RMS_ADD,
          linearStaticQuantize: GEMMA4_QAT_E4B_LINEAR_SRQ,
          geluMulStaticQuantize: GEMMA4_QAT_E4B_GELU_MUL_SRQ,
        },
      },
    },
  },
  {
    // The distribution mirror's `i4-fast` (`level: "reorder"` with the packed activations; ③'s
    // parallel-fused is not derived — file header).
    family: "gemma4-qat",
    models: E4B,
    seats: {
      linearGemvReduce: "parallel",
      packedStaticQuantize: true,
      stateAttentionReduce: "parallel-fused",
    },
    census: {
      target: {
        prefill: { rmsNormAdd: GEMMA4_QAT_E4B_RMS_ADD },
        decode: {
          linearGemvReduce: GEMMA4_QAT_E4B_GEMV,
          rmsNormAdd: GEMMA4_QAT_E4B_RMS_ADD,
          linearStaticQuantize: GEMMA4_QAT_E4B_LINEAR_SRQ,
          geluMulStaticQuantize: GEMMA4_QAT_E4B_GELU_MUL_SRQ,
          packedStaticQuantize: GEMMA4_QAT_E4B_PACKED,
        },
        verify: {
          linearGemvReduce: GEMMA4_QAT_E4B_GEMV,
          rmsNormAdd: GEMMA4_QAT_E4B_RMS_ADD,
          linearStaticQuantize: GEMMA4_QAT_E4B_LINEAR_SRQ,
          geluMulStaticQuantize: GEMMA4_QAT_E4B_GELU_MUL_SRQ,
          packedStaticQuantize: GEMMA4_QAT_E4B_PACKED,
        },
      },
    },
  },
  // --- irodori (the preset's session reaches only dit and dit_context) ------
  {
    family: "irodori",
    models: ["v4-small", "v4.1-small"],
    seats: { linearCompute: "a8" },
    census: {
      dit: { step: { linearCompute: IRODORI_DIT_A8 } },
      "dit-context": { run: { linearCompute: IRODORI_DIT_CONTEXT_A8 } },
    },
  },
  {
    family: "irodori",
    models: ["v4.1-small-mf"],
    seats: { linearCompute: "a8" },
    census: {
      dit: { step: { linearCompute: IRODORI_MEANFLOW_DIT_A8 } },
      "dit-context": { run: { linearCompute: IRODORI_DIT_CONTEXT_A8 } },
    },
  },
  {
    family: "irodori",
    models: ["v4-large"],
    seats: { linearCompute: "a8" },
    census: {
      dit: { step: { linearCompute: IRODORI_V4_LARGE_DIT_A8 } },
      "dit-context": { run: { linearCompute: IRODORI_V4_LARGE_DIT_CONTEXT_A8 } },
    },
  },
  // --- anima (the preset's session reaches only the DiT = transformer) ------
  {
    family: "anima",
    models: ANIMA_MODELS,
    seats: { linearCompute: "a8" },
    census: { transformer: { step: { linearCompute: ANIMA_DIT_LINEARS } } },
  },
  {
    // attentionCompute is not derived (file header).
    family: "anima",
    models: ANIMA_MODELS,
    seats: { linearCompute: "a8", attentionCompute: "a8" },
    census: { transformer: { step: { linearCompute: ANIMA_DIT_LINEARS } } },
  },
  {
    // attentionCompute / attentionScoreStorage are not derived (file header).
    family: "anima",
    models: ANIMA_MODELS,
    seats: { linearCompute: "a8", attentionCompute: "a8", attentionScoreStorage: "f16" },
    census: { transformer: { step: { linearCompute: ANIMA_DIT_LINEARS } } },
  },
  {
    // The seats of `bf16-c16`: every bf16-stored linear goes to the c16 GEMM skeleton, which
    // decodes bf16 into its f16 tiles (the M=1 GEMV computes in f32 only).
    // attentionCompute is not derived (file header).
    family: "anima",
    models: ANIMA_MODELS,
    seats: { linearCompute: "f16", attentionCompute: "f16" },
    census: { transformer: { step: { linearCompute: ANIMA_DIT_LINEARS } } },
  },
  // --- sbv2 (the preset's session reaches only front / voice) ---------------
  {
    // front has 2 linears and voice 4 (the rest are conv1d, outside linearCompute's scope).
    family: "sbv2",
    models: SBV2_MODELS,
    seats: { linearCompute: "a8" },
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
    seats: { linearCompute: "a8", attentionCompute: "a8", attentionScoreStorage: "f16" },
    census: { transformer: { pass: { linearCompute: WAN_DIT_LINEARS } } },
  },
  {
    // The Wan2.2 distribution mirror's practical preset `f16+dit8-a8-attn8-s16` (ADR 0121 step 8a —
    // the same bundle as 2.1). attentionCompute / attentionScoreStorage are not derived, as in the
    // 2.1 row (file header).
    family: "wan-ti2v",
    models: ["ti2v-5b"],
    seats: { linearCompute: "a8", attentionCompute: "a8", attentionScoreStorage: "f16" },
    census: { transformer: { pass: { linearCompute: WAN22_DIT_LINEARS } } },
  },
  {
    // Wan2.2's online-attention preset `f16+dit8-a8-attn8-online` (ADR 0132 decision 6): the same
    // linears as the s16 row (only the attention route differs). attentionCompute /
    // attentionScoreStorage stay underived as in the s16 row (top of the file); the e2e checks the
    // online route through `assertAttentionSeatsComplete` with the "none" seat.
    family: "wan-ti2v",
    models: ["ti2v-5b"],
    seats: { linearCompute: "a8", attentionCompute: "a8", attentionScoreStorage: "none" },
    census: { transformer: { pass: { linearCompute: WAN22_DIT_LINEARS } } },
  },
  {
    // Wan2.2's conv8 preset `i8-a8-attn8-online-conv8` (ADR 0133 decision 6): the online bundle plus
    // the VAE decoder's conv knob. The transformer has no conv3d, so its passes carry exactly 0
    // conv3d w8a8 keys (the knob reaches the decoder Sessions — `e2e_wan_ti2v_vae_a8_test.ts` checks
    // them).
    family: "wan-ti2v",
    models: ["ti2v-5b"],
    seats: {
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
    seats: { linearCompute: "a8", attentionCompute: "a8", attentionScoreStorage: "none" },
    census: {
      dit: { pass: { linearCompute: MOSS_DIT_LINEARS } },
      dit_skip22: { pass: { linearCompute: MOSS_DIT_SKIP22_LINEARS } },
    },
  },
];

// --------------------------------------------------------------------------- Lookup and checks
// ---------------------------------------------------------------------------

/**
 * The row of (family, model, effective SessionOptions) in `distribution` (the mirror's directory
 * under `models/`). `undefined` when there is none (= the census is at least one / zero).
 *
 * MUST: a family whose rows name their distribution (pipeline `gemma4`) is looked up with one, or
 * this throws — without it the lookup would pick one line's counts for the other's run.
 */
export const censusRowOf = (
  family: string,
  model: string,
  effective: SessionOptions,
  distribution?: string,
): BundleCensusRow | undefined => {
  if (
    distribution === undefined &&
    CENSUS_TABLE.some((row) => row.family === family && row.distribution !== undefined)
  ) {
    throw new Error(
      `census table: the rows of ${family} name their distribution — pass the mirror the run reads`,
    );
  }
  const key = bundleKey(censusSeats(effective));
  return CENSUS_TABLE.find(
    (row) =>
      row.family === family &&
      (row.distribution === undefined || row.distribution === distribution) &&
      row.models.includes(model) &&
      bundleKey(row.seats) === key,
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
 * Per seat it yields "exactly `variant` variant keys of the non-reference value" and, for seats
 * with a reference-variant predicate, also "exactly `reference` keys left on that seat's reference
 * path"; per fusion rule, "exactly `variant` fused-kernel keys".
 */
export const phaseExpectations = (
  row: BundleCensusRow,
  component: string,
  phase: string,
): readonly CensusExpectation[] | undefined => {
  const census = phaseOf(row, component, phase);
  if (census === undefined) return undefined;
  const fusions = (Object.keys(FUSION_SIGNATURES) as FusionRule[]).flatMap(
    (rule): CensusExpectation[] => {
      const count = census[rule];
      return count === undefined
        ? []
        : [
            {
              label: `fused kernel of ${rule}`,
              match: FUSION_SIGNATURES[rule],
              count: count.variant,
            },
          ];
    },
  );
  const seats = NUMERIC_SEATS.flatMap((seat): CensusExpectation[] => {
    const count: { readonly variant: number; readonly reference?: number } | undefined =
      census[seat];
    if (count === undefined) return [];
    const value = row.seats[seat];
    const signatures: Readonly<Record<string, KeyPredicate>> = SEAT_SIGNATURES[seat];
    const predicate = value === undefined ? undefined : signatures[String(value)];
    if (predicate === undefined) {
      throw new Error(
        `census table: ${row.family} [${bundleKey(row.seats)}]: ${component}/${phase} holds a ` +
          `count for seat ${seat}, which the bundle does not set (or sets to its reference value)`,
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
  return [...seats, ...fusions];
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
