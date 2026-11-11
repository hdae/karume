/**
 * Shared helpers of the census (the check of how many dispatches **actually** ran the variant when
 * an option is set — ADR 0058 decision 4 ③ / ADR 0110 decision 5).
 *
 * The source is `SessionDiagnostics.lastRunPipelines` (the **planned** dispatch count of the latest
 * run — filled independently of timing, on the resident path too). With the timing table
 * (`lastRunTiming`) as the source, the census would be skipped entirely on machines without
 * timestamp-query and on machines where timing drops out for the whole device (the Metal query set
 * limit), leaving not a single check that detects a silent fallback of an option there.
 * NOTE: the states form (dispatches whose workgroup count is computed from the logical length)
 * counts as one dispatch even on a step with no work (it is a census of the plan, not of what was
 * issued — `pipelineCensus` in src/runtime/recipe.ts).
 *
 * Predicates are spelled from the key functions / constants of src (keeping a second copy of the
 * strings here would let a key rename make only the predicates miss, turning the census
 * tautological). Distinguishing fragments (`:c16` / `:subgroup32` / `i8a8` and so on) are shared by
 * several families, so a predicate is always written **paired with the family (the part of the key
 * before the first `:`)**.
 */

import { assert } from "@std/assert";
import {
  ATTENTION_STATS_KEY,
  attentionPvKey,
  attentionQkKey,
} from "../../src/kernels/attention.ts";
import { attentionPvI8a8Key, attentionQkI8a8Key } from "../../src/kernels/attention-i8a8.ts";
import { attentionOnlineI8a8Key } from "../../src/kernels/attention-online-i8a8.ts";
import { conv3dI8a8Key } from "../../src/kernels/conv3d-i8a8.ts";
import { geluMulStaticQuantizeKey } from "../../src/kernels/gelu-mul-static-quantize.ts";
import { DEFAULT_GEOMETRY_PROFILE } from "../../src/kernels/geometry-profile.ts";
import { gemmComputeKeyPart } from "../../src/kernels/gemm.ts";
import { linearKey } from "../../src/kernels/linear.ts";
import {
  linearGemvParallelKey,
  linearGemvParallelPackedKey,
  linearGemvStaticQuantizeKey,
  linearGemvSubgroupKey,
} from "../../src/kernels/linear-gemv.ts";
import { integerGemvKey } from "../../src/kernels/linear-gemv-integer.ts";
import { planesGemvKey } from "../../src/kernels/linear-gemv-planes.ts";
import { linearI8a8Key } from "../../src/kernels/linear-i8a8.ts";
import { RMS_NORM_KEY, rmsNormAddKey } from "../../src/kernels/rms-norm.ts";
import { rmsNormSubgroupKey } from "../../src/kernels/rms-norm-subgroup.ts";
import { scoreKeyPart } from "../../src/kernels/score-storage.ts";
import {
  statePvParallelKey,
  statePvParallelReadonlyKey,
  stateQkFoldedKey,
  stateQkParallelKey,
  stateQkParallelReadonlyKey,
} from "../../src/kernels/state-attention.ts";
import { stateStatsPvKey } from "../../src/kernels/state-attention-stats-pv.ts";
import { stateStorageKeyPart } from "../../src/kernels/state-storage.ts";
import { STATIC_QUANTIZE_PACKED_KEY } from "../../src/kernels/static-quantize.ts";
import type { SessionDiagnostics, SessionOptions } from "../../src/runtime/session-types.ts";

/** A census row (an element of `lastRunPipelines`). */
export type CensusRow = NonNullable<SessionDiagnostics["lastRunPipelines"]>[number];

/** Predicate that recognizes a pipeline key. */
export type KeyPredicate = (key: string) => boolean;

// ---------------------------------------------------------------------------
// The table covering every option
// ---------------------------------------------------------------------------

/**
 * Fields of `SessionOptions` that do not change numerics (outside the census).
 *
 * MUST: an exclusion list (not an allow list). Adding a string-keyed field to `SessionOptions` puts
 * it into {@link NumericSeat} automatically, and the coverage of {@link SEAT_REFERENCE} /
 * {@link SEAT_SIGNATURES} fails the type check — whoever adds it must decide "is this an option
 * that changes numerics".
 * NOTE: the test-only knobs keyed by unique symbols (`I8A8_DOT` / `ROW_BLOCK_SPLIT`) drop out via
 * `Extract<…, string>`. Both select a variant whose contract is an integer match / Uint32 match
 * with the reference, and do not change numerics. `attentionRowBlockMaxBytes` is a resource policy
 * that changes only the number of row blocks (the output matches the no-budget run in Uint32 —
 * ADR 0124 decisions 3 / 4). `expertCache` is exact: a run with no miss matches full residency bit
 * for bit (ADR 0146 decision 1). `fusionRules` is not one seat with one reference value and one key
 * signature: each rule it carries declares its own contract class (ADR 0147 decision 2), and a class
 * E rule is u32-identical to the plan without it (ADR 0148).
 */
type NonNumericField =
  | "submitPolicy"
  | "sharedWeights"
  | "planBackingBudgetBytes"
  | "linearGemvRowsThreadTarget"
  | "attentionRowBlockMaxBytes"
  | "expertCache"
  | "fusionRules";

/**
 * Options of `SessionOptions` that change numerics (the string keys minus {@link NonNumericField}).
 */
export type NumericSeat = Exclude<Extract<keyof SessionOptions, string>, NonNumericField>;

type SeatValue<Seat extends NumericSeat> = NonNullable<SessionOptions[Seat]>;

/**
 * Reference value of each option (= the runtime's value when omitted — the same values as the
 * defaults assigned in src/runtime/session-build.ts). The type pins each to the option's union, so
 * a misspelling fails the type check.
 */
export const SEAT_REFERENCE = {
  linearCompute: "f32",
  convCompute: "f32",
  attentionCompute: "f32",
  attentionScoreStorage: "f32",
  stateAttentionReduce: "sequential",
  fullStateStorage: "f32",
  linearGemvReduce: "sequential",
  fuseRmsNormAdd: false,
  fuseLinearStaticQuantize: false,
  fuseGeluMulStaticQuantize: false,
  packedStaticQuantize: false,
  integerGemv: false,
  linearGemvPlanes: false,
  rmsNormReduce: "workgroup",
} as const satisfies { readonly [Seat in NumericSeat]-?: SeatValue<Seat> };

type NonReference<Seat extends NumericSeat> = Exclude<
  SeatValue<Seat>,
  (typeof SEAT_REFERENCE)[Seat]
>;

/** Option × non-reference value → predicate (boolean options use `"true"` as the key). */
export type SeatSignatures = {
  readonly [Seat in NumericSeat]: {
    readonly [Value in NonReference<Seat> as `${Value}`]: KeyPredicate;
  };
};

/**
 * Family of a key (before the first `:` — `linear` / `attention_qk` / `linear_gemv_parallel` …).
 */
const familyOf = (key: string): string => {
  const colon = key.indexOf(":");
  return colon < 0 ? key : key.slice(0, colon);
};

/**
 * dtype field of a key (the third of `family:generation:dtype:…` — `i8a8` for the i8a8 variants).
 */
const dtypeOf = (key: string): string | undefined => key.split(":")[2];

/**
 * Whether the key contains a `:`-delimited fragment (matched at delimiter boundaries, not as a
 * substring — so that `:s16` does not match `:s16x`).
 */
const hasFragment = (key: string, fragment: string): boolean => `${key}:`.includes(`${fragment}:`);

/**
 * The fragment a variant key appends to the base key (for families whose key functions in src build
 * "base + fragment").
 */
const suffixOf = (variant: string, base: string): string => {
  assert(
    variant.startsWith(base),
    `fragment derivation: '${variant}' does not start with '${base}'`,
  );
  return variant.slice(base.length);
};

/**
 * The key function evaluated for all 4 combinations of `(sliding, gqa)` (the states form has only
 * two variant bits).
 */
const stateVariants = (
  ...keyOf: readonly ((sliding: boolean, gqa: boolean) => string)[]
): ReadonlySet<string> =>
  new Set(
    keyOf.flatMap((key) =>
      [false, true].flatMap((sliding) => [false, true].map((gqa) => key(sliding, gqa))),
    ),
  );

// Representative arguments for deriving fragments and families (any arguments give the same
// fragment — the key functions build "base + fragment").
const GEMV_PARALLEL_SAMPLE = linearGemvParallelKey("i8", undefined, 4);

const LINEAR_FAMILY = familyOf(linearKey("f32", true));
// The conv3d w8a8 family (ADR 0133) — its keys carry the i8a8 dtype like linear's (any geometry
// gives the same family).
const CONV3D_FAMILY = familyOf(conv3dI8a8Key(true, true, DEFAULT_GEOMETRY_PROFILE.i8a8.conv3d));
const I8A8_DTYPE = dtypeOf(linearI8a8Key(true, true));
const C16 = gemmComputeKeyPart("f16");
const S16 = scoreKeyPart("f16");
const ATTENTION_FAMILIES: ReadonlySet<string> = new Set(
  [attentionQkKey(true), attentionPvKey(true), ATTENTION_STATS_KEY].map(familyOf),
);
const ATTENTION_I8A8_FAMILIES: ReadonlySet<string> = new Set(
  [attentionQkI8a8Key(true, true), attentionPvI8a8Key(true, true)].map(familyOf),
);
// The online kernel's family (ADR 0132 — `attentionScoreStorage: "none"`). Its keys carry the i8a8
// dtype too: the kernel is the a8 compute with S never stored.
const ATTENTION_ONLINE_FAMILY = familyOf(attentionOnlineI8a8Key(true, 128, "sg32"));
// The folded ①' (perf-ledger K-55) is ①' on selected full GQA classes (F = r = 8 today; F = 4 is
// still generated, for perf-ledger K-95) — the same evidence of "parallel", outside the
// (sliding, gqa) grid because its key carries F.
const STATE_PARALLEL_KEYS: ReadonlySet<string> = new Set([
  ...stateVariants(
    stateQkParallelKey,
    statePvParallelKey,
    stateQkParallelReadonlyKey,
    statePvParallelReadonlyKey,
  ),
  stateQkFoldedKey(4),
  stateQkFoldedKey(8),
  // The same forms on an f16 full slot: their keys only gain the storage part, so they still read as
  // "parallel ran". A sliding slot never has f16.
  ...[false, true].flatMap((gqa) => [
    stateQkParallelKey(false, gqa, "f16"),
    statePvParallelKey(false, gqa, "f16"),
    stateQkParallelReadonlyKey(false, gqa, "f16"),
    statePvParallelReadonlyKey(false, gqa, "f16"),
  ]),
  stateQkFoldedKey(4, "f16"),
  stateQkFoldedKey(8, "f16"),
]);
const STATE_STATS_PV_KEYS = stateVariants(stateStatsPvKey);
const GEMV_PARALLEL_FAMILY = familyOf(GEMV_PARALLEL_SAMPLE);
const GEMV_SUBGROUP = suffixOf(linearGemvSubgroupKey("i8", undefined, 4), GEMV_PARALLEL_SAMPLE);
const GEMV_STATIC_QUANTIZE = suffixOf(
  linearGemvStaticQuantizeKey("i8", undefined, 4),
  GEMV_PARALLEL_SAMPLE,
);
const GEMV_PACKED = suffixOf(linearGemvParallelPackedKey("i8", undefined, 4), GEMV_PARALLEL_SAMPLE);
const RMS_NORM_ADD_FAMILY = familyOf(rmsNormAddKey("norm-residual", 1536));
// The gelu_tanh → mul → SRQ kernel (perf-ledger K-48) — one family for both operand orders.
const GELU_MUL_STATIC_QUANTIZE_FAMILY = familyOf(geluMulStaticQuantizeKey("gelu-other"));
const RMS_FAMILIES: ReadonlySet<string> = new Set([familyOf(RMS_NORM_KEY), RMS_NORM_ADD_FAMILY]);
const RMS_SUBGROUP = suffixOf(rmsNormSubgroupKey(), RMS_NORM_KEY);
// The integer GEMV (`integerGemv`): one family for every variant, plain or with the SRQ epilogue.
const GEMV_INTEGER_FAMILY = familyOf(
  integerGemvKey({ weight: { storage: "i2" }, lanes: 8, rows: 4 }, false),
);
// The byte-plane GEMV (`linearGemvPlanes`): one family for every (lanes, rows) variant.
const GEMV_PLANES_FAMILY = familyOf(planesGemvKey({ lanes: 8, rows: 4 }));

const isGemvParallel = (key: string): boolean => familyOf(key) === GEMV_PARALLEL_FAMILY;

/**
 * Numerics-changing option × non-reference value → predicate on the key showing "that variant ran".
 *
 * MUST: the predicates of different non-reference values of one option are disjoint
 * ({@link assertSeatsApplied} counts per value). Overlap across options is by design (SRQ fusion
 * and packed sit on top of the parallel GEMV family; the subgroup32 RMS→add fusion key is evidence
 * for both `fuseRmsNormAdd` and `rmsNormReduce`).
 */
export const SEAT_SIGNATURES: SeatSignatures = {
  linearCompute: {
    a8: (key) => familyOf(key) === LINEAR_FAMILY && dtypeOf(key) === I8A8_DTYPE,
    f16: (key) => familyOf(key) === LINEAR_FAMILY && hasFragment(key, C16),
  },
  convCompute: {
    a8: (key) => familyOf(key) === CONV3D_FAMILY && dtypeOf(key) === I8A8_DTYPE,
  },
  attentionCompute: {
    f16: (key) => ATTENTION_FAMILIES.has(familyOf(key)) && hasFragment(key, C16),
    // The online kernel is evidence of a8 as well (a `"none"` run has no attention_qk / attention_pv).
    a8: (key) =>
      (ATTENTION_I8A8_FAMILIES.has(familyOf(key)) || familyOf(key) === ATTENTION_ONLINE_FAMILY) &&
      dtypeOf(key) === I8A8_DTYPE,
  },
  attentionScoreStorage: {
    f16: (key) => ATTENTION_FAMILIES.has(familyOf(key)) && hasFragment(key, S16),
    none: (key) => familyOf(key) === ATTENTION_ONLINE_FAMILY,
  },
  stateAttentionReduce: {
    parallel: (key) => STATE_PARALLEL_KEYS.has(key),
    "parallel-fused": (key) => STATE_STATS_PV_KEYS.has(key),
  },
  fullStateStorage: {
    f16: (key) => key.endsWith(stateStorageKeyPart("f16")),
  },
  linearGemvReduce: {
    parallel: (key) => isGemvParallel(key) && !hasFragment(key, GEMV_SUBGROUP),
    "parallel-subgroup32": (key) => isGemvParallel(key) && hasFragment(key, GEMV_SUBGROUP),
  },
  fuseRmsNormAdd: {
    true: (key) => familyOf(key) === RMS_NORM_ADD_FAMILY,
  },
  fuseLinearStaticQuantize: {
    true: (key) => isGemvParallel(key) && hasFragment(key, GEMV_STATIC_QUANTIZE),
  },
  fuseGeluMulStaticQuantize: {
    true: (key) => familyOf(key) === GELU_MUL_STATIC_QUANTIZE_FAMILY,
  },
  packedStaticQuantize: {
    // True on both the producer side (the packed variant of standalone SRQ) and the consumer side
    // (the packed variant of parallel GEMV).
    true: (key) =>
      key === STATIC_QUANTIZE_PACKED_KEY || (isGemvParallel(key) && hasFragment(key, GEMV_PACKED)),
  },
  integerGemv: {
    true: (key) => familyOf(key) === GEMV_INTEGER_FAMILY,
  },
  linearGemvPlanes: {
    true: (key) => familyOf(key) === GEMV_PLANES_FAMILY,
  },
  rmsNormReduce: {
    subgroup32: (key) => RMS_FAMILIES.has(familyOf(key)) && hasFragment(key, RMS_SUBGROUP),
  },
};

// ---------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------

/**
 * Takes the census rows. undefined / empty fails loudly.
 *
 * MUST: empty never passes as "nothing to look at" — a match of two empty tables or an assertion of
 * zero dispatches is tautological, and the census would turn green without checking anything.
 */
export const requireCensus = (
  rows: SessionDiagnostics["lastRunPipelines"],
  where: string,
): readonly CensusRow[] => {
  assert(
    rows !== undefined,
    `${where}: lastRunPipelines is not filled (no latest run / it failed)`,
  );
  assert(
    rows.length > 0,
    `${where}: the census is empty (would hide a miscount behind an assertion of zero)`,
  );
  return rows;
};

/** Sum of the dispatch counts of the rows matching the predicate (or the exact key). */
export const countDispatches = (
  rows: readonly CensusRow[],
  match: string | KeyPredicate,
): number => {
  const test: KeyPredicate = typeof match === "string" ? (key) => key === match : match;
  return rows.reduce((total, row) => total + (test(row.key) ? row.dispatchCount : 0), 0);
};

/**
 * Folds the census of several runs into one table (counts summed per key, keys in lexicographic
 * order).
 *
 * MUST: an undefined element fails loudly (no census right after a successful run means the source
 * is broken; skipping it would silently drop "that run's variants" from the totals).
 */
export const mergeCensus = (
  runs: readonly SessionDiagnostics["lastRunPipelines"][],
  where: string,
): readonly CensusRow[] => {
  const counts = new Map<string, number>();
  runs.forEach((rows, index) => {
    assert(rows !== undefined, `${where}: run ${index} has no lastRunPipelines`);
    for (const { key, dispatchCount } of rows) {
      counts.set(key, (counts.get(key) ?? 0) + dispatchCount);
    }
  });
  return [...counts]
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .map(([key, dispatchCount]) => ({ key, dispatchCount }));
};

/** One census expectation ("exactly this count" or "at least"). */
export type CensusExpectation = {
  /** Name shown in the failure message. */
  readonly label: string;
  /** Predicate (a string is an exact key match). */
  readonly match: string | KeyPredicate;
} & ({ readonly count: number } | { readonly atLeast: number });

/** Checks the count per expectation. undefined / empty rows fail loudly ({@link requireCensus}). */
export const assertPipelineCensus = (
  rows: SessionDiagnostics["lastRunPipelines"],
  expected: readonly CensusExpectation[],
  where: string,
): void => {
  const census = requireCensus(rows, where);
  for (const expectation of expected) {
    const actual = countDispatches(census, expectation.match);
    const keys = census.map((row) => `${row.key}×${row.dispatchCount}`).join(" / ");
    if ("count" in expectation) {
      assert(
        actual === expectation.count,
        `${where}: ${expectation.label} ran ${actual} times (expected ${expectation.count}) — ${keys}`,
      );
    } else {
      assert(
        actual >= expectation.atLeast,
        `${where}: ${expectation.label} ran ${actual} times (expected at least ${expectation.atLeast}) — ${keys}`,
      );
    }
  }
};

/**
 * Option names of the coverage table (`Object.keys` returns string[], so the names are re-typed
 * from the table's type).
 */
const NUMERIC_SEATS = Object.keys(SEAT_REFERENCE) as readonly NumericSeat[];

/**
 * Two-sided check that the options passed actually took effect.
 *
 * - An option with a non-reference value: at least one dispatch matches that value's predicate.
 * - An option **explicitly set** to its reference value: no dispatch matches any of its
 *   non-reference predicates.
 *
 * Options not passed (undefined) are not checked — the effective value when omitted is decided by
 * the caller (a preset declaration and the like), so it is not assumed to be the reference here.
 * NOTE: "at least one" is evidence that the option took effect **somewhere**, not evidence of full
 * eligibility (every target went to the variant). Per-target counts belong in
 * {@link assertPipelineCensus}.
 */
export const assertSeatsApplied = (
  rows: SessionDiagnostics["lastRunPipelines"],
  options: SessionOptions,
  where: string,
): void => {
  const census = requireCensus(rows, where);
  for (const seat of NUMERIC_SEATS) {
    const value = options[seat];
    if (value === undefined) continue;
    const variants: Readonly<Record<string, KeyPredicate>> = SEAT_SIGNATURES[seat];
    if (value === SEAT_REFERENCE[seat]) {
      for (const [variant, predicate] of Object.entries(variants)) {
        const hits = census.filter((row) => predicate(row.key)).map((row) => row.key);
        assert(
          hits.length === 0,
          `${where}: ${seat} is the reference value ${JSON.stringify(value)} yet the ${variant} variant ran — ${hits.join(
            " / ",
          )}`,
        );
      }
      continue;
    }
    const predicate = variants[String(value)];
    assert(
      predicate !== undefined,
      `${where}: ${seat}=${JSON.stringify(value)} has no predicate in the coverage table`,
    );
    assert(
      countDispatches(census, predicate) > 0,
      `${where}: no dispatch of the ${seat}=${JSON.stringify(value)} variant ran — ${census
        .map((row) => row.key)
        .join(" / ")}`,
    );
  }
};

/**
 * **Full-eligibility** check of the fused-attention options — the declared a8 / s16 took effect on
 * **every** fused-attention dispatch in the census (ADR 0058 addendum 2026-10-10).
 *
 * - `attentionCompute: "a8"`: in each of the ①QK and ③PV families, total count == i8a8 variant
 *   count.
 * - `attentionScoreStorage: "f16"`: in each of the 3 families ①QK / ②row statistics / ③PV, total
 *   count == count with `:s16`.
 *
 * It compares against the total (a ratio), so the number of row blocks (which varies with the
 * device's binding limits) multiplies both sides equally and cancels — compatible with census
 * tables that do not write the counts themselves (census-table.ts in models).
 * Not mixed into {@link assertSeatsApplied}: that one checks "took effect somewhere" and is also
 * called by runtime tests that hit fallback shapes on purpose. A fallback by the shape-eligibility
 * decision (src/runtime/attention-route.ts) is allowed by the spec (ADR 0058 decision 3), so this
 * is applied only to the census of **shapes that can be expected to take the declared path**.
 * MUST: the census passed covers only runs of the Session that declared a8 / s16 — mixing in the
 * f32 attention of another Session (VAE, text encoder) would make legitimate f32 look like a
 * fallback and turn red.
 * MUST: a census without any dispatch of a target family fails loudly (no green from a tautological
 * 0 == 0).
 * NOTE: an unfused decomposed attention (the bmm path) has no attention-family key, so it is out of
 * scope. `attentionCompute: "f16"` has no shape-based fallback, so it is not checked.
 */
export const assertAttentionSeatsComplete = (
  rows: SessionDiagnostics["lastRunPipelines"],
  options: SessionOptions,
  where: string,
): void => {
  const census = requireCensus(rows, where);
  const assertComplete = (
    seat: string,
    families: ReadonlySet<string>,
    variant: KeyPredicate,
    variantLabel: string,
  ): void => {
    for (const family of families) {
      const inFamily: KeyPredicate = (key) => familyOf(key) === family;
      const total = countDispatches(census, inFamily);
      const applied = countDispatches(census, (key) => inFamily(key) && variant(key));
      const keys = census
        .filter((row) => inFamily(row.key))
        .map((row) => `${row.key}×${row.dispatchCount}`)
        .join(" / ");
      assert(total > 0, `${where}: ${seat} was declared but there is no ${family} dispatch`);
      assert(
        applied === total,
        `${where}: ${seat}: of the ${family} dispatches, ${total} in all, only ${applied} are ${variantLabel}` +
          ` (the rest fell back to f32 — shape eligibility in attentionRoute) — ${keys}`,
      );
    }
  };
  // `"none"` (the online kernel — ADR 0132): complete when the online kernel ran and none of the three
  // dispatches did (every node's shape and the device admitted it). The online kernel is the a8
  // compute, so this also covers the a8 check below (which would look for attention_qk / attention_pv).
  if (options.attentionScoreStorage === "none") {
    const online = countDispatches(census, (key) => familyOf(key) === ATTENTION_ONLINE_FAMILY);
    const threePass = census
      .filter((row) => ATTENTION_FAMILIES.has(familyOf(row.key)))
      .map((row) => `${row.key}×${row.dispatchCount}`);
    assert(
      online > 0,
      `${where}: attentionScoreStorage="none" was declared but no ${ATTENTION_ONLINE_FAMILY} dispatch ran`,
    );
    assert(
      threePass.length === 0,
      `${where}: attentionScoreStorage="none" fell back to the three dispatches for some node` +
        ` (the shape or the device did not admit the online kernel — attentionRoute) — ${threePass.join(" / ")}`,
    );
    return;
  }
  if (options.attentionCompute === "a8") {
    assertComplete(
      'attentionCompute="a8"',
      ATTENTION_I8A8_FAMILIES,
      (key) => dtypeOf(key) === I8A8_DTYPE,
      "i8a8",
    );
  }
  if (options.attentionScoreStorage === "f16") {
    assertComplete(
      'attentionScoreStorage="f16"',
      ATTENTION_FAMILIES,
      (key) => hasFragment(key, S16),
      S16,
    );
  }
};
