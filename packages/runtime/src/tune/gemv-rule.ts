/**
 * The numbers of the GEMV derivation's rule of one steady record (the maintainer, 2026-11-28) and the
 * pure pieces of a record they are read with: a record's steadiness class, a row's gap and a
 * reading's break-even margin. The derivation (gemv-derive.ts), the sweep that states its record's
 * class (gemv-sweep.ts) and a reader of one record (the benchmark page) share them, so a record's
 * class and a gain's bar are computed in one place.
 *
 * The figures below come from the GEMV sweep records of 2026-11-18 to 27 of four devices — the RTX
 * 5070 Ti (Chrome, D3D12), the Apple M2 and the Apple M5 (Chrome, Metal; the M5 also without its
 * adapter description) and the RTX 3080 Ti (Node, Vulkan) — as docs/research/2026-11-28-gemv-adoption-rule.md
 * gives them.
 *
 * SPEC: docs/spec/geometry-profiles.md#deriving-the-gemv-choices
 */

/**
 * The margin of the rule of one record, by default: at M = 1 a candidate qualifies when its gain over
 * the reference in the same group is at least this × gap(candidate) × gap(reference).
 *
 * Source: the least margin that covers 99 % of the gains' differences between records of one device
 * is ×1.060, ×1.066 and ×1.058 on the three steady devices (M = 1, the candidates a table can hold;
 * 3,633, 4,414 and 8,338 rows); against a reference that is the default's choice, 0 of 1,481 adoptions
 * is slower than ×1/1.02 by the other records. It bounds the error between records of one session:
 * across days it is exceeded in 0.7 to 3.8 % of the rows, because single kernels move between
 * sessions by more than ×1.065 in 0.6 to 0.9 % of the cases with nothing to see inside a record.
 */
export const GEMV_MARGIN = 1.065;

/**
 * The smallest margin a derivation's output is meant to be re-decided at (×1.03, the alternative the
 * maintainer weighs): the sweep carries on to M = 4 and 8 every reorder candidate at least this much
 * faster than the default's choice at M = 1 in a steady group, so that a re-decision at it reads the
 * times it needs from the same record. A gap is at least 1, so a gain below it qualifies at no margin
 * of at least it.
 */
export const GEMV_LOWEST_MARGIN = 1.03;

/**
 * The fewest groups with a baseline spread for a record to have a class. Source: of the records with
 * 20 groups or more none is called wrongly (35 of three steady devices, 36 of the RTX 3080 Ti); below
 * that six RTX 3080 Ti records of 12 or 13 groups read as steady and four steady devices' records of
 * 2 to 4 groups read as unsteady.
 */
export const GEMV_STEADY_MIN_GROUPS = 20;

/**
 * The largest share of a record's groups whose baseline spread exceeds
 * {@link GEMV_STEADY_GROUP_SPREAD} for the record to be steady. Source: the 35 steady records read 0
 * to 15.2 %, the 36 of the RTX 3080 Ti 31.8 to 84.6 %; no record lies between, so 20 % is a point
 * inside an empty band.
 */
export const GEMV_STEADY_MAX_SHARE = 0.2;

/** The baseline spread (its largest round ÷ its smallest) above which a group counts as unsteady. */
export const GEMV_STEADY_GROUP_SPREAD = 1.05;

/**
 * The shortest pass (ms) of a kernel whose group is evidence under the rule of one record. Source:
 * the data hold 3 such rows on the RTX 5070 Ti and 5 on the RTX 3080 Ti, 3 of them wrong; between
 * 20 ms and the 80 ms target the gap term widens the bar by itself (the Apple M5's 40 to 60 ms rows
 * miss the bound in 0.44 % against 0.40 % from 60 ms up).
 */
export const GEMV_MIN_PASS_MS = 20;

/**
 * A record's class: `steady` (one record adopts by the rule of one record), `unsteady`, or
 * `unclassified` — fewer than {@link GEMV_STEADY_MIN_GROUPS} groups with a baseline spread, read as
 * unsteady.
 */
export type GemvSteadinessClass = "steady" | "unsteady" | "unclassified";

/** A record's class with the counts it was read from. */
export type GemvSteadiness = {
  readonly class: GemvSteadinessClass;
  /** The groups (summaries) with a baseline spread, at every row count. */
  readonly groups: number;
  /** Of those, the groups whose spread exceeds {@link GEMV_STEADY_GROUP_SPREAD}. */
  readonly unsteadyGroups: number;
  /** `unsteadyGroups ÷ groups` (absent with no such group). */
  readonly share?: number;
};

/**
 * The class of a record from its summaries: steady when at least {@link GEMV_STEADY_MIN_GROUPS} of
 * its groups have a baseline spread and at most {@link GEMV_STEADY_MAX_SHARE} of those spread beyond
 * {@link GEMV_STEADY_GROUP_SPREAD}. Every group counts, whatever its row count or case. Pure.
 */
export const gemvSteadiness = (
  summaries: readonly { readonly baselineSpread?: number }[],
): GemvSteadiness => {
  const spreads = summaries.flatMap(({ baselineSpread }) =>
    baselineSpread === undefined ? [] : [baselineSpread],
  );
  const groups = spreads.length;
  const unsteadyGroups = spreads.filter((spread) => spread > GEMV_STEADY_GROUP_SPREAD).length;
  const share = groups === 0 ? undefined : unsteadyGroups / groups;
  const steadinessClass: GemvSteadinessClass =
    groups < GEMV_STEADY_MIN_GROUPS || share === undefined
      ? "unclassified"
      : share <= GEMV_STEADY_MAX_SHARE
        ? "steady"
        : "unsteady";
  return {
    class: steadinessClass,
    groups,
    unsteadyGroups,
    ...(share === undefined ? {} : { share }),
  };
};

/**
 * A row's gap: its second-smallest positive round ÷ its smallest (rounds clamped to 0 left out);
 * undefined with fewer than two positive rounds. Pure.
 */
export const gemvRoundsGap = (rounds: readonly number[] | undefined): number | undefined => {
  const positive = (rounds ?? []).filter((round) => round > 0).toSorted((a, b) => a - b);
  const [smallest, second] = positive;
  return smallest === undefined || second === undefined ? undefined : second / smallest;
};

/**
 * The largest margin at which a gain clears the bar margin × gap × referenceGap: the gain ÷ the two
 * gaps. Pure.
 */
export const gemvBreakEven = (gain: number, gap: number, referenceGap: number): number =>
  gain / (gap * referenceGap);

/**
 * The margin a derivation runs with: {@link GEMV_MARGIN} when not given, otherwise a finite number of
 * at least 1 (below 1 a slower candidate would qualify). Throws for any other value.
 */
export const gemvMarginOf = (margin: number | undefined): number => {
  if (margin === undefined) return GEMV_MARGIN;
  if (Number.isFinite(margin) && margin >= 1) return margin;
  throw new Error(`The GEMV margin must be a finite number of at least 1 (${margin})`);
};
