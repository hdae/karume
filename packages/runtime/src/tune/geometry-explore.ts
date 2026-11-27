/**
 * The geometry sweep's `explore` set (geometries.ts's `EXPLORE_SET`; device-free): which order its
 * cases are measured in, and which candidates a case no longer measures.
 *
 * The derivation (derive.ts) adopts a geometry for a profile field — a class of cases
 * (`sweepCaseSlot`) — only if in every case of the class its output is the default's and its ratio
 * to the default is at least `minSpeedup`. A candidate that loses in one case of a class is never
 * adopted for that class, so it need not be measured in the class's other cases:
 * 1. the cases are measured class by class ({@link geometryExploreOrder});
 * 2. the first case of a class measures every candidate of `full`, after the quick look against the
 *    default at the set's margin as any set does;
 * 3. a candidate is **dropped** for the class once, in a case of the class, it was screened out by
 *    the quick look, failed, wrote an output that is not the default's, or was timed at a ratio to
 *    the default below {@link EXPLORE_KEEP_SPEEDUP} ({@link exploreRulesOut}); the later cases of the
 *    class neither prepare nor measure it, and write a skipped row for it
 *    ({@link geometryExploreDropped}). The default geometry is always measured.
 *
 * A record of the set is a derivation input as a `full` record that holds part of the candidates
 * is: a skipped row is no observation, as a screened row is.
 *
 * MUST: nothing is computed at import time (functions only).
 */

import { SWEEP_CASES, type SweepCase } from "./cases.ts";
import { type ProfileSlot, sweepCaseSlot } from "./derive.ts";
import type { SweepRow } from "./report.ts";

/**
 * The ratio to the default below which a timed candidate is dropped for its class: 1, not slower
 * than the default. It stays below the derivation's threshold (`DEFAULT_MIN_SPEEDUP`, ×1.05) so that
 * a candidate a little short of it in one record can still reach it in the geometric mean of two.
 *
 * Source: replayed on the two `full` geometry-sweep records of an RTX 5070 Ti under Chrome of
 * 2026-11-18 (78 cases, 6,191 candidate rows each), dropping at a ratio below 1 prepares 2,198 and
 * 2,201 of the candidate rows and times 1,439 and 1,443 where `full` timed 3,278 and 3,279, an
 * estimated 58 % less time in the cases, and the fields the derivation adopts from the two records
 * together are the same eleven decisions. Dropping below ×1.05 also left them unchanged (1,141 and
 * 1,152 timed rows); 1 was chosen for the slack.
 */
export const EXPLORE_KEEP_SPEEDUP = 1;

/** The case of the explore set's table (`full`'s, {@link SWEEP_CASES}) with `id`; another id throws. */
const exploreCase = (id: string): SweepCase => {
  const found = SWEEP_CASES.find((sweepCase) => sweepCase.id === id);
  if (found === undefined) {
    throw new Error(`Case ${id} is not in the shape table of the explore set`);
  }
  return found;
};

/**
 * The class of a case of the explore set's table (`full`'s) by its id: the profile field the
 * derivation reads the case in (`sweepCaseSlot` — a `gemmRows` or `i8a8.linear` bucket, an attention
 * stage, `conv2d.rows64` or `conv2d.rows32`, `i8a8.conv3d`). Another id throws. Pure.
 */
export const geometryCaseClass = (caseId: string): ProfileSlot =>
  sweepCaseSlot(exploreCase(caseId));

/**
 * The order the explore set measures `cases` in: class by class ({@link geometryCaseClass}), the
 * classes in the order of their first case among `cases`, the cases of a class in the order given.
 * `cases` come in the case table's order (the sweep's plan and the workbench's case list do), each a
 * case of the explore set's table — another id throws. Pure.
 */
export const geometryExploreOrder = <T extends { readonly id: string }>(
  cases: readonly T[],
): T[] => {
  const classes = new Map<ProfileSlot, T[]>();
  for (const entry of cases) {
    const slot = geometryCaseClass(entry.id);
    classes.set(slot, [...(classes.get(slot) ?? []), entry]);
  }
  return [...classes.values()].flat();
};

/**
 * Whether a candidate's row in a case rules the candidate out of the later cases of the case's
 * class: a row that is not the default's whose geometry the quick look screened out, that failed,
 * whose output is not the default's, or whose ratio to the default is below
 * {@link EXPLORE_KEEP_SPEEDUP} — or a skipped row (an earlier case ruled it out already). A row
 * without a ratio or an output comparison (the default's row failed) rules nothing out. Pure.
 */
export const exploreRulesOut = (
  row: Pick<
    SweepRow,
    "isDefault" | "screened" | "skipped" | "error" | "identicalToDefault" | "speedupVsDefault"
  >,
): boolean =>
  !row.isDefault &&
  (row.skipped === true ||
    row.screened === true ||
    row.error !== undefined ||
    row.identicalToDefault === false ||
    (row.speedupVsDefault !== undefined && row.speedupVsDefault < EXPLORE_KEEP_SPEEDUP));

/**
 * The candidates the explore set no longer measures in `target`, by geometry name: those that a row
 * of `rows` — the rows of the cases measured before it, by the same call or by earlier calls — of a
 * case of `target`'s class rules out ({@link exploreRulesOut}). Every row is of a case of the explore
 * set's table. Pure.
 */
export const geometryExploreDropped = (
  target: SweepCase,
  rows: readonly SweepRow[],
): ReadonlySet<string> => {
  const slot = sweepCaseSlot(target);
  const slots = new Map<string, ProfileSlot>();
  const slotOfCase = (caseId: string): ProfileSlot => {
    const known = slots.get(caseId);
    if (known !== undefined) return known;
    const found = geometryCaseClass(caseId);
    slots.set(caseId, found);
    return found;
  };
  return new Set(
    rows
      .filter((row) => exploreRulesOut(row) && slotOfCase(row.caseId) === slot)
      .map((row) => row.geometry),
  );
};
