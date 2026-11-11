/**
 * The expected fusion counter record of a plan made with the built-in rule list, for tests that
 * compare the whole record (`assertEquals(plan.counts, builtInCounts({ silu: 1 }))`).
 *
 * A test spells only the counters it expects to move; every other counter is 0, so an unexpected
 * non-zero counter still fails the comparison. The zero record is the scan's own
 * (`zeroFusionCounts` in src/runtime/fusion.ts), so a counter added to the built-in list needs no
 * edit here.
 */

import { FUSION_RULES, type FusionPlan, zeroFusionCounts } from "../../src/runtime/fusion.ts";

/** The counter record of a plan made with the built-in rule list (no external rule's key). */
type BuiltInCounts = FusionPlan["counts"];

/** The built-in counter record with `hits` and every other counter at 0. */
export const builtInCounts = (hits: Partial<BuiltInCounts> = {}): BuiltInCounts => ({
  ...zeroFusionCounts(FUSION_RULES),
  ...hits,
});
