/**
 * The table that converts one unit of a timestamp-query value to ns (per environment key — shared
 * by the Wan DiT / umT5 e2e).
 *
 * Deno returns wgpu's raw ticks without converting them (docs/known-issues.md, section "Intel Arc B570" — on the B570
 * 1 tick = 52.0833 ns). The conversion of an environment that is not in the table is not guessed: callers that check
 * times fail loudly through {@link timestampUnitNs}, and callers that only record them see `undefined` from
 * {@link lookupTimestampUnitNs} and skip the conversion.
 *
 * The environment key comes in as an argument (this module imports nothing — zero side effects).
 */

/** ns per timestamp unit (environment key → ns). Add a row only for an environment whose period was measured. */
export const TIMESTAMP_UNIT_NS: Readonly<Record<string, number>> = {
  "deno-intel-graphics-bmg-g21": 52.0833,
  // RTX 3080 Ti (NVIDIA driver 615.71): Vulkan's timestampPeriod is 1. Measured on 2026-10-06: the raw ticks of one
  // pass matched the wall clock minus the fixed cost of a submit (about 11 ms) within 1% (6 runs of 133–153 ms passes,
  // with both a throw-away and a reused query set).
  "deno-nvidia-geforce-rtx-3080-ti": 1,
  // Dawn (Node) reports timestamps already converted to ns (the Vulkan timestampPeriod is applied
  // inside Dawn); the test setup disables Dawn's 65,536 ns quantisation, so 1 unit = 1 ns.
  "node-nvidia-geforce-rtx-3080-ti": 1,
};

/** The table's row (`undefined` when missing — the conversion is not guessed). */
export const lookupTimestampUnitNs = (key: string | undefined): number | undefined =>
  key !== undefined && Object.hasOwn(TIMESTAMP_UNIT_NS, key) ? TIMESTAMP_UNIT_NS[key] : undefined;

/**
 * ns per timestamp unit of the environment key `key` ({@link TIMESTAMP_UNIT_NS}). Fails loudly when not in the table.
 */
export const timestampUnitNs = (key: string | undefined): number => {
  const unit = lookupTimestampUnitNs(key);
  if (unit === undefined) {
    throw new Error(
      `TIMESTAMP_UNIT_NS has no ns-per-timestamp-unit row for the environment key '${key}'` +
        " (the conversion is not guessed — see docs/known-issues.md and add a row)",
    );
  }
  return unit;
};
