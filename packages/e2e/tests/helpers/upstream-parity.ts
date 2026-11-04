/**
 * Upstream-parity cases (model-exit or per-component comparisons against torch / diffusers
 * goldens with a tolerance band) are opt-in (maintainer ruling 2026-10-17): parity is proven when a
 * family is implemented, and later changes are caught by the sha256 reference rows and the self
 * A/B checks, so the default lanes skip these cases. Run them with `KARUME_UPSTREAM_PARITY=1`.
 */
export const UPSTREAM_PARITY: boolean = process.env.KARUME_UPSTREAM_PARITY === "1";

/** The title suffix of an opt-in parity case, so a skipped run says why. */
export const UPSTREAM_PARITY_OPT_IN = " (upstream parity, opt-in KARUME_UPSTREAM_PARITY=1)";
