/**
 * Implementation-time checks read the exporter's series (`outputs/series/`) and are opt-in
 * (ADR 0139 decision 6): the tests of graphs that are not shipped, and of families with no
 * distribution (MiniCPM5, the vowel detector; EmbeddingGemma until its distribution is published).
 * Series are transient — they exist while a model is exported and checked, and may be deleted once
 * its distribution, goldens and sha256 rows are in place (decision 8) — so the default lanes skip
 * these cases explicitly. Run them with `KARUME_SERIES=1` on a machine that holds the series.
 */
export const SERIES_CHECKS: boolean = process.env.KARUME_SERIES === "1";

/** The title suffix of an implementation-time case, so a skipped run says why. */
export const SERIES_CHECKS_OPT_IN = " (implementation-time series check, opt-in KARUME_SERIES=1)";
