/**
 * The margin of the headroom probe (**pipeline-independent shared value** — used by every family
 * that checks a stage's bytes with a trial allocation before opening it: Anima's DiT residency
 * (ADR 0112) and the MOSS-Image stages (ADR 0138)).
 *
 * MUST: not exported from the barrel (for the same reason as the co-located `options.ts` /
 * `dispose-steps.ts` — an internal mechanism, not a surface the user touches).
 */

/**
 * Margin added to the headroom probe (512 MiB).
 *
 * WHY: the estimate (`AdmissionReport.peakAccountedBytes`) + the staging of the largest part is
 * "what is accounted for", and real allocations can exceed it — ① the allocator's (gpu-allocator)
 * blocks are 256 MiB, and an allocation below block size that opens a new block takes up to 256 MiB
 * more than the probe saw ② wgpu loses the device when it exceeds 99 % of the budget after submit /
 * poll (the OOM line of allocation is 97 %) — this difference is about 178 MiB on a 9.93 GiB
 * machine (Arc B570) ③ the estimate's `unaccounted` (params, quantization scratch, etc.) is outside
 * the accounting. 512 MiB is a value that covers ① + ② and leaves a little for ③ (small against the
 * resident share of +2,646 MiB at 1024² with the default preset — ADR 0112).
 */
export const HEADROOM_MARGIN_BYTES = 512 * 1024 * 1024;
