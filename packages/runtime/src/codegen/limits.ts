/**
 * Range limits that come from a kernel implementation and that **the op contract layer guards
 * as well**.
 *
 * MUST: they live here, in one place. A limit that exists only on the params side turns into "the
 * contract check passed, then an internal error at execution time", and the user's diagnostics do
 * not show which value of which op is out of range (the same split as the strided family's rank
 * limit {@link "./strided.ts"} `STRIDED_RANK`). It is also a neutral place that avoids an
 * ops → kernels dependency (ops and kernels both already depend on codegen).
 */

/**
 * Upper limit of the gru_scan hidden width H (= the kernel's workgroup size). The assignment is
 * **one lane = one hidden unit**, and the generated WGSL's `h_shared` / `stage` are workgroup
 * arrays of this length (kernels/gru-scan.ts).
 *
 * MUST: exceeding it fails loudly both in the contract layer (gru_scan in ops/shapes.ts) and in
 * params (`gruScanParams`) — a silent fallback writes the workgroup arrays out of range and
 * corrupts other units' state without an exception. Raising the limit needs a workgroup-internal
 * grid-stride and a double-buffered `h` = a separate design decision, which MAY be made on demand
 * (re-graded 2026-10-18 — ADR 0056 decision 5).
 */
export const GRU_SCAN_MAX_HIDDEN = 256;
