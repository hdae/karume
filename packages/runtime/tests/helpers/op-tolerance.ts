/**
 * **Per-op tolerance table** of the op vocabulary sweep (checkAll in gpu_ops_test.ts).
 *
 * Source of truth = [docs/research/2026-08-31-op-tolerance-measurement.md](../../../../docs/research/2026-08-31-op-tolerance-measurement.md)
 * §7 (derived from real-GPU measurements + the K-dependence rule maxAbs ≈ C·eps·√K·max|expected|;
 * every band was confirmed mechanically, by usage ratio, to keep at least 4× margin over the
 * measurement). The old DEFAULT_TOLERANCE (atol 1e-5 / rtol 1e-3) used a usage ratio of only
 * 2.6e-2 even at worst as the sweep's check, a band that let regressions through (same §2.2).
 *
 * MUST: **an op missing from the table fails loudly** (it does not fall back to a default band).
 * Adding a new op means measuring it and adding a row — closing off "silently passing with a
 * generic band" is why this table exists. Per-case exceptions are an `OpCase.tolerance` override
 * (the escape hatch for ill-conditioned inputs — the table stays unchanged).
 *
 * NOTE: among the "exact (0, 0)" ops, the GEMM / reduction ones are exact only because **the
 * inputs of this corpus lie on a binary lattice** (the lattice argument of the research §2 — it
 * holds for any implementation conforming to IEEE-754 f32). For general inputs the band of the
 * GEMM / reduction class is authoritative, and the bit-identity check (gpu_ops_test.ts) pins this
 * exactness as a separate test.
 */

import type { Tolerance } from "../../src/reference/allclose.ts";

const EXACT: Tolerance = { atol: 0, rtol: 0 };
/** Single division op (measured 1 ULP; 6.4× the spec's 2.5 ULP). */
const ULP_DIV: Tolerance = { atol: 1e-7, rtol: 2e-6 };
/** Single transcendental op (measured ≤13 ULP / 5× the spec's ≤12 ULP + an absolute floor). */
const TRANSCENDENTAL: Tolerance = { atol: 2e-6, rtol: 8e-6 };
/**
 * Composite ops containing transcendentals — cancellation points (the negative tail of gelu and the
 * like) are covered by the absolute floor (not cut off by the relative term).
 */
const COMPOSITE: Tolerance = { atol: 4e-6, rtol: 8e-6 };
/** The softmax family (16× the measured abs 6.0e-8 — K independence confirmed by measurement). */
const SOFTMAX: Tolerance = { atol: 1e-6, rtol: 8e-6 };
/** 4-tap sum (bilinear). */
const SMALL_REDUCE: Tolerance = { atol: 4e-6, rtol: 4e-6 };
/** Row reductions (from the √K rule for K ≤ 16384). */
const ROW_REDUCE: Tolerance = { atol: 4e-5, rtol: 4e-6 };
/**
 * attention (6× the abs 6.9e-6 of the saturated-softmax case; the relative term diverges, so the
 * absolute floor dominates).
 */
const ATTENTION: Tolerance = { atol: 4e-5, rtol: 8e-6 };
/**
 * The GEMM / conv family (from the rule 4·eps·√K at the corpus maximum K=72; re-derive the band
 * when adding a case whose K is orders of magnitude larger). The GPU vs reference comparisons of
 * the quantized-weight paths (gpu_{f16,i8,i4}_weights / skinny / gemv) also use this class
 * explicitly.
 */
export const GEMM_TOLERANCE: Tolerance = { atol: 2e-4, rtol: 4e-6 };

const OP_TOLERANCE: Readonly<Record<string, Tolerance>> = {
  // Exact — ops that create no new values / correctly rounded single operations (f32 output)
  static_quantize: EXACT,
  neg: EXACT,
  abs: EXACT,
  relu: EXACT,
  clamp: EXACT,
  clamp_min: EXACT,
  leaky_relu: EXACT,
  where: EXACT,
  cast: EXACT,
  add: EXACT,
  sub: EXACT,
  mul: EXACT,
  // Exact — data movement only
  reshape: EXACT,
  permute: EXACT,
  expand: EXACT,
  slice: EXACT,
  cat: EXACT,
  sym_prefix_slice: EXACT,
  pad: EXACT,
  flip: EXACT,
  gather: EXACT,
  embedding: EXACT,
  masked_fill: EXACT,
  upsample2x: EXACT,
  // Exact — integer / bool output (the dtype branch of compareTensors makes these EXACT anyway,
  // but they keep rows so the table covers every op (fail loudly))
  bitwise_and: EXACT,
  bitwise_not: EXACT,
  ge: EXACT,
  ge_scalar: EXACT,
  gt_scalar: EXACT,
  le_scalar: EXACT,
  // Exact — selection / indices / matching sequential rounding (for cumsum the reference mirrors
  // the GPU's sequential rounding)
  amax: EXACT,
  amin: EXACT,
  argmax: EXACT,
  topk: EXACT,
  cumsum: EXACT,
  // ULP level
  div: ULP_DIV,
  // Transcendental (single op)
  exp: TRANSCENDENTAL,
  log: TRANSCENDENTAL,
  log1p: TRANSCENDENTAL,
  sqrt: TRANSCENDENTAL,
  sin: TRANSCENDENTAL,
  tanh: TRANSCENDENTAL,
  sigmoid: TRANSCENDENTAL,
  // Composite (cancellation points covered by the absolute floor)
  gelu: COMPOSITE,
  gelu_tanh: COMPOSITE,
  silu: COMPOSITE,
  layer_norm: COMPOSITE,
  group_norm: COMPOSITE,
  rms_norm: COMPOSITE,
  gru_scan: COMPOSITE,
  gru_scan_reverse: COMPOSITE,
  softmax: SOFTMAX,
  safe_softmax: SOFTMAX,
  // Reduction / GEMM
  upsample_bilinear2d: SMALL_REDUCE,
  sum: ROW_REDUCE,
  attention: ATTENTION,
  matmul: GEMM_TOLERANCE,
  bmm: GEMM_TOLERANCE,
  linear: GEMM_TOLERANCE,
  conv1d: GEMM_TOLERANCE,
  conv2d: GEMM_TOLERANCE,
  conv_transpose1d: GEMM_TOLERANCE,
  deform_conv2d: GEMM_TOLERANCE,
};

/**
 * Per-op tolerance used by the sweep. An op missing from the table fails loudly (the MUST above).
 * For i32 / bool outputs compareTensors picks EXACT by dtype, so this holds only the f32 bands.
 */
export const opTolerance = (op: string): Tolerance => {
  if (!Object.hasOwn(OP_TOLERANCE, op)) {
    throw new Error(
      `op-tolerance: op '${op}' has no tolerance in the table — measure it and add a row` +
        " (procedure in docs/research/2026-08-31-op-tolerance-measurement.md §7; there is no default band)",
    );
  }
  return OP_TOLERANCE[op];
};

/**
 * The set of ops pinned to be **bit-identical between the GPU and the f64 oracle** on the current
 * corpus (research §7.1 — the inputs lie on a binary lattice and the intermediate sums stay in the
 * exact-integer range of f32, so no rounding ever happens in any IEEE-754 f32 conforming
 * implementation). A mismatch is not a reason to loosen the tolerance but a signal that "the driver
 * computes below f32 / the generated WGSL fell back to a different operation". If a particular
 * driver breaks it, take that op out of the check and record the reason (the same practice as the
 * tanh saturation canary).
 */
export const BIT_IDENTICAL_OPS: ReadonlySet<string> = new Set([
  "matmul",
  "bmm",
  "linear",
  "conv1d",
  "conv2d",
  "conv_transpose1d",
  "deform_conv2d",
  "sum",
  "upsample_bilinear2d",
]);
