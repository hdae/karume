/**
 * The fixed f32 kernel of the GRU **hidden-side scan** (ops `gru_scan` / `gru_scan_reverse`, an
 * extended composite op — ADR 0056). The caller prepares the input-side GEMM with the existing
 * `linear` (this op does not include it).
 *
 * ```
 * gh      = W_hh · h + b_hh                     // [3H] (bias last = the GEMM epilogue form)
 * r       = sigmoid(gh_r + gi_r)
 * z       = sigmoid(gh_z + gi_z)
 * n       = tanh(gi_n + gh_n · r)
 * h'      = (h − n) · z + n
 * ```
 *
 * MUST: **the order and grouping of the operations follow torch's GRU decomposition verbatim**
 * (ADR 0056 decision 3). The mathematically equivalent `(1 − z)·n + z·h` gives a different rounding
 * sequence in f32 (measured: 44,345 of 100,000 elements differ in bits). The order of the gate sums
 * (hidden side as the first operand) and of the `n` product (`gh_n · r`) also match the
 * decomposition.
 * MUST: bias is **last** (`(Σ W·h) + b`). Copying the conv family's bias-first (ADR 0024) changes
 * the rounding sequence and breaks bit identity with the decomposed path's `linear`.
 * MUST: the reduction runs sequentially in ascending `k`, spelled `acc = acc + w * h` (the same
 * form as GEMM — `accumulatorUpdate` in src/kernels/gemm.ts). GEMM splits by K tiles of 16, but the
 * addition order per output stays ascending in k, so the rounding sequence matches this plain
 * sequential loop.
 * MUST: the body of `sigmoid` shares **the same string** as the elementwise codegen (the same
 * reason as silu.ts — a copy could split the rounding sequence between the primitive and the fused
 * version).
 *
 * MUST: the body of `tanh` also shares `TANH_STABLE_WGSL` of the elementwise codegen as **the same
 * string** (the same reason as sigmoid). A plain built-in `tanh` computed as
 * `(exp(2x)−1)/(exp(2x)+1)` (Metal fast-math) silently produces NaN for |gi_n + gh_n·r| > 44.36,
 * and bit identity with the decomposed path (already tanh_stable) also breaks in the saturation
 * range. Measured on the distributed assets, the pre-activation reaches 17.81, already in the
 * saturation range 9.5 (the static upper bound towards the NaN boundary is 31.57 in layer 1 —
 * docs/research/2026-08-31-op-numerics-review.md). The clamp is bit-identical to an IEEE-faithful
 * tanh (the grounds are in the doc of TANH_STABLE_WGSL).
 *
 * ## Rounding barrier (workgroup memory round trip)
 *
 * Writing `a * b + c` as one expression **gets contracted to an fma** (measured on a real GPU in
 * this worktree: 15,371 of 65,536 elements differ in bits from the result of two dispatches, mul /
 * add. Naming it `let p = a * b;` gives the same result, and a `bitcast` round trip does not stop
 * it either). In the decomposed path `mul` and `add` are separate dispatches = a storage round
 * trip, so no contraction happens. So **the product is written to workgroup memory and read back
 * after a barrier**, leaving an explicit materialization point (the technique of silu.ts; 0
 * mismatches in the same measurement).
 *
 * **This rounding barrier is not guaranteed by the WGSL specification** — the specification only
 * allows fusion and does not require a workgroup memory round trip to be respected as an
 * optimisation barrier. If the bit-identity check (tests/gpu_gru_scan_parity_test.ts) breaks,
 * suspect this first.
 * NOTE (measured on M2, 2026-08-31): on Metal this check does break (1–64 ULP at H=128; the same
 * signature before and after the tanh_stable change = pre-existing; the Metal section of
 * docs/known-issues.md). Real quality is covered by the vowel-detector golden being green on M2 —
 * strengthening the barrier's expression form is filed as a root-cause candidate.
 *
 * ## The scan decision (for correctness, not performance)
 *
 * The recurrences of different batch elements are independent, so **one workgroup = one batch
 * element** (the batch direction crosses the dispatch limit with the same grid-stride as
 * layer-norm). The assignment is fixed at **one lane = one hidden unit**, and shapes above
 * `H ≤ {@link GRU_SCAN_MAX_HIDDEN}` fail loudly in {@link gruScanParams} (ADR 0056 decision 5).
 * The time loop and the barriers sit only inside workgroup-uniform control flow (dims is uniform —
 * the same as the MUST of layer-norm.ts).
 *
 * MUST: the reverse direction reverses **only the scan order**; output is always written at the
 * forward time index (a form without an interposed `flip` is one of the reasons this op exists —
 * ADR 0056 decision 2).
 */

import {
  IS_NAN_BITS_WGSL,
  SIGMOID_STABLE_WGSL,
  TANH_STABLE_WGSL,
} from "../codegen/numerics-wgsl.ts";
import { CodegenError } from "../codegen/errors.ts";
import { GRU_SCAN_MAX_HIDDEN } from "../codegen/limits.ts";
import { assertU32Params } from "../codegen/params.ts";

/**
 * Lanes per workgroup. **One lane = one hidden unit**, so it equals the hidden-width limit
 * {@link GRU_SCAN_MAX_HIDDEN} and is derived from it rather than kept as a number in two places.
 * The limit itself is also checked by the contract layer (ops/shapes.ts), so it lives in the
 * neutral codegen/limits.ts (the only measured shape is vowel-detector's 2-layer BiGRU with
 * H = 128 — ADR 0056 decision 5).
 */
export const GRU_SCAN_WORKGROUP_SIZE = GRU_SCAN_MAX_HIDDEN;

/** Scan direction (one-to-one with the two ops `gru_scan` / `gru_scan_reverse`). */
export type GruScanDirection = "forward" | "reverse";

const canonicalizeDirection = (direction: GruScanDirection): GruScanDirection => {
  if (direction === "forward" || direction === "reverse") return direction;
  throw new CodegenError(`gru_scan codegen: 走査方向が不正（${direction}）`);
};

/**
 * MUST: bump the key when the WGSL changes (the pipeline cache does not look at the source).
 * v2: the candidate gate's tanh was switched to the shared TANH_STABLE_WGSL (root-cause fix of the
 * silent NaN under Metal fast-math).
 */
export const gruScanKey = (direction: GruScanDirection): string =>
  `gru_scan:v2:f32:${canonicalizeDirection(
    direction,
  )}:wg${GRU_SCAN_WORKGROUP_SIZE}:h${GRU_SCAN_MAX_HIDDEN}`;

export const gruScanWgsl = (direction: GruScanDirection): string => {
  const canonical = canonicalizeDirection(direction);
  // The reverse direction flips only the scan order (the output time index `t` is in forward
  // order in both).
  const timeIndex = canonical === "reverse" ? "dims.time - 1u - step" : "step";
  return `// karume gru_scan (${canonical}, f32, 1 workgroup = 1 batch element / 1 lane = 1 hidden unit)
struct Dims {
  time: u32,
  batch: u32,
  hidden: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> gi: array<f32>;
@group(0) @binding(2) var<storage, read> h0: array<f32>;
@group(0) @binding(3) var<storage, read> w_hh: array<f32>;
@group(0) @binding(4) var<storage, read> b_hh: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

var<workgroup> h_shared: array<f32, ${GRU_SCAN_MAX_HIDDEN}>;
// relay for the rounding barriers (holds the f32 bit pattern as is). Each lane touches only its
// own slot.
var<workgroup> stage: array<u32, ${GRU_SCAN_MAX_HIDDEN}>;

${SIGMOID_STABLE_WGSL}

${IS_NAN_BITS_WGSL}

${TANH_STABLE_WGSL}

@compute @workgroup_size(${GRU_SCAN_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let hidden = dims.hidden;
  let gates = hidden * 3u;
  let lane_used = lid < hidden;
  // W_hh is [3H, H], row-major. The gates are laid out H rows each, in the order r / z / n.
  let row_r = lid * hidden;
  let row_z = (hidden + lid) * hidden;
  let row_n = (hidden * 2u + lid) * hidden;
  var item = wid.x;
  while (item < dims.batch) {
    if (lane_used) {
      h_shared[lid] = h0[item * hidden + lid];
    }
    workgroupBarrier();
    var step = 0u;
    while (step < dims.time) {
      let t = ${timeIndex};
      let gi_base = (t * dims.batch + item) * gates;
      var h_prev = 0.0;
      var gate_z = 0.0;
      if (lane_used) {
        h_prev = h_shared[lid];
        var acc_r = 0.0;
        var acc_z = 0.0;
        var acc_n = 0.0;
        for (var k = 0u; k < hidden; k = k + 1u) {
          let hk = h_shared[k];
          acc_r = acc_r + w_hh[row_r + k] * hk;
          acc_z = acc_z + w_hh[row_z + k] * hk;
          acc_n = acc_n + w_hh[row_n + k] * hk;
        }
        // bias last (the same as the GEMM epilogue). The gate sums take the hidden side as the
        // first operand.
        let gate_r = sigmoid_stable((acc_r + b_hh[lid]) + gi[gi_base + lid]);
        gate_z = sigmoid_stable((acc_z + b_hh[hidden + lid]) + gi[gi_base + hidden + lid]);
        let gh_n = acc_n + b_hh[hidden * 2u + lid];
        // rounding barrier ①: the decomposed path's mul is a separate dispatch, so no fma
        // contraction here
        stage[lid] = bitcast<u32>(gh_n * gate_r);
      }
      workgroupBarrier();
      var cand = 0.0;
      if (lane_used) {
        // n = tanh(i_n + h_n·r) — the input side is the first operand
        cand = tanh_stable(gi[gi_base + hidden * 2u + lid] + bitcast<f32>(stage[lid]));
        // rounding barrier ②: between the mul and the add of h' = (h − n)·z + n
        stage[lid] = bitcast<u32>((h_prev - cand) * gate_z);
      }
      workgroupBarrier();
      if (lane_used) {
        let h_next = bitcast<f32>(stage[lid]) + cand;
        h_shared[lid] = h_next;
        out[(t * dims.batch + item) * hidden + lid] = h_next;
      }
      // make this write complete before the next step's reduction reads h_shared
      workgroupBarrier();
      step = step + 1u;
    }
    item = item + nwg.x;
  }
}
`;
};

/** Geometry of gru_scan (the one input type of params). */
type GruScanDims = {
  /** Time length T (the actual length after the symbolic dimension is bound). */
  readonly time: number;
  /** Batch N. */
  readonly batch: number;
  /** Hidden width H. */
  readonly hidden: number;
};

/**
 * The uniform Dims: 12 bytes of content allocated as 16 bytes (allocating in 16-byte units is a
 * project convention (the fusion pass fixes the same form in ADR 0040), not a requirement of the
 * specification — the same treatment as layer-norm.ts).
 *
 * MUST: the limit on `hidden` is checked here (redundant with the contract check, but a check that
 * the direct kernel-call path also passes). Letting an excess through silently writes `h_shared`
 * out of range and corrupts other units' state without an exception.
 */
export const gruScanParams = (dims: GruScanDims): Uint32Array<ArrayBuffer> => {
  const { time, batch, hidden } = dims;
  assertU32Params("gru_scan params", { time, batch });
  if (!Number.isSafeInteger(hidden) || hidden < 1) {
    throw new CodegenError(`gru_scan params: hidden は正整数（${hidden}）`);
  }
  if (hidden > GRU_SCAN_MAX_HIDDEN) {
    throw new CodegenError(
      `gru_scan params: hidden ${hidden} が上限 ${GRU_SCAN_MAX_HIDDEN} を超える（1 lane = 1 隠れユニットの割り当て — ADR 0056 決定 5）`,
    );
  }
  return new Uint32Array([time, batch, hidden, 0]);
};
