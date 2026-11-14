// Fixture of gpu_ops_test.ts (the OpCase type, deterministic input generators, the case arrays per
// op). The test bodies (runCase / checkAll / the test set) stay in gpu_ops_test.ts; this file
// holds only "the data passed to checkAll" — dependencies go one way only, helpers → src.

import type { Tolerance } from "../../src/reference/allclose.ts";
import type { RefTensor } from "../../src/reference/ops.ts";
import { fill } from "./model-fixture.ts";

export type OpCase = {
  readonly name: string;
  readonly op: string;
  readonly inputs: readonly RefTensor[];
  /** Shapes in ascending output-slot order (ADR 0068 decision 1 — only topk has two). */
  readonly outShapes: readonly (readonly number[])[];
  /**
   * dtypes in ascending output-slot order (default: the input's — only cast / argmax / the index
   * side of topk differ).
   */
  readonly outDtypes?: readonly ("f32" | "i32" | "bool")[];
  readonly attrs?: Record<string, unknown>;
  /**
   * Per-case tolerance override (the escape hatch for ill-conditioned inputs — the table of
   * helpers/op-tolerance.ts stays unchanged). Default: the per-op table (`opTolerance` of
   * {@link "./op-tolerance.ts"}).
   */
  readonly tolerance?: Tolerance;
};

/** Deterministic sequence spanning negative values, 0 and positive values. */
export const SIGNED = (i: number): number => ((i % 13) - 6) * 0.75;
/** Sequence kept inside the domain of log / sqrt. */
export const POSITIVE = (i: number): number => 0.125 + (i % 17) * 0.5;
/** Divisors for div (no 0). */
const NONZERO = (i: number): number => ((i % 7) + 1) * 0.5;

const UNARY_INPUTS: readonly (readonly [string, (index: number) => number])[] = [
  ["neg", SIGNED],
  ["abs", SIGNED],
  ["exp", SIGNED],
  ["log", POSITIVE],
  ["sqrt", POSITIVE],
  ["sin", SIGNED],
  ["tanh", SIGNED],
  ["sigmoid", SIGNED],
  ["relu", SIGNED],
  ["gelu", SIGNED],
  ["gelu_tanh", SIGNED],
];

export const UNARY_CASES: readonly OpCase[] = UNARY_INPUTS.map(([op, generator]) => ({
  name: `unary ${op} [3,5]`,
  op,
  inputs: [fill([3, 5], generator)],
  outShapes: [[3, 5]],
}));

export const BINARY_CASES: readonly OpCase[] = [
  {
    name: "add same shape [4,3]",
    op: "add",
    inputs: [fill([4, 3], SIGNED), fill([4, 3], POSITIVE)],
    outShapes: [[4, 3]],
  },
  {
    name: "sub right-aligned broadcast [4,3] - [3]",
    op: "sub",
    inputs: [fill([4, 3], SIGNED), fill([3], POSITIVE)],
    outShapes: [[4, 3]],
  },
  {
    name: "mul broadcast on both sides [4,1] * [1,3]",
    op: "mul",
    inputs: [fill([4, 1], SIGNED), fill([1, 3], POSITIVE)],
    outShapes: [[4, 3]],
  },
  {
    name: "div rank3 broadcast [2,3,4] / [3,1]",
    op: "div",
    inputs: [fill([2, 3, 4], SIGNED), fill([3, 1], NONZERO)],
    outShapes: [[2, 3, 4]],
  },
  {
    name: "add scalar broadcast [5] + []",
    op: "add",
    inputs: [fill([5], SIGNED), fill([], () => 2.5)],
    outShapes: [[5]],
  },
];

/** Deterministic integer sequence spanning 0/1 (matching the shapes seen on the mask path). */
const MASK = (i: number): number => (i % 3 === 0 ? 0 : 1);
/** Integer sequence spanning negative values, 0 and positive values. */
export const INTEGERS = (i: number): number => (i % 11) - 5;
/**
 * Fractional sequence on which truncation is distinguishable from round / floor (includes ±.5 and
 * ±.7).
 */
const FRACTIONS = (i: number): number => ((i % 9) - 4) * 0.6;

/**
 * The i32 / bool execution path (ADR 0009). Only the forms seen on the mask path of real graphs are
 * enabled: the mask outer product `mul(i32, i32)` / the `sub(i32)` of `1 - mask` / casts to
 * boolean / `bitwise_not`.
 */
export const DTYPE_CASES: readonly OpCase[] = [
  {
    // mask outer product [4,1] × [1,4] (in practice [1,1,T,1]×[1,1,1,T] via unsqueeze)
    name: "mul i32 outer-product broadcast [4,1] * [1,4]",
    op: "mul",
    inputs: [fill([4, 1], MASK, "i32"), fill([1, 4], MASK, "i32")],
    outShapes: [[4, 4]],
  },
  {
    name: "sub i32 same shape [3,5]",
    op: "sub",
    inputs: [fill([3, 5], INTEGERS, "i32"), fill([3, 5], MASK, "i32")],
    outShapes: [[3, 5]],
  },
  {
    name: "bitwise_not bool [2,6]",
    op: "bitwise_not",
    inputs: [fill([2, 6], MASK, "bool")],
    outShapes: [[2, 6]],
  },
  {
    name: "cast f32 → i32 (truncation towards 0) [3,3]",
    op: "cast",
    inputs: [fill([3, 3], FRACTIONS)],
    outShapes: [[3, 3]],
    outDtypes: ["i32"],
    attrs: { to: "i32" },
  },
  {
    name: "cast i32 → bool（x != 0）[3,4]",
    op: "cast",
    inputs: [fill([3, 4], INTEGERS, "i32")],
    outShapes: [[3, 4]],
    outDtypes: ["bool"],
    attrs: { to: "bool" },
  },
  {
    name: "cast bool → f32 (0/1 as weights) [2,5]",
    op: "cast",
    inputs: [fill([2, 5], MASK, "bool")],
    outShapes: [[2, 5]],
    outDtypes: ["f32"],
    attrs: { to: "f32" },
  },
  {
    name: "cast i32 → f32 [7]",
    op: "cast",
    inputs: [fill([7], INTEGERS, "i32")],
    outShapes: [[7]],
    outDtypes: ["f32"],
    attrs: { to: "f32" },
  },
  {
    name: "cast f32 → bool（x != 0）[6]",
    op: "cast",
    inputs: [fill([6], FRACTIONS)],
    outShapes: [[6]],
    outDtypes: ["bool"],
    attrs: { to: "bool" },
  },
];

/**
 * The math ops of M1-P3 step 3 (the spline of sdp / the leaky_relu of dec — recon §2).
 *
 * MUST: ops with a "direction" are exercised with asymmetric inputs. The branches of where, the
 * inequality of comparisons and the accumulation direction of cumsum can all go wrong without
 * changing shape or dtype, so only the values can detect it.
 */
export const MATH_CASES: readonly OpCase[] = [
  {
    // Domain x > -1. Small x (where the correction matters) is checked for precision by a dedicated
    // test
    name: "log1p [3,5]",
    op: "log1p",
    inputs: [fill([3, 5], POSITIVE)],
    outShapes: [[3, 5]],
  },
  {
    // SIGNED spans ±4.5, so all 3 branches (below / inside / above) are taken
    name: "clamp [3,5] min=-1.5 max=2.25",
    op: "clamp",
    inputs: [fill([3, 5], SIGNED)],
    outShapes: [[3, 5]],
    attrs: { min: -1.5, max: 2.25 },
  },
  {
    name: "clamp collapsing form min == max",
    op: "clamp",
    inputs: [fill([6], SIGNED)],
    outShapes: [[6]],
    attrs: { min: 1.5, max: 1.5 },
  },
  {
    // One-sided clamp (ADR 0017). SIGNED spans ±4.5, so both branches (below / above) are taken
    name: "clamp_min [3,5] min=-1.5",
    op: "clamp_min",
    inputs: [fill([3, 5], SIGNED)],
    outShapes: [[3, 5]],
    attrs: { min: -1.5 },
  },
  {
    // The shape seen in practice (clamp(min=eps) of channel L2 normalization) — a positive-only
    // sequence also takes the side where the lower bound has no effect
    name: "clamp_min [4] min=1e-12 (the eps form seen in practice)",
    op: "clamp_min",
    inputs: [fill([4], (i) => [0, -1, 1e-20, 2][i])],
    outShapes: [[4]],
    attrs: { min: 1e-12 },
  },
  {
    // The two slopes seen in practice (ADR 0015). They go through params, so only the values change
    // on the same pipeline
    name: "leaky_relu slope=0.1 [3,5]",
    op: "leaky_relu",
    inputs: [fill([3, 5], SIGNED)],
    outShapes: [[3, 5]],
    attrs: { negative_slope: 0.1 },
  },
  {
    name: "leaky_relu slope=0.01 [3,5]",
    op: "leaky_relu",
    inputs: [fill([3, 5], SIGNED)],
    outShapes: [[3, 5]],
    attrs: { negative_slope: 0.01 },
  },
  {
    // SIGNED is exactly 0 at i % 13 == 6 = always hits the point where ge and gt differ
    name: "ge_scalar value=0 (including the equality point)",
    op: "ge_scalar",
    inputs: [fill([3, 5], SIGNED)],
    outShapes: [[3, 5]],
    outDtypes: ["bool"],
    attrs: { value: 0 },
  },
  {
    name: "gt_scalar value=0 (including the equality point)",
    op: "gt_scalar",
    inputs: [fill([3, 5], SIGNED)],
    outShapes: [[3, 5]],
    outDtypes: ["bool"],
    attrs: { value: 0 },
  },
  {
    name: "le_scalar value=0 (including the equality point)",
    op: "le_scalar",
    inputs: [fill([3, 5], SIGNED)],
    outShapes: [[3, 5]],
    outDtypes: ["bool"],
    attrs: { value: 0 },
  },
  {
    // Right-aligned broadcast shrunk from the searchsorted form (inputs[…,None] >= bl)
    name: "ge [4,1] >= [3] (right-aligned broadcast)",
    op: "ge",
    inputs: [fill([4, 1], SIGNED), fill([3], (i) => (i - 1) * 0.75)],
    outShapes: [[4, 3]],
    outDtypes: ["bool"],
  },
  {
    name: "bitwise_and bool [2,6] & [6]",
    op: "bitwise_and",
    inputs: [fill([2, 6], MASK, "bool"), fill([6], (i) => (i % 2 === 0 ? 1 : 0), "bool")],
    outShapes: [[2, 6]],
    outDtypes: ["bool"],
  },
  {
    // a is filled with positive values and b with negative ones, so swapped branches show in the
    // values
    name: "where cond[1,4] ? a[3,4] : b[3,1]",
    op: "where",
    inputs: [fill([1, 4], MASK, "bool"), fill([3, 4], POSITIVE), fill([3, 1], (i) => -10 - i)],
    outShapes: [[3, 4]],
    // The output has the dtype of **the value side**, not of the condition (slot 0) (the contract
    // table's mapping bool → f32)
    outDtypes: ["f32"],
  },
  {
    name: "where same shape [2,3,4]",
    op: "where",
    inputs: [
      fill([2, 3, 4], (i) => (i % 3 === 0 ? 1 : 0), "bool"),
      fill([2, 3, 4], SIGNED),
      fill([2, 3, 4], POSITIVE),
    ],
    outShapes: [[2, 3, 4]],
    outDtypes: ["f32"],
  },
  {
    // The sum of bool is **the count of trues** (output i32 — the contract table's mapping)
    name: "sum bool [6,10] → i32 count",
    op: "sum",
    inputs: [fill([6, 10], MASK, "bool")],
    outShapes: [[6]],
    outDtypes: ["i32"],
    attrs: { dim: 1 },
  },
  {
    // Row length exceeds the workgroup size (256) → the path where one thread folds several
    // elements (bool version)
    name: "sum bool [3,700] → i32",
    op: "sum",
    inputs: [fill([3, 700], MASK, "bool")],
    outShapes: [[3]],
    outDtypes: ["i32"],
    attrs: { dim: 1 },
  },
  {
    // An asymmetric sequence on which the accumulation direction shows in the values (avoiding
    // arithmetic sequences, which can give the same values in reverse)
    name: "cumsum [4,7]",
    op: "cumsum",
    inputs: [fill([4, 7], SIGNED)],
    outShapes: [[4, 7]],
    attrs: { dim: 1 },
  },
  {
    name: "cumsum rank1 [10]",
    op: "cumsum",
    inputs: [fill([10], POSITIVE)],
    outShapes: [[10]],
    attrs: { dim: 0 },
  },
  {
    // One thread runs one row sequentially, so a long row also exercises the accumulation length
    name: "cumsum row length 300 rank3 [2,3,300]",
    op: "cumsum",
    inputs: [fill([2, 3, 300], SIGNED)],
    outShapes: [[2, 3, 300]],
    attrs: { dim: 2 },
  },
  {
    // Making the relative position embedding 4-D (recon §2) — the form that enabled f32 expand
    name: "expand f32 [1,4,1] → [1,4,3]",
    op: "expand",
    inputs: [fill([1, 4, 1], SIGNED)],
    outShapes: [[1, 4, 3]],
  },
  {
    name: "expand f32 rank increase [3] → [2,4,3]",
    op: "expand",
    inputs: [fill([3], POSITIVE)],
    outShapes: [[2, 4, 3]],
  },
];

/**
 * matmul (64×64 register tiles — src/kernels/gemm.ts).
 *
 * MUST: there are cases that **cross** the tile edge 64. Small shapes like `[7,5]×[5,3]` all
 * collapse into "less than one tile" and exercise none of the tile origin, the quad guard or the
 * partial K tile.
 * MUST: **both variants**, v4 (`k%4==0 && n%4==0`) and scalar, are exercised. With only one, a
 * silently broken fallback for partial shapes would go unnoticed.
 */
export const MATMUL_CASES: readonly OpCase[] = [
  {
    // Partial tiles of 16 (none of m/n/k aligns to a tile boundary)
    name: "matmul [7,5] × [5,3]",
    op: "matmul",
    inputs: [fill([7, 5], SIGNED), fill([5, 3], POSITIVE)],
    outShapes: [[7, 3]],
  },
  {
    name: "matmul [32,16] × [16,32]",
    op: "matmul",
    inputs: [fill([32, 16], SIGNED), fill([16, 32], POSITIVE)],
    outShapes: [[32, 32]],
  },
  {
    name: "matmul [1,64] × [64,1]",
    op: "matmul",
    inputs: [fill([1, 64], SIGNED), fill([64, 1], POSITIVE)],
    outShapes: [[1, 1]],
  },
  {
    // Tile boundaries on the v4 path: n=68 leaves 1 valid quad of 16 in the last tile (without the
    // quad guard it clobbers the neighbouring row), k=20 is a multiple of 4 but not of 16 (zero
    // fill of the partial K tile), m=65 spans 2 row tiles. m/n/k all differ.
    name: "matmul v4 tile boundaries [65,20] × [20,68]",
    op: "matmul",
    inputs: [fill([65, 20], SIGNED), fill([20, 68], POSITIVE)],
    outShapes: [[65, 68]],
  },
  {
    // Tile boundaries of the scalar variant (k=19 / n=23 are not multiples of 4, m=70 spans 2 row
    // tiles)
    name: "matmul scalar variant tile boundaries [70,19] × [19,23]",
    op: "matmul",
    inputs: [fill([70, 19], SIGNED), fill([19, 23], NONZERO)],
    outShapes: [[70, 23]],
  },
  {
    // MUST: the v4 decision looks at **both** k and n. If a check of only one passed through, the
    // vec4 binding and the actual byte count would disagree, giving wrong values without an
    // exception. Cases with only k / only n a multiple of 4 come in a pair.
    name: "matmul only k a multiple of 4 [66,20] × [20,19]",
    op: "matmul",
    inputs: [fill([66, 20], SIGNED), fill([20, 19], POSITIVE)],
    outShapes: [[66, 19]],
  },
  {
    name: "matmul only n a multiple of 4 [66,19] × [19,20]",
    op: "matmul",
    inputs: [fill([66, 19], SIGNED), fill([19, 20], POSITIVE)],
    outShapes: [[66, 20]],
  },
];

/**
 * bmm (rank-3 batched matmul).
 *
 * MUST: there is a case with **B / M / K / N all of different lengths** (the Pitfalls of
 * ACTIVE_DESIGN). Building the batch stride from the product of the wrong neighbouring dimensions
 * does not show in the numbers for square shapes or shapes with two axes of the same length.
 */
export const BMM_CASES: readonly OpCase[] = [
  {
    // Asymmetric shape shrunk from the real shape [16,T,64] × [16,64,T] (B=3 / M=5 / K=7 / N=2)
    name: "bmm asymmetric [3,5,7] × [3,7,2]",
    op: "bmm",
    inputs: [fill([3, 5, 7], SIGNED), fill([3, 7, 2], POSITIVE)],
    outShapes: [[3, 5, 2]],
  },
  {
    // Partial tiles of 16 over a batch of 2 (none of M / N / K aligns to a tile boundary)
    name: "bmm partial tiles [2,17,19] × [2,19,23]",
    op: "bmm",
    inputs: [fill([2, 17, 19], SIGNED), fill([2, 19, 23], NONZERO)],
    outShapes: [[2, 17, 23]],
  },
  {
    // K crossing one tile (the path where the reduction splits over several tiles) and a batch of 4
    name: "bmm reduction over several tiles [4,3,40] × [4,40,5]",
    op: "bmm",
    inputs: [fill([4, 3, 40], SIGNED), fill([4, 40, 5], POSITIVE)],
    outShapes: [[4, 3, 5]],
  },
  {
    name: "bmm batch of 1 [1,4,6] × [1,6,3]",
    op: "bmm",
    inputs: [fill([1, 4, 6], SIGNED), fill([1, 6, 3], POSITIVE)],
    outShapes: [[1, 4, 3]],
  },
  {
    // MUST: the batch base on the v4 path is **in quads** (`m * k4`). Building it in elements shows
    // only with batch ≥ 2, and may not show in the numbers if any two of B/M/K/N are equal.
    // M=68 gives 2 row tiles, K=20 a partial K tile, N=12 less than one column tile.
    name: "bmm v4 tile boundaries [3,68,20] × [3,20,12]",
    op: "bmm",
    inputs: [fill([3, 68, 20], SIGNED), fill([3, 20, 12], POSITIVE)],
    outShapes: [[3, 68, 12]],
  },
  {
    // The same trap on the scalar variant (a base in elements is correct here, but do all three go
    // through the base?)
    name: "bmm scalar variant tile boundaries [2,70,19] × [2,19,23]",
    op: "bmm",
    inputs: [fill([2, 70, 19], SIGNED), fill([2, 19, 23], NONZERO)],
    outShapes: [[2, 70, 23]],
  },
];

/**
 * Fused attention (ADR 0023). The contract is rank-4 head-first, 1 node = 3 dispatches.
 *
 * MUST: there is a case with **B / H / M / N / D all of different lengths**. The kernel folds B
 * and H into one batch axis, so with the real shape (B=1) swapped axes do not show in the values
 * (the detection limit of design recon §4.6).
 * MUST: there is a case with **D ≠ 128** (in practice only the DiT's 128 and the VAE's 384). D is
 * not expanded into the WGSL, so it adds no variant, but D decides which way `gemmUsesVec4` goes.
 * MUST: there are cases with **partial M / N**. Real models have M,N ∈ {512,1024,4096}, all
 * multiples of 64, so the unit tests are the only detector for the partial-tile path (the same
 * shape as the hole ADR 0022 recorded).
 */
const attentionCase = (
  name: string,
  [b, h, m, n, d]: readonly [number, number, number, number, number],
  options: {
    readonly query?: (index: number) => number;
    readonly key?: (index: number) => number;
    readonly scale?: number;
    /** Number of k / v heads (GQA — default: the classic H = Hkv. ADR 0067 decision 1). */
    readonly kvHeads?: number;
  } = {},
): OpCase => {
  const kvHeads = options.kvHeads ?? h;
  return {
    name,
    op: "attention",
    inputs: [
      fill([b, h, m, d], options.query ?? SIGNED),
      fill([b, kvHeads, n, d], options.key ?? POSITIVE),
      fill([b, kvHeads, n, d], NONZERO),
    ],
    outShapes: [[b, h, m, d]],
    // Default: the contract's half scale (torch's `√(1/√D)`).
    attrs: { scale: options.scale ?? Math.fround(Math.sqrt(1 / Math.sqrt(d))) },
  };
};

export const ATTENTION_CASES: readonly OpCase[] = [
  // B=2 / H=3 / M=5 / N=11 / D=7 — all 5 axes differ (swapped axes always show in the values)
  attentionCase("attention all axes differ [2,3,5,7] × [2,3,11,7]", [2, 3, 5, 11, 7]),
  // Partial tiles (M % 64 ≠ 0 / N % 64 ≠ 0) + D % 4 ≠ 0 for the scalar variant
  attentionCase("attention partial [3,1,17,13] × [3,1,19,13]", [3, 1, 17, 19, 13]),
  // Crossing row tiles on the v4 path (M=68 > 64)
  attentionCase("attention v4 crossing tiles [1,2,68,12]", [1, 2, 68, 20, 12]),
  // Mixed: ① on v4, ③ falls back to scalar (pins that the variant choice differs per stage)
  attentionCase("attention variant differs per stage [2,2,9,6]", [2, 2, 9, 8, 6]),
  // D=64 (conditioner scale) and D=384 (VAE decoder scale — the staged fusion also handles a D
  // that the flash form cannot take)
  attentionCase("attention D=64 [1,3,7,64]", [1, 3, 7, 5, 64]),
  attentionCase("attention D=384 [1,1,3,384]", [1, 1, 3, 5, 384]),
  // N of 1 (the reduction has 1 element — a degenerate form where softmax is the identity)
  attentionCase("attention N=1 [2,2,3,4]", [2, 2, 3, 1, 4]),
  {
    // MUST: a shape that turns red the moment the amax subtraction of the row statistics is lost.
    // With q large and negative and k all positive, S drops to around −200, and a naive softmax
    // would flush exp to 0 in f32, giving 0/0 = NaN.
    ...attentionCase(
      "attention large negative values [1,2,4,4] (naive form underflows)",
      [1, 2, 4, 6, 4],
      {
        query: () => -15,
        key: (i) => 6 + (i % 5) * 0.25,
      },
    ),
    name: "attention large negative values [1,2,4,4] (naive form underflows)",
  },
  // Explicit scale (SDPA's `scale` argument given — values other than the default 1/√D follow the
  // contract too)
  attentionCase("attention explicit scale [2,2,5,4]", [2, 2, 5, 7, 4], { scale: 0.25 }),
  // GQA / MQA (divisible broadcast — ADR 0067 decision 1). The job here is **the comparison with
  // the CPU reference**, a detector separate from the comparison with materialized repeat_kv
  // (tests/gpu_attention_gqa_test.ts) (the materializing side and the kernel share the same
  // mapping, so the reference implementation serves as an independent witness).
  // MUST: there is a GQA case with B ≥ 2 (the b term of `⌊(b·H + h)/r⌋ = b·Hkv + ⌊h/r⌋` is not
  // verified with B=1).
  attentionCase("attention GQA r=4 [2,8,5,4] × [2,2,7,4]", [2, 8, 5, 7, 4], { kvHeads: 2 }),
  attentionCase("attention MQA r=8 [1,8,3,8] × [1,1,5,8]", [1, 8, 3, 5, 8], { kvHeads: 1 }),
];

/** Deterministic indices within the last dimension D=9 of src (neither identity nor monotonic). */
const PICKS = (i: number): number => (i * 5 + 3) % 9;
/** Indices within the last dimension D=4. */
const PICKS4 = (i: number): number => (i * 3 + 1) % 4;

/**
 * gather (fixed to the last dimension). In practice src f32[16,T,512] / index i32[16,T,T].
 * MUST: the indices pick different columns per row (identity indices would not show a row-offset
 * bug).
 */
export const GATHER_CASES: readonly OpCase[] = [
  {
    name: "gather rank3 src [4,5,9] / index [4,5,6]",
    op: "gather",
    inputs: [fill([4, 5, 9], SIGNED), fill([4, 5, 6], PICKS, "i32")],
    outShapes: [[4, 5, 6]],
  },
  {
    // The output's last dimension is longer than src's (the same index may be picked repeatedly)
    name: "gather more columns src [3,4] / index [3,7]",
    op: "gather",
    inputs: [fill([3, 4], POSITIVE), fill([3, 7], PICKS4, "i32")],
    outShapes: [[3, 7]],
  },
  {
    name: "gather rank1 src [9] / index [5]",
    op: "gather",
    inputs: [fill([9], SIGNED), fill([5], PICKS, "i32")],
    outShapes: [[5]],
  },
  {
    // Too large for 1 workgroup (256 elements). **A reduced grid-stride dispatch is not exercised
    // here** (262144 elements = 1024 workgroups, inside the dispatch limit, so every workgroup
    // needed is granted) — tests/gpu_gridstride_test.ts exercises the reduced path directly by
    // limiting the dispatch count.
    name: "gather larger src [4096,9] / index [4096,64]",
    op: "gather",
    inputs: [fill([4096, 9], SIGNED), fill([4096, 64], PICKS, "i32")],
    outShapes: [[4096, 64]],
  },
];

export const REDUCE_CASES: readonly OpCase[] = [
  {
    name: "sum [6,10]",
    op: "sum",
    inputs: [fill([6, 10], SIGNED)],
    outShapes: [[6]],
    attrs: { dim: 1 },
  },
  {
    name: "amax [6,10]",
    op: "amax",
    inputs: [fill([6, 10], SIGNED)],
    outShapes: [[6]],
    attrs: { dim: 1 },
  },
  {
    name: "amin [6,10]",
    op: "amin",
    inputs: [fill([6, 10], SIGNED)],
    outShapes: [[6]],
    attrs: { dim: 1 },
  },
  // Row length exceeds the workgroup size (256) → the path where one thread folds several
  // elements
  {
    name: "sum [3,700]",
    op: "sum",
    inputs: [fill([3, 700], SIGNED)],
    outShapes: [[3]],
    attrs: { dim: 1 },
  },
  {
    name: "amax rank3 [2,3,7]",
    op: "amax",
    inputs: [fill([2, 3, 7], SIGNED)],
    outShapes: [[2, 3]],
    attrs: { dim: 2 },
  },
  // rank 1 → rank 0 (scalar output)
  {
    name: "sum [4] → scalar",
    op: "sum",
    inputs: [fill([4], SIGNED)],
    outShapes: [[]],
    attrs: { dim: 0 },
  },
  // Axes other than the last (the axis reduce variant — a different execution kernel)
  {
    name: "sum axis 1 [5,9,11]",
    op: "sum",
    inputs: [fill([5, 9, 11], SIGNED)],
    outShapes: [[5, 11]],
    attrs: { dim: 1 },
  },
  {
    name: "amax axis 0 [7,13]",
    op: "amax",
    inputs: [fill([7, 13], SIGNED)],
    outShapes: [[13]],
    attrs: { dim: 0 },
  },
  {
    // Reduction length > 256 (the path where the axis variant's slot scan loops twice). A middle
    // axis, so inner is not 1 either.
    name: "sum axis 1 with reduction length 300 [2,300,5]",
    op: "sum",
    inputs: [fill([2, 300, 5], SIGNED)],
    outShapes: [[2, 5]],
    attrs: { dim: 1 },
  },
  {
    // Axis sum of bool (the i32 accumulator + the cast to boolean are in the axis variant too)
    name: "sum bool axis 1 [3,13,5] → i32",
    op: "sum",
    inputs: [fill([3, 13, 5], MASK, "bool")],
    outShapes: [[3, 5]],
    outDtypes: ["i32"],
    attrs: { dim: 1 },
  },
];

/**
 * argmax (last dimension, rank kept, output i32 — ADR 0068 decision 2). The comparison is an
 * **exact** i32 match (indices have no rounding differences). The pinned behaviour on tie-breaks /
 * NaN / all −inf rows is checked by a dedicated check with literal expected values
 * (gpu_ops_test.ts), so this covers shapes and paths.
 */
export const ARGMAX_CASES: readonly OpCase[] = [
  {
    name: "argmax [6,10]",
    op: "argmax",
    inputs: [fill([6, 10], SIGNED)],
    outShapes: [[6, 1]],
    outDtypes: ["i32"],
  },
  {
    // Row length exceeds the workgroup size (256) → the path where one thread folds several
    // elements and the index is carried through a tree reduction across lanes. The maximum sits in
    // the second pass (index ≥ 256) **at a different position per row** (if the same index were
    // correct for every row, "reads another row" would not show in the values).
    name: "argmax [3,700] (maximum in the second pass of the scan loop, different position per row)",
    op: "argmax",
    inputs: [fill([3, 700], (i) => (i % 700 === 300 + Math.floor(i / 700) * 100 ? 99 : SIGNED(i)))],
    outShapes: [[3, 1]],
    outDtypes: ["i32"],
  },
  {
    name: "argmax rank3 [2,3,7]",
    op: "argmax",
    inputs: [fill([2, 3, 7], SIGNED)],
    outShapes: [[2, 3, 1]],
    outDtypes: ["i32"],
  },
  {
    // rank 1 → [1] (the reduce family drops to a scalar here — a shape where keeping the rank makes
    // a difference)
    name: "argmax [4] → [1]",
    op: "argmax",
    inputs: [fill([4], (i) => [-1.5, 0.25, 3.5, 3.5][i])],
    outShapes: [[1]],
    outDtypes: ["i32"],
  },
  {
    name: "argmax last dimension 1 [5,1]",
    op: "argmax",
    inputs: [fill([5, 1], SIGNED)],
    outShapes: [[5, 1]],
    outDtypes: ["i32"],
  },
  {
    // Two-phase form (row length ≥ 16,384 — "two-phase split" in src/kernels/argmax.ts). 5 spans of
    // 4,096 elements, with the maximum **in a different span per row** (row 0 → span 1, row 1 →
    // span 3, row 2 → end of span 4) (if every row had its maximum in span 0, a merge that "reads
    // only span 0" would not show in the values).
    name: "argmax [3,20000] (two-phase, maximum in a different span per row)",
    op: "argmax",
    inputs: [
      fill([3, 20000], (i) => {
        const row = Math.floor(i / 20000);
        const at = i % 20000;
        const peak = [4096 + 7, 3 * 4096 + 4000, 19999][row];
        return at === peak ? 99 : SIGNED(i);
      }),
    ],
    outShapes: [[3, 1]],
    outDtypes: ["i32"],
  },
  {
    // Span boundary of the two-phase form: equal maxima at the end of span 1 and the start of span
    // 2 → the smallest index (span 1 side).
    name: "argmax [1,16384] (two-phase, a tie across a span boundary)",
    op: "argmax",
    inputs: [fill([1, 16384], (i) => (i === 8191 || i === 8192 ? 42 : SIGNED(i)))],
    outShapes: [[1, 1]],
    outDtypes: ["i32"],
  },
];

/**
 * Input table for the tie-break / NaN / all −inf rows of argmax (the 3 MUSTs of ADR 0068 decision
 * 2).
 *
 * Meaning of the rows:
 * 0. Two equal values (3.0 at index 1,2) → **smallest index = 1**
 * 1. All elements equal → **index 0** (a "last maximum" implementation would give 3)
 * 2. All elements −inf → **index 0** (with a finite sentinel as identity there is no candidate and
 *    the sentinel leaks)
 * 3. Two NaNs (index 1,3) → **NaN is the maximum, smallest index = 1** (the side consistent within
 *    the family with the NaN propagation of `amax`)
 * 4. All elements NaN → **index 0**
 * 5. −inf mixed with equal maxima (2.0 at index 1,3) → **1**
 */
const ARGMAX_TIEBREAK_ROWS: readonly (readonly number[])[] = [
  [1, 3, 3, 2],
  [5, 5, 5, 5],
  [
    Number.NEGATIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
  ],
  [1, Number.NaN, 3, Number.NaN],
  [Number.NaN, Number.NaN, Number.NaN, Number.NaN],
  [Number.NEGATIVE_INFINITY, 2, Number.NEGATIVE_INFINITY, 2],
];

/**
 * The table above as an f32 `[6,4]` input.
 *
 * MUST: **both** the real-GPU side (gpu_ops_test.ts) and the CPU reference side
 * (reference_ops_test.ts) use this one table — with the literal on only one side, the check that
 * pins the direction of the CPU reference would disappear in environments without an adapter (a
 * GPU comparison turns green when both sides are wrong in the same direction).
 */
export const ARGMAX_TIEBREAK_INPUT = fill(
  [ARGMAX_TIEBREAK_ROWS.length, 4],
  (index) => ARGMAX_TIEBREAK_ROWS[Math.floor(index / 4)][index % 4],
);

/**
 * torch's measured values for `ARGMAX_TIEBREAK_INPUT` (measured 2026-08-17 / torch 2.13:
 * `torch.argmax(x, dim=-1, keepdim=True)`).
 */
export const ARGMAX_TIEBREAK_TORCH: readonly number[] = [1, 0, 0, 1, 0, 1];

/**
 * topk (last dimension, static k, **two outputs** — ADR 0068 decision 3). Values are compared with
 * f32 allclose and indices by exact i32 match, **both** against the CPU reference (`checkAll` goes
 * by the output count). The pinned behaviour on tie-breaks / NaN / all −inf rows is checked by a
 * dedicated check with literal expected values (gpu_ops_test.ts), so this covers shapes and paths
 * (lane-local top-k → tournament merge).
 */
export const TOPK_CASES: readonly OpCase[] = [
  {
    name: "topk [6,10] k=3",
    op: "topk",
    inputs: [fill([6, 10], SIGNED)],
    outShapes: [
      [6, 3],
      [6, 3],
    ],
    outDtypes: ["f32", "i32"],
    attrs: { k: 3 },
  },
  {
    // Row length exceeds one lane's share (700 / 32 = 21.9 passes) → the path where inserts into
    // the lane-local list happen many times. The top entries sit **at different positions per row**
    // (if the same indices were correct for every row, neither "reads another row" nor "the row
    // loop stops after one pass" would show in the values).
    name: "topk [3,700] k=4 (repeated inserts into the lane-local list, different positions per row)",
    op: "topk",
    inputs: [
      fill([3, 700], (i) => {
        const row = Math.floor(i / 700);
        const col = i % 700;
        // The top 4 of row r are 4 points 7 apart from 300+100r (across lanes too)
        const rank = (col - (300 + row * 100)) / 7;
        return Number.isInteger(rank) && rank >= 0 && rank < 4 ? 90 - rank : SIGNED(i);
      }),
    ],
    outShapes: [
      [3, 4],
      [3, 4],
    ],
    outDtypes: ["f32", "i32"],
    attrs: { k: 4 },
  },
  {
    // A shape where **all top k entries sit in the same lane** (every index a multiple of 32). The
    // path where one lane's cursor advances k times in a row; unless the tournament advances "only
    // the owner of the winning element", the same element comes out k times.
    name: "topk [2,320] k=5 (all top 5 in lane 0)",
    op: "topk",
    inputs: [
      fill([2, 320], (i) => {
        const col = i % 320;
        return col % 32 === 0 && col / 32 < 5 ? 100 - col / 32 : SIGNED(i);
      }),
    ],
    outShapes: [
      [2, 5],
      [2, 5],
    ],
    outDtypes: ["f32", "i32"],
    attrs: { k: 5 },
  },
  {
    name: "topk rank3 [2,3,7] k=2",
    op: "topk",
    inputs: [fill([2, 3, 7], SIGNED)],
    outShapes: [
      [2, 3, 2],
      [2, 3, 2],
    ],
    outDtypes: ["f32", "i32"],
    attrs: { k: 2 },
  },
  {
    // k = the last dimension (the top of the accepted range = a descending sort of the whole row).
    // Exercises the rank-1 form at the same time.
    name: "topk [4] k=4 (k = last dimension, rank 1)",
    op: "topk",
    inputs: [fill([4], (i) => [-1.5, 0.25, 3.5, 3.5][i])],
    outShapes: [[4], [4]],
    outDtypes: ["f32", "i32"],
    attrs: { k: 4 },
  },
  {
    // k > lane count (33 > 32) — a shape where each lane's block is 33 words.
    name: "topk [2,64] k=33 (k > lane count)",
    op: "topk",
    inputs: [fill([2, 64], (i) => SIGNED(i) + (i % 64) * 0.01)],
    outShapes: [
      [2, 33],
      [2, 33],
    ],
    outDtypes: ["f32", "i32"],
    attrs: { k: 33 },
  },
  {
    name: "topk last dimension 1 [5,1] k=1",
    op: "topk",
    inputs: [fill([5, 1], SIGNED)],
    outShapes: [
      [5, 1],
      [5, 1],
    ],
    outDtypes: ["f32", "i32"],
    attrs: { k: 1 },
  },
];

/**
 * Layout ops (ADR 0011). reshape is a buffer alias and issues no dispatch at all; permute / expand
 * run on one kernel family, a strided materializing copy.
 * The enabled dtypes are the ones seen in practice (f32 for permute, i32 / bool for expand).
 */
export const LAYOUT_CASES: readonly OpCase[] = [
  {
    name: "reshape [2,3,4] → [6,4] (alias)",
    op: "reshape",
    inputs: [fill([2, 3, 4], SIGNED)],
    outShapes: [[6, 4]],
  },
  {
    name: "reshape i32 [6] → [1,6,1] (axes inserted)",
    op: "reshape",
    inputs: [fill([6], INTEGERS, "i32")],
    outShapes: [[1, 6, 1]],
  },
  {
    name: "reshape bool [2,3] → [6] (axis removed)",
    op: "reshape",
    inputs: [fill([2, 3], MASK, "bool")],
    outShapes: [[6]],
  },
  {
    // Real shape [0,2,1,3] (head reshaping in attention)
    name: "permute rank4 [1,5,3,4] dims=[0,2,1,3]",
    op: "permute",
    inputs: [fill([1, 5, 3, 4], SIGNED)],
    outShapes: [[1, 3, 5, 4]],
    attrs: { dims: [0, 2, 1, 3] },
  },
  {
    // Real shape [0,2,1] (transposing scores)
    name: "permute rank3 [3,5,7] dims=[0,2,1]",
    op: "permute",
    inputs: [fill([3, 5, 7], POSITIVE)],
    outShapes: [[3, 7, 5]],
    attrs: { dims: [0, 2, 1] },
  },
  {
    name: "permute rank2 transpose [4,6] dims=[1,0]",
    op: "permute",
    inputs: [fill([4, 6], SIGNED)],
    outShapes: [[6, 4]],
    attrs: { dims: [1, 0] },
  },
  {
    // MUST: there is one permutation with a cycle of length 3 or more. The forms seen in practice
    // ([0,2,1,3] / [0,2,1] / [1,0]) are all involutions (their own inverse), so none of them can
    // detect a stride table built with the inverse permutation.
    name: "permute rank3 3-cycle [2,3,5] dims=[1,2,0]",
    op: "permute",
    inputs: [fill([2, 3, 5], SIGNED)],
    outShapes: [[3, 5, 2]],
    attrs: { dims: [1, 2, 0] },
  },
  {
    name: "permute rank4 4-cycle [2,3,4,5] dims=[1,2,3,0]",
    op: "permute",
    inputs: [fill([2, 3, 4, 5], POSITIVE)],
    outShapes: [[3, 4, 5, 2]],
    attrs: { dims: [1, 2, 3, 0] },
  },
  {
    // Real shape of the gather indices [1,T,T] → [16,T,T]
    name: "expand i32 [1,4,4] → [5,4,4] (leading axis replicated)",
    op: "expand",
    inputs: [fill([1, 4, 4], INTEGERS, "i32")],
    outShapes: [[5, 4, 4]],
  },
  {
    // bool mask of the conv path [1,T,1] → [1,T,C]
    name: "expand bool [1,6,1] → [1,6,9] (last axis replicated)",
    op: "expand",
    inputs: [fill([1, 6, 1], MASK, "bool")],
    outShapes: [[1, 6, 9]],
  },
  {
    name: "expand i32 rank increase [3] → [2,4,3]",
    op: "expand",
    inputs: [fill([3], INTEGERS, "i32")],
    outShapes: [[2, 4, 3]],
  },
];

/**
 * The second group of layout ops (ADR 0014) — slice / cat / pad / flip.
 *
 * MUST: there is always **a slice not starting at 0** and **a cat input that is not the first**
 * (with offset-0 shapes only, "forgetting / mixing up the offset" would not show in the values).
 * MUST: there is one flipped axis of **length 3 or more** (flipping length 2 cancels an off-by-one
 * symmetrically).
 * MUST: pad has a left/right asymmetric shape (with [w,w] only, swapping left and right would not
 * show in the values).
 */
export const LAYOUT2_CASES: readonly OpCase[] = [
  {
    // Same shape as stats[:, c:] of enc_p (the second half of the channels) — a middle-axis slice
    // not starting at 0
    name: "slice second half of a middle axis [1,6,5] dim=1 [3,6)",
    op: "slice",
    inputs: [fill([1, 6, 5], SIGNED)],
    outShapes: [[1, 3, 5]],
    attrs: { dim: 1, start: 3, end: 6 },
  },
  {
    // Same shape as bin_locations[..., :-1] of the spline (last dimension, dropping the end)
    name: "slice last dimension [4,7] dim=1 [1,6)",
    op: "slice",
    inputs: [fill([4, 7], POSITIVE)],
    outShapes: [[4, 5]],
    attrs: { dim: 1, start: 1, end: 6 },
  },
  {
    // Slicing the leading axis (the row stride does not change — only the offset matters)
    name: "slice leading axis [5,3] dim=0 [2,4)",
    op: "slice",
    inputs: [fill([5, 3], SIGNED)],
    outShapes: [[2, 3]],
    attrs: { dim: 0, start: 2, end: 4 },
  },
  {
    // A middle axis of rank 4 (a rank that does not hit the left-alignment of the strided params)
    name: "slice rank4 [2,5,3,4] dim=1 [1,4)",
    op: "slice",
    inputs: [fill([2, 5, 3, 4], SIGNED)],
    outShapes: [[2, 3, 3, 4]],
    attrs: { dim: 1, start: 1, end: 4 },
  },
  {
    // Same shape as cat([x0,x1], 1) of the coupling reverse (concatenating the channel axis — rows
    // alternate)
    name: "cat channel axis [1,3,5] + [1,2,5] dim=1",
    op: "cat",
    inputs: [fill([1, 3, 5], SIGNED), fill([1, 2, 5], POSITIVE)],
    outShapes: [[1, 5, 5]],
    attrs: { dim: 1 },
  },
  {
    // The cats of the spline (concatenating the last dimension — prepending a 1-element band)
    name: "cat last dimension [3,1] + [3,4] dim=1",
    op: "cat",
    inputs: [fill([3, 1], NONZERO), fill([3, 4], SIGNED)],
    outShapes: [[3, 5]],
    attrs: { dim: 1 },
  },
  {
    // 3 inputs (variable arity) — the middle input has a non-zero offset and length
    name: "cat 3 inputs [2,1]+[2,3]+[2,2] dim=1",
    op: "cat",
    inputs: [fill([2, 1], POSITIVE), fill([2, 3], SIGNED), fill([2, 2], NONZERO)],
    outShapes: [[2, 6]],
    attrs: { dim: 1 },
  },
  {
    // Concatenating the leading axis (rows simply stack up)
    name: "cat leading axis [2,4] + [3,4] dim=0",
    op: "cat",
    inputs: [fill([2, 4], SIGNED), fill([3, 4], POSITIVE)],
    outShapes: [[5, 4]],
    attrs: { dim: 0 },
  },
  {
    // Same shape as F.pad(p_attn, [w,w]) (w=4) on the value side of relative position
    name: "pad symmetric [1,2,3,5] [4,4]",
    op: "pad",
    inputs: [fill([1, 2, 3, 5], SIGNED)],
    outShapes: [[1, 2, 3, 13]],
    attrs: { left: 4, right: 4 },
  },
  {
    name: "pad asymmetric [3,4] [2,1]",
    op: "pad",
    inputs: [fill([3, 4], POSITIVE)],
    outShapes: [[3, 7]],
    attrs: { left: 2, right: 1 },
  },
  {
    name: "pad one side 0 [2,3] [0,3]",
    op: "pad",
    inputs: [fill([2, 3], SIGNED)],
    outShapes: [[2, 6]],
    attrs: { left: 0, right: 3 },
  },
  {
    // Shrunk from the Flip of flow (192ch) — a middle axis of length 5 (odd, so the centre stays
    // put)
    name: "flip middle axis [2,5,3] dim=1",
    op: "flip",
    inputs: [fill([2, 5, 3], SIGNED)],
    outShapes: [[2, 5, 3]],
    attrs: { dim: 1 },
  },
  {
    // Same shape as the Flip of sdp (2ch)
    name: "flip 2ch [1,2,6] dim=1",
    op: "flip",
    inputs: [fill([1, 2, 6], POSITIVE)],
    outShapes: [[1, 2, 6]],
    attrs: { dim: 1 },
  },
  {
    name: "flip last dimension [3,4] dim=1",
    op: "flip",
    inputs: [fill([3, 4], SIGNED)],
    outShapes: [[3, 4]],
    attrs: { dim: 1 },
  },
  {
    name: "flip leading axis rank4 [4,2,3,2] dim=0",
    op: "flip",
    inputs: [fill([4, 2, 3, 2], SIGNED)],
    outShapes: [[4, 2, 3, 2]],
    attrs: { dim: 0 },
  },
];

/** The fill value of masked_fill seen in practice (the lowest finite f32 — ADR 0012). */
const NEG_F32_MAX = -3.4028234663852886e38;
/**
 * The range where a naive softmax flushes exp to 0 in f32 (exercises why safe-softmax is needed).
 */
const HUGE_NEGATIVE = (i: number): number => -200 + (i % 7) * 3;
/** 0/1 bool mask (one true in every 3 elements). */
const MASKED = (i: number): number => (i % 3 === 0 ? 1 : 0);
/**
 * `[3,5]` scores for safe_softmax (ADR 0044). **Row 1 is all -inf** (the row on which torch's guard
 * fires), and rows 0 / 2 are normal rows mixed with -inf. Scores after the additive mask
 * `where(mask, 0, -inf)` have exactly this shape.
 */
const EMPTY_ROW_SCORES = (i: number): number => {
  const row = Math.floor(i / 5);
  if (row === 1) return -Infinity;
  return i % 4 === 3 ? -Infinity : SIGNED(i);
};

/**
 * The 10 fused ops (ADR 0012 / recon §5 + conv_transpose1d of ADR 0015 + rms_norm / conv2d of
 * ADR 0017 + safe_softmax of ADR 0044).
 *
 * MUST: each op has one case where "every axis has a different length". The m/n/k of linear and
 * the B/Cin/Cout/L/K of conv1d build indices from products, so with two axes of the same length a
 * swap does not show in the numbers.
 */
export const FUSED_CASES: readonly OpCase[] = [
  {
    // In practice [1,T,1024] × W[1024,1024]. This is a shrunk shape with m/n/k all different
    name: "linear rank2 [5,7] × W[3,7] + b[3]",
    op: "linear",
    inputs: [fill([5, 7], SIGNED), fill([3, 7], POSITIVE), fill([3], NONZERO)],
    outShapes: [[5, 3]],
  },
  {
    // Two leading dimensions (the path that flattens them into a GEMM)
    name: "linear rank3 [2,4,6] × W[5,6] + b[5]",
    op: "linear",
    inputs: [fill([2, 4, 6], SIGNED), fill([5, 6], NONZERO), fill([5], SIGNED)],
    outShapes: [[2, 4, 5]],
  },
  {
    // Partial tiles of 16 (none of m/n/k aligns to a tile boundary) + the reduction splits over
    // several tiles. n=19 is not a multiple of 4, so it takes the scalar variant, and the
    // shared-memory transpose (column quad = wc/4, component = wc%4) crosses quads within one tile
    // (W differs per column, so a swap shows in the numbers).
    name: "linear partial tiles [17,40] × W[19,40] + b[19]",
    op: "linear",
    inputs: [fill([17, 40], SIGNED), fill([19, 40], POSITIVE), fill([19], NONZERO)],
    outShapes: [[17, 19]],
  },
  {
    // Tile boundaries on the v4 path: n=68 (1 valid quad of 16 in the last tile; the shared
    // transpose crosses quads too) / k=20 (partial K tile) / m=65 (2 row tiles). m/n/k all differ.
    name: "linear v4 tile boundaries [65,20] × W[68,20] + b[68]",
    op: "linear",
    inputs: [fill([65, 20], SIGNED), fill([68, 20], POSITIVE), fill([68], SIGNED)],
    outShapes: [[65, 68]],
  },
  {
    // Crossing the tile edge 64 on the scalar variant (k=37 is a multiple of neither 4 nor 2)
    name: "linear scalar variant tile boundaries [70,37] × W[23,37] + b[23]",
    op: "linear",
    inputs: [fill([70, 37], SIGNED), fill([23, 37], NONZERO), fill([23], SIGNED)],
    outShapes: [[70, 23]],
  },
  {
    name: "layer_norm [4,10] affine",
    op: "layer_norm",
    inputs: [fill([4, 10], SIGNED), fill([10], POSITIVE), fill([10], SIGNED)],
    outShapes: [[4, 10]],
    attrs: { normalized_shape: [10], eps: 1e-5 },
  },
  {
    // Row length exceeds the workgroup size (256) → the path where one thread folds several
    // elements
    name: "layer_norm row length 300 rank3 [2,3,300]",
    op: "layer_norm",
    inputs: [fill([2, 3, 300], SIGNED), fill([300], NONZERO), fill([300], SIGNED)],
    outShapes: [[2, 3, 300]],
    attrs: { normalized_shape: [300], eps: 1e-7 },
  },
  {
    // The eps seen in practice (1e-07) and **rows with variance exactly 0** (constant per row).
    // Without eps this path gives 1/sqrt(0) = inf and a NaN output; the result is the bias itself.
    name: "layer_norm rows with variance 0 [3,8] eps=1e-7",
    op: "layer_norm",
    inputs: [fill([3, 8], (i) => Math.floor(i / 8) + 1), fill([8], POSITIVE), fill([8], SIGNED)],
    outShapes: [[3, 8]],
    attrs: { normalized_shape: [8], eps: 1e-7 },
  },
  {
    // ADR 0017: rms_norm does not subtract the mean (arity 2). SIGNED spans 0, so an implementation
    // copied from layer_norm (subtracting the mean) always disagrees with the CPU reference.
    name: "rms_norm [4,10] weight",
    op: "rms_norm",
    inputs: [fill([4, 10], SIGNED), fill([10], POSITIVE)],
    outShapes: [[4, 10]],
    attrs: { eps: 1e-6 },
  },
  {
    // Row length exceeds the workgroup size (256) → the path where one thread folds several
    // elements
    name: "rms_norm row length 300 rank3 [2,3,300]",
    op: "rms_norm",
    inputs: [fill([2, 3, 300], SIGNED), fill([300], NONZERO)],
    outShapes: [[2, 3, 300]],
    attrs: { eps: 1e-6 },
  },
  {
    // **Rows of all zeros** (sum of squares 0) — with eps outside the square root, 1/sqrt(0) = inf
    // turns into NaN. A correct implementation returns exactly 0 (the same safeguard as the
    // "rows with variance 0" of layer_norm).
    name: "rms_norm rows of all zeros [3,8] eps=1e-6",
    op: "rms_norm",
    inputs: [fill([3, 8], () => 0), fill([8], POSITIVE)],
    outShapes: [[3, 8]],
    attrs: { eps: 1e-6 },
  },
  {
    name: "softmax last dimension [4,9]",
    op: "softmax",
    inputs: [fill([4, 9], SIGNED)],
    outShapes: [[4, 9]],
    attrs: { dim: 1 },
  },
  {
    // MUST: the range where the naive form flushes exp(-200) to 0 in f32, giving 0/0 = NaN.
    // This turns red the moment safe-softmax (the amax subtraction) is lost.
    name: "softmax large negative values [3,11] (naive form underflows)",
    op: "softmax",
    inputs: [fill([3, 11], HUGE_NEGATIVE)],
    outShapes: [[3, 11]],
    attrs: { dim: 1 },
  },
  {
    // Rows whose elements are all equal (the same shape as a row fully masked by masked_fill) —
    // a uniform distribution
    name: "softmax all elements equal [2,6]",
    op: "softmax",
    inputs: [fill([2, 6], () => NEG_F32_MAX)],
    outShapes: [[2, 6]],
    attrs: { dim: 1 },
  },
  {
    name: "softmax row length 300 rank3 [2,3,300]",
    op: "softmax",
    inputs: [fill([2, 3, 300], SIGNED)],
    outShapes: [[2, 3, 300]],
    attrs: { dim: 2 },
  },
  {
    // safe_softmax (ADR 0044) — on input without -inf it gives the same values as plain softmax.
    name: "safe_softmax last dimension [4,9] (no -inf)",
    op: "safe_softmax",
    inputs: [fill([4, 9], SIGNED)],
    outShapes: [[4, 9]],
    attrs: { dim: 1 },
  },
  {
    // MUST: include **rows of all -inf**. They are the very reason this op exists; plain softmax
    // gives 0/0 = NaN (allclose fails on NaN, so losing it always turns red).
    // Row 1 is all -inf / rows 0 and 2 are only partly -inf.
    name: "safe_softmax all -inf rows [3,5]",
    op: "safe_softmax",
    inputs: [fill([3, 5], EMPTY_ROW_SCORES)],
    outShapes: [[3, 5]],
    attrs: { dim: 1 },
  },
  {
    // The empty-row decision works with row lengths beyond the workgroup size (256) too (the
    // identity of ① goes through both the grid-stride scan and the tree reduction). Row 1 is all
    // -inf.
    name: "safe_softmax row length 300 rank3 [2,3,300] (with an all -inf row)",
    op: "safe_softmax",
    inputs: [fill([2, 3, 300], (i) => (Math.floor(i / 300) === 1 ? -Infinity : SIGNED(i)))],
    outShapes: [[2, 3, 300]],
    attrs: { dim: 2 },
  },
  {
    // The fill value of masked_fill (-F32_MAX) is **finite**, so the row is not empty — it gives
    // the same uniform distribution as plain softmax (an implementation mixing up -inf and
    // -F32_MAX fails here).
    name: "safe_softmax all -F32_MAX rows [2,6] (not empty rows)",
    op: "safe_softmax",
    inputs: [fill([2, 6], () => NEG_F32_MAX)],
    outShapes: [[2, 6]],
    attrs: { dim: 1 },
  },
  {
    // In practice weight f32[22012,1024] × index i32[1,T]. A shrunk shape with V/H/index count all
    // different
    name: "embedding weight [7,4] × index [2,3]",
    op: "embedding",
    inputs: [fill([7, 4], SIGNED), fill([2, 3], (i) => (i * 3 + 1) % 7, "i32")],
    outShapes: [[2, 3, 4]],
    attrs: { padding_idx: -1 },
  },
  {
    // padding_idx has no effect on forward (accepted and inert — ADR 0012). Exercised with indices
    // that always pick 0; an implementation that zeroes the padding row disagrees with the CPU
    // reference.
    name: "embedding padding_idx=0 has no effect on forward",
    op: "embedding",
    inputs: [fill([5, 3], POSITIVE), fill([4], (i) => i % 5, "i32")],
    outShapes: [[4, 3]],
    attrs: { padding_idx: 0 },
  },
  {
    // Shrunk from the real shape mask [1,1,T,T] × x [1,16,T,T] (right-aligned broadcast)
    name: "masked_fill rank4 broadcast x[1,3,4,5] mask[1,1,4,5]",
    op: "masked_fill",
    inputs: [fill([1, 3, 4, 5], SIGNED), fill([1, 1, 4, 5], MASKED, "bool")],
    outShapes: [[1, 3, 4, 5]],
    attrs: { value: NEG_F32_MAX },
  },
  {
    // Real shape mask [1,T,1024] × x [1,T,1024] (same shape, fill value 0)
    name: "masked_fill same shape x[2,3,4] mask[2,3,4] value=0",
    op: "masked_fill",
    inputs: [fill([2, 3, 4], SIGNED), fill([2, 3, 4], MASKED, "bool")],
    outShapes: [[2, 3, 4]],
    attrs: { value: 0 },
  },
  {
    // Right-aligned broadcast across ranks (mask has a lower rank than x)
    name: "masked_fill different ranks x[3,4,5] mask[5]",
    op: "masked_fill",
    inputs: [fill([3, 4, 5], SIGNED), fill([5], MASKED, "bool")],
    outShapes: [[3, 4, 5]],
    attrs: { value: -1.5 },
  },
  {
    // The real shape (kernel 3 / stride 1 / padding 1) with B/Cin/Cout/L all different
    name: "conv1d [2,3,9] * W[4,3,3] stride=1 padding=1",
    op: "conv1d",
    inputs: [fill([2, 3, 9], SIGNED), fill([4, 3, 3], POSITIVE), fill([4], NONZERO)],
    outShapes: [[2, 4, 9]],
    attrs: { stride: 1, padding: 1, dilation: 1, groups: 1 },
  },
  {
    // No padding (the output shrinks) + stride 2 (exercises the params path)
    name: "conv1d [1,2,11] * W[3,2,4] stride=2 padding=0",
    op: "conv1d",
    inputs: [fill([1, 2, 11], SIGNED), fill([3, 2, 4], NONZERO), fill([3], SIGNED)],
    outShapes: [[1, 3, 4]],
    attrs: { stride: 2, padding: 0, dilation: 1, groups: 1 },
  },
  {
    // Padding longer than the kernel (some outputs at both ends see only the padding region)
    name: "conv1d [1,1,3] * W[2,1,3] stride=1 padding=2",
    op: "conv1d",
    inputs: [fill([1, 1, 3], SIGNED), fill([2, 1, 3], POSITIVE), fill([2], NONZERO)],
    outShapes: [[1, 2, 5]],
    attrs: { stride: 1, padding: 2, dilation: 1, groups: 1 },
  },
  // ---- conv family extensions (ADR 0015) -----------------------------------
  {
    // Shrunk from the DDSConv of sdp (depthwise groups=C, dilation 3, k=5). groups=Cin=Cout, so the
    // weight's second axis is 1 — reading across groups shows in the values as "values of another
    // channel mixed in".
    name: "conv1d depthwise [1,6,17] * W[6,1,5] dilation=3 groups=6",
    op: "conv1d",
    inputs: [fill([1, 6, 17], SIGNED), fill([6, 1, 5], POSITIVE), fill([6], NONZERO)],
    outShapes: [[1, 6, 17]],
    attrs: { stride: 1, padding: 6, dilation: 3, groups: 6 },
  },
  {
    // depthwise with dilation 9 (the 3rd layer of DDSConv). Padding grows to d·(K−1)/2 = 18 too
    name: "conv1d depthwise [1,4,13] * W[4,1,5] dilation=9 groups=4",
    op: "conv1d",
    inputs: [fill([1, 4, 13], SIGNED), fill([4, 1, 5], NONZERO), fill([4], SIGNED)],
    outShapes: [[1, 4, 13]],
    attrs: { stride: 1, padding: 18, dilation: 9, groups: 4 },
  },
  {
    // Intermediate groups (1 < g < C) — both Cin and Cout have 2ch or more per group, so dropping
    // the band's start offset shows in the values. An asymmetric shape with Cin ≠ Cout.
    name: "conv1d grouped [1,6,11] * W[9,2,3] groups=3",
    op: "conv1d",
    inputs: [fill([1, 6, 11], SIGNED), fill([9, 2, 3], POSITIVE), fill([9], NONZERO)],
    outShapes: [[1, 9, 11]],
    attrs: { stride: 1, padding: 1, dilation: 1, groups: 3 },
  },
  {
    // Shrunk from ResBlock1 of dec (dilation 5, k=3, full channels)
    name: "conv1d dilated [1,3,15] * W[5,3,3] dilation=5",
    op: "conv1d",
    inputs: [fill([1, 3, 15], SIGNED), fill([5, 3, 3], NONZERO), fill([5], SIGNED)],
    outShapes: [[1, 5, 15]],
    attrs: { stride: 1, padding: 5, dilation: 5, groups: 1 },
  },
  // ---- conv2d（ADR 0017）------------------------------------------------------
  // MUST: implementation mix-ups do not show in the numbers on symmetric shapes. The following 5
  // cases cover all of (a) Cin/Cout both ≥ 2 and different (b) Kh ≠ Kw (c) H/W-asymmetric
  // stride/padding (d) dilation > 1 (e) groups > 1 (including depthwise).
  {
    // The real shape of the VAE's plain conv2d (k3 / pad1 / same), with B/Cin/Cout/H/W all
    // different
    name: "conv2d [2,3,7,9] * W[5,3,3,3] stride=1 padding=1",
    op: "conv2d",
    inputs: [fill([2, 3, 7, 9], SIGNED), fill([5, 3, 3, 3], POSITIVE), fill([5], NONZERO)],
    outShapes: [[2, 5, 7, 9]],
    attrs: { stride: [1, 1], padding: [1, 1], dilation: [1, 1], groups: 1 },
  },
  {
    // Kh ≠ Kw and H/W-asymmetric stride / padding
    // H: (9 + 2 − 2)/2 + 1 = 5 / W: (11 + 0 − 4)/3 + 1 = 3
    name: "conv2d [1,2,9,11] * W[4,2,2,4] stride=[2,3] padding=[1,0]",
    op: "conv2d",
    inputs: [fill([1, 2, 9, 11], SIGNED), fill([4, 2, 2, 4], NONZERO), fill([4], SIGNED)],
    outShapes: [[1, 4, 5, 3]],
    attrs: { stride: [2, 3], padding: [1, 0], dilation: [1, 1], groups: 1 },
  },
  {
    // H/W-asymmetric dilation too (H: 12 − 3·2 = 6 / W: 10 − 2·1 = 8)
    name: "conv2d [1,2,12,10] * W[3,2,3,2] dilation=[3,2]",
    op: "conv2d",
    inputs: [fill([1, 2, 12, 10], SIGNED), fill([3, 2, 3, 2], POSITIVE), fill([3], NONZERO)],
    outShapes: [[1, 3, 6, 8]],
    attrs: { stride: [1, 1], padding: [0, 0], dilation: [3, 2], groups: 1 },
  },
  {
    // depthwise (groups = Cin = Cout → the weight's second axis is 1). Reading across groups
    // always shows in the values as "values of another channel mixed in"
    name: "conv2d depthwise [1,6,5,7] * W[6,1,3,3] groups=6",
    op: "conv2d",
    inputs: [fill([1, 6, 5, 7], SIGNED), fill([6, 1, 3, 3], POSITIVE), fill([6], NONZERO)],
    outShapes: [[1, 6, 5, 7]],
    attrs: { stride: [1, 1], padding: [1, 1], dilation: [1, 1], groups: 6 },
  },
  {
    // Intermediate groups (1 < g < C) with Cin ≠ Cout, Kh ≠ Kw — one case that catches both a
    // dropped band start offset and swapped kernel axes
    name: "conv2d grouped [1,6,6,8] * W[9,2,3,1] groups=3",
    op: "conv2d",
    inputs: [fill([1, 6, 6, 8], SIGNED), fill([9, 2, 3, 1], NONZERO), fill([9], SIGNED)],
    outShapes: [[1, 9, 4, 8]],
    attrs: { stride: [1, 1], padding: [0, 0], dilation: [1, 1], groups: 3 },
  },
  {
    // **Tile boundaries** of the implicit GEMM (groups == 1 — ADR 0024): m = 70 gives 2 row tiles,
    // n = Hout·Wout = 72 2 column tiles, K = 8·3·3 = 72 a partial K tile (16·4 + 8).
    // kFlat % 4 == 0, Wout % 4 == 0 and stride_w == 1, so it takes the v4 variant.
    name: "conv2d igemm v4 tile boundaries [1,8,9,8] * W[70,8,3,3] padding=1",
    op: "conv2d",
    inputs: [fill([1, 8, 9, 8], SIGNED), fill([70, 8, 3, 3], POSITIVE), fill([70], NONZERO)],
    outShapes: [[1, 70, 9, 8]],
    attrs: { stride: [1, 1], padding: [1, 1], dilation: [1, 1], groups: 1 },
  },
  {
    // **Detector for a wrong v4 decision** (MUST ② of ADR 0024): Hout = Wout = 2 gives N = 4, so a
    // decision on `N % 4` would pick v4, but column quads cross output rows and give wrong values.
    // kFlat = 4·1·5 = 20 and stride_w = 1 meet the other v4 conditions, so `Wout % 4` alone
    // decides. The real shapes (Wout always a multiple of 4) would never expose it.
    name: "conv2d igemm Hout=Wout=2 (N%4==0 but Wout%4≠0) [1,4,2,6] * W[3,4,1,5]",
    op: "conv2d",
    inputs: [fill([1, 4, 2, 6], SIGNED), fill([3, 4, 1, 5], POSITIVE), fill([3], NONZERO)],
    outShapes: [[1, 3, 2, 2]],
    attrs: { stride: [1, 1], padding: [0, 0], dilation: [1, 1], groups: 1 },
  },
  {
    // Padding wider than the kernel span (some outputs at both ends see only the padding region)
    name: "conv2d [1,1,2,3] * W[2,1,3,1] padding=[2,0]",
    op: "conv2d",
    inputs: [fill([1, 1, 2, 3], SIGNED), fill([2, 1, 3, 1], POSITIVE), fill([2], NONZERO)],
    outShapes: [[1, 2, 4, 3]],
    attrs: { stride: [1, 1], padding: [2, 0], dilation: [1, 1], groups: 1 },
  },
  {
    // The last 2 ups of dec (up 2 / k 2 / pad 0). Asymmetric channels pin the weight transpose
    name: "conv_transpose1d [1,5,7] * W[5,3,2] stride=2 padding=0",
    op: "conv_transpose1d",
    inputs: [fill([1, 5, 7], SIGNED), fill([5, 3, 2], POSITIVE), fill([3], NONZERO)],
    outShapes: [[1, 3, 14]],
    attrs: { stride: 2, padding: 0 },
  },
  {
    // Shrunk from the first up of dec (up 8 / k 16 / pad 4) — several k contribute to one output
    name: "conv_transpose1d [1,3,5] * W[3,2,16] stride=8 padding=4",
    op: "conv_transpose1d",
    inputs: [fill([1, 3, 5], SIGNED), fill([3, 2, 16], NONZERO), fill([2], SIGNED)],
    outShapes: [[1, 2, 40]],
    attrs: { stride: 8, padding: 4 },
  },
  {
    // k = stride (pad 0) — the boundary shape where exactly one k contributes
    name: "conv_transpose1d [1,2,6] * W[2,4,4] stride=4 padding=0",
    op: "conv_transpose1d",
    inputs: [fill([1, 2, 6], SIGNED), fill([2, 4, 4], POSITIVE), fill([4], NONZERO)],
    outShapes: [[1, 4, 24]],
    attrs: { stride: 4, padding: 0 },
  },
];

/**
 * group_norm (ADR 0138 decision 7): several samples and groups (a group indexed without its
 * sample, or a channel without its group, reads another group's statistics or affine), a rank-3
 * input whose groups span two partial chunks (2 × 600 = 1,200 > 1,024 elements), and a rank-2
 * input (no trailing axis). The four-point check and the large group are
 * `gpu_group_norm_test.ts`.
 */
export const GROUP_NORM_CASES: readonly OpCase[] = [
  {
    name: "group_norm [2,6,3,5] 3 groups affine",
    op: "group_norm",
    inputs: [fill([2, 6, 3, 5], SIGNED), fill([6], POSITIVE), fill([6], SIGNED)],
    outShapes: [[2, 6, 3, 5]],
    attrs: { num_groups: 3, eps: 1e-6 },
  },
  {
    name: "group_norm [1,4,600] 2 groups spanning two chunks",
    op: "group_norm",
    inputs: [fill([1, 4, 600], SIGNED), fill([4], NONZERO), fill([4], SIGNED)],
    outShapes: [[1, 4, 600]],
    attrs: { num_groups: 2, eps: 1e-5 },
  },
  {
    name: "group_norm [5,8] rank 2, 4 groups",
    op: "group_norm",
    inputs: [fill([5, 8], SIGNED), fill([8], POSITIVE), fill([8], SIGNED)],
    outShapes: [[5, 8]],
    attrs: { num_groups: 4, eps: 1e-5 },
  },
];

// Bilinear resample (a layer-1 op). It is a new primitive without an A/B oracle, so the comparison
// with the CPU reference here and the comparison of the golden with torch are the main checks.
// MUST: cover upscaling / downscaling / edge clamping / an axis with scale 0, and with B·C > 1 also
// **how the plane index is folded** (N·C collapsed into one).
export const UPSAMPLE_CASES: readonly OpCase[] = [
  {
    // Upscaling by a non-integer factor (H 3/6, W 4/8). H≠W and Hout≠Wout, so swapped axes show in
    // the values
    name: "upsample_bilinear2d upscale [2,3,4,5] → 7×9",
    op: "upsample_bilinear2d",
    inputs: [fill([2, 3, 4, 5], SIGNED)],
    outShapes: [[2, 3, 7, 9]],
    attrs: { output_size: [7, 9] },
  },
  {
    // Downscaling (the same shape as forward_enc / cxt of BiRefNet) — reading only 2 taps is
    // torch's behaviour
    name: "upsample_bilinear2d downscale [1,3,8,10] → 3×4",
    op: "upsample_bilinear2d",
    inputs: [fill([1, 3, 8, 10], POSITIVE)],
    outShapes: [[1, 3, 3, 4]],
    attrs: { output_size: [3, 4] },
  },
  {
    // Input height 1 → the H scale is 0 (the end special case `index1 = index0` fires for every
    // output). Only W is downscaled, so the two axes take different branches at once
    name: "upsample_bilinear2d height 1 [1,2,1,6] → 4×3",
    op: "upsample_bilinear2d",
    inputs: [fill([1, 2, 1, 6], SIGNED)],
    outShapes: [[1, 2, 4, 3]],
    attrs: { output_size: [4, 3] },
  },
  {
    // 1×1 → N×M (the GAP branch of ASPP) — scale 0 on both axes, effectively a broadcast
    name: "upsample_bilinear2d 1×1 → 5×4 [2,3,1,1]",
    op: "upsample_bilinear2d",
    inputs: [fill([2, 3, 1, 1], NONZERO)],
    outShapes: [[2, 3, 5, 4]],
    attrs: { output_size: [5, 4] },
  },
  {
    // Output 1×1 (the out == 1 special case of the scale = 0). torch reads only the top-left point
    name: "upsample_bilinear2d 6×7 → 1×1 [1,2,6,7]",
    op: "upsample_bilinear2d",
    inputs: [fill([1, 2, 6, 7], SIGNED)],
    outShapes: [[1, 2, 1, 1]],
    attrs: { output_size: [1, 1] },
  },
  {
    // Same size (in == out) — every λ is 0, an identity copy (the scale is exactly 1.0)
    name: "upsample_bilinear2d same size [1,2,5,4] → 5×4",
    op: "upsample_bilinear2d",
    inputs: [fill([1, 2, 5, 4], SIGNED)],
    outShapes: [[1, 2, 5, 4]],
    attrs: { output_size: [5, 4] },
  },
];

/**
 * **Shapes whose scale is not exactly representable in f32** (the 6 cases of
 * {@link UPSAMPLE_CASES} all have exact scales in {0, 0.5, 1, 2.5, 3, 3.5}, so λ collapses to 0 / 1
 * and no rounding sequence ever arises). H: in=4 → out=6 gives `scale = fl(3/5) = 0.6` / W: in=5 →
 * out=4 gives `scale = fl(4/3)`; H and W take different inexact scales, so swapped axes show at the
 * same time.
 *
 * MUST: **not mixed into** {@link UPSAMPLE_CASES}. Those also feed the bit-identity check of
 * `BIT_IDENTICAL_OPS` in {@link "./op-tolerance.ts"}, which inexact scales cannot meet by
 * definition — the GPU rounds every stage in f32, while the CPU reference keeps `top` / `bottom` in
 * f64 and rounds only once at the end, so on this shape 11 of 48 elements differ by 1 ulp
 * (measured). The comparison band is the measured table of `upsample_bilinear2d` (SMALL_REDUCE),
 * which it meets.
 */
export const UPSAMPLE_FRACTIONAL_CASES: readonly OpCase[] = [
  {
    name: "upsample_bilinear2d inexact scale [1,2,4,5] → 6×4",
    op: "upsample_bilinear2d",
    inputs: [fill([1, 2, 4, 5], SIGNED)],
    outShapes: [[1, 2, 6, 4]],
    attrs: { output_size: [6, 4] },
  },
];

/** offset of deform (non-integers up to ±2.5 — reaching outside the input plane). */
const OFFSET = (i: number): number => ((i % 11) - 5) * 0.5;
/** modulator of deform (the same range [0,2] as BiRefNet's `2·sigmoid(...)`). */
const MODULATOR = (i: number): number => (i % 9) * 0.25;

// DCNv2 (a layer-1' op — ADR 0055). It is a new primitive without an A/B oracle, so the comparison
// with the CPU reference here and the comparison of the golden with torch are the main checks (the
// bit match of the degenerate case is in gpu_deform_conv2d_test.ts).
// MUST: cover the two forms of k (the 4 branches of BiRefNet have k ∈ {1,1,3,7}), B > 1,
// Cin ≠ Cout, Kh ≠ Kw and padding with H≠W.
export const DEFORM_CASES: readonly OpCase[] = [
  {
    // Kh ≠ Kw / padding with H≠W (the y/x order of offset and the weight axes matter at once)
    name: "deform_conv2d [1,2,4,5] * W[3,2,3,2] padding=[1,0]",
    op: "deform_conv2d",
    inputs: [
      fill([1, 2, 4, 5], SIGNED),
      fill([3, 2, 3, 2], POSITIVE),
      fill([1, 12, 4, 4], OFFSET),
      fill([1, 6, 4, 4], MODULATOR),
      fill([3], SIGNED),
    ],
    outShapes: [[1, 3, 4, 4]],
    attrs: { padding: [1, 0] },
  },
  {
    // k=1 (the aspp1 branch of ASPP) with **batch 2** (the batch stride of offset / mask matters)
    name: "deform_conv2d k=1 batch 2 [2,3,3,4] * W[4,3,1,1]",
    op: "deform_conv2d",
    inputs: [
      fill([2, 3, 3, 4], SIGNED),
      fill([4, 3, 1, 1], POSITIVE),
      fill([2, 2, 3, 4], OFFSET),
      fill([2, 1, 3, 4], MODULATOR),
      fill([4], NONZERO),
    ],
    outShapes: [[2, 4, 3, 4]],
    attrs: { padding: [0, 0] },
  },
  {
    // Same-size output (BiRefNet's real shape — k=3 / padding 1) with a longer reduction
    name: "deform_conv2d [1,4,6,7] * W[5,4,3,3] padding=[1,1]",
    op: "deform_conv2d",
    inputs: [
      fill([1, 4, 6, 7], SIGNED),
      fill([5, 4, 3, 3], POSITIVE),
      fill([1, 18, 6, 7], OFFSET),
      fill([1, 9, 6, 7], MODULATOR),
      fill([5], SIGNED),
    ],
    outShapes: [[1, 5, 6, 7]],
    attrs: { padding: [1, 1] },
  },
];

// GRU hidden-side scan (a layer-2 op — ADR 0056). The bit-identical A/B oracle lives in
// gpu_gru_scan_parity_test.ts, so this is **an independent comparison with the CPU reference**
// (checking the reduction indices and the scan direction with an implementation that shares
// nothing with the GEMM or the WGSL).
// MUST: the reverse direction runs on the same sequences (with the forward direction only, an
// identity mapping of `t` would be green).

/**
 * Hidden-side weights of the GRU (kept small enough that the gates do not saturate over a reduction
 * of H terms).
 */
const SMALL_WEIGHT = (i: number): number => ((i % 19) - 9) * 0.031;

export const GRU_SCAN_CASES: readonly OpCase[] = ["gru_scan", "gru_scan_reverse"].flatMap((op) => [
  {
    // Smallest shape (T = 2 = the minimum at which the state is carried once); N ≠ H, so swapped
    // axes show in the values
    name: `${op} [4,1,9] h0[1,3]`,
    op,
    inputs: [
      fill([4, 1, 9], SIGNED),
      fill([1, 3], NONZERO),
      fill([9, 3], SMALL_WEIGHT),
      fill([9], SIGNED),
    ],
    outShapes: [[4, 1, 3]],
  },
  {
    // Batch > 1 (the boundary of 1 workgroup = 1 batch element; mixed-up states show in the values)
    name: `${op} batch 3 [3,3,15] h0[3,5]`,
    op,
    inputs: [
      fill([3, 3, 15], SIGNED),
      fill([3, 5], NONZERO),
      fill([15, 5], SMALL_WEIGHT),
      fill([15], SIGNED),
    ],
    outShapes: [[3, 3, 5]],
  },
]);

// Boundary cases. MUST: "larger inputs" do not exercise a **reduced** grid-stride dispatch — every
// workgroup the element count needs is granted, so a bug that makes `stride` a constant passes
// green here. The reduced dispatch itself is verified directly by tests/gpu_gridstride_test.ts,
// which limits the dispatch count. What this covers is "index computation does not break at sizes
// beyond one workgroup".
export const BOUNDARY_CASES: readonly OpCase[] = [
  {
    // 4096 workgroups (inside the dispatch limit) — a size covered entirely at 1 element per thread
    name: "relu one larger tensor [1048576]",
    op: "relu",
    inputs: [fill([1 << 20], SIGNED)],
    outShapes: [[1 << 20]],
  },
  {
    // The strided copy at the same size (output 1048576 elements = 4096 workgroups)
    name: "permute larger [512,2048] dims=[1,0]",
    op: "permute",
    inputs: [fill([512, 2048], SIGNED)],
    outShapes: [[2048, 512]],
    attrs: { dims: [1, 0] },
  },
  {
    // Row count > maxComputeWorkgroupsPerDimension (default 65535) — this one case alone hits a
    // reduced dispatch with a real shape (gridStrideWorkgroups caps at the limit, and one workgroup
    // loops over several rows)
    name: "sum many rows [70000,4]",
    op: "sum",
    inputs: [fill([70000, 4], SIGNED)],
    outShapes: [[70000]],
    attrs: { dim: 1 },
  },
];
