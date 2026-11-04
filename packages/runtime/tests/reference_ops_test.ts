import { test } from "vite-plus/test";
import { assertAlmostEquals, assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import {
  allclose,
  AllcloseError,
  compareTensors,
  DEFAULT_TOLERANCE,
  EXACT_TOLERANCE,
  formatAllclose,
} from "../src/reference/allclose.ts";
import {
  applyReferenceOp,
  applyReferenceOpOutputs,
  referenceAttention,
  referenceBinary,
  referenceBmm,
  referenceCast,
  referenceCat,
  referenceConv1d,
  referenceConv2d,
  referenceConv3d,
  referenceConvTranspose1d,
  referenceCumsum,
  referenceDeformConv2d,
  referenceEmbedding,
  referenceExpand,
  referenceFlip,
  referenceGather,
  referenceGroupNorm,
  referenceGruScan,
  referenceLayerNorm,
  referenceLinear,
  referenceMaskedFill,
  referenceMatmul,
  ReferenceOpError,
  referencePad,
  referencePermute,
  referenceReshape,
  referenceRmsNorm,
  referenceRowReduce,
  referenceSafeSoftmax,
  referenceSlice,
  referenceSoftmax,
  referenceSymPrefixSlice,
  referenceUnary,
  referenceUpsampleBilinear2d,
  referenceWhere,
  type RefTensor,
  refTensor,
} from "../src/reference/ops.ts";
import { OpContractError, type UnaryOpName } from "../src/ops.ts";
import { ARGMAX_TIEBREAK_INPUT, ARGMAX_TIEBREAK_TORCH } from "./helpers/gpu_op_cases.ts";

const t = (shape: readonly number[], values: readonly number[]): RefTensor =>
  refTensor(shape, Float32Array.from(values));

const i32 = (shape: readonly number[], values: readonly number[]): RefTensor =>
  refTensor(shape, Int32Array.from(values));

const bools = (shape: readonly number[], values: readonly number[]): RefTensor =>
  refTensor(shape, Uint32Array.from(values));

test("unary known values (match torch's default definitions)", () => {
  const at = (op: UnaryOpName, x: number): number => referenceUnary(op, t([1], [x])).data[0];
  assertEquals(at("neg", -2), 2);
  assertEquals(at("abs", -2.5), 2.5);
  assertAlmostEquals(at("exp", 1), Math.E, 1e-6);
  assertAlmostEquals(at("log", Math.E), 1, 1e-6);
  assertEquals(at("sqrt", 9), 3);
  // sin is odd and its zeros are the integer multiples of π (the domain DACVAE's Snake activation
  // hits — ADR 0043 layer 1).
  assertEquals(at("sin", 0), 0);
  assertAlmostEquals(at("sin", 1), 0.8414709848078965, 1e-6);
  assertAlmostEquals(at("sin", -1), -0.8414709848078965, 1e-6);
  assertAlmostEquals(at("sin", Math.PI / 2), 1, 1e-6);
  assertAlmostEquals(at("sin", Math.PI), 0, 1e-6);
  assertAlmostEquals(at("tanh", 1), 0.7615941559557649, 1e-6);
  assertEquals(at("sigmoid", 0), 0.5);
  assertAlmostEquals(at("sigmoid", 2), 0.8807970779778823, 1e-6);
  assertEquals(at("relu", -1), 0);
  assertEquals(at("relu", 3), 3);
  // gelu(x) = x·Φ(x) (approximate="none") — Φ(1)=0.8413447460685429
  assertEquals(at("gelu", 0), 0);
  assertAlmostEquals(at("gelu", 1), 0.8413447460685429, 1e-6);
  assertAlmostEquals(at("gelu", -1), -0.15865525393145707, 1e-6);
  assertAlmostEquals(at("gelu", 3), 2.99595030590511, 1e-6);
  // For |x| > 4, rounding erf to ±1 is indistinguishable in f32
  assertAlmostEquals(at("gelu", 8), 8, 1e-6);
  // gelu_tanh(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715x³))) (approximate="tanh").
  // It differs from the erf form by ~3e-4 at x = ±1 — that gap shows the two cannot share one op.
  assertEquals(at("gelu_tanh", 0), 0);
  assertAlmostEquals(at("gelu_tanh", 1), 0.8411919906082768, 1e-6);
  assertAlmostEquals(at("gelu_tanh", -1), -0.15880800939172324, 1e-6);
  assertAlmostEquals(at("gelu_tanh", 3), 2.996362607918227, 1e-6);
  assertAlmostEquals(at("gelu_tanh", 8), 8, 1e-6);
});

test("binary evaluates with right-aligned broadcasting", () => {
  const sum = referenceBinary("add", t([2, 3], [1, 2, 3, 4, 5, 6]), t([3], [10, 20, 30]));
  assertEquals(sum.shape, [2, 3]);
  assertEquals([...sum.data], [11, 22, 33, 14, 25, 36]);

  const product = referenceBinary("mul", t([2, 1], [1, 2]), t([1, 3], [10, 20, 30]));
  assertEquals(product.shape, [2, 3]);
  assertEquals([...product.data], [10, 20, 30, 20, 40, 60]);

  const diff = referenceBinary("sub", t([2], [5, 7]), t([2], [1, 2]));
  assertEquals([...diff.data], [4, 5]);
  const quotient = referenceBinary("div", t([2], [6, 9]), t([2], [2, 3]));
  assertEquals([...quotient.data], [3, 3]);
});

// Wave-3 math ops. The known values are torch's own definitions (clamp / leaky_relu / log1p /
// comparisons).
test("unary ops with attrs follow torch's definitions and round scalars to f32", () => {
  const clamp = referenceUnary("clamp", t([5], [-3, -1, 0, 1, 3]), { min: -1, max: 1 });
  assertEquals([...clamp.data], [-1, -1, 0, 1, 1]);
  // Both bounds pass through unchanged (getting < / > backwards changes the value at the bound)
  assertEquals([...referenceUnary("clamp", t([2], [-1, 1]), { min: -1, max: 1 }).data], [-1, 1]);

  // clamp_min has only a lower bound (no upper limit). The bound itself is returned unchanged.
  const clampMin = referenceUnary("clamp_min", t([5], [-3, -1, 0, 1, 3]), { min: -1 });
  assertEquals([...clampMin.data], [-1, -1, 0, 1, 3]);
  // MUST: NaN propagates (torch's clamp(NaN, min=m) = NaN). Written as `x >= min ? x : min`, NaN
  // turns into min, so this one case detects the comparison direction.
  assertEquals(Number.isNaN(referenceUnary("clamp_min", t([1], [NaN]), { min: 0 }).data[0]), true);
  // The observed channel L2 normalization (clamp(min=eps)) — even a tiny lower bound lifts 0
  assertEquals(
    [...referenceUnary("clamp_min", t([2], [0, 5]), { min: 1e-12 }).data],
    [Math.fround(1e-12), 5],
  );
  assertThrows(() => referenceUnary("clamp_min", t([1], [0]), {}), OpContractError);

  // Two slopes (ADR 0015 — observed graphs mix 0.1 and 0.01)
  const steep = referenceUnary("leaky_relu", t([4], [-2, -1, 0, 3]), { negative_slope: 0.1 });
  assertEquals(
    [...steep.data],
    [-0.2, -0.1, 0, 3].map((v) => Math.fround(v)),
  );
  const shallow = referenceUnary("leaky_relu", t([2], [-2, 2]), { negative_slope: 0.01 });
  assertEquals([...shallow.data], [Math.fround(-0.02), 2]);
  // MUST: NaN propagates (torch's leaky_relu(NaN) = NaN)
  assertEquals(
    Number.isNaN(referenceUnary("leaky_relu", t([1], [NaN]), { negative_slope: 0.1 }).data[0]),
    true,
  );

  // log1p follows Math.log1p — it keeps a value where the naive log(1+x) collapses to 0
  assertEquals(referenceUnary("log1p", t([1], [0]), {}).data[0], 0);
  assertAlmostEquals(referenceUnary("log1p", t([1], [Math.E - 1]), {}).data[0], 1, 1e-6);
  assertEquals(referenceUnary("log1p", t([1], [1e-8]), {}).data[0], Math.fround(1e-8));

  // Comparisons return bool (u32 0/1). ge and gt differ at the equality point.
  const ge = referenceUnary("ge_scalar", t([3], [-1, 0, 1]), { value: 0 });
  assertEquals(ge.dtype, "bool");
  assertEquals([...ge.data], [0, 1, 1]);
  assertEquals([...referenceUnary("gt_scalar", t([3], [-1, 0, 1]), { value: 0 }).data], [0, 0, 1]);
  assertEquals([...referenceUnary("le_scalar", t([3], [-1, 0, 1]), { value: 0 }).data], [1, 1, 0]);
  // The contract rejects a missing required attr and an out-of-range value
  assertThrows(() => referenceUnary("clamp", t([1], [0]), { min: 0 }), OpContractError);
  assertThrows(() => referenceUnary("clamp", t([1], [0]), { min: 1, max: 0 }), OpContractError);
  assertThrows(() => referenceUnary("leaky_relu", t([1], [0]), {}), OpContractError);
});

test("ge / bitwise_and broadcast and then return bool", () => {
  // The searchsorted shape (inputs[…,None] >= bl)
  const ge = referenceBinary("ge", t([3, 1], [-1, 0.5, 2]), t([3], [0, 1, 2]));
  assertEquals(ge.dtype, "bool");
  assertEquals(ge.shape, [3, 3]);
  assertEquals([...ge.data], [0, 0, 0, 1, 0, 0, 1, 1, 1]);

  const and = referenceBinary("bitwise_and", bools([2, 2], [1, 1, 0, 0]), bools([2], [1, 0]));
  assertEquals(and.dtype, "bool");
  assertEquals([...and.data], [1, 0, 0, 0]);
  // Mixed dtypes and dtypes outside the contract are rejected
  assertThrows(() => referenceBinary("bitwise_and", t([1], [1]), t([1], [1])), OpContractError);
  assertThrows(() => referenceBinary("ge", bools([1], [1]), bools([1], [1])), OpContractError);
});

// MUST: the branch direction (true takes the second argument) is only visible in the values — fill
// a / b with different ranges.
test("where takes a where the condition is true and b where it is false, broadcasting all three right-aligned", () => {
  const out = referenceWhere(
    bools([2, 3], [1, 0, 1, 0, 1, 0]),
    t([2, 3], [1, 2, 3, 4, 5, 6]),
    t([3], [-10, -20, -30]),
  );
  assertEquals(out.dtype, "f32");
  assertEquals(out.shape, [2, 3]);
  assertEquals([...out.data], [1, -20, 3, -10, 5, -30]);
  // A condition of lower rank than the values (the shape of the spline's inside test)
  const wide = referenceWhere(bools([2], [1, 0]), t([1], [7]), t([2, 2], [1, 2, 3, 4]));
  assertEquals(wide.shape, [2, 2]);
  assertEquals([...wide.data], [7, 2, 7, 4]);
  assertThrows(
    () => referenceWhere(bools([1], [1]), bools([1], [1]), t([1], [0])),
    OpContractError,
  );
});

// MUST: the accumulation direction is only visible on an asymmetric sequence ([1,1,1] is the same
// reversed).
test("cumsum is the prefix sum over the last dimension, independent per row", () => {
  const out = referenceCumsum(t([2, 4], [1, 2, 3, 4, 10, 20, 30, 40]), { dim: 1 });
  assertEquals(out.shape, [2, 4]);
  assertEquals([...out.data], [1, 3, 6, 10, 10, 30, 60, 100]);
  // Reversed it would give [10,9,7,4] (a sequence that tells prefix from suffix)
  assertEquals([...referenceCumsum(t([1, 4], [1, 2, 3, 4]), { dim: 1 }).data], [1, 3, 6, 10]);
  // rank 1 (a single row) and rejection of dimensions other than the last
  assertEquals([...referenceCumsum(t([3], [2, -1, 5]), { dim: 0 }).data], [2, 1, 6]);
  assertThrows(() => referenceCumsum(t([2, 2], [1, 2, 3, 4]), { dim: 0 }), OpContractError);
});

test("matmul returns row-major known values", () => {
  const out = referenceMatmul(t([2, 2], [1, 2, 3, 4]), t([2, 2], [5, 6, 7, 8]));
  assertEquals(out.shape, [2, 2]);
  assertEquals([...out.data], [19, 22, 43, 50]);

  const rect = referenceMatmul(t([2, 3], [1, 2, 3, 4, 5, 6]), t([3, 1], [1, 1, 1]));
  assertEquals(rect.shape, [2, 1]);
  assertEquals([...rect.data], [6, 15]);
});

// MUST: check B / M / K / N with all-different lengths (ACTIVE_DESIGN's Pitfalls — a swapped axis
// does not show in the numbers for square or symmetric shapes). Pinned here with hand-computed
// values.
test("bmm returns an independent rank-3 matrix product per batch", () => {
  // B=2, M=3, K=2, N=1 (all different lengths). Batch 1 has every element multiplied by 10.
  const a = t([2, 3, 2], [1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 60]);
  const b = t([2, 2, 1], [1, 10, 100, 1000]);
  const out = referenceBmm(a, b);
  assertEquals(out.shape, [2, 3, 1]);
  // Batch 0: [1·1+2·10, 3·1+4·10, 5·1+6·10] / batch 1: [10·100+20·1000, ...]
  assertEquals([...out.data], [21, 43, 65, 21000, 43000, 65000]);

  // A single batch is still rank 3 (does not degenerate to matmul)
  const single = referenceBmm(t([1, 2, 2], [1, 2, 3, 4]), t([1, 2, 2], [5, 6, 7, 8]));
  assertEquals(single.shape, [1, 2, 2]);
  assertEquals([...single.data], [19, 22, 43, 50]);

  // rank 2 / batch mismatch / contraction mismatch are rejected by the contract
  assertThrows(
    () => referenceBmm(t([2, 2], [1, 2, 3, 4]), t([2, 2], [1, 2, 3, 4])),
    OpContractError,
  );
  assertThrows(
    () => referenceBmm(t([2, 1, 2], [1, 2, 3, 4]), t([1, 2, 1], [1, 2])),
    OpContractError,
  );
  assertThrows(() => referenceBmm(i32([1, 1, 1], [1]), t([1, 1, 1], [1])), OpContractError);
});

// Contract: out[..., j] = src[..., index[..., j]] (last dimension fixed). Out-of-range indices
// **always throw** (the GPU side poisons with NaN — the ruling in src/kernels/gather.ts; the oracle
// side is not relaxed).
test("gather re-reads the last dimension per row and rejects out-of-range indices", () => {
  const src = t([2, 4], [1, 2, 3, 4, 10, 20, 30, 40]);
  const index = i32([2, 3], [3, 0, 2, 1, 1, 0]);
  const out = referenceGather(src, index);
  assertEquals(out.dtype, "f32");
  assertEquals(out.shape, [2, 3]);
  // Row 0 reads only from its own row (crossing rows would mix in 10/20)
  assertEquals([...out.data], [4, 1, 3, 20, 20, 10]);

  // A rank-3 case of the same type as the observed shape (src f32[16,T,512] / index i32[16,T,T])
  const cube = referenceGather(
    t([2, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    i32([2, 2, 2], [2, 0, 1, 1, 0, 2, 2, 2]),
  );
  assertEquals(cube.shape, [2, 2, 2]);
  assertEquals([...cube.data], [3, 1, 5, 5, 7, 9, 12, 12]);

  // Out of range (above / negative) fails loudly
  assertThrows(() => referenceGather(src, i32([2, 3], [4, 0, 0, 0, 0, 0])), ReferenceOpError);
  assertThrows(() => referenceGather(src, i32([2, 3], [0, 0, 0, 0, -1, 0])), ReferenceOpError);
  // A slot mix-up (src and index passed the other way round) is rejected by the contract
  assertThrows(() => referenceGather(index, src as never), OpContractError);
  assertThrows(() => referenceGather(src, t([2, 3], [0, 1, 2, 0, 1, 2])), OpContractError);
  // Mismatched leading dimensions
  assertThrows(() => referenceGather(src, i32([3, 3], new Array(9).fill(0))), OpContractError);
});

test("row reduce folds the last dimension without keepdim", () => {
  const x = t([2, 3], [1, 2, 3, 4, 5, 6]);
  assertEquals([...referenceRowReduce("sum", x, 1).data], [6, 15]);
  assertEquals([...referenceRowReduce("amax", x, 1).data], [3, 6]);
  assertEquals([...referenceRowReduce("amin", x, 1).data], [1, 4]);
  assertEquals(referenceRowReduce("sum", x, 1).shape, [2]);

  // rank 1 → rank 0 (scalar)
  const scalar = referenceRowReduce("sum", t([3], [1, 2, 3]), 0);
  assertEquals(scalar.shape, []);
  assertEquals([...scalar.data], [6]);

  const cube = referenceRowReduce("amax", t([2, 2, 2], [1, 8, 3, 2, 9, 4, 5, 6]), 2);
  assertEquals(cube.shape, [2, 2]);
  assertEquals([...cube.data], [8, 3, 9, 6]);
});

// Axis reduce (dimensions other than the last). **A shape that cannot pass trivially**: asymmetric
// data whose values differ on every axis, so a swapped axis always gives a different number (the
// answer differs for each of the 3 axes of a rank-3 tensor).
test("reduce folds only the attrs.dim axis (the answer differs per axis)", () => {
  // x[i,j,k] = 100i + 10j + k (folding any axis gives different values)
  const shape = [2, 3, 4];
  const data = new Array<number>(24);
  for (let i = 0; i < 2; i += 1) {
    for (let j = 0; j < 3; j += 1) {
      for (let k = 0; k < 4; k += 1) data[(i * 3 + j) * 4 + k] = 100 * i + 10 * j + k;
    }
  }
  const x = t(shape, data);

  const axis0 = referenceRowReduce("sum", x, 0);
  assertEquals(axis0.shape, [3, 4]);
  // Σ_i (100i + 10j + k) = 100 + 2·(10j + k)
  assertEquals([...axis0.data], [100, 102, 104, 106, 120, 122, 124, 126, 140, 142, 144, 146]);

  const axis1 = referenceRowReduce("sum", x, 1);
  assertEquals(axis1.shape, [2, 4]);
  // Σ_j (100i + 10j + k) = 300i + 30 + 3k
  assertEquals([...axis1.data], [30, 33, 36, 39, 330, 333, 336, 339]);

  const axis2 = referenceRowReduce("sum", x, 2);
  assertEquals(axis2.shape, [2, 3]);
  // Σ_k (100i + 10j + k) = 400i + 40j + 6
  assertEquals([...axis2.data], [6, 46, 86, 406, 446, 486]);

  // amax / amin also differ per axis (fixing only sum's index expression does not pass)
  assertEquals([...referenceRowReduce("amax", x, 1).data], [20, 21, 22, 23, 120, 121, 122, 123]);
  assertEquals(
    [...referenceRowReduce("amin", x, 0).data],
    [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23],
  );

  // The axis is the same through attrs (the wiring of applyReferenceOp)
  assertEquals([...applyReferenceOp("sum", [x], { dim: 1 }).data], [...axis1.data]);
  // Declaration required (no default filled in)
  assertThrows(() => applyReferenceOp("sum", [x]), OpContractError);
  assertThrows(() => applyReferenceOp("sum", [x], { dim: 3 }), OpContractError);
});

/**
 * The reduction identity of amax / amin (ruling 4a) — pins that the CPU reference returns **the
 * same value** as the GPU side (identity ∓F32_MAX in `codegen/reduce.ts`) for all-−Infinity and
 * all-+Infinity rows.
 *
 * MUST: the expected value is a literal of the largest finite f32, imported from neither the
 * reference nor codegen (pulling either constant would carry a regression that moves the identity
 * into the expected value too, making the test trivially true). The contract layer rejects only
 * zero-length axes (`ops/shapes.ts`), so an all-−inf row is valid input.
 */
test("amax / amin fold from the identity (an all-±inf row gives ∓F32_MAX, as on the GPU)", () => {
  const f32Max = 3.4028234663852886e38;
  const negInf = Number.NEGATIVE_INFINITY;
  const posInf = Number.POSITIVE_INFINITY;

  // Row 0 = all −inf (the identity remains) / row 1 = has finite values (the identity does not win)
  const amax = referenceRowReduce("amax", t([2, 3], [negInf, negInf, negInf, -5, -1, -9]), 1);
  assertEquals([...amax.data], [-f32Max, -1]);

  const amin = referenceRowReduce("amin", t([2, 3], [posInf, posInf, posInf, 5, 1, 9]), 1);
  assertEquals([...amin.data], [f32Max, 1]);

  // amax of a row containing +inf stays +inf (the identity does not clip the top end)
  assertEquals([...referenceRowReduce("amax", t([1, 2], [posInf, 1]), 1).data], [posInf]);
  // NaN propagation is unchanged when starting from the identity (like the GPU's NaN-propagating
  // max/min)
  assertEquals(
    Number.isNaN(referenceRowReduce("amax", t([1, 2], [Number.NaN, 1]), 1).data[0]),
    true,
  );
});

/**
 * The CPU reference of argmax (ADR 0068 decision 2) — pins tie-breaking / NaN / all-−inf rows with
 * literals of values measured on torch.
 *
 * MUST: do not put this inside the real-GPU comparison. The oracle's direction is not settled by
 * comparing with the GPU (if both sides are wrong the same way, it is green), so this test, which
 * runs with or without an adapter, is the only check on the CPU reference's values. The real-GPU
 * side (tests/gpu_ops_test.ts) applies the same table to the GPU output.
 */
test("argmax's CPU reference matches torch's measured values for ties / NaN / all-−inf rows", () => {
  const rows = ARGMAX_TIEBREAK_INPUT.shape[0];
  const out = applyReferenceOp("argmax", [ARGMAX_TIEBREAK_INPUT], {}, [rows, 1]);
  assertEquals(out.dtype, "i32");
  assertEquals(out.shape, [rows, 1]);
  assertEquals(
    [...out.data],
    [...ARGMAX_TIEBREAK_TORCH],
    "the CPU reference's argmax differs from torch's measured values",
  );
});

/**
 * The CPU reference of topk (ADR 0068 decision 3) — the wiring of the **unified multi-output
 * entry point** and the accepted domain.
 *
 * The value and index columns themselves are compared with literals of torch's measured values by
 * the dedicated real-GPU check (tests/gpu_ops_test.ts), so this pins "the reference returns two
 * outputs", "the single-output entry point cannot take it" and "inputs outside the accepted domain
 * fail".
 */
test("topk's CPU reference returns values and indices, and the single-output entry point rejects it", () => {
  const x = t([2, 4], [1, 3, 3, 2, -1, -1, 5, 0]);
  const outputs = applyReferenceOpOutputs("topk", [x], { k: 2 });
  assertEquals(outputs.length, 2);
  assertEquals(outputs[0].dtype, "f32");
  assertEquals(outputs[1].dtype, "i32");
  assertEquals(outputs[0].shape, [2, 2]);
  assertEquals(outputs[1].shape, [2, 2]);
  // Values in descending order, ties prefer the lowest index (row 0 has two 3.0s)
  assertEquals([...outputs[0].data], [3, 3, 5, 0]);
  assertEquals([...outputs[1].data], [1, 2, 2, 3]);
  // MUST: the single-output entry point does not silently return only slot 0 of a multi-output op
  // (that would allow a test whose indices are compared against nothing)
  assertThrows(() => applyReferenceOp("topk", [x], { k: 2 }), ReferenceOpError, "has 2 outputs");
  // Outside the accepted domain (k=0 / k > last dimension / missing k / symbolic k) all fail loudly
  assertThrows(() => applyReferenceOpOutputs("topk", [x], { k: 0 }), OpContractError);
  assertThrows(() => applyReferenceOpOutputs("topk", [x], { k: 5 }), OpContractError);
  assertThrows(() => applyReferenceOpOutputs("topk", [x], {}), OpContractError);
  assertThrows(() => applyReferenceOpOutputs("topk", [x], { k: "T" }), OpContractError);
  // i32 input is not in the vocabulary (f32 only)
  assertThrows(
    () => applyReferenceOpOutputs("topk", [i32([2], [1, 2])], { k: 1 }),
    OpContractError,
  );
});

// Contract (src/ops.ts): f32 → i32 truncates as torch does (toward 0).
// MUST: use a value that tells round / floor apart — -2.7 is trunc -2 / floor -3 / round -3.
test("cast truncates f32 → i32 toward 0 (neither round nor floor)", () => {
  const out = referenceCast(t([6], [2.7, -2.7, 0.5, -0.5, 1.5, -1.5]), "i32");
  assertEquals(out.dtype, "i32");
  assertEquals([...out.data], [2, -2, 0, 0, 1, -1]);
});

test("cast converts to bool by x != 0 and reads bool as u32 0/1", () => {
  const fromFloat = referenceCast(t([4], [0, -0.25, 3, -7]), "bool");
  assertEquals(fromFloat.dtype, "bool");
  assertEquals([...fromFloat.data], [0, 1, 1, 1]);
  assertEquals([...referenceCast(i32([3], [0, -1, 5]), "bool").data], [0, 1, 1]);
  // bool → number gives 0/1 as is
  assertEquals([...referenceCast(bools([2], [0, 1]), "f32").data], [0, 1]);
  assertEquals([...referenceCast(bools([2], [0, 1]), "i32").data], [0, 1]);
  // A same-type cast is an identity copy
  assertEquals([...referenceCast(i32([2], [7, -7]), "i32").data], [7, -7]);
});

test("bitwise_not negates bool, and the contract rejects anything other than bool", () => {
  const out = referenceUnary("bitwise_not", bools([4], [0, 1, 1, 0]));
  assertEquals(out.dtype, "bool");
  assertEquals([...out.data], [1, 0, 0, 1]);
  assertThrows(() => referenceUnary("bitwise_not", t([2], [0, 1])), OpContractError);
  assertThrows(() => referenceUnary("relu", bools([2], [0, 1])), OpContractError);
});

test("i32 binary evaluates only the enabled ops with 32-bit arithmetic", () => {
  // Mask outer product (the shape of the observed graph): [3,1] × [1,3] broadcast
  const outer = referenceBinary("mul", i32([3, 1], [1, 0, 1]), i32([1, 3], [1, 1, 0]));
  assertEquals(outer.dtype, "i32");
  assertEquals([...outer.data], [1, 1, 0, 0, 0, 0, 1, 1, 0]);
  assertEquals([...referenceBinary("sub", i32([2], [1, 1]), i32([2], [0, 1])).data], [1, 0]);
  // i32 wraps around in two's complement (the same as GPU i32 arithmetic)
  assertEquals([...referenceBinary("mul", i32([1], [65536]), i32([1], [65536])).data], [0]);
  // Combinations the contract table does not enable
  assertThrows(() => referenceBinary("add", i32([1], [1]), i32([1], [1])), OpContractError);
  assertThrows(() => referenceBinary("div", i32([1], [4]), i32([1], [2])), OpContractError);
  // Mixed dtypes (both are contract dtypes of mul, but mixed-type elementwise is not in the
  // vocabulary)
  assertThrows(() => referenceBinary("mul", i32([1], [1]), t([1], [1])), ReferenceOpError);
});

test("reshape changes only the shape without reordering elements", () => {
  const out = referenceReshape(t([2, 3], [1, 2, 3, 4, 5, 6]), [3, 2]);
  assertEquals(out.shape, [3, 2]);
  assertEquals([...out.data], [1, 2, 3, 4, 5, 6]);
  // The dtype also passes through (an op that does not touch elements)
  assertEquals(referenceReshape(bools([4], [0, 1, 1, 0]), [2, 2]).dtype, "bool");
  assertThrows(() => referenceReshape(t([2, 3], [1, 2, 3, 4, 5, 6]), [4, 2]), ReferenceOpError);
});

// Expected values are hand-computed (independent of codegen's stride table — the reference derives
// them in scatter form).
test("permute redistributes elements by reordering dims", () => {
  // Transpose of [2,3]: row-major [1,2,3 / 4,5,6] → [1,4 / 2,5 / 3,6]
  const flat = referencePermute(t([2, 3], [1, 2, 3, 4, 5, 6]), [1, 0]);
  assertEquals(flat.shape, [3, 2]);
  assertEquals([...flat.data], [1, 4, 2, 5, 3, 6]);
  // The observed shape [0,2,1,3] (attention's head reshaping) exercised at rank 4
  const src = Array.from({ length: 24 }, (_, i) => i);
  const heads = referencePermute(t([1, 2, 3, 4], src), [0, 2, 1, 3]);
  assertEquals(heads.shape, [1, 3, 2, 4]);
  assertEquals(
    [...heads.data].join(","),
    "0,1,2,3,12,13,14,15,4,5,6,7,16,17,18,19,8,9,10,11,20,21,22,23",
  );
  // MUST: keep one permutation with a 3-cycle (every observed shape is an involution, so a mix-up
  // with the inverse permutation is undetectable). Unlike [1,0] on [2,3], the inverse of [1,2,0] is
  // [2,0,1], so the result changes.
  const cycled = referencePermute(t([1, 2, 3], [1, 2, 3, 4, 5, 6]), [1, 2, 0]);
  assertEquals(cycled.shape, [2, 3, 1]);
  assertEquals([...cycled.data], [1, 2, 3, 4, 5, 6]);
  const rotated = referencePermute(t([2, 1, 3], [1, 2, 3, 4, 5, 6]), [1, 2, 0]);
  assertEquals(rotated.shape, [1, 3, 2]);
  assertEquals([...rotated.data], [1, 4, 2, 5, 3, 6]);
  // The identity permutation does not reorder elements
  assertEquals([...referencePermute(t([2, 2], [1, 2, 3, 4]), [0, 1]).data], [1, 2, 3, 4]);
  assertThrows(() => referencePermute(t([2, 3], [1, 2, 3, 4, 5, 6]), [0, 0]), ReferenceOpError);
  assertThrows(() => referencePermute(t([2, 3], [1, 2, 3, 4, 5, 6]), [0]), ReferenceOpError);
});

test("expand replicates only dimensions of length 1", () => {
  // The observed gather index shape [1,T,T] → [16,T,T], shrunk to [1,2,2] → [3,2,2]
  const spread = referenceExpand(i32([1, 2, 2], [1, 2, 3, 4]), [3, 2, 2]);
  assertEquals(spread.shape, [3, 2, 2]);
  assertEquals([...spread.data], [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]);
  // The observed bool mask shape [1,T,1] → [1,T,C]
  const mask = referenceExpand(bools([1, 2, 1], [1, 0]), [1, 2, 3]);
  assertEquals([...mask.data], [1, 1, 1, 0, 0, 0]);
  assertEquals(mask.dtype, "bool");
  // A shape whose rank grows (right-aligned)
  assertEquals([...referenceExpand(i32([2], [7, 8]), [2, 2]).data], [7, 8, 7, 8]);
  assertThrows(() => referenceExpand(i32([2, 2], [1, 2, 3, 4]), [2, 4]), ReferenceOpError);
  assertThrows(() => referenceExpand(i32([2, 2], [1, 2, 3, 4]), [4]), ReferenceOpError);
});

// ---- Layout group 2 (ADR 0014) ----------------------------------------------

test("slice takes only [start, end) of a static axis", () => {
  // enc_p's stats[:, :c] / stats[:, c:], shrunk (channel axis 4 → first 2 / last 2)
  const stats = t([1, 4, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
  const head = referenceSlice(stats, { dim: 1, start: 0, end: 2 });
  assertEquals(head.shape, [1, 2, 2]);
  assertEquals([...head.data], [1, 2, 3, 4]);
  const tail = referenceSlice(stats, { dim: 1, start: 2, end: 4 });
  assertEquals([...tail.data], [5, 6, 7, 8]);
  // Slicing the last dimension (same type as the spline's bin_locations[..., :-1]) — the row stride
  // stays the input's
  const bins = t([2, 3], [1, 2, 3, 4, 5, 6]);
  assertEquals([...referenceSlice(bins, { dim: 1, start: 0, end: 2 }).data], [1, 2, 4, 5]);
  assertEquals([...referenceSlice(bins, { dim: 1, start: 1, end: 3 }).data], [2, 3, 5, 6]);
  // Slicing the leading axis, and the full length (identity copy)
  assertEquals([...referenceSlice(bins, { dim: 0, start: 1, end: 2 }).data], [4, 5, 6]);
  assertEquals([...referenceSlice(bins, { dim: 0, start: 0, end: 2 }).data], [...bins.data]);
  // Contract violations (out of range, reversed, outside the axes) are rejected by the shape layer
  assertThrows(() => referenceSlice(bins, { dim: 1, start: 1, end: 4 }), OpContractError);
  assertThrows(() => referenceSlice(bins, { dim: 1, start: 2, end: 1 }), OpContractError);
  assertThrows(() => referenceSlice(bins, { dim: 2, start: 0, end: 1 }), OpContractError);
});

test("cat lays the inputs out in order along the concatenation axis and fills the whole output", () => {
  // coupling reverse's cat([x0,x1], 1), shrunk (2 + 1 on axis 1)
  const a = t([2, 2], [1, 2, 3, 4]);
  const b = t([2, 1], [5, 6]);
  const joined = referenceCat([a, b], { dim: 1 });
  assertEquals(joined.shape, [2, 3]);
  // MUST: rows interleave (building the offset "by the input's element count" would give
  // [1,2,3,4,5,6])
  assertEquals([...joined.data], [1, 2, 5, 3, 4, 6]);
  // Concatenation on the leading axis is plain concatenation (rows in order)
  assertEquals([...referenceCat([a, t([1, 2], [9, 8])], { dim: 0 }).data], [1, 2, 3, 4, 9, 8]);
  // 3 inputs (variadic) — a shape where the order shows in the values
  const three = referenceCat([t([1, 1], [1]), t([1, 2], [2, 3]), t([1, 1], [4])], { dim: 1 });
  assertEquals([...three.data], [1, 2, 3, 4]);
  // Contract violations are rejected by the shape layer (mismatch outside the concatenation axis, a
  // single input)
  assertThrows(() => referenceCat([a, t([3, 1], [1, 2, 3])], { dim: 1 }), OpContractError);
  assertThrows(() => referenceCat([a], { dim: 1 }), OpContractError);
});

test("pad fills the last dimension with 0, and the padded area is exactly 0", () => {
  // The relative-position value side's F.pad(p_attn, [w, w]), shrunk (w = 1)
  const attn = t([2, 2], [1, 2, 3, 4]);
  const padded = referencePad(attn, { left: 1, right: 1 });
  assertEquals(padded.shape, [2, 4]);
  assertEquals([...padded.data], [0, 1, 2, 0, 0, 3, 4, 0]);
  // Asymmetric, one side 0 (a shape where an off-by-one at the boundary shows in the values)
  assertEquals([...referencePad(attn, { left: 2, right: 0 }).data], [0, 0, 1, 2, 0, 0, 3, 4]);
  assertEquals([...referencePad(attn, { left: 0, right: 1 }).data], [1, 2, 0, 3, 4, 0]);
  // Width 0 is an identity copy
  assertEquals([...referencePad(attn, { left: 0, right: 0 }).data], [...attn.data]);
  assertThrows(() => referencePad(attn, { left: -1, right: 0 }), OpContractError);
});

test("flip reverses the indices of a static axis", () => {
  // The flow's Flip, shrunk (axis 1 of length 3 — with length 2 an off-by-one cancels out by
  // symmetry)
  const x = t([2, 3, 2], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  const flipped = referenceFlip(x, { dim: 1 });
  assertEquals(flipped.shape, [2, 3, 2]);
  assertEquals([...flipped.data], [5, 6, 3, 4, 1, 2, 11, 12, 9, 10, 7, 8]);
  // A swapped axis gives a different value (leading axis and last axis each)
  assertEquals([...referenceFlip(x, { dim: 0 }).data], [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]);
  assertEquals([...referenceFlip(x, { dim: 2 }).data], [2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]);
  // Flipping is an involution (twice gives the original)
  assertEquals([...referenceFlip(referenceFlip(x, { dim: 1 }), { dim: 1 }).data], [...x.data]);
  assertThrows(() => referenceFlip(x, { dim: 3 }), OpContractError);
});

test("sym_prefix_slice takes only the leading part of a Tmax constant", () => {
  // The observed shape of the relative-position bucket table — take the leading 2×2 for T=2 from a
  // Tmax 4×4 constant
  const table = i32([4, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
  const sliced = referenceSymPrefixSlice(table, [2, 2]);
  assertEquals(sliced.shape, [2, 2]);
  // MUST: the row stride is **the Tmax shape (4)** — building it from the output shape (2) gives
  // [0,1,2,3]
  assertEquals([...sliced.data], [0, 1, 4, 5]);
  assertEquals(sliced.dtype, "i32");
  // Shrinking only one axis (position table), and shrinking none (identity)
  assertEquals([...referenceSymPrefixSlice(table, [2, 4]).data], [0, 1, 2, 3, 4, 5, 6, 7]);
  assertEquals([...referenceSymPrefixSlice(table, [4, 4]).data], [...table.data]);
  // f32 takes the same path (position scale table)
  const scale = t([3, 2], [0, 1, 2, 3, 4, 5]);
  assertEquals([...referenceSymPrefixSlice(scale, [2, 1]).data], [0, 2]);
  // A prefix that does not fit in the constant and a rank mismatch fail loudly
  assertThrows(() => referenceSymPrefixSlice(table, [5, 4]), ReferenceOpError);
  assertThrows(() => referenceSymPrefixSlice(table, [4]), ReferenceOpError);
});

// ---- Known values of fused ops (ADR 0012) ------------------------------------

test("linear returns known values with weights in transposed layout", () => {
  // x[2,3] × W[2,3] (W is [out,in]) + b[2]
  const out = referenceLinear(
    t([2, 3], [1, 2, 3, 4, 5, 6]),
    t([2, 3], [1, 0, -1, 2, 2, 2]),
    t([2], [10, -1]),
  );
  assertEquals(out.shape, [2, 2]);
  // Row 0: [1·1+2·0+3·(−1)+10, 1·2+2·2+3·2+(−1)] = [8, 11]
  // Row 1: [4·1+5·0+6·(−1)+10, 4·2+5·2+6·2+(−1)] = [8, 29]
  assertEquals([...out.data], [8, 11, 8, 29]);
  // Leading dimensions are flattened (the weights are the same at rank 3)
  assertEquals(
    referenceLinear(
      t([2, 1, 3], [1, 2, 3, 4, 5, 6]),
      t([2, 3], [1, 0, -1, 2, 2, 2]),
      t([2], [10, -1]),
    ).shape,
    [2, 1, 2],
  );
  // A mismatched feature count or bias length is rejected by the contract
  assertThrows(
    () => referenceLinear(t([2, 3], [1, 2, 3, 4, 5, 6]), t([2, 2], [1, 0, 1, 0]), t([2], [0, 0])),
    OpContractError,
  );
});

// MUST: the variance is the population variance (divide by N). torch.var's default (N−1) drifts
// further the smaller dim is.
test("layer_norm normalizes with the population variance (correction=0)", () => {
  const attrs = { normalized_shape: [4], eps: 1e-12 };
  const out = referenceLayerNorm(
    t([1, 4], [1, 2, 3, 4]),
    t([4], [1, 1, 1, 1]),
    t([4], [0, 0, 0, 0]),
    attrs,
  );
  // mean = 2.5 / population variance = 1.25 (N−1 would give 1.6667, 15% off the expected values
  // below)
  const inv = 1 / Math.sqrt(1.25);
  assertAlmostEquals(out.data[0], -1.5 * inv, 1e-5);
  assertAlmostEquals(out.data[3], 1.5 * inv, 1e-5);
  // affine applies (weight scales, bias shifts)
  const affine = referenceLayerNorm(
    t([1, 4], [1, 2, 3, 4]),
    t([4], [2, 2, 2, 2]),
    t([4], [5, 5, 5, 5]),
    attrs,
  );
  assertAlmostEquals(affine.data[0], -1.5 * inv * 2 + 5, 1e-5);
  // A row with variance 0 keeps only eps (the output is the bias itself)
  const constant = referenceLayerNorm(
    t([1, 4], [7, 7, 7, 7]),
    t([4], [3, 3, 3, 3]),
    t([4], [1, 2, 3, 4]),
    { normalized_shape: [4], eps: 1e-7 },
  );
  assertEquals([...constant.data], [1, 2, 3, 4]);
});

// group_norm: a group is a contiguous channel band of one sample (population variance, per-channel
// affine) — hand-computed on two samples × two groups, plus the cancellation case.
test("group_norm normalises each sample's channel bands and applies the per-channel affine", () => {
  // x[2, 4, 2]: groups of 2 channels × 2 elements. Sample 0 group 0 = [1, 2, 3, 4] (mean 2.5,
  // population variance 1.25), group 1 = [10, 10, 10, 10] (variance 0 → the bias).
  const x = t([2, 4, 2], [1, 2, 3, 4, 10, 10, 10, 10, -1, -2, -3, -4, 0, 4, 0, 4]);
  const weight = t([4], [1, 2, 3, 4]);
  const bias = t([4], [0, 0.5, -1, 2]);
  const out = referenceGroupNorm(x, weight, bias, { num_groups: 2, eps: 1e-12 });
  const inv = 1 / Math.sqrt(1.25);
  // channel 0 = elements 0, 1; channel 1 = elements 2, 3 (weight 2, bias 0.5)
  assertAlmostEquals(out.data[0], -1.5 * inv * 1 + 0, 1e-6);
  assertAlmostEquals(out.data[3], 1.5 * inv * 2 + 0.5, 1e-6);
  assertEquals([...out.data.slice(4, 8)], [-1, -1, 2, 2]);
  // sample 1 has its own statistics: group 0 = [-1, -2, -3, -4] (mean -2.5)
  assertAlmostEquals(out.data[8], 1.5 * inv * 1 + 0, 1e-6);
  // group 1 = [0, 4, 0, 4] (mean 2, variance 4): element 15 is channel 3
  assertAlmostEquals(out.data[15], (2 / 2) * 4 + 2, 1e-6);
  assertThrows(
    () => referenceGroupNorm(x, weight, bias, { num_groups: 3, eps: 1e-6 }),
    OpContractError,
  );
});

test("group_norm keeps its variance at a mean offset of 2^30 (two passes in f64)", () => {
  // x = 2^30 + 128·k with k = (5i mod 7) − 3 (every value an f32). Mean / spread ≈ 2^23, where
  // even an f64 one-pass E[x²] − E[x]² loses the variance's leading digits (Σx² ≈ 2^66 has an
  // ulp of 2^14 against n·variance ≈ 2^22); the two-pass form is exact here. The expected values
  // come from the k alone (the offset and the factor 128 cancel; eps is negligible).
  const ks = Array.from({ length: 64 }, (_, i) => ((5 * i) % 7) - 3);
  const values = ks.map((k) => 2 ** 30 + 128 * k);
  const out = referenceGroupNorm(t([1, 1, 64], values), t([1], [1]), t([1], [0]), {
    num_groups: 1,
    eps: 1e-12,
  });
  const mean = ks.reduce((sum, k) => sum + k, 0) / 64;
  const deviation = Math.sqrt(ks.reduce((sum, k) => sum + (k - mean) ** 2, 0) / 64);
  for (let i = 0; i < 64; i += 1) {
    assertAlmostEquals(out.data[i], (ks[i] - mean) / deviation, 1e-6);
  }
});

test("group_norm puts eps inside the square root (a group variance of the order of eps)", () => {
  // Group [−a, a, −a, a] with a = f32(1e-3): variance a² ≈ 1e-6 = eps, so 1 / sqrt(var + eps)
  // (≈ 0.707 / a), 1 / (sqrt(var) + eps) (≈ 0.999 / a) and a missing eps (1 / a) differ by far
  // more than the tolerance.
  const a = Math.fround(1e-3);
  const out = referenceGroupNorm(t([1, 1, 4], [-a, a, -a, a]), t([1], [1]), t([1], [0]), {
    num_groups: 1,
    eps: 1e-6,
  });
  const expected = a / Math.sqrt(a * a + Math.fround(1e-6));
  assertAlmostEquals(expected, Math.SQRT1_2, 1e-3);
  for (let i = 0; i < 4; i += 1) {
    assertAlmostEquals(out.data[i], i % 2 === 0 ? -expected : expected, 1e-6);
  }
});

// MUST: rms_norm does **not subtract the mean** (pins a copy-paste from layer_norm by hand
// calculation).
test("rms_norm normalizes by the root mean square and does not subtract the mean", () => {
  const attrs = { eps: 1e-12 };
  // mean(x²) = (1+4+9+16)/4 = 7.5 → rms = √7.5. layer_norm would subtract the mean 2.5, so the
  // first element would be negative (here all stay positive = evidence the mean is not subtracted).
  const out = referenceRmsNorm(t([1, 4], [1, 2, 3, 4]), t([4], [1, 1, 1, 1]), attrs);
  const inv = 1 / Math.sqrt(7.5);
  assertAlmostEquals(out.data[0], 1 * inv, 1e-6);
  assertAlmostEquals(out.data[3], 4 * inv, 1e-6);
  // weight applies per element (there is no bias — arity 2)
  const scaled = referenceRmsNorm(t([1, 4], [1, 2, 3, 4]), t([4], [2, 0, -1, 0.5]), attrs);
  assertAlmostEquals(scaled.data[0], 1 * inv * 2, 1e-6);
  assertEquals(scaled.data[1], 0);
  assertAlmostEquals(scaled.data[2], 3 * inv * -1, 1e-6);
  // Independent per row (leading dimensions are flattened correctly)
  const rows = referenceRmsNorm(t([2, 2], [3, 4, 6, 8]), t([2], [1, 1]), attrs);
  assertAlmostEquals(rows.data[0], 3 / Math.sqrt(12.5), 1e-6);
  assertAlmostEquals(rows.data[2], 6 / Math.sqrt(50), 1e-6);
  // MUST: eps is **inside the square root** (added outside, an all-zero row gives eps times a value
  // instead of 0). Without eps an all-zero row is 0/0 = NaN. If this is not 0, eps is in the wrong
  // place.
  const zeros = referenceRmsNorm(t([1, 3], [0, 0, 0]), t([3], [5, 5, 5]), { eps: 1e-6 });
  assertEquals([...zeros.data], [0, 0, 0]);
  // Pin how eps acts by value (a 1-element row with x=1: 1/√(1+eps))
  const epsy = referenceRmsNorm(t([1, 1], [1]), t([1], [1]), { eps: 0.5 });
  assertAlmostEquals(epsy.data[0], 1 / Math.sqrt(1.5), 1e-6);
  // A mismatched weight length or rank is rejected by the contract
  assertThrows(
    () => referenceRmsNorm(t([1, 4], [1, 2, 3, 4]), t([3], [1, 1, 1]), attrs),
    OpContractError,
  );
  assertThrows(
    () => referenceRmsNorm(t([1, 4], [1, 2, 3, 4]), t([1, 4], [1, 1, 1, 1]), attrs),
    OpContractError,
  );
});

// MUST: safe softmax (amax subtraction). The naive form gives NaN on large inputs, as in the
// ledger's counterexamples.
test("softmax subtracts amax and does not break on large inputs", () => {
  const attrs = { dim: 1 };
  const out = referenceSoftmax(t([1, 3], [1, 2, 3]), attrs);
  const total = Math.exp(-2) + Math.exp(-1) + 1;
  assertAlmostEquals(out.data[0], Math.exp(-2) / total, 1e-6);
  assertAlmostEquals(out.data[0] + out.data[1] + out.data[2], 1, 1e-6);
  // The region where the naive form would collapse exp(-200) to 0 in f32 and give 0/0 = NaN
  const huge = referenceSoftmax(t([1, 2], [-200, -199]), attrs);
  assertEquals(
    huge.data.every((v) => Number.isFinite(v)),
    true,
  );
  assertAlmostEquals(huge.data[0] + huge.data[1], 1, 1e-6);
  assertAlmostEquals(huge.data[1] / huge.data[0], Math.E, 1e-4);
  // A row with all elements equal (a row fully masked by masked_fill) is uniform
  const masked = referenceSoftmax(t([1, 4], Array(4).fill(-3.4028234663852886e38)), attrs);
  assertEquals([...masked.data], [0.25, 0.25, 0.25, 0.25]);
  // Dimensions other than the last are not in the vocabulary
  assertThrows(() => referenceSoftmax(t([2, 3], [1, 2, 3, 4, 5, 6]), { dim: 0 }), OpContractError);
});

// safe_softmax (ADR 0044) — everything softmax does + "a row whose max is −inf is all 0". Pins that
// it gives the same values as torch's SDPA guard (`where(¬any(¬eq(src,−inf)), 0, softmax(src))`).
test("safe_softmax writes 0 to all-−inf rows and equals softmax elsewhere", () => {
  const attrs = { dim: 1 };
  const neg = Number.NEGATIVE_INFINITY;
  // Row 0 = normal / row 1 = all −inf (the row where torch's guard fires) / row 2 = partly −inf
  const x = t([3, 3], [1, 2, 3, neg, neg, neg, neg, 0, 1]);
  const out = referenceSafeSoftmax(x, attrs);
  assertEquals([...out.data.slice(3, 6)], [0, 0, 0]);
  // Plain softmax gives 0/0 = NaN on the same row (this one row is exactly "why it is a separate
  // op")
  assertEquals(referenceSoftmax(x, attrs).data.slice(3, 6).every(Number.isNaN), true);
  // Rows without −inf / rows that are only partly −inf match plain softmax **bit for bit**
  const plain = referenceSoftmax(x, attrs);
  assertEquals([...out.data.slice(0, 3)], [...plain.data.slice(0, 3)]);
  assertEquals([...out.data.slice(6, 9)], [...plain.data.slice(6, 9)]);
  assertAlmostEquals(out.data[6], 0, 1e-12);
  assertAlmostEquals(out.data[7] + out.data[8], 1, 1e-6);
  // masked_fill's fill value (−F32_MAX) is **finite**, so the row is not empty (it stays uniform)
  const filled = referenceSafeSoftmax(t([1, 4], Array(4).fill(-3.4028234663852886e38)), attrs);
  assertEquals([...filled.data], [0.25, 0.25, 0.25, 0.25]);
  // The contract is the same as softmax (last dimension only)
  assertThrows(
    () => referenceSafeSoftmax(t([2, 3], [1, 2, 3, 4, 5, 6]), { dim: 0 }),
    OpContractError,
  );
});

// Fused attention (ADR 0023). The oracle's discipline is "the plain 3 steps", not copying the GPU's
// fused form (P not materialized). Pins the **half scale** (scale applies to both q and k) and the
// safe form.
test("attention applies scale to both q and k and does not break large inputs with safe softmax", () => {
  // A hand-computed case with B=1 / H=1 / M=1 / N=2 / D=2. With scale=2 the dot product is
  // multiplied by 4.
  const q = t([1, 1, 1, 2], [1, 0]);
  const key = t([1, 1, 2, 2], [1, 0, 0, 1]);
  const value = t([1, 1, 2, 2], [10, 20, 30, 40]);
  const out = referenceAttention(q, key, value, { scale: 2 });
  assertEquals(out.shape, [1, 1, 1, 2]);
  // S = [(1·2)·(1·2), (1·2)·(0·2)] = [4, 0] — **applying it to one side only gives [2, 0]**
  const weight = 1 / (1 + Math.exp(-4));
  assertAlmostEquals(out.data[0], weight * 10 + (1 - weight) * 30, 1e-5);
  assertAlmostEquals(out.data[1], weight * 20 + (1 - weight) * 40, 1e-5);
  // Explicitly pin that it does not match the value of one-sided scaling (the wrong implementation)
  const halfWeight = 1 / (1 + Math.exp(-2));
  assertEquals(Math.abs(out.data[0] - (halfWeight * 10 + (1 - halfWeight) * 30)) > 1e-3, true);

  // A row whose logits are all around −200 (naive softmax would collapse exp to 0 in f32 and give
  // 0/0 = NaN)
  const huge = referenceAttention(
    t([1, 1, 1, 2], [-10, -10]),
    t([1, 1, 2, 2], [10, 10, 10, 11]),
    value,
    { scale: 1 },
  );
  assertEquals(
    [...huge.data].every((v) => Number.isFinite(v)),
    true,
  );

  // B and H are treated as separate axes (a shape with the same product does not hide a mix-up)
  const batched = referenceAttention(
    t([2, 1, 1, 1], [1, 2]),
    t([2, 1, 1, 1], [1, 1]),
    t([2, 1, 1, 1], [5, 7]),
    { scale: 1 },
  );
  // N=1, so softmax is the identity (probability 1) — the output is v itself
  assertEquals([...batched.data], [5, 7]);

  // Contract violations (rank 3 / D mismatch / missing scale) fail on the oracle side as well
  // MUST: all 3 are `OpContractError`. Expecting the base `Error` would stay green even if the rank
  // check disappeared and the later shape destructuring threw a TypeError / RangeError instead.
  assertThrows(
    () => referenceAttention(t([1, 1, 1], [1]), key, value, { scale: 1 }),
    OpContractError,
  );
  assertThrows(
    () => referenceAttention(q, key, t([1, 1, 2, 3], [1, 2, 3, 4, 5, 6]), { scale: 1 }),
    OpContractError,
  );
  assertThrows(() => referenceAttention(q, key, value, {}), OpContractError);
  // The unified entry point gives the same result (wiring of the kind branch)
  assertEquals(
    [...applyReferenceOp("attention", [q, key, value], { scale: 2 }).data],
    [...out.data],
  );

  // Additive mask (the optional fourth input — ADR 0023 revision). Added **after** S is rounded.
  // In the same hand-computed case as above, dropping column 1 makes the output row 0 of v itself.
  const masked = referenceAttention(q, key, value, { scale: 2 }, t([1, 1, 1, 2], [0, -1e30]));
  assertAlmostEquals(masked.data[0], 10, 1e-5);
  assertAlmostEquals(masked.data[1], 20, 1e-5);
  // MUST: the mask broadcasts over B·H (every head reads the one [1,1,M,N] plane).
  // Applying a mask that drops column 0 to a 2-head input makes both heads row 1 of v.
  const twoHeads = referenceAttention(
    t([1, 2, 1, 2], [1, 0, 1, 0]),
    t([1, 2, 2, 2], [1, 0, 0, 1, 1, 0, 0, 1]),
    t([1, 2, 2, 2], [10, 20, 30, 40, 50, 60, 70, 80]),
    { scale: 2 },
    t([1, 1, 1, 2], [-1e30, 0]),
  );
  assertEquals(
    [...twoHeads.data].map((v) => Math.round(v)),
    [30, 40, 70, 80],
  );
  // The unified entry point also passes the fourth input through (forgetting it silently returns
  // the unmasked values)
  assertEquals(
    [
      ...applyReferenceOp("attention", [q, key, value, t([1, 1, 1, 2], [0, -1e30])], { scale: 2 })
        .data,
    ],
    [...masked.data],
  );
  // A mask violating the contract (a shape other than [1,1,M,N]) fails on the oracle side as well
  assertThrows(
    () => referenceAttention(q, key, value, { scale: 2 }, t([1, 1, 2, 2], [0, 0, 0, 0])),
    OpContractError,
  );
});

test("the unified entry point rejects attrs keys the contract rejects instead of using them (the same match as the node contract check)", () => {
  const q = t([1, 1, 1, 2], [1, 0]);
  const key = t([1, 1, 2, 2], [1, 0, 0, 1]);
  const value = t([1, 1, 2, 2], [10, 20, 30, 40]);
  // window is for the states form only. Without the match, the window would be silently ignored and
  // the values of the full context returned.
  assertThrows(
    () => applyReferenceOp("attention", [q, key, value], { scale: 0.5, window: 2 }),
    OpContractError,
    "attrs.window can be declared only on a node with a states field",
  );
  assertThrows(
    () =>
      applyReferenceOp(
        "conv1d",
        [t([1, 1, 4], [1, 2, 3, 4]), t([1, 1, 3], [1, 1, 1]), t([1], [0])],
        { ...CONV, extra: 1 },
      ),
    OpContractError,
    "attrs outside the contract [extra]",
  );
});

test("embedding re-reads rows and rejects out-of-range indices", () => {
  const weight = t([3, 2], [1, 2, 3, 4, 5, 6]);
  const out = referenceEmbedding(weight, i32([2, 2], [2, 0, 1, 2]));
  assertEquals(out.shape, [2, 2, 2]);
  assertEquals([...out.data], [5, 6, 1, 2, 3, 4, 5, 6]);
  // padding_idx has no effect on forward (it is not taken, so it cannot have any)
  assertEquals(
    [...applyReferenceOp("embedding", [weight, i32([1], [0])], { padding_idx: 0 }).data],
    [1, 2],
  );
  // MUST: out of range throws (not relaxed to match the GPU side's NaN poisoning)
  assertThrows(() => referenceEmbedding(weight, i32([1], [3])), ReferenceOpError);
  assertThrows(() => referenceEmbedding(weight, i32([1], [-1])), ReferenceOpError);
});

test("masked_fill fills only where the mask is true and broadcasts the mask right-aligned", () => {
  const out = referenceMaskedFill(t([2, 3], [1, 2, 3, 4, 5, 6]), bools([3], [1, 0, 1]), {
    value: -7.5,
  });
  assertEquals(out.shape, [2, 3]);
  assertEquals([...out.data], [-7.5, 2, -7.5, -7.5, 5, -7.5]);
  // The observed fill value (the smallest finite f32) is not moved by rounding
  const extreme = referenceMaskedFill(t([2], [1, 2]), bools([2], [1, 0]), {
    value: -3.4028234663852886e38,
  });
  assertEquals([...extreme.data], [-3.4028234663852886e38, 2]);
  // A mask that would widen x is rejected by the contract (the fill value does not leak into
  // elements that do not exist)
  assertThrows(
    () => referenceMaskedFill(t([1], [1]), bools([3], [1, 0, 1]), { value: 0 }),
    OpContractError,
  );
});

const CONV = { stride: 1, padding: 0, dilation: 1, groups: 1 } as const;

test("conv1d treats padding as zero fill, and stride determines the output length", () => {
  // x[1,1,4] = [1,2,3,4] / W[1,1,3] = [1,1,1] / bias = 0, stride 1 / padding 1
  const same = referenceConv1d(t([1, 1, 4], [1, 2, 3, 4]), t([1, 1, 3], [1, 1, 1]), t([1], [0]), {
    ...CONV,
    padding: 1,
  });
  assertEquals(same.shape, [1, 1, 4]);
  // Both ends skip the padding area ([0+1+2, 1+2+3, 2+3+4, 3+4+0])
  assertEquals([...same.data], [3, 6, 9, 7]);
  // With padding 0 the output length shrinks to 2 (a different shape)
  const valid = referenceConv1d(
    t([1, 1, 4], [1, 2, 3, 4]),
    t([1, 1, 3], [1, 1, 1]),
    t([1], [0]),
    CONV,
  );
  assertEquals(valid.shape, [1, 1, 2]);
  assertEquals([...valid.data], [6, 9]);
  // stride 2 (output length floor((4+0−3)/2)+1 = 1) and bias
  const strided = referenceConv1d(
    t([1, 1, 4], [1, 2, 3, 4]),
    t([1, 1, 3], [1, 1, 1]),
    t([1], [100]),
    { ...CONV, stride: 2 },
  );
  assertEquals(strided.shape, [1, 1, 1]);
  assertEquals([...strided.data], [106]);
  // Contracts per input channel (Cin=2 / Cout=2)
  const multi = referenceConv1d(
    t([1, 2, 3], [1, 2, 3, 10, 20, 30]),
    t([2, 2, 1], [1, 0, 0, 1]),
    t([2], [0, 0]),
    CONV,
  );
  assertEquals(multi.shape, [1, 2, 3]);
  assertEquals([...multi.data], [1, 2, 3, 10, 20, 30]);
});

test("conv1d's dilation spaces out kernel positions, and groups narrows the input channel band", () => {
  // dilation 2 / k 3 sees [x0, x2, x4] (span 5 — output length L − 4)
  const dilated = referenceConv1d(
    t([1, 1, 7], [1, 2, 3, 4, 5, 6, 7]),
    t([1, 1, 3], [1, 1, 1]),
    t([1], [0]),
    { ...CONV, dilation: 2 },
  );
  assertEquals(dilated.shape, [1, 1, 3]);
  assertEquals([...dilated.data], [1 + 3 + 5, 2 + 4 + 6, 3 + 5 + 7]);
  // MUST: dilation applies to k (applying it to ox has the same form as stride, distinguishable
  // only in this case where "the input length stays and only the output length shrinks")
  const strided = referenceConv1d(
    t([1, 1, 7], [1, 2, 3, 4, 5, 6, 7]),
    t([1, 1, 3], [1, 1, 1]),
    t([1], [0]),
    { ...CONV, stride: 2 },
  );
  assertEquals([...strided.data], [1 + 2 + 3, 3 + 4 + 5, 5 + 6 + 7]);

  // depthwise (groups = Cin = Cout = 2) — each output channel sees only the channel of the same
  // number. Reading across groups always shows in the values as "another channel's values mixed
  // in".
  const depthwise = referenceConv1d(
    t([1, 2, 3], [1, 2, 3, 10, 20, 30]),
    t([2, 1, 1], [1, 100]),
    t([2], [0, 0]),
    { ...CONV, groups: 2 },
  );
  assertEquals(depthwise.shape, [1, 2, 3]);
  assertEquals([...depthwise.data], [1, 2, 3, 1000, 2000, 3000]);

  // Intermediate groups (Cin 4 / Cout 2 / g 2) — the weights' second axis is Cin/groups = 2
  const grouped = referenceConv1d(
    t([1, 4, 2], [1, 2, 3, 4, 10, 20, 30, 40]),
    t([2, 2, 1], [1, 1, 1, 1]),
    t([2], [0, 0]),
    { ...CONV, groups: 2 },
  );
  assertEquals(grouped.shape, [1, 2, 2]);
  assertEquals([...grouped.data], [1 + 3, 2 + 4, 10 + 30, 20 + 40]);
});

/** Default conv2d attrs (each case replaces only the axes it needs). */
const CONV2D = {
  stride: [1, 1],
  padding: [0, 0],
  dilation: [1, 1],
  groups: 1,
} as const;

// MUST: the weights are [Cout, Cin/groups, Kh, Kw]. An implementation that reads Kh and Kw swapped
// **matches even in value for a square kernel**, so hand-computed cases always use Kh ≠ Kw and
// asymmetric weights.
test("conv2d convolves with [Cout,Cin/g,Kh,Kw] weights (hand-computed, Kh≠Kw)", () => {
  // x [[1,2,3],[4,5,6]] × w [[1,2,3],[4,5,6]] (Kh=2 / Kw=3) → one output point
  // Σ = 1·1 + 2·2 + 3·3 + 4·4 + 5·5 + 6·6 = 91.
  // Reading Kh/Kw transposed gives 1·1 + 2·3 + 3·5 + 4·2 + 5·4 + 6·6 = 86, which always disagrees.
  const exact = referenceConv2d(
    t([1, 1, 2, 3], [1, 2, 3, 4, 5, 6]),
    t([1, 1, 2, 3], [1, 2, 3, 4, 5, 6]),
    t([1], [0]),
    CONV2D,
  );
  assertEquals(exact.shape, [1, 1, 1, 1]);
  assertEquals([...exact.data], [91]);
  // bias is added once per output channel
  const biased = referenceConv2d(
    t([1, 1, 2, 3], [1, 2, 3, 4, 5, 6]),
    t([1, 1, 2, 3], [1, 2, 3, 4, 5, 6]),
    t([1], [9]),
    CONV2D,
  );
  assertEquals([...biased.data], [100]);
});

// MUST: the two spatial axes act independently. With only cases that use the same value for H and
// W, a swapped stride / padding / dilation would not go red even in the numbers.
test("conv2d's stride / padding / dilation act independently on H and W", () => {
  // A 3×4 input and distinguishable 2×2 weights (1 / 10 / 100 / 1000), so the values show which tap
  // applied
  const x = t([1, 1, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  const taps = t([1, 1, 2, 2], [1, 10, 100, 1000]);
  // stride [2,1] / padding [1,0]: H is (3+2−1−1)/2+1 = 2, W is (4−1−1)/1+1 = 3
  const asym = referenceConv2d(x, taps, t([1], [0]), {
    ...CONV2D,
    stride: [2, 1],
    padding: [1, 0],
  });
  assertEquals(asym.shape, [1, 1, 2, 3]);
  assertEquals([...asym.data], [2100, 3200, 4300, 10965, 12076, 13187]);

  // dilation [2,1]: only H's kernel span grows to 3 (H is 1, W is 4)
  const dilated = referenceConv2d(
    t([1, 1, 3, 5], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
    taps,
    t([1], [0]),
    { ...CONV2D, dilation: [2, 1] },
  );
  assertEquals(dilated.shape, [1, 1, 1, 4]);
  assertEquals([...dilated.data], [13121, 14232, 15343, 16454]);

  // The padding area is zero-filled and **skipped** (a shape with outputs at both ends that see
  // only padding)
  const padded = referenceConv2d(t([1, 1, 1, 1], [7]), t([1, 1, 1, 1], [3]), t([1], [0]), {
    ...CONV2D,
    padding: [1, 0],
  });
  assertEquals(padded.shape, [1, 1, 3, 1]);
  assertEquals([...padded.data], [0, 21, 0]);
});

// MUST: include a case where Cin and Cout are both at least 2 and differ (if either is 1, an error
// that drops the group's band offset matches by chance — the lesson from conv_transpose1d).
test("conv2d's groups narrows the input channel band (depthwise and intermediate groups)", () => {
  // Cin 4 / Cout 6 / groups 2 → in_per_group 2 / out_per_group 3.
  // oc 0..2 see ic 0,1, and oc 3..5 see only ic 2,3.
  const grouped = referenceConv2d(
    t([1, 4, 1, 2], [1, 2, 3, 4, 5, 6, 7, 8]),
    t([6, 2, 1, 1], [1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1]),
    t([6], [0, 0, 0, 0, 0, 0]),
    { ...CONV2D, groups: 2 },
  );
  assertEquals(grouped.shape, [1, 6, 1, 2]);
  // Dropping the band offset turns oc 3..5 into [1,2] / [3,4] / [4,6]
  assertEquals([...grouped.data], [1, 2, 3, 4, 4, 6, 5, 6, 7, 8, 12, 14]);

  // depthwise (groups = Cin = Cout = 3 → the weights' second axis is 1)
  const depthwise = referenceConv2d(
    t([1, 3, 1, 1], [1, 1, 1]),
    t([3, 1, 1, 1], [2, 3, 4]),
    t([3], [0, 0, 0]),
    { ...CONV2D, groups: 3 },
  );
  assertEquals([...depthwise.data], [2, 3, 4]);
});

/** Default conv3d attrs (each case replaces only the axes it needs — ADR 0118 decision 1). */
const CONV3D = {
  stride: [1, 1, 1],
  padding: [0, 0, 0],
  dilation: [1, 1, 1],
  groups: 1,
} as const;

// MUST: the weights are [Cout, Cin/groups, Kt, Kh, Kw]. An implementation that reads the axes
// swapped **matches even in value between axes of the same length**, so hand-computed cases use
// Kt ≠ Kh ≠ Kw and asymmetric weights. x is unbatched rank 4 ([Cin, T, H, W]).
test("conv3d convolves with [Cout,Cin/g,Kt,Kh,Kw] weights (hand-computed, Kt≠Kh≠Kw, unbatched)", () => {
  // x[t][h] = 1..6 (T=2, H=3, W=1) × w[kt][kh] = 1..6 (Kt=2, Kh=3, Kw=1) → one output point
  // Σ = 1·1 + 2·2 + … + 6·6 = 91. Reading Kt and Kh swapped gives
  // 1·1 + 2·3 + 3·5 + 4·2 + 5·4 + 6·6 = 86, which always disagrees.
  const exact = referenceConv3d(
    t([1, 2, 3, 1], [1, 2, 3, 4, 5, 6]),
    t([1, 1, 2, 3, 1], [1, 2, 3, 4, 5, 6]),
    t([1], [0]),
    CONV3D,
  );
  assertEquals(exact.shape, [1, 1, 1, 1]);
  assertEquals([...exact.data], [91]);
  // bias is added once per output channel
  const biased = referenceConv3d(
    t([1, 2, 3, 1], [1, 2, 3, 4, 5, 6]),
    t([1, 1, 2, 3, 1], [1, 2, 3, 4, 5, 6]),
    t([1], [9]),
    CONV3D,
  );
  assertEquals([...biased.data], [100]);
});

// MUST: the 3 axes act independently. With only cases that use the same value on two axes, a
// swapped stride / padding / dilation would not go red even in the numbers (each case moves only
// one axis, and the output shape changes per axis too).
test("conv3d's stride / padding / dilation act independently on T / H / W", () => {
  // stride T = 2: Tout = (4−1−1)/2 + 1 = 2. out[ot] = x[2ot]·1 + x[2ot+1]·10
  const strided = referenceConv3d(
    t([1, 4, 1, 1], [1, 2, 3, 4]),
    t([1, 1, 2, 1, 1], [1, 10]),
    t([1], [0]),
    { ...CONV3D, stride: [2, 1, 1] },
  );
  assertEquals(strided.shape, [1, 2, 1, 1]);
  assertEquals([...strided.data], [21, 43]);

  // dilation T = 2: x[t][h] = 3t + h + 1 (T=3, H=3) × w[kt][kh] = [[1,10],[100,1000]] →
  // Tout = 3 − 2·1 = 1, Hout = 3 − 1 = 2. out[oy] = Σ x[2kt][oy+kh]·w[kt][kh]
  // oy=0: 1 + 2·10 + 7·100 + 8·1000 = 8721 / oy=1: 2 + 3·10 + 8·100 + 9·1000 = 9832
  const dilated = referenceConv3d(
    t([1, 3, 3, 1], [1, 2, 3, 4, 5, 6, 7, 8, 9]),
    t([1, 1, 2, 2, 1], [1, 10, 100, 1000]),
    t([1], [0]),
    { ...CONV3D, dilation: [2, 1, 1] },
  );
  assertEquals(dilated.shape, [1, 1, 2, 1]);
  assertEquals([...dilated.data], [8721, 9832]);

  // padding T = 2 (Kt=3, T=1): the padding area is zero-filled and **skipped** — every output reads
  // the one real frame exactly once through a different tap (out[0] is w[2], out[1] is w[1], out[2]
  // is w[0])
  const padded = referenceConv3d(t([1, 1, 1, 1], [7]), t([1, 1, 3, 1, 1], [1, 3, 5]), t([1], [0]), {
    ...CONV3D,
    padding: [2, 0, 0],
  });
  assertEquals(padded.shape, [1, 3, 1, 1]);
  assertEquals([...padded.data], [35, 21, 7]);

  // stride W = 2, padding H = 1: Hout = 2 + 2 − 1 = 4, Wout = (5−1−1)/2 + 1 = 2. Rows −1 and 2 are
  // padding
  const mixed = referenceConv3d(
    t([1, 1, 2, 5], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    t([1, 1, 1, 1, 2], [1, 10]),
    t([1], [0]),
    { ...CONV3D, stride: [1, 1, 2], padding: [0, 1, 0] },
  );
  assertEquals(mixed.shape, [1, 1, 4, 2]);
  assertEquals([...mixed.data], [0, 0, 21, 43, 76, 98, 0, 0]);
});

// MUST: include a case where Cin and Cout are both at least 2 and differ (if either is 1, an error
// that drops the group's band offset matches by chance — the same lesson as conv2d).
test("conv3d's groups narrows the input channel band (general groups semantics — intermediate groups and depthwise)", () => {
  // Cin 4 / Cout 6 / groups 2 → oc 0..2 see ic 0,1, and oc 3..5 see only ic 2,3 (T=1, H=1, W=2)
  const grouped = referenceConv3d(
    t([4, 1, 1, 2], [1, 2, 3, 4, 5, 6, 7, 8]),
    t([6, 2, 1, 1, 1], [1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1]),
    t([6], [0, 0, 0, 0, 0, 0]),
    { ...CONV3D, groups: 2 },
  );
  assertEquals(grouped.shape, [6, 1, 1, 2]);
  // Dropping the band offset turns oc 3..5 into [1,2] / [3,4] / [4,6]
  assertEquals([...grouped.data], [1, 2, 3, 4, 4, 6, 5, 6, 7, 8, 12, 14]);

  // depthwise (groups = Cin = Cout = 3 → the weights' second axis is 1). A temporal kernel of 2
  // reads 2 frames per band
  const depthwise = referenceConv3d(
    t([3, 2, 1, 1], [1, 2, 3, 4, 5, 6]),
    t([3, 1, 2, 1, 1], [1, 10, 2, 20, 3, 30]),
    t([3], [0, 0, 0]),
    { ...CONV3D, groups: 3 },
  );
  assertEquals(depthwise.shape, [3, 1, 1, 1]);
  assertEquals([...depthwise.data], [1 + 20, 6 + 80, 15 + 180]);
});

// A conv3d with Kt = 1 is "a conv2d with the frames placed in conv2d's batch", down to the order of
// the contraction (ic, kh, kw), so it matches the **independently implemented** referenceConv2d
// exactly (the CPU version of identity check ① on the GPU side).
test("conv3d (Kt = 1) matches per-frame conv2d exactly (comparing independent implementations)", () => {
  const [cin, time, height, width, cout] = [3, 4, 5, 6, 2];
  const x = t(
    [cin, time, height, width],
    Array.from({ length: cin * time * height * width }, (_, i) => ((i * 7) % 23) / 8 - 1.25),
  );
  const w = t(
    [cout, cin, 1, 3, 2],
    Array.from({ length: cout * cin * 3 * 2 }, (_, i) => ((i * 11) % 19) / 16 - 0.5),
  );
  const b = t([cout], [0.375, -0.625]);
  const out3d = referenceConv3d(x, w, b, {
    stride: [1, 2, 1],
    padding: [0, 1, 0],
    dilation: [1, 1, 2],
    groups: 1,
  });
  // x [Cin,T,H,W] → [T,Cin,H,W] (frames into the batch), weights are [Cout,Cin,Kh,Kw] with Kt
  // dropped
  const frames = new Float32Array(x.data.length);
  for (let c = 0; c < cin; c += 1) {
    for (let f = 0; f < time; f += 1) {
      for (let i = 0; i < height * width; i += 1) {
        frames[(f * cin + c) * height * width + i] = x.data[(c * time + f) * height * width + i];
      }
    }
  }
  const out2d = referenceConv2d(
    t([time, cin, height, width], [...frames]),
    t([cout, cin, 3, 2], [...w.data]),
    b,
    {
      stride: [2, 1],
      padding: [1, 0],
      dilation: [1, 2],
      groups: 1,
    },
  );
  const [, tOut, hOut, wOut] = out3d.shape;
  assertEquals(out2d.shape, [time, cout, hOut, wOut]);
  assertEquals(tOut, time);
  for (let oc = 0; oc < cout; oc += 1) {
    for (let f = 0; f < time; f += 1) {
      for (let i = 0; i < hOut * wOut; i += 1) {
        assertEquals(
          out3d.data[(oc * time + f) * hOut * wOut + i],
          out2d.data[(f * cout + oc) * hOut * wOut + i],
          `oc ${oc} / frame ${f} / pixel ${i}`,
        );
      }
    }
  }
});

test("conv_transpose1d stretches the input by stride with [Cin,Cout,K] weights", () => {
  // x[1,1,3] = [1,2,3] / W[1,1,2] = [1,10] / stride 2 / padding 0 → output length 6
  // out[2i] = x[i]·w[0] / out[2i+1] = x[i]·w[1]
  const up = referenceConvTranspose1d(t([1, 1, 3], [1, 2, 3]), t([1, 1, 2], [1, 10]), t([1], [0]), {
    stride: 2,
    padding: 0,
  });
  assertEquals(up.shape, [1, 1, 6]);
  assertEquals([...up.data], [1, 10, 2, 20, 3, 30]);

  // MUST: the weights are [Cin, Cout, K]. Asymmetric channels (Cin 2 / Cout 1) pin a transposition
  // mix-up. Read transposed, w[1] is picked up at w[0]'s position and the values are no longer
  // [1+20, …].
  const asym = referenceConvTranspose1d(
    t([1, 2, 2], [1, 2, 10, 20]),
    t([2, 1, 2], [1, 0, 0, 1]),
    t([1], [0]),
    { stride: 2, padding: 0 },
  );
  assertEquals(asym.shape, [1, 1, 4]);
  // ic=0 is w=[1,0] (to even positions) / ic=1 is w=[0,1] (to odd positions)
  assertEquals([...asym.data], [1, 10, 2, 20]);

  // padding trims both ends of the output (2P == K − S: with K 4 / S 2 / P 1 the output length is
  // L·S)
  const padded = referenceConvTranspose1d(
    t([1, 1, 3], [1, 2, 3]),
    t([1, 1, 4], [1, 2, 3, 4]),
    t([1], [0]),
    { stride: 2, padding: 1 },
  );
  assertEquals(padded.shape, [1, 1, 6]);
  // Hand calculation (out[ox] = Σ_k x[(ox+1−k)/2]·w[k], only when (ox+1−k) is a non-negative even
  // number with quotient < 3):
  //   ox0: x0·w1                       = 2
  //   ox1: x1·w0 + x0·w2               = 2 + 3   = 5
  //   ox2: x1·w1 + x0·w3               = 4 + 4   = 8
  //   ox3: x2·w0 + x1·w2               = 3 + 6   = 9
  //   ox4: x2·w1 + x1·w3               = 6 + 8   = 14
  //   ox5: x2·w2                       = 9
  // Treating padding as 0 shifts everything one to the left (the asymmetry of the two ends is the
  // detector).
  assertEquals([...padded.data], [2, 5, 8, 9, 14, 9]);

  // MUST: keep one case where Cin and Cout are **both at least 2 and differ**. A transposed read
  // has matching indices whether Cin == 1 or Cout == 1, so the two cases above alone stay green
  // (shown by fault injection — ADR 0015's "pin with asymmetric channel counts" includes this).
  const wide = referenceConvTranspose1d(
    t([1, 2, 2], [1, 2, 10, 20]),
    // W[ic][oc][k] = 1..12 (ic outermost, then oc, k innermost)
    t([2, 3, 2], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    t([3], [0, 0, 0]),
    { stride: 2, padding: 0 },
  );
  assertEquals(wide.shape, [1, 3, 4]);
  // Hand calculation: out[oc][2i+j] = x0[i]·W[0][oc][j] + x1[i]·W[1][oc][j]
  assertEquals(
    [...wide.data],
    [
      71,
      82,
      142,
      164, // oc0
      93,
      104,
      186,
      208, // oc1
      115,
      126,
      230,
      252, // oc2
    ],
  );

  // bias is added once per output channel (not in proportion to the number of contributions)
  const biased = referenceConvTranspose1d(
    t([1, 1, 2], [1, 1]),
    t([1, 2, 2], [1, 1, 1, 1]),
    t([2], [5, 7]),
    { stride: 2, padding: 0 },
  );
  assertEquals([...biased.data], [6, 6, 6, 6, 8, 8, 8, 8]);
});

// deform_conv2d (ADR 0055). Narrowed to a 1×1 kernel and a single channel, the output is exactly
// "mask · bilinear(x, base + offset)", so the three offset branches (inside / only a corner out of
// range / the center out of range) can be pinned by hand calculation.
test("deform_conv2d's 1×1 branch exercises the three offset branches as hand-computed", () => {
  const x = t([1, 1, 2, 2], [1, 2, 3, 4]);
  const weight = t([1, 1, 1, 1], [1]);
  const bias = t([1], [0]);
  const ones = t([1, 1, 2, 2], [1, 1, 1, 1]);
  /**
   * The offset is `[y plane: 4 elements, x plane: 4 elements]` (even channels = y / odd = x).
   */
  const offset = (shiftY: number, shiftX: number): RefTensor =>
    t([1, 2, 2, 2], [shiftY, shiftY, shiftY, shiftY, shiftX, shiftX, shiftX, shiftX]);
  const run = (off: RefTensor, mask: RefTensor = ones): readonly number[] => [
    ...applyReferenceOp("deform_conv2d", [x, weight, off, mask, bias], { padding: [0, 0] }).data,
  ];
  // Offset 0 is a plain 1×1 conv = identity (mask 1, bias 0)
  assertEquals(run(offset(0, 0)), [1, 2, 3, 4]);
  // +0.5 on y: oy=0 is the midpoint of rows 0/1, oy=1 is at y=1.5 with **only the lower corners out
  // of range** (zero fill)
  assertEquals(run(offset(0.5, 0)), [2, 3, 1.5, 2]);
  // −1.5 on y: oy=0 has its center at −1.5 ≤ −1, so **the whole tap is 0**; oy=1 is at y=−0.5 with
  // only the upper corners 0
  assertEquals(run(offset(-1.5, 0)), [0, 0, 0.5, 1]);
  // MUST: even = y / odd = x. +1 on x shifts along columns and gives a different value from +1 on y
  assertEquals(run(offset(0, 1)), [2, 0, 4, 0]);
  assertEquals(run(offset(1, 0)), [3, 4, 0, 0]);
  // The modulator applies **after** interpolation (the top of BiRefNet's range [0,2])
  assertEquals(run(offset(0.5, 0), t([1, 1, 2, 2], [2, 2, 2, 2])), [4, 6, 3, 4]);
});

// GRU hidden-side scan (ADR 0056). Setting `W_hh = 0` removes the hidden-side contraction, leaving
// only **the gate order (r / z / n), what the reset gate applies to, and how time advances**, which
// can then be pinned by hand calculation (the contraction itself is covered by the CPU reference
// comparison in gpu_ops_test).
//
// MUST: make `b_hh` non-zero. The reset gate applies to **the hidden-side product including b_hh**
// (`n = tanh(i_n + (Σ W·h + b_n)·r)`), so the error of moving the bias outside reset goes red only
// here.
test("gru_scan with W_hh = 0 follows the gate order and time steps as hand-computed", () => {
  const hidden = 1;
  // gi[T=2, N=1, 3H] = [i_r, i_z, i_n] × 2 steps (different values per gate = an order mix-up
  // shows)
  const gi = t([2, 1, 3], [0.5, -0.25, 0.75, -1.5, 0.125, -0.5]);
  const h0 = t([1, 1], [0.4]);
  const weight = t([3, 1], [0, 0, 0]);
  const bias = t([3], [0.1, -0.2, 0.3]);
  /**
   * Expected values solving the contract's formula directly in JS f64 (the f32 rounding sequence is
   * the reference's job).
   */
  const advance = (state: number, step: number): number => {
    const gate = (index: number): number => gi.data[step * 3 + index];
    const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));
    const reset = sigmoid(bias.data[0] + gate(0));
    const update = sigmoid(bias.data[1] + gate(1));
    const candidate = Math.tanh(gate(2) + bias.data[2] * reset);
    return (state - candidate) * update + candidate;
  };
  const first = advance(0.4, 0);
  const second = advance(first, 1);

  const forward = applyReferenceOp("gru_scan", [gi, h0, weight, bias]);
  assertEquals(forward.shape, [2, 1, hidden]);
  assertAlmostEquals(forward.data[0], first, 1e-6);
  assertAlmostEquals(forward.data[1], second, 1e-6);

  // The reverse direction scans from the end, so the initial state h0 is received at **t = 1**. The
  // output stays in forward time order (no `flip` in between = a direct check that the op folds the
  // scan direction).
  const reverseFirst = advance(0.4, 1);
  const reverseSecond = advance(reverseFirst, 0);
  const backward = applyReferenceOp("gru_scan_reverse", [gi, h0, weight, bias]);
  assertAlmostEquals(backward.data[1], reverseFirst, 1e-6);
  assertAlmostEquals(backward.data[0], reverseSecond, 1e-6);
});

// A check that the state **does not mix across batches** and that `W_hh` really applies only to the
// hidden side. With only batch 0's h0 non-zero, W_hh's contribution appears only in that batch.
test("gru_scan applies W_hh only to each batch's own state", () => {
  const gi = t([1, 2, 6], [0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3]);
  const weight = t([6, 2], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
  const bias = t([6], [0, 0, 0, 0, 0, 0]);
  // Only batch 0's state is non-zero (gi is identical across batches, so any difference can only
  // come from W_hh·h)
  const mixed = applyReferenceOp("gru_scan", [gi, t([2, 2], [0.7, -0.3, 0, 0]), weight, bias]);
  const zeroed = applyReferenceOp("gru_scan", [gi, t([2, 2], [0, 0, 0, 0]), weight, bias]);
  assertEquals(mixed.shape, [1, 2, 2]);
  // Batch 1 (the last 2 elements) has h0 = 0 and matches in both; batch 0 does not match
  assertEquals([...mixed.data.slice(2)], [...zeroed.data.slice(2)]);
  assertNotEquals([...mixed.data.slice(0, 2)], [...zeroed.data.slice(0, 2)]);
});

// The degenerate case (offset all 0, mask all 1) **matches plain conv2d exactly**. It is the only
// A/B oracle for the new primitive, and the real-GPU side (gpu_deform_conv2d_test.ts) checks it bit
// for bit.
// MUST: pin with an asymmetric shape (Cin ≠ Cout / Kh ≠ Kw / padding H ≠ W) — a symmetric shape
// does not show a mix-up of the weight layout or the offset's channel order in the values.
test("deform_conv2d matches conv2d exactly with offset 0 and mask 1", () => {
  const series = (count: number, step: number, base: number): readonly number[] =>
    Array.from({ length: count }, (_unused, index) => base + step * ((index % 13) - 6));
  const x = t([1, 2, 3, 4], series(24, 0.25, 0.5));
  const weight = t([3, 2, 2, 3], series(36, 0.125, -0.25));
  const bias = t([3], [0.5, -0.25, 0.75]);
  // padding [1,0] → Hout = 3 + 2 − 1 = 4 / Wout = 4 + 0 − 2 = 2
  const offset = t([1, 12, 4, 2], new Array(96).fill(0));
  const mask = t([1, 6, 4, 2], new Array(48).fill(1));
  const deform = applyReferenceOp("deform_conv2d", [x, weight, offset, mask, bias], {
    padding: [1, 0],
  });
  const plain = applyReferenceOp("conv2d", [x, weight, bias], {
    stride: [1, 1],
    padding: [1, 0],
    dilation: [1, 1],
    groups: 1,
  });
  assertEquals(deform.shape, [1, 3, 4, 2]);
  assertEquals(deform.shape, plain.shape);
  assertEquals([...deform.data], [...plain.data]);
});

// An offset NaN is not "out of range". With only the positive-form range test (`> −1 && < in`), NaN
// falls to the false side = a 0 contribution and becomes a silent wrong value, so NaN propagates to
// the output (ADR 0055 decision 5).
test("deform_conv2d propagates a NaN offset instead of dropping it to 0", () => {
  const x = t([1, 1, 2, 2], [1, 2, 3, 4]);
  const weight = t([1, 1, 1, 1], [1]);
  const bias = t([1], [0]);
  const mask = t([1, 1, 2, 2], [1, 1, 1, 1]);
  // Only the first element of the y plane is NaN (the rest 0) → only the first output element is
  // NaN
  const offset = t([1, 2, 2, 2], [Number.NaN, 0, 0, 0, 0, 0, 0, 0]);
  const out = applyReferenceOp("deform_conv2d", [x, weight, offset, mask, bias], {
    padding: [0, 0],
  });
  assertEquals(Number.isNaN(out.data[0]), true, "a NaN offset propagates to the output");
  assertEquals([...out.data.slice(1)], [2, 3, 4], "other elements are not affected");
  // ±Inf is "out of range = 0" as in torch (handled separately from NaN)
  const infinite = t(
    [1, 2, 2, 2],
    [Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, 0, 0, 0, 0, 0, 0],
  );
  const clipped = applyReferenceOp("deform_conv2d", [x, weight, infinite, mask, bias], {
    padding: [0, 0],
  });
  assertEquals([...clipped.data], [0, 0, 3, 4]);
});

// MUST: pin only with shapes whose 4 neighbors and weights can be followed by hand (the new
// primitive has no A/B oracle, so the check is that the reference itself "reads as obviously
// correct"). All values are exact in f32.
test("upsample_bilinear2d mixes the 4 neighbors of align_corners as hand-computed", () => {
  // 2×2 → 3×3. scale = (2−1)/(3−1) = 0.5, so even output positions are the input as is and
  // odd positions are the midpoint of two adjacent points.
  const up = applyReferenceOp("upsample_bilinear2d", [t([1, 1, 2, 2], [1, 2, 3, 4])], {
    output_size: [3, 3],
  });
  assertEquals(up.shape, [1, 1, 3, 3]);
  assertEquals([...up.data], [1, 1.5, 2, 2, 2.5, 3, 3, 3.5, 4]);

  // 3×3 → 2×2 (downscale). scale = 2, so λ is 0 and **only the 2 points at the ends are read**
  // (information loss exactly as torch specifies, unlike antialias / area).
  const down = applyReferenceOp(
    "upsample_bilinear2d",
    [t([1, 1, 3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9])],
    { output_size: [2, 2] },
  );
  assertEquals([...down.data], [1, 3, 7, 9]);

  // Input height 1 (same type as ASPP's GAP branch). H's scale is 0, so every row reads the same
  // row, and only W is really interpolated.
  const broadcast = applyReferenceOp("upsample_bilinear2d", [t([1, 1, 1, 2], [10, 20])], {
    output_size: [2, 3],
  });
  assertEquals([...broadcast.data], [10, 15, 20, 10, 15, 20]);
});

// The definition of align_corners = True: the output's 4 corners match the input's 4 corners. The
// exact match holds only where `fl(fl((in−1)/(out−1)) · (out−1))` round-trips to `in−1` (λ does not
// generally collapse to 0 / 1; for a shape like 2→42 the last λ is 0.99999994 — torch gives the
// same value). The 2→5 / 3→7 pinned here are on the round-tripping side. H and W have different
// lengths so an axis mix-up is checked at the same time.
test("upsample_bilinear2d matches the output's 4 corners exactly to the input's 4 corners", () => {
  const x = t([1, 1, 2, 3], [1, 2, 3, 4, 5, 6]);
  const out = applyReferenceOp("upsample_bilinear2d", [x], { output_size: [5, 7] });
  assertEquals(out.shape, [1, 1, 5, 7]);
  const at = (row: number, column: number): number => out.data[row * 7 + column];
  assertEquals(at(0, 0), 1, "top left");
  assertEquals(at(0, 6), 3, "top right");
  assertEquals(at(4, 0), 4, "bottom left");
  assertEquals(at(4, 6), 6, "bottom right");
  // The edge **rows and columns** are also the 1-D interpolation of the input's edges (the 4
  // corners alone leave an axis mix-up)
  assertEquals([at(0, 0), at(0, 3), at(0, 6)], [1, 2, 3], "top row");
  assertEquals([at(0, 0), at(2, 0), at(4, 0)], [1, 2.5, 4], "left column");
});

// The 3 tests from here (deform_conv2d / upsample_bilinear2d / gru_scan) are **checks that call
// directly, not through applyReferenceOp**. Unlike the entry-point checks above, they pin the path
// that calls the exported function itself (if the wiring is swapped, the function's own contract
// still goes red independently).

// MUST: the oracle is referenceConv2d (an independent implementation). With a uniform integer
// offset the source coordinate is `oy − ph + kh + dy` = "a conv2d with the padding changed to
// `ph − dy`", so without recomputing deform's bilinear taps, the conv2d values constrain
// ① the taps' integer positions ② the offset's channel order (even = y / odd = x) ③ zero fill
// out of range.
// MUST: make dy ≠ dx. With the same shift, a read that swaps the y / x planes matches in value too.
test("deform_conv2d treats a uniform integer offset as a shifted conv2d window (direct call)", () => {
  // A fixed sequence in steps of 1/8 (exact in f32, with no 0 and no symmetry)
  const series = (count: number, seed: number): readonly number[] =>
    Array.from({ length: count }, (_unused, index) => (((index * 7 + seed) % 17) - 8) / 8);
  // Cin 2 ≠ Cout 3 / Kh 2 ≠ Kw 3 (a shape where a weight layout mix-up shows in the values)
  const x = t([1, 2, 4, 5], series(40, 0));
  const weight = t([3, 2, 2, 3], series(36, 5));
  const bias = t([3], [0.5, -0.25, 0.75]);
  // padding [2,2] → Hout = 4 + 4 − 1 = 7 / Wout = 5 + 4 − 2 = 7
  const attrs = { padding: [2, 2] } as const;
  const mask = t([1, 6, 7, 7], new Array(6 * 49).fill(1));
  /** An offset with the same shift for every tap and pixel (even channels = y / odd = x). */
  const uniformOffset = (shiftY: number, shiftX: number): RefTensor =>
    t(
      [1, 12, 7, 7],
      Array.from({ length: 12 * 49 }, (_unused, index) =>
        Math.floor(index / 49) % 2 === 0 ? shiftY : shiftX,
      ),
    );

  // ① The degenerate case (offset 0, mask 1) matches plain conv2d exactly
  const degenerate = referenceDeformConv2d(x, weight, uniformOffset(0, 0), mask, bias, attrs);
  const plain = referenceConv2d(x, weight, bias, { ...CONV2D, padding: [2, 2] });
  assertEquals(degenerate.shape, [1, 3, 7, 7]);
  assertEquals(degenerate.shape, plain.shape);
  assertEquals([...degenerate.data], [...plain.data]);

  // ② dy = 2 / dx = 1 → the source coordinates are `oy + kh` and `ox − 1 + kw` = a conv2d with
  // padding [0,1]. The conv2d output (3×5) should sit as is in the top-left window of the deform
  // output (7×7).
  const shifted = referenceDeformConv2d(x, weight, uniformOffset(2, 1), mask, bias, attrs);
  const window = referenceConv2d(x, weight, bias, { ...CONV2D, padding: [0, 1] });
  assertEquals(window.shape, [1, 3, 3, 5]);
  const cropped: number[] = [];
  for (let channel = 0; channel < 3; channel += 1) {
    for (let row = 0; row < 3; row += 1) {
      for (let column = 0; column < 5; column += 1) {
        cropped.push(shifted.data[(channel * 7 + row) * 7 + column]);
      }
    }
  }
  // Swapping y / x gives a conv2d with padding [1,0], and this window does not match
  assertEquals(cropped, [...window.data]);
});

// MUST: λ goes in steps of 1/4. In the existing case with a shift of 1/2, `(1 − λ)` and `λ` are
// equal, so an implementation that swaps the 4 corner weights left/right or top/bottom stays green.
// MUST: the input is not a plane (linear in the coordinates). On a plane the bilinear cross term
// vanishes and the values match even if the weights are distributed wrongly (here the 4 points are
// 1,2,3,7 = not a plane).
test("deform_conv2d's 4 corner weights with λ in steps of 1/4 match the hand calculation (direct call)", () => {
  // x[0,0] = [[1,2],[3,7]] / 1×1 kernel w = 2 / mask = 3 / bias = 0.5, padding 0
  const x = t([1, 1, 2, 2], [1, 2, 3, 7]);
  const weight = t([1, 1, 1, 1], [2]);
  const bias = t([1], [0.5]);
  const mask = t([1, 1, 2, 2], [3, 3, 3, 3]);
  const attrs = { padding: [0, 0] } as const;
  /** An offset that shifts only output (0,0) (`[y plane: 4 elements, x plane: 4 elements]`). */
  const corner = (shiftY: number, shiftX: number): RefTensor =>
    t([1, 2, 2, 2], [shiftY, 0, 0, 0, shiftX, 0, 0, 0]);

  // Source (0.25, 0.5): the 4 corner weights (1−λy)(1−λx), (1−λy)λx, λy(1−λx), λyλx = 0.375, 0.375,
  //   0.125, 0.125 → interpolated value = 0.375·1 + 0.375·2 + 0.125·3 + 0.125·7 = 2.375
  //   → 0.5 + (3 · 2.375) · 2 = 14.75 (mask after interpolation, weight after that, bias once)
  // The 3 unshifted pixels are plain samples: 0.5 + 3·x·2 = 12.5 / 18.5 / 42.5
  const tilted = referenceDeformConv2d(x, weight, corner(0.25, 0.5), mask, bias, attrs);
  assertEquals(tilted.shape, [1, 1, 2, 2]);
  assertEquals([...tilted.data], [14.75, 12.5, 18.5, 42.5]);

  // Source (0.5, 0.25) only swaps the λs from above, but on a non-planar input the value differs:
  //   0.375·1 + 0.125·2 + 0.375·3 + 0.125·7 = 2.625 → 0.5 + (3 · 2.625) · 2 = 16.25
  const mirrored = referenceDeformConv2d(x, weight, corner(0.5, 0.25), mask, bias, attrs);
  assertEquals([...mirrored.data], [16.25, 12.5, 18.5, 42.5]);
});

// Constrains the definition of align_corners = True (source coordinate `s = i·(In − 1)/(Out − 1)`)
// with **an input linear in the coordinates**. Bilinear interpolation reproduces a linear function
// exactly (on one axis `(1 − λ)·f(k) + λ·f(k+1) = f(k + λ)`, and likewise on the second), so the
// expected value is "the source coordinate substituted into the linear formula" — independent of
// how the implementation folds it. With align_corners = False (`s = (i + 0.5)·In/Out − 0.5`) the
// source coordinate shifts and it always goes red.
// MUST: make N and C **both 2** so each plane has a different intercept (a mix-up of the plane
// index `n·C + c` does not show if either is 1). H and W get different slopes so an axis mix-up is
// checked too.
test("upsample_bilinear2d reproduces a linear input at align_corners source coordinates (direct call)", () => {
  const heightIn = 3;
  const widthIn = 4;
  /**
   * f(n,c,y,x) = (0.5·n + 0.25·c) + 0.75·y − 0.5·x (linear in the coordinates = a fixed point of
   * bilinear interpolation).
   */
  const value = (item: number, channel: number, y: number, x: number): number =>
    0.5 * item + 0.25 * channel + 0.75 * y - 0.5 * x;
  const input: number[] = [];
  for (let item = 0; item < 2; item += 1) {
    for (let channel = 0; channel < 2; channel += 1) {
      for (let row = 0; row < heightIn; row += 1) {
        for (let column = 0; column < widthIn; column += 1) {
          input.push(value(item, channel, row, column));
        }
      }
    }
  }
  const x = t([2, 2, heightIn, widthIn], input);
  /** The source coordinate of align_corners = True (fixed at 0 for Out = 1). */
  const source = (index: number, lengthIn: number, lengthOut: number): number =>
    lengthOut > 1 ? index * ((lengthIn - 1) / (lengthOut - 1)) : 0;
  const check = (heightOut: number, widthOut: number): void => {
    const up = referenceUpsampleBilinear2d(x, { output_size: [heightOut, widthOut] });
    assertEquals(up.shape, [2, 2, heightOut, widthOut]);
    for (let item = 0; item < 2; item += 1) {
      for (let channel = 0; channel < 2; channel += 1) {
        for (let row = 0; row < heightOut; row += 1) {
          for (let column = 0; column < widthOut; column += 1) {
            const expected = value(
              item,
              channel,
              source(row, heightIn, heightOut),
              source(column, widthIn, widthOut),
            );
            const flat = ((item * 2 + channel) * heightOut + row) * widthOut + column;
            assertAlmostEquals(
              up.data[flat],
              expected,
              1e-6,
              `[${item},${channel},${row},${column}] @ ${heightOut}×${widthOut}`,
            );
          }
        }
      }
    }
  };
  // Both integer factors (scale = exactly 0.5) and non-integer factors (scale = 2/3 and 0.6)
  check(5, 7);
  check(4, 6);
});

// A direct check that includes the W_hh contraction. Setting gi so it exactly cancels the hidden
// side gives r = σ(0) = 0.5 / z = σ(0) = 0.5 / n = tanh(0) = 0, and the update collapses to
// `h' = h·0.5`. The cancellation itself requires ① the row order (gate-major, row `(g·H + j)`)
// ② the order of column `k` ③ b_hh going **inside** reset ④ the state carried between steps, so
// if any one breaks, the σ / tanh arguments move off 0 and the values no longer match (all exact
// in f32).
//
// Hand calculation (h = [2,1], W and b are the literals below):
//   gh_r = [1·2 − 2·1 + 0.5, 0.5·2 + 1·1 − 1] = [0.5, 1]
//   gh_z = [−1·2 + 3·1 + 0.25, 2·2 − 0.5·1 + 0] = [1.25, 3.5]
//   gh_n = [3·2 + 1·1 − 1, −2·2 + 2·1 + 0.5]   = [6, −1.5]
//   → with i_r = −gh_r / i_z = −gh_z / i_n = −gh_n·0.5, h' = [2,1]·0.5 = [1, 0.5]
//   The next step uses the same setting for h = [1, 0.5]: gh_r = [0.5, 0] / gh_z = [0.75, 1.75] /
//   gh_n = [2.5, −0.5] → h'' = [0.5, 0.25]
test("gru_scan follows the W_hh contraction and the carried state as hand-computed (direct call)", () => {
  const weight = t([6, 2], [1, -2, 0.5, 1, -1, 3, 2, -0.5, 3, 1, -2, 2]);
  const bias = t([6], [0.5, -1, 0.25, 0, -1, 0.5]);
  const h0 = t([1, 2], [2, 1]);
  // Each step's gi is [i_r(2), i_z(2), i_n(2)]
  const firstStep = [-0.5, -1, -1.25, -3.5, -3, 0.75] as const;
  const secondStep = [-0.5, 0, -0.75, -1.75, -1.25, 0.25] as const;

  const forward = referenceGruScan(
    "gru_scan",
    t([2, 1, 6], [...firstStep, ...secondStep]),
    h0,
    weight,
    bias,
  );
  assertEquals(forward.shape, [2, 1, 2]);
  assertEquals([...forward.data], [1, 0.5, 0.5, 0.25]);

  // The reverse direction reverses **only the scan order**. Given the same sequence with time
  // swapped, h0 is received on the t = 1 side, and the output keeps the forward time index (step 1
  // at t=1, step 2 at t=0).
  const backward = referenceGruScan(
    "gru_scan_reverse",
    t([2, 1, 6], [...secondStep, ...firstStep]),
    h0,
    weight,
    bias,
  );
  assertEquals([...backward.data], [0.5, 0.25, 1, 0.5]);
});

test("applyReferenceOp branches on the contract table's kind and rejects arity violations", () => {
  assertEquals([...applyReferenceOp("relu", [t([2], [-1, 2])]).data], [0, 2]);
  assertEquals([...applyReferenceOp("sum", [t([1, 2], [3, 4])], { dim: 1 }).data], [7]);
  assertEquals([...applyReferenceOp("cast", [t([2], [1.9, 0])], { to: "i32" }).data], [1, 0]);
  assertThrows(() => applyReferenceOp("add", [t([2], [1, 2])]), ReferenceOpError);
  // Ops with per-slot dtype contracts go through the same entry point (tests can follow the graph
  // as is)
  assertEquals(
    [...applyReferenceOp("bmm", [t([1, 1, 2], [1, 2]), t([1, 2, 1], [3, 4])]).data],
    [11],
  );
  assertEquals(
    [...applyReferenceOp("gather", [t([1, 3], [7, 8, 9]), i32([1, 2], [2, 0])]).data],
    [9, 7],
  );
  // cast's attrs fail on the reference side even without passing the contract check
  assertThrows(() => applyReferenceOp("cast", [t([2], [1, 2])]), OpContractError);
  assertThrows(() => refTensor([2, 2], Float32Array.from([1, 2])), ReferenceOpError);

  // Layout ops (ADR 0011) — reshape / expand need a target shape
  assertEquals(applyReferenceOp("reshape", [t([4], [1, 2, 3, 4])], {}, [2, 2]).shape, [2, 2]);
  const transposed = applyReferenceOp("permute", [t([2, 2], [1, 2, 3, 4])], { dims: [1, 0] });
  assertEquals([...transposed.data], [1, 3, 2, 4]);
  assertEquals(applyReferenceOp("expand", [i32([1, 2], [5, 6])], {}, [2, 2]).shape, [2, 2]);
  // MUST: do not silently guess when no target shape is passed
  assertThrows(() => applyReferenceOp("reshape", [t([4], [1, 2, 3, 4])]), ReferenceOpError);
  assertThrows(() => applyReferenceOp("expand", [i32([1, 2], [5, 6])]), ReferenceOpError);
  assertThrows(() => applyReferenceOp("permute", [t([2, 2], [1, 2, 3, 4])]), OpContractError);

  // Layout group 2 (ADR 0014) — the output shape comes from attrs, so no target shape is needed
  assertEquals(
    [...applyReferenceOp("slice", [t([2, 2], [1, 2, 3, 4])], { dim: 1, start: 1, end: 2 }).data],
    [2, 4],
  );
  assertEquals(
    [...applyReferenceOp("cat", [t([1], [1]), t([2], [2, 3])], { dim: 0 }).data],
    [1, 2, 3],
  );
  assertEquals(
    [...applyReferenceOp("pad", [t([1, 2], [1, 2])], { left: 1, right: 0 }).data],
    [0, 1, 2],
  );
  assertEquals([...applyReferenceOp("flip", [t([3], [1, 2, 3])], { dim: 0 }).data], [3, 2, 1]);
  // Variadic arity has a **lower bound** (a cat with one input is not accepted)
  assertThrows(() => applyReferenceOp("cat", [t([2], [1, 2])], { dim: 0 }), ReferenceOpError);

  // The 3 ops of ADR 0017 go through the same entry point too (the kind exhaustiveness lets the
  // types catch missing wiring, but the execution path is separate)
  assertEquals([...applyReferenceOp("clamp_min", [t([2], [-1, 2])], { min: 0 }).data], [0, 2]);
  // mean(x²) = 9 → rms 3 → 3/3·2 = 2 (eps is 1e-12, small enough to ignore)
  assertAlmostEquals(
    applyReferenceOp("rms_norm", [t([1, 1], [3]), t([1], [2])], { eps: 1e-12 }).data[0],
    2,
    1e-6,
  );
  assertEquals(
    [
      ...applyReferenceOp("conv2d", [t([1, 1, 1, 1], [2]), t([1, 1, 1, 1], [3]), t([1], [1])], {
        stride: [1, 1],
        padding: [0, 0],
        dilation: [1, 1],
        groups: 1,
      }).data,
    ],
    [7],
  );
  assertThrows(
    () => applyReferenceOp("rms_norm", [t([1, 1], [3])], { eps: 1e-6 }),
    ReferenceOpError,
  );
});

test("refTensor rejects negative and non-integer dimensions (looking only at the product, [-1,-1] passes as 1 element)", () => {
  assertThrows(() => refTensor([-1, -1], Float32Array.from([7])), ReferenceOpError, "dimension");
  assertThrows(() => refTensor([0.5, 2], Float32Array.from([7])), ReferenceOpError, "dimension");
  assertThrows(() => refTensor([Number.NaN], new Float32Array(0)), ReferenceOpError, "dimension");
  assertEquals(refTensor([0, 3], new Float32Array(0)).shape, [0, 3]);
});

// The reference must be able to detect contract violations in test data (if the oracle returns a
// value for input outside the contract, a combination the GPU side correctly rejects could be
// written as "the reference has an answer" rather than "does not match the reference"). attrs
// values, input dtypes and ranks go through the same checks of the op contract table.
test("applyReferenceOp rejects attrs values, input dtypes and ranks outside the op contract", () => {
  const weight = t([3, 2], [1, 2, 3, 4, 5, 6]);
  const index = i32([2], [0, 2]);
  // embedding's input order is weight → index (reversed, the dtype check fails first and the attrs
  // value cannot be exercised).
  assertEquals(applyReferenceOp("embedding", [weight, index], { padding_idx: -1 }).shape, [2, 2]);
  assertThrows(
    () => applyReferenceOp("embedding", [weight, index], { padding_idx: "bad" }),
    OpContractError,
    "padding_idx",
  );
  assertThrows(
    () => applyReferenceOp("permute", [i32([2, 2], [1, 2, 3, 4])], { dims: [1, 0] }),
    OpContractError,
    "dtype 'i32'",
  );
  assertThrows(
    () => applyReferenceOp("expand", [t([], [5])], {}, [2, 2]),
    OpContractError,
    "rank 0",
  );
  assertThrows(
    () =>
      applyReferenceOp(
        "sym_prefix_slice",
        [bools([2, 2], [1, 0, 0, 1])],
        { sym: "T", slices: [{ dim: 0, coeff: 1, offset: 0 }] },
        [1, 2],
      ),
    OpContractError,
    "dtype 'bool'",
  );
});

// Ops that touch state have no CPU reference (they are an effect on a context-owned buffer, not a
// value). Letting them through and returning an empty output list would allow a parity test that
// counts as "the reference and the GPU agreed", so this pins that **both** forms of the unified
// entry point (plural and singular) fail here.
test("applyReferenceOp does not let state_append through and fails on the missing CPU reference", () => {
  const x = t([1, 1, 1, 1], [1]);
  assertThrows(
    () => applyReferenceOpOutputs("state_append", [x], {}),
    ReferenceOpError,
    "has no CPU reference",
  );
  assertThrows(
    () => applyReferenceOp("state_append", [x], {}),
    ReferenceOpError,
    "has no CPU reference",
  );
});

test("compareTensors chooses the tolerance by dtype and rejects mixed dtypes", () => {
  assertEquals(EXACT_TOLERANCE, { atol: 0, rtol: 0 });
  // A difference that f32 passes with the default tolerance
  assertEquals(compareTensors(t([1], [1.000001]), t([1], [1])).pass, true);
  // Integers fail on a difference of even 1
  assertEquals(compareTensors(i32([2], [1, 2]), i32([2], [1, 2])).pass, true);
  assertEquals(compareTensors(i32([2], [1, 3]), i32([2], [1, 2])).pass, false);
  assertEquals(compareTensors(bools([2], [0, 1]), bools([2], [0, 0])).pass, false);
  assertThrows(() => compareTensors(i32([1], [1]), t([1], [1])), AllcloseError);
});

test("allclose judges by atol + rtol·|ref|", () => {
  const expected = Float32Array.from([0, 1, 100]);
  assertEquals(allclose(expected, expected).pass, true);
  // Within atol=1e-5 / within the range where rtol=1e-3 applies
  assertEquals(allclose(Float32Array.from([5e-6, 1.0005, 100.05]), expected).pass, true);
  const report = allclose(Float32Array.from([0, 1, 101]), expected);
  assertEquals(report.pass, false);
  assertEquals(report.failCount, 1);
  assertEquals(report.worstIndex, 2);
  assertEquals(formatAllclose(report).startsWith("fail=1"), true);
});

test("allclose fails NaN / Inf on either side", () => {
  const expected = Float32Array.from([1, 2]);
  const withNan = allclose(Float32Array.from([Number.NaN, 2]), expected);
  assertEquals(withNan.pass, false);
  assertEquals(withNan.nonFiniteCount, 1);
  assertEquals(withNan.maxAbsError, Number.POSITIVE_INFINITY);

  const referenceInf = allclose(expected, Float32Array.from([Number.POSITIVE_INFINITY, 2]));
  assertEquals(referenceInf.pass, false);
  assertEquals(referenceInf.nonFiniteCount, 1);

  assertEquals(DEFAULT_TOLERANCE.atol, 1e-5);
  assertEquals(DEFAULT_TOLERANCE.rtol, 1e-3);
  assertThrows(() => allclose(expected, Float32Array.from([1])), AllcloseError);
});

// Checks the tolerance itself. The test `abs > atol + rtol·|ref|` is always false when the right
// side is NaN, passing any error. A negative band is the mix-up "fails even with zero error"; both
// make the check meaningless.
test("allclose rejects NaN / negative tolerances with AllcloseError and accepts atol = +Infinity as a recording-only band", () => {
  const actual = Float32Array.from([999]);
  const expected = Float32Array.from([0]);
  for (const tolerance of [
    { atol: Number.NaN, rtol: 0 },
    { atol: 0, rtol: Number.NaN },
    { atol: -1e-5, rtol: 1e-3 },
    { atol: 1e-5, rtol: -1e-3 },
  ]) {
    assertThrows(
      () => allclose(actual, expected, tolerance),
      AllcloseError,
      "Tolerance",
      JSON.stringify(tolerance),
    );
    // An invalid tolerance is rejected even for empty arrays (having no elements does not make the
    // tolerance valid).
    assertThrows(
      () => allclose([], [], tolerance),
      AllcloseError,
      "Tolerance",
      JSON.stringify(tolerance),
    );
  }
  // Control: e2e intentionally passes atol = +Infinity as a recording-only band (finiteness is not
  // required).
  assertEquals(allclose(actual, expected, { atol: Number.POSITIVE_INFINITY, rtol: 0 }).pass, true);
});

test("allclose rejects rtol = +Infinity with AllcloseError (elements whose reference is 0 do not turn into passes)", () => {
  // For an element whose reference is 0, rtol·|ref| = Infinity·0 = NaN, so the right side of the
  // test is NaN. Without the rejection, an error of 999 passes even with atol = 0.
  assertThrows(
    () =>
      allclose(Float32Array.from([999]), Float32Array.from([0]), {
        atol: 0,
        rtol: Number.POSITIVE_INFINITY,
      }),
    AllcloseError,
    "Tolerance rtol",
  );
});

// Constrains the **meaning** of the report fields (looking only at pass, nobody pins what
// worstIndex / maxRelError point to). Covers non-finite values appearing "later" and the floor on
// the relative error for a zero reference.
test("allclose's report fields prefer non-finite values and floor the relative error for a zero reference", () => {
  // Even after seeing an absolute error of 10 first, worstIndex moves to the non-finite value once
  // it appears
  const mixed = allclose(Float32Array.from([10, Number.NaN, 0]), Float32Array.from([0, 1, 2]));
  assertEquals(mixed.failCount, 3);
  assertEquals(mixed.nonFiniteCount, 1);
  assertEquals(mixed.worstIndex, 1);
  assertEquals(mixed.maxAbsError, Number.POSITIVE_INFINITY);

  // The relative error for a zero reference divides by the floor 1e-12 on |y| (not Infinity from
  // dividing by 0)
  const atZero = allclose(Float32Array.from([0, 10]), Float32Array.from([0, 0]));
  assertEquals(atZero.maxRelError, 10 / 1e-12);

  // NOTE: two zero-length arrays **currently** give pass:true (the loop never runs). Whether to
  // allow empty input or make it an AllcloseError is undecided (review L-R3-4), so this records the
  // current observed value and catches only unintended changes — once decided, flip this one
  // deliberately.
  const empty = allclose(new Float32Array(0), new Float32Array(0));
  assertEquals(
    { pass: empty.pass, failCount: empty.failCount, worstIndex: empty.worstIndex },
    { pass: true, failCount: 0, worstIndex: 0 },
  );
});
