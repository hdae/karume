// **CPU-only** checks of the w8a8 reference oracle (src/reference/i8a8.ts).
//
// The reference is the source of truth of a numeric contract that claims "it matches the GPU with
// atol=0", and that claim is carried by the real-GPU comparisons such as tests/gpu_i8a8_test.ts.
// But a comparison **passes when the reference and the GPU break in the same direction**
// (tautology), so that the reference itself meets the spec (design doc §4.2 / the MUSTs of each
// docstring) has to be pinned separately, by known-value checks that do not depend on the GPU.
//
// MUST: this file has no `ignore: !GPU_AVAILABLE`. Detecting regressions on the reference side even
// in environments without an adapter (CI) is the very reason this file exists.
// MUST: the expected values are known values written out **independently from the spec**, not the
// reference's implementation formulas (copying the same formula makes a tautological check that
// fault injection cannot break). The same rule as tests/reference_ops_test.ts.
//
// Mutations it detects (fault injections confirmed at design time):
// - removing the floor of `Math.max(..., F32_TINY)` → the scale of an all-zero row becomes 0 and ①
//   fails
// - dropping the `sawNan` branch for a plain `Math.max` → NaN is swallowed and ② fails
// - making the quantization lattice ±128 → the largest magnitude lands on 128 (= −128 in i8) and ③
//   fails
// - breaking the multiplication order into the sequential form → ⑤ fails (the data is chosen so the
//   folded and sequential forms give different values)
// - accumulating across groups in i32 (one rounding fewer) → ⑤c fails (the w4a8 folding order)
// - removing the w4a8 group-divisibility check → ⑤d fails (a partial k returns NaN / mixes in the
//   next row)
// - replacing `roundTiesToEven` with `Math.round` → ⑦ fails (ties go towards +∞)

import { test } from "vite-plus/test";
import { assert, assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import {
  I8a8Error,
  quantizeRowsReference,
  referenceAttentionPvI8a8Core,
  referenceAttentionPvQuant,
  referenceAttentionQkI8a8,
  referenceLinearI8a8,
  referenceLinearW4a8,
} from "../src/reference/i8a8.ts";

/**
 * Smallest normal f32 (`torch.finfo(float32).tiny`).
 *
 * MUST: the reference's constant is not imported (if the reference dropped the floor and the
 * expected value moved along with it, the check would disappear).
 */
const F32_TINY = 1.1754943508222875e-38;

// ---------------------------------------------------------------------------
// ① quantizeRowsReference — scale floor of an all-zero row
// ---------------------------------------------------------------------------

test("i8a8 reference: the scale of an all-zero row is f32 tiny (without the floor it would be 0/0 = NaN)", () => {
  const { q, scale } = quantizeRowsReference([0, 0, 0, 0], 1, 4);
  assertEquals(scale[0], F32_TINY, "scale of the all-zero row");
  // With the floor, x/s is 0/tiny = 0, restored exactly (without it, NaN turns into 0 when cast to
  // i8, so the regression would not show in q alone — the scale side is the detector)
  assertEquals([...q], [0, 0, 0, 0], "q of the all-zero row");
});

// ---------------------------------------------------------------------------
// ② quantizeRowsReference — propagation of non-finite values (including row independence)
// ---------------------------------------------------------------------------

test("i8a8 reference: input whose length does not match rows × dim, or whose dimensions are not non-negative integers, fails before computing", () => {
  // Without the length check, missing elements are undefined → NaN, propagating into the row scale
  // and returning an out-of-contract "NaN row" as the expected value (reading [1] as 1×2 gives
  // scale=[NaN]).
  assertThrows(() => quantizeRowsReference([1], 1, 2), I8a8Error, "element count");
  assertThrows(() => quantizeRowsReference([1, 2, 3], 1, 2), I8a8Error, "element count");
  assertThrows(() => quantizeRowsReference([], -1, 0), I8a8Error, "rows");
  assertThrows(() => quantizeRowsReference([1, 2], 1, 2.5), I8a8Error, "dim");
});

test("i8a8 reference: NaN / ±Inf propagate into the row scale without contaminating neighbouring rows", () => {
  // row 0 = contains NaN / row 1 = +Inf / row 2 = −Inf / row 3 = normal
  const x = [
    1,
    Number.NaN,
    -3,
    2,
    Number.POSITIVE_INFINITY,
    1,
    2,
    Number.NEGATIVE_INFINITY,
    1,
    1,
    2,
    3,
  ];
  const { q, scale } = quantizeRowsReference(x, 4, 3);
  // MUST: a plain Math.max swallows NaN (`NaN > amax` is false), giving a finite scale with amax =
  // 3
  assert(Number.isNaN(scale[0]), `scale of the NaN row is ${scale[0]}`);
  assertEquals(scale[1], Number.POSITIVE_INFINITY, "scale of the +Inf row");
  // |−Inf| = +Inf, so the largest magnitude is +Inf (no sign is left in the scale)
  assertEquals(scale[2], Number.POSITIVE_INFINITY, "scale of the −Inf row");
  // The normal row is untouched — s = f32(3 · (1/127)), q = round(x/s)
  assertEquals(scale[3], 0.023622047156095505, "scale of the normal row");
  assertEquals([...q.slice(9, 12)], [42, 85, 127], "q of the normal row");
});

// ---------------------------------------------------------------------------
// ③ quantizeRowsReference — the ends of the lattice (±127)
// ---------------------------------------------------------------------------

test("i8a8 reference: the element with the largest magnitude lands exactly on ±127 (a ±128 lattice breaks it)", () => {
  // With amax = 127, s = f32(127 · (1/127)) = exactly 1.0 and x/s is the input itself (division by
  // 1.0 is exact), so only the ends of the lattice are isolated.
  const { q, scale } = quantizeRowsReference([127, -127, 3, 0], 1, 4);
  assertEquals(scale[0], 1, "s is exactly 1.0");
  // With a ±128 lattice s = f32(127/128) < 1, and 127/s = 128 → turns into −128 in i8
  assertEquals([...q], [127, -127, 3, 0], "the ±127 end points");
});

// ---------------------------------------------------------------------------
// ④⑤ referenceLinearI8a8 — known values and multiplication order
// ---------------------------------------------------------------------------

test("i8a8 reference: known values of linear (integer dot product → f32 dequant → bias)", () => {
  // x = [8, 3] → amax = 8, s = f32(8/127) = 0.06299212574958801, q = [127, 48]
  const x = [8, 3];
  const quantized = quantizeRowsReference(x, 1, 2);
  assertEquals([...quantized.q], [127, 48], "quantized activations");
  assertEquals(quantized.scale[0], 0.06299212574958801, "row scale of the activations");
  // Quantized values of W[2,2] (row = output channel) and the per-channel scale
  //   acc[0] = 127·1 + 48·2   =  223 → 223·(s·0.5)  + 1  ≈  7.0236218 + 1
  //   acc[1] = 127·(−3) + 48·4 = −189 → −189·(s·0.25) − 1 ≈ −2.9763780 − 1
  const out = referenceLinearI8a8({
    x,
    weight: [1, 2, -3, 4],
    weightScale: [0.5, 0.25],
    bias: [1, -1],
    m: 1,
    n: 2,
    k: 2,
  });
  assertEquals([...out], [8.023621559143066, -3.9763779640197754], "output of the w8a8 linear");
});

test("i8a8 reference: linear folds xs·wscale first (pinned with data on which the sequential form differs)", () => {
  // The combination acc = 127, s = f32(0.001/127), wscale = f32(0.001) rounds at different places:
  //   folded     f32(127 · f32(s·w))   = 9.999999974752427e-7
  //   sequential f32(f32(127·s) · w)   = 1.0000001111620804e-6
  // (the GPU side folds the second argument of fma into one f32 — design §4.2).
  const out = referenceLinearI8a8({
    x: [0.001, 0],
    weight: [1, 0],
    weightScale: [0.001],
    bias: [0],
    m: 1,
    n: 1,
    k: 2,
  });
  assertEquals(out[0], 9.999999974752427e-7, "value of the folded form");
  assertNotEquals(out[0], 1.0000001111620804e-6, "degraded into the sequential form");
});

// ---------------------------------------------------------------------------
// ⑤b⑤c referenceLinearW4a8 — known values and the per-group folding order
// ---------------------------------------------------------------------------

test("w4a8 reference: known values of linear (i32 per group → f32 flush at the group boundary → xs)", () => {
  // x = [8, 3, −2, 5] → amax = 8, s = f32(8 · f32(1/127)) = 0.06299212574958801
  //   q = [127, 48, −32, 79] (far enough from half-integers that the round-to-even branch is not
  //   taken)
  // group_size 2, so a row has 2 groups. The scales are powers of 2 only, which keeps the
  // arithmetic up to the flush exact in f32 (only the final multiplication by xs rounds = checkable
  //   by hand). row 0: g0 acc = 127·1 + 48·2   =  223 → 223·0.5   = 111.5
  //          g1 acc = −32·(−3) + 79·4 =  412 → 412·0.25  = 103   → accf = 214.5
  //          out = f32(214.5 · s + 1)
  //   row 1: g0 acc = 127·7 + 48·(−7) =  553 → 553·0.125 =  69.125
  //          g1 acc = −32·5 + 79·(−6) = −634 → −634·2    = −1268 → accf = −1198.875
  //          out = f32(−1198.875 · s − 1)
  const out = referenceLinearW4a8({
    x: [8, 3, -2, 5],
    weight: [1, 2, -3, 4, 7, -7, 5, -6],
    weightScale: [0.5, 0.25, 0.125, 2],
    bias: [1, -1],
    m: 1,
    n: 2,
    k: 4,
    groupSize: 2,
  });
  assertEquals([...out], [14.511811256408691, -76.519683837890625], "output of the w4a8 linear");
});

test("w4a8 reference: folds into f32 per group (differs from summing all groups into one)", () => {
  // A row with amax = 127 has s = f32(127 · f32(1/127)) = exactly 1.0 (division by 1.0 is exact),
  // so q = x itself and the final `accf · xs` is the identity too; **only the folding order**
  // remains in the output.
  //   g0 acc = 127·(−7) + 37·(−3) = −1000 / g1 acc = 121·(−7) + 20·(−6) = −967
  //   per group  : f32(f32(−967 · s) + f32(−1000 · s)) = −33.45867156982422
  //   single sum : f32(−1967 · s)                      = −33.45866775512695
  // MUST: an implementation that accumulates across groups in i32 (one rounding fewer) fails here.
  const scale = 0.017009999603033066;
  const out = referenceLinearW4a8({
    x: [127, 37, 121, 20],
    weight: [-7, -3, -7, -6],
    weightScale: [scale, scale],
    bias: [0],
    m: 1,
    n: 1,
    k: 4,
    groupSize: 2,
  });
  assertEquals(out[0], -33.45867156982422, "per-group folding");
  assertNotEquals(out[0], -33.45866775512695, "sums all groups into one");
});

// ⑤d group-divisibility check
test("w4a8 reference: input whose k is not divisible by the group length fails before computing", () => {
  // groups = 48 / 32 = 1.5. Without the check, the group loop runs up to `group = 1` and the inner
  // loop reads `q[row·48 + 32..63]` — measured: with m=1 every element is NaN (out of range of q =
  // undefined); with m ≥ 2 only row 0 / col 0 becomes a finite wrong value that adds in **the next
  // row's activations** (both are "the oracle silently returns a different number", and the atol=0
  // claim loses its meaning).
  assertThrows(
    () =>
      referenceLinearW4a8({
        x: new Array(48).fill(1),
        weight: new Array(96).fill(1),
        weightScale: new Array(4).fill(1),
        bias: [0, 0],
        m: 1,
        n: 2,
        k: 48,
        groupSize: 32,
      }),
    I8a8Error,
    "group",
  );
});

test("w4a8 reference: a non-integer or negative group length fails before computing even when it divides k", () => {
  const input = {
    weightScale: new Array(4).fill(1),
    bias: [0, 0],
    m: 1,
    n: 2,
  };
  // k=5 / g=2.5 passes the divisibility check with groups = 2, but the activation and weight
  // indices become non-integers and it returns NaN
  assertThrows(
    () =>
      referenceLinearW4a8({
        ...input,
        x: new Array(5).fill(1),
        weight: new Array(10).fill(1),
        k: 5,
        groupSize: 2.5,
      }),
    I8a8Error,
    "positive integer",
  );
  // k=4 / g=-2 gives groups = -2, the group loop runs 0 times, and it returns the bias as is
  assertThrows(
    () =>
      referenceLinearW4a8({
        ...input,
        x: new Array(4).fill(1),
        weight: new Array(8).fill(1),
        k: 4,
        groupSize: -2,
      }),
    I8a8Error,
    "positive integer",
  );
});

// ---------------------------------------------------------------------------
// ⑥ referenceAttentionQkI8a8 — the half scale applies to both q and k
// ---------------------------------------------------------------------------

test("i8a8 reference: the QK half scale applies to both q and k (scale ×2 → output ×4)", () => {
  // q = k = [8, 3] → on both sides q̂ = [127, 48], s = f32(8/127)
  //   acc = 127² + 48² = 18433 / S = f32(acc · f32((s·scale)·(s·scale)))
  const input = { q: [8, 3], k: [8, 3], batch: 1, m: 1, n: 1, d: 2 };
  const half = referenceAttentionQkI8a8({ ...input, scale: 0.5 });
  const full = referenceAttentionQkI8a8({ ...input, scale: 1 });
  assertEquals(half[0], 18.285572052001953, "S with scale = 0.5");
  // Doubling is exact in f32, so if it applied to only one side the output would only double
  assertEquals(full[0], 73.14228820800781, "S with scale = 1.0");
  assertEquals(full[0], half[0] * 4, "the scale does not take effect squared");
});

// ---------------------------------------------------------------------------
// ⑦ referenceAttentionPvQuant — 127·exp(0) and round-to-even
// ---------------------------------------------------------------------------

test("i8a8 reference: P̃ is quantized as round(127·exp(S−m)), with ties going to even", () => {
  // At the position where the row maximum m is subtracted, exp(0) = 1 → exactly 127 (it sticks to
  // the top of the lattice)
  const qp = referenceAttentionPvQuant([5, 4], [5], 1, 2);
  assertEquals([...qp], [127, 47], "exp(0) → 127 / exp(−1)·127 ≈ 46.7 → 47");
  // A real tie: with S−m = f32(−0.195), f32(f32(exp(x))·127) lands exactly on 104.5.
  // Round-to-even gives 104 and Math.round (half-up) gives 105, so the two differ.
  const tie = referenceAttentionPvQuant([Math.fround(-0.195)], [0], 1, 1);
  assertEquals(tie[0], 104, "a tie goes to even (Math.round would give 105)");
});

// ---------------------------------------------------------------------------
// ⑧ referenceAttentionPvI8a8Core — known values (the pure function after the integers are taken)
// ---------------------------------------------------------------------------

test("i8a8 reference: ③PV matches the known value of f32(Σ qP·vq) · ((inv·(1/127))·vs)", () => {
  //   acc  = 127·100 + 64·(−50) = 9500
  //   prow = f32(0.5 · f32(1/127)) = 0.003937007859349251
  //   O    = f32(9500 · f32(prow · 2)) ≈ 74.803149
  const out = referenceAttentionPvI8a8Core({
    qp: [127, 64],
    vq: [100, -50],
    rowInv: [0.5],
    vs: [2],
    batch: 1,
    m: 1,
    n: 2,
    d: 1,
  });
  assertEquals([...out], [74.80314636230469], "output of ③PV");
});
