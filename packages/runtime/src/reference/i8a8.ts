/**
 * **CPU reference** of w8a8 linear (per-token i8 activations × per-channel i8 weights) (stage 2 of
 * ADR 0005).
 *
 * The numeric contract of this path is **agreement with the GPU at atol=0**, a property that sets
 * it apart from the other kernels (whose f32 reduction order varies with the implementation). It
 * rests on two facts:
 *
 * 1. The dot product is an **exact `i32` addition**, so neither tiling, zero-padding of the
 *    remainder nor the addition order changes the value.
 * 2. There are only three floating-point operations per output element — `f32(acc)` / `xs·wscale`
 *    / `fma`. The WGSL spec requires the conversion and the multiplication to be correctly rounded.
 *    **Only `fma` has no spec guarantee** (WGSL §15.7.4.1 states that its accuracy is inherited from
 *    `x*y+z` — a fused implementation rounds once and matches this reference, while a non-fused
 *    expansion is also spec-conforming and can differ within a 1 ULP band. Measured: atol=0 holds on
 *    the fused side on Vulkan, Metal differs by 1 ULP = docs/known-issues.md). atol=0 is kept not
 *    as "a consequence of the spec" but as **a per-device conformance check**.
 *
 * MUST: do not copy the GPU side's tile structure (the 64×64 tiles, shared memory and zero-padding
 * of the K remainder of src/kernels/linear-i8a8.ts are **not here**). Writing a plain triple loop is
 * the only way to keep the handling of tile boundaries and remainders under verification (the same
 * discipline as referenceLinear).
 *
 * ## Tie-breaking of rounding (settled by measurement, 2026-08-03)
 *
 * WGSL's `round` is **round-half-to-even** (measured: `round(0.5)=0` / `round(1.5)=2` /
 * `round(-0.5)=-0` / `round(2.5)=2`), which matches `torch.round`. JS's `Math.round` is half-up
 * (toward +∞) and **does not match** — the source of truth is confined to the single
 * {@link roundTiesToEven}.
 *
 * ## Mirroring `fma` (settled by measurement, 2026-08-03)
 *
 * The GPU side's final expression is `fma(f32(acc), xs·ws, bias)` (rounding once without rounding
 * the product). JS has no fma, but `f32(acc)` and `xs·ws` are both f32, so **their product is exact
 * in double precision** (24+24 = 48 ≤ 53 bits), and `Math.fround(a * s + b)` matches the
 * single-rounding result. Measured on this machine: all 100,000 samples matched (the plain `a*b+c`,
 * an explicit `fma` and a `bitcast`-pinned form all gave the same values).
 * NOTE: for pairs whose exact sum exceeds 54 bits, the intermediate rounding in double precision can
 * in theory take effect (when the exponents of `|f32(acc)·s|` and `|bias|` differ widely). Not a
 * single case has been measured, but it is not a guarantee in principle.
 *
 * ## Only division is outside atol=0 (found by measurement, 2026-08-03)
 *
 * The WGSL spec allows f32 division up to 2.5 ULP, and on this machine `a / b` measured 1 ULP off
 * IEEE correct rounding in 55,605 of 200,000 samples (multiplication, addition and `fma`: 0).
 * Therefore `q = round(x / s)` **can move by ±1 step only for elements where `x/s` is near a half
 * integer**. The scale side is built by the **multiplication** `amax * (1/127)` and matches exactly
 * (the MUST of src/kernels/quantize-rows.ts). The atol=0 comparison is a contract that holds only
 * on "data far enough from the rounding boundaries", and the test side measures that margin and
 * makes it a check (tests/gpu_i8a8_test.ts).
 */

/**
 * A value passed to the reference does not meet the contract (`k` not divisible by the group
 * length, etc.).
 *
 * MUST: do not import `ReferenceOpError` from `reference/ops.ts`. This module is self-contained
 * with no import at all; importing it would grow the `../ops.ts` module graph all the way into
 * production code (the import chain of src/gpu/attention-dp4a-canary.ts). It keeps a "light Error
 * for this layer only", like `I4Error` in `format/i4.ts`.
 */
export class I8a8Error extends Error {
  override readonly name = "I8a8Error";
}

/**
 * The edge of the quantization grid and the floor of the scale.
 *
 * MUST: do not **import** the constants of src/kernels/quantize-rows.ts. Sharing them would put a
 * regression such as "make the grid ±128" or "drop the floor" into the reference to the same
 * extent, and the comparison would become vacuous (demonstrated by fault injection ②). The
 * reference is written out independently from the spec (design doc §4.2).
 */
const ABS_MAX = 127;
/** The smallest normal f32 (`torch.finfo(float32).tiny`). */
const F32_TINY = 1.1754943508222875e-38;
/**
 * MUST: the scale is a **multiplication** by 1/127, not a division by 127 (division is not
 * correctly rounded).
 */
const INV_ABS_MAX = Math.fround(1 / ABS_MAX);

/**
 * Round-half-to-even. The same rule as WGSL's `round` / `torch.round`.
 *
 * MUST: do not use `Math.round` (it is half-up, so `-0.5 → 0` matches but `0.5 → 1` /
 * `2.5 → 3` diverge). Inputs that land exactly on a quantization grid boundary occur routinely in
 * real data.
 */
export const roundTiesToEven = (value: number): number => {
  if (!Number.isFinite(value)) return value;
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction > 0.5) return floor + 1;
  if (fraction < 0.5) return floor;
  // A tie goes to the even side (floor when floor is even, floor + 1 when it is odd)
  return floor % 2 === 0 ? floor : floor + 1;
};

type QuantizedRows = {
  /**
   * The quantized activations (flat i8 of `[rows, dim]` — the GPU's `xq` decoded from its 4-per-word
   * packing).
   *
   * NOTE: when a row's scale is non-finite (NaN / Inf), these values are **outside the contract**
   * (on the GPU side `vec4<i32>(NaN)` is an indeterminate value). Only `scale` and the final output
   * are compared.
   */
  readonly q: Int8Array<ArrayBuffer>;
  /** Per-row scale (`[rows]`). */
  readonly scale: Float32Array<ArrayBuffer>;
};

/**
 * Symmetric i8 quantization per row (= token) (the mirror of src/kernels/quantize-rows.ts).
 *
 * MUST: NaN propagates into the row's scale (if `amax` is NaN, so is `s`). The GPU side guarantees
 * this with a bit-pattern check; if the reference swallowed NaN with a plain `Math.max`, the
 * comparison would become vacuous.
 */
export const quantizeRowsReference = (
  x: ArrayLike<number>,
  rows: number,
  dim: number,
): QuantizedRows => {
  // MUST: check shape and length before computing. Missing elements would propagate as
  // undefined → NaN into the row's scale and return an out-of-contract "NaN row" as the expected
  // value (NaN inside the tensor is a legitimate input, so it is not checked).
  for (const [name, value] of [
    ["rows", rows],
    ["dim", dim],
  ] as const) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new I8a8Error(`行量子化の参照: ${name}=${value} が非負の整数でない`);
    }
  }
  if (x.length !== rows * dim) {
    throw new I8a8Error(
      `行量子化の参照: 要素数 ${x.length} が rows × dim = ${rows * dim} と合わない`,
    );
  }
  const q = new Int8Array(rows * dim);
  const scale = new Float32Array(rows);
  for (let row = 0; row < rows; row += 1) {
    const base = row * dim;
    let amax = 0;
    let sawNan = false;
    for (let i = 0; i < dim; i += 1) {
      const value = Math.abs(x[base + i]);
      if (Number.isNaN(value)) sawNan = true;
      else if (value > amax) amax = value;
    }
    const s = sawNan ? Number.NaN : Math.max(Math.fround(amax * INV_ABS_MAX), F32_TINY);
    scale[row] = s;
    for (let i = 0; i < dim; i += 1) {
      const rounded = roundTiesToEven(Math.fround(x[base + i] / s));
      q[base + i] = Math.max(-ABS_MAX, Math.min(ABS_MAX, rounded));
    }
  }
  return { q, scale };
};

/**
 * The smallest distance of `x/s` from a rounding boundary (a half integer).
 *
 * GPU division is allowed up to 2.5 ULP, so the GPU's and the reference's quantized values match
 * exactly **only on data where this margin is large** (the docstring above). Tests measure this
 * value and make it a check before claiming atol=0 — claiming atol=0 without noticing that the
 * margin shrank would in effect mean "it only happens to pass". 2.5 ULP at |x/s| ≤ 127 is about
 * 1.9e-5, so a margin of 1e-3 is safe by orders of magnitude.
 */
export const quantizeRowsTieMargin = (x: ArrayLike<number>, rows: number, dim: number): number => {
  const { scale } = quantizeRowsReference(x, rows, dim);
  let margin = Number.POSITIVE_INFINITY;
  for (let row = 0; row < rows; row += 1) {
    for (let i = 0; i < dim; i += 1) {
      const ratio = Math.fround(x[row * dim + i] / scale[row]);
      if (!Number.isFinite(ratio)) continue;
      const fraction = ratio - Math.floor(ratio);
      margin = Math.min(margin, Math.abs(fraction - 0.5));
    }
  }
  return margin;
};

type LinearI8a8Input = {
  /** Activations `[m, k]` (f32, before quantization). */
  readonly x: ArrayLike<number>;
  /** **Quantized integer values** of the weight `[n, k]` (±127). */
  readonly weight: ArrayLike<number>;
  /** Per-channel scale of the weight `[n]`. */
  readonly weightScale: ArrayLike<number>;
  /** bias `[n]` (always f32 — ADR 0006). */
  readonly bias: ArrayLike<number>;
  readonly m: number;
  readonly n: number;
  readonly k: number;
};

/**
 * Reference of w8a8 linear — `out = f32(Σ xq·wq) · (xs[row] · wscale[col]) + bias[col]`.
 *
 * MUST: the multiplication order is the same as the GPU's (`xs · wscale` is folded into one f32
 * first). The sequential form (`f32(acc) · xs · wscale`) moves the rounding points and breaks
 * atol=0.
 */
export const referenceLinearI8a8 = (input: LinearI8a8Input): Float32Array<ArrayBuffer> => {
  const { x, weight, weightScale, bias, m, n, k } = input;
  const { q, scale } = quantizeRowsReference(x, m, k);
  const out = new Float32Array(m * n);
  for (let row = 0; row < m; row += 1) {
    for (let col = 0; col < n; col += 1) {
      // Exact integer sum (|acc| ≤ k·127·128 is far below 2^53, so a JS number is exact)
      let acc = 0;
      for (let i = 0; i < k; i += 1) acc += q[row * k + i] * weight[col * k + i];
      const combined = Math.fround(scale[row] * weightScale[col]);
      out[row * n + col] = Math.fround(Math.fround(acc) * combined + bias[col]);
    }
  }
  return out;
};

type LinearW4a8Input = {
  /** Activations `[m, k]` (f32, before quantization). */
  readonly x: ArrayLike<number>;
  /** **Quantized integer values** of the weight `[n, k]` (±7 — offset 8 already decoded). */
  readonly weight: ArrayLike<number>;
  /** Group scales of the weight (flat `[n, k/g]`). */
  readonly weightScale: ArrayLike<number>;
  /** bias `[n]` (always f32 — ADR 0006). */
  readonly bias: ArrayLike<number>;
  readonly m: number;
  readonly n: number;
  readonly k: number;
  /** Group length along K (divides `k`). */
  readonly groupSize: number;
};

/**
 * Reference of w4a8 linear — **fold each group exactly in i32 and flush to f32 only at group
 * boundaries**:
 *
 * ```
 * accf = Σ_gi fma(f32(Σ_{i∈gi} xq·wq), wscale[col, gi], accf)   // k/g roundings
 * out  = fma(accf, xs[row], bias[col])                          // + 1
 * ```
 *
 * MUST: do not bring `xs` into the group (the MUST of {@link referenceLinearI8a8} that folds
 * `xs·wscale` first **does not hold for w4a8** — wscale changes per group, so folding adds `k/g`
 * roundings and breaks atol=0 against the GPU (src/kernels/linear-i8a8.ts)).
 * MUST: the folding order is "ascending group". Collapsing all groups into one integer sum reduces
 * the roundings to one, and data exists on which the values diverge (pinned by
 * tests/i8a8_reference_test.ts).
 * MUST: do not copy the GPU side's tile structure (a plain quadruple loop — the same discipline as
 * {@link referenceLinearI8a8}. A group is **the quantization axis**, not a tile, so this is the spec
 * itself, not a copy of the structure).
 * NOTE: the `fma` mirror is the same text as in {@link referenceLinearI8a8}
 * (`Math.fround(a * b + c)`); for pairs whose exact sum exceeds 54 bits the intermediate rounding in
 * double precision can in theory take effect.
 */
export const referenceLinearW4a8 = (input: LinearW4a8Input): Float32Array<ArrayBuffer> => {
  const { x, weight, weightScale, bias, m, n, k, groupSize } = input;
  // MUST: check the range of the group length itself first. A non-integer (k=5, g=2.5) gives NaN
  // through non-integer activation and weight indices even when groups is an integer; a negative
  // one (k=4, g=-2) runs the group loop 0 times and silently returns bias.
  if (!Number.isSafeInteger(groupSize) || groupSize < 1) {
    throw new I8a8Error(`w4a8 参照: group 長 ${groupSize} が正の整数でない`);
  }
  const groups = k / groupSize;
  // MUST: fail before the comparison. With a `k` that leaves a remainder, the group loop adds the
  // activations of **the next row** of the flat `[m,k]` (col=0) and returns NaN through a
  // non-integer scale index (col≥1) — both are forms where "the oracle silently returns another
  // number", and the check that claims atol=0 loses its meaning.
  if (!Number.isSafeInteger(groups)) {
    throw new I8a8Error(
      `w4a8 参照: k=${k} が group 長 ${groupSize} で割り切れない（K 方向の量子化軸）`,
    );
  }
  const { q, scale } = quantizeRowsReference(x, m, k);
  const out = new Float32Array(m * n);
  for (let row = 0; row < m; row += 1) {
    for (let col = 0; col < n; col += 1) {
      let accf = 0;
      for (let group = 0; group < groups; group += 1) {
        // Exact integer sum (|acci| ≤ g·127·8 is far below 2^53, so a JS number is exact)
        let acci = 0;
        for (let i = group * groupSize; i < (group + 1) * groupSize; i += 1) {
          acci += q[row * k + i] * weight[col * k + i];
        }
        accf = Math.fround(Math.fround(acci) * weightScale[col * groups + group] + accf);
      }
      out[row * n + col] = Math.fround(accf * scale[row] + bias[col]);
    }
  }
  return out;
};

/**
 * The edge of P̃'s quantization grid and `1/127`.
 *
 * MUST: do not **merge** them with {@link ABS_MAX} above. That one is the grid of "an adaptive
 * scale built from the row amax"; this one rests on a different fact, "the in-row max of
 * `P̃ = exp(S−m)` is 1.0 by construction, so the scale degenerates to 1/127" (design §2.2). The
 * values agree almost by coincidence, and merging them would silently propagate a change of one to
 * the other.
 */
const P_ABS_MAX = 127;
const INV_P_ABS_MAX = Math.fround(1 / P_ABS_MAX);

/**
 * Reference of the quantized sequence `qP = round(127·exp(S−m))` that ③PV builds while filling the
 * A tile (flat `[batch·m, n]`).
 *
 * **It does not match the GPU bit for bit** (implementation differences of `exp` — design §4.2).
 * The role of this reference is the check "a mismatch is always ±1 step" and recording the mismatch
 * rate, which the test side asserts.
 * MUST: rounding is round-half-to-even ({@link roundTiesToEven}). `Math.round` is half-up and
 * diverges.
 * NOTE: there is no clamp for the same reason as on the GPU side (`exp(S−m) ≤ 1` when `m` is the
 * exact row max). Passing a `rowMax` that is not the row max can exceed 127, but that is a contract
 * violation by the caller. In that case 128 **wraps implicitly to −128** when stored into the
 * Int8Array (no throw) — on the GPU side `pack4xI8` packs the low 8 bits and gives the same −128,
 * so the reference and the GPU go wrong silently **together**. Throwing only in the reference would
 * behave differently from the GPU on contract-violating input, so it is only documented here (the
 * boundary: 128 is reached at `S − m > ln(127.5/127) ≈ 0.00393` — round-half-to-even rounds 127.5
 * to 128).
 */
export const referenceAttentionPvQuant = (
  s: ArrayLike<number>,
  rowMax: ArrayLike<number>,
  rows: number,
  n: number,
): Int8Array<ArrayBuffer> => {
  const qp = new Int8Array(rows * n);
  for (let row = 0; row < rows; row += 1) {
    const max = rowMax[row];
    for (let i = 0; i < n; i += 1) {
      const shifted = Math.fround(s[row * n + i] - max);
      const p = Math.fround(Math.fround(Math.exp(shifted)) * P_ABS_MAX);
      qp[row * n + i] = roundTiesToEven(p);
    }
  }
  return qp;
};

type AttentionPvI8a8CoreInput = {
  /**
   * `qP` (flat `[batch, m, n]`, integers 0..127). Exactly the sequence the GPU built in the A tile.
   */
  readonly qp: ArrayLike<number>;
  /** Quantized values of Vᵀ (flat `[batch, d, n]`, integers ±127 — **N contiguous**). */
  readonly vq: ArrayLike<number>;
  /** The row statistic `1/Σexp(S−m)` (`[batch·m]`). */
  readonly rowInv: ArrayLike<number>;
  /** Row scale of Vᵀ (`[batch·d]`) = the **column** scale of V. */
  readonly vs: ArrayLike<number>;
  readonly batch: number;
  readonly m: number;
  readonly n: number;
  readonly d: number;
};

/**
 * Reference of the i8a8 variant of fused attention ③PV — **a pure function from the integers on**.
 *
 * `O[b,m,d] = f32(Σ_n qP·vq) · ((inv·(1/127)) · vs[b,d])`, with only four floating-point
 * operations: `f32(acc)` / `inv·(1/127)` / `prow·vs` / the product. Hence **atol=0 against the GPU**
 * (only the generation of `qP` carries the implementation differences of `exp`, and that is split
 * off into a separate check on the {@link referenceAttentionPvQuant} side — the two-step split of
 * design §4.2).
 *
 * MUST: the multiplication order is the same as the GPU's (`prow · vs` is folded into one f32
 * first). The sequential form (`f32(acc)·prow·vs`) moves the rounding points and breaks atol=0.
 * MUST: `1/127` is built by multiplication (the same as the GPU side — division is not correctly
 * rounded).
 * MUST: do not copy the GPU side's tile structure (a plain quadruple loop — the same discipline as
 * the other references).
 */
export const referenceAttentionPvI8a8Core = (
  input: AttentionPvI8a8CoreInput,
): Float32Array<ArrayBuffer> => {
  const { qp, vq, rowInv, vs, batch, m, n, d } = input;
  const out = new Float32Array(batch * m * d);
  for (let b = 0; b < batch; b += 1) {
    for (let row = 0; row < m; row += 1) {
      const prow = Math.fround(rowInv[b * m + row] * INV_P_ABS_MAX);
      for (let col = 0; col < d; col += 1) {
        // Exact integer sum (|acc| ≤ n·127² is far below 2^53, so a JS number is exact)
        let acc = 0;
        for (let i = 0; i < n; i += 1) {
          acc += qp[(b * m + row) * n + i] * vq[(b * d + col) * n + i];
        }
        const combined = Math.fround(prow * vs[b * d + col]);
        out[(b * m + row) * d + col] = Math.fround(Math.fround(acc) * combined);
      }
    }
  }
  return out;
};

type AttentionQkI8a8Input = {
  /** q `[batch, m, d]` (f32, before quantization; batch is one axis folding B·H). */
  readonly q: ArrayLike<number>;
  /** k `[batch, n, d]` (f32, before quantization). */
  readonly k: ArrayLike<number>;
  readonly batch: number;
  readonly m: number;
  readonly n: number;
  readonly d: number;
  /** **The half scale** (`√scale_factor` of ADR 0023). Applied to both the q side and the k side. */
  readonly scale: number;
};

/**
 * Reference of the i8a8 variant of fused attention ①QK —
 * `S[b,m,n] = f32(Σ_d qq·kq) · ((qs[b,m]·scale) · (ks[b,n]·scale))`.
 *
 * MUST: the half scale is applied **after quantization** (on the dequant side). Applying it before
 * quantization adds one rounding step and breaks atol=0 against the GPU
 * (src/kernels/attention-i8a8.ts).
 * MUST: the multiplication order is the same as the GPU's too (`qs'·ks'` is folded into one f32
 * first). The sequential form (`f32(acc)·qs'·ks'`) moves the rounding points.
 * MUST: do not copy the GPU side's tile structure (writing a plain quadruple loop is the only way
 * to keep the handling of tile boundaries and the K remainder under verification — the same
 * discipline as {@link referenceLinearI8a8}).
 */
export const referenceAttentionQkI8a8 = (
  input: AttentionQkI8a8Input,
): Float32Array<ArrayBuffer> => {
  const { batch, m, n, d } = input;
  const scale = Math.fround(input.scale);
  // The reduced axis D is innermost-contiguous for both q and k, so quantization is per-token over
  // the rows (b, m) / (b, n) as they are
  const query = quantizeRowsReference(input.q, batch * m, d);
  const key = quantizeRowsReference(input.k, batch * n, d);
  const out = new Float32Array(batch * m * n);
  for (let b = 0; b < batch; b += 1) {
    for (let row = 0; row < m; row += 1) {
      const qs = Math.fround(query.scale[b * m + row] * scale);
      for (let col = 0; col < n; col += 1) {
        // Exact integer sum (|acc| ≤ d·127² is far below 2^53, so a JS number is exact)
        let acc = 0;
        for (let i = 0; i < d; i += 1) {
          acc += query.q[(b * m + row) * d + i] * key.q[(b * n + col) * d + i];
        }
        const ks = Math.fround(key.scale[b * n + col] * scale);
        out[(b * m + row) * n + col] = Math.fround(Math.fround(acc) * Math.fround(qs * ks));
      }
    }
  }
  return out;
};
