/**
 * WGSL generation for elementwise kernels (unary / binary / where / cast).
 *
 * MUST: generation is deterministic — the same key gives byte-identical WGSL. The key of the
 * browser's implicit pipeline cache is the WGSL string itself and there is no explicit control API,
 * so any wobble assigns another kernel to the same key. Key and WGSL are built from the same
 * canonicalization ({@link canonicalize}), so the same generation input always fixes both at once.
 * MUST: **the element type is part of the key** (ADR 0009). The same op at the same rank is a
 * different kernel for a different element type; without it in the key an f32 pipeline is assigned
 * to an i32 dispatch and the bit pattern is reinterpreted without an exception.
 * MUST: grid-stride loops are assumed. Real models exceed the per-dimension dispatch limit (65535 by
 * default in the spec).
 *
 * The output is a contiguous layout; each input is read through "right-aligned broadcast strides"
 * (stride 0 for a dimension of 1). The shape is passed in the params buffer, so a pipeline is
 * reused for the same rank. The attrs scalars (clamp's min/max, clamp_min's min, leaky_relu's slope,
 * the comparison value) also travel **at the end of params** — baking them into the WGSL would add a
 * pipeline per value (the same discipline as src/kernels/masked-fill.ts).
 */

import type { IrDtype } from "../format/ir.ts";
import {
  BINARY_OPS,
  type BinaryOpName,
  CAST_OP,
  outputDtypeOf,
  resolveOpContract,
  scalarParamCount,
  slotDtypesOf,
  UNARY_OPS,
  type UnaryOpName,
  WHERE_OP,
} from "../ops.ts";
import { CodegenError } from "./errors.ts";
import {
  f32Literal,
  IS_NAN_BITS_WGSL,
  nanGuard,
  SIGMOID_STABLE_WGSL,
  TANH_STABLE_WGSL,
} from "./numerics-wgsl.ts";
import { assertU32Params } from "./params.ts";

export const ELEMENTWISE_WORKGROUP_SIZE = 256;

export type ElementwiseSpec =
  | {
      readonly op: UnaryOpName | BinaryOpName | typeof WHERE_OP;
      /** Output rank. At least 1 (the caller normalizes a rank-0 scalar to rank 1, length 1). */
      readonly rank: number;
      /**
       * Semantic dtype of the **value slots**. The output dtype follows from the contract table's
       * mapping (bool for comparison ops). where's condition slot is always bool, so only the value
       * side goes here.
       */
      readonly dtype: IrDtype;
    }
  | {
      readonly op: typeof CAST_OP;
      readonly rank: number;
      /** Semantic dtype converted from. */
      readonly dtype: IrDtype;
      /** Semantic dtype converted to. */
      readonly to: IrDtype;
    };

/**
 * The canonicalization result is a union discriminated by kind. The type of the op name is fixed
 * by kind, so the generator does not have to write an unchecked cast such as `op as UnaryOpName`
 * (the same approach as OpContract in ops.ts). `arity` is the number of inputs.
 */
type CanonicalSpec = {
  readonly rank: number;
  /** dtype of the value slots (the side that goes into the key). */
  readonly from: IrDtype;
  readonly to: IrDtype;
  /** Element type of each input slot (baked into the bindings' `array<T>`). */
  readonly slots: readonly IrDtype[];
  /** Number of f32 scalars at the end of params. */
  readonly scalars: number;
} & (
  | { readonly kind: "unary"; readonly op: UnaryOpName; readonly arity: 1 }
  | { readonly kind: "binary"; readonly op: BinaryOpName; readonly arity: 2 }
  | { readonly kind: "cast"; readonly op: typeof CAST_OP; readonly arity: 1 }
  | { readonly kind: "where"; readonly op: typeof WHERE_OP; readonly arity: 3 }
);

const isUnary = (op: string): op is UnaryOpName => UNARY_OPS.some((name) => name === op);
const isBinary = (op: string): op is BinaryOpName => BINARY_OPS.some((name) => name === op);

/**
 * The dtype of the generation input is checked against the contract table (copying the table would
 * split the allowed range into two places).
 *
 * MUST: check per slot, not against the **union** of accepted sets. where's union is {bool, f32},
 * so checking the union lets the condition's bool pass as a value-slot generation input.
 */
const assertSlotDtype = (op: string, slot: number, dtype: IrDtype): void => {
  const accept = slotDtypesOf(resolveOpContract(op))[slot];
  if (accept === undefined || !accept.includes(dtype)) {
    throw new CodegenError(
      `elementwise codegen: op '${op}' slot ${slot} cannot run dtype '${dtype}'`,
    );
  }
};

/**
 * The output dtype comes from the contract table's mapping (codegen does not decide it on its own)
 * — elementwise has a single output.
 */
const outputDtype = (op: string, slotZeroDtype: IrDtype): IrDtype =>
  outputDtypeOf(resolveOpContract(op), 0, slotZeroDtype, "elementwise codegen");

const canonicalize = (spec: ElementwiseSpec): CanonicalSpec => {
  if (!Number.isSafeInteger(spec.rank) || spec.rank < 1) {
    throw new CodegenError(`elementwise codegen: rank must be an integer >= 1 (${spec.rank})`);
  }
  const { op, rank, dtype } = spec;
  if (op === CAST_OP) {
    assertSlotDtype(op, 0, dtype);
    assertSlotDtype(op, 0, spec.to);
    return {
      kind: "cast",
      op,
      rank,
      arity: 1,
      from: dtype,
      to: spec.to,
      slots: [dtype],
      scalars: 0,
    };
  }
  if (op === WHERE_OP) {
    // Slot 0 is the condition (always bool), 1 / 2 are the values. The generation input's dtype
    // is the value side.
    assertSlotDtype(op, 1, dtype);
    return {
      kind: "where",
      op,
      rank,
      arity: 3,
      from: dtype,
      to: outputDtype(op, "bool"),
      slots: ["bool", dtype, dtype],
      scalars: 0,
    };
  }
  if (isUnary(op)) {
    assertSlotDtype(op, 0, dtype);
    return {
      kind: "unary",
      op,
      rank,
      arity: 1,
      from: dtype,
      to: outputDtype(op, dtype),
      slots: [dtype],
      scalars: scalarParamCount(op),
    };
  }
  if (isBinary(op)) {
    assertSlotDtype(op, 0, dtype);
    return {
      kind: "binary",
      op,
      rank,
      arity: 2,
      from: dtype,
      to: outputDtype(op, dtype),
      slots: [dtype, dtype],
      scalars: scalarParamCount(op),
    };
  }
  throw new CodegenError(`elementwise codegen: op '${op}' is not in the elementwise vocabulary`);
};

/** Semantic dtype → WGSL scalar type. A bool is stored as u32 0 / 1 (ADR 0009). */
const WGSL_SCALAR: Readonly<Record<IrDtype, string>> = {
  f32: "f32",
  i32: "i32",
  bool: "u32",
};

/** Zero literal per dtype (used for the `!= 0` truth test). */
const WGSL_ZERO: Readonly<Record<IrDtype, string>> = {
  f32: "0.0",
  i32: "0",
  bool: "0u",
};

/**
 * Abramowitz–Stegun 7.1.26. WGSL has no built-in erf.
 * gelu needs it because it implements torch's default `approximate="none"` (the erf form).
 * The CPU reference computes erf with a different formula (a Chebyshev fit) — copying the same
 * approximation would cancel the error and make the test vacuous.
 *
 * Error accounting (three layers — do not mix them up): (1) the approximation's own theoretical
 * error ≈1.4e-7 (evaluated in double precision, largest at x≈0.045); (2) this f32 Horner + exp
 * evaluated with f32 rounding at every step, measured ≈5.3e-7 (largest at x≈0.035 — CPU f32
 * emulation sweep 2026-08-31); (3) the WGSL spec allows exp up to 3+2|x| ULP, so the bound for a
 * spec-conforming implementation is higher still (contribution ≈3.6e-7 as a→0). Do not quote
 * "1.5e-7" as the shader's guaranteed value. For gelu as a whole the maximum absolute error is
 * ≈6.8e-7 (≈1.4 ULP at |gelu|≈4.2), but the negative tail (x≈−3.4) loses digits and has a
 * **relative** error of ≈2.2e-4 — a per-op tolerance cut as a relative band gives false positives
 * here.
 */
const ERF_FN = `fn erf_approx(x: f32) -> f32 {
  let s = sign(x);
  let a = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  var p = 1.061405429;
  p = -1.453152027 + p * t;
  p = 1.421413741 + p * t;
  p = -0.284496736 + p * t;
  p = 0.254829592 + p * t;
  return s * (1.0 - p * t * exp(-a * a));
}`;

/**
 * The range that switches to the series (`|x| < LOG1P_SERIES_THRESHOLD`). Rationale in the doc of
 * {@link LOG1P_FN}.
 */
const LOG1P_SERIES_THRESHOLD = 0.25;
/** Number of terms of the Maclaurin series (`x^1 … x^n`). */
const LOG1P_SERIES_TERMS = 10;

/**
 * Builds the Maclaurin series of `log1p` in Horner form `x·(1 − x·(1/2 − x·(… ∓ x/n)))`.
 * The coefficients are `1/k` themselves, so the term count alone determines the whole expression
 * (no hand-written coefficient table = typos cannot happen by construction).
 */
const log1pSeries = (): string => {
  let expr = f32Literal(1 / LOG1P_SERIES_TERMS);
  for (let k = LOG1P_SERIES_TERMS - 1; k >= 1; k -= 1) {
    expr = `${f32Literal(1 / k)} - x * (${expr})`;
  }
  return `x * (${expr})`;
};

/**
 * log1p (WGSL has no built-in). **The series for small |x|, `log(1+x)` for large |x|**.
 *
 * MUST: do not use `log(1 + x)` over the whole range. In f32, `1 + x` at x = 1e-8 is exactly 1.0
 * and the answer is 0 (relative error 1.0); moreover the WGSL spec guarantees `log` only up to an
 * **absolute** error on [0.5, 2.0] (not a relative error), so the smaller x is, the less it can be
 * trusted. Measured on this repository's verification environment, `log(1 + 1e-7)` was about 1.94
 * times the true value.
 *
 * MUST: do not use the classic correction that extracts the rounding residual of `u = fl(1+x)` as
 * `d = u − 1` (`log(u)·x/d`). Measurement confirmed that **the shader compiler folds
 * `(1.0 + x) - 1.0` algebraically into `x`** (d always equalled x), so the correction factor becomes
 * identically 1 and degenerates to the naive form. WGSL has no way to forbid reassociation, so this
 * technique cannot be expressed.
 *
 * Bound of the truncation error: with n terms and |x| ≤ a the relative error is ≈ `a^n/(n+1)`.
 * a = 0.25 / n = 10 gives 8.7e-8, below the f32 eps (1.19e-7). Above the switch point the
 * precision of `log` shows as is (raising a needs more series terms, lowering it increases the
 * relative error on the `log` side — the threshold sits about where the two balance). **The
 * measured worst relative error is 2.7e-7 (x = 0.2501, log side)**, more than six orders of
 * magnitude away from the naive form's 1.0 (x = 1e-8). tests/gpu_ops_test.ts pins the deviation by
 * comparing against the CPU reference (`Math.log1p`).
 *
 * NOTE: the domain is x > −1 (the actual argument `exp(scaled)` of the softplus decomposition is
 * always positive). x < −1 is a negative argument to `log`, indeterminate in WGSL — torch gives NaN
 * too, so nothing is silently approximated.
 */
const LOG1P_FN = `fn log1p_series(x: f32) -> f32 {
  // The coefficients are 1/k (Horner form). Outside the switch point, log(1+x) is used as is.
  let series = ${log1pSeries()};
  return select(log(1.0 + x), series, abs(x) < ${f32Literal(LOG1P_SERIES_THRESHOLD)});
}`;

/**
 * Unary value expressions. The second argument returns the WGSL name of a scalar at the end of
 * params (in the order of {@link SCALAR_PARAM_ATTRS}).
 *
 * MUST: `clamp` / `clamp_min` / `relu` are wrapped in {@link nanGuard}. torch propagates NaN in
 * all three, but the plain expressions swallow NaN through folding (mechanism in
 * {@link IS_NAN_BITS_WGSL}).
 * MUST: `clamp` does not use the built-in `clamp(x, lo, hi)`. Its result for `lo > hi` is
 * indeterminate (although the contract side rejects anything but min <= max).
 * MUST: `leaky_relu` uses **the select form**, not `max` / `min` (ADR 0015). A form such as
 * `relu(x) + s·min(x, 0)` breaks propagation because `min` swallows NaN. In the select form the
 * comparison falls to false and returns `s · NaN = NaN`, so **with x in both branches** this form
 * alone keeps NaN even when folded (pinned on a real GPU by tests/gpu_ops_test.ts).
 */
const UNARY_WGSL: Readonly<
  Record<UnaryOpName, (a: string, scalar: (index: number) => string) => string>
> = {
  neg: (a) => `-${a}`,
  abs: (a) => `abs(${a})`,
  exp: (a) => `exp(${a})`,
  log: (a) => `log(${a})`,
  log1p: (a) => `log1p_series(${a})`,
  sqrt: (a) => `sqrt(${a})`,
  sin: (a) => `sin(${a})`,
  // MUST: do not pass straight through to the built-in `tanh` (mechanism and threshold in
  // {@link TANH_STABLE_WGSL}). This op also runs inside the logits softcap
  // (`div` → `tanh` → `mul`), so it would open the same hole as gelu_tanh.
  tanh: (a) => `tanh_stable(${a})`,
  sigmoid: (a) => `sigmoid_stable(${a})`,
  relu: (a) => nanGuard(a, `max(${a}, 0.0)`),
  // 0.7071067811865476 = 1/√2
  gelu: (a) => `0.5 * ${a} * (1.0 + erf_approx(${a} * 0.7071067811865476))`,
  // torch's approximate="tanh". 0.7978845608028654 = √(2/π). The expression itself is the
  // definition, so (unlike the erf form) there is no room to make the approximation more precise.
  // MUST: the inner part goes through {@link TANH_STABLE_WGSL}. With the plain `tanh`, the cubic
  // term pushes the inner argument past 44.36 at a pre-activation x ≳ 10.05, and an exp-based
  // implementation returns a silent NaN.
  // The NaN shell ({@link nanGuard}) is not applied to the whole expression — the factor `x` stays
  // outside, so `0.5 · NaN · (…)` stays NaN and propagation does not depend on folding (the same
  // treatment as the erf-form gelu).
  gelu_tanh: (a) =>
    `0.5 * ${a} * (1.0 + tanh_stable(0.7978845608028654 * (${a} + 0.044715 * ${a} * ${a} * ${a})))`,
  // A bool is u32 0 / 1. `1u - x` would leave 0/1 when the storage convention is broken.
  bitwise_not: (a) => `select(1u, 0u, ${a} != 0u)`,
  clamp: (a, scalar) =>
    nanGuard(
      a,
      `select(select(${a}, ${scalar(0)}, ${a} < ${scalar(0)}), ${scalar(1)}, ${a} > ${scalar(1)})`,
    ),
  // The select form, as ruled in ADR 0017 (WGSL does not guarantee NaN propagation for
  // `max(x, m)`).
  // NOTE: the original assessment of ADR 0015 / op-vocabulary.md that "a two-step select
  // propagates NaN" was refuted by measurement (folding turns it into `max` —
  // {@link IS_NAN_BITS_WGSL}). Propagation is carried only by the shell's bit-pattern check, not
  // by the shape of the select.
  clamp_min: (a, scalar) => nanGuard(a, `select(${a}, ${scalar(0)}, ${a} < ${scalar(0)})`),
  leaky_relu: (a, scalar) => `select(${scalar(0)} * ${a}, ${a}, ${a} >= 0.0)`,
  ge_scalar: (a, scalar) => `select(0u, 1u, ${a} >= ${scalar(0)})`,
  le_scalar: (a, scalar) => `select(0u, 1u, ${a} <= ${scalar(0)})`,
  gt_scalar: (a, scalar) => `select(0u, 1u, ${a} > ${scalar(0)})`,
};

const BINARY_WGSL: Readonly<Record<BinaryOpName, (a: string, b: string) => string>> = {
  add: (a, b) => `${a} + ${b}`,
  sub: (a, b) => `${a} - ${b}`,
  mul: (a, b) => `${a} * ${b}`,
  div: (a, b) => `${a} / ${b}`,
  ge: (a, b) => `select(0u, 1u, ${a} >= ${b})`,
  // A form that stays a logical AND of truth values even when the storage convention (0 / 1) is
  // broken. `a & b` is a bitwise operation, so it would not.
  bitwise_and: (a, b) => `select(0u, 1u, (${a} != 0u) && (${b} != 0u))`,
};

/**
 * The value expression of cast. Only the conversion to bool is the `x != 0` truth test; everything
 * else is a WGSL type conversion (`i32(f32)` truncates toward 0 by spec = matches torch's truncate).
 */
const castWgsl = (a: string, from: IrDtype, to: IrDtype): string =>
  to === "bool" ? `select(0u, 1u, ${a} != ${WGSL_ZERO[from]})` : `${WGSL_SCALAR[to]}(${a})`;

/**
 * The injection order of helper functions is fixed (not the order they appear in the expression)
 * — for same key → byte-identical WGSL.
 */
const HELPERS: readonly (readonly [string, string])[] = [
  ["erf_approx", ERF_FN],
  ["is_nan_bits", IS_NAN_BITS_WGSL],
  ["log1p_series", LOG1P_FN],
  ["sigmoid_stable", SIGMOID_STABLE_WGSL],
  ["tanh_stable", TANH_STABLE_WGSL],
];

/**
 * Selects the helper functions the value expression calls, **closing over the dependencies between
 * helpers too** (`tanh_stable` calls `is_nan_bits`).
 *
 * MUST: do not keep a dependency table by hand. The text of the bodies is the source of truth; a
 * separate table silently rots every time a callee is added or removed. Filtering by the value
 * expression's text alone emits WGSL in which a callee is left undefined (it fails at compile time,
 * so it is not a silent wrong value, but it is blocked at the generation entry).
 * The returned order is that of {@link HELPERS}; the selection does not affect the order.
 */
const usedHelpers = (value: string): readonly string[] => {
  const selected = new Set<string>();
  let scope = value;
  for (;;) {
    const added = HELPERS.filter(([name]) => !selected.has(name) && scope.includes(name));
    if (added.length === 0) {
      return HELPERS.filter(([name]) => selected.has(name)).map(([, body]) => body);
    }
    for (const [name, body] of added) {
      selected.add(name);
      scope = `${scope}\n${body}`;
    }
  }
};

/**
 * MUST: every generation parameter baked into the WGSL goes into the key. The workgroup size is
 * baked into `@workgroup_size` and the grid-stride stride expression, so changing the constant
 * without it in the key reuses a pipeline compiled with the old size under the same key, out of
 * step with the dispatch count. The element type goes into the key for the same reason (it is baked
 * into the bindings' `array<T>`).
 *
 * NOTE: **the value slots' dtype and the output dtype** are all the key needs. The condition slot
 * (where's bool) and the scalar count follow uniquely from the op name, so they are not independent
 * axes.
 *
 * v2: clamp / clamp_min / relu gained the NaN-propagating shell and their WGSL body changed
 * (IS_NAN_BITS_WGSL).
 * v3: tanh / gelu_tanh now go through the saturation clipping ({@link TANH_STABLE_WGSL}).
 * The version is bumped **per family** (ADR 0020 decision 3) — a per-op exception table would mean
 * tracking "which op is at which version now" twice at the next bump (the other ops only change
 * their key; their generated code is identical).
 */
export const elementwiseKey = (spec: ElementwiseSpec): string => {
  const { op, rank, from, to } = canonicalize(spec);
  return `ew:v3:${op}:${from}>${to}:r${rank}:wg${ELEMENTWISE_WORKGROUP_SIZE}`;
};

/**
 * The params layout (storage, a u32 array):
 * `[0]=element count n, [1..rank]=output dims, then rank strides per input, and at the end the
 * scalar attrs as f32 bit patterns (in the order of {@link SCALAR_PARAM_ATTRS})`.
 * It is passed as storage rather than uniform because this family has no workgroupBarrier and is
 * not constrained by the uniformity analysis (reduce / matmul, which have barriers, use uniform).
 */
export const elementwiseWgsl = (spec: ElementwiseSpec): string => {
  const canonical = canonicalize(spec);
  const { op, rank, arity, from, to, slots, scalars } = canonical;
  const strideBase = (input: number): number => 1 + rank + input * rank;
  const scalarAt = (index: number): number => 1 + rank + arity * rank + index;

  const bindings = [`@group(0) @binding(0) var<storage, read> params: array<u32>;`];
  for (let k = 0; k < arity; k += 1) {
    bindings.push(
      `@group(0) @binding(${k + 1}) var<storage, read> in${k}: array<${WGSL_SCALAR[slots[k]]}>;`,
    );
  }
  bindings.push(
    `@group(0) @binding(${arity + 1}) var<storage, read_write> out: array<${WGSL_SCALAR[to]}>;`,
  );

  const decode: string[] = ["    var rem = i;"];
  for (let d = rank - 1; d >= 1; d -= 1) {
    decode.push(`    let c${d} = rem % params[${1 + d}u]; rem = rem / params[${1 + d}u];`);
  }
  decode.push("    let c0 = rem;");

  const loads: string[] = [];
  for (let k = 0; k < arity; k += 1) {
    const terms = Array.from({ length: rank }, (_, d) => `c${d} * params[${strideBase(k) + d}u]`);
    loads.push(`    let v${k} = in${k}[${terms.join(" + ")}];`);
  }

  // Scalars are not re-read per element; they are bound to names once, outside the loop.
  const constants = Array.from(
    { length: scalars },
    (_, index) => `  let s${index} = bitcast<f32>(params[${scalarAt(index)}u]);`,
  );
  const scalarName = (index: number): string => `s${index}`;

  const value =
    canonical.kind === "cast"
      ? castWgsl("v0", from, to)
      : canonical.kind === "unary"
        ? UNARY_WGSL[canonical.op]("v0", scalarName)
        : canonical.kind === "binary"
          ? BINARY_WGSL[canonical.op]("v0", "v1")
          : // torch's where(cond, a, b) = cond ? a : b. WGSL's select(f, t, cond) takes them reversed.
            `select(v2, v1, v0 != 0u)`;
  const used = usedHelpers(value);

  return [
    `// karume elementwise ${op} (rank ${rank}, generated)`,
    ...bindings,
    "",
    ...used.flatMap((fn) => [fn, ""]),
    `@compute @workgroup_size(${ELEMENTWISE_WORKGROUP_SIZE})`,
    "fn main(",
    "  @builtin(global_invocation_id) gid: vec3<u32>,",
    "  @builtin(num_workgroups) nwg: vec3<u32>,",
    ") {",
    "  let n = params[0u];",
    ...constants,
    `  let stride = nwg.x * ${ELEMENTWISE_WORKGROUP_SIZE}u;`,
    "  var i = gid.x;",
    "  while (i < n) {",
    ...decode,
    ...loads,
    `    out[i] = ${value};`,
    "    i = i + stride;",
    "  }",
    "}",
    "",
  ].join("\n");
};

/** The shape right-aligned to rank (the missing part padded with 1). */
const padShape = (shape: readonly number[], rank: number): number[] => {
  if (shape.length > rank) {
    throw new CodegenError(
      `elementwise params: input rank ${shape.length} exceeds output rank ${rank}`,
    );
  }
  return [...Array<number>(rank - shape.length).fill(1), ...shape];
};

/**
 * Builds the contents of the params buffer. The condition is that **the spec used for generation is
 * passed**; four points — rank, number of inputs, number of scalars, and each axis's compatibility —
 * are checked against the same canonicalization ({@link canonicalize}).
 *
 * MUST: do not drop the four-point check. The WGSL side bakes the read position of the scalar attrs
 * from **the `arity` at generation time** as `1 + rank + arity·rank + index`, so if the position
 * written here (derived from the number of inputs) disagrees, it reads outside the params array —
 * WebGPU's bounded access raises no exception, and the result is a silent wrong value such as
 * `clamp_min(x, 1e-12)` turning into a clamp with another value.
 * MUST: axis compatibility (`1` or equal to the output) is checked here for the same reason. For a
 * pair that cannot broadcast, building "contiguous stride unless 1" silently makes params that
 * point outside **the input buffer**.
 * MUST: scalars are rounded to f32 before they are stored. IR attrs are JSON f64, so without
 * rounding the CPU reference (`Math.fround`) and the GPU (the f32 words of params) could see
 * different values.
 */
export const elementwiseParams = (
  spec: ElementwiseSpec,
  outShape: readonly number[],
  inputShapes: readonly (readonly number[])[],
  scalars: readonly number[] = [],
): Uint32Array<ArrayBuffer> => {
  const { rank, arity, scalars: scalarCount } = canonicalize(spec);
  if (outShape.length !== rank) {
    throw new CodegenError(
      `elementwise params: output rank ${outShape.length} does not match the rank ${rank} at generation time`,
    );
  }
  if (inputShapes.length !== arity) {
    throw new CodegenError(
      `elementwise params: ${inputShapes.length} inputs do not match op '${spec.op}' arity ${arity}`,
    );
  }
  if (scalars.length !== scalarCount) {
    throw new CodegenError(
      `elementwise params: ${scalars.length} scalars do not match op '${spec.op}' scalar count ${scalarCount}`,
    );
  }
  const n = outShape.reduce((count, dim) => count * dim, 1);
  assertU32Params("elementwise params", {
    ...Object.fromEntries(outShape.map((dim, d) => [`out_dims[${d}]`, dim])),
    n,
  });
  const params = new Uint32Array(1 + rank + inputShapes.length * rank + scalars.length);
  params[0] = n;
  for (let d = 0; d < rank; d += 1) params[1 + d] = outShape[d];
  inputShapes.forEach((shape, k) => {
    const padded = padShape(shape, rank);
    for (let d = 0; d < rank; d += 1) {
      if (padded[d] !== 1 && padded[d] !== outShape[d]) {
        throw new CodegenError(
          `elementwise params: input ${k} axis ${d} length ${padded[d]} matches neither output ${
            outShape[d]
          } nor 1`,
        );
      }
    }
    // Accumulate the contiguous strides from the right. A dimension of 1 gets stride 0 to absorb
    // the broadcast.
    let running = 1;
    const strides = new Array<number>(rank).fill(0);
    for (let d = rank - 1; d >= 0; d -= 1) {
      strides[d] = padded[d] === 1 ? 0 : running;
      running *= padded[d];
    }
    // With a length-0 axis in between, the strides to its left can exceed u32 even when n is 0
    // (the product does not collapse to 0).
    assertU32Params(
      "elementwise params",
      Object.fromEntries(strides.map((stride, d) => [`in${k}_strides[${d}]`, stride])),
    );
    for (let d = 0; d < rank; d += 1) params[1 + rank + k * rank + d] = strides[d];
  });
  const floats = new Float32Array(params.buffer);
  scalars.forEach((value, index) => {
    // MUST: the finiteness check is done **as an f32** (the destination is an f32 word, so `1e39`,
    // finite in f64, silently turns into `+Inf` — the same check as the contract layer's
    // `assertFiniteAttr`).
    if (!Number.isFinite(Math.fround(value))) {
      throw new CodegenError(`elementwise params: scalar attr is not finite as f32 (${value})`);
    }
    floats[1 + rank + inputShapes.length * rank + index] = value;
  });
  return params;
};
