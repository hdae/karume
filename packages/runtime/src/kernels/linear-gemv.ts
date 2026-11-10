/**
 * The **GEMV family** of linear (weights stored as i2 / i4 / i8, and f16 / f32 storage at M=1 —
 * ADR 0082 / 0097). The second kernel family of `linear`: output, bindings and uniform stay the same
 * as on the default path (linear in src/kernels/gemm.ts), and **only the work assignment** changes to
 * "1 thread = 1 output column". It has 2 forms: the M=1 (decode) variant and the **row-block
 * variant** for small M (2–{@link LINEAR_GEMV_MAX_ROWS}, short prefill / speculative verification)
 * (the latter is ADR 0082 addendum 5 / perf-ledger K-21).
 *
 * ## Why the default GEMM skeleton is not enough (mechanism — ADR 0082 / research 2026-08-30 §7)
 *
 * The default chooses the geometry by M bucket (`gemmGeometryForRows` in
 * src/kernels/gemm-geometry.ts), but every bucket shares one skeleton: "shared tiles + a double
 * barrier per K tile of 16". With `GEOMETRY_M16N16` (64 threads), which M=1 hits:
 *
 * - **Only 4 of the 64 threads write output** (the row with `lid.y == 0`). The shared A tile (16
 *   rows × K16) has only 1 real row; the other 15 rows are dead weight filled with 0.
 * - Each K tile passes `workgroupBarrier()` twice, so **the latency of the weight reads is exposed
 *   sequentially, once per tile**. The measurement is proportional to k, `k / 16 × ≈1.3µs`, and
 *   **almost independent of n**; small shapes whose weights fit in L2 are just as slow — so the
 *   bottleneck is not bandwidth starvation but **serialised issue**.
 *
 * This family drops the shared memory and the barriers altogether and prefetches weight words to
 * create memory parallelism. `n` independent reductions run at once, so the latency is hidden by
 * parallelism across columns.
 * This cost stays as is for M=2–32 too, independent of M (the same geometry M16N16 takes M ≤ 64 —
 * in research 2026-09-07 §7.2 the GEMM path's 65–67 ms is unchanged across M=2–16), so the row-block
 * variant removes the same cost with the same means.
 *
 * ## Row block (M ≥ 2) — one read of a weight word shared by `rows` rows, y tiles along rows
 *
 * One thread holds the reduction of 1 column × `rows` rows, reads the weight word (and the i4 group
 * scale) once and multiply-adds it with each row's x. M is split into `ceil(m / rows)` y tiles (the
 * weights are re-read per tile). `rows` is decided by {@link linearGemvRowsForShape} as **a pure
 * function of (storage, m, n, parallelism target)** (the target is a static knob fixed at Session
 * creation — {@link ROWS_THREAD_TARGET}). The mechanism seen in the sweep (RTX 3080 Ti, the 13 census
 * shapes of gemma4 E2B × M ∈ {1..64} — docs/research/2026-09-07-gemv-rows-k21.md):
 *
 * - **While the thread count (= n × y tiles) is below ≈16K, buying parallelism with y tiles is faster
 *   than folding rows into one thread** (L2 absorbs the weight re-reads within one dispatch). Shapes
 *   with small n (256 / 512 / 1536 columns) are best at `rows = 1` even at M=32.
 * - Beyond 16K the weight traffic matters, and increasing `rows` to reduce re-reads is faster
 *   (lm_head with 262,144 columns is best at `rows = M`).
 * - However, the elements one thread holds per word (`rows × step`) have 2 ceilings. The run-time
 *   ceiling is 512 (i4 16 rows / i8 32 rows — above that, register pressure slows it down), and the
 *   ceiling of **shader compile cost** comes before it: {@link ROWS_ELEMENTS_PER_WORD_CAP} = 256
 *   (generated text grows with the row count, and the cost of parsing and validation is
 *   super-linear in the text size — see the constant's doc).
 *
 * The target {@link ROWS_THREAD_TARGET} and one ceiling reproduce both sides; the gap to the best
 * per shape (oracle) is, census-weighted, 1–5% for M ≤ 32 and ≤ 13% for M ≤ 64. M=1 is not turned
 * into a row block (the same text as the generated M=1 variant — equal or better in measurement too,
 * and the generated decode shaders do not move by a single byte).
 *
 * ## The storage axis (i4 / i8)
 *
 * The mechanism (serialised issue caused by the barriers) does not depend on the storage dtype, so
 * inside the family the storage is a **variant axis**. One weight word is **`vec4<u32>` = 16 B for
 * both**, and only the number of elements it carries (= the reduction step {@link linearGemvUnit})
 * differs: 32 elements for i4, 16 for i8. The scale lookup differs too: i4 per group (depends on k,
 * so it is looked up again per word), i8 one per output channel (invariant in k, so it is bound once
 * outside the reduction — the loop-invariant hoisting of ADR 0019).
 *
 * ## Numeric contract (bit identity MUST)
 *
 * MUST: the reduction per output element is **sequential in ascending k**, the literal form of the
 * multiply-add is `acc = acc + a * b`, and the bias is added once at the end. Weight decode uses
 * **the same literal form** as the default path (`dequant4` in weight-storage.ts): a per-component
 * f32 multiply, `f32(i32(u) − 8) * scale` for i4 and `f32(q) * scale` for i8 (the scale is not
 * factored out of the reduction — ADR 0019). What changes is only the **work assignment** that ADR
 * 0022 decision 3 grants as free = **bit-identical to the default path**. In the row block too, the
 * reduction order per output element, the pairing of products and the order of adding the bias are
 * the same as at M=1. The row block calls a per-word function per row, but the rounding point of the
 * decode stays the same single f32 multiply (ADR 0082 addenda 8 and 9).
 * MUST: prefetching (`unroll`) only "reads words together"; **the multiply-add order stays in
 * ascending word order**. Mixing multiply-adds across words would move the addition order per output
 * element and break the contract.
 * NOTE: f32 reduction has no theoretical guarantee of order independence, so bit identity is, as in
 * gemm-geometry, **an empirical claim** (the check = exact u32 match in tests/gpu_linear_gemv_test.ts).
 *
 * MUST NOT: the form that splits k and adds partial sums back together (split-K) changes the
 * reduction order, so it is not put into this family. The separate opt-in kernel
 * `linearGemvParallelWgsl` follows ADR 0098.
 *
 * ## Relation to the geometry (the reach of gemm-geometry's "only selection point" MUST)
 *
 * gemm-geometry says "the selection point of the work assignment is the one place
 * `gemmGeometryForRows`", but its reach stays **inside the GEMM skeleton** (ADR 0082 decision 2).
 * This family is a separate kernel that does not share the skeleton, and selection has 2 stages —
 * **family selection** (whether to enter the GEMV family) in one place, `buildLinear`, and **variant
 * selection** (M=1 variant or row block, and the row block's `rows`) in one place,
 * {@link defaultLinearGemvRowsVariant} — both **pure functions of plan-time shapes**. The
 * discriminator is in the pipeline key as the family name `linear_gemv` (for the row block, the
 * variant's `r<rows>`) — both the ban on run-time autotuning and "same key → byte-identical WGSL"
 * are kept.
 */

import { CodegenError } from "../codegen/errors.ts";
import { gemmParams, gemmUsesVec4 } from "./gemm.ts";
import { staticQuantizeGuessWgsl, staticQuantizeParams } from "./static-quantize.ts";
import {
  i4GroupKeyPart,
  i4GroupShift,
  WEIGHT_SCALE_VAR,
  weightKeyPart,
  weightNote,
  weightScaleWgsl,
  type WeightStorage,
} from "./weight-storage.ts";

/**
 * Elements carried by one weight word (`vec4<u32>` or `vec4<f32>` = 16 B) = **the reduction step**
 * (per storage).
 *
 * It is also the divisibility unit the eligibility check (`buildLinear` in
 * src/runtime/recipe-builders/linear.ts) imposes on k and the group length, so the check and the
 * kernel read the same single derivation point per storage.
 * f16 / bf16 carry 8 elements per word and f32 4, all M=1 only (ADR 0082 addenda 6 and 7; bf16 is
 * ADR 0131).
 */
export const linearGemvUnit = (storage: WeightStorage): number => {
  if (storage === "i2") return 64;
  if (storage === "i4") return 32;
  if (storage === "i8") return 16;
  if (storage === "f16" || storage === "bf16") return 8;
  if (storage === "f32") return 4;
  // Unreachable; this line exists to turn a missing step for a newly added storage red at compile time.
  const unhandled: never = storage;
  throw new CodegenError(`linear_gemv: unhandled weight storage: ${JSON.stringify(unhandled)}`);
};

/** f16 / bf16 / f32 are accepted at M=1 only. Do not widen them implicitly to the row block. */
const assertRowsStorage = (storage: WeightStorage): void => {
  if (storage === "f16" || storage === "bf16" || storage === "f32") {
    throw new CodegenError(
      `linear_gemv: row blocks are not supported for ${storage} storage (M=1 variant only)`,
    );
  }
};

/** The per-workgroup thread limit WebGPU core guarantees (upper bound of `cols`). */
const MAX_THREADS = 256;

/**
 * Generation parameters (the counterpart of the geometry). **{@link defaultLinearGemvVariant} is the
 * only selection point**, and nothing re-chooses at run time (ADR 0022's ban on run-time autotuning
 * applies to this family too).
 */
export type LinearGemvVariant = {
  /** Output columns one workgroup handles (= threads per workgroup — 1 thread per column). */
  readonly cols: number;
  /**
   * Weight words prefetched per iteration (memory parallelism). Does not affect the multiply-add
   * order.
   */
  readonly unroll: number;
};

/**
 * Generation parameters of the row-block variant (M ≥ 2). `cols` / `unroll` mean the same as in the
 * M=1 variant, and `rows` = the rows one thread holds (the height of a y tile).
 * **{@link defaultLinearGemvRowsVariant} is the only selection point**.
 */
export type LinearGemvRowsVariant = LinearGemvVariant & {
  readonly rows: number;
};

/**
 * The default variant. Census-weighted best in **the sweep of the 12 real shapes + 4 remainder
 * shapes of gemma4 E2B decode on an RTX 3080 Ti** (`c32 u4`, ×8.45 over the default — ADR 0082 /
 * docs/research/2026-08-30-gemma4-decode-wallclock.md §7). When a shape is passed, the
 * large-vocabulary INT8 shape measured at M=1 (addendum 10) selects c16.
 * MUST: a change of the default goes together with re-measuring the bit-identity check
 * (tests/gpu_linear_gemv_test.ts).
 */
export const defaultLinearGemvVariant = (shape?: {
  readonly storage: WeightStorage;
  readonly n: number;
  readonly k: number;
}): LinearGemvVariant => ({
  // DECIDED: only the measured large-vocabulary INT8 shape gets fewer cols/workgroup (ADR 0082 addendum 10).
  // docs/decisions/0082-linear-gemv-decode.md
  cols: shape?.storage === "i8" && shape.n === 262144 && shape.k === 1536 ? 16 : 32,
  unroll: 4,
});

/**
 * The upper bound of M the row-block variant takes = **the guard's upper bound** (the range of row
 * counts `buildLinear` sends to this family). It is exactly the swept range (M ≤ 64 — the same width
 * as the M16N16 bucket of the default GEMM skeleton); outside it, the default GEMM skeleton stays
 * (ADR 0082 decision 4, "Eligibility stays inside the measured range").
 */
export const LINEAR_GEMV_MAX_ROWS = 64;

/**
 * The parallelism target that decides `rows` (thread count = output columns n × y tiles
 * `ceil(m / rows)`). In the RTX 3080 Ti sweep, shapes below this value were faster buying
 * parallelism with y tiles, and shapes above it were faster reducing weight re-reads with `rows`
 * (the mechanism in the module doc).
 * NOTE: **a value baked from the saturation point of the reference device (RTX 3080 Ti, 80 SMs)**,
 * not a portable optimum. On GPUs whose saturation point is an order of magnitude smaller
 * (integrated GPUs, Apple M series) it buys too many y tiles and the weight re-reads exceed the
 * optimum. For other devices the value is replaced with `SessionOptions.linearGemvRowsThreadTarget`
 * (a **static** knob fixed at Session creation = the limit that flows into the 4th argument of
 * {@link linearGemvRowsForShape}; the selection stays a pure function). Nothing chooses it
 * automatically by looking at the device (the ban on run-time autotuning (ADR 0022) —
 * docs/limitations.md).
 * NOTE: even with another target, `rows` goes into the key (`r<rows>` of {@link linearGemvRowsKey}),
 * so "same key → byte-identical WGSL" is kept — a different target changes the chosen `rows` and so
 * the key.
 * MUST: a change goes together with a re-measured sweep (the check in tests/gpu_linear_gemv_test.ts
 * does not pin the value — it only checks that the selection is a pure function of (storage, m, n,
 * target)).
 */
const ROWS_THREAD_TARGET = 16384;

/**
 * The ceiling of elements one thread holds per weight word (`rows × ` {@link linearGemvUnit}) =
 * i4 8 rows / i8 16 rows. For run time alone it would go up to 512 (i4 16 rows / i8 32 rows; above
 * that, register pressure slows it down), but **generated text grows with the row count and becomes
 * shader compile cost**, so it stops at 256: at 512, i4 16 rows is 138 KB / ≈165 ms and i8 32 rows is
 * 137 KB / ≈0.9 s (naga's parsing and validation are super-linear in text size), adding ≈1 s to the
 * first turn. At 256 the largest shapes that can be chosen (i4 8 rows 73 KB, i8 16 rows 70 KB) take
 * ≈55–60 ms each, ≈150 ms in total for the first turn. The run time lost is, census-weighted, +3% at
 * M=32 / 64 (i4) and lm_head +1.6 ms/run (i8) — research 2026-09-07-gemv-rows-k21 §6.
 * MUST: a change goes together with a re-measured sweep (run time + compile time).
 */
const ROWS_ELEMENTS_PER_WORD_CAP = 256;

/**
 * Chooses the row-block height `rows` as **a pure function of (storage, m, n, target)**: start from
 * the largest power of 2 not above m (within the ceiling {@link ROWS_ELEMENTS_PER_WORD_CAP} / step)
 * and halve it until the thread count `n · ceil(m / rows)` reaches `threadTarget` (1 if it never
 * does). Shapes with small n always get 1 (= the M=1 kernel laid out M times along y); for n of
 * lm_head size it grows up to the ceiling.
 *
 * `threadTarget` defaults to {@link ROWS_THREAD_TARGET} (the saturation point of the reference
 * device); on other devices the value `SessionOptions.linearGemvRowsThreadTarget` fixed at Session
 * creation arrives here. The range guard lives in one place on the Session side (the same guard is
 * not implemented twice).
 *
 * MUST: the return value goes into the key ({@link linearGemvRowsKey}). Being a pure function
 * guarantees both "same key → byte-identical WGSL" and the ban on run-time autotuning.
 */
export const linearGemvRowsForShape = (
  storage: WeightStorage,
  m: number,
  n: number,
  threadTarget: number = ROWS_THREAD_TARGET,
): number => {
  assertRowsStorage(storage);
  if (!Number.isSafeInteger(m) || m < 1 || m > LINEAR_GEMV_MAX_ROWS) {
    throw new CodegenError(`linear_gemv: row count ${m} is outside 1..${LINEAR_GEMV_MAX_ROWS}`);
  }
  if (!Number.isSafeInteger(n) || n < 1) {
    throw new CodegenError(`linear_gemv: column count ${n} is not a positive integer`);
  }
  const cap = ROWS_ELEMENTS_PER_WORD_CAP / linearGemvUnit(storage);
  let rows = 1;
  while (rows * 2 <= Math.min(m, cap)) rows *= 2;
  while (rows > 1 && n * Math.ceil(m / rows) < threadTarget) rows /= 2;
  return rows;
};

/**
 * The default of the row-block variant (`cols` / `unroll` the same as the M=1 default, `rows` from
 * the storage, the shape and the target). For the meaning and default of `threadTarget`, see
 * {@link linearGemvRowsForShape}.
 */
export const defaultLinearGemvRowsVariant = (
  storage: WeightStorage,
  m: number,
  n: number,
  threadTarget?: number,
): LinearGemvRowsVariant => ({
  ...defaultLinearGemvVariant(),
  rows: linearGemvRowsForShape(storage, m, n, threadTarget),
});

const assertVariant = (variant: LinearGemvVariant): void => {
  const { cols, unroll } = variant;
  if (!Number.isSafeInteger(cols) || cols < 1 || cols > MAX_THREADS) {
    throw new CodegenError(`linear_gemv: cols must be an integer in 1..${MAX_THREADS} (${cols})`);
  }
  if (!Number.isSafeInteger(unroll) || unroll < 1) {
    throw new CodegenError(`linear_gemv: unroll must be a positive integer (${unroll})`);
  }
};

const assertRowsVariant = (variant: LinearGemvRowsVariant): void => {
  assertVariant(variant);
  const { rows } = variant;
  if (!Number.isSafeInteger(rows) || rows < 1 || rows > LINEAR_GEMV_MAX_ROWS) {
    throw new CodegenError(
      `linear_gemv: rows must be an integer in 1..${LINEAR_GEMV_MAX_ROWS} (${rows})`,
    );
  }
};

/**
 * Group length → the shift baked into the WGSL (`undefined` for i2 / i8, which have no group).
 *
 * "Power of 2 ≥ 16" and "storage and group length come in pairs" are checked by
 * {@link i4GroupShift} (the same derivation point as the declaration layer). This family further
 * requires **group ≥ {@link linearGemvUnit}** — if the elements of one word crossed a group, one
 * scale per word would not be enough, and values with the wrong scale applied would come out
 * silently.
 */
const gemvGroupShift = (
  storage: WeightStorage,
  groupSize: number | undefined,
): number | undefined => {
  const unit = linearGemvUnit(storage);
  const shift = i4GroupShift("linear_gemv", storage, groupSize);
  // i2 / i8 have no group (i4GroupShift has already checked the pairing)
  if (groupSize === undefined) return shift;
  if (shift === undefined || groupSize < unit) {
    throw new CodegenError(
      `linear_gemv: group_size ${groupSize} is not a power of two that is at least ${unit}` +
        ` (1 word = ${unit} elements would cross a group)`,
    );
  }
  return shift;
};

/**
 * The uniform Dims (byte-identical to the default path's `linearParams(m, n, k)` — the contract not
 * to split the binding layout). Only the checks are family-specific; the u32 range of m / n / k is
 * delegated to {@link gemmParams}.
 *
 * MUST: k is limited to **multiples of {@link linearGemvUnit}**. The WGSL `units = dims.k / <step>u`
 * truncates the remainder, so otherwise the reduction returns values that silently dropped the end
 * of the row (no exception).
 * MUST: for i4, k is also limited to **multiples of group_size** (the WGSL
 * `scale_base = col * (k >> shift)` derives the number of scales per row by division). The
 * declaration layer (ADR 0069 decision 2) and the recipe builder's eligibility check guarantee the
 * same condition, but a direct kernel call does not go through them.
 * MUST: m is 1..{@link LINEAR_GEMV_MAX_ROWS} (the M=1 variant only 1; the row-block variant takes it
 * in `ceil(m / rows)` y tiles).
 */
export const linearGemvParams = (
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
  groupSize?: number,
): Uint32Array<ArrayBuffer> => {
  const unit = linearGemvUnit(storage);
  gemvGroupShift(storage, groupSize);
  if ((storage === "f16" || storage === "bf16" || storage === "f32") && m !== 1) {
    throw new CodegenError(`linear_gemv params: ${storage} storage supports only m=1 (${m})`);
  }
  if (!Number.isSafeInteger(m) || m < 1 || m > LINEAR_GEMV_MAX_ROWS) {
    throw new CodegenError(
      `linear_gemv params: m must be an integer in 1..${LINEAR_GEMV_MAX_ROWS} (${m})`,
    );
  }
  if (!Number.isSafeInteger(k) || k < 0 || k % unit !== 0) {
    throw new CodegenError(
      `linear_gemv params: k must be a non-negative integer multiple of ${unit} (${k})`,
    );
  }
  if (groupSize !== undefined && k % groupSize !== 0) {
    throw new CodegenError(
      `linear_gemv params: k=${k} is not divisible by group_size ${groupSize}`,
    );
  }
  return gemmParams("linear", m, n, k);
};

/**
 * Pipeline key (M=1 variant). The family name `linear_gemv` discriminates it from the default path
 * (`linear`); variant, storage and group length are all baked into the WGSL, so they go into the key
 * (codegen determinism: same key → byte-identical WGSL).
 *
 * The storage discriminator (`:wi4` / `:wi8`) and the group fragment (`g32`) use the spelling of
 * weight-storage.ts as is — aligned with the existing reading in which diagnostics and the census
 * identify the path by `:wi4g32` / `:wi8` (ADR 0069 decision 5).
 */
export const linearGemvKey = (
  storage: WeightStorage,
  groupSize?: number,
  variant: LinearGemvVariant = defaultLinearGemvVariant(),
): string => {
  assertVariant(variant);
  gemvGroupShift(storage, groupSize);
  return `linear_gemv:v1:f32:c${variant.cols}u${variant.unroll}${weightKeyPart(storage)}${i4GroupKeyPart(
    groupSize,
  )}`;
};

/**
 * Pipeline key (row-block variant). The M=1 variant's key plus `r<rows>`; `rows` is baked into the
 * WGSL (accumulator count, y-tile height), so it goes into the key.
 * NOTE: `r1` is a different text from the M=1 variant (a row block tiled along y) — they are told
 * apart by the presence of the discriminator.
 */
export const linearGemvRowsKey = (
  storage: WeightStorage,
  groupSize: number | undefined,
  variant: LinearGemvRowsVariant,
): string => {
  assertRowsVariant(variant);
  assertRowsStorage(storage);
  gemvGroupShift(storage, groupSize);
  return `linear_gemv:v2:f32:c${variant.cols}u${variant.unroll}r${variant.rows}${weightKeyPart(
    storage,
  )}${i4GroupKeyPart(groupSize)}`;
};

/**
 * Bindings. **The same numbers and meanings as the default path's linear** (0 dims / 1 x / 2 w /
 * 3 bias / 4 out / 5 wscale), taking the binding list `buildLinear` builds as is.
 *
 * Only 2 element types change: the weight is `vec4<u32>` (16 B = 32 i4 / 16 i8 elements read at
 * once), and the output is `f32` (a scalar write of 1 column per thread — unlike the `vec4<f32>` of
 * the default v4 path, n need not be divisible).
 */
const bindings = (unit: number, storage: WeightStorage, quantize = false, packed = false): string =>
  `@group(0) @binding(1) var<storage, read> x: array<vec4<${packed ? "u32" : "f32"}>>;
// Row starts are 16 B aligned because k % ${unit} == 0 (guaranteed by the eligibility check)
@group(0) @binding(2) var<storage, read> w: array<vec4<${storage === "f32" ? "f32" : "u32"}>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<${quantize ? "u32" : "f32"}>;
${
  storage === "f16" || storage === "bf16" || storage === "f32"
    ? ""
    : "@group(0) @binding(5) var<storage, read> wscale: array<f32>;"
}`;

/**
 * The reads for one word (the weight word + the first x quad + the group scale, for i4 only).
 *
 * The i8 scale is per output channel and invariant in the K loop, so it is bound once outside the
 * reduction rather than here ({@link weightScaleWgsl} — the same hoisting as ADR 0019).
 */
const unitLoads = (
  storage: WeightStorage,
  slot: string,
  unitExpr: string,
  shift: number | undefined,
  inputBase = "",
  packed = false,
  /** false for the staged-activation variant, which reads x from workgroup memory instead. */
  activation = true,
): string => {
  const unit = linearGemvUnit(storage);
  const groupScale =
    storage === "i4"
      ? `
    let ws${slot} = wscale[scale_base + ((unit${slot} * ${unit}u) >> ${shift}u)];`
      : "";
  // With packed activations one binding element (`vec4<u32>`) carries 16 elements (an f32 quad
  // carries 4).
  const xq = activation
    ? `
    let xq${slot} = ${inputBase}unit${slot} * ${unit / (packed ? 16 : 4)}u;`
    : "";
  return `    let unit${slot} = ${unitExpr};
    let pw${slot} = w[row_base + unit${slot}];${groupScale}${xq}`;
};

/**
 * Component names within a word (static `vec4` indices — the discipline that avoids dynamic
 * indexing on Metal).
 */
const LANES = ["x", "y", "z", "w"] as const;

/**
 * The workgroup-memory slot of activation quad `index` in the staged-activation variant (K-90): the
 * low 3 bits are XORed with the next 3, so the 32 lanes of a column, which read quads `V` apart
 * (`V` = quads per word), spread over the banks instead of hitting the same one. The map only
 * permutes slots inside each block of 8, so it stays within the array.
 */
const stagedSlot = (index: string): string => `(${index}) ^ (((${index}) >> 3u) & 7u)`;

/**
 * One line that reads one activation quad (4 elements). By default it reads a `vec4<f32>` quad as
 * is.
 *
 * With packed activations (ADR 0105) the binding becomes `array<vec4<u32>>`, and **16 elements = 1
 * entry** can be read. The word is read once per 4 quads, and per quad it is turned back into int8
 * codes with `unpack4xI8`, then `f32(code) * x_scale` makes the f32 value. This brings the activation
 * loads per weight word from 16 → 4 for i2, 8 → 2 for i4 and 4 → 1 for i8 (the bottleneck in
 * research 2026-09-19 §14).
 *
 * MUST: the literal form of the restoration is `vec4<f32>(unpack4xI8(…)) * dims.x_scale` = **one f32
 * multiply per element**. It matches the producer-side SRQ output table (`params[129 + level]` =
 * `Math.fround(level*scale)` in static-quantize.ts) u32 for u32 per element — the product of
 * level ≤ 128 and an f32 scale is exact in f64, so both the table and this multiply equal "the exact
 * product correctly rounded to f32" (ADR 0105 decision 3).
 * MUST: the components of a word are read with static indices (`.x` / `.y` / `.z` / `.w`) (the same
 * Metal discipline as the default path).
 * NOTE: no rounding barrier is placed on the restored value. The divergence from the f32 path on
 * Metal came from how the multiply-add was contracted to fma; spelling the parallel family's
 * multiply-add with explicit `fma()` makes both paths agree ({@link mac} — ADR 0105 addendum 4).
 */
const activationQuad = (
  packed: boolean,
  slot: string,
  name: string,
  quad: number,
  staged = false,
): string => {
  if (staged) {
    if (packed) throw new CodegenError("linear_gemv: staged activations take f32 quads only");
    return `    let ${name} = sx[${stagedSlot(`sx_base + ${quad}u`)}];`;
  }
  if (!packed) return `    let ${name} = x[xq${slot} + ${quad}u];`;
  const word = quad >> 2;
  const load = quad % 4 === 0 ? `    let xp${slot}_${word} = x[xq${slot} + ${word}u];\n` : "";
  return `${load}    let ${name} = vec4<f32>(unpack4xI8(xp${slot}_${word}.${
    LANES[quad & 3]
  })) * dims.x_scale;`;
};

/**
 * One multiply-add line. The parallel family (`parallelWgsl`'s f32 / packed × without / with
 * fusion) and the subgroup32 variant spell it with explicit `fma()`.
 *
 * MUST: the parallel family and subgroup32 use `fma`. The spelling `acc + x * d` leaves contraction
 * to fma to the compiler; on RTX / Vulkan it is always contracted (u32-identical to explicit fma — a
 * sweep of 540 combinations), but Metal (M2) contracts differently per expression form, and 2
 * kernels with the same formula (f32 and packed) diverged in u32. Explicit fma removes the freedom to
 * contract itself, so both paths round the same way (ADR 0105 addendum 4). subgroup32 is a variant
 * that computes the same sum as parallel through subgroup value exchange, so its multiply-add
 * spelling is also identical to parallel's (ADR 0101 decision 3, addendum 2026-09-25). The
 * sequential GEMV and the row block keep the existing spelling so as not to move the numerics of the
 * reference path.
 */
const mac = (fma: boolean, x: string, d: string): string =>
  fma ? `    acc = fma(${x}, ${d}, acc);` : `    acc = acc + ${x} * ${d};`;

/**
 * Unrolled multiply-adds for one word (32 i4 elements) (M=1 variant).
 *
 * The nibble order is the same as `dequant4` in weight-storage.ts (element 2i = low / 2i+1 = high,
 * stored value `u = q + 8` — the canonical source is the exporter's `karume/emit.py: pack_int4`).
 * MUST: the unroll order is ascending element order within the word (component x→w × byte x→w ×
 * low→high nibble) — that is exactly "sequential in ascending k", and breaking it breaks bit
 * identity with the default path.
 * MUST: x is read from the `vec4<f32>` binding with **static components** (a dynamic component
 * index spills to the local region on Metal — the same discipline as `storeBTransposed` in
 * gemm.ts).
 * MUST: this text does not change by a single byte from before row blocks were introduced (the
 * generated decode shaders — tests/fixtures/wgsl/linear_gemv_*.wgsl are the detectors).
 */
const unitMacsI4 = (slot: string, packed = false, fma = false, staged = false): string => {
  const lanes = ["x", "y", "z", "w"] as const;
  return lanes
    .map((component, quad) => {
      const bytes = `b${slot}_${quad}`;
      const xa = `xa${slot}_${quad}`;
      const xb = `xb${slot}_${quad}`;
      // Component `quad` of the word is elements 8·quad..8·quad+7 = 2 quads of x.
      const macs = lanes
        .flatMap((byte, lane) => {
          const source = lane < 2 ? xa : xb;
          const low = lanes[(lane * 2) % 4];
          const high = lanes[(lane * 2 + 1) % 4];
          return [
            mac(fma, `${source}.${low}`, `(f32(i32(${bytes}.${byte} & 0xFu) - 8) * ws${slot})`),
            mac(fma, `${source}.${high}`, `(f32(i32(${bytes}.${byte} >> 4u) - 8) * ws${slot})`),
          ];
        })
        .join("\n");
      return `    let ${bytes} = unpack4xU8(pw${slot}.${component});
${activationQuad(packed, slot, xa, quad * 2, staged)}
${activationQuad(packed, slot, xb, quad * 2 + 1, staged)}
${macs}`;
    })
    .join("\n");
};

/**
 * Unrolled multiply-adds for one word (16 i8 elements) (M=1 variant).
 *
 * The lane order is the same as `dequant4` in weight-storage.ts (the i8 quad version =
 * `vec4<f32>(unpack4xI8(…)) * scale`), and the 4 elements of component `quad` correspond exactly to
 * one quad of x (the only difference from the 2 quads of i4).
 * MUST: the unroll order is ascending element order within the word (component x→w × lane x→w), and
 * the literal form is the per-element multiply `f32(q) * ws`.
 * MUST: x is read from the `vec4<f32>` binding with **static components** (the same Metal discipline
 * as the i4 version).
 */
const unitMacsI8 = (slot: string, packed = false, fma = false, staged = false): string => {
  const lanes = ["x", "y", "z", "w"] as const;
  return lanes
    .map((component, quad) => {
      const bytes = `b${slot}_${quad}`;
      const xa = `xa${slot}_${quad}`;
      const macs = lanes
        .map((lane) => mac(fma, `${xa}.${lane}`, `(f32(${bytes}.${lane}) * ${WEIGHT_SCALE_VAR})`))
        .join("\n");
      return `    let ${bytes} = unpack4xI8(pw${slot}.${component});
${activationQuad(packed, slot, xa, quad, staged)}
${macs}`;
    })
    .join("\n");
};

/**
 * f16, 8 elements per word. unpack2x16float is the same decode as the default GEMM, with no scale.
 * MUST: multiply-add in the order word x→w, then low→high within each pair. The activation's
 * component indices are unrolled statically.
 */
const unitMacsF16 = (slot: string): string => {
  const lanes = ["x", "y", "z", "w"] as const;
  return lanes
    .map((component, pair) => {
      const weights = `h${slot}_${pair}`;
      const activation = `x${slot}_${pair}`;
      return `    let ${weights} = unpack2x16float(pw${slot}.${component});
    let ${activation} = x[xq${slot} + ${Math.floor(pair / 2)}u];
    acc = acc + ${activation}.${lanes[(pair * 2) % 4]} * ${weights}.x;
    acc = acc + ${activation}.${lanes[(pair * 2 + 1) % 4]} * ${weights}.y;`;
    })
    .join("\n");
};

/**
 * bf16, 8 elements per word. The packing and the ascending-K multiply-add are the same text as the
 * f16 version; only the decode is a bitcast (low half of the word = `<< 16u`, high half =
 * `& 0xFFFF0000u` — ADR 0131).
 */
const unitMacsBf16 = (slot: string): string => {
  const lanes = ["x", "y", "z", "w"] as const;
  return lanes
    .map((component, pair) => {
      const word = `b${slot}_${pair}`;
      const activation = `x${slot}_${pair}`;
      return `    let ${word} = pw${slot}.${component};
    let ${activation} = x[xq${slot} + ${Math.floor(pair / 2)}u];
    acc = acc + ${activation}.${lanes[(pair * 2) % 4]} * bitcast<f32>(${word} << 16u);
    acc = acc + ${activation}.${lanes[(pair * 2 + 1) % 4]} * bitcast<f32>(${word} & 0xFFFF0000u);`;
    })
    .join("\n");
};

/**
 * f32, 4 elements per word. Unrolls the same ascending-K multiply-add as the regular GEMM with
 * static components.
 */
const unitMacsF32 = (slot: string): string => {
  const macs = ["x", "y", "z", "w"]
    .map((lane) => `    acc = acc + xf${slot}.${lane} * pw${slot}.${lane};`)
    .join("\n");
  return `    let xf${slot} = x[xq${slot}];
${macs}`;
};

/**
 * Multiply-adds an INT2 16 B word in ascending K. Static component indices avoid dynamic indexing
 * on Metal.
 */
const unitMacsI2 = (slot: string, packed = false, fma = false, staged = false): string => {
  const lanes = ["x", "y", "z", "w"] as const;
  return lanes
    .flatMap((component, word) =>
      lanes.map((_, byte) => {
        const index = word * 4 + byte;
        const quantized = `q${slot}_${index}`;
        const decoded = `d${slot}_${index}`;
        const activation = `x${slot}_${index}_0`;
        const products = `${activationQuad(packed, slot, activation, index, staged)}
${lanes.map((lane) => mac(fma, `${activation}.${lane}`, `${decoded}.${lane}`)).join("\n")}`;
        return `    let ${quantized} = (pw${slot}.${component} >> ${byte * 8}u) & 255u;
    let ${decoded} = vec4<f32>(vec4<i32>(vec4<u32>(${quantized}, ${quantized} >> 2u, ${quantized} >> 4u, ${quantized} >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * ${WEIGHT_SCALE_VAR};
${products}`;
      }),
    )
    .join("\n");
};

/**
 * Puts only the INT2 row block into a function, reducing the first-time parse cost of unrolling per
 * word and per row. The ascending K per output, the static component indices and the rounding
 * points of decode and multiply-add are the same as at M=1.
 * DECIDED: docs/decisions/0082-linear-gemv-decode.md#addendum-8-2026-09-11--smaller-int2-row-block-shaders-k-30
 */
const i2WordWgsl = (): string => `
fn linear_i2_word(pwa: vec4<u32>, xqa: u32, wscale_v: f32, initial: f32) -> f32 {
  var acc = initial;
${unitMacsI2("a")}
  return acc;
}
`;

const rowsMacsI2 = (slot: string, rows: number): string =>
  Array.from(
    { length: rows },
    (_, row) =>
      `    acc${row} = linear_i2_word(pw${slot}, xr${row} + xq${slot}, wscale_v, acc${row});`,
  ).join("\n");

/**
 * The multiply-adds of one word, by storage. Also the body of `expert_linear`'s kernel
 * (src/kernels/expert-linear.ts), so the two decode the same text.
 */
export const unitMacs = (
  storage: WeightStorage,
  slot: string,
  packed = false,
  fma = false,
  staged = false,
): string =>
  storage === "f32"
    ? unitMacsF32(slot)
    : storage === "f16"
      ? unitMacsF16(slot)
      : storage === "bf16"
        ? unitMacsBf16(slot)
        : storage === "i4"
          ? unitMacsI4(slot, packed, fma, staged)
          : storage === "i2"
            ? unitMacsI2(slot, packed, fma, staged)
            : unitMacsI8(slot, packed, fma, staged);

/**
 * I4/I8 also put the per-row multiply-add into a small function. The ascending K and the f32
 * rounding points are the same as at M=1. Calling a function instead of duplicating the dequantise
 * expression per row reduces the amount of WGSL to parse and compile.
 * DECIDED: docs/decisions/0082-linear-gemv-decode.md#addendum-9-2026-09-11--the-function-form-for-int4-and-int8-row-blocks-too
 */
const rowsWordCall = (storage: "i4" | "i8", slot: string, rows: number): string =>
  Array.from({ length: rows }, (_, row) => {
    const scale = storage === "i4" ? `ws${slot}` : "wscale_v";
    return `    acc${row} = linear_${storage}_word(pw${slot}, xr${row} + xq${slot}, ${scale}, acc${row});`;
  }).join("\n");

const wordHelperWgsl = (storage: WeightStorage): string => {
  if (storage === "i2") return i2WordWgsl();
  if (storage === "f16" || storage === "bf16" || storage === "f32") return "";
  const scale = storage === "i4" ? "wsa" : "wscale_v";
  return `
fn linear_${storage}_word(pwa: vec4<u32>, xqa: u32, ${scale}: f32, initial: f32) -> f32 {
  var acc = initial;
${storage === "i4" ? unitMacsI4("a") : unitMacsI8("a")}
  return acc;
}
`;
};

const rowsMacs = (storage: WeightStorage, slot: string, rows: number): string =>
  storage === "i4"
    ? rowsWordCall("i4", slot, rows)
    : storage === "i2"
      ? rowsMacsI2(slot, rows)
      : rowsWordCall("i8", slot, rows);

/**
 * i4 derives the group start from the number of scales per row / i8 hoists the single
 * per-output-channel scale.
 */
const scaleSetupWgsl = (
  storage: WeightStorage,
  shift: number | undefined,
  column = "col",
): string =>
  storage === "i4"
    ? `
  let scale_base = ${column} * (dims.k >> ${shift}u);`
    : weightScaleWgsl(storage, column, "  ");

/**
 * The GEMV WGSL (`out[n] = x[k] · wᵀ[n,k] + bias[n]`, M=1, weights stored as f32 / f16 / i2 / i4 /
 * i8).
 *
 * One thread holds the whole reduction of one output column, so parallelism is bounded by `n`. That
 * is the very price of bit identity: splitting along k would raise parallelism but move the
 * reduction order (MUST NOT — module doc).
 */
export const linearGemvWgsl = (
  storage: WeightStorage,
  groupSize?: number,
  variant: LinearGemvVariant = defaultLinearGemvVariant(),
): string => {
  assertVariant(variant);
  const unit = linearGemvUnit(storage);
  const shift = gemvGroupShift(storage, groupSize);
  const { cols, unroll } = variant;
  const slots = Array.from({ length: unroll }, (_, slot) => `${slot}`);
  const loads = slots.map((slot) => unitLoads(storage, slot, `unit + ${slot}u`, shift)).join("\n");
  const macs = slots.map((slot) => unitMacs(storage, slot)).join("\n");
  return `// karume linear gemv (M=1: out[n] = x[k] · wᵀ[n,k] + bias[n], f32${weightNote(
    storage,
  )}, ${cols} cols / wg, ${unroll} words prefetched)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
${bindings(unit, storage)}

@compute @workgroup_size(${cols})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  // No shared memory and no barrier, so edge workgroups may return early
  if (col >= dims.n) {
    return;
  }
  let units = dims.k / ${unit}u;
  let row_base = col * units;${scaleSetupWgsl(storage, shift)}
  var acc = 0.0;
  var unit = 0u;
  // Issue all prefetched weight words **first**, then do the multiply-adds (memory parallelism).
  // Words are still processed in ascending order, so the addition order per output element is the
  // same whatever the prefetch count
  for (; unit + ${unroll}u <= units; unit = unit + ${unroll}u) {
${loads}
${macs}
  }
  // Remainder words (units % ${unroll} of them) — walk the same order one word at a time
  for (; unit < units; unit = unit + 1u) {
${unitLoads(storage, "t", "unit", shift)}
${unitMacs(storage, "t")}
  }
  out[col] = acc + bias[col];
}
`;
};

/**
 * The row-block GEMV WGSL (`out[m,n] = x[m,k] · wᵀ[n,k] + bias[n]`, M ≥ 2, weights stored as i2 /
 * i4 / i8).
 *
 * The dispatch is `[ceil(n / cols), ceil(m / rows), 1]`. One thread holds column `gid.x` × rows
 * `gid.y · rows .. +rows`, reads the weight word and the scale once and multiply-adds them with each
 * row's x.
 * MUST: rows beyond m **re-read the last row of x** (to keep reads inside the binding), but **only
 * rows with `row < m` are written back**. The output shape is `[m, n]`; column c of row r is
 * `out[r·n + c]`.
 * MUST: `dims.m ≥ 1` is assumed (`min(row0 + r, dims.m - 1u)` wraps the u32 around at m = 0). It is
 * guaranteed by the m range check of {@link linearGemvParams} (1..{@link LINEAR_GEMV_MAX_ROWS}) —
 * the WGSL side does not check it.
 * MUST: the literal form and order of the reduction are the same as the M=1 variant
 * ({@link linearGemvWgsl}) (the numeric contract in the module doc).
 */
export const linearGemvRowsWgsl = (
  storage: WeightStorage,
  groupSize: number | undefined,
  variant: LinearGemvRowsVariant,
): string => {
  assertRowsVariant(variant);
  assertRowsStorage(storage);
  const unit = linearGemvUnit(storage);
  const shift = gemvGroupShift(storage, groupSize);
  const { cols, unroll, rows } = variant;
  const slots = Array.from({ length: unroll }, (_, slot) => `${slot}`);
  const rowList = Array.from({ length: rows }, (_, row) => row);
  const loads = slots.map((slot) => unitLoads(storage, slot, `unit + ${slot}u`, shift)).join("\n");
  const macs = slots.map((slot) => rowsMacs(storage, slot, rows)).join("\n");
  const rowHeads = rowList
    .map((row) => `  let xr${row} = min(row0 + ${row}u, dims.m - 1u) * kq;`)
    .join("\n");
  const accs = rowList.map((row) => `  var acc${row} = 0.0;`).join("\n");
  const stores = rowList
    .map(
      (row) =>
        `  if (row0 + ${row}u < dims.m) {
    out[(row0 + ${row}u) * dims.n + col] = acc${row} + bias[col];
  }`,
    )
    .join("\n");
  return `// karume linear gemv rows (M≥2: out[m,n] = x[m,k] · wᵀ[n,k] + bias[n], f32${weightNote(
    storage,
  )}, ${cols} cols / wg, ${unroll} words prefetched, ${rows} rows / thread)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
${bindings(unit, storage)}${wordHelperWgsl(storage)}

@compute @workgroup_size(${cols})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  // No shared memory and no barrier, so edge workgroups may return early
  if (col >= dims.n) {
    return;
  }
  let units = dims.k / ${unit}u;
  let row_base = col * units;
  // The first row of the row block and the start of x per row (in quads). Rows beyond m re-read the
  // last row (only to keep reads in range; write-back is only inside the row count)
  let kq = dims.k / 4u;
  let row0 = gid.y * ${rows}u;
${rowHeads}${scaleSetupWgsl(storage, shift)}
${accs}
  var unit = 0u;
  // Issue all prefetched weight words **first**, then do the multiply-adds (memory parallelism).
  // Words are still processed in ascending order, so the addition order per output element is the
  // same whatever the prefetch count
  for (; unit + ${unroll}u <= units; unit = unit + ${unroll}u) {
${loads}
${macs}
  }
  // Remainder words (units % ${unroll} of them) — walk the same order one word at a time
  for (; unit < units; unit = unit + 1u) {
${unitLoads(storage, "t", "unit", shift)}
${rowsMacs(storage, "t", rows)}
  }
${stores}
}
`;
};

/** Number of lanes sharing K. One workgroup is always 128 threads (512 bytes of shared partial sums). */
export type LinearGemvParallelLanes = 2 | 4 | 8 | 16 | 32;

type ParallelShape = {
  readonly storage: WeightStorage;
  readonly n: number;
  readonly k: number;
  readonly group?: number;
  readonly lanes: LinearGemvParallelLanes;
  /**
   * Whether the shape takes packed int8 activations (ADR 0105). **true only for rows that measured
   * faster**. The E4B rows are the exception: an unmeasured provisional value copied from the E2B row
   * with the same role (the E4B section of the table).
   *
   * It pays off only for "long K with lanes 32" = shapes where one thread reads many activations per
   * weight word; for K=1536 with lanes 2 / 4 and for i8, the extra unpack / conversion / multiply eat
   * the gain (per-key measurement of ADR 0105 addendum 1).
   */
  readonly packedActivations: boolean;
  /**
   * Whether the shape takes the staged-activation variant (K-90): the workgroup loads each run of x
   * quads once into workgroup memory and the columns read it from there, instead of every column
   * re-reading x from storage. **true only for rows that measured faster** (RTX 3080 Ti, 2026-10-25:
   * the lanes-32 rows with K ≥ 2048 at ×1.37–1.63; K = 1536 and the lanes-4 rows were ×0.86–1.06).
   * The E4B rows copy the E2B row with the same role, as for `packedActivations`. Never set together
   * with `packedActivations` (packed words already cut the activation loads).
   */
  readonly stagedActivations?: true;
};

// DECIDED: only measured shapes are eligible for the opt-in option. Not an automatic selection by
// GPU name.
// docs/decisions/0098-linear-gemv-parallel.md
const PARALLEL_SHAPES: readonly ParallelShape[] = [
  { storage: "i4", n: 8960, k: 1536, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 2048, k: 1536, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 256, k: 1536, group: 32, lanes: 32, packedActivations: false },
  {
    storage: "i4",
    n: 1536,
    k: 2048,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  { storage: "i4", n: 6144, k: 1536, group: 32, lanes: 4, packedActivations: false },
  {
    storage: "i4",
    n: 1536,
    k: 6144,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  { storage: "i4", n: 1536, k: 256, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 4096, k: 1536, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 512, k: 1536, group: 32, lanes: 32, packedActivations: false },
  {
    storage: "i4",
    n: 1536,
    k: 4096,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  { storage: "i4", n: 12288, k: 1536, group: 32, lanes: 4, packedActivations: false },
  {
    storage: "i4",
    n: 1536,
    k: 12288,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  { storage: "i8", n: 262144, k: 1536, lanes: 16, packedActivations: false },
  { storage: "i4", n: 2048, k: 1536, group: 512, lanes: 4, packedActivations: false },
  { storage: "i4", n: 256, k: 1536, group: 512, lanes: 32, packedActivations: false },
  { storage: "i4", n: 1536, k: 2048, group: 2048, lanes: 32, packedActivations: true },
  { storage: "i4", n: 6144, k: 1536, group: 512, lanes: 4, packedActivations: false },
  { storage: "i4", n: 1536, k: 6144, group: 2048, lanes: 32, packedActivations: true },
  { storage: "i8", n: 256, k: 1536, lanes: 32, packedActivations: false },
  { storage: "i8", n: 1536, k: 256, lanes: 4, packedActivations: false },
  { storage: "i4", n: 4096, k: 1536, group: 512, lanes: 4, packedActivations: false },
  { storage: "i4", n: 512, k: 1536, group: 512, lanes: 32, packedActivations: false },
  { storage: "i4", n: 1536, k: 4096, group: 4096, lanes: 32, packedActivations: true },
  { storage: "i2", n: 12288, k: 1536, lanes: 2, packedActivations: false },
  { storage: "i2", n: 1536, k: 12288, lanes: 32, packedActivations: true },
  // ---- E4B (added 2026-09-26) — **none of the rows below is measured**. lanes / packedActivations
  // are guesses copied from the E2B row with the same role, to be revisited with E4B's flag-bench
  // (per-key census and GPU time) (ADR 0098 addendum 2026-09-26). The shapes are the real ones, read
  // for regular E4B from the upstream config (hidden 2560, intermediate 10240, kv 2, 42 layers, PLE
  // 256) and for QAT E4B from the container descriptor of the distribution mirror.
  // E4B regular i4 g32: per_layer_model_projection (42×256) — lanes from E2B 8960×1536 (unmeasured)
  { storage: "i4", n: 10752, k: 2560, group: 32, lanes: 4, packedActivations: false },
  // E4B regular i4 g32: q (sliding) — lanes from E2B 2048×1536 (unmeasured)
  { storage: "i4", n: 2048, k: 2560, group: 32, lanes: 4, packedActivations: false },
  // E4B regular i4 g32: k / v (sliding) — lanes from E2B 256×1536 (unmeasured)
  { storage: "i4", n: 512, k: 2560, group: 32, lanes: 32, packedActivations: false },
  // E4B regular i4 g32: per_layer_input_gate — lanes from E2B 256×1536 (unmeasured)
  { storage: "i4", n: 256, k: 2560, group: 32, lanes: 32, packedActivations: false },
  // E4B regular i4 g32: o (sliding) — lanes from E2B 1536×2048 (unmeasured)
  {
    storage: "i4",
    n: 2560,
    k: 2048,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  // E4B regular i4 g32: gate / up — lanes from E2B 6144×1536 (unmeasured)
  { storage: "i4", n: 10240, k: 2560, group: 32, lanes: 4, packedActivations: false },
  // E4B regular i4 g32: down — lanes from E2B 1536×6144 (unmeasured)
  {
    storage: "i4",
    n: 2560,
    k: 10240,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  // E4B regular i4 g32: per_layer_projection — lanes from E2B 1536×256 (unmeasured)
  { storage: "i4", n: 2560, k: 256, group: 32, lanes: 4, packedActivations: false },
  // E4B regular i4 g32: q (full) — lanes from E2B 4096×1536 (unmeasured)
  { storage: "i4", n: 4096, k: 2560, group: 32, lanes: 4, packedActivations: false },
  // E4B regular i4 g32: k / v (full) — lanes from E2B 512×1536 (unmeasured)
  { storage: "i4", n: 1024, k: 2560, group: 32, lanes: 32, packedActivations: false },
  // E4B regular i4 g32: o (full) — lanes from E2B 1536×4096 (unmeasured)
  {
    storage: "i4",
    n: 2560,
    k: 4096,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  // E4B regular i8: lm_head (tied embedding) — lanes from E2B 262144×1536 (unmeasured)
  { storage: "i8", n: 262144, k: 2560, lanes: 16, packedActivations: false },
  // E4B QAT i4 g512: q (sliding) — lanes from E2B QAT 2048×1536 g512 (unmeasured)
  { storage: "i4", n: 2048, k: 2560, group: 512, lanes: 4, packedActivations: false },
  // E4B QAT i4 g512: k / v (sliding) — lanes from E2B QAT 256×1536 g512 (unmeasured)
  { storage: "i4", n: 512, k: 2560, group: 512, lanes: 32, packedActivations: false },
  // E4B QAT i4 g2048: o (sliding) — lanes / packed from E2B QAT 1536×2048 g2048 (unmeasured)
  { storage: "i4", n: 2560, k: 2048, group: 2048, lanes: 32, packedActivations: true },
  // E4B QAT i4 g512: gate / up — lanes from E2B QAT 6144×1536 g512 (unmeasured)
  { storage: "i4", n: 10240, k: 2560, group: 512, lanes: 4, packedActivations: false },
  // E4B QAT i4 g2048: down — lanes / packed from E2B QAT 1536×6144 g2048 (unmeasured)
  { storage: "i4", n: 2560, k: 10240, group: 2048, lanes: 32, packedActivations: true },
  // E4B QAT i8: per_layer_input_gate — lanes from E2B QAT 256×1536 i8 (unmeasured)
  { storage: "i8", n: 256, k: 2560, lanes: 32, packedActivations: false },
  // E4B QAT i8: per_layer_projection — lanes from E2B QAT 1536×256 i8 (unmeasured)
  { storage: "i8", n: 2560, k: 256, lanes: 4, packedActivations: false },
  // E4B QAT i4 g512: q (full) — lanes from E2B QAT 4096×1536 g512 (unmeasured)
  { storage: "i4", n: 4096, k: 2560, group: 512, lanes: 4, packedActivations: false },
  // E4B QAT i4 g512: k / v (full) — lanes from E2B QAT 512×1536 g512 (unmeasured)
  { storage: "i4", n: 1024, k: 2560, group: 512, lanes: 32, packedActivations: false },
  // E4B QAT i4 g4096: o (full) — lanes / packed from E2B QAT 1536×4096 g4096 (unmeasured)
  { storage: "i4", n: 2560, k: 4096, group: 4096, lanes: 32, packedActivations: true },
  // ---- 12B q4_0 QAT (added 2026-11-06) — lanes measured as whole-model decode on the RTX 3080 Ti
  // (ADR 0098 addendum 2026-11-06): 4 beats 16 and 32 on the rows that widen, staged 32 ties 16 and
  // beats plain 32 on the rows that narrow, and the output head takes 16 (32 does not build).
  // gate / up
  { storage: "i4", n: 15360, k: 3840, group: 32, lanes: 4, packedActivations: false },
  // q (full), q (sliding), k / v (sliding)
  { storage: "i4", n: 8192, k: 3840, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 4096, k: 3840, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 2048, k: 3840, group: 32, lanes: 4, packedActivations: false },
  // k (full)
  { storage: "i4", n: 512, k: 3840, group: 32, lanes: 32, packedActivations: false },
  // o (sliding), o (full), down
  {
    storage: "i4",
    n: 3840,
    k: 4096,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  {
    storage: "i4",
    n: 3840,
    k: 8192,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  {
    storage: "i4",
    n: 3840,
    k: 15360,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  // lm_head (tied embedding)
  { storage: "i4", n: 262144, k: 3840, group: 32, lanes: 16, packedActivations: false },
  // ---- 26B A4B q4_0 QAT (added 2026-11-10) — the dense linears of the mixture of experts (the
  // routed experts run on `expert_linear`, which has no parallel form). Lanes measured per kernel
  // as whole-model decode on the RTX 3080 Ti (ADR 0098 addendum 2026-11-10): on the rows that
  // widen 4, 16 and 32 tie and 2 and 8 lose, so they take 12B's 4; the router and the full k take
  // 32; the rows that narrow take staged 32 (staged 16 ties, unstaged 32 and plain 4 lose); and
  // the output head takes 4, which beats 8 and 16 here (32 does not build).
  // q (full), q (sliding), k / v (sliding), gate / up
  { storage: "i4", n: 8192, k: 2816, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 4096, k: 2816, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 2048, k: 2816, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 2112, k: 2816, group: 32, lanes: 4, packedActivations: false },
  // k (full), router
  { storage: "i4", n: 1024, k: 2816, group: 32, lanes: 32, packedActivations: false },
  { storage: "i4", n: 128, k: 2816, group: 32, lanes: 32, packedActivations: false },
  // o (sliding), o (full), down
  {
    storage: "i4",
    n: 2816,
    k: 4096,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  {
    storage: "i4",
    n: 2816,
    k: 8192,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  {
    storage: "i4",
    n: 2816,
    k: 2112,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  // lm_head (tied embedding)
  { storage: "i4", n: 262144, k: 2816, group: 32, lanes: 4, packedActivations: false },
];

/** Row lookup in the measured table (including the M range, in one place). */
const parallelShapeFor = (
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
  group?: number,
): ParallelShape | undefined => {
  if (m < 1 || m > 8) return undefined;
  return PARALLEL_SHAPES.find(
    (shape) => shape.storage === storage && shape.n === n && shape.k === k && shape.group === group,
  );
};

/**
 * M=1/4/8 of the same shape have the same addition order. Prefill with M>8 keeps the existing row
 * block.
 */
export const linearGemvParallelLanes = (
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
  group?: number,
): LinearGemvParallelLanes | undefined => parallelShapeFor(storage, m, n, k, group)?.lanes;

/**
 * The **only pure function** that returns "whether this shape goes to the parallel GEMV (the
 * {@link linearGemvParallelWgsl} family)".
 *
 * {@link linearGemvParallelLanes} (the lookup of measured shapes) overlaid with the conditions under
 * which `buildLinear` in recipe-builders/linear.ts sends a shape to the GEMV family (storage,
 * `k % step`, the i4 group length, the measured range of vec4 output). The pairing of packed
 * activations (ADR 0105) reads {@link linearGemvPackedEligible}, which narrows this further by the
 * per-row adoption.
 */
export const linearGemvParallelEligible = (
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
  group?: number,
): LinearGemvParallelLanes | undefined => {
  if (storage !== "i2" && storage !== "i4" && storage !== "i8") return undefined;
  if (!gemmUsesVec4(k, n)) return undefined;
  if (k <= 0 || k % linearGemvUnit(storage) !== 0) return undefined;
  if (storage === "i4") {
    if (group === undefined || group % linearGemvUnit("i4") !== 0) return undefined;
  } else if (group !== undefined) return undefined;
  return linearGemvParallelLanes(storage, m, n, k, group);
};

/**
 * The **only pure function** that returns "whether this shape takes **packed int8 activations**
 * (ADR 0105)".
 *
 * {@link linearGemvParallelEligible} (whether it goes to the parallel GEMV) overlaid with the
 * `packedActivations` of the measured table's row. The WGSL and params of the packed variant can be
 * generated for every shape (tests and snapshots keep every shape), but **the production plan
 * chooses only shapes for which this predicate returns a lane count**.
 *
 * MUST: the pairing (fusion.ts) and the recipe builder's guard read this same single function. Kept
 * separately, when one of them is wider, f32 words would flow into a `vec4<u32>` binding (a silent
 * wrong value with no exception).
 */
export const linearGemvPackedEligible = (
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
  group?: number,
): LinearGemvParallelLanes | undefined => {
  const lanes = linearGemvParallelEligible(storage, m, n, k, group);
  if (lanes === undefined) return undefined;
  return parallelShapeFor(storage, m, n, k, group)?.packedActivations === true ? lanes : undefined;
};

/**
 * The **only pure function** that returns "whether this shape takes the **staged-activation**
 * parallel GEMV (K-90)": {@link linearGemvParallelEligible} narrowed by the measured table row's
 * `stagedActivations`. The variant is bit-identical to {@link linearGemvParallelWgsl} (same words per
 * lane in the same order, same `fma` text, same tree), so taking it changes speed only.
 *
 * MUST: the recipe builder reads this one function. A shape that is packed (ADR 0105) never takes
 * it — the packed binding is `vec4<u32>`, which the staged loads do not read.
 */
export const linearGemvStagedEligible = (
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
  group?: number,
): LinearGemvParallelLanes | undefined => {
  const lanes = linearGemvParallelEligible(storage, m, n, k, group);
  if (lanes === undefined) return undefined;
  const shape = parallelShapeFor(storage, m, n, k, group);
  return shape?.stagedActivations === true && !shape.packedActivations ? lanes : undefined;
};

export const linearGemvParallelKey = (
  storage: WeightStorage,
  group: number | undefined,
  lanes: LinearGemvParallelLanes,
): string => `linear_gemv_parallel${weightKeyPart(storage)}${i4GroupKeyPart(group)}:l${lanes}`;

/**
 * The opt-in version that distributes compressed words over K lanes round-robin. Dequantisation and
 * the multiply-adds within a word are shared with the sequential version, but the addition order
 * across words differs. It completes with a tree reduction inside 128 threads and requires no extra
 * feature. M is not baked into the shader, so decode and speculative verification of a few rows use
 * the same key and the same arithmetic.
 * DECIDED: docs/decisions/0098-linear-gemv-parallel.md
 */
export const linearGemvParallelWgsl = (
  storage: WeightStorage,
  group: number | undefined,
  lanes: LinearGemvParallelLanes,
): string => parallelWgsl(storage, group, lanes, false);

/**
 * A separate key that also fuses the fixed SRQ. The reference parallel body is kept byte for byte
 * (ADR 0103).
 */
export const linearGemvStaticQuantizeKey = (
  storage: WeightStorage,
  group: number | undefined,
  lanes: LinearGemvParallelLanes,
): string => `${linearGemvParallelKey(storage, group, lanes)}:static-quantize:v2`;

export const linearGemvStaticQuantizeParams = (
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
  scale: number,
  group?: number,
): Uint32Array<ArrayBuffer> => {
  if (m < 1 || m > 8) throw new CodegenError("linear static_quantize: supports only m in 1..8");
  assertRowsStorage(storage);
  const params = new Uint32Array(264);
  params.set(linearGemvParams(storage, m, n, k, group));
  // The 4th word is a run-time rounding barrier. Do not replace the XOR with 0 by a constant expression.
  params.set(staticQuantizeParams(m * n, scale), 4);
  return params;
};

export const linearGemvStaticQuantizeWgsl = (
  storage: WeightStorage,
  group: number | undefined,
  lanes: LinearGemvParallelLanes,
): string => parallelWgsl(storage, group, lanes, true);

/**
 * The key fragment for the packed int8 activation variant (ADR 0105). It is **appended to the end of
 * the existing key**, so none of the storage discriminators diagnostics and the census read
 * (`:wi4g512` / `:l4` / `:static-quantize:v2`) moves.
 */
const PACKED_ACTIVATION_KEY_PART = ":packed-x-i8";

export const linearGemvParallelPackedKey = (
  storage: WeightStorage,
  group: number | undefined,
  lanes: LinearGemvParallelLanes,
): string => `${linearGemvParallelKey(storage, group, lanes)}${PACKED_ACTIVATION_KEY_PART}`;

/**
 * The packed-activation variant of the parallel GEMV. Geometry, lane table, reduction tree and the
 * order of adding the bias do not differ by a single byte from {@link linearGemvParallelWgsl}; **only
 * the activation read and restoration** differ.
 */
export const linearGemvParallelPackedWgsl = (
  storage: WeightStorage,
  group: number | undefined,
  lanes: LinearGemvParallelLanes,
): string => parallelWgsl(storage, group, lanes, false, true);

export const linearGemvStaticQuantizePackedKey = (
  storage: WeightStorage,
  group: number | undefined,
  lanes: LinearGemvParallelLanes,
): string => `${linearGemvStaticQuantizeKey(storage, group, lanes)}${PACKED_ACTIVATION_KEY_PART}`;

/**
 * The packed-activation variant of the parallel GEMV with the fused SRQ epilogue (the output-side
 * SRQ stays f32).
 */
export const linearGemvStaticQuantizePackedWgsl = (
  storage: WeightStorage,
  group: number | undefined,
  lanes: LinearGemvParallelLanes,
): string => parallelWgsl(storage, group, lanes, true, true);

/**
 * The key of the staged-activation variant (K-90): appended to the parallel key, like the packed
 * fragment, so the storage and lane discriminators that diagnostics read do not move.
 */
export const linearGemvParallelStagedKey = (
  storage: WeightStorage,
  group: number | undefined,
  lanes: LinearGemvParallelLanes,
): string => `${linearGemvParallelKey(storage, group, lanes)}:staged-x`;

/**
 * The staged-activation variant of the parallel GEMV (K-90). Per chunk the workgroup copies the next
 * `lanes × quads-per-word` activation quads into workgroup memory with whole-`vec4` stores, then each
 * lane reads its own word's quads from there. The GEMV is limited by the activation loads, not the
 * weights (research 2026-09-19 §14, 2026-10-25 recon §9.1): the 128 / lanes columns of a workgroup
 * share one activation row, so each quad is read from storage once instead of once per column.
 *
 * Bit identity with {@link linearGemvParallelWgsl}: lane `l` still handles words `l, l + lanes, …` in
 * ascending order with the same multiply-add text, and the partial-sum tree and the store are the same
 * text. Uses the same uniform and bindings (params are {@link linearGemvParams}).
 */
export const linearGemvParallelStagedWgsl = (
  storage: WeightStorage,
  group: number | undefined,
  lanes: LinearGemvParallelLanes,
): string => parallelWgsl(storage, group, lanes, false, false, true);

/**
 * Turns the restoration scale of packed activations (= the producer-side SRQ's f32 scale) into u32
 * bits.
 *
 * MUST: a positive, finite value exactly representable in f32 (the same guard as the producer side,
 * {@link staticQuantizeParams}). 0 is the identity SRQ and cannot be turned into packed int8, so it
 * is rejected here too (fail loudly).
 */
const activationScaleBits = (scale: number): number => {
  if (!Number.isFinite(scale) || scale <= 0 || Math.fround(scale) !== scale) {
    throw new CodegenError(
      `linear_gemv packed: activation scale ${scale} must be positive, finite and exactly representable as f32`,
    );
  }
  const scratch = new Float32Array(1);
  scratch[0] = scale;
  return new Uint32Array(scratch.buffer)[0];
};

/** Params of the packed-activation variant (only one word added for the last Dims member `x_scale`). */
export const linearGemvParallelPackedParams = (
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
  xScale: number,
  group?: number,
): Uint32Array<ArrayBuffer> => {
  assertRowsStorage(storage);
  const params = linearGemvParams(storage, m, n, k, group);
  params[3] = activationScaleBits(xScale);
  return params;
};

/**
 * Params of the packed-activation variant with the fused SRQ epilogue. It keeps the existing layout
 * (`m/n/k/rounding_mask` + the 260-word output-side SRQ table) as is and appends one word, `x_scale`,
 * **at the end** (the 16 B alignment of `array<vec4<u32>>` puts it at word 264 — the same derivation
 * as the WGSL Dims).
 */
export const linearGemvStaticQuantizePackedParams = (
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
  xScale: number,
  scale: number,
  group?: number,
): Uint32Array<ArrayBuffer> => {
  const params = new Uint32Array(268);
  params.set(linearGemvStaticQuantizeParams(storage, m, n, k, scale, group));
  params[264] = activationScaleBits(xScale);
  return params;
};

/**
 * Shares the SRQ integer table and the standalone kernel's level guess (v2 — perf-ledger K-97:
 * {@link staticQuantizeGuessWgsl}); a failed guess falls back to the upper-bound search over the
 * 128 boundaries unrolled into a fixed count (the v1 epilogue's search, unchanged).
 */
const staticQuantizeEpilogue = (): string => `
fn srqWord(index: u32) -> u32 { return dims.srq[index >> 2u][index & 3u]; }
fn quantize(bits: u32) -> u32 {
  if (srqWord(258u) != 0u) { return bits; }
  let magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u) { return bits | 0x00400000u; }
  let sign = bits & 0x80000000u;
  let cap = select(127u, 128u, sign != 0u);
${staticQuantizeGuessWgsl("srqWord")}
  let guessed = srqWord(129u + min(c_guess, cap));
  if (!off_guess) { return guessed | sign; }
  var lo = 0u;
${[64, 32, 16, 8, 4, 2, 1]
  .map((step) => `  lo += select(0u, ${step}u, magnitude >= srqWord(lo + ${step}u));`)
  .join("\n")}
  lo += select(0u, 1u, magnitude >= srqWord(128u));
  let level = min(lo, cap);
  return srqWord(129u + level) | sign;
}
`;

/** The K loop of the parallel family: each lane walks words `lane, lane + lanes, …` of its column. */
const plainReduction = (
  storage: WeightStorage,
  lanes: LinearGemvParallelLanes,
  shift: number | undefined,
  packed: boolean,
): string => {
  const unit = linearGemvUnit(storage);
  return `  // Edge columns also write a partial sum of 0, so every thread passes the same barriers.
  if (col < dims.n) {
    let units = dims.k / ${unit}u;
    let row_base = col * units;${scaleSetupWgsl(storage, shift)}
    for (var unit = lane; unit < units; unit += ${lanes}u) {
${unitLoads(storage, "t", "unit", shift, `wg.y * (dims.k / ${packed ? 16 : 4}u) + `, packed)}
${unitMacs(storage, "t", packed, true)}
    }
  }`;
};

/**
 * The K loop of the staged-activation variant (K-90): the same words per lane in the same ascending
 * order as {@link plainReduction}, taken in chunks of `lanes` words. Each chunk first copies the
 * chunk's activation quads into `sx` (one whole-`vec4` store per quad, zero past K), then every lane
 * multiply-adds its word against `sx`.
 * MUST: both barriers sit in uniform control flow (`chunks` comes from the uniform) and every thread
 * of the workgroup — edge columns too — runs every chunk, because the copy is shared.
 */
const stagedReduction = (
  storage: WeightStorage,
  lanes: LinearGemvParallelLanes,
  shift: number | undefined,
): string => {
  const unit = linearGemvUnit(storage);
  const quads = unit / 4;
  const perRound = lanes * quads;
  return `  // Edge columns load their share of the staged activation and pass every barrier; their hoisted
  // reads take the last column, and their sums are never stored.
  let units = dims.k / ${unit}u;
  let col_read = min(col, dims.n - 1u);
  let row_base = col_read * units;${scaleSetupWgsl(storage, shift, "col_read")}
  let sx_base = lane * ${quads}u;
  let chunks = (units + ${lanes - 1}u) / ${lanes}u;
  for (var chunk = 0u; chunk < chunks; chunk += 1u) {
    // Chunk c stages the activation quads of words c·${lanes} … c·${lanes} + ${lanes - 1}.
    for (var q = lid; q < ${perRound}u; q += 128u) {
      let v = chunk * ${perRound}u + q;
      var staged_quad = vec4<f32>(0.0);
      if (v < dims.k / 4u) {
        staged_quad = x[wg.y * (dims.k / 4u) + v];
      }
      sx[${stagedSlot("q")}] = staged_quad;
    }
    workgroupBarrier();
    let unit = chunk * ${lanes}u + lane;
    if (col < dims.n && unit < units) {
${unitLoads(storage, "t", "unit", shift, "", false, false)}
${unitMacs(storage, "t", false, true, true)}
    }
    workgroupBarrier();
  }`;
};

const parallelWgsl = (
  storage: WeightStorage,
  group: number | undefined,
  lanes: LinearGemvParallelLanes,
  quantize: boolean,
  packed = false,
  staged = false,
): string => {
  if (![2, 4, 8, 16, 32].includes(lanes)) {
    throw new CodegenError("linear_gemv_parallel: invalid lane count");
  }
  if (staged && (packed || quantize)) {
    throw new CodegenError(
      "linear_gemv_parallel: staged activations exist for the plain variant only",
    );
  }
  assertRowsStorage(storage);
  const unit = linearGemvUnit(storage);
  const shift = gemvGroupShift(storage, group);
  // MUST: `x_scale` is the **last member** of Dims (the packed variant's params add one word at
  // this position).
  return `// karume linear gemv K parallel (${storage}, ${lanes} lanes/output${
    packed ? ", packed int8 activations" : ""
  }${staged ? ", staged activations" : ""})
struct Dims {
  m: u32,
  n: u32,
  k: u32,${quantize ? "\n  rounding_mask: u32,\n  srq: array<vec4<u32>, 65>," : ""}${
    packed ? "\n  x_scale: f32," : ""
  }
}
@group(0) @binding(0) var<uniform> dims: Dims;
${bindings(unit, storage, quantize, packed)}${quantize ? staticQuantizeEpilogue() : ""}
var<workgroup> partial: array<f32, 128>;${
    staged ? `\nvar<workgroup> sx: array<vec4<f32>, ${lanes * (unit / 4)}u>;` : ""
  }

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let lane = lid % ${lanes}u;
  let col = wg.x * ${128 / lanes}u + lid / ${lanes}u;
  var acc = 0.0;
${staged ? stagedReduction(storage, lanes, shift) : plainReduction(storage, lanes, shift, packed)}
  partial[lid] = acc;
  workgroupBarrier();
  for (var width = ${lanes / 2}u; width > 0u; width /= 2u) {
    if (lane < width) {
      partial[lid] = partial[lid] + partial[lid + width];
    }
    workgroupBarrier();
  }
  if (col < dims.n && lane == 0u) {
    out[wg.y * dims.n + col] = ${
      quantize
        ? "quantize(bitcast<u32>(partial[lid] + bias[col]) ^ dims.rounding_mask)"
        : "partial[lid] + bias[col]"
    };
  }
}
`;
};

export const linearGemvSubgroupKey = (
  storage: WeightStorage,
  group: number | undefined,
  lanes: LinearGemvParallelLanes,
): string => `${linearGemvParallelKey(storage, group, lanes)}:subgroup32`;

/**
 * Runs the same input distribution and addition tree as parallel with value exchange inside a
 * subgroup. The existing WGSL and keys are kept. It explicitly requires the fixed-32-lane feature and
 * uses no shared memory. The multiply-add is the same explicit `fma()` as parallel ({@link mac} — a
 * different spelling makes it diverge from parallel in u32 on Metal).
 * DECIDED: docs/decisions/0101-linear-gemv-subgroup.md
 */
export const linearGemvSubgroupWgsl = (
  storage: WeightStorage,
  group: number | undefined,
  lanes: LinearGemvParallelLanes,
): string => {
  if (![2, 4, 8, 16, 32].includes(lanes)) {
    throw new CodegenError("linear_gemv_parallel: invalid lane count");
  }
  assertRowsStorage(storage);
  const unit = linearGemvUnit(storage);
  const shift = gemvGroupShift(storage, group);
  return `enable subgroups, subgroup_size_control;
// karume linear gemv K subgroup32 (${storage}, ${lanes} lanes/output)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
${bindings(unit, storage)}

@compute @workgroup_size(128) @subgroup_size(32)
fn main(
  @builtin(subgroup_invocation_id) sub: u32,
  @builtin(subgroup_id) sg: u32,
  @builtin(workgroup_id) wg: vec3<u32>,
) {
  let lane = sub % ${lanes}u;
  let col = wg.x * ${128 / lanes}u + sg * ${32 / lanes}u + sub / ${lanes}u;
  var acc = 0.0;
  // All lanes of edge columns also take part in the shuffle. No layout of local ID vs subgroup ID
  // is assumed.
  if (col < dims.n) {
    let units = dims.k / ${unit}u;
    let row_base = col * units;${scaleSetupWgsl(storage, shift)}
    for (var unit = lane; unit < units; unit += ${lanes}u) {
${unitLoads(storage, "t", "unit", shift, "wg.y * (dims.k / 4u) + ")}
${unitMacs(storage, "t", false, true)}
    }
  }
  for (var width = ${lanes / 2}u; width > 0u; width /= 2u) {
    let other = subgroupShuffleXor(acc, width);
    if (lane < width) { acc = acc + other; }
  }
  if (col < dims.n && lane == 0u) {
    out[wg.y * dims.n + col] = acc + bias[col];
  }
}
`;
};
