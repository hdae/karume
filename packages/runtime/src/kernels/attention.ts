/**
 * The three kernels of fused attention (ADR 0023). 1 node = 3 dispatches:
 *
 * | Step | Key                                         | Role                                         |
 * | ---- | ------------------------------------------- | -------------------------------------------- |
 * | ①    | `attention_qk:v1:f32:reg64x64r8x4w16{v4}`   | **materialises** `S = (q·scale) @ (k·scale)ᵀ` |
 * | ②    | `attention_stats:v2:f32:lastdim:safe:wg256` | per-row `m = amax S`, `inv = 1/Σexp(S−m)`    |
 * | ③    | `attention_pv:v1:f32:reg64x64r8x4w16{v4}`   | `O = P @ v` (`P = exp(S−m)·inv` not stored)  |
 *
 * ① and ③ share fragments of the GEMM skeleton (src/kernels/gemm.ts), so the inner-product loop
 * keeps a single source of truth. Only ② has its body here.
 *
 * ## Bit identity is the core of the design (MUST)
 *
 * The output is **bit-identical** to the decomposed path (`mul → permute → bmm → softmax → expand →
 * bmm`). It rests on four points; breaking any one of them changes the rounding sequence:
 *
 * 1. ① reduces in K-ascending order with K tile 16 (the same skeleton as the current bmm).
 * 2. `scale` is applied to **both the q side and the k side while filling tiles** (the half-scale
 *    contract — src/ops.ts). Applying it once after the inner product moves the points where each
 *    product of `(q·s)·(k·s)` is rounded to f32.
 * 3. ② is **cut verbatim from passes ①② of the current softmax.ts** (the same 256-wide tree
 *    reduction, the same scan order `i += 256`, the same `inv = 1.0 / Σ`). From v2 the identity
 *    and the empty-row guard also follow **safe_softmax** (identity −inf, an empty row gives stats
 *    (0,0)) — on rows with at least one finite element the amax / inv bits equal the plain softmax
 *    form, so the decomposition parity of the mask-free path (plain softmax) still holds. Only
 *    all −inf rows widened from "contract violation, undefined value" to "valid input with output
 *    0" (ADR 0044 addendum 2026-08-31).
 * 4. ③'s A element is `exp(S − m) · inv` evaluated in f32, the same expression and operation order
 *    as the value the current softmax writes in pass ③. The reduction after that is the same as
 *    the current bmm.
 *
 * MUST: do not rewrite ② into a "one-pass online softmax". It is faster but changes the reduction
 * order, breaks point 3 above and loses the bit-identity check (PNG sha256 / measured atol=0
 * match). A variant with another order comes **with a separate key and new evidence** (ADR 0022
 * decision 3; since ADR 0110 decision 2 the evidence is the one its declared contract class
 * requires).
 *
 * ## The additive mask is an axis of ①'s epilogue only (the ADR 0023 addendum of 2026-08-11)
 *
 * The optional fourth input `mask[1,1,M,N]` (the contract in src/ops.ts) enters by **adding
 * `S' = fl(S + mask[m·N+n])` once, just before ①QK writes its output**. The rounding position and
 * count match the value the decomposed path builds with the two nodes `bmm` (materialises S) →
 * `add` (adds the mask), so it is bit-identical.
 * MUST: ②③ **never look at** the mask (S is already masked). The snapshots pin that neither the
 * output nor the key moves by a single byte.
 *
 * ## The storage form of S is an independent axis (option γ, phase 1)
 *
 * The **storage form** of the S handed between the three kernels is **a separate knob**
 * (src/kernels/score-storage.ts) from the compute form (`GemmCompute}` in {@link "./gemm.ts"}).
 * `s16` packs S into `array<u32>` with `pack2x16float`; since ① writes and ②③ read it, **all three
 * kernels switch together** (no per-step mix is possible). The only added rounding is the single
 * storage point, so the four points above stay valid as "the f32 path with S rounded to f16".
 */

import { IS_NAN_BITS_WGSL, NAN_MAX_WGSL } from "../codegen/numerics-wgsl.ts";
import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";
import { SAFE_SOFTMAX_NEG_INF_BITS } from "./softmax.ts";
import {
  assertGemmRowWindow,
  type GemmCompute,
  gemmComputeKeyPart,
  gemmKeyPart,
  gemmParams,
  type GemmRowWindowSpan,
  gemmWgsl,
} from "./gemm.ts";
import type { GemmGeometry } from "./gemm-geometry.ts";
import {
  assertScoreStorageSupported,
  scoreArrayType,
  scoreKeyPart,
  scoreNote,
  scoreReadAt,
  scoreScalarLoaderWgsl,
  type ScoreStorage,
} from "./score-storage.ts";

/**
 * Workgroup width of ②. **256, as in softmax** (the number of tree-reduction levels is the reduction
 * order itself).
 */
export const ATTENTION_STATS_WORKGROUP_SIZE = 256;

/** Words per row of the row statistics (`[0]` = row maximum / `[1]` = `1/Σexp(S−m)`). */
export const ATTENTION_STATS_STRIDE = 2;

/**
 * **Additive mask** discriminator of ①QK (the ADR 0023 addendum of 2026-08-11).
 *
 * MUST: absent is the empty string (existing keys do not move by a single character). MUST: the
 * word goes **after** s16 — the three words compute `:c16` / storage `:s16` / mask can be set at
 * once, so unless the order is fixed here in one place the same configuration gets two keys.
 */
const maskKeyPart = (mask: boolean): string => (mask ? ":mask" : "");

/**
 * Discriminator of **GQA** (divisible broadcast — ADR 0067 decisions 1 / 2).
 *
 * SHOULD: absent is the empty string (existing keys with `r = 1` do not move by a single character
 * — the "r=1 is byte-identical" of ADR 0067 decision 2 applies to the key side too) (re-graded
 * 2026-10-18 — ADR 0067 decision 2). MUST: the word goes **after** mask (the order is fixed here in
 * one place — the rule that keeps one configuration from having two keys).
 * MUST: the value of `r` itself is not in the key (it travels in the uniform — putting it in would
 * add a pipeline per distinct r; the same rule as scale / row_offset).
 */
const gqaKeyPart = (gqa: boolean): string => (gqa ? ":gqa" : "");

/**
 * Discriminator of the **row window** (block execution of query rows — `GemmRowWindow` in
 * {@link "./gemm.ts"}).
 *
 * MUST: absent is the empty string (with n = 1 the key does not differ by a single character from
 * the existing one — row blocking's "a shape where the split does not trigger is exactly the
 * existing one" applies to the key side too). MUST: the word goes **after** GQA (the order is fixed
 * here in one place). MUST: the side (`a` / `c`) follows from the op, so the spelling is fixed —
 * the same vocabulary as bmm's `:rwa` / `:rwc`, so the diagnostic key list reads side by side.
 * MUST: the row offset and the full M are not in the key (they travel in the uniform — putting
 * them in would add a pipeline per block).
 */
const rowWindowKeyPart = (side: "a" | "c", rowWindow: boolean): string =>
  rowWindow ? `:rw${side}` : "";

/**
 * Key of ①QK. `geometry` is the explicit geometry (`GemmSpec` in src/kernels/gemm.ts — the
 * profile's value in a Session, a sweep candidate in measurement), and MUST: pass **the same value**
 * to {@link attentionQkWgsl} (omitted = the default geometry = the existing key).
 */
export const attentionQkKey = (
  v4: boolean,
  compute: GemmCompute = "f32",
  score: ScoreStorage = "f32",
  mask = false,
  gqa = false,
  rowWindow = false,
  geometry?: GemmGeometry,
): string =>
  `attention_qk:v1:f32:${gemmKeyPart(v4, undefined, geometry)}${gemmComputeKeyPart(compute)}${scoreKeyPart(
    score,
  )}${maskKeyPart(mask)}${gqaKeyPart(gqa)}${rowWindowKeyPart("a", rowWindow)}`;

/** Key of ③PV (the `geometry` rule is the same as {@link attentionQkKey}). */
export const attentionPvKey = (
  v4: boolean,
  compute: GemmCompute = "f32",
  score: ScoreStorage = "f32",
  gqa = false,
  rowWindow = false,
  geometry?: GemmGeometry,
): string =>
  `attention_pv:v1:f32:${gemmKeyPart(v4, undefined, geometry)}${gemmComputeKeyPart(compute)}${scoreKeyPart(
    score,
  )}${gqaKeyPart(gqa)}${rowWindowKeyPart("c", rowWindow)}`;

/**
 * Upper bound on the elements per thread that ②'s **regcache variant** (reads S once and keeps it
 * in registers) can take. Self-attention (dim = 4096) is 16 and cross-attention (dim = 512) is 2;
 * measured ×1.672 / ×1.115 (docs/research/2026-08-10-kernel-variant-sweep.md §3.1).
 *
 * The bound exists because `epc` is directly the register count — leaving it unbounded for shapes
 * with a large dim (long sequences such as 32k) spills and defeats keeping values in registers.
 * Shapes above it **fall back to the current two-read loop** (both give bit-identical values, so
 * this is a choice of speed variant, not an approximation).
 */
export const ATTENTION_STATS_REG_CACHE_MAX = 32;

/**
 * **The only pure function** that derives the regcache variant's elements per thread from dim
 * (`undefined` = stay with the two-read loop). The recipe derivation layer (`buildAttention` in
 * recipe-builders/attention.ts) passes it to both the key and the WGSL, so a state where "the
 * generated `epc` disagrees with the actual dim" cannot arise by construction.
 */
export const attentionStatsRegCache = (dim: number): number | undefined => {
  const perThread = Math.ceil(dim / ATTENTION_STATS_WORKGROUP_SIZE);
  return perThread <= ATTENTION_STATS_REG_CACHE_MAX ? perThread : undefined;
};

// v2: ① uses nan_max + identity −inf (via params), and an all −inf row writes stats (0,0)
// (a port of safe_softmax's three guards — the bits of finite rows do not change)
export const attentionStatsKey = (
  compute: GemmCompute = "f32",
  score: ScoreStorage = "f32",
  regCache?: number,
): string =>
  `attention_stats:v2:f32:lastdim:safe:wg${ATTENTION_STATS_WORKGROUP_SIZE}${gemmComputeKeyPart(
    compute,
  )}${scoreKeyPart(score)}${regCache === undefined ? "" : `:rc${regCache}`}`;

/** Key of the f32 variant (the existing call surface — source of truth: {@link attentionStatsKey}). */
export const ATTENTION_STATS_KEY = attentionStatsKey();

export const attentionQkWgsl = (
  v4: boolean,
  compute: GemmCompute = "f32",
  score: ScoreStorage = "f32",
  mask = false,
  gqa = false,
  rowWindow = false,
  geometry?: GemmGeometry,
): string => gemmWgsl({ op: "attention_qk", v4, compute, score, mask, gqa, rowWindow, geometry });

export const attentionPvWgsl = (
  v4: boolean,
  compute: GemmCompute = "f32",
  score: ScoreStorage = "f32",
  gqa = false,
  rowWindow = false,
  geometry?: GemmGeometry,
): string => gemmWgsl({ op: "attention_pv", v4, compute, score, gqa, rowWindow, geometry });

/**
 * ② row statistics. **Cut as is from passes ①② of the current softmax.ts**; only the output
 * changed to "2 words per row" (the second evaluation of `exp` and the write-back of the whole row
 * are gone).
 *
 * MUST: match softmax in scan order (`i = lid; i += 256`) and tree-reduction levels
 * (`stride = 128, 64, …`). The last ulp of a row sum moves with the summation order, so a
 * difference here breaks bit identity with the decomposed path.
 * MUST: row max is `nan_max`, the identity is −inf via params, and an empty row gives stats (0,0)
 * (the same form as safe_softmax's three guards — point 3 of the module doc. For an empty row ③PV
 * multiplies `exp(S − 0) = exp(−inf) = 0` by `inv = 0`, so the output is exactly 0).
 */
export const attentionStatsWgsl = (
  compute: GemmCompute = "f32",
  score: ScoreStorage = "f32",
  regCache?: number,
): string => {
  const f16 = compute === "f16";
  // ② **only reads** S, so the v4 constraint (the write side's RMW) does not apply — the only
  // check is "not set together with `:c16`".
  assertScoreStorageSupported("attention_stats", score, true, f16);
  // MUST: widen S to f32 before reducing (widening is exact, so the same S yields **bit-identical**
  // m / inv to the f32 variant). Taking max / exp in f16 changes the rounding sequence of the
  // reduction and breaks ADR 0023's "② matches softmax passes ①② verbatim".
  const read = (index: string): string =>
    f16 ? `f32(s[${index}])` : scoreReadAt("s", score, index);
  // The regcache variant is **exactly the same** as the two-read version in both "element →
  // thread assignment" and "reduction order"; the only difference is reading S twice versus
  // reading it once and keeping it in registers (out-of-range elements are touched in neither
  // pass). So m / inv do not move by a single bit — point 3 of ADR 0023's bit identity holds.
  const slots = regCache === undefined ? [] : Array.from({ length: regCache }, (_, at) => at);
  const indexPrologue =
    regCache === undefined
      ? ""
      : `${slots
          .map(
            (at) =>
              `  let i${at} = lid${at === 0 ? "" : ` + ${at * ATTENTION_STATS_WORKGROUP_SIZE}u`};`,
          )
          .join("\n")}\n`;
  const maxPass =
    regCache === undefined
      ? `    var i = lid;
    while (i < dim) {
      hi = nan_max(hi, ${read("base + i")});
      i = i + ${ATTENTION_STATS_WORKGROUP_SIZE}u;
    }`
      : slots
          .map(
            (at) =>
              `    var c${at} = 0.0;
    if (i${at} < dim) {
      c${at} = ${read(`base + i${at}`)};
      hi = nan_max(hi, c${at});
    }`,
          )
          .join("\n");
  const sumPass =
    regCache === undefined
      ? `    var j = lid;
    while (j < dim) {
      acc = acc + exp(${read("base + j")} - amax);
      j = j + ${ATTENTION_STATS_WORKGROUP_SIZE}u;
    }`
      : slots
          .map(
            (at) =>
              `    if (i${at} < dim) {
      acc = acc + exp(c${at} - amax);
    }`,
          )
          .join("\n");
  return `// karume attention_stats (per-row m = amax(S) and inv = 1/Σexp(S - m), f32${
    f16 ? ", S in f16 storage" : ""
  }${scoreNote(score)}${
    regCache === undefined
      ? ""
      : `, S read once and held in registers (${regCache} elements per thread)`
  })${f16 ? "\nenable f16;" : ""}
struct Params {
  rows: u32,
  dim: u32,
  neg_inf: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> s: array<${scoreArrayType(score, f16 ? "f16" : "f32")}>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;
${scoreScalarLoaderWgsl("s", score)}
${IS_NAN_BITS_WGSL}

${NAN_MAX_WGSL}

var<workgroup> scratch: array<f32, ${ATTENTION_STATS_WORKGROUP_SIZE}>;

@compute @workgroup_size(${ATTENTION_STATS_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let dim = params.dim;
  let neg_inf = bitcast<f32>(params.neg_inf);
${indexPrologue}  var row = wid.x;
  while (row < params.rows) {
    let base = row * dim;

    // ① Row maximum (the subtracted term of safe-softmax — identity is -inf)
    var hi = neg_inf;
${maxPass}
    scratch[lid] = hi;
    workgroupBarrier();
    var stride = ${ATTENTION_STATS_WORKGROUP_SIZE / 2}u;
    while (stride > 0u) {
      if (lid < stride) {
        scratch[lid] = nan_max(scratch[lid], scratch[lid + stride]);
      }
      workgroupBarrier();
      stride = stride / 2u;
    }
    let row_max = scratch[0u];
    // A row with every element -inf (no key visible) — make the subtracted term 0 so no NaN is
    // produced (same form as safe_softmax)
    let empty = row_max == neg_inf;
    let amax = select(row_max, 0.0, empty);
    // Make sure scratch has been read before ② overwrites it
    workgroupBarrier();

    // ② Σ exp(S - amax) (the largest element gives exp(0) = 1, so the denominator is at least 1)
    var acc = 0.0;
${sumPass}
    scratch[lid] = acc;
    workgroupBarrier();
    var stride2 = ${ATTENTION_STATS_WORKGROUP_SIZE / 2}u;
    while (stride2 > 0u) {
      if (lid < stride2) {
        scratch[lid] = scratch[lid] + scratch[lid + stride2];
      }
      workgroupBarrier();
      stride2 = stride2 / 2u;
    }
    // MUST: the reciprocal is built here (going back to a division in ③ would change the operation
    // versus softmax pass ③)
    let inv = 1.0 / scratch[0u];
    if (lid == 0u) {
      // An empty row is (0.0, 0.0) — ③ multiplies exp(S - 0) = exp(-inf) = 0 by inv = 0, so the
      // output is exactly 0
      stats[row * ${ATTENTION_STATS_STRIDE}u] = amax;
      stats[row * ${ATTENTION_STATS_STRIDE}u + 1u] = select(inv, 0.0, empty);
    }
    // Make sure scratch[0] has been read before the next row overwrites scratch[lid]
    workgroupBarrier();
    row = row + nwg.x;
  }
}
`;
};

/** WGSL of the f32 variant (the existing call surface — source of truth: {@link attentionStatsWgsl}). */
export const ATTENTION_STATS_WGSL: string = attentionStatsWgsl();

/**
 * Range check of the GQA repeat count `r = H / Hkv` (a derived value carried in the uniform — ADR
 * 0067 decision 2).
 *
 * MUST: do not let 0 through. WGSL u32 division by zero does not trap and returns an
 * implementation-dependent value, so `r = 0` corrupts the K / V base with neither an exception nor
 * NaN (a silent wrong value).
 */
const assertKvRepeat = (where: string, kvRepeat: number): void => {
  assertU32Params(where, { kv_repeat: kvRepeat });
  if (kvRepeat < 1) {
    throw new CodegenError(`${where}: kv_repeat must be a positive integer (${kvRepeat})`);
  }
};

/**
 * Uniform of ① (`{m, n, k, scale}`). The 4th word is the f32 bits of the **half scale**.
 *
 * MUST: do not bake scale into the WGSL (a pipeline per distinct value — the same reason as the
 * fill value of masked_fill).
 *
 * `kvRepeat` is the 5th word passed **only by the GQA variant** (ADR 0067 decision 2). Rounded up
 * to 16-byte units it grows to 32 bytes (the same form as bmm's row-window variant — allocating in
 * 16-byte units is a project convention, not a requirement of the spec). SHOULD: when omitted the
 * bytes stay exactly 16 as before — the purpose of this branch is that the existing `r = 1` path
 * does not move by a single byte (re-graded 2026-10-18 — ADR 0067 decision 2).
 *
 * `window` is the 2 trailing words passed **only by the row-window variant** (paired with
 * `ROW_WINDOW_DIMS_EXTRA` in {@link "./gemm.ts"}).
 * MUST: its position is **after** `kv_repeat` (the GQA word position does not move with the row
 * window — the generator's dimsExtra spells them in the same order). MUST: `m` is the block row
 * count (not the full M).
 */
export const attentionQkParams = (
  m: number,
  n: number,
  k: number,
  scale: number,
  kvRepeat?: number,
  window?: GemmRowWindowSpan,
): Uint32Array<ArrayBuffer> => {
  // MUST: check finiteness **as f32** (the destination is an f32 word — `1e39`, finite in f64,
  // becomes `+Inf`, and `0 * Inf = NaN` silently breaks the scores. The same check as the
  // contract layer's `assertFiniteAttr`).
  if (!Number.isFinite(Math.fround(scale))) {
    throw new CodegenError(`attention_qk params: scale must be a finite number as f32 (${scale})`);
  }
  if (kvRepeat !== undefined) assertKvRepeat("attention_qk params", kvRepeat);
  if (window !== undefined) assertGemmRowWindow("attention_qk row window params", m, window);
  const base = gemmParams("attention_qk", m, n, k);
  if (kvRepeat === undefined && window === undefined) {
    new Float32Array(base.buffer)[3] = scale;
    return base;
  }
  const params = new Uint32Array(8);
  params.set(base.subarray(0, 3));
  new Float32Array(params.buffer)[3] = scale;
  if (kvRepeat !== undefined) params[4] = kvRepeat;
  if (window !== undefined) {
    const at = kvRepeat === undefined ? 4 : 5;
    params[at] = window.offset;
    params[at + 1] = window.rowsFull;
  }
  return params;
};

/**
 * Uniform of ③ (`{m, n, k}` — the same 3 words as bmm).
 *
 * `kvRepeat` is the 4th word passed **only by the GQA variant** (ADR 0067 decision 2). It fits in
 * the slack of the 16-byte allocation, so GQA does not change the byte count (when omitted the 4th
 * word is 0 as before).
 *
 * `window` is the 2 trailing words passed **only by the row-window variant** (the same rule as ①QK
 * — after `kv_repeat`). Only here does the struct exceed 16 bytes, so 32 bytes are allocated.
 * MUST: `m` is the block row count.
 */
export const attentionPvParams = (
  m: number,
  n: number,
  k: number,
  kvRepeat?: number,
  window?: GemmRowWindowSpan,
): Uint32Array<ArrayBuffer> => {
  if (kvRepeat !== undefined) assertKvRepeat("attention_pv params", kvRepeat);
  if (window !== undefined) assertGemmRowWindow("attention_pv row window params", m, window);
  const params = gemmParams("attention_pv", m, n, k);
  if (window === undefined) {
    if (kvRepeat !== undefined) params[3] = kvRepeat;
    return params;
  }
  const wide = new Uint32Array(8);
  wide.set(params.subarray(0, 3));
  if (kvRepeat !== undefined) wide[3] = kvRepeat;
  const at = kvRepeat === undefined ? 3 : 4;
  wide[at] = window.offset;
  wide[at + 1] = window.rowsFull;
  return wide;
};

/**
 * Uniform of ②. 16 bytes are allocated even though the content is 2 words (as in softmax —
 * allocating in 16-byte units is a project convention (the fusion pass fixes the same form in ADR
 * 0040), not a requirement of the spec).
 *
 * MUST: when the regcache variant is used, pass its `regCache` here too so the geometry is checked
 * (redundant with the generator, but a check that the direct kernel call path also goes through —
 * the same shape as gru_scan's hidden limit). `epc` is baked at generation while `dim` is a
 * run-time value, so with `dim > epc · 256` the leftover elements enter neither max nor Σ and the
 * probabilities silently shift (a silent wrong value with neither an exception nor NaN).
 */
export const attentionStatsParams = (
  rows: number,
  dim: number,
  regCache?: number,
): Uint32Array<ArrayBuffer> => {
  assertU32Params("attention_stats params", { rows, dim });
  if (dim < 1) {
    throw new CodegenError(`attention_stats params: dim must be a positive integer (${dim})`);
  }
  if (regCache !== undefined) {
    // MUST: check that it is an integer first. The generator's `Array.from({ length: regCache })`
    // **truncates** a non-integer length when laying out the slots, while the coverage computation
    // below claims `regCache · 256` with the non-integer as is, so `1.5` passes claiming "384
    // elements covered although only 1 slot = 256 elements is unrolled", and the leftover elements
    // enter neither max nor Σ — a silent wrong value.
    if (!Number.isInteger(regCache)) {
      throw new CodegenError(`attention_stats params: regCache ${regCache} is not an integer`);
    }
    if (regCache > ATTENTION_STATS_REG_CACHE_MAX) {
      throw new CodegenError(
        `attention_stats params: regCache ${regCache} exceeds the limit ${ATTENTION_STATS_REG_CACHE_MAX}`,
      );
    }
    const covered = regCache * ATTENTION_STATS_WORKGROUP_SIZE;
    if (dim > covered) {
      throw new CodegenError(
        `attention_stats params: dim ${dim} exceeds the range ${covered} covered by the regcache variant (${regCache} × ${ATTENTION_STATS_WORKGROUP_SIZE})`,
      );
    }
  }
  const params = new Uint32Array(4);
  params[0] = rows;
  params[1] = dim;
  // Bits of −inf (identity and the empty-row test — via params to avoid the shader-creation error
  // of a constant-expression bitcast; the same as the MUST in softmax.ts)
  params[2] = SAFE_SOFTMAX_NEG_INF_BITS;
  return params;
};
