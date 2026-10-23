/**
 * The **i8a8 variants** of fused attention ①QK / ③PV (an **integer dot product** of i8
 * activations) — design in docs/research/2026-08-04-attention-a8-design.md §2.1 / §2.2 / §2.3 /
 * §4.1 / §4.2, opt-in through `SessionOptions.attentionCompute: "a8"` (default `"f32"`).
 *
 * ```
 * ①QK: acc[i32] = Σ_d qq[b,m,d] · kq[b,n,d]          (exact integers, 0 roundings)
 *       S        = f32(acc) · (qs'[b,m] · ks'[b,n])   (qs' = qs·scale / ks' = ks·scale)
 * ③PV: acc[i32] = Σ_n qP[b,m,n] · vq[b,d,n]          (qP: built in the A tile fill, not stored)
 *       O        = f32(acc) · (prow[b,m] · vs[b,d])   (prow = inv·(1/127))
 * ```
 *
 * ## Relation to linear's i8a8 (src/kernels/linear-i8a8.ts)
 *
 * In `S[b,m,n] = Σ_d q[b,m,d]·k[b,n,d]` **the reduction axis D is innermost and contiguous in both
 * q and k**, so the index shape matches linear's `x[m,k]` (k contiguous) × `w[n,k]` (k contiguous)
 * exactly. Activation quantization therefore just **calls {@link "./quantize-rows.ts"} twice,
 * unmodified**, and the inner-product loop, the `[pack][row]` layout of the shared tiles and the
 * zero fill of the K remainder take the same shape as linear's i8a8.
 *
 * The generator is not shared with linear because the differences are scattered **where fragment
 * splicing cannot absorb them** (5 batch bases / dequant with scales on both sides + the half scale
 * / no bias). What is shared is **the tile geometry and its arithmetic**
 * ({@link "./i8a8-geometry.ts"}) and **the body of the integer dot product**
 * ({@link i8a8IdotWgsl}), so the parts that affect the numeric contract and the index arithmetic
 * keep a single source of truth.
 *
 * MUST: ①QK and ③PV take **different geometries** as their defaults (for ③, `M64N128`, which fits
 * N = D = 128 into one tile, wins — docs/research/2026-08-10-kernel-variant-sweep.md §3.2). Tying the
 * two steps together on the assumption of "one best geometry" always leaves one of them behind.
 *
 * ## Numeric contract (GPU vs TS reference at atol=0 — src/reference/i8a8.ts)
 *
 * The integer reduction is an order-independent exact value, so the floating-point operations per
 * output element are only six: `qs = amax_q·(1/127)` / `ks = amax_k·(1/127)` / `qs·scale` /
 * `ks·scale` / `qs'·ks'` / `f32(acc)·(…)`. Each is deterministic with round-to-nearest-even (the
 * only weak point is the quantization's `x/s` = a data-dependent division, where elements near a
 * rounding boundary may move by ±1 step within WGSL's 2.5 ULP allowance — the tests measure the
 * margin with `quantizeRowsTieMargin` and make it a check).
 *
 * MUST: fix the multiplication order of dequant — **fold `qs' · ks'` into one f32 first**, then
 * multiply `f32(acc)` by it. The sequential form `f32(acc)·qs'·ks'` moves the rounding points and
 * breaks bit identity with the reference (the same MUST as linear's `xs·wscale`). **There is no
 * bias, so `fma` is not used**.
 * MUST: the half scale (ADR 0023's `√scale_factor`) is applied **on the dequant side**, to both q
 * and k. Applying it **before** quantization (quantizing `q·scale`) moves amax by the same factor,
 * so the integers are almost the same, but it only adds one f32 rounding — a pure loss — and
 * `quantize_rows` could no longer be used unmodified (the ruling of design §2.1). The attrs
 * contract (`scale` stays √scale) does not change.
 * MUST: the row scale is `qs[batch·M + arow]` and the column scale `ks[batch·N + wcol]`. Mixing
 * them up is a wrong value without an exception, and the only detector is a test with **B·H ≥ 2 and
 * per-head statistics that differ** × **2 or more m / n tiles** (the same trap as ADR 0024 decision
 * 6).
 *
 * ## Structure of ③PV (design §2.2 / §2.3 — three points where it decisively differs from ①QK)
 *
 * 1. **The A side (P̃) does not go through the quantization kernel**. `P̃ = exp(S−m)` has an in-row
 *    max of 1.0 by construction (`exp(0)` is exact in f32), so per-token amax quantization would
 *    give a scale that **collapses to 1/127**. Taking amax is therefore pointless, and the A tile
 *    fill builds `round(127·exp(S−m))` on the spot and packs 4 per word (**no division**, and the
 *    upper bound 127 is structural, so no clamp is needed either).
 *    MUST: do not write it in a form that can take amax — it would be misread as "adapting per
 *    row" and drift silently toward the optimistic bound that the runtime cannot implement (the
 *    granularity table of design §5.1).
 * 2. **V's scale must be per-column (amax over all of N)**. The reduction axis is n, so the
 *    condition for the `f32(acc)·s` form to hold is exactly **s not depending on n**. Per-token
 *    (per row n of V) makes the scale vary along the reduction axis so it cannot be factored out of
 *    the sum — used silently, it is a wrong value without an exception (the same error as reusing
 *    `wcol` for rows in linear — ADR 0024 decision 6).
 *    The recipe derivation layer (`buildAttention` in recipe-builders/attention.ts) builds
 *    Vᵀ`[B·H, D, N]` with the existing `strided` (permute) and then calls
 *    `quantize_rows` (rows = `(b,h,d)`) — **per-column quantization and the N-contiguous packing
 *    that dp4a needs come at once**, so no new kernel is needed.
 * 3. **The row's `1/l` moves to the dequant side**. `prow = inv·(1/127)` is invariant over the row
 *    loop, so `O = f32(acc)·(prow·vs[d])` takes the same shape as linear's `xs·wscale`. ② row
 *    statistics **stay f32, unchanged to the byte** (using the post-quantization sum as the
 *    denominator was measured clearly worse in phase Q0 and rejected for good — a refutation of
 *    design §4.3).
 *
 * ## Numeric contract of ③PV (unlike ②, atol=0 does not hold for O — design §4.2)
 *
 * The `exp` in `qP = round(127·exp(S−m))` is not guaranteed to be bit-identical between WGSL and
 * JS, and `127·exp` is densely distributed over [0,127], so some elements always land on a rounding
 * boundary. The contract is therefore **split in two**:
 *
 * - **The pure function after the integers are received** (`referenceAttentionPvI8a8Core`) matches
 *   at **atol=0**. It has only three operations: `f32(acc)` / `prow·vs` / the product.
 * - Generating `qP` is a separate check: a mismatch with the TS reference is **always ±1 step** (a
 *   single ±2 means an implementation bug), and the mismatch rate itself is recorded
 *   (tests/gpu_attention_pv_i8a8_test.ts).
 *
 * MUST: fix the multiplication order of dequant — **fold `prow · vs[d]` into one f32 first**, then
 * multiply `f32(acc)` by it (the same shape as ①QK). MUST: build `1/127` by **multiplication**
 * (division is allowed up to 2.5 ULP — the measurement in quantize-rows.ts).
 * NOTE (outside the contract): `qP` of a row with non-finite values in S is
 * **implementation-dependent** (the value of `vec4<i32>(NaN)` is undefined in WGSL). The f32
 * variant propagates NaN into O as is, so **only there does the behaviour differ**. It is treated
 * like quantize-rows.ts's "quantized values of non-finite rows are outside the contract" and is not
 * compared.
 *
 * ## Phases (staged introduction)
 *
 * This file covers ①QK and ③PV; ② row statistics run in f32 as they are (phases 1 + 2 of design
 * §7). Eligibility is **independent per step** (① needs `D % 4 == 0`, ③ needs `N % 4 == 0`), so
 * **a mix with only one of them in i8a8** can occur (`buildAttention` in
 * recipe-builders/attention.ts).
 */

import { CodegenError } from "../codegen/errors.ts";
import {
  assertGemmRowWindow,
  gemmParams,
  type GemmRowWindowSpan,
  ROW_WINDOW_DIMS_EXTRA,
} from "./gemm.ts";
import {
  assertI8a8Geometry,
  defaultI8a8Geometry,
  I8A8_PACK,
  i8a8AccumulatorInit,
  i8a8ASlots,
  i8a8BSlots,
  i8a8FillStride,
  type I8a8Geometry,
  i8a8GeometryKeyPart,
  i8a8GeometryNote,
  i8a8InnerProductLoop,
  i8a8KPacks,
  i8a8Threads,
  i8a8TileM,
  i8a8TileN,
} from "./i8a8-geometry.ts";
import { i8a8IdotWgsl, LINEAR_I8A8_MAX_K } from "./linear-i8a8.ts";
import {
  assertScoreStorageSupported,
  scoreArrayType,
  scoreKeyPart,
  scoreNote,
  scoreQuadLoaderWgsl,
  scoreReadQuad,
  type MaterialisedScoreStorage,
  scoreStoreQuad,
  scoreStoreWgsl,
} from "./score-storage.ts";

/** Binding of q's per-token scale (`[B·H·M]`) (the number after the output S). */
export const ATTENTION_QK_Q_SCALE_BINDING = 4;
/** Binding of k's per-token scale (`[B·H·N]`). It amounts to **a per-output-column scale**. */
export const ATTENTION_QK_K_SCALE_BINDING = 5;
/**
 * Binding of Vᵀ's per-row scale (`[B·H·D]`) (the number after ③PV's output O). A "row" of Vᵀ =
 * `(b,h,d)`, so this is exactly **V's per-column scale (amax over all of N)** (point 2 above).
 */
export const ATTENTION_PV_V_SCALE_BINDING = 5;

/**
 * Whether ①QK can be dispatched with the i8a8 variant. q / k pack along the reduction axis D, so
 * the condition is that `D` divides into words of 4 elements. A step that is not eligible falls
 * back to the f32 path (ADR 0030 decision 5).
 * This defines the step's eligibility; the source of truth for the path decision is
 * `attentionRoute` (src/runtime/attention-route.ts). Both the recipe derivation (`buildAttention`)
 * and the tests of the accepted set go through it (writing the decision twice lets one copy move
 * alone and disagree).
 */
export const attentionQkI8a8Eligible = (depth: number): boolean => depth % I8A8_PACK === 0;

/**
 * Whether ③PV can be dispatched with the i8a8 variant. P̃ / Vᵀ pack along the reduction axis N (the
 * key length), so the condition is that `N` divides into words of 4 elements (the axis differs from
 * ①, so shapes eligible for only one of them do exist).
 */
export const attentionPvI8a8Eligible = (cols: number): boolean => cols % I8A8_PACK === 0;

/**
 * Edge of P̃'s quantization lattice. **Do not import `QUANTIZE_ROWS_ABS_MAX` from
 * quantize-rows.ts** — that one is "the lattice of an adaptive scale built from the row amax",
 * while this one is **a different fact**, "the in-row max is 1.0 by construction, so the scale
 * collapses to 1/127" (design §2.2). That the values match is close to a coincidence, and tying
 * them together would silently propagate a change of one into the other.
 */
const P_LATTICE_MAX = 127;
/** MUST: build `1/127` by multiplication (WGSL f32 division is allowed up to 2.5 ULP). */
const INV_P_LATTICE = 1 / P_LATTICE_MAX;

/** Additive index offset (0 is omitted — keeps the output readable). */
const at = (offset: number): string => (offset === 0 ? "" : ` + ${offset}u`);

/** Output row numbers (unrolled from `orow0`). */
const rowDecls = (geometry: I8a8Geometry): string =>
  Array.from({ length: geometry.regM }, (_, row) =>
    row === 0
      ? `  let orow0 = wid.y * ${i8a8TileM(geometry)}u + lid.y * ${geometry.regM}u;`
      : `  let orow${row} = orow0 + ${row}u;`,
  ).join("\n");

/** Output column quads (v4 path — unrolled from `ocq0`). */
const quadDecls = (geometry: I8a8Geometry): string => {
  const quads = geometry.regN / I8A8_PACK;
  return Array.from({ length: quads }, (_, quad) =>
    quad === 0
      ? `  let ocq0 = wid.x * ${i8a8TileN(geometry) / I8A8_PACK}u + lid.x * ${quads}u;`
      : `  let ocq${quad} = ocq0 + ${quad}u;`,
  ).join("\n");
};

/**
 * Assignment of the A (q / P̃) tile — row numbers are invariant over the K tile loop, so the
 * prologue builds them once.
 */
const prologueA = (geometry: I8a8Geometry, base: string, label: string): string => {
  const tileM = i8a8TileM(geometry);
  const stride = i8a8FillStride(geometry);
  const slots = Array.from(
    { length: i8a8ASlots(geometry) },
    (_, slot) =>
      `  let arow${slot} = ${slot === 0 ? `wid.y * ${tileM}u + ar` : `arow0 + ${slot * stride}u`};
  let arow_base${slot} = ${base} + arow${slot} * k4;`,
  ).join("\n");
  return `  // ${label} (${tileM} rows × ${i8a8KPacks(geometry)} packs over ${i8a8Threads(
    geometry,
  )} threads in ${i8a8ASlots(geometry)} rounds)
  let ar = tid / ${i8a8KPacks(geometry)}u;
  let ap = tid % ${i8a8KPacks(geometry)}u;
  let sa_at = ap * ${tileM}u + ar;
${slots}`;
};

/**
 * Assignment of the B (k / Vᵀ) tile — in both, the pack direction is innermost and contiguous, so
 * no transpose is needed.
 */
const prologueB = (
  geometry: I8a8Geometry,
  base: string,
  baseName: string,
  note: string,
): string => {
  const tileN = i8a8TileN(geometry);
  const stride = i8a8FillStride(geometry);
  const slots = Array.from(
    { length: i8a8BSlots(geometry) },
    (_, slot) =>
      `  let wcol${slot} = ${slot === 0 ? `wid.x * ${tileN}u + wc` : `wcol0 + ${slot * stride}u`};
  let ${baseName}${slot} = ${base} + wcol${slot} * k4;`,
  ).join("\n");
  return `  // B tile assignment (${tileN} columns × ${i8a8KPacks(geometry)} packs over ${i8a8Threads(
    geometry,
  )} threads in ${i8a8BSlots(geometry)} rounds).
  // ${note}
  let wc = tid / ${i8a8KPacks(geometry)}u;
  let wp = tid % ${i8a8KPacks(geometry)}u;
  let sb_at = wp * ${tileN}u + wc;
${slots}`;
};

/** A tile fill of ①QK (reads q as is — the same shape as linear's x). */
const fillA = (geometry: I8a8Geometry, name: string): string => {
  const stride = i8a8FillStride(geometry);
  const slots = Array.from(
    { length: i8a8ASlots(geometry) },
    (_, slot) =>
      `    var av${slot} = 0u;
    if (arow${slot} < dims.m && apack < k4) {
      av${slot} = ${name}[arow_base${slot} + apack];
    }
    sa[sa_at${at(slot * stride)}] = av${slot};`,
  ).join("\n");
  return `    let apack = t * ${i8a8KPacks(geometry)}u + ap;
${slots}`;
};

/**
 * B tile fill (out of range is zero-filled — `idot(0, x) == 0`, so it is exact even with a K
 * remainder).
 *
 * MUST: the slot value variables are fixed as `bv<number>`, not built from the caller's prefix
 * (`value`). Built from ①QK's prefix `k`, the 5th slot becomes `var k4` and shadows the
 * function-scope `let k4` (the K pack count) — shadowing is legal in WGSL, so compilation and
 * validation pass, the range test `kpack < k4` of the following slots is always false, and columns
 * of S silently become 0 (measured on the 16 geometries with 5 or more fill slots).
 * The detectors are tests/codegen_i8a8_shadowing_test.ts (overlapping declaration names) and
 * tests/gpu_attention_pv_i8a8_test.ts (atol=0 against the reference on geometries with 8 / 16
 * slots).
 */
const fillB = (geometry: I8a8Geometry, name: string, baseName: string, value: string): string => {
  const stride = i8a8FillStride(geometry);
  const slots = Array.from(
    { length: i8a8BSlots(geometry) },
    (_, slot) =>
      `    var bv${slot} = 0u;
    if (wcol${slot} < dims.n && ${value}pack < k4) {
      bv${slot} = ${name}[${baseName}${slot} + ${value}pack];
    }
    sb[sb_at${at(slot * stride)}] = bv${slot};`,
  ).join("\n");
  return `    let ${value}pack = t * ${i8a8KPacks(geometry)}u + wp;
${slots}`;
};

/**
 * Pipeline key. It strictly follows the precedent of `linear:v4:i8a8:…` (**bump the generation,
 * set the dtype field to `i8a8`, include the whole geometry discriminator, and append the
 * dot-product variant**). The geometry became a parameter, so a key with only the tile edges does
 * not determine the output ({@link i8a8GeometryKeyPart}).
 * MUST: the existing f32 / `:c16` keys do not move by a single character (the snapshots are the
 * detector).
 */
export const attentionQkI8a8Key = (
  v4: boolean,
  dp4a: boolean,
  score: MaterialisedScoreStorage = "f32",
  geometry: I8a8Geometry = defaultI8a8Geometry("attention_qk"),
  rowWindow = false,
): string =>
  `attention_qk:v3:i8a8:${i8a8GeometryKeyPart(geometry, v4)}:${dp4a ? "dp4a" : "dp4aEmu"}${scoreKeyPart(
    score,
  )}${rowWindow ? ":rwa" : ""}`;

/**
 * Whether S can be written as `vec4<f32>`. **The D-side condition is already covered by the
 * eligibility check (`D % 4 == 0`)**, so this looks only at N (the output columns) (the same rule as
 * linear's i8a8).
 */
export const attentionQkI8a8UsesVec4 = (n: number): boolean => n % I8A8_PACK === 0;

/**
 * Store of ①QK. Both rows and column quads are unrolled at codegen and `acc{row}_{column quad}` is
 * read statically (unrolled so that no dynamic `acc[i]` index remains — the reason is in
 * {@link i8a8AccumulatorInit}). The guard structure, the folding order of `qs'·ks'` and the
 * position of the half scale are the same for every geometry.
 */
const store = (geometry: I8a8Geometry, v4: boolean, score: MaterialisedScoreStorage): string => {
  const { regM, regN } = geometry;
  const quads = regN / I8A8_PACK;
  const rows = rowDecls(geometry);
  if (v4) {
    const blocks = Array.from({ length: quads }, (_, quad) => {
      const rowStores = Array.from(
        { length: regM },
        (_, row) =>
          `    if (orow${row} < dims.m) {
      ${scoreStoreQuad(
        "s",
        score,
        `sbase + orow${row} * n4 + ocq${quad}`,
        `vec4<f32>(acc${row}_${quad}) * ((qscale[qsbase + orow${row}] * dims.scale) * ks)`,
      )}
    }`,
      ).join("\n");
      return `  if (ocq${quad} < n4) {
    // n % ${I8A8_PACK} == 0 and ocq${quad} < n4, so oc + ${I8A8_PACK - 1} < n.
    // MUST: the half scale goes to **both** q and k on the dequant side (the column side once, here)
    let oc = ocq${quad} * ${I8A8_PACK}u;
    let ks = vec4<f32>(
      kscale[ksbase + oc],
      kscale[ksbase + oc + 1u],
      kscale[ksbase + oc + 2u],
      kscale[ksbase + oc + 3u],
    ) * dims.scale;
    // MUST: fold qs'·ks' into one f32 first, then multiply f32(acc) by it (no bias, so no fma)
${rowStores}
  }`;
    }).join("\n");
    return `${quadDecls(geometry)}
${rows}
${blocks}`;
  }
  const lanes = ["x", "y", "z", "w"] as const;
  const rowStores = Array.from({ length: regM }, (_, row) => {
    const writes = Array.from({ length: regN }, (_, col) => {
      const offset = at(col);
      const acc = `acc${row}_${Math.floor(col / I8A8_PACK)}.${lanes[col % I8A8_PACK]}`;
      return `    if (ocol${offset} < dims.n) {
      s[obase + ocol${offset}] = f32(${acc}) * (qs * (kscale[ksbase + ocol${offset}] * dims.scale));
    }`;
    }).join("\n");
    return `  if (orow${row} < dims.m) {
    let obase = sbase + orow${row} * dims.n;
    // MUST: fold qs'·ks' into one f32 first, then multiply f32(acc) by it (no bias, so no fma)
    let qs = qscale[qsbase + orow${row}] * dims.scale;
${writes}
  }`;
  }).join("\n");
  return `  let ocol = wid.x * ${i8a8TileN(geometry)}u + lid.x * ${regN}u;
${rows}
${rowStores}`;
};

export const attentionQkI8a8Wgsl = (
  v4: boolean,
  dp4a: boolean,
  score: MaterialisedScoreStorage = "f32",
  geometry: I8a8Geometry = defaultI8a8Geometry("attention_qk"),
  rowWindow = false,
): string => {
  assertScoreStorageSupported("attention_qk i8a8", score, v4);
  assertI8a8Geometry(geometry, "attention_qk i8a8");
  const kPacks = i8a8KPacks(geometry);
  // Row window (`"a"` of `GemmRowWindow` in {@link "./gemm.ts"}): only **the 2 bases of the
  // quantized q** count with the full M stride + the row offset. S / row statistics stay
  // block-relative, so the inner-product loop, the shared tile fill and dequant do not move by a
  // single character = bit-identical to a single-block run.
  // MUST: the payload counts in packs and the scale in rows (mixing up the units reads the next head).
  const qRows = rowWindow ? "dims.rows_full" : "dims.m";
  return `// karume attention_qk (S[b,m,n] = q[b,m,d] · k[b,n,d]ᵀ, integer dot product of per-token i8 q and k${
    dp4a ? "" : " (emulated)"
  }, half scale on the dequant side${scoreNote(score)}${rowWindow ? ", row window a" : ""}, ${i8a8GeometryNote(
    geometry,
  )}${v4 ? " + vec4" : ""})
struct Dims {
  m: u32,
  n: u32,
  k: u32,
  scale: f32,
${rowWindow ? ROW_WINDOW_DIMS_EXTRA : ""}}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> qq: array<u32>;
@group(0) @binding(2) var<storage, read> kq: array<u32>;
@group(0) @binding(3) var<storage, read_write> s: array<${scoreArrayType(
    score,
    v4 ? "vec4<f32>" : "f32",
  )}>;
@group(0) @binding(${ATTENTION_QK_Q_SCALE_BINDING}) var<storage, read> qscale: array<f32>;
@group(0) @binding(${ATTENTION_QK_K_SCALE_BINDING}) var<storage, read> kscale: array<f32>;
${scoreStoreWgsl("s", score)}
${i8a8IdotWgsl(dp4a)}

// Shared tiles are laid out **[pack][row] / [pack][column]** (as in linear's i8a8 — the inner reads
// get word stride ${I8A8_PACK} and bank conflicts drop from 8-way to 2-way)
var<workgroup> sa: array<u32, ${kPacks * i8a8TileM(geometry)}>;
var<workgroup> sb: array<u32, ${kPacks * i8a8TileN(geometry)}>;

@compute @workgroup_size(${geometry.wgX}, ${geometry.wgY})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * ${geometry.wgX}u + lid.x;
  // Eligibility requires D % 4 == 0, so a row always starts on a word boundary (the i8 payload
  // packs 4 per flat index)
  let k4 = dims.k / ${I8A8_PACK}u;
${
  v4 ? `  let n4 = dims.n / ${I8A8_PACK}u;\n` : ""
}  // The batch axis folding B and H (uniform per workgroup — a precondition for the inner
  // workgroupBarrier to meet WGSL's uniformity requirement). MUST: the q / k payloads count in
  // **packs**, the scales in rows, S in elements${v4 ? " (quads in v4)" : ""} — mixing up the units
  // reads and writes the next head when B·H ≥ 2 (a wrong value without an exception, invisible
  // to tests with B=H=1)
  let qbase = wid.z * ${qRows} * k4${rowWindow ? " + dims.row_offset * k4" : ""};
  let kbase = wid.z * dims.n * k4;
  let qsbase = wid.z * ${qRows}${rowWindow ? " + dims.row_offset" : ""};
  let ksbase = wid.z * dims.n;
  let sbase = wid.z * dims.m * ${v4 ? "n4" : "dims.n"};
${prologueA(geometry, "qbase", "q tile assignment")}
${prologueB(
  geometry,
  "kbase",
  "krow_base",
  "k is read as [N,D] as is (the contiguous direction D is the pack direction, so **no transpose is needed**)",
)}
  // The loop condition is uniform (dims is a uniform buffer) — needed for the inner
  // workgroupBarrier to meet WGSL's uniformity requirement
  let tiles = (k4 + ${kPacks - 1}u) / ${kPacks}u;
${i8a8AccumulatorInit(geometry)}
  for (var t = 0u; t < tiles; t = t + 1u) {
    // Zero-fill out of range (dot4I8Packed(0, x) == 0, so the result is exact even with a K remainder)
${fillA(geometry, "qq")}
${fillB(geometry, "kq", "krow_base", "k")}
    workgroupBarrier();
${i8a8InnerProductLoop(geometry)}
    workgroupBarrier();
  }
${store(geometry, v4, score)}
}
`;
};

/**
 * Uniform Dims (`{m,n,k}` + the 4th word, the f32 bits of the **half scale** — the same order as
 * the f32 variant's `attentionQkParams` in {@link "./attention.ts"}).
 *
 * MUST: do not bake scale into the WGSL (a pipeline per distinct value).
 *
 * `window` is the 5th–6th words passed **only by the row-window variant** (the same rule as the f32
 * variant's `attentionQkParams`; the struct grows to 32 bytes). MUST: `m` is the block row count
 * (not the full M).
 */
export const attentionQkI8a8Params = (
  m: number,
  n: number,
  k: number,
  scale: number,
  window?: GemmRowWindowSpan,
): Uint32Array<ArrayBuffer> => {
  // MUST: check finiteness on the f32 word (as in attention.ts / state-attention.ts — a value finite
  // in f64 that becomes +Inf in f32 contaminates S through 0·Inf = NaN. Redundant with the attrs
  // check on the public API side).
  if (!Number.isFinite(Math.fround(scale))) {
    throw new CodegenError(
      `attention_qk i8a8 params: scale must be a finite number as f32 (${scale})`,
    );
  }
  if (!Number.isSafeInteger(k) || k < 0 || k % 4 !== 0) {
    throw new CodegenError(
      `attention_qk i8a8 params: k must be a non-negative integer multiple of 4 (${k})`,
    );
  }
  if (k > LINEAR_I8A8_MAX_K) {
    throw new CodegenError(
      `attention_qk i8a8 params: k=${k} exceeds the i32 reduction guard ${LINEAR_I8A8_MAX_K}`,
    );
  }
  const params = gemmParams("attention_qk", m, n, k);
  if (window === undefined) {
    new Float32Array(params.buffer)[3] = scale;
    return params;
  }
  assertGemmRowWindow("attention_qk i8a8 row window params", m, window);
  const wide = new Uint32Array(8);
  wide.set(params.subarray(0, 3));
  new Float32Array(wide.buffer)[3] = scale;
  wide[4] = window.offset;
  wide[5] = window.rowsFull;
  return wide;
};

// ---------------------------------------------------------------------------
// ③PV (P̃ is turned into integers without being materialised / V is per-column i8)
// ---------------------------------------------------------------------------

/**
 * Key of ③PV's i8a8 variant (the same convention as ①QK — generation ++ / dtype field i8a8 /
 * geometry / dot-product variant).
 */
export const attentionPvI8a8Key = (
  v4: boolean,
  dp4a: boolean,
  score: MaterialisedScoreStorage = "f32",
  geometry: I8a8Geometry = defaultI8a8Geometry("attention_pv"),
  rowWindow = false,
): string =>
  `attention_pv:v3:i8a8:${i8a8GeometryKeyPart(geometry, v4)}:${dp4a ? "dp4a" : "dp4aEmu"}${scoreKeyPart(
    score,
  )}${rowWindow ? ":rwc" : ""}`;

/**
 * Whether O can be written as `vec4<f32>`. **The reduction-side (N) condition is already covered by
 * the eligibility check (`N % 4 == 0`)**, so this looks only at the output columns = D (the same
 * rule as ①QK / linear's i8a8).
 */
export const attentionPvI8a8UsesVec4 = (depth: number): boolean => depth % I8A8_PACK === 0;

/**
 * Fill of the A tile (P̃). It **reads S, builds `round(127·exp(S−m))` on the spot and packs 4 per
 * word**, so not a single byte of P is materialised (keeping the property of ADR 0023 as is).
 *
 * MUST: apply the conversion **only to in-range elements**. Applying `exp(0−m)·127` out of range
 * mixes non-zero integers into the dot product, and only the K-remainder tile goes silently wrong
 * (the same MUST as `fillA` in gemm.ts).
 * MUST: `exp` is **a per-component scalar expression** (not grouped into a `vec4` — vector versions
 * of transcendental functions are implementation-dependent).
 * MUST: the row max `m` is the one of **the row this fill slot handles** (`row_max{slot}`). Reusing
 * a single one mixes in the next row's max on geometries where one thread fills 2 or more rows
 * (`tileM > threads / kPacks`) — a wrong value without an exception.
 * NOTE: there is no clamp because `exp(S−m) ≤ 1` follows structurally from `m = amax` (design
 * §2.2). The detector of a "lattice becomes 128" regression is **this 127 and the 1/127 on the
 * dequant side**, not a clamp; comparison with the reference catches both.
 */
const fillPTile = (geometry: I8a8Geometry, score: MaterialisedScoreStorage): string => {
  const stride = i8a8FillStride(geometry);
  const lane = (slot: number, component: string): string =>
    `round(exp(raw${slot}.${component} - row_max${slot}) * ${P_LATTICE_MAX}.0)`;
  const slots = Array.from(
    { length: i8a8ASlots(geometry) },
    (_, slot) =>
      `    var av${slot} = 0u;
    if (arow${slot} < dims.m && apack < k4) {
      let raw${slot} = ${scoreReadQuad("s", score, `arow_base${slot} + apack`)};
      // P̃'s scale is **fixed at 1/127** (the in-row max is exp(0) = 1 by construction — no amax).
      // There is not a single division, so this stays outside WGSL's 2.5 ULP problem
      let p${slot} = vec4<f32>(
        ${["x", "y", "z", "w"].map((component) => lane(slot, component)).join(",\n        ")},
      );
      av${slot} = pack4xI8(vec4<i32>(p${slot}));
    }
    sa[sa_at${at(slot * stride)}] = av${slot};`,
  ).join("\n");
  return `    // Zero-fill out of range (qP = 0 adds nothing to the dot product, so the result is exact
    // even with a K remainder)
    let apack = t * ${i8a8KPacks(geometry)}u + ap;
${slots}`;
};

/** Max of the row each fill slot handles (invariant over the K tile loop, so read once). */
const prologuePvStats = (geometry: I8a8Geometry): string =>
  Array.from(
    { length: i8a8ASlots(geometry) },
    (_, slot) =>
      `  let stat_at${slot} = select(0u, (rbase + arow${slot}) * 2u, arow${slot} < dims.m);
  let row_max${slot} = stats[stat_at${slot}];`,
  ).join("\n");

/**
 * Store of ③PV. As in ①QK's {@link store}, rows and column quads are unrolled at codegen and
 * `acc{row}_{column quad}` is read statically. The guard structure, the folding order of `prow·vs`
 * and the multiplication by `1/127` are the same for every geometry.
 */
const pvStore = (geometry: I8a8Geometry, v4: boolean): string => {
  const { regM, regN } = geometry;
  const quads = regN / I8A8_PACK;
  const rows = rowDecls(geometry);
  // MUST: prow is the statistic of **the output row** (`rbase + orow`). Reusing the `arow` side
  // read in the A tile fill applies the row inv of another thread's row (it matches only with
  // B·H = M = 1).
  const prow = (row: number, indent: string): string =>
    `${indent}let prow = stats[(rbase + orow${row}) * 2u + 1u] * ${INV_P_LATTICE};`;
  if (v4) {
    const blocks = Array.from({ length: quads }, (_, quad) => {
      const rowStores = Array.from(
        { length: regM },
        (_, row) =>
          `    if (orow${row} < dims.m) {
${prow(row, "      ")}
      o[obase + orow${row} * n4 + ocq${quad}] = vec4<f32>(acc${row}_${quad}) * (prow * vs);
    }`,
      ).join("\n");
      return `  if (ocq${quad} < n4) {
    // D % ${I8A8_PACK} == 0 and ocq${quad} < n4, so oc + ${I8A8_PACK - 1} < D.
    // MUST: the column scale is a **row** of Vᵀ = (b, h, d) (= V's per-column scale). Mixing it
    // up with the row scale raises no exception
    let oc = ocq${quad} * ${I8A8_PACK}u;
    let vs = vec4<f32>(
      vscale[vsbase + oc],
      vscale[vsbase + oc + 1u],
      vscale[vsbase + oc + 2u],
      vscale[vsbase + oc + 3u],
    );
    // MUST: fold prow·vs into one f32 first, then multiply f32(acc) by it (no bias, so no fma)
${rowStores}
  }`;
    }).join("\n");
    return `${quadDecls(geometry)}
${rows}
${blocks}`;
  }
  const lanes = ["x", "y", "z", "w"] as const;
  const rowStores = Array.from({ length: regM }, (_, row) => {
    const writes = Array.from({ length: regN }, (_, col) => {
      const offset = at(col);
      const acc = `acc${row}_${Math.floor(col / I8A8_PACK)}.${lanes[col % I8A8_PACK]}`;
      return `    if (ocol${offset} < dims.n) {
      o[orow_base + ocol${offset}] = f32(${acc}) * (prow * vscale[vsbase + ocol${offset}]);
    }`;
    }).join("\n");
    return `  if (orow${row} < dims.m) {
    let orow_base = obase + orow${row} * dims.n;
    // MUST: fold prow·vs into one f32 first, then multiply f32(acc) by it (no bias, so no fma)
${prow(row, "    ")}
${writes}
  }`;
  }).join("\n");
  return `  let ocol = wid.x * ${i8a8TileN(geometry)}u + lid.x * ${regN}u;
${rows}
${rowStores}`;
};

export const attentionPvI8a8Wgsl = (
  v4: boolean,
  dp4a: boolean,
  score: MaterialisedScoreStorage = "f32",
  geometry: I8a8Geometry = defaultI8a8Geometry("attention_pv"),
  rowWindow = false,
): string => {
  assertI8a8Geometry(geometry, "attention_pv i8a8");
  const kPacks = i8a8KPacks(geometry);
  // Row window (`"c"` of `GemmRowWindow` in {@link "./gemm.ts"}): only **the 1 base of O** counts
  // with the full M stride + the row offset. S / row statistics / Vᵀ stay block-relative, so
  // neither P̃ generation nor dequant moves by a single character = bit-identical to a single-block
  // run.
  const oUnit = v4 ? "n4" : "dims.n";
  return `// karume attention_pv (O[b,m,d] = P[b,m,n] · v[b,n,d], P̃ = round(127·exp(S − m)) not materialised${
    dp4a ? "" : " (emulated)"
  }, v is per-column i8 (Vᵀ contiguous)${scoreNote(score)}${rowWindow ? ", row window c" : ""}, ${i8a8GeometryNote(
    geometry,
  )}${v4 ? " + vec4" : ""})
struct Dims {
  m: u32,
  n: u32,
  k: u32,
${rowWindow ? ROW_WINDOW_DIMS_EXTRA : ""}}
@group(0) @binding(0) var<uniform> dims: Dims;
// Eligibility requires N % 4 == 0, so a row of S always starts on a quad boundary (the same step
// as P̃'s packing of 4)
@group(0) @binding(1) var<storage, read> s: array<${scoreArrayType(score, "vec4<f32>")}>;
@group(0) @binding(2) var<storage, read> vq: array<u32>;
@group(0) @binding(3) var<storage, read> stats: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<${v4 ? "vec4<f32>" : "f32"}>;
@group(0) @binding(${ATTENTION_PV_V_SCALE_BINDING}) var<storage, read> vscale: array<f32>;
${scoreQuadLoaderWgsl("s", score)}
${i8a8IdotWgsl(dp4a)}

// Shared tiles are laid out **[pack][row] / [pack][column]** (as in ①QK / linear's i8a8 — the inner
// reads get word stride ${I8A8_PACK} and bank conflicts drop from 8-way to 2-way)
var<workgroup> sa: array<u32, ${kPacks * i8a8TileM(geometry)}>;
var<workgroup> sb: array<u32, ${kPacks * i8a8TileN(geometry)}>;

@compute @workgroup_size(${geometry.wgX}, ${geometry.wgY})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * ${geometry.wgX}u + lid.x;
  // The reduction axis N counts in quads on the S side and in packs on the vq side (the same step
  // of 4 elements)
  let k4 = dims.k / ${I8A8_PACK}u;
${
  v4 ? `  let n4 = dims.n / ${I8A8_PACK}u;\n` : ""
}  // The batch axis folding B and H (uniform per workgroup — a precondition for the inner
  // workgroupBarrier to meet WGSL's uniformity requirement). MUST: S counts in quads, vq in packs,
  // the row statistics in 2 words per row, vscale in D entries, O in elements${v4 ? " (quads in v4)" : ""}
  // — mixing up the units reads and writes the next head when B·H ≥ 2 (a wrong value without an
  // exception, invisible with B=H=1)
  let sbase = wid.z * dims.m * k4;
  let vbase = wid.z * dims.n * k4;
  let rbase = wid.z * dims.m;
  let vsbase = wid.z * dims.n;
  let obase = wid.z * ${rowWindow ? "dims.rows_full" : "dims.m"} * ${oUnit}${
    rowWindow ? ` + dims.row_offset * ${oUnit}` : ""
  };
${prologueA(geometry, "sbase", "P̃ tile assignment")}
  // Row max m (invariant over the K tile loop, so read once). On the edge tile arow >= M can
  // happen, so the index falls back to 0 (the value read is used only in the arow < M branch)
${prologuePvStats(geometry)}
${prologueB(
  geometry,
  "vbase",
  "vrow_base",
  "columns are D. vq is [D,N] with N contiguous (= the pack direction), so **no transpose is needed**",
)}
  // The loop condition is uniform (dims is a uniform buffer) — needed for the inner
  // workgroupBarrier to meet WGSL's uniformity requirement
  let tiles = (k4 + ${kPacks - 1}u) / ${kPacks}u;
${i8a8AccumulatorInit(geometry)}
  for (var t = 0u; t < tiles; t = t + 1u) {
${fillPTile(geometry, score)}
${fillB(geometry, "vq", "vrow_base", "v")}
    workgroupBarrier();
${i8a8InnerProductLoop(geometry)}
    workgroupBarrier();
  }
${pvStore(geometry, v4)}
}
`;
};

/**
 * Uniform Dims (`{m, n, k}` = `{M, D, N}` — the same order as the f32 variant's
 * `attentionPvParams` in {@link "./attention.ts"}. The half scale does not appear in ③PV).
 *
 * `window` is the 4th–5th words passed **only by the row-window variant** (the struct grows to 32
 * bytes).
 * MUST: `m` is the block row count (not the full M).
 */
export const attentionPvI8a8Params = (
  m: number,
  n: number,
  k: number,
  window?: GemmRowWindowSpan,
): Uint32Array<ArrayBuffer> => {
  if (!Number.isSafeInteger(k) || k < 0 || k % 4 !== 0) {
    throw new CodegenError(
      `attention_pv i8a8 params: k (= N) must be a non-negative integer multiple of 4 (${k})`,
    );
  }
  if (k > LINEAR_I8A8_MAX_K) {
    throw new CodegenError(
      `attention_pv i8a8 params: k=${k} exceeds the i32 reduction guard ${LINEAR_I8A8_MAX_K}`,
    );
  }
  const params = gemmParams("attention_pv", m, n, k);
  if (window === undefined) return params;
  assertGemmRowWindow("attention_pv i8a8 row window params", m, window);
  const wide = new Uint32Array(8);
  wide.set(params.subarray(0, 3));
  wide[3] = window.offset;
  wide[4] = window.rowsFull;
  return wide;
};
