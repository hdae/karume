/**
 * The **w8a8 variant** of linear (an **integer dot product** of per-token i8 activations × per-channel
 * i8 weights) — design in docs/research/2026-08-03-dp4a-w8a8-design.md §4.3, opt-in through
 * `SessionOptions.linearCompute: "a8"` (default `"f32"`).
 *
 * ```
 * acc[i32] = Σ_k xq[m,k] · wq[n,k]                    (exact integers, 0 roundings)
 * out      = f32(acc) · (xs[row] · wscale[col]) + bias[col]
 * ```
 *
 * ## Why the skeleton is not shared with gemm.ts (relation to the MUST of ADR 0022)
 *
 * ADR 0022's "one source of truth for the inner-product loop" is a MUST that **keeps the f32
 * reduction order the same across all ops**; this kernel reduces by exact `i32` addition
 * (order-independent), so the property that MUST protects holds from the start. The element type of
 * the shared A/B tiles (`u32` packing 4), the fill assignment and the shape of the inner product
 * (`dot4I8Packed` taking 4 elements at once) all differ from the f32 skeleton, so splicing in
 * fragments would hurt the readability of both. **The tile geometry is independent of the f32
 * skeleton** ({@link "./i8a8-geometry.ts"}), a freedom that can be chosen by measurement precisely
 * because the integer reduction is order-independent — the geometry arithmetic and the
 * inner-product loop keep a single source of truth on the i8a8 side (that file).
 *
 * ## Numeric contract (the core of this path)
 *
 * - **The integer reduction is an order-independent exact value**, so `(Σ xq·wq)` is the same single
 *   integer in every implementation. The reason ADR 0019 rejected the `(Σ x·q)·s` form (bit mismatch
 *   between GPU dequant and the CPU decode) disappears with w8a8, and **GPU and TS reference match
 *   at atol=0** (src/reference/i8a8.ts).
 * - MUST: fix the multiplication order of dequant — **fold `xs[row] · wscale[col]` into one f32
 *   first**, then multiply `f32(acc)` by it, and add bias last. The sequential form
 *   `f32(acc)·xs·wscale` moves the rounding points and breaks bit identity with the reference.
 * - MUST: the final multiply-add is **`fma`**. A plain `a * b + c` is a form the driver **may**
 *   fuse, which makes it implementation-dependent (on the development machine's RTX 3080 Ti it was
 *   measured as fused — the plain expression, via `let`, and pinned with `bitcast`, all three forms
 *   gave the same fused value). However the WGSL spec only says the accuracy of `fma` is
 *   **inherited** from `x*y+z` (§15.7.4.1 — a non-fused expansion also conforms), so a single
 *   rounding is **an implementation property, not a spec guarantee**. Fused implementations match
 *   the TS reference (which rounds once in double precision) bit for bit; non-fused ones can differ
 *   within 1 ULP (the Metal measurement = docs/known-issues.md).
 * - **The K remainder is exact with zero fill** (`dot4I8Packed(0, x) == 0`). Detection limit (1) of
 *   ADR 0022 (Consequences: a zero-fill regression on only one of A/B is hidden because the products
 *   cancel) carries over as is.
 *
 * ## The two integer dot-product variants (numerically identical)
 *
 * `dot4I8Packed` (WGSL language extension `packed_4x8_integer_dot_product`) and
 * `dot(unpack4xI8(a), unpack4xI8(b))` (core WGSL) **return the same integer** — integer addition is
 * associative and exact, and intermediate values are at most 4·127·128 = 65,024 (activations are
 * ±127, weights may include the i8 value −128 — the same bound as the K limit check below), so no
 * wrap-around can occur. Whether the extension exists therefore **affects only speed**, and neither
 * fail loudly nor a tolerance branch is needed. The choice is made from the enumeration of
 * `navigator.gpu.wgslLanguageFeatures` ({@link dp4aAvailable}), and the key carries `:dp4a` /
 * `:dp4aEmu`, so the diagnostics (ADR 0021) show which one ran. The only mechanical detector is the
 * test that "runs both variants on a machine with the extension and compares them at atol=0"
 * (tests/gpu_i8a8_test.ts, and for the QK side of fused attention
 * tests/gpu_attention_pv_i8a8_test.ts).
 *
 * ## w4a8 variant (`weight: "i4"` — i4-resident weights × per-token i8 activations)
 *
 * i4 residency saves download size and VRAM, but it fell outside the i8a8 predicate (i8-resident),
 * flowed to the f32 compute path and lost exactly the dp4a gain
 * (docs/research/2026-08-21-anima-i4-seat-speed.md). This variant carries a second numeric contract
 * for **putting i4 onto the integer dot product as is**:
 *
 * ```
 * accf = 0
 * for gi in 0..k/g-1:
 *     acci = Σ_{i∈group gi} xq[row,i] · wq[col,i]   // exact in i32 (|acci| ≤ g·127·8)
 *     accf = fma(f32(acci), wscale[col, gi], accf)  // flush only at group boundaries
 * out = fma(accf, xs[row], bias[col])
 * ```
 *
 * - Rounding happens in **`k/g + 1` fma operations**, and the flush is **exactly at the group
 *   boundary** (not at the K tile boundary). This count treats fma as one rounding and holds only
 *   for `g ≤ 16,513`, where the conversion `f32(acci)` is lossless (the check
 *   {@link LINEAR_W4A8_MAX_GROUP} lets values beyond that through — ADR 0076 §2 and the addendum of
 *   2026-08-31). So the i8a8 freedom "tile geometry is outside the numeric contract" carries over
 *   to w4a8 as is — however the tiles are split, the inside of a group is exact i32 addition, and
 *   only g decides where the flush happens.
 * - **MUST: the i8 variant's MUST of folding `xs · wscale` into one f32 first does not hold for
 *   w4a8.** wscale changes per group, so folding would build `xs·wscale` per group, adding `k/g`
 *   roundings and breaking the structure "xs is a per-row quantity". xs goes to **the single final
 *   fma** (this asymmetry is the only difference in numeric contract between w4a8 and i8a8).
 * - The inside of a group is exact in i32, so as with the i8 variant **GPU and TS reference match
 *   at atol=0** (`referenceLinearW4a8` in {@link "../reference/i8a8.ts"}).
 * - A K tile that crosses a group boundary mixes the weights of 2 groups into one i32, so
 *   `groupSize % tileK == 0` is rejected at generation (`assertI8a8GroupGeometry` in
 *   {@link "./i8a8-geometry.ts"}). Together with k % g == 0, **a K remainder cannot occur by
 *   construction**.
 *
 * NOTE: the w4a8 K loop has two levels, "group outside × tile inside", so the shared fragments
 * ({@link i8a8InnerProductLoop} / the A-side fill) are embedded with one level less indentation.
 * **A single source of truth for the fragments** took priority over the look of the output
 * (copying the geometry's index arithmetic invites getting the pack stride wrong on one side only).
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";
import { gemmParams, LINEAR_SCALE_BINDING } from "./gemm.ts";
import {
  i4GroupKeyPart,
  i4GroupShift,
  i4IntegerLanesWgsl,
  weightKeyPart,
  type WeightStorage,
} from "./weight-storage.ts";
import {
  assertI8a8Geometry,
  assertI8a8GroupGeometry,
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

export { LINEAR_SCALE_BINDING } from "./gemm.ts";

/**
 * Binding of the activation scale (per-token). **Not a weight-storage concept**, so it lives here
 * rather than in weight-storage.ts (after linear's current 0..5).
 */
export const LINEAR_ACT_SCALE_BINDING = LINEAR_SCALE_BINDING + 1;

/**
 * Overflow limit of the i32 reduction. quantize_rows closes activations at ±127, but **weights may
 * include the i8 value −128**, so the bound is `|acc| ≤ k · 127 · 128 = k · 16256` — from the
 * positive i32 maximum 2,147,483,647 that gives `k ≤ 132,104` (the earlier derivation assuming ±127,
 * k ≤ 133,144, is 2 too loose). The limit 2^17 = 131,072 is inside both bounds (131,072 · 16256 =
 * 2,130,706,432 < 2^31), with 16 times headroom over the DiT's largest k = 8,192. A larger k fails
 * loudly — letting it through silently would corrupt the value, sign included, by i32 wrap-around.
 */
export const LINEAR_I8A8_MAX_K = 131072;

/**
 * i32 reduction limit **of w4a8** (on the group length). i4 lanes have `|q| ≤ 8`, so one group has
 * `|acci| ≤ g · 127 · 8 = g · 1016`, and the i32 maximum 2,147,483,647 gives `g ≤ 2,113,665`. The
 * limit takes 2,113,664, one inside that (the safe side — the exact-equality case is not let
 * through).
 *
 * MUST: do not apply {@link LINEAR_I8A8_MAX_K} to w4a8. The flush is per group, so only one group
 * sits in an i32, and however long k is it does not wrap around (the k-side quantity moves to a
 * separate contract, the f32 accumulator's rounding count `k/g + 1`). Using the k limit instead
 * would reject long k values that should pass while letting the real danger, "g is abnormally
 * large", through.
 */
export const LINEAR_W4A8_MAX_GROUP = 2113664;

/** Name of the WGSL language extension used for the integer dot product (not a device feature). */
export const DP4A_WGSL_FEATURE = "packed_4x8_integer_dot_product";

/**
 * Whether the `dot4I8Packed` variant can be used.
 *
 * MUST: use this only to **choose between numerically identical variants** (paired with the
 * docstring in src/gpu/device.ts). The enumeration guarantees that compilation succeeds, not that
 * the hardware supports it, so it cannot serve as feature detection — it works here because a wrong
 * choice does not change the result by a single bit on this path.
 */
export const dp4aAvailable = (wgslLanguageFeatures: ReadonlySet<string>): boolean =>
  wgslLanguageFeatures.has(DP4A_WGSL_FEATURE);

/**
 * Whether the output can be written as `vec4<f32>`. **The k-side condition is already covered by
 * the eligibility check (`k % 4 == 0`)**, so this looks only at n (in contrast to the f32
 * skeleton's {@link gemmUsesVec4}, which looks at both k and n — for i8a8 the k condition is
 * mandatory, and the n condition only falls back to the scalar store variant).
 */
export const linearI8a8UsesVec4 = (n: number): boolean => n % I8A8_PACK === 0;

/**
 * Pipeline key. The geometry became a parameter, so **the tile edges alone do not determine the
 * output** ({@link i8a8GeometryKeyPart}) — the generation is bumped to `v4` and the whole geometry
 * discriminator goes in.
 *
 * MUST: w4a8 appends the storage discriminator + group length (`:wi4g32`). With the default
 * arguments the i8 variant stays empty as before (existing keys are byte-identical — the browser's
 * implicit shader cache is not rebuilt). Forgetting the group length gives the silent wrong value
 * **a group-32 pipeline running on group-64 assets** (g is baked into the WGSL as a shift — ADR
 * 0069).
 */
export const linearI8a8Key = (
  v4: boolean,
  dp4a: boolean,
  geometry: I8a8Geometry = defaultI8a8Geometry("linear"),
  weight: WeightStorage = "i8",
  groupSize?: number,
): string => {
  // MUST: run the same two checks as {@link linearI8a8Wgsl} with the same wording (side effects
  // only — the return value for valid input does not change by a single byte). Storage other than
  // i8 leaves the fragment above empty, so without the check f32 storage could not be told apart
  // from i8 by key, and another path sharing the slot would run on the same assets.
  if (weight !== "i8" && weight !== "i4") {
    throw new CodegenError(`linear i8a8: 重み格納は i8 / i4 のみ（${weight}）`);
  }
  i4GroupShift("linear i8a8", weight, groupSize);
  // The generation that keeps w4a8's group scale in range on edge columns too. The i8 output stays
  // as before.
  return (
    `linear:v${weight === "i4" ? 5 : 4}:i8a8:${i8a8GeometryKeyPart(geometry, v4)}:${
      dp4a ? "dp4a" : "dp4aEmu"
    }` + (weight === "i8" ? "" : `${weightKeyPart(weight)}${i4GroupKeyPart(groupSize)}`)
  );
};

/**
 * The two integer dot-product variants. **The returned integers are exactly the same**; they differ
 * only in speed (the docstring above).
 *
 * MUST: every i8a8 kernel uses this one function (the QK of fused attention too —
 * src/kernels/attention-i8a8.ts). The claim "the emulation is numerically identical" rests on
 * `idot` having a single body; copying it would let each kernel drift separately.
 *
 * The comments inside the WGSL below are free to edit: the kernel fingerprint hashes the WGSL
 * without its comments (ADR 0117 addendum 2026-10-19), so a comment edit moves no embedded table.
 * Changing a token of the WGSL still does, and then the embedded geometry profile tables are
 * regenerated (ADR 0117 decision 4).
 */
export const i8a8IdotWgsl = (dp4a: boolean): string =>
  dp4a
    ? `// integer dot product (WGSL language extension packed_4x8_integer_dot_product); returns **the same integer** as the emulated variant
fn idot(a: u32, b: u32) -> i32 {
  return dot4I8Packed(a, b);
}`
    : `// emulation of the integer dot product (core WGSL only); returns **the same integer** as dot4I8Packed —
// integer addition is associative and exact, and the intermediates stay at most 4·127·128 = 65,024, so nothing wraps
fn idot(a: u32, b: u32) -> i32 {
  return dot(unpack4xI8(a), unpack4xI8(b));
}`;

/** Additive index offset (0 is omitted — keeps the output readable). */
const at = (offset: number): string => (offset === 0 ? "" : ` + ${offset}u`);

/**
 * Assignment of the x tile (row numbers and row-start offsets that are invariant over the K tile
 * loop). One thread takes {@link i8a8ASlots} of them at the fill stride.
 */
const prologueA = (geometry: I8a8Geometry): string => {
  const tileM = i8a8TileM(geometry);
  const stride = i8a8FillStride(geometry);
  const slots = Array.from(
    { length: i8a8ASlots(geometry) },
    (_, slot) =>
      `  let arow${slot} = ${slot === 0 ? `wid.y * ${tileM}u + ar` : `arow0 + ${slot * stride}u`};
  let arow_base${slot} = arow${slot} * k4;`,
  ).join("\n");
  return `  // x tile assignment (${tileM} rows × ${i8a8KPacks(geometry)} packs over ${i8a8Threads(
    geometry,
  )} threads in ${i8a8ASlots(geometry)} rounds)
  let ar = tid / ${i8a8KPacks(geometry)}u;
  let ap = tid % ${i8a8KPacks(geometry)}u;
  let sa_at = ap * ${tileM}u + ar;
${slots}`;
};

/**
 * Assignment of the W tile. Weights are read as `[n,k]` (the contiguous direction is k = the pack
 * direction, so no transpose is needed).
 *
 * `rowStride` is the row-start step. For i8 it is the **word count** `k4` (4 per word); for i4 a
 * word holds 8 elements and a pack is half a word, so it is the **flat element count** `dims.k`
 * (the word split is done by {@link i4IntegerLanesWgsl}).
 * MUST: the default is `"k4"` — the i8 variant's output does not move by a single byte.
 */
const prologueB = (geometry: I8a8Geometry, rowStride = "k4"): string => {
  const tileN = i8a8TileN(geometry);
  const stride = i8a8FillStride(geometry);
  const slots = Array.from(
    { length: i8a8BSlots(geometry) },
    (_, slot) =>
      `  let wcol${slot} = ${slot === 0 ? `wid.x * ${tileN}u + wc` : `wcol0 + ${slot * stride}u`};
  let wrow_base${slot} = wcol${slot} * ${rowStride};`,
  ).join("\n");
  return `  // W tile assignment (${tileN} output channels × ${i8a8KPacks(geometry)} packs over ${i8a8Threads(
    geometry,
  )} threads in ${i8a8BSlots(geometry)} rounds).
  // Weights are read as [n,k] (the contiguous direction is k = the pack direction, so **no
  // transpose is needed**)
  let wc = tid / ${i8a8KPacks(geometry)}u;
  let wp = tid % ${i8a8KPacks(geometry)}u;
  let sb_at = wp * ${tileN}u + wc;
${slots}`;
};

/**
 * Shared tile fill on the A (activation) side (out of range is zero-filled — `idot(0, x) == 0`, so
 * it is exact even with a K remainder).
 *
 * MUST: both the i8 and the w4a8 variant use this one function. `t` is **the running K tile
 * number**, so in w4a8's two-level loop (group outside × tile inside) the expression is identical
 * as long as `t` is rebuilt from both levels.
 */
const fillTilesA = (geometry: I8a8Geometry): string => {
  const kPacks = i8a8KPacks(geometry);
  const stride = i8a8FillStride(geometry);
  const a = Array.from(
    { length: i8a8ASlots(geometry) },
    (_, slot) =>
      `    var av${slot} = 0u;
    if (arow${slot} < dims.m && apack < k4) {
      av${slot} = xq[arow_base${slot} + apack];
    }
    sa[sa_at${at(slot * stride)}] = av${slot};`,
  ).join("\n");
  return `    // Zero-fill out of range (dot4I8Packed(0, x) == 0, so the result is exact even with a K
    // remainder)
    let apack = t * ${kPacks}u + ap;
${a}`;
};

/** Shared tile fill on the B (i8 weight) side. */
const fillTilesB = (geometry: I8a8Geometry): string => {
  const kPacks = i8a8KPacks(geometry);
  const stride = i8a8FillStride(geometry);
  const b = Array.from(
    { length: i8a8BSlots(geometry) },
    (_, slot) =>
      `    var wv${slot} = 0u;
    if (wcol${slot} < dims.n && wpack < k4) {
      wv${slot} = w[wrow_base${slot} + wpack];
    }
    sb[sb_at${at(slot * stride)}] = wv${slot};`,
  ).join("\n");
  return `    let wpack = t * ${kPacks}u + wp;
${b}`;
};

/** Shared tile fill (zero-filled out of range — `idot(0, x) == 0`, exact even with a K remainder). */
const fillTiles = (geometry: I8a8Geometry): string =>
  `${fillTilesA(geometry)}\n${fillTilesB(geometry)}`;

/**
 * Shared tile fill on the B (i4 weight) side. 1 pack = 4 elements is half a word (8 elements), so
 * the index is built in **flat elements**, and {@link i4IntegerLanesWgsl} handles the word split
 * and nibble extraction.
 *
 * There is no K-side guard because `k % g == 0` (an invariant of the distribution) and
 * `g % tileK == 0` ({@link assertI8a8GroupGeometry}) mean **a K remainder cannot occur by
 * construction**. The A side's `apack < k4` stays because the fragment is shared, but it is always
 * true for w4a8.
 */
const fillTilesBI4 = (geometry: I8a8Geometry): string => {
  const stride = i8a8FillStride(geometry);
  const b = Array.from(
    { length: i8a8BSlots(geometry) },
    (_, slot) =>
      `    var wv${slot} = 0u;
    if (wcol${slot} < dims.n) {
      wv${slot} = i4lanes(wrow_base${slot} + welem);
    }
    sb[sb_at${at(slot * stride)}] = wv${slot};`,
  ).join("\n");
  return `    // i4 has 8 elements per word, so the pack index is in flat elements (k % g == 0 ∧
    // g % tileK == 0 leaves no K remainder)
    let welem = t * ${geometry.tileK}u + wp * ${I8A8_PACK}u;
${b}`;
};

/**
 * Output store. Both rows and column quads are unrolled at codegen and `acc{row}_{column quad}` is
 * read statically (unrolled so that no dynamic `acc[i]` index remains — the reason is in
 * {@link i8a8AccumulatorInit}). The guard structure, the single rounding of `fma` and the folding
 * order of `xs·wscale` are the same for every geometry.
 */
const store = (geometry: I8a8Geometry, v4: boolean): string => {
  const { regM, regN } = geometry;
  const quads = regN / I8A8_PACK;
  const rows = Array.from({ length: regM }, (_, row) =>
    row === 0
      ? `  let orow0 = wid.y * ${i8a8TileM(geometry)}u + lid.y * ${regM}u;`
      : `  let orow${row} = orow0 + ${row}u;`,
  ).join("\n");
  if (v4) {
    const quadDecls = Array.from({ length: quads }, (_, quad) =>
      quad === 0
        ? `  let ocq0 = wid.x * ${i8a8TileN(geometry) / I8A8_PACK}u + lid.x * ${quads}u;`
        : `  let ocq${quad} = ocq0 + ${quad}u;`,
    ).join("\n");
    const blocks = Array.from({ length: quads }, (_, quad) => {
      const rowStores = Array.from(
        { length: regM },
        (_, row) =>
          `    if (orow${row} < dims.m) {
      out[orow${row} * n4 + ocq${quad}] = fma(vec4<f32>(acc${row}_${quad}), xscale[orow${row}] * ws, biasv);
    }`,
      ).join("\n");
      return `  if (ocq${quad} < n4) {
    // n % ${I8A8_PACK} == 0 and ocq${quad} < n4, so oc + ${
      I8A8_PACK - 1
    } < n (bias is always f32 — ADR 0006)
    let oc = ocq${quad} * ${I8A8_PACK}u;
    let ws = vec4<f32>(wscale[oc], wscale[oc + 1u], wscale[oc + 2u], wscale[oc + 3u]);
    let biasv = vec4<f32>(bias[oc], bias[oc + 1u], bias[oc + 2u], bias[oc + 3u]);
    // MUST: fold xs·wscale into one f32 first; the multiply-add is fma (single rounding)
${rowStores}
  }`;
    }).join("\n");
    return `  let n4 = dims.n / ${I8A8_PACK}u;
${quadDecls}
${rows}
${blocks}`;
  }
  const lanes = ["x", "y", "z", "w"] as const;
  const rowStores = Array.from({ length: regM }, (_, row) => {
    const writes = Array.from({ length: regN }, (_, col) => {
      const offset = at(col);
      const acc = `acc${row}_${Math.floor(col / I8A8_PACK)}.${lanes[col % I8A8_PACK]}`;
      return `    if (ocol${offset} < dims.n) {
      out[obase + ocol${offset}] = fma(f32(${acc}), xs * wscale[ocol${offset}], bias[ocol${offset}]);
    }`;
    }).join("\n");
    return `  if (orow${row} < dims.m) {
    let obase = orow${row} * dims.n;
    // MUST: fold xs·wscale into one f32 first; the multiply-add is fma (single rounding)
    let xs = xscale[orow${row}];
${writes}
  }`;
  }).join("\n");
  return `  let ocol = wid.x * ${i8a8TileN(geometry)}u + lid.x * ${regN}u;
${rows}
${rowStores}`;
};

/** WGSL expression of the assigned column (`col` relative to `ocol`). */
const columnAt = (col: number): string => (col === 0 ? "ocol" : `ocol + ${col}u`);

/**
 * w4a8 bindings **before the loop** — the assigned output rows / columns and the per-column row
 * start of the group scales.
 *
 * The i8 variant uses these only in the store and can put them at the end, but in w4a8 **the flush
 * at a group boundary looks up per-column scales**, so they are hoisted out of the loop (the group
 * count per row, `groups`, is loop-invariant).
 */
const prologueW4a8 = (geometry: I8a8Geometry, shift: number): string => {
  const { regM, regN } = geometry;
  const rows = Array.from({ length: regM }, (_, row) =>
    row === 0
      ? `  let orow0 = wid.y * ${i8a8TileM(geometry)}u + lid.y * ${regM}u;`
      : `  let orow${row} = orow0 + ${row}u;`,
  ).join("\n");
  const bases = Array.from(
    { length: regN },
    (_, col) => `  let wsb${col} = min(${columnAt(col)}, dims.n - 1u) * groups;`,
  ).join("\n");
  const accf = Array.from({ length: regM }, (_, row) =>
    Array.from(
      { length: regN / I8A8_PACK },
      (_, quad) => `  var accf${row}_${quad} = vec4<f32>(0.0);`,
    ).join("\n"),
  ).join("\n");
  return `  // Output assignment is **outside the loop** (the flush at a group boundary looks up
  // per-column scales)
  let ocol = wid.x * ${i8a8TileN(geometry)}u + lid.x * ${regN}u;
${rows}
  // group scales are a flat [n, k/g]. Invalid columns are clamped to the last valid column, so
  // nothing out of range is read. The row start is loop-invariant and hoisted (output of invalid
  // columns is discarded by the store guard)
  let groups = dims.k >> ${shift}u;
${bases}
  // f32 accumulator (written only at group boundaries — k/g roundings)
${accf}`;
};

/**
 * Flush at the group boundary. The only place that **folds the i32 accumulator into f32, once**, and
 * thus the rounding point (= the numeric contract) itself.
 *
 * MUST: the multiply-add is `fma` (single rounding). MUST: do not multiply by `xs` here — wscale
 * changes per group, so the i8 variant's "fold `xs·wscale` first" form does not hold, and
 * multiplying would add `k/g` roundings (the w4a8 section at the top of the file).
 */
const flushW4a8 = (geometry: I8a8Geometry): string => {
  const { regM, regN } = geometry;
  const quads = regN / I8A8_PACK;
  const blocks = Array.from({ length: quads }, (_, quad) => {
    const lanes = Array.from(
      { length: I8A8_PACK },
      (_, lane) => `wscale[wsb${quad * I8A8_PACK + lane} + gi]`,
    ).join(", ");
    const rows = Array.from(
      { length: regM },
      (_, row) =>
        `    accf${row}_${quad} = fma(vec4<f32>(acc${row}_${quad}), ws${quad}, accf${row}_${quad});`,
    ).join("\n");
    return `    let ws${quad} = vec4<f32>(${lanes});
${rows}`;
  }).join("\n");
  return `    // Flush to f32 exactly at the group boundary (MUST: single rounding of fma; xs is not
    // applied until the end)
${blocks}`;
};

/**
 * K loop of w4a8 (two levels, **group outside × tile inside**).
 *
 * `t` is rebuilt as the running K tile number, so the A-side fill and the inner-product loop use
 * **the same fragments** as the i8 variant as is. The inner bound is the constant `g / tileK`, and
 * only `groups` is a uniform run-time value (both levels meet the uniformity requirement of the
 * inner `workgroupBarrier`).
 */
const loopW4a8 = (geometry: I8a8Geometry, groupSize: number): string => {
  const tilesPerGroup = groupSize / geometry.tileK;
  const accInit = Array.from({ length: geometry.regM }, (_, row) =>
    Array.from(
      { length: geometry.regN / I8A8_PACK },
      (_, quad) => `    var acc${row}_${quad} = vec4<i32>(0);`,
    ).join("\n"),
  ).join("\n");
  return `  // The loop condition is uniform (dims is a uniform buffer) — needed for the inner
  // workgroupBarrier to meet WGSL's uniformity requirement
  for (var gi = 0u; gi < groups; gi = gi + 1u) {
    // The i32 accumulator is reset per group (|acci| ≤ g·127·8 — limit: LINEAR_W4A8_MAX_GROUP)
${accInit}
    // 1 group = exactly ${tilesPerGroup} K tiles (the g % tileK == 0 check). t is the running number
    let gbase = gi * ${tilesPerGroup}u;
    for (var gt = 0u; gt < ${tilesPerGroup}u; gt = gt + 1u) {
      let t = gbase + gt;
${fillTilesA(geometry)}
${fillTilesBI4(geometry)}
      workgroupBarrier();
${i8a8InnerProductLoop(geometry)}
      workgroupBarrier();
    }
${flushW4a8(geometry)}
  }`;
};

/**
 * Store of w4a8. `out = fma(accf, xs[row], bias[col])`, **xs is applied here for the first time**
 * (the asymmetry with the i8 variant's `xs·wscale` folding — the w4a8 section at the top of the
 * file). The unrolling of rows / column quads and the guard structure are the same as the i8
 * variant.
 */
const storeW4a8 = (geometry: I8a8Geometry, v4: boolean): string => {
  const { regM, regN } = geometry;
  const quads = regN / I8A8_PACK;
  if (v4) {
    const quadDecls = Array.from({ length: quads }, (_, quad) =>
      quad === 0 ? `  let ocq0 = ocol / ${I8A8_PACK}u;` : `  let ocq${quad} = ocq0 + ${quad}u;`,
    ).join("\n");
    const blocks = Array.from({ length: quads }, (_, quad) => {
      const rowStores = Array.from(
        { length: regM },
        (_, row) =>
          `    if (orow${row} < dims.m) {
      out[orow${row} * n4 + ocq${quad}] = fma(accf${row}_${quad}, vec4<f32>(xscale[orow${row}]), biasv);
    }`,
      ).join("\n");
      return `  if (ocq${quad} < n4) {
    // n % ${I8A8_PACK} == 0 and ocq${quad} < n4, so oc + ${
      I8A8_PACK - 1
    } < n (bias is always f32 — ADR 0006)
    let oc = ocq${quad} * ${I8A8_PACK}u;
    let biasv = vec4<f32>(bias[oc], bias[oc + 1u], bias[oc + 2u], bias[oc + 3u]);
    // MUST: xs goes to the final fma (single rounding) — wscale was folded at the group boundaries
${rowStores}
  }`;
    }).join("\n");
    return `  let n4 = dims.n / ${I8A8_PACK}u;
${quadDecls}
${blocks}`;
  }
  const lanes = ["x", "y", "z", "w"] as const;
  const rowStores = Array.from({ length: regM }, (_, row) => {
    const writes = Array.from({ length: regN }, (_, col) => {
      const column = columnAt(col);
      const acc = `accf${row}_${Math.floor(col / I8A8_PACK)}.${lanes[col % I8A8_PACK]}`;
      return `    if (${column} < dims.n) {
      out[obase + ${column}] = fma(${acc}, xs, bias[${column}]);
    }`;
    }).join("\n");
    return `  if (orow${row} < dims.m) {
    let obase = orow${row} * dims.n;
    // MUST: xs goes to the final fma (single rounding) — wscale was folded at the group boundaries
    let xs = xscale[orow${row}];
${writes}
  }`;
  }).join("\n");
  return rowStores;
};

export const linearI8a8Wgsl = (
  v4: boolean,
  dp4a: boolean,
  geometry: I8a8Geometry = defaultI8a8Geometry("linear"),
  weight: WeightStorage = "i8",
  groupSize?: number,
): string => {
  assertI8a8Geometry(geometry, "linear i8a8");
  if (weight !== "i8" && weight !== "i4") {
    throw new CodegenError(`linear i8a8: 重み格納は i8 / i4 のみ（${weight}）`);
  }
  // MUST: i4 and the group length come as a pair (a missing or extra one is a wiring bug —
  // weight-storage.ts)
  const shift = i4GroupShift("linear i8a8", weight, groupSize);
  if (shift !== undefined && groupSize !== undefined) {
    assertI8a8GroupGeometry(geometry, groupSize, "linear w4a8");
  }
  const kPacks = i8a8KPacks(geometry);
  return `// karume linear (x[m,k] · wᵀ[k,n] + bias[n], integer dot product of per-token i8 activations × ${
    shift === undefined ? "i8" : `i4 群 ${groupSize}`
  } weights${dp4a ? "" : "・エミュ"}, ${i8a8GeometryNote(geometry)}${v4 ? " + vec4" : ""})
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> xq: array<u32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<${v4 ? "vec4<f32>" : "f32"}>;
@group(0) @binding(${LINEAR_SCALE_BINDING}) var<storage, read> wscale: array<f32>;
@group(0) @binding(${LINEAR_ACT_SCALE_BINDING}) var<storage, read> xscale: array<f32>;

${i8a8IdotWgsl(dp4a)}
${shift === undefined ? "" : i4IntegerLanesWgsl("w")}
// Shared tiles are laid out **[pack][row] / [pack][column]** (rearranged from the prototype's
// [row][pack] — the inner reads get word stride ${I8A8_PACK} and bank conflicts drop from 8-way to
// 2-way)
var<workgroup> sa: array<u32, ${kPacks * i8a8TileM(geometry)}>;
var<workgroup> sb: array<u32, ${kPacks * i8a8TileN(geometry)}>;

@compute @workgroup_size(${geometry.wgX}, ${geometry.wgY})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * ${geometry.wgX}u + lid.x;
  // Eligibility requires k % 4 == 0, so a row always starts on a word boundary (the i8 payload
  // packs 4 per flat index)
  let k4 = dims.k / ${I8A8_PACK}u;
${prologueA(geometry)}
${prologueB(geometry, shift === undefined ? "k4" : "dims.k")}
${
  shift === undefined || groupSize === undefined
    ? `  // The loop condition is uniform (dims is a uniform buffer) — needed for the inner
  // workgroupBarrier to meet WGSL's uniformity requirement
  let tiles = (k4 + ${kPacks - 1}u) / ${kPacks}u;
${i8a8AccumulatorInit(geometry)}
  for (var t = 0u; t < tiles; t = t + 1u) {
${fillTiles(geometry)}
    workgroupBarrier();
${i8a8InnerProductLoop(geometry)}
    workgroupBarrier();
  }
${store(geometry, v4)}`
    : `${prologueW4a8(geometry, shift)}
${loopW4a8(geometry, groupSize)}
${storeW4a8(geometry, v4)}`
}
}
`;
};

/**
 * Uniform Dims (`{m,n,k}` — the same 3 words as the f32 skeleton, allocated as 16 bytes).
 *
 * Passing `groupSize` switches to **the w4a8 checks** (group-side overflow and divisibility of k).
 * MUST: the i8 variant's k limit ({@link LINEAR_I8A8_MAX_K}) is not applied to w4a8 (the reason
 * is at {@link LINEAR_W4A8_MAX_GROUP}).
 *
 * MUST: w4a8 also checks the logical weight element count `n·k ≤ 2^32 − 1`. The row start is first
 * built in u32 as the logical element number `wcol·dims.k`, and `i4lanes` turns it into a word
 * position, so if the product wraps it reads another output channel's weights without an exception
 * (the same mechanism as the check on linear's GEMM path — linear.ts). The i8 variant builds the row
 * start as the word count `wcol·k4`, so this check does not apply.
 */
export const linearI8a8Params = (
  m: number,
  n: number,
  k: number,
  groupSize?: number,
): Uint32Array<ArrayBuffer> => {
  if (!Number.isSafeInteger(k) || k < 0 || k % 4 !== 0) {
    throw new CodegenError(`linear i8a8 params: k は 4 の倍数の非負整数（${k}）`);
  }
  if (groupSize === undefined) {
    if (k > LINEAR_I8A8_MAX_K) {
      throw new CodegenError(
        `linear i8a8 params: k=${k} が i32 縮約の門 ${LINEAR_I8A8_MAX_K} を超える`,
      );
    }
    return gemmParams("linear", m, n, k);
  }
  // If the group does not divide K the last group is short and the flush position drifts from what
  // the WGSL bakes in (tiles/group is a constant) — letting it through silently makes the reduction
  // run past k into the next row.
  if (k % groupSize !== 0) {
    throw new CodegenError(`linear w4a8 params: k=${k} が group_size ${groupSize} で割り切れない`);
  }
  if (groupSize > LINEAR_W4A8_MAX_GROUP) {
    throw new CodegenError(
      `linear w4a8 params: group_size ${groupSize} が i32 縮約の門 ${LINEAR_W4A8_MAX_GROUP} を超える`,
    );
  }
  const params = gemmParams("linear", m, n, k);
  assertU32Params("linear w4a8 params", { "n * k": n * k });
  return params;
};
