/**
 * **Online int8 attention** (ADR 0132, perf-ledger K-82 / K-84) — the kernel behind
 * `attentionScoreStorage: "none"` with `attentionCompute: "a8"`. One dispatch computes
 * `O = softmax((q·scale)(k·scale)ᵀ) · v` for a window of query rows **without ever writing S**:
 * it streams the keys in tiles, keeps the running row max `m`, the running row sum `l` and an f32
 * output accumulator, and folds every tile's integer products into them (the FlashAttention-2
 * work partition, with `dot4I8Packed` in place of matrix units).
 *
 * ```
 * per key tile (64 keys):
 *   S      = f32(Σ_d qq·kq) · ((qs·scale) · (ks·scale))     (the a8 dequant of ①QK — ADR 0030)
 *   tm     = max over the tile's keys;  m' = max(m, tm);  α = exp(m − m')
 *   l      = l·α + Σ exp(S − m')                           (per-lane partial, reduced once at the end)
 *   P̃      = round(127 · exp(S − m'))                       (the lattice of ③PV — fixed 1/127)
 *   O      = O·α + f32(Σ_n P̃·vq)                            (f32 accumulator, integer products per tile)
 * at the end: O · (vs[d] · (1/127) / l)
 * ```
 *
 * ## Why these layouts (they are what makes the kernel fast — ADR 0132 decision 3, addendum 2026-10-25)
 *
 * 256 threads; 64 query rows per workgroup; 64 keys per tile. Thread `(tx, ty)` (`tx` in 0..15,
 * `ty` in 0..15) owns rows `ty·4..+3`; in the score phase it owns the keys `{tx, tx+16, tx+32,
 * tx+48}` of the tile, in the PV phase the output columns `{64·g + 4·tx + c}` (column group `g`,
 * component `c`). All four tiles are arrays of `vec4<u32>`, so every shared-memory access of the
 * two dot-product loops is one 16-byte load. Measured against the first layout (key `v1`) on the
 * RTX 3080 Ti: self-attention ×1.44–1.46, cross-attention ×2.0, bit-identical (ADR 0132 addendum
 * of 2026-10-25):
 *
 * - the `q` and `k` tiles hold a row as `D/16` vec4 words padded by one (`qkStride`), so the 16
 *   lanes of a row read 16 different 16-byte bank groups;
 * - the Vᵀ tile is `[pack][d/4]`: vec4 `(w, d4)` holds the words of the columns `4·d4..+3`, and
 *   word `w` of a column packs the keys `{w, w+16, w+32, w+48}` (gathered byte-wise from the
 *   quantised Vᵀ, whose words pack four consecutive keys). It is filled by **owner** threads: a
 *   thread owns one whole vec4 `(pk, d4)`, consecutive lanes take consecutive `pk` of the same
 *   columns (so a half-warp's four byte loads of a column cover one 64-byte row segment), and the vec4 is
 *   stored at once. The row stride `D/4 + 1` breaks the conflicts of those stores;
 * - the P̃ tile is **transposed**: vec4 `(w, ty)` holds word `w` of the four rows of `ty`, one
 *   whole-vector store per thread; the PV loop reads one vec4 per key word and takes component
 *   `r` for row `r`;
 * - the PV phase runs once per 64-column group, so only one group's 16 integer accumulators are
 *   live at a time (the register margin below the 128-register occupancy step — K-84).
 *
 * MUST: never write a component of a workgroup vector through a dynamic index (`vt[i][d % 4u] =
 * …`): naga/Metal drops such writes silently (the MUST of gemm.ts `storeBTransposed`). Every
 * workgroup store here is a whole vec4 or a whole scalar.
 *
 * ## The two reduction forms are bit-identical (MUST)
 *
 * The row max is exact under any order. The final sum of `l` across the 16 lanes of a row uses
 * the same pairing in both forms — the butterfly `((l0+l8)+(l4+l12)) + ((l2+l10)+(l6+l14)) +
 * ((l1+l9)+(l5+l13)) + ((l3+l11)+(l7+l15))` that `subgroupShuffleXor` with masks 8, 4, 2, 1
 * produces in every lane (f32 addition is commutative, so every lane holds the same bits), spelled
 * out explicitly in the workgroup-memory form. Hence the build may pick `sg32` or `wg` from the
 * device's capabilities without changing a bit (the same discipline as `idot` against its
 * emulation); tests/gpu_attention_online_i8a8_test.ts holds the Uint32 equality.
 * MUST: `sg32` addresses its tiles by `local_invocation_index` and assumes the 16 lanes of a row
 * (`t / 16`) share a subgroup — the linear lane mapping (lane = t % 32 at `@subgroup_size(32)`).
 * WGSL defines no such relationship, so the mapping is **measured** on the device by `acquireGpu`
 * (`GpuContext.subgroupLanesLinear`) and the route takes `sg32` only where it holds. Identifying
 * the lanes by `subgroup_invocation_id` / `subgroup_id` instead was measured 16% slower (the tile
 * addresses no longer derive from the thread index — 2026-10-23, RTX 3080 Ti).
 *
 * ## Numerics (contract class Q — ADR 0110)
 *
 * The weight of key `j` in the output is `P̃_j / 127 · exp(m_t − m_final)` against the exact
 * `exp(S_j − m_final)`; the later rescales only shrink the lattice step, so each weight is off by
 * at most `0.5 / 127`, and `|ΔO[d]| ≤ (0.5/127) · Σ_j |ṽ_j[d]| / l` plus f32 accumulation. The
 * identity of keys beyond N is `NEG` (−3.0e38), not −inf: with −inf the first tile's `α =
 * exp(m − m')` would be `exp(−inf − (−inf)) = NaN`.
 * The layouts above change which thread computes which element, never the value: the integer sums
 * are exact (at most 128·127·127 per tile, far below 2³¹), and every f32 operation keeps its
 * operands and order. So `v2` is bit-identical to `v1` (checked at the switch, 2026-10-25).
 * MUST: `exp` is a per-component scalar expression (vector transcendentals are
 * implementation-dependent). MUST: the dequant folds `(qs·scale)·(ks·scale)` into one f32 first
 * (the order of ①QK); the output folds `O · vs[d]` first and multiplies by `(1/127) / l` after.
 * MUST: `1/127` is a multiplication by the constant (division is allowed 2.5 ULP).
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";
import { I8A8_PACK } from "./i8a8-geometry.ts";
import { i8a8IdotWgsl } from "./linear-i8a8.ts";
import { workgroupVariableBytes } from "./workgroup-storage.ts";

/** Query rows per workgroup. */
export const ATTENTION_ONLINE_ROWS = 64;
/** Keys per tile. */
export const ATTENTION_ONLINE_KEYS = 64;
const THREADS = 256;
/** Lanes per row (the score phase spreads a tile's 64 keys over 16 lanes × 4 keys). */
const LANES = 16;
const ROWS_PER_THREAD = ATTENTION_ONLINE_ROWS / LANES;
/** Row groups of a workgroup (`ty`). */
const ROW_GROUPS = THREADS / LANES;
/** Words per vec4. */
const VEC = 4;
/** vec4 words per q / k row (D / 16). */
const qkQuads = (depth: number): number => depth / (I8A8_PACK * VEC);
/** Padded row stride of the `q` / `k` tiles (vec4) — one quad of padding breaks the bank conflicts. */
const qkStride = (depth: number): number => qkQuads(depth) + 1;
/** Words per Vᵀ column in a tile (64 keys / 4) — also the P̃ words of a row. */
const V_WORDS = ATTENTION_ONLINE_KEYS / I8A8_PACK;
/** Padded row stride of the Vᵀ tile (vec4): D/4 column quads plus one. */
const vtStride = (depth: number): number => depth / VEC + 1;
/** Padded row stride of the transposed P̃ tile (vec4): one per row group plus one. */
const P_STRIDE = ROW_GROUPS + 1;
/** Output columns of one PV column group (16 lanes × 4 columns). */
const GROUP_COLUMNS = LANES * VEC;
const P_LATTICE_MAX = 127;
/** The identity for keys beyond N (not −inf — see the module doc). */
const NEG = "-3.0e38";

/** The row-reduction form: `sg32` = `subgroupShuffleXor` across the 16 lanes of a row; `wg` = workgroup memory. */
export type AttentionOnlineReduce = "sg32" | "wg";
export const ATTENTION_ONLINE_REDUCES: readonly AttentionOnlineReduce[] = ["sg32", "wg"];

/**
 * The largest head depth the generator takes. The `q`, `k` and Vᵀ tiles grow with D (at D = 256
 * they need 58 KiB, above the 48 KiB of Dawn / Vulkan); a deeper head needs a tiling without the
 * `q` tile, which is not built.
 */
export const ATTENTION_ONLINE_MAX_DEPTH = 128;

/**
 * Whether the online kernel can run the shape: the PV phase hands each lane `D / 16` columns in
 * vec4 groups (so `D % 64 == 0`), the accumulators are bounded by {@link ATTENTION_ONLINE_MAX_DEPTH},
 * and there is at least one key (the clamped `ks` reads and `1 / l` need one). Any key length
 * `N ≥ 1` runs: `N % 4 != 0` takes the {@link AttentionOnlineKeyWords} `padded` form (ADR 0132
 * addendum 2026-10-27). Shapes outside fall back to the three dispatches (ADR 0058 decision 3 — the
 * route shows it).
 * NOTE: `D % 64 == 0` also makes every binding the kernel reads as `vec4` (q, k, O, vs) a whole
 * number of 16-byte elements, and the plan places offsets at multiples of
 * `minStorageBufferOffsetAlignment` — the two facts the vec4 bindings rely on.
 * NOTE: ③PV's i8a8 kernel keeps `N % 4 == 0` (`attentionPvI8a8Eligible`); only this kernel learned
 * the key valid length, so the three-dispatch routes (Anima's a8 + s16) do not move.
 */
export const attentionOnlineI8a8Eligible = (depth: number, cols: number): boolean =>
  depth % 64 === 0 && depth <= ATTENTION_ONLINE_MAX_DEPTH && cols >= 1;

/**
 * How the quantised Vᵀ packs the key axis (ADR 0132 addendum 2026-10-27). `whole`: `N % 4 == 0`,
 * each row of Vᵀ is `N / 4` words of four keys. `padded`: `N % 4 != 0`, each row is `ceil(N / 4)`
 * words and the bytes of the keys `N..` in the last word are zero (the producer's padding —
 * `quantizeRowsPaddedWgsl`). The kernel never weighs those bytes: keys `≥ N` take the score `NEG`
 * (decision 4), so their P̃ is `round(127 · exp(NEG − m)) = 0` and their products with any byte are
 * 0. The kernel's only difference is the row length `n4` of its Vᵀ reads.
 *
 * MUST: the kernel ({@link attentionOnlineI8a8Wgsl} and its uniform
 * {@link attentionOnlineI8a8Params}), the Vᵀ producer (recipe-builders/attention.ts
 * `attentionValueQuantization`) and the size of the quantised Vᵀ (`attentionI8a8TempBytes`, through
 * {@link attentionOnlineValueRowWords}) take the form from this one function — a row length two of
 * them disagree on reads the wrong words without an exception.
 * MUST: `whole` keeps the key and the WGSL of the kernel before the addendum byte for byte, so the
 * outputs of every `N % 4 == 0` shape (every accepted Wan2.1 / Wan2.2 size) are unchanged by
 * construction.
 */
export type AttentionOnlineKeyWords = "whole" | "padded";

export const attentionOnlineKeyWords = (cols: number): AttentionOnlineKeyWords =>
  cols % I8A8_PACK === 0 ? "whole" : "padded";

/**
 * Words per row of the quantised Vᵀ — the row length the kernel's `n4` reads: `N / 4` in the
 * {@link attentionOnlineKeyWords} `whole` form and `ceil(N / 4)` in the `padded` one, so one
 * expression serves both.
 */
export const attentionOnlineValueRowWords = (cols: number): number => Math.ceil(cols / I8A8_PACK);

/** Bytes of workgroup storage the kernel declares (what the device compares with its limit). */
export const attentionOnlineI8a8WorkgroupStorageBytes = (
  depth: number,
  reduce: AttentionOnlineReduce,
): number =>
  workgroupVariableBytes(ATTENTION_ONLINE_ROWS * qkStride(depth) * 16) +
  workgroupVariableBytes(ATTENTION_ONLINE_KEYS * qkStride(depth) * 16) +
  workgroupVariableBytes(V_WORDS * vtStride(depth) * 16) +
  workgroupVariableBytes(V_WORDS * P_STRIDE * 16) +
  (reduce === "wg" ? workgroupVariableBytes(ATTENTION_ONLINE_ROWS * LANES * 4) : 0);

/**
 * Where the kernel writes O. `bhsd` = head-major `[B·H, M, D]`, the attention node's own output.
 * `bshd` = token-major `[B, M, H, D]`: the attention absorbed the permute that follows it, so the
 * output projection reads O as stored (perf-ledger F-3 A4). Only the store address differs (each
 * row is still 4·D contiguous bytes), so the values are bit-identical to "store → permute".
 */
export type AttentionOnlineOutput = "bhsd" | "bshd";

/** The key adds `:kpad` for the {@link AttentionOnlineKeyWords} `padded` form only (`whole` keys are unchanged). */
export const attentionOnlineI8a8Key = (
  dp4a: boolean,
  depth: number,
  reduce: AttentionOnlineReduce,
  output: AttentionOnlineOutput = "bhsd",
  keyWords: AttentionOnlineKeyWords = "whole",
): string =>
  `attention_online:v2:i8a8:d${depth}:r${ATTENTION_ONLINE_ROWS}k${ATTENTION_ONLINE_KEYS}:${
    dp4a ? "dp4a" : "emu"
  }:${reduce}${output === "bshd" ? ":obshd" : ""}${keyWords === "padded" ? ":kpad" : ""}`;

/**
 * The uniform: `m` = query rows of the whole node, `n` = keys, `row_offset` = the first row of this
 * dispatch's window (the dispatch covers `ceil(windowRows / 64)` workgroups along x and B·H along
 * y), `scale` = the half scale (`√scale_factor` of ADR 0023, applied to both q and k). With a token-major
 * output ({@link AttentionOnlineOutput} `bshd`), `heads` = H follows in a second 16-byte block.
 * `keyWords` is the key-axis form the pipeline was compiled with: the uniform's `n` must be a key
 * length of that form, since a `whole` kernel given `N % 4 != 0` (or a `padded` one given
 * `N % 4 == 0`) reads Vᵀ at the wrong row length without an exception — so a mismatch throws here.
 */
export const attentionOnlineI8a8Params = (
  m: number,
  n: number,
  rowOffset: number,
  scale: number,
  keyWords: AttentionOnlineKeyWords,
  heads?: number,
): Uint32Array<ArrayBuffer> => {
  assertU32Params("attention_online i8a8", { m, n, row_offset: rowOffset, heads: heads ?? 1 });
  if (!Number.isFinite(scale)) {
    throw new CodegenError(`attention_online i8a8: scale must be finite (${scale})`);
  }
  if (keyWords !== attentionOnlineKeyWords(n)) {
    throw new CodegenError(
      `attention_online i8a8: N = ${n} takes the '${attentionOnlineKeyWords(n)}' key axis, but the pipeline was compiled for '${keyWords}'`,
    );
  }
  const words = new Uint32Array(heads === undefined ? 4 : 8);
  words[0] = m;
  words[1] = n;
  words[2] = rowOffset;
  new Float32Array(words.buffer, 12, 1)[0] = scale;
  if (heads !== undefined) words[4] = heads;
  return words;
};

const assertDepth = (depth: number): void => {
  if (!attentionOnlineI8a8Eligible(depth, I8A8_PACK)) {
    throw new CodegenError(
      `attention_online i8a8: depth ${depth} is not eligible (D % 64 == 0 and D ≤ ${ATTENTION_ONLINE_MAX_DEPTH})`,
    );
  }
};

const range = (length: number): number[] => Array.from({ length }, (_, i) => i);
const rows = range(ROWS_PER_THREAD);
const lanes = ["x", "y", "z", "w"] as const;
/** The four keys of a lane in the score phase: `tx + 16·j`. */
const keySlots = range(4);
/** Additive index offset (0 is omitted — keeps the output readable). */
const plus = (offset: number): string => (offset === 0 ? "" : `${offset}u + `);

/** The 16-lane butterfly sum of `l` in its two spellings (the same pairing — module doc). */
const finalRowSum = (reduce: AttentionOnlineReduce): string =>
  reduce === "sg32"
    ? rows
        .map(
          (r) =>
            `  l${r} = l${r} + subgroupShuffleXor(l${r}, 8u);
  l${r} = l${r} + subgroupShuffleXor(l${r}, 4u);
  l${r} = l${r} + subgroupShuffleXor(l${r}, 2u);
  l${r} = l${r} + subgroupShuffleXor(l${r}, 1u);`,
        )
        .join("\n")
    : `${rows.map((r) => `  red[(ty * ${ROWS_PER_THREAD}u + ${r}u) * ${LANES}u + tx] = l${r};`).join("\n")}
  workgroupBarrier();
${rows
  .map(
    (r) => `  {
    let b = (ty * ${ROWS_PER_THREAD}u + ${r}u) * ${LANES}u;
    // The butterfly pairing of the sg32 form, spelled out: a_i = l_i + l_{i+8}, b_i = a_i + a_{i+4},
    // c_i = b_i + b_{i+2}, l = c_0 + c_1 (MUST: the same pairing, or the two forms diverge in bits)
    let a0 = red[b] + red[b + 8u];   let a1 = red[b + 1u] + red[b + 9u];
    let a2 = red[b + 2u] + red[b + 10u]; let a3 = red[b + 3u] + red[b + 11u];
    let a4 = red[b + 4u] + red[b + 12u]; let a5 = red[b + 5u] + red[b + 13u];
    let a6 = red[b + 6u] + red[b + 14u]; let a7 = red[b + 7u] + red[b + 15u];
    let b0 = a0 + a4; let b1 = a1 + a5; let b2 = a2 + a6; let b3 = a3 + a7;
    let c0 = b0 + b2; let c1 = b1 + b3;
    l${r} = c0 + c1;
  }`,
  )
  .join("\n")}`;

/** The tile's row max across the 16 lanes (exact under any order). */
const tileRowMax = (reduce: AttentionOnlineReduce): string =>
  reduce === "sg32"
    ? rows
        .map(
          (r) =>
            `    tm${r} = max(tm${r}, subgroupShuffleXor(tm${r}, 8u));
    tm${r} = max(tm${r}, subgroupShuffleXor(tm${r}, 4u));
    tm${r} = max(tm${r}, subgroupShuffleXor(tm${r}, 2u));
    tm${r} = max(tm${r}, subgroupShuffleXor(tm${r}, 1u));`,
        )
        .join("\n")
    : `${rows.map((r) => `    red[(ty * ${ROWS_PER_THREAD}u + ${r}u) * ${LANES}u + tx] = tm${r};`).join("\n")}
    workgroupBarrier();
${rows
  .map(
    (r) => `    {
      let b = (ty * ${ROWS_PER_THREAD}u + ${r}u) * ${LANES}u;
      var mx = red[b];
      for (var i = 1u; i < ${LANES}u; i = i + 1u) { mx = max(mx, red[b + i]); }
      tm${r} = mx;
    }`,
  )
  .join("\n")}`;

/**
 * Word `c` of a Vᵀ owner vec4: the bytes of the keys `kg + 16·i` (i = 0..3) of column `d4·4 + c`,
 * each in the global word `kg/4 + 4·i` at byte `kg % 4` (the tile start is a multiple of 64, so
 * the byte is `pk % 4`). The guards keep the reads of a partial last tile inside the row.
 */
const vtOwnerWord = (c: number): string =>
  `      var vw${c} = 0u;
      { let base = v_bh + (d4 * ${VEC}u + ${c}u) * n4 + gw; ${range(4)
        .map(
          (i) =>
            `if (gw + ${4 * i}u < n4) { vw${c} = vw${c} | (((vq[base + ${4 * i}u] >> shift) & 0xffu) << ${8 * i}u); }`,
        )
        .join(" ")} }`;

export const attentionOnlineI8a8Wgsl = (
  dp4a: boolean,
  depth: number,
  reduce: AttentionOnlineReduce,
  output: AttentionOnlineOutput = "bhsd",
  keyWords: AttentionOnlineKeyWords = "whole",
): string => {
  const bshd = output === "bshd";
  const padded = keyWords === "padded";
  assertDepth(depth);
  const quads = qkQuads(depth);
  const qkS = qkStride(depth);
  const vtS = vtStride(depth);
  /** vec4 column quads per output row (D/4). */
  const columnQuads = depth / VEC;
  /** PV column groups (64 columns each). */
  const groupIds = range(depth / GROUP_COLUMNS);
  /** Fill slots per thread and tile: K vec4 words, Vᵀ vec4 words. */
  const kSlots = range((ATTENTION_ONLINE_KEYS * quads) / THREADS);
  const vSlots = range((V_WORDS * columnQuads) / THREADS);
  const sg = reduce === "sg32";
  const header = sg
    ? `enable subgroups, subgroup_size_control;
diagnostic(off, subgroup_uniformity);
`
    : "";
  return `${header}// karume attention_online (O = softmax((q·scale)(k·scale)ᵀ) · v with S never stored — ADR 0132; q / k per-token i8, v per-column i8, P̃ = round(127·exp(S − m)) against the running max${
    dp4a ? "" : " (emulated)"
  }, D = ${depth}, 64 rows × 64-key tiles, 256 threads, vec4 tiles, ${sg ? "subgroup shuffle" : "workgroup memory"} reductions${
    padded ? ", Vᵀ rows of ceil(N/4) words" : ""
  })
struct Dims {
  m: u32,
  n: u32,
  row_offset: u32,
  scale: f32,${bshd ? "\n  heads: u32," : ""}
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> qq: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> kq: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> vq: array<u32>;
@group(0) @binding(4) var<storage, read_write> o: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> qs: array<f32>;
@group(0) @binding(6) var<storage, read> ks: array<f32>;
@group(0) @binding(7) var<storage, read> vs: array<vec4<f32>>;
${i8a8IdotWgsl(dp4a)}
// q tile: 64 rows × ${quads} vec4 words, padded to ${qkS} per row (loaded once per workgroup)
var<workgroup> qt: array<vec4<u32>, ${ATTENTION_ONLINE_ROWS * qkS}>;
// k tile: 64 keys × ${quads} vec4 words, padded to ${qkS} per key
var<workgroup> kt: array<vec4<u32>, ${ATTENTION_ONLINE_KEYS * qkS}>;
// Vᵀ tile: [pack 0..15][column quad 0..${columnQuads - 1}], padded to ${vtS} per pack; word w of a column packs the keys {w, w+16, w+32, w+48}
var<workgroup> vt: array<vec4<u32>, ${V_WORDS * vtS}>;
// P̃ tile, transposed: [word 0..15][row group 0..15] = that word of the rows ty·4..+3, padded to ${P_STRIDE} per word
var<workgroup> pt: array<vec4<u32>, ${V_WORDS * P_STRIDE}>;
${reduce === "wg" ? `var<workgroup> red: array<f32, ${ATTENTION_ONLINE_ROWS * LANES}>;` : ""}
const INV_P: f32 = ${1 / P_LATTICE_MAX};
const NEG: f32 = ${NEG};
@compute @workgroup_size(${THREADS})${sg ? " @subgroup_size(32)" : ""}
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) t: u32,
) {
  let bh = wid.y;
  let row0 = dims.row_offset + wid.x * ${ATTENTION_ONLINE_ROWS}u;
  // the 16 lanes of a row: ${sg ? "one half of a subgroup under the measured linear lane mapping" : "reduced through workgroup memory"}
  let tx = t % ${LANES}u;
  let ty = t / ${LANES}u;
${
  padded
    ? `  // Vᵀ rows hold ceil(N/4) words: the producer zero-pads the last one, and its keys from N on
  // score NEG, so their P̃ is 0 and the padding never weighs in
  let n4 = (dims.n + ${I8A8_PACK - 1}u) / ${I8A8_PACK}u;`
    : `  let n4 = dims.n / ${I8A8_PACK}u;`
}
  // this head's offsets: q / k in vec4 words, Vᵀ in words, the k scales in elements
  let q_bh = bh * dims.m * ${quads}u;
  let k_bh = bh * dims.n * ${quads}u;
  let v_bh = bh * ${depth}u * n4;
  let ks_bh = bh * dims.n;
${rows
  .map(
    (r) => `  let qrow${r} = row0 + ty * ${ROWS_PER_THREAD}u + ${r}u;
  // the half scale on the q side (the scale of a row beyond M is never used — clamp keeps the read in range)
  let qsc${r} = qs[bh * dims.m + min(qrow${r}, dims.m - 1u)] * dims.scale;`,
  )
  .join("\n")}
  for (var w = t; w < ${ATTENTION_ONLINE_ROWS * quads}u; w = w + ${THREADS}u) {
    let r = w / ${quads}u;
    let pk = w % ${quads}u;
    qt[r * ${qkS}u + pk] = select(vec4<u32>(0u), qq[q_bh + (row0 + r) * ${quads}u + pk], row0 + r < dims.m);
  }
${rows.map((r) => `  var m${r}: f32 = NEG;\n  var l${r}: f32 = 0.0;`).join("\n")}
${rows.map((r) => groupIds.map((g) => `  var o${r}_${g}: vec4<f32> = vec4<f32>(0.0);`).join("\n")).join("\n")}
${kSlots.map((s) => `  var kr${s}: vec4<u32>;`).join("\n")}
${vSlots.map((s) => `  var vr${s}: vec4<u32>;`).join("\n")}
  let tiles = (dims.n + ${ATTENTION_ONLINE_KEYS - 1}u) / ${ATTENTION_ONLINE_KEYS}u;
  for (var tile = 0u; tile < tiles; tile = tile + 1u) {
    let key0 = tile * ${ATTENTION_ONLINE_KEYS}u;
    // every load of the tile into registers first, then every store (the loads overlap)
${kSlots
  .map(
    (s) =>
      `    { let w = t + ${s * THREADS}u; let r = w / ${quads}u; let pk = w % ${quads}u; kr${s} = select(vec4<u32>(0u), kq[k_bh + (key0 + r) * ${quads}u + pk], key0 + r < dims.n); }`,
  )
  .join("\n")}
${vSlots
  .map(
    (
      s,
    ) => `    // Vᵀ owner (pk, d4): consecutive lanes take consecutive keys of the same four columns
    { let w = t + ${s * THREADS}u; let pk = w % ${V_WORDS}u; let d4 = w / ${V_WORDS}u; let kg = key0 + pk; let shift = (kg % ${I8A8_PACK}u) * 8u; let gw = kg / ${I8A8_PACK}u;
${range(VEC).map(vtOwnerWord).join("\n")}
      vr${s} = vec4<u32>(vw0, vw1, vw2, vw3); }`,
  )
  .join("\n")}
${kSlots
  .map(
    (s) =>
      `    { let w = t + ${s * THREADS}u; let r = w / ${quads}u; let pk = w % ${quads}u; kt[r * ${qkS}u + pk] = kr${s}; }`,
  )
  .join("\n")}
${vSlots
  .map(
    (s) =>
      `    { let w = t + ${s * THREADS}u; let pk = w % ${V_WORDS}u; let d4 = w / ${V_WORDS}u; vt[pk * ${vtS}u + d4] = vr${s}; }`,
  )
  .join("\n")}
    workgroupBarrier();
    // ---- scores: my 4 rows × the keys {tx, tx+16, tx+32, tx+48} ----
${rows.map((r) => `    var acc${r}: vec4<i32> = vec4<i32>(0);`).join("\n")}
    for (var p4 = 0u; p4 < ${quads}u; p4 = p4 + 1u) {
${rows.map((r) => `      let qv${r} = qt[(ty * ${ROWS_PER_THREAD}u + ${r}u) * ${qkS}u + p4];`).join("\n")}
${keySlots.map((j) => `      let kv${j} = kt[(tx + ${16 * j}u) * ${qkS}u + p4];`).join("\n")}
${rows
  .map(
    (r) =>
      `      acc${r} = acc${r} + vec4<i32>(${keySlots
        .map((j) => lanes.map((c) => `idot(qv${r}.${c}, kv${j}.${c})`).join(" + "))
        .join(", ")});`,
  )
  .join("\n")}
    }
    let kidx = key0 + tx;
    let kok = vec4<bool>(kidx < dims.n, kidx + 16u < dims.n, kidx + 32u < dims.n, kidx + 48u < dims.n);
    let ksc = vec4<f32>(
      ks[ks_bh + min(kidx, dims.n - 1u)],
      ks[ks_bh + min(kidx + 16u, dims.n - 1u)],
      ks[ks_bh + min(kidx + 32u, dims.n - 1u)],
      ks[ks_bh + min(kidx + 48u, dims.n - 1u)],
    ) * dims.scale;
${rows
  .map(
    (r) => `    // MUST: fold (qs·scale)·(ks·scale) into one f32 first (the dequant order of ①QK)
    let sv${r} = select(vec4<f32>(NEG), vec4<f32>(acc${r}) * (qsc${r} * ksc), kok);
    var tm${r} = max(max(sv${r}.x, sv${r}.y), max(sv${r}.z, sv${r}.w));`,
  )
  .join("\n")}
${tileRowMax(reduce)}
${rows
  .map(
    (r) => `    let mn${r} = max(m${r}, tm${r});
    let alpha${r} = exp(m${r} - mn${r});
    m${r} = mn${r};
    l${r} = l${r} * alpha${r};
${groupIds.map((g) => `    o${r}_${g} = o${r}_${g} * alpha${r};`).join("\n")}
    // MUST: exp per component (vector transcendentals are implementation-dependent)
    let pe${r} = vec4<f32>(${lanes.map((c) => `exp(sv${r}.${c} - mn${r})`).join(", ")});
    l${r} = l${r} + pe${r}.x + pe${r}.y + pe${r}.z + pe${r}.w;
    let pq${r} = pack4xI8(vec4<i32>(round(pe${r} * ${P_LATTICE_MAX}.0)));`,
  )
  .join("\n")}
    pt[tx * ${P_STRIDE}u + ty] = vec4<u32>(${rows.map((r) => `pq${r}`).join(", ")});
    workgroupBarrier();
    // ---- PV: my 4 rows × the columns {64·g + 4·tx + c}, one column group at a time ----
${groupIds
  .map(
    (g) => `    {
${rows.map((r) => `      var p${r}_${g}: vec4<i32> = vec4<i32>(0);`).join("\n")}
      for (var p4 = 0u; p4 < ${V_WORDS / VEC}u; p4 = p4 + 1u) {
${range(VEC)
  .map(
    (c) => `        {
          let pw = pt[(p4 * ${VEC}u + ${c}u) * ${P_STRIDE}u + ty];
          let vv = vt[(p4 * ${VEC}u + ${c}u) * ${vtS}u + ${plus(g * LANES)}tx];
${rows
  .map(
    (r) =>
      `          p${r}_${g} = p${r}_${g} + vec4<i32>(${lanes.map((v) => `idot(pw.${lanes[r]}, vv.${v})`).join(", ")});`,
  )
  .join("\n")}
        }`,
  )
  .join("\n")}
      }
${rows.map((r) => `      o${r}_${g} = o${r}_${g} + vec4<f32>(p${r}_${g});`).join("\n")}
    }`,
  )
  .join("\n")}
    workgroupBarrier(); // the next tile overwrites kt / vt / pt
  }
${finalRowSum(reduce)}
${groupIds.map((g) => `  let vs${g} = vs[bh * ${columnQuads}u + ${plus(g * LANES)}tx];`).join("\n")}
${bshd ? "  // Token-major O: row qrow of batch b, head h sits at ((b·M + qrow)·H + h)·D\n  let ob_b = bh / dims.heads;\n  let ob_h = bh % dims.heads;\n" : ""}${rows
    .map(
      (r) => `  if (qrow${r} < dims.m) {
    // MUST: 1/127 by multiplication; O · vs[d] first, then the row factor
    let f${r} = INV_P / l${r};
    let ob${r} = ${bshd ? `((ob_b * dims.m + qrow${r}) * dims.heads + ob_h)` : `(bh * dims.m + qrow${r})`} * ${columnQuads}u;
${groupIds.map((g) => `    o[ob${r} + ${plus(g * LANES)}tx] = (o${r}_${g} * vs${g}) * f${r};`).join("\n")}
  }`,
    )
    .join("\n")}
}
`;
};
