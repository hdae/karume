/**
 * **Online int8 attention** (ADR 0132, perf-ledger K-82) — the kernel behind
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
 * ## Why these layouts (they are what makes the kernel faster than the three dispatches)
 *
 * 256 threads; 64 query rows per workgroup; 64 keys per tile. Thread `(tx, ty)` (`tx` in 0..15,
 * `ty` in 0..15) owns rows `ty·4..+3`; in the score phase it owns the keys `{tx, tx+16, tx+32,
 * tx+48}` of the tile, in the PV phase the output columns `{tx + 16·j}`. The spike of 2026-10-23
 * (research 2026-10-23-flash-attention-webgpu-recon §10.2) measured the natural layouts at half
 * the speed of the three dispatches — the 16 lanes of one row hit one shared-memory bank — and
 * these at ×1.46 against them at S = 27,280:
 *
 * - the `k` and `q` tiles pad each row to 33 words (`KEY_STRIDE`), so the 16 lanes of a row read
 *   16 different banks;
 * - the Vᵀ tile is stored `[pack][d]`: word `w` of a column packs the keys `{w, w+16, w+32, w+48}`
 *   (gathered byte-wise from the quantised Vᵀ, whose words pack four consecutive keys), so that a
 *   lane's P̃ word pairs with it in one `dot4I8Packed` and the 16 lanes read consecutive words;
 * - the P̃ tile pads each row to 17 words (`P_STRIDE`).
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
/** Padded row stride of the `q` / `k` tiles (words) — `D/4 + 1` breaks the bank conflicts. */
const keyStride = (depth: number): number => depth / I8A8_PACK + 1;
/** Padded row stride of the P̃ tile (words). */
const P_STRIDE = ATTENTION_ONLINE_KEYS / I8A8_PACK + 1;
/** Words per Vᵀ column in a tile (64 keys / 4). */
const V_WORDS = ATTENTION_ONLINE_KEYS / I8A8_PACK;
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
 * and P̃ / Vᵀ pack four keys per word (`N % 4 == 0`, the ③PV condition). Shapes outside fall back
 * to the three dispatches (ADR 0058 decision 3 — the route shows it).
 */
export const attentionOnlineI8a8Eligible = (depth: number, cols: number): boolean =>
  depth % 64 === 0 && depth <= ATTENTION_ONLINE_MAX_DEPTH && cols % I8A8_PACK === 0;

/** Bytes of workgroup storage the kernel declares (what the device compares with its limit). */
export const attentionOnlineI8a8WorkgroupStorageBytes = (
  depth: number,
  reduce: AttentionOnlineReduce,
): number =>
  workgroupVariableBytes(ATTENTION_ONLINE_ROWS * keyStride(depth) * 4) +
  workgroupVariableBytes(ATTENTION_ONLINE_KEYS * keyStride(depth) * 4) +
  workgroupVariableBytes(V_WORDS * depth * 4) +
  workgroupVariableBytes(ATTENTION_ONLINE_ROWS * P_STRIDE * 4) +
  (reduce === "wg" ? workgroupVariableBytes(ATTENTION_ONLINE_ROWS * LANES * 4) : 0);

export const attentionOnlineI8a8Key = (
  dp4a: boolean,
  depth: number,
  reduce: AttentionOnlineReduce,
): string =>
  `attention_online:v1:i8a8:d${depth}:r${ATTENTION_ONLINE_ROWS}k${ATTENTION_ONLINE_KEYS}:${
    dp4a ? "dp4a" : "emu"
  }:${reduce}`;

/**
 * The uniform: `m` = query rows of the whole node, `n` = keys, `row_offset` = the first row of this
 * dispatch's window (the dispatch covers `ceil(windowRows / 64)` workgroups along x and B·H along
 * y), `scale` = the half scale (`√scale_factor` of ADR 0023, applied to both q and k).
 */
export const attentionOnlineI8a8Params = (
  m: number,
  n: number,
  rowOffset: number,
  scale: number,
): Uint32Array<ArrayBuffer> => {
  assertU32Params("attention_online i8a8", { m, n, row_offset: rowOffset });
  if (!Number.isFinite(scale)) {
    throw new CodegenError(`attention_online i8a8: scale must be finite (${scale})`);
  }
  const words = new Uint32Array(4);
  words[0] = m;
  words[1] = n;
  words[2] = rowOffset;
  new Float32Array(words.buffer, 12, 1)[0] = scale;
  return words;
};

const assertDepth = (depth: number): void => {
  if (!attentionOnlineI8a8Eligible(depth, I8A8_PACK)) {
    throw new CodegenError(
      `attention_online i8a8: depth ${depth} is not eligible (D % 64 == 0 and D ≤ ${ATTENTION_ONLINE_MAX_DEPTH})`,
    );
  }
};

const rows = Array.from({ length: ROWS_PER_THREAD }, (_, r) => r);
const lanes = ["x", "y", "z", "w"] as const;

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

export const attentionOnlineI8a8Wgsl = (
  dp4a: boolean,
  depth: number,
  reduce: AttentionOnlineReduce,
): string => {
  assertDepth(depth);
  const packs = depth / I8A8_PACK;
  const stride = keyStride(depth);
  /** Output columns per lane and their vec4 groups. */
  const columns = depth / LANES;
  const groups = columns / I8A8_PACK;
  const groupIds = Array.from({ length: groups }, (_, g) => g);
  const column = (g: number, c: number): string =>
    `tx${g * 16 * I8A8_PACK + c * 16 === 0 ? "" : ` + ${(g * I8A8_PACK + c) * 16}u`}`;
  const sg = reduce === "sg32";
  const header = sg
    ? `enable subgroups, subgroup_size_control;
diagnostic(off, subgroup_uniformity);
`
    : "";
  return `${header}// karume attention_online (O = softmax((q·scale)(k·scale)ᵀ) · v with S never stored — ADR 0132; q / k per-token i8, v per-column i8, P̃ = round(127·exp(S − m)) against the running max${
    dp4a ? "" : " (emulated)"
  }, D = ${depth}, 64 rows × 64-key tiles, 256 threads, ${sg ? "subgroup shuffle" : "workgroup memory"} reductions)
struct Dims {
  m: u32,
  n: u32,
  row_offset: u32,
  scale: f32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> qq: array<u32>;
@group(0) @binding(2) var<storage, read> kq: array<u32>;
@group(0) @binding(3) var<storage, read> vq: array<u32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
@group(0) @binding(5) var<storage, read> qs: array<f32>;
@group(0) @binding(6) var<storage, read> ks: array<f32>;
@group(0) @binding(7) var<storage, read> vs: array<f32>;
${i8a8IdotWgsl(dp4a)}
// q tile: 64 rows × ${packs} words, padded to ${stride} per row (loaded once per workgroup)
var<workgroup> qt: array<u32, ${ATTENTION_ONLINE_ROWS * stride}>;
// k tile: 64 keys × ${packs} words, padded to ${stride} per key
var<workgroup> kt: array<u32, ${ATTENTION_ONLINE_KEYS * stride}>;
// Vᵀ tile: [pack 0..15][d 0..${depth - 1}]; word w of a column packs the keys {w, w+16, w+32, w+48}
var<workgroup> vt: array<u32, ${V_WORDS * depth}>;
// P̃ tile: 64 rows × 16 words, padded to ${P_STRIDE} per row
var<workgroup> pt: array<u32, ${ATTENTION_ONLINE_ROWS * P_STRIDE}>;
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
  let n4 = dims.n / ${I8A8_PACK}u;
  let q_bh = bh * dims.m * ${packs}u;
  let k_bh = bh * dims.n * ${packs}u;
  let v_bh = bh * ${depth}u * n4;
${rows
  .map(
    (r) => `  let qrow${r} = row0 + ty * ${ROWS_PER_THREAD}u + ${r}u;
  // the half scale on the q side (the scale of a row beyond M is never used — clamp keeps the read in range)
  let qsc${r} = qs[bh * dims.m + min(qrow${r}, dims.m - 1u)] * dims.scale;`,
  )
  .join("\n")}
  for (var w = t; w < ${ATTENTION_ONLINE_ROWS * packs}u; w = w + ${THREADS}u) {
    let r = w / ${packs}u;
    let pk = w % ${packs}u;
    qt[r * ${stride}u + pk] = select(0u, qq[q_bh + (row0 + r) * ${packs}u + pk], row0 + r < dims.m);
  }
${rows.map((r) => `  var m${r}: f32 = NEG;\n  var l${r}: f32 = 0.0;`).join("\n")}
${rows.map((r) => groupIds.map((g) => `  var o${r}_${g}: vec4<f32> = vec4<f32>(0.0);`).join("\n")).join("\n")}
  let tiles = (dims.n + ${ATTENTION_ONLINE_KEYS - 1}u) / ${ATTENTION_ONLINE_KEYS}u;
  for (var tile = 0u; tile < tiles; tile = tile + 1u) {
    let key0 = tile * ${ATTENTION_ONLINE_KEYS}u;
    for (var w = t; w < ${ATTENTION_ONLINE_KEYS * packs}u; w = w + ${THREADS}u) {
      let r = w / ${packs}u;
      let pk = w % ${packs}u;
      kt[r * ${stride}u + pk] = select(0u, kq[k_bh + (key0 + r) * ${packs}u + pk], key0 + r < dims.n);
    }
    // Vᵀ: word (pk, d) = the bytes of keys key0 + pk + 16·i (i = 0..3), each in the global word
    // (key0 + pk)/4 + 4·i at byte (key0 + pk) % 4 (key0 is a multiple of 64, so the byte is pk % 4)
    for (var w = t; w < ${V_WORDS * depth}u; w = w + ${THREADS}u) {
      let pk = w / ${depth}u;
      let d = w % ${depth}u;
      let kg = key0 + pk;
      let shift = (kg % ${I8A8_PACK}u) * 8u;
      let base = v_bh + d * n4 + kg / ${I8A8_PACK}u;
      var word = 0u;
${[0, 1, 2, 3]
  .map(
    (i) =>
      `      if (kg / ${I8A8_PACK}u + ${4 * i}u < n4) { word = word | (((vq[base + ${4 * i}u] >> shift) & 0xffu) << ${8 * i}u); }`,
  )
  .join("\n")}
      vt[w] = word;
    }
    workgroupBarrier();
    // ---- scores: my 4 rows × the keys {tx, tx+16, tx+32, tx+48} ----
${rows.map((r) => `    var acc${r}: vec4<i32> = vec4<i32>(0);`).join("\n")}
    for (var pk = 0u; pk < ${packs}u; pk = pk + 1u) {
${rows.map((r) => `      let qw${r} = qt[(ty * ${ROWS_PER_THREAD}u + ${r}u) * ${stride}u + pk];`).join("\n")}
      let kw = vec4<u32>(
        kt[tx * ${stride}u + pk],
        kt[(tx + 16u) * ${stride}u + pk],
        kt[(tx + 32u) * ${stride}u + pk],
        kt[(tx + 48u) * ${stride}u + pk],
      );
${rows
  .map(
    (r) =>
      `      acc${r} = acc${r} + vec4<i32>(idot(qw${r}, kw.x), idot(qw${r}, kw.y), idot(qw${r}, kw.z), idot(qw${r}, kw.w));`,
  )
  .join("\n")}
    }
    let kidx = key0 + tx;
    let kok = vec4<bool>(kidx < dims.n, kidx + 16u < dims.n, kidx + 32u < dims.n, kidx + 48u < dims.n);
    let ksc = vec4<f32>(
      ks[k_bh / ${packs}u + min(kidx, dims.n - 1u)],
      ks[k_bh / ${packs}u + min(kidx + 16u, dims.n - 1u)],
      ks[k_bh / ${packs}u + min(kidx + 32u, dims.n - 1u)],
      ks[k_bh / ${packs}u + min(kidx + 48u, dims.n - 1u)],
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
    pt[(ty * ${ROWS_PER_THREAD}u + ${r}u) * ${P_STRIDE}u + tx] = pack4xI8(vec4<i32>(round(pe${r} * ${P_LATTICE_MAX}.0)));`,
  )
  .join("\n")}
    workgroupBarrier();
    // ---- PV: my 4 rows × the columns {tx + 16·j} ----
${rows.map((r) => groupIds.map((g) => `    var p${r}_${g}: vec4<i32> = vec4<i32>(0);`).join("\n")).join("\n")}
    for (var pk = 0u; pk < ${V_WORDS}u; pk = pk + 1u) {
${rows.map((r) => `      let pw${r} = pt[(ty * ${ROWS_PER_THREAD}u + ${r}u) * ${P_STRIDE}u + pk];`).join("\n")}
      let vb = pk * ${depth}u;
${groupIds
  .map(
    (g) =>
      `      let vw${g} = vec4<u32>(${[0, 1, 2, 3].map((c) => `vt[vb + ${column(g, c)}]`).join(", ")});`,
  )
  .join("\n")}
${rows
  .map((r) =>
    groupIds
      .map(
        (g) =>
          `      p${r}_${g} = p${r}_${g} + vec4<i32>(idot(pw${r}, vw${g}.x), idot(pw${r}, vw${g}.y), idot(pw${r}, vw${g}.z), idot(pw${r}, vw${g}.w));`,
      )
      .join("\n"),
  )
  .join("\n")}
    }
${rows.map((r) => groupIds.map((g) => `    o${r}_${g} = o${r}_${g} + vec4<f32>(p${r}_${g});`).join("\n")).join("\n")}
    workgroupBarrier(); // the next tile overwrites kt / vt / pt
  }
${finalRowSum(reduce)}
${groupIds
  .map(
    (g) =>
      `  let vs${g} = vec4<f32>(${[0, 1, 2, 3].map((c) => `vs[bh * ${depth}u + ${column(g, c)}]`).join(", ")});`,
  )
  .join("\n")}
${rows
  .map(
    (r) => `  if (qrow${r} < dims.m) {
    // MUST: 1/127 by multiplication; O · vs[d] first, then the row factor
    let f${r} = INV_P / l${r};
    let ob${r} = (bh * dims.m + qrow${r}) * ${depth}u;
${groupIds
  .map(
    (g) => `    let ov${r}_${g} = (o${r}_${g} * vs${g}) * f${r};
${[0, 1, 2, 3].map((c) => `    o[ob${r} + ${column(g, c)}] = ov${r}_${g}.${lanes[c]};`).join("\n")}`,
  )
  .join("\n")}
  }`,
  )
  .join("\n")}
}
`;
};
