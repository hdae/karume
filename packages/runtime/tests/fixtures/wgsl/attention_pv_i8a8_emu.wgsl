// karume attention_pv (O[b,m,d] = P[b,m,n] · v[b,n,d], P̃ = round(127·exp(S − m)) not materialised (emulated), v is per-column i8 (Vᵀ contiguous), tile 64x128 / 8x8 per thread / wg 16x8 / K 16)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
// Eligibility requires N % 4 == 0, so a row of S always starts on a quad boundary (the same step
// as P̃'s packing of 4)
@group(0) @binding(1) var<storage, read> s: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vq: array<u32>;
@group(0) @binding(3) var<storage, read> stats: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
@group(0) @binding(5) var<storage, read> vscale: array<f32>;

// emulation of the integer dot product (core WGSL only); returns **the same integer** as dot4I8Packed —
// integer addition is associative and exact, and the intermediates stay at most 4·127·128 = 65,024, so nothing wraps
fn idot(a: u32, b: u32) -> i32 {
  return dot(unpack4xI8(a), unpack4xI8(b));
}

// Shared tiles are laid out **[pack][row] / [pack][column]** (as in ①QK / linear's i8a8 — the inner
// reads get word stride 4 and bank conflicts drop from 8-way to 2-way)
var<workgroup> sa: array<u32, 256>;
var<workgroup> sb: array<u32, 512>;

@compute @workgroup_size(16, 8)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 16u + lid.x;
  // The reduction axis N counts in quads on the S side and in packs on the vq side (the same step
  // of 4 elements)
  let k4 = dims.k / 4u;
  // The batch axis folding B and H (uniform per workgroup — a precondition for the inner
  // workgroupBarrier to meet WGSL's uniformity requirement). MUST: S counts in quads, vq in packs,
  // the row statistics in 2 words per row, vscale in D entries, O in elements
  // — mixing up the units reads and writes the next head when B·H ≥ 2 (a wrong value without an
  // exception, invisible with B=H=1)
  let sbase = wid.z * dims.m * k4;
  let vbase = wid.z * dims.n * k4;
  let rbase = wid.z * dims.m;
  let vsbase = wid.z * dims.n;
  let obase = wid.z * dims.m * dims.n;
  // P̃ tile assignment (64 rows × 4 packs over 128 threads in 2 rounds)
  let ar = tid / 4u;
  let ap = tid % 4u;
  let sa_at = ap * 64u + ar;
  let arow0 = wid.y * 64u + ar;
  let arow_base0 = sbase + arow0 * k4;
  let arow1 = arow0 + 32u;
  let arow_base1 = sbase + arow1 * k4;
  // Row max m (invariant over the K tile loop, so read once). On the edge tile arow >= M can
  // happen, so the index falls back to 0 (the value read is used only in the arow < M branch)
  let stat_at0 = select(0u, (rbase + arow0) * 2u, arow0 < dims.m);
  let row_max0 = stats[stat_at0];
  let stat_at1 = select(0u, (rbase + arow1) * 2u, arow1 < dims.m);
  let row_max1 = stats[stat_at1];
  // B tile assignment (128 columns × 4 packs over 128 threads in 4 rounds).
  // columns are D. vq is [D,N] with N contiguous (= the pack direction), so **no transpose is needed**
  let wc = tid / 4u;
  let wp = tid % 4u;
  let sb_at = wp * 128u + wc;
  let wcol0 = wid.x * 128u + wc;
  let vrow_base0 = vbase + wcol0 * k4;
  let wcol1 = wcol0 + 32u;
  let vrow_base1 = vbase + wcol1 * k4;
  let wcol2 = wcol0 + 64u;
  let vrow_base2 = vbase + wcol2 * k4;
  let wcol3 = wcol0 + 96u;
  let vrow_base3 = vbase + wcol3 * k4;
  // The loop condition is uniform (dims is a uniform buffer) — needed for the inner
  // workgroupBarrier to meet WGSL's uniformity requirement
  let tiles = (k4 + 3u) / 4u;
  var acc0_0 = vec4<i32>(0);
  var acc0_1 = vec4<i32>(0);
  var acc1_0 = vec4<i32>(0);
  var acc1_1 = vec4<i32>(0);
  var acc2_0 = vec4<i32>(0);
  var acc2_1 = vec4<i32>(0);
  var acc3_0 = vec4<i32>(0);
  var acc3_1 = vec4<i32>(0);
  var acc4_0 = vec4<i32>(0);
  var acc4_1 = vec4<i32>(0);
  var acc5_0 = vec4<i32>(0);
  var acc5_1 = vec4<i32>(0);
  var acc6_0 = vec4<i32>(0);
  var acc6_1 = vec4<i32>(0);
  var acc7_0 = vec4<i32>(0);
  var acc7_1 = vec4<i32>(0);
  for (var t = 0u; t < tiles; t = t + 1u) {
    // Zero-fill out of range (qP = 0 adds nothing to the dot product, so the result is exact
    // even with a K remainder)
    let apack = t * 4u + ap;
    var av0 = 0u;
    if (arow0 < dims.m && apack < k4) {
      let raw0 = s[arow_base0 + apack];
      // P̃'s scale is **fixed at 1/127** (the in-row max is exp(0) = 1 by construction — no amax).
      // There is not a single division, so this stays outside WGSL's 2.5 ULP problem
      let p0 = vec4<f32>(
        round(exp(raw0.x - row_max0) * 127.0),
        round(exp(raw0.y - row_max0) * 127.0),
        round(exp(raw0.z - row_max0) * 127.0),
        round(exp(raw0.w - row_max0) * 127.0),
      );
      av0 = pack4xI8(vec4<i32>(p0));
    }
    sa[sa_at] = av0;
    var av1 = 0u;
    if (arow1 < dims.m && apack < k4) {
      let raw1 = s[arow_base1 + apack];
      // P̃'s scale is **fixed at 1/127** (the in-row max is exp(0) = 1 by construction — no amax).
      // There is not a single division, so this stays outside WGSL's 2.5 ULP problem
      let p1 = vec4<f32>(
        round(exp(raw1.x - row_max1) * 127.0),
        round(exp(raw1.y - row_max1) * 127.0),
        round(exp(raw1.z - row_max1) * 127.0),
        round(exp(raw1.w - row_max1) * 127.0),
      );
      av1 = pack4xI8(vec4<i32>(p1));
    }
    sa[sa_at + 32u] = av1;
    let vpack = t * 4u + wp;
    var bv0 = 0u;
    if (wcol0 < dims.n && vpack < k4) {
      bv0 = vq[vrow_base0 + vpack];
    }
    sb[sb_at] = bv0;
    var bv1 = 0u;
    if (wcol1 < dims.n && vpack < k4) {
      bv1 = vq[vrow_base1 + vpack];
    }
    sb[sb_at + 32u] = bv1;
    var bv2 = 0u;
    if (wcol2 < dims.n && vpack < k4) {
      bv2 = vq[vrow_base2 + vpack];
    }
    sb[sb_at + 64u] = bv2;
    var bv3 = 0u;
    if (wcol3 < dims.n && vpack < k4) {
      bv3 = vq[vrow_base3 + vpack];
    }
    sb[sb_at + 96u] = bv3;
    workgroupBarrier();
    // 16 shared loads (8 words of B + 8 words of A) give 64 integer inner products = 256 MACs.
    // The reduction is exact i32 addition, so it **does not depend on order** (unlike the f32
    // skeleton, the addition order is not part of the numeric contract)
    for (var p = 0u; p < 4u; p = p + 1u) {
      let bcol = p * 128u + lid.x * 8u;
      let b0 = sb[bcol];
      let b1 = sb[bcol + 1u];
      let b2 = sb[bcol + 2u];
      let b3 = sb[bcol + 3u];
      let b4 = sb[bcol + 4u];
      let b5 = sb[bcol + 5u];
      let b6 = sb[bcol + 6u];
      let b7 = sb[bcol + 7u];
      let arow_at = p * 64u + lid.y * 8u;
      let a0 = sa[arow_at];
      acc0_0 = acc0_0 + vec4<i32>(idot(a0, b0), idot(a0, b1), idot(a0, b2), idot(a0, b3));
      acc0_1 = acc0_1 + vec4<i32>(idot(a0, b4), idot(a0, b5), idot(a0, b6), idot(a0, b7));
      let a1 = sa[arow_at + 1u];
      acc1_0 = acc1_0 + vec4<i32>(idot(a1, b0), idot(a1, b1), idot(a1, b2), idot(a1, b3));
      acc1_1 = acc1_1 + vec4<i32>(idot(a1, b4), idot(a1, b5), idot(a1, b6), idot(a1, b7));
      let a2 = sa[arow_at + 2u];
      acc2_0 = acc2_0 + vec4<i32>(idot(a2, b0), idot(a2, b1), idot(a2, b2), idot(a2, b3));
      acc2_1 = acc2_1 + vec4<i32>(idot(a2, b4), idot(a2, b5), idot(a2, b6), idot(a2, b7));
      let a3 = sa[arow_at + 3u];
      acc3_0 = acc3_0 + vec4<i32>(idot(a3, b0), idot(a3, b1), idot(a3, b2), idot(a3, b3));
      acc3_1 = acc3_1 + vec4<i32>(idot(a3, b4), idot(a3, b5), idot(a3, b6), idot(a3, b7));
      let a4 = sa[arow_at + 4u];
      acc4_0 = acc4_0 + vec4<i32>(idot(a4, b0), idot(a4, b1), idot(a4, b2), idot(a4, b3));
      acc4_1 = acc4_1 + vec4<i32>(idot(a4, b4), idot(a4, b5), idot(a4, b6), idot(a4, b7));
      let a5 = sa[arow_at + 5u];
      acc5_0 = acc5_0 + vec4<i32>(idot(a5, b0), idot(a5, b1), idot(a5, b2), idot(a5, b3));
      acc5_1 = acc5_1 + vec4<i32>(idot(a5, b4), idot(a5, b5), idot(a5, b6), idot(a5, b7));
      let a6 = sa[arow_at + 6u];
      acc6_0 = acc6_0 + vec4<i32>(idot(a6, b0), idot(a6, b1), idot(a6, b2), idot(a6, b3));
      acc6_1 = acc6_1 + vec4<i32>(idot(a6, b4), idot(a6, b5), idot(a6, b6), idot(a6, b7));
      let a7 = sa[arow_at + 7u];
      acc7_0 = acc7_0 + vec4<i32>(idot(a7, b0), idot(a7, b1), idot(a7, b2), idot(a7, b3));
      acc7_1 = acc7_1 + vec4<i32>(idot(a7, b4), idot(a7, b5), idot(a7, b6), idot(a7, b7));
    }
    workgroupBarrier();
  }
  let ocol = wid.x * 128u + lid.x * 8u;
  let orow0 = wid.y * 64u + lid.y * 8u;
  let orow1 = orow0 + 1u;
  let orow2 = orow0 + 2u;
  let orow3 = orow0 + 3u;
  let orow4 = orow0 + 4u;
  let orow5 = orow0 + 5u;
  let orow6 = orow0 + 6u;
  let orow7 = orow0 + 7u;
  if (orow0 < dims.m) {
    let orow_base = obase + orow0 * dims.n;
    // MUST: fold prow·vs into one f32 first, then multiply f32(acc) by it (no bias, so no fma)
    let prow = stats[(rbase + orow0) * 2u + 1u] * 0.007874015748031496;
    if (ocol < dims.n) {
      o[orow_base + ocol] = f32(acc0_0.x) * (prow * vscale[vsbase + ocol]);
    }
    if (ocol + 1u < dims.n) {
      o[orow_base + ocol + 1u] = f32(acc0_0.y) * (prow * vscale[vsbase + ocol + 1u]);
    }
    if (ocol + 2u < dims.n) {
      o[orow_base + ocol + 2u] = f32(acc0_0.z) * (prow * vscale[vsbase + ocol + 2u]);
    }
    if (ocol + 3u < dims.n) {
      o[orow_base + ocol + 3u] = f32(acc0_0.w) * (prow * vscale[vsbase + ocol + 3u]);
    }
    if (ocol + 4u < dims.n) {
      o[orow_base + ocol + 4u] = f32(acc0_1.x) * (prow * vscale[vsbase + ocol + 4u]);
    }
    if (ocol + 5u < dims.n) {
      o[orow_base + ocol + 5u] = f32(acc0_1.y) * (prow * vscale[vsbase + ocol + 5u]);
    }
    if (ocol + 6u < dims.n) {
      o[orow_base + ocol + 6u] = f32(acc0_1.z) * (prow * vscale[vsbase + ocol + 6u]);
    }
    if (ocol + 7u < dims.n) {
      o[orow_base + ocol + 7u] = f32(acc0_1.w) * (prow * vscale[vsbase + ocol + 7u]);
    }
  }
  if (orow1 < dims.m) {
    let orow_base = obase + orow1 * dims.n;
    // MUST: fold prow·vs into one f32 first, then multiply f32(acc) by it (no bias, so no fma)
    let prow = stats[(rbase + orow1) * 2u + 1u] * 0.007874015748031496;
    if (ocol < dims.n) {
      o[orow_base + ocol] = f32(acc1_0.x) * (prow * vscale[vsbase + ocol]);
    }
    if (ocol + 1u < dims.n) {
      o[orow_base + ocol + 1u] = f32(acc1_0.y) * (prow * vscale[vsbase + ocol + 1u]);
    }
    if (ocol + 2u < dims.n) {
      o[orow_base + ocol + 2u] = f32(acc1_0.z) * (prow * vscale[vsbase + ocol + 2u]);
    }
    if (ocol + 3u < dims.n) {
      o[orow_base + ocol + 3u] = f32(acc1_0.w) * (prow * vscale[vsbase + ocol + 3u]);
    }
    if (ocol + 4u < dims.n) {
      o[orow_base + ocol + 4u] = f32(acc1_1.x) * (prow * vscale[vsbase + ocol + 4u]);
    }
    if (ocol + 5u < dims.n) {
      o[orow_base + ocol + 5u] = f32(acc1_1.y) * (prow * vscale[vsbase + ocol + 5u]);
    }
    if (ocol + 6u < dims.n) {
      o[orow_base + ocol + 6u] = f32(acc1_1.z) * (prow * vscale[vsbase + ocol + 6u]);
    }
    if (ocol + 7u < dims.n) {
      o[orow_base + ocol + 7u] = f32(acc1_1.w) * (prow * vscale[vsbase + ocol + 7u]);
    }
  }
  if (orow2 < dims.m) {
    let orow_base = obase + orow2 * dims.n;
    // MUST: fold prow·vs into one f32 first, then multiply f32(acc) by it (no bias, so no fma)
    let prow = stats[(rbase + orow2) * 2u + 1u] * 0.007874015748031496;
    if (ocol < dims.n) {
      o[orow_base + ocol] = f32(acc2_0.x) * (prow * vscale[vsbase + ocol]);
    }
    if (ocol + 1u < dims.n) {
      o[orow_base + ocol + 1u] = f32(acc2_0.y) * (prow * vscale[vsbase + ocol + 1u]);
    }
    if (ocol + 2u < dims.n) {
      o[orow_base + ocol + 2u] = f32(acc2_0.z) * (prow * vscale[vsbase + ocol + 2u]);
    }
    if (ocol + 3u < dims.n) {
      o[orow_base + ocol + 3u] = f32(acc2_0.w) * (prow * vscale[vsbase + ocol + 3u]);
    }
    if (ocol + 4u < dims.n) {
      o[orow_base + ocol + 4u] = f32(acc2_1.x) * (prow * vscale[vsbase + ocol + 4u]);
    }
    if (ocol + 5u < dims.n) {
      o[orow_base + ocol + 5u] = f32(acc2_1.y) * (prow * vscale[vsbase + ocol + 5u]);
    }
    if (ocol + 6u < dims.n) {
      o[orow_base + ocol + 6u] = f32(acc2_1.z) * (prow * vscale[vsbase + ocol + 6u]);
    }
    if (ocol + 7u < dims.n) {
      o[orow_base + ocol + 7u] = f32(acc2_1.w) * (prow * vscale[vsbase + ocol + 7u]);
    }
  }
  if (orow3 < dims.m) {
    let orow_base = obase + orow3 * dims.n;
    // MUST: fold prow·vs into one f32 first, then multiply f32(acc) by it (no bias, so no fma)
    let prow = stats[(rbase + orow3) * 2u + 1u] * 0.007874015748031496;
    if (ocol < dims.n) {
      o[orow_base + ocol] = f32(acc3_0.x) * (prow * vscale[vsbase + ocol]);
    }
    if (ocol + 1u < dims.n) {
      o[orow_base + ocol + 1u] = f32(acc3_0.y) * (prow * vscale[vsbase + ocol + 1u]);
    }
    if (ocol + 2u < dims.n) {
      o[orow_base + ocol + 2u] = f32(acc3_0.z) * (prow * vscale[vsbase + ocol + 2u]);
    }
    if (ocol + 3u < dims.n) {
      o[orow_base + ocol + 3u] = f32(acc3_0.w) * (prow * vscale[vsbase + ocol + 3u]);
    }
    if (ocol + 4u < dims.n) {
      o[orow_base + ocol + 4u] = f32(acc3_1.x) * (prow * vscale[vsbase + ocol + 4u]);
    }
    if (ocol + 5u < dims.n) {
      o[orow_base + ocol + 5u] = f32(acc3_1.y) * (prow * vscale[vsbase + ocol + 5u]);
    }
    if (ocol + 6u < dims.n) {
      o[orow_base + ocol + 6u] = f32(acc3_1.z) * (prow * vscale[vsbase + ocol + 6u]);
    }
    if (ocol + 7u < dims.n) {
      o[orow_base + ocol + 7u] = f32(acc3_1.w) * (prow * vscale[vsbase + ocol + 7u]);
    }
  }
  if (orow4 < dims.m) {
    let orow_base = obase + orow4 * dims.n;
    // MUST: fold prow·vs into one f32 first, then multiply f32(acc) by it (no bias, so no fma)
    let prow = stats[(rbase + orow4) * 2u + 1u] * 0.007874015748031496;
    if (ocol < dims.n) {
      o[orow_base + ocol] = f32(acc4_0.x) * (prow * vscale[vsbase + ocol]);
    }
    if (ocol + 1u < dims.n) {
      o[orow_base + ocol + 1u] = f32(acc4_0.y) * (prow * vscale[vsbase + ocol + 1u]);
    }
    if (ocol + 2u < dims.n) {
      o[orow_base + ocol + 2u] = f32(acc4_0.z) * (prow * vscale[vsbase + ocol + 2u]);
    }
    if (ocol + 3u < dims.n) {
      o[orow_base + ocol + 3u] = f32(acc4_0.w) * (prow * vscale[vsbase + ocol + 3u]);
    }
    if (ocol + 4u < dims.n) {
      o[orow_base + ocol + 4u] = f32(acc4_1.x) * (prow * vscale[vsbase + ocol + 4u]);
    }
    if (ocol + 5u < dims.n) {
      o[orow_base + ocol + 5u] = f32(acc4_1.y) * (prow * vscale[vsbase + ocol + 5u]);
    }
    if (ocol + 6u < dims.n) {
      o[orow_base + ocol + 6u] = f32(acc4_1.z) * (prow * vscale[vsbase + ocol + 6u]);
    }
    if (ocol + 7u < dims.n) {
      o[orow_base + ocol + 7u] = f32(acc4_1.w) * (prow * vscale[vsbase + ocol + 7u]);
    }
  }
  if (orow5 < dims.m) {
    let orow_base = obase + orow5 * dims.n;
    // MUST: fold prow·vs into one f32 first, then multiply f32(acc) by it (no bias, so no fma)
    let prow = stats[(rbase + orow5) * 2u + 1u] * 0.007874015748031496;
    if (ocol < dims.n) {
      o[orow_base + ocol] = f32(acc5_0.x) * (prow * vscale[vsbase + ocol]);
    }
    if (ocol + 1u < dims.n) {
      o[orow_base + ocol + 1u] = f32(acc5_0.y) * (prow * vscale[vsbase + ocol + 1u]);
    }
    if (ocol + 2u < dims.n) {
      o[orow_base + ocol + 2u] = f32(acc5_0.z) * (prow * vscale[vsbase + ocol + 2u]);
    }
    if (ocol + 3u < dims.n) {
      o[orow_base + ocol + 3u] = f32(acc5_0.w) * (prow * vscale[vsbase + ocol + 3u]);
    }
    if (ocol + 4u < dims.n) {
      o[orow_base + ocol + 4u] = f32(acc5_1.x) * (prow * vscale[vsbase + ocol + 4u]);
    }
    if (ocol + 5u < dims.n) {
      o[orow_base + ocol + 5u] = f32(acc5_1.y) * (prow * vscale[vsbase + ocol + 5u]);
    }
    if (ocol + 6u < dims.n) {
      o[orow_base + ocol + 6u] = f32(acc5_1.z) * (prow * vscale[vsbase + ocol + 6u]);
    }
    if (ocol + 7u < dims.n) {
      o[orow_base + ocol + 7u] = f32(acc5_1.w) * (prow * vscale[vsbase + ocol + 7u]);
    }
  }
  if (orow6 < dims.m) {
    let orow_base = obase + orow6 * dims.n;
    // MUST: fold prow·vs into one f32 first, then multiply f32(acc) by it (no bias, so no fma)
    let prow = stats[(rbase + orow6) * 2u + 1u] * 0.007874015748031496;
    if (ocol < dims.n) {
      o[orow_base + ocol] = f32(acc6_0.x) * (prow * vscale[vsbase + ocol]);
    }
    if (ocol + 1u < dims.n) {
      o[orow_base + ocol + 1u] = f32(acc6_0.y) * (prow * vscale[vsbase + ocol + 1u]);
    }
    if (ocol + 2u < dims.n) {
      o[orow_base + ocol + 2u] = f32(acc6_0.z) * (prow * vscale[vsbase + ocol + 2u]);
    }
    if (ocol + 3u < dims.n) {
      o[orow_base + ocol + 3u] = f32(acc6_0.w) * (prow * vscale[vsbase + ocol + 3u]);
    }
    if (ocol + 4u < dims.n) {
      o[orow_base + ocol + 4u] = f32(acc6_1.x) * (prow * vscale[vsbase + ocol + 4u]);
    }
    if (ocol + 5u < dims.n) {
      o[orow_base + ocol + 5u] = f32(acc6_1.y) * (prow * vscale[vsbase + ocol + 5u]);
    }
    if (ocol + 6u < dims.n) {
      o[orow_base + ocol + 6u] = f32(acc6_1.z) * (prow * vscale[vsbase + ocol + 6u]);
    }
    if (ocol + 7u < dims.n) {
      o[orow_base + ocol + 7u] = f32(acc6_1.w) * (prow * vscale[vsbase + ocol + 7u]);
    }
  }
  if (orow7 < dims.m) {
    let orow_base = obase + orow7 * dims.n;
    // MUST: fold prow·vs into one f32 first, then multiply f32(acc) by it (no bias, so no fma)
    let prow = stats[(rbase + orow7) * 2u + 1u] * 0.007874015748031496;
    if (ocol < dims.n) {
      o[orow_base + ocol] = f32(acc7_0.x) * (prow * vscale[vsbase + ocol]);
    }
    if (ocol + 1u < dims.n) {
      o[orow_base + ocol + 1u] = f32(acc7_0.y) * (prow * vscale[vsbase + ocol + 1u]);
    }
    if (ocol + 2u < dims.n) {
      o[orow_base + ocol + 2u] = f32(acc7_0.z) * (prow * vscale[vsbase + ocol + 2u]);
    }
    if (ocol + 3u < dims.n) {
      o[orow_base + ocol + 3u] = f32(acc7_0.w) * (prow * vscale[vsbase + ocol + 3u]);
    }
    if (ocol + 4u < dims.n) {
      o[orow_base + ocol + 4u] = f32(acc7_1.x) * (prow * vscale[vsbase + ocol + 4u]);
    }
    if (ocol + 5u < dims.n) {
      o[orow_base + ocol + 5u] = f32(acc7_1.y) * (prow * vscale[vsbase + ocol + 5u]);
    }
    if (ocol + 6u < dims.n) {
      o[orow_base + ocol + 6u] = f32(acc7_1.z) * (prow * vscale[vsbase + ocol + 6u]);
    }
    if (ocol + 7u < dims.n) {
      o[orow_base + ocol + 7u] = f32(acc7_1.w) * (prow * vscale[vsbase + ocol + 7u]);
    }
  }
}
