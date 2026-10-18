// karume attention_pv (O[b,m,d] = P[b,m,n] · v[b,n,d], P̃ = round(127·exp(S − m)) not materialised, v is per-column i8 (Vᵀ contiguous), S は f16 格納（pack2x16float）, タイル 64x128 / 1 スレッド 8x8 / wg 16x8 / K 16 + vec4)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
// Eligibility requires N % 4 == 0, so a row of S always starts on a quad boundary (the same step
// as P̃'s packing of 4)
@group(0) @binding(1) var<storage, read> s: array<u32>;
@group(0) @binding(2) var<storage, read> vq: array<u32>;
@group(0) @binding(3) var<storage, read> stats: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> vscale: array<f32>;

// Quad read of f16 storage: quad q = the 2 words s[2q] and s[2q + 1] (unpack is exact)
fn score_quad(q: u32) -> vec4<f32> {
  let w = q * 2u;
  let lo = unpack2x16float(s[w]);
  let hi = unpack2x16float(s[w + 1u]);
  return vec4<f32>(lo.x, lo.y, hi.x, hi.y);
}

// integer dot product (WGSL language extension packed_4x8_integer_dot_product); returns **the same integer** as the emulated variant
fn idot(a: u32, b: u32) -> i32 {
  return dot4I8Packed(a, b);
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
  let n4 = dims.n / 4u;
  // The batch axis folding B and H (uniform per workgroup — a precondition for the inner
  // workgroupBarrier to meet WGSL's uniformity requirement). MUST: S counts in quads, vq in packs,
  // the row statistics in 2 words per row, vscale in D entries, O in elements（v4 では quad）
  // — mixing up the units reads and writes the next head when B·H ≥ 2 (a wrong value without an
  // exception, invisible with B=H=1)
  let sbase = wid.z * dims.m * k4;
  let vbase = wid.z * dims.n * k4;
  let rbase = wid.z * dims.m;
  let vsbase = wid.z * dims.n;
  let obase = wid.z * dims.m * n4;
  // P̃ タイルの担当 (64 rows × 4 packs over 128 threads in 2 rounds)
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
  // 列は D。vq は [D,N] の N 連続（= パック方向）なので**転置が要らない**
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
      let raw0 = score_quad(arow_base0 + apack);
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
      let raw1 = score_quad(arow_base1 + apack);
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
  let ocq0 = wid.x * 32u + lid.x * 2u;
  let ocq1 = ocq0 + 1u;
  let orow0 = wid.y * 64u + lid.y * 8u;
  let orow1 = orow0 + 1u;
  let orow2 = orow0 + 2u;
  let orow3 = orow0 + 3u;
  let orow4 = orow0 + 4u;
  let orow5 = orow0 + 5u;
  let orow6 = orow0 + 6u;
  let orow7 = orow0 + 7u;
  if (ocq0 < n4) {
    // D % 4 == 0 and ocq0 < n4, so oc + 3 < D.
    // MUST: the column scale is a **row** of Vᵀ = (b, h, d) (= V's per-column scale). Mixing it
    // up with the row scale raises no exception
    let oc = ocq0 * 4u;
    let vs = vec4<f32>(
      vscale[vsbase + oc],
      vscale[vsbase + oc + 1u],
      vscale[vsbase + oc + 2u],
      vscale[vsbase + oc + 3u],
    );
    // MUST: fold prow·vs into one f32 first, then multiply f32(acc) by it (no bias, so no fma)
    if (orow0 < dims.m) {
      let prow = stats[(rbase + orow0) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow0 * n4 + ocq0] = vec4<f32>(acc0_0) * (prow * vs);
    }
    if (orow1 < dims.m) {
      let prow = stats[(rbase + orow1) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow1 * n4 + ocq0] = vec4<f32>(acc1_0) * (prow * vs);
    }
    if (orow2 < dims.m) {
      let prow = stats[(rbase + orow2) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow2 * n4 + ocq0] = vec4<f32>(acc2_0) * (prow * vs);
    }
    if (orow3 < dims.m) {
      let prow = stats[(rbase + orow3) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow3 * n4 + ocq0] = vec4<f32>(acc3_0) * (prow * vs);
    }
    if (orow4 < dims.m) {
      let prow = stats[(rbase + orow4) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow4 * n4 + ocq0] = vec4<f32>(acc4_0) * (prow * vs);
    }
    if (orow5 < dims.m) {
      let prow = stats[(rbase + orow5) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow5 * n4 + ocq0] = vec4<f32>(acc5_0) * (prow * vs);
    }
    if (orow6 < dims.m) {
      let prow = stats[(rbase + orow6) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow6 * n4 + ocq0] = vec4<f32>(acc6_0) * (prow * vs);
    }
    if (orow7 < dims.m) {
      let prow = stats[(rbase + orow7) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow7 * n4 + ocq0] = vec4<f32>(acc7_0) * (prow * vs);
    }
  }
  if (ocq1 < n4) {
    // D % 4 == 0 and ocq1 < n4, so oc + 3 < D.
    // MUST: the column scale is a **row** of Vᵀ = (b, h, d) (= V's per-column scale). Mixing it
    // up with the row scale raises no exception
    let oc = ocq1 * 4u;
    let vs = vec4<f32>(
      vscale[vsbase + oc],
      vscale[vsbase + oc + 1u],
      vscale[vsbase + oc + 2u],
      vscale[vsbase + oc + 3u],
    );
    // MUST: fold prow·vs into one f32 first, then multiply f32(acc) by it (no bias, so no fma)
    if (orow0 < dims.m) {
      let prow = stats[(rbase + orow0) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow0 * n4 + ocq1] = vec4<f32>(acc0_1) * (prow * vs);
    }
    if (orow1 < dims.m) {
      let prow = stats[(rbase + orow1) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow1 * n4 + ocq1] = vec4<f32>(acc1_1) * (prow * vs);
    }
    if (orow2 < dims.m) {
      let prow = stats[(rbase + orow2) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow2 * n4 + ocq1] = vec4<f32>(acc2_1) * (prow * vs);
    }
    if (orow3 < dims.m) {
      let prow = stats[(rbase + orow3) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow3 * n4 + ocq1] = vec4<f32>(acc3_1) * (prow * vs);
    }
    if (orow4 < dims.m) {
      let prow = stats[(rbase + orow4) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow4 * n4 + ocq1] = vec4<f32>(acc4_1) * (prow * vs);
    }
    if (orow5 < dims.m) {
      let prow = stats[(rbase + orow5) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow5 * n4 + ocq1] = vec4<f32>(acc5_1) * (prow * vs);
    }
    if (orow6 < dims.m) {
      let prow = stats[(rbase + orow6) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow6 * n4 + ocq1] = vec4<f32>(acc6_1) * (prow * vs);
    }
    if (orow7 < dims.m) {
      let prow = stats[(rbase + orow7) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow7 * n4 + ocq1] = vec4<f32>(acc7_1) * (prow * vs);
    }
  }
}
