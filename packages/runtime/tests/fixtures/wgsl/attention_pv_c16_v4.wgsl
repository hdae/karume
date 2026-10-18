// karume attention_pv (O[b,m,d] = P[b,m,n] · v[b,n,d], P = exp(S − m)·inv not materialised, f32, f16 タイル計算・S は f16, レジスタ 128x128 タイル / 1 スレッド 8x8 / wg 16x16 + vec4)
enable f16;
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> s: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> v: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> stats: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<vec4<f32>>;

// Shared A tile (128 rows × K 16, scalar storage) and
// shared B tile (K 16 × 32 column quads, column direction grouped into vec4)。f16 変種は共有バイトが半分
// (16384 B → 8192 B / WG) — the expected gain rests entirely on this one mechanism
var<workgroup> sa: array<f16, 2048>;
var<workgroup> sb: array<vec4<f16>, 512>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 16u + lid.x;
  let k4 = dims.k / 4u;
  let n4 = dims.n / 4u;
  // The batch is uniform per workgroup (one z index = the tiles of one batch) — the premise for the
  // inner workgroupBarrier to meet WGSL's uniformity requirement。MUST: base は quad 単位（要素単位で組むと隣のバッチを読む）
  let abase = wid.z * dims.m * k4;
  let bbase = wid.z * dims.k * n4;
  let cbase = wid.z * dims.m * n4;
  let rbase = wid.z * dims.m;
  // A tile assignment (128 rows × 4 quads over 256 threads in 2 rounds)
  let ar = tid / 4u;
  let aq = tid % 4u;
  let arow0 = wid.y * 128u + ar;
  let arow_base0 = abase + arow0 * k4;
  let sa_base0 = ar * 16u + aq * 4u;
  let arow1 = arow0 + 64u;
  let arow_base1 = arow_base0 + 64u * k4;
  let sa_base1 = sa_base0 + 1024u;
  // On edge tiles arow >= m can happen, so the index falls back to 0 (the value read is used only in the
  // arow < m branch). Reading stats out of range is itself safe under WGSL's bounded access
  let stat_at0 = select(0u, (rbase + arow0) * 2u, arow0 < dims.m);
  let row_max0 = stats[stat_at0];
  let row_inv0 = stats[stat_at0 + 1u];
  let stat_at1 = select(0u, (rbase + arow1) * 2u, arow1 < dims.m);
  let row_max1 = stats[stat_at1];
  let row_inv1 = stats[stat_at1 + 1u];
  // B tile assignment (K 16 rows × 32 column quads over 256 threads in 2 rounds)
  let bk0 = tid / 32u;
  let bcq = tid % 32u;
  let bc4 = wid.x * 32u + bcq;
  let bk1 = bk0 + 8u;
  // The loop condition is uniform (dims is a uniform buffer) — needed for the inner workgroupBarrier
  // to meet WGSL's uniformity requirement
  let tiles = (dims.k + 15u) / 16u;
  var acc0_0 = vec4<f32>(0.0);
  var acc0_1 = vec4<f32>(0.0);
  var acc1_0 = vec4<f32>(0.0);
  var acc1_1 = vec4<f32>(0.0);
  var acc2_0 = vec4<f32>(0.0);
  var acc2_1 = vec4<f32>(0.0);
  var acc3_0 = vec4<f32>(0.0);
  var acc3_1 = vec4<f32>(0.0);
  var acc4_0 = vec4<f32>(0.0);
  var acc4_1 = vec4<f32>(0.0);
  var acc5_0 = vec4<f32>(0.0);
  var acc5_1 = vec4<f32>(0.0);
  var acc6_0 = vec4<f32>(0.0);
  var acc6_1 = vec4<f32>(0.0);
  var acc7_0 = vec4<f32>(0.0);
  var acc7_1 = vec4<f32>(0.0);
  for (var t = 0u; t < tiles; t = t + 1u) {
    // Fill out-of-range entries with 0. They do not contribute to the inner product, so
    // remainder shapes give the same result
    let ak0 = t * 16u + aq * 4u;
    var av0 = vec4<f32>(0.0);
    if (arow0 < dims.m && ak0 < dims.k) {
      let raw0 = s[arow_base0 + t * 4u + aq];
      av0 = vec4<f32>(
        exp(f32(raw0.x) - row_max0) * row_inv0,
        exp(f32(raw0.y) - row_max0) * row_inv0,
        exp(f32(raw0.z) - row_max0) * row_inv0,
        exp(f32(raw0.w) - row_max0) * row_inv0,
      );
    }
    sa[sa_base0] = f16(av0.x);
    sa[sa_base0 + 1u] = f16(av0.y);
    sa[sa_base0 + 2u] = f16(av0.z);
    sa[sa_base0 + 3u] = f16(av0.w);
    var av1 = vec4<f32>(0.0);
    if (arow1 < dims.m && ak0 < dims.k) {
      let raw1 = s[arow_base1 + t * 4u + aq];
      av1 = vec4<f32>(
        exp(f32(raw1.x) - row_max1) * row_inv1,
        exp(f32(raw1.y) - row_max1) * row_inv1,
        exp(f32(raw1.z) - row_max1) * row_inv1,
        exp(f32(raw1.w) - row_max1) * row_inv1,
      );
    }
    sa[sa_base1] = f16(av1.x);
    sa[sa_base1 + 1u] = f16(av1.y);
    sa[sa_base1 + 2u] = f16(av1.z);
    sa[sa_base1 + 3u] = f16(av1.w);
    let brow0 = t * 16u + bk0;
    var bv4_0 = vec4<f32>(0.0);
    if (brow0 < dims.k && bc4 < n4) {
      bv4_0 = v[bbase + brow0 * n4 + bc4];
    }
    sb[bk0 * 32u + bcq] = vec4<f16>(bv4_0);
    let brow1 = t * 16u + bk1;
    var bv4_1 = vec4<f32>(0.0);
    if (brow1 < dims.k && bc4 < n4) {
      bv4_1 = v[bbase + brow1 * n4 + bc4];
    }
    sb[bk1 * 32u + bcq] = vec4<f16>(bv4_1);
    workgroupBarrier();
    // 10 shared loads (2 vec4 of B + 8 scalars of A) give 64 MACs.
    // The reduction is sequential in ascending k, and the addition order per output element matches
    // the 16×16 kernel with 1 output per thread exactly.
    // MUST: widening to f32 happens **once at register load** (10 times / 64 MACs). Writing
    // f32(av * bv) per MAC would raise the conversions to 64 and also drop the product to f16 precision (rounding 2 of the ADR 0028 sequence).
    // Widening f16 → f32 is exact, so the values do not differ by a single bit from
    // "the f32 variant with inputs rounded to f16".
    for (var kk = 0u; kk < 16u; kk = kk + 1u) {
      let bv0 = vec4<f32>(sb[kk * 32u + lid.x * 2u]);
      let bv1 = vec4<f32>(sb[kk * 32u + lid.x * 2u + 1u]);
      acc0_0 = acc0_0 + f32(sa[(lid.y * 8u + 0u) * 16u + kk]) * bv0;
      acc0_1 = acc0_1 + f32(sa[(lid.y * 8u + 0u) * 16u + kk]) * bv1;
      acc1_0 = acc1_0 + f32(sa[(lid.y * 8u + 1u) * 16u + kk]) * bv0;
      acc1_1 = acc1_1 + f32(sa[(lid.y * 8u + 1u) * 16u + kk]) * bv1;
      acc2_0 = acc2_0 + f32(sa[(lid.y * 8u + 2u) * 16u + kk]) * bv0;
      acc2_1 = acc2_1 + f32(sa[(lid.y * 8u + 2u) * 16u + kk]) * bv1;
      acc3_0 = acc3_0 + f32(sa[(lid.y * 8u + 3u) * 16u + kk]) * bv0;
      acc3_1 = acc3_1 + f32(sa[(lid.y * 8u + 3u) * 16u + kk]) * bv1;
      acc4_0 = acc4_0 + f32(sa[(lid.y * 8u + 4u) * 16u + kk]) * bv0;
      acc4_1 = acc4_1 + f32(sa[(lid.y * 8u + 4u) * 16u + kk]) * bv1;
      acc5_0 = acc5_0 + f32(sa[(lid.y * 8u + 5u) * 16u + kk]) * bv0;
      acc5_1 = acc5_1 + f32(sa[(lid.y * 8u + 5u) * 16u + kk]) * bv1;
      acc6_0 = acc6_0 + f32(sa[(lid.y * 8u + 6u) * 16u + kk]) * bv0;
      acc6_1 = acc6_1 + f32(sa[(lid.y * 8u + 6u) * 16u + kk]) * bv1;
      acc7_0 = acc7_0 + f32(sa[(lid.y * 8u + 7u) * 16u + kk]) * bv0;
      acc7_1 = acc7_1 + f32(sa[(lid.y * 8u + 7u) * 16u + kk]) * bv1;
    }
    workgroupBarrier();
  }
  let ocq0 = wid.x * 32u + lid.x * 2u;
  let ocq1 = ocq0 + 1u;
  let orow0 = wid.y * 128u + lid.y * 8u;
  let orow1 = orow0 + 1u;
  let orow2 = orow0 + 2u;
  let orow3 = orow0 + 3u;
  let orow4 = orow0 + 4u;
  let orow5 = orow0 + 5u;
  let orow6 = orow0 + 6u;
  let orow7 = orow0 + 7u;
  if (ocq0 < n4) {
    if (orow0 < dims.m) {
      o[cbase + orow0 * n4 + ocq0] = acc0_0;
    }
    if (orow1 < dims.m) {
      o[cbase + orow1 * n4 + ocq0] = acc1_0;
    }
    if (orow2 < dims.m) {
      o[cbase + orow2 * n4 + ocq0] = acc2_0;
    }
    if (orow3 < dims.m) {
      o[cbase + orow3 * n4 + ocq0] = acc3_0;
    }
    if (orow4 < dims.m) {
      o[cbase + orow4 * n4 + ocq0] = acc4_0;
    }
    if (orow5 < dims.m) {
      o[cbase + orow5 * n4 + ocq0] = acc5_0;
    }
    if (orow6 < dims.m) {
      o[cbase + orow6 * n4 + ocq0] = acc6_0;
    }
    if (orow7 < dims.m) {
      o[cbase + orow7 * n4 + ocq0] = acc7_0;
    }
  }
  if (ocq1 < n4) {
    if (orow0 < dims.m) {
      o[cbase + orow0 * n4 + ocq1] = acc0_1;
    }
    if (orow1 < dims.m) {
      o[cbase + orow1 * n4 + ocq1] = acc1_1;
    }
    if (orow2 < dims.m) {
      o[cbase + orow2 * n4 + ocq1] = acc2_1;
    }
    if (orow3 < dims.m) {
      o[cbase + orow3 * n4 + ocq1] = acc3_1;
    }
    if (orow4 < dims.m) {
      o[cbase + orow4 * n4 + ocq1] = acc4_1;
    }
    if (orow5 < dims.m) {
      o[cbase + orow5 * n4 + ocq1] = acc5_1;
    }
    if (orow6 < dims.m) {
      o[cbase + orow6 * n4 + ocq1] = acc6_1;
    }
    if (orow7 < dims.m) {
      o[cbase + orow7 * n4 + ocq1] = acc7_1;
    }
  }
}
