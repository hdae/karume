// karume attention_qk (S[b,m,n] = q[b,m,d] · k[b,n,d]ᵀ, integer dot product of per-token i8 q and k, half scale on the dequant side, tile 128x64 / 8x8 per thread / wg 8x16 / K 16 + vec4)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
  scale: f32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> qq: array<u32>;
@group(0) @binding(2) var<storage, read> kq: array<u32>;
@group(0) @binding(3) var<storage, read_write> s: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> qscale: array<f32>;
@group(0) @binding(5) var<storage, read> kscale: array<f32>;

// integer dot product (WGSL language extension packed_4x8_integer_dot_product); returns **the same integer** as the emulated variant
fn idot(a: u32, b: u32) -> i32 {
  return dot4I8Packed(a, b);
}

// Shared tiles are laid out **[pack][row] / [pack][column]** (as in linear's i8a8 — the inner reads
// get word stride 4 and bank conflicts drop from 8-way to 2-way)
var<workgroup> sa: array<u32, 512>;
var<workgroup> sb: array<u32, 256>;

@compute @workgroup_size(8, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 8u + lid.x;
  // Eligibility requires D % 4 == 0, so a row always starts on a word boundary (the i8 payload
  // packs 4 per flat index)
  let k4 = dims.k / 4u;
  let n4 = dims.n / 4u;
  // The batch axis folding B and H (uniform per workgroup — a precondition for the inner
  // workgroupBarrier to meet WGSL's uniformity requirement). MUST: the q / k payloads count in
  // **packs**, the scales in rows, S in elements (quads in v4) — mixing up the units
  // reads and writes the next head when B·H ≥ 2 (a wrong value without an exception, invisible
  // to tests with B=H=1)
  let qbase = wid.z * dims.m * k4;
  let kbase = wid.z * dims.n * k4;
  let qsbase = wid.z * dims.m;
  let ksbase = wid.z * dims.n;
  let sbase = wid.z * dims.m * n4;
  // q tile assignment (128 rows × 4 packs over 128 threads in 4 rounds)
  let ar = tid / 4u;
  let ap = tid % 4u;
  let sa_at = ap * 128u + ar;
  let arow0 = wid.y * 128u + ar;
  let arow_base0 = qbase + arow0 * k4;
  let arow1 = arow0 + 32u;
  let arow_base1 = qbase + arow1 * k4;
  let arow2 = arow0 + 64u;
  let arow_base2 = qbase + arow2 * k4;
  let arow3 = arow0 + 96u;
  let arow_base3 = qbase + arow3 * k4;
  // B tile assignment (64 columns × 4 packs over 128 threads in 2 rounds).
  // k is read as [N,D] as is (the contiguous direction D is the pack direction, so **no transpose is needed**)
  let wc = tid / 4u;
  let wp = tid % 4u;
  let sb_at = wp * 64u + wc;
  let wcol0 = wid.x * 64u + wc;
  let krow_base0 = kbase + wcol0 * k4;
  let wcol1 = wcol0 + 32u;
  let krow_base1 = kbase + wcol1 * k4;
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
    // Zero-fill out of range (dot4I8Packed(0, x) == 0, so the result is exact even with a K remainder)
    let apack = t * 4u + ap;
    var av0 = 0u;
    if (arow0 < dims.m && apack < k4) {
      av0 = qq[arow_base0 + apack];
    }
    sa[sa_at] = av0;
    var av1 = 0u;
    if (arow1 < dims.m && apack < k4) {
      av1 = qq[arow_base1 + apack];
    }
    sa[sa_at + 32u] = av1;
    var av2 = 0u;
    if (arow2 < dims.m && apack < k4) {
      av2 = qq[arow_base2 + apack];
    }
    sa[sa_at + 64u] = av2;
    var av3 = 0u;
    if (arow3 < dims.m && apack < k4) {
      av3 = qq[arow_base3 + apack];
    }
    sa[sa_at + 96u] = av3;
    let kpack = t * 4u + wp;
    var bv0 = 0u;
    if (wcol0 < dims.n && kpack < k4) {
      bv0 = kq[krow_base0 + kpack];
    }
    sb[sb_at] = bv0;
    var bv1 = 0u;
    if (wcol1 < dims.n && kpack < k4) {
      bv1 = kq[krow_base1 + kpack];
    }
    sb[sb_at + 32u] = bv1;
    workgroupBarrier();
    // 16 shared loads (8 words of B + 8 words of A) give 64 integer inner products = 256 MACs.
    // The reduction is exact i32 addition, so it **does not depend on order** (unlike the f32
    // skeleton, the addition order is not part of the numeric contract)
    for (var p = 0u; p < 4u; p = p + 1u) {
      let bcol = p * 64u + lid.x * 8u;
      let b0 = sb[bcol];
      let b1 = sb[bcol + 1u];
      let b2 = sb[bcol + 2u];
      let b3 = sb[bcol + 3u];
      let b4 = sb[bcol + 4u];
      let b5 = sb[bcol + 5u];
      let b6 = sb[bcol + 6u];
      let b7 = sb[bcol + 7u];
      let arow_at = p * 128u + lid.y * 8u;
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
  let ocq0 = wid.x * 16u + lid.x * 2u;
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
    // n % 4 == 0 and ocq0 < n4, so oc + 3 < n.
    // MUST: the half scale goes to **both** q and k on the dequant side (the column side once, here)
    let oc = ocq0 * 4u;
    let ks = vec4<f32>(
      kscale[ksbase + oc],
      kscale[ksbase + oc + 1u],
      kscale[ksbase + oc + 2u],
      kscale[ksbase + oc + 3u],
    ) * dims.scale;
    // MUST: fold qs'·ks' into one f32 first, then multiply f32(acc) by it (no bias, so no fma)
    if (orow0 < dims.m) {
      s[sbase + orow0 * n4 + ocq0] = vec4<f32>(acc0_0) * ((qscale[qsbase + orow0] * dims.scale) * ks);
    }
    if (orow1 < dims.m) {
      s[sbase + orow1 * n4 + ocq0] = vec4<f32>(acc1_0) * ((qscale[qsbase + orow1] * dims.scale) * ks);
    }
    if (orow2 < dims.m) {
      s[sbase + orow2 * n4 + ocq0] = vec4<f32>(acc2_0) * ((qscale[qsbase + orow2] * dims.scale) * ks);
    }
    if (orow3 < dims.m) {
      s[sbase + orow3 * n4 + ocq0] = vec4<f32>(acc3_0) * ((qscale[qsbase + orow3] * dims.scale) * ks);
    }
    if (orow4 < dims.m) {
      s[sbase + orow4 * n4 + ocq0] = vec4<f32>(acc4_0) * ((qscale[qsbase + orow4] * dims.scale) * ks);
    }
    if (orow5 < dims.m) {
      s[sbase + orow5 * n4 + ocq0] = vec4<f32>(acc5_0) * ((qscale[qsbase + orow5] * dims.scale) * ks);
    }
    if (orow6 < dims.m) {
      s[sbase + orow6 * n4 + ocq0] = vec4<f32>(acc6_0) * ((qscale[qsbase + orow6] * dims.scale) * ks);
    }
    if (orow7 < dims.m) {
      s[sbase + orow7 * n4 + ocq0] = vec4<f32>(acc7_0) * ((qscale[qsbase + orow7] * dims.scale) * ks);
    }
  }
  if (ocq1 < n4) {
    // n % 4 == 0 and ocq1 < n4, so oc + 3 < n.
    // MUST: the half scale goes to **both** q and k on the dequant side (the column side once, here)
    let oc = ocq1 * 4u;
    let ks = vec4<f32>(
      kscale[ksbase + oc],
      kscale[ksbase + oc + 1u],
      kscale[ksbase + oc + 2u],
      kscale[ksbase + oc + 3u],
    ) * dims.scale;
    // MUST: fold qs'·ks' into one f32 first, then multiply f32(acc) by it (no bias, so no fma)
    if (orow0 < dims.m) {
      s[sbase + orow0 * n4 + ocq1] = vec4<f32>(acc0_1) * ((qscale[qsbase + orow0] * dims.scale) * ks);
    }
    if (orow1 < dims.m) {
      s[sbase + orow1 * n4 + ocq1] = vec4<f32>(acc1_1) * ((qscale[qsbase + orow1] * dims.scale) * ks);
    }
    if (orow2 < dims.m) {
      s[sbase + orow2 * n4 + ocq1] = vec4<f32>(acc2_1) * ((qscale[qsbase + orow2] * dims.scale) * ks);
    }
    if (orow3 < dims.m) {
      s[sbase + orow3 * n4 + ocq1] = vec4<f32>(acc3_1) * ((qscale[qsbase + orow3] * dims.scale) * ks);
    }
    if (orow4 < dims.m) {
      s[sbase + orow4 * n4 + ocq1] = vec4<f32>(acc4_1) * ((qscale[qsbase + orow4] * dims.scale) * ks);
    }
    if (orow5 < dims.m) {
      s[sbase + orow5 * n4 + ocq1] = vec4<f32>(acc5_1) * ((qscale[qsbase + orow5] * dims.scale) * ks);
    }
    if (orow6 < dims.m) {
      s[sbase + orow6 * n4 + ocq1] = vec4<f32>(acc6_1) * ((qscale[qsbase + orow6] * dims.scale) * ks);
    }
    if (orow7 < dims.m) {
      s[sbase + orow7 * n4 + ocq1] = vec4<f32>(acc7_1) * ((qscale[qsbase + orow7] * dims.scale) * ks);
    }
  }
}
