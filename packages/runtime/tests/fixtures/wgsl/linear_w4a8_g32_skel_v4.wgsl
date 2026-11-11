// karume linear (x[m,k] · wᵀ[k,n] + bias[n], integer dot product of per-token i8 activations × i4 group 32 weights, tile 128x64 / 16x4 per thread / wg 16x8 / K 32 / vec4 skeleton + vec4)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> xq: array<u32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> wscale: array<f32>;
@group(0) @binding(6) var<storage, read> xscale: array<f32>;

// integer dot product (WGSL language extension packed_4x8_integer_dot_product); returns **the same integer** as the emulated variant
fn idot(a: u32, b: u32) -> i32 {
  return dot4I8Packed(a, b);
}

// integer-lane decode of i4 storage: flat elements i..i+3 (i a multiple of 4) → a u32 of four i8
// lanes. 1 word = 8 elements, so the 4 elements are the upper or lower 16 bits of the word
// ((i >> 2) & 1 picks). Restored as q = u − 8 **as an integer** (the scale is applied by the f32
// flush at the group boundary — ADR 0069 decision 4)
fn i4lanes(i: u32) -> u32 {
  let half = (w[i >> 3u] >> (((i >> 2u) & 1u) * 16u)) & 0xFFFFu;
  let u = vec4<u32>(half, half >> 4u, half >> 8u, half >> 12u) & vec4<u32>(0xFu);
  return pack4xI8(vec4<i32>(u) - vec4<i32>(8));
}

// Shared tiles are laid out **[pack][row] / [pack][column]** (rearranged from the prototype's
// [row][pack] — the inner reads get word stride 4 and bank conflicts drop from 8-way to
// 2-way)
var<workgroup> sa: array<u32, 1024>;
var<workgroup> sb: array<u32, 512>;

@compute @workgroup_size(16, 8)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 16u + lid.x;
  // Eligibility requires k % 4 == 0, so a row always starts on a word boundary (the i8 payload
  // packs 4 per flat index)
  let k4 = dims.k / 4u;
  // x tile assignment (128 rows × 8 packs over 128 threads in 8 rounds)
  let ar = tid / 8u;
  let ap = tid % 8u;
  let sa_at = ap * 128u + ar;
  let arow0 = wid.y * 128u + ar;
  let arow_base0 = arow0 * k4;
  let arow1 = arow0 + 16u;
  let arow_base1 = arow1 * k4;
  let arow2 = arow0 + 32u;
  let arow_base2 = arow2 * k4;
  let arow3 = arow0 + 48u;
  let arow_base3 = arow3 * k4;
  let arow4 = arow0 + 64u;
  let arow_base4 = arow4 * k4;
  let arow5 = arow0 + 80u;
  let arow_base5 = arow5 * k4;
  let arow6 = arow0 + 96u;
  let arow_base6 = arow6 * k4;
  let arow7 = arow0 + 112u;
  let arow_base7 = arow7 * k4;
  // W tile assignment (64 output channels × 8 packs over 128 threads in 4 rounds).
  // Weights are read as [n,k] (the contiguous direction is k = the pack direction, so **no
  // transpose is needed**)
  let wc = tid / 8u;
  let wp = tid % 8u;
  let sb_at = wp * 64u + wc;
  let wcol0 = wid.x * 64u + wc;
  let wrow_base0 = wcol0 * dims.k;
  let wcol1 = wcol0 + 16u;
  let wrow_base1 = wcol1 * dims.k;
  let wcol2 = wcol0 + 32u;
  let wrow_base2 = wcol2 * dims.k;
  let wcol3 = wcol0 + 48u;
  let wrow_base3 = wcol3 * dims.k;
  // Output assignment is **outside the loop** (the flush at a group boundary looks up
  // per-column scales)
  let ocol = wid.x * 64u + lid.x * 4u;
  let orow0 = wid.y * 128u + lid.y * 16u;
  let orow1 = orow0 + 1u;
  let orow2 = orow0 + 2u;
  let orow3 = orow0 + 3u;
  let orow4 = orow0 + 4u;
  let orow5 = orow0 + 5u;
  let orow6 = orow0 + 6u;
  let orow7 = orow0 + 7u;
  let orow8 = orow0 + 8u;
  let orow9 = orow0 + 9u;
  let orow10 = orow0 + 10u;
  let orow11 = orow0 + 11u;
  let orow12 = orow0 + 12u;
  let orow13 = orow0 + 13u;
  let orow14 = orow0 + 14u;
  let orow15 = orow0 + 15u;
  // group scales are a flat [n, k/g]. Invalid columns are clamped to the last valid column, so
  // nothing out of range is read. The row start is loop-invariant and hoisted (output of invalid
  // columns is discarded by the store guard)
  let groups = dims.k >> 5u;
  let wsb0 = min(ocol, dims.n - 1u) * groups;
  let wsb1 = min(ocol + 1u, dims.n - 1u) * groups;
  let wsb2 = min(ocol + 2u, dims.n - 1u) * groups;
  let wsb3 = min(ocol + 3u, dims.n - 1u) * groups;
  // f32 accumulator (written only at group boundaries — k/g roundings)
  var accf0_0 = vec4<f32>(0.0);
  var accf1_0 = vec4<f32>(0.0);
  var accf2_0 = vec4<f32>(0.0);
  var accf3_0 = vec4<f32>(0.0);
  var accf4_0 = vec4<f32>(0.0);
  var accf5_0 = vec4<f32>(0.0);
  var accf6_0 = vec4<f32>(0.0);
  var accf7_0 = vec4<f32>(0.0);
  var accf8_0 = vec4<f32>(0.0);
  var accf9_0 = vec4<f32>(0.0);
  var accf10_0 = vec4<f32>(0.0);
  var accf11_0 = vec4<f32>(0.0);
  var accf12_0 = vec4<f32>(0.0);
  var accf13_0 = vec4<f32>(0.0);
  var accf14_0 = vec4<f32>(0.0);
  var accf15_0 = vec4<f32>(0.0);
  // The loop condition is uniform (dims is a uniform buffer) — needed for the inner
  // workgroupBarrier to meet WGSL's uniformity requirement
  for (var gi = 0u; gi < groups; gi = gi + 1u) {
    // The i32 accumulator is reset per group (|acci| ≤ g·127·8 — limit: LINEAR_W4A8_MAX_GROUP)
    var acc0_0 = vec4<i32>(0);
    var acc1_0 = vec4<i32>(0);
    var acc2_0 = vec4<i32>(0);
    var acc3_0 = vec4<i32>(0);
    var acc4_0 = vec4<i32>(0);
    var acc5_0 = vec4<i32>(0);
    var acc6_0 = vec4<i32>(0);
    var acc7_0 = vec4<i32>(0);
    var acc8_0 = vec4<i32>(0);
    var acc9_0 = vec4<i32>(0);
    var acc10_0 = vec4<i32>(0);
    var acc11_0 = vec4<i32>(0);
    var acc12_0 = vec4<i32>(0);
    var acc13_0 = vec4<i32>(0);
    var acc14_0 = vec4<i32>(0);
    var acc15_0 = vec4<i32>(0);
    // 1 group = exactly 1 K tiles (the g % tileK == 0 check). t is the running number
    let gbase = gi * 1u;
    for (var gt = 0u; gt < 1u; gt = gt + 1u) {
      let t = gbase + gt;
    // Zero-fill out of range (dot4I8Packed(0, x) == 0, so the result is exact even with a K
    // remainder)
    let apack = t * 8u + ap;
    var av0 = 0u;
    if (arow0 < dims.m && apack < k4) {
      av0 = xq[arow_base0 + apack];
    }
    sa[sa_at] = av0;
    var av1 = 0u;
    if (arow1 < dims.m && apack < k4) {
      av1 = xq[arow_base1 + apack];
    }
    sa[sa_at + 16u] = av1;
    var av2 = 0u;
    if (arow2 < dims.m && apack < k4) {
      av2 = xq[arow_base2 + apack];
    }
    sa[sa_at + 32u] = av2;
    var av3 = 0u;
    if (arow3 < dims.m && apack < k4) {
      av3 = xq[arow_base3 + apack];
    }
    sa[sa_at + 48u] = av3;
    var av4 = 0u;
    if (arow4 < dims.m && apack < k4) {
      av4 = xq[arow_base4 + apack];
    }
    sa[sa_at + 64u] = av4;
    var av5 = 0u;
    if (arow5 < dims.m && apack < k4) {
      av5 = xq[arow_base5 + apack];
    }
    sa[sa_at + 80u] = av5;
    var av6 = 0u;
    if (arow6 < dims.m && apack < k4) {
      av6 = xq[arow_base6 + apack];
    }
    sa[sa_at + 96u] = av6;
    var av7 = 0u;
    if (arow7 < dims.m && apack < k4) {
      av7 = xq[arow_base7 + apack];
    }
    sa[sa_at + 112u] = av7;
    // i4 has 8 elements per word, so the pack index is in flat elements (k % g == 0 ∧
    // g % tileK == 0 leaves no K remainder)
    let welem = t * 32u + wp * 4u;
    var wv0 = 0u;
    if (wcol0 < dims.n) {
      wv0 = i4lanes(wrow_base0 + welem);
    }
    sb[sb_at] = wv0;
    var wv1 = 0u;
    if (wcol1 < dims.n) {
      wv1 = i4lanes(wrow_base1 + welem);
    }
    sb[sb_at + 16u] = wv1;
    var wv2 = 0u;
    if (wcol2 < dims.n) {
      wv2 = i4lanes(wrow_base2 + welem);
    }
    sb[sb_at + 32u] = wv2;
    var wv3 = 0u;
    if (wcol3 < dims.n) {
      wv3 = i4lanes(wrow_base3 + welem);
    }
    sb[sb_at + 48u] = wv3;
      workgroupBarrier();
    {
      let bcol = 0u + lid.x * 4u;
      let b0 = sb[bcol];
      let b1 = sb[bcol + 1u];
      let b2 = sb[bcol + 2u];
      let b3 = sb[bcol + 3u];
      let arow_at = 0u + lid.y * 16u;
      let a0 = sa[arow_at];
      acc0_0 = acc0_0 + vec4<i32>(idot(a0, b0), idot(a0, b1), idot(a0, b2), idot(a0, b3));
      let a1 = sa[arow_at + 1u];
      acc1_0 = acc1_0 + vec4<i32>(idot(a1, b0), idot(a1, b1), idot(a1, b2), idot(a1, b3));
      let a2 = sa[arow_at + 2u];
      acc2_0 = acc2_0 + vec4<i32>(idot(a2, b0), idot(a2, b1), idot(a2, b2), idot(a2, b3));
      let a3 = sa[arow_at + 3u];
      acc3_0 = acc3_0 + vec4<i32>(idot(a3, b0), idot(a3, b1), idot(a3, b2), idot(a3, b3));
      let a4 = sa[arow_at + 4u];
      acc4_0 = acc4_0 + vec4<i32>(idot(a4, b0), idot(a4, b1), idot(a4, b2), idot(a4, b3));
      let a5 = sa[arow_at + 5u];
      acc5_0 = acc5_0 + vec4<i32>(idot(a5, b0), idot(a5, b1), idot(a5, b2), idot(a5, b3));
      let a6 = sa[arow_at + 6u];
      acc6_0 = acc6_0 + vec4<i32>(idot(a6, b0), idot(a6, b1), idot(a6, b2), idot(a6, b3));
      let a7 = sa[arow_at + 7u];
      acc7_0 = acc7_0 + vec4<i32>(idot(a7, b0), idot(a7, b1), idot(a7, b2), idot(a7, b3));
      let a8 = sa[arow_at + 8u];
      acc8_0 = acc8_0 + vec4<i32>(idot(a8, b0), idot(a8, b1), idot(a8, b2), idot(a8, b3));
      let a9 = sa[arow_at + 9u];
      acc9_0 = acc9_0 + vec4<i32>(idot(a9, b0), idot(a9, b1), idot(a9, b2), idot(a9, b3));
      let a10 = sa[arow_at + 10u];
      acc10_0 = acc10_0 + vec4<i32>(idot(a10, b0), idot(a10, b1), idot(a10, b2), idot(a10, b3));
      let a11 = sa[arow_at + 11u];
      acc11_0 = acc11_0 + vec4<i32>(idot(a11, b0), idot(a11, b1), idot(a11, b2), idot(a11, b3));
      let a12 = sa[arow_at + 12u];
      acc12_0 = acc12_0 + vec4<i32>(idot(a12, b0), idot(a12, b1), idot(a12, b2), idot(a12, b3));
      let a13 = sa[arow_at + 13u];
      acc13_0 = acc13_0 + vec4<i32>(idot(a13, b0), idot(a13, b1), idot(a13, b2), idot(a13, b3));
      let a14 = sa[arow_at + 14u];
      acc14_0 = acc14_0 + vec4<i32>(idot(a14, b0), idot(a14, b1), idot(a14, b2), idot(a14, b3));
      let a15 = sa[arow_at + 15u];
      acc15_0 = acc15_0 + vec4<i32>(idot(a15, b0), idot(a15, b1), idot(a15, b2), idot(a15, b3));
    }
    {
      let bcol = 64u + lid.x * 4u;
      let b0 = sb[bcol];
      let b1 = sb[bcol + 1u];
      let b2 = sb[bcol + 2u];
      let b3 = sb[bcol + 3u];
      let arow_at = 128u + lid.y * 16u;
      let a0 = sa[arow_at];
      acc0_0 = acc0_0 + vec4<i32>(idot(a0, b0), idot(a0, b1), idot(a0, b2), idot(a0, b3));
      let a1 = sa[arow_at + 1u];
      acc1_0 = acc1_0 + vec4<i32>(idot(a1, b0), idot(a1, b1), idot(a1, b2), idot(a1, b3));
      let a2 = sa[arow_at + 2u];
      acc2_0 = acc2_0 + vec4<i32>(idot(a2, b0), idot(a2, b1), idot(a2, b2), idot(a2, b3));
      let a3 = sa[arow_at + 3u];
      acc3_0 = acc3_0 + vec4<i32>(idot(a3, b0), idot(a3, b1), idot(a3, b2), idot(a3, b3));
      let a4 = sa[arow_at + 4u];
      acc4_0 = acc4_0 + vec4<i32>(idot(a4, b0), idot(a4, b1), idot(a4, b2), idot(a4, b3));
      let a5 = sa[arow_at + 5u];
      acc5_0 = acc5_0 + vec4<i32>(idot(a5, b0), idot(a5, b1), idot(a5, b2), idot(a5, b3));
      let a6 = sa[arow_at + 6u];
      acc6_0 = acc6_0 + vec4<i32>(idot(a6, b0), idot(a6, b1), idot(a6, b2), idot(a6, b3));
      let a7 = sa[arow_at + 7u];
      acc7_0 = acc7_0 + vec4<i32>(idot(a7, b0), idot(a7, b1), idot(a7, b2), idot(a7, b3));
      let a8 = sa[arow_at + 8u];
      acc8_0 = acc8_0 + vec4<i32>(idot(a8, b0), idot(a8, b1), idot(a8, b2), idot(a8, b3));
      let a9 = sa[arow_at + 9u];
      acc9_0 = acc9_0 + vec4<i32>(idot(a9, b0), idot(a9, b1), idot(a9, b2), idot(a9, b3));
      let a10 = sa[arow_at + 10u];
      acc10_0 = acc10_0 + vec4<i32>(idot(a10, b0), idot(a10, b1), idot(a10, b2), idot(a10, b3));
      let a11 = sa[arow_at + 11u];
      acc11_0 = acc11_0 + vec4<i32>(idot(a11, b0), idot(a11, b1), idot(a11, b2), idot(a11, b3));
      let a12 = sa[arow_at + 12u];
      acc12_0 = acc12_0 + vec4<i32>(idot(a12, b0), idot(a12, b1), idot(a12, b2), idot(a12, b3));
      let a13 = sa[arow_at + 13u];
      acc13_0 = acc13_0 + vec4<i32>(idot(a13, b0), idot(a13, b1), idot(a13, b2), idot(a13, b3));
      let a14 = sa[arow_at + 14u];
      acc14_0 = acc14_0 + vec4<i32>(idot(a14, b0), idot(a14, b1), idot(a14, b2), idot(a14, b3));
      let a15 = sa[arow_at + 15u];
      acc15_0 = acc15_0 + vec4<i32>(idot(a15, b0), idot(a15, b1), idot(a15, b2), idot(a15, b3));
    }
    {
      let bcol = 128u + lid.x * 4u;
      let b0 = sb[bcol];
      let b1 = sb[bcol + 1u];
      let b2 = sb[bcol + 2u];
      let b3 = sb[bcol + 3u];
      let arow_at = 256u + lid.y * 16u;
      let a0 = sa[arow_at];
      acc0_0 = acc0_0 + vec4<i32>(idot(a0, b0), idot(a0, b1), idot(a0, b2), idot(a0, b3));
      let a1 = sa[arow_at + 1u];
      acc1_0 = acc1_0 + vec4<i32>(idot(a1, b0), idot(a1, b1), idot(a1, b2), idot(a1, b3));
      let a2 = sa[arow_at + 2u];
      acc2_0 = acc2_0 + vec4<i32>(idot(a2, b0), idot(a2, b1), idot(a2, b2), idot(a2, b3));
      let a3 = sa[arow_at + 3u];
      acc3_0 = acc3_0 + vec4<i32>(idot(a3, b0), idot(a3, b1), idot(a3, b2), idot(a3, b3));
      let a4 = sa[arow_at + 4u];
      acc4_0 = acc4_0 + vec4<i32>(idot(a4, b0), idot(a4, b1), idot(a4, b2), idot(a4, b3));
      let a5 = sa[arow_at + 5u];
      acc5_0 = acc5_0 + vec4<i32>(idot(a5, b0), idot(a5, b1), idot(a5, b2), idot(a5, b3));
      let a6 = sa[arow_at + 6u];
      acc6_0 = acc6_0 + vec4<i32>(idot(a6, b0), idot(a6, b1), idot(a6, b2), idot(a6, b3));
      let a7 = sa[arow_at + 7u];
      acc7_0 = acc7_0 + vec4<i32>(idot(a7, b0), idot(a7, b1), idot(a7, b2), idot(a7, b3));
      let a8 = sa[arow_at + 8u];
      acc8_0 = acc8_0 + vec4<i32>(idot(a8, b0), idot(a8, b1), idot(a8, b2), idot(a8, b3));
      let a9 = sa[arow_at + 9u];
      acc9_0 = acc9_0 + vec4<i32>(idot(a9, b0), idot(a9, b1), idot(a9, b2), idot(a9, b3));
      let a10 = sa[arow_at + 10u];
      acc10_0 = acc10_0 + vec4<i32>(idot(a10, b0), idot(a10, b1), idot(a10, b2), idot(a10, b3));
      let a11 = sa[arow_at + 11u];
      acc11_0 = acc11_0 + vec4<i32>(idot(a11, b0), idot(a11, b1), idot(a11, b2), idot(a11, b3));
      let a12 = sa[arow_at + 12u];
      acc12_0 = acc12_0 + vec4<i32>(idot(a12, b0), idot(a12, b1), idot(a12, b2), idot(a12, b3));
      let a13 = sa[arow_at + 13u];
      acc13_0 = acc13_0 + vec4<i32>(idot(a13, b0), idot(a13, b1), idot(a13, b2), idot(a13, b3));
      let a14 = sa[arow_at + 14u];
      acc14_0 = acc14_0 + vec4<i32>(idot(a14, b0), idot(a14, b1), idot(a14, b2), idot(a14, b3));
      let a15 = sa[arow_at + 15u];
      acc15_0 = acc15_0 + vec4<i32>(idot(a15, b0), idot(a15, b1), idot(a15, b2), idot(a15, b3));
    }
    {
      let bcol = 192u + lid.x * 4u;
      let b0 = sb[bcol];
      let b1 = sb[bcol + 1u];
      let b2 = sb[bcol + 2u];
      let b3 = sb[bcol + 3u];
      let arow_at = 384u + lid.y * 16u;
      let a0 = sa[arow_at];
      acc0_0 = acc0_0 + vec4<i32>(idot(a0, b0), idot(a0, b1), idot(a0, b2), idot(a0, b3));
      let a1 = sa[arow_at + 1u];
      acc1_0 = acc1_0 + vec4<i32>(idot(a1, b0), idot(a1, b1), idot(a1, b2), idot(a1, b3));
      let a2 = sa[arow_at + 2u];
      acc2_0 = acc2_0 + vec4<i32>(idot(a2, b0), idot(a2, b1), idot(a2, b2), idot(a2, b3));
      let a3 = sa[arow_at + 3u];
      acc3_0 = acc3_0 + vec4<i32>(idot(a3, b0), idot(a3, b1), idot(a3, b2), idot(a3, b3));
      let a4 = sa[arow_at + 4u];
      acc4_0 = acc4_0 + vec4<i32>(idot(a4, b0), idot(a4, b1), idot(a4, b2), idot(a4, b3));
      let a5 = sa[arow_at + 5u];
      acc5_0 = acc5_0 + vec4<i32>(idot(a5, b0), idot(a5, b1), idot(a5, b2), idot(a5, b3));
      let a6 = sa[arow_at + 6u];
      acc6_0 = acc6_0 + vec4<i32>(idot(a6, b0), idot(a6, b1), idot(a6, b2), idot(a6, b3));
      let a7 = sa[arow_at + 7u];
      acc7_0 = acc7_0 + vec4<i32>(idot(a7, b0), idot(a7, b1), idot(a7, b2), idot(a7, b3));
      let a8 = sa[arow_at + 8u];
      acc8_0 = acc8_0 + vec4<i32>(idot(a8, b0), idot(a8, b1), idot(a8, b2), idot(a8, b3));
      let a9 = sa[arow_at + 9u];
      acc9_0 = acc9_0 + vec4<i32>(idot(a9, b0), idot(a9, b1), idot(a9, b2), idot(a9, b3));
      let a10 = sa[arow_at + 10u];
      acc10_0 = acc10_0 + vec4<i32>(idot(a10, b0), idot(a10, b1), idot(a10, b2), idot(a10, b3));
      let a11 = sa[arow_at + 11u];
      acc11_0 = acc11_0 + vec4<i32>(idot(a11, b0), idot(a11, b1), idot(a11, b2), idot(a11, b3));
      let a12 = sa[arow_at + 12u];
      acc12_0 = acc12_0 + vec4<i32>(idot(a12, b0), idot(a12, b1), idot(a12, b2), idot(a12, b3));
      let a13 = sa[arow_at + 13u];
      acc13_0 = acc13_0 + vec4<i32>(idot(a13, b0), idot(a13, b1), idot(a13, b2), idot(a13, b3));
      let a14 = sa[arow_at + 14u];
      acc14_0 = acc14_0 + vec4<i32>(idot(a14, b0), idot(a14, b1), idot(a14, b2), idot(a14, b3));
      let a15 = sa[arow_at + 15u];
      acc15_0 = acc15_0 + vec4<i32>(idot(a15, b0), idot(a15, b1), idot(a15, b2), idot(a15, b3));
    }
    {
      let bcol = 256u + lid.x * 4u;
      let b0 = sb[bcol];
      let b1 = sb[bcol + 1u];
      let b2 = sb[bcol + 2u];
      let b3 = sb[bcol + 3u];
      let arow_at = 512u + lid.y * 16u;
      let a0 = sa[arow_at];
      acc0_0 = acc0_0 + vec4<i32>(idot(a0, b0), idot(a0, b1), idot(a0, b2), idot(a0, b3));
      let a1 = sa[arow_at + 1u];
      acc1_0 = acc1_0 + vec4<i32>(idot(a1, b0), idot(a1, b1), idot(a1, b2), idot(a1, b3));
      let a2 = sa[arow_at + 2u];
      acc2_0 = acc2_0 + vec4<i32>(idot(a2, b0), idot(a2, b1), idot(a2, b2), idot(a2, b3));
      let a3 = sa[arow_at + 3u];
      acc3_0 = acc3_0 + vec4<i32>(idot(a3, b0), idot(a3, b1), idot(a3, b2), idot(a3, b3));
      let a4 = sa[arow_at + 4u];
      acc4_0 = acc4_0 + vec4<i32>(idot(a4, b0), idot(a4, b1), idot(a4, b2), idot(a4, b3));
      let a5 = sa[arow_at + 5u];
      acc5_0 = acc5_0 + vec4<i32>(idot(a5, b0), idot(a5, b1), idot(a5, b2), idot(a5, b3));
      let a6 = sa[arow_at + 6u];
      acc6_0 = acc6_0 + vec4<i32>(idot(a6, b0), idot(a6, b1), idot(a6, b2), idot(a6, b3));
      let a7 = sa[arow_at + 7u];
      acc7_0 = acc7_0 + vec4<i32>(idot(a7, b0), idot(a7, b1), idot(a7, b2), idot(a7, b3));
      let a8 = sa[arow_at + 8u];
      acc8_0 = acc8_0 + vec4<i32>(idot(a8, b0), idot(a8, b1), idot(a8, b2), idot(a8, b3));
      let a9 = sa[arow_at + 9u];
      acc9_0 = acc9_0 + vec4<i32>(idot(a9, b0), idot(a9, b1), idot(a9, b2), idot(a9, b3));
      let a10 = sa[arow_at + 10u];
      acc10_0 = acc10_0 + vec4<i32>(idot(a10, b0), idot(a10, b1), idot(a10, b2), idot(a10, b3));
      let a11 = sa[arow_at + 11u];
      acc11_0 = acc11_0 + vec4<i32>(idot(a11, b0), idot(a11, b1), idot(a11, b2), idot(a11, b3));
      let a12 = sa[arow_at + 12u];
      acc12_0 = acc12_0 + vec4<i32>(idot(a12, b0), idot(a12, b1), idot(a12, b2), idot(a12, b3));
      let a13 = sa[arow_at + 13u];
      acc13_0 = acc13_0 + vec4<i32>(idot(a13, b0), idot(a13, b1), idot(a13, b2), idot(a13, b3));
      let a14 = sa[arow_at + 14u];
      acc14_0 = acc14_0 + vec4<i32>(idot(a14, b0), idot(a14, b1), idot(a14, b2), idot(a14, b3));
      let a15 = sa[arow_at + 15u];
      acc15_0 = acc15_0 + vec4<i32>(idot(a15, b0), idot(a15, b1), idot(a15, b2), idot(a15, b3));
    }
    {
      let bcol = 320u + lid.x * 4u;
      let b0 = sb[bcol];
      let b1 = sb[bcol + 1u];
      let b2 = sb[bcol + 2u];
      let b3 = sb[bcol + 3u];
      let arow_at = 640u + lid.y * 16u;
      let a0 = sa[arow_at];
      acc0_0 = acc0_0 + vec4<i32>(idot(a0, b0), idot(a0, b1), idot(a0, b2), idot(a0, b3));
      let a1 = sa[arow_at + 1u];
      acc1_0 = acc1_0 + vec4<i32>(idot(a1, b0), idot(a1, b1), idot(a1, b2), idot(a1, b3));
      let a2 = sa[arow_at + 2u];
      acc2_0 = acc2_0 + vec4<i32>(idot(a2, b0), idot(a2, b1), idot(a2, b2), idot(a2, b3));
      let a3 = sa[arow_at + 3u];
      acc3_0 = acc3_0 + vec4<i32>(idot(a3, b0), idot(a3, b1), idot(a3, b2), idot(a3, b3));
      let a4 = sa[arow_at + 4u];
      acc4_0 = acc4_0 + vec4<i32>(idot(a4, b0), idot(a4, b1), idot(a4, b2), idot(a4, b3));
      let a5 = sa[arow_at + 5u];
      acc5_0 = acc5_0 + vec4<i32>(idot(a5, b0), idot(a5, b1), idot(a5, b2), idot(a5, b3));
      let a6 = sa[arow_at + 6u];
      acc6_0 = acc6_0 + vec4<i32>(idot(a6, b0), idot(a6, b1), idot(a6, b2), idot(a6, b3));
      let a7 = sa[arow_at + 7u];
      acc7_0 = acc7_0 + vec4<i32>(idot(a7, b0), idot(a7, b1), idot(a7, b2), idot(a7, b3));
      let a8 = sa[arow_at + 8u];
      acc8_0 = acc8_0 + vec4<i32>(idot(a8, b0), idot(a8, b1), idot(a8, b2), idot(a8, b3));
      let a9 = sa[arow_at + 9u];
      acc9_0 = acc9_0 + vec4<i32>(idot(a9, b0), idot(a9, b1), idot(a9, b2), idot(a9, b3));
      let a10 = sa[arow_at + 10u];
      acc10_0 = acc10_0 + vec4<i32>(idot(a10, b0), idot(a10, b1), idot(a10, b2), idot(a10, b3));
      let a11 = sa[arow_at + 11u];
      acc11_0 = acc11_0 + vec4<i32>(idot(a11, b0), idot(a11, b1), idot(a11, b2), idot(a11, b3));
      let a12 = sa[arow_at + 12u];
      acc12_0 = acc12_0 + vec4<i32>(idot(a12, b0), idot(a12, b1), idot(a12, b2), idot(a12, b3));
      let a13 = sa[arow_at + 13u];
      acc13_0 = acc13_0 + vec4<i32>(idot(a13, b0), idot(a13, b1), idot(a13, b2), idot(a13, b3));
      let a14 = sa[arow_at + 14u];
      acc14_0 = acc14_0 + vec4<i32>(idot(a14, b0), idot(a14, b1), idot(a14, b2), idot(a14, b3));
      let a15 = sa[arow_at + 15u];
      acc15_0 = acc15_0 + vec4<i32>(idot(a15, b0), idot(a15, b1), idot(a15, b2), idot(a15, b3));
    }
    {
      let bcol = 384u + lid.x * 4u;
      let b0 = sb[bcol];
      let b1 = sb[bcol + 1u];
      let b2 = sb[bcol + 2u];
      let b3 = sb[bcol + 3u];
      let arow_at = 768u + lid.y * 16u;
      let a0 = sa[arow_at];
      acc0_0 = acc0_0 + vec4<i32>(idot(a0, b0), idot(a0, b1), idot(a0, b2), idot(a0, b3));
      let a1 = sa[arow_at + 1u];
      acc1_0 = acc1_0 + vec4<i32>(idot(a1, b0), idot(a1, b1), idot(a1, b2), idot(a1, b3));
      let a2 = sa[arow_at + 2u];
      acc2_0 = acc2_0 + vec4<i32>(idot(a2, b0), idot(a2, b1), idot(a2, b2), idot(a2, b3));
      let a3 = sa[arow_at + 3u];
      acc3_0 = acc3_0 + vec4<i32>(idot(a3, b0), idot(a3, b1), idot(a3, b2), idot(a3, b3));
      let a4 = sa[arow_at + 4u];
      acc4_0 = acc4_0 + vec4<i32>(idot(a4, b0), idot(a4, b1), idot(a4, b2), idot(a4, b3));
      let a5 = sa[arow_at + 5u];
      acc5_0 = acc5_0 + vec4<i32>(idot(a5, b0), idot(a5, b1), idot(a5, b2), idot(a5, b3));
      let a6 = sa[arow_at + 6u];
      acc6_0 = acc6_0 + vec4<i32>(idot(a6, b0), idot(a6, b1), idot(a6, b2), idot(a6, b3));
      let a7 = sa[arow_at + 7u];
      acc7_0 = acc7_0 + vec4<i32>(idot(a7, b0), idot(a7, b1), idot(a7, b2), idot(a7, b3));
      let a8 = sa[arow_at + 8u];
      acc8_0 = acc8_0 + vec4<i32>(idot(a8, b0), idot(a8, b1), idot(a8, b2), idot(a8, b3));
      let a9 = sa[arow_at + 9u];
      acc9_0 = acc9_0 + vec4<i32>(idot(a9, b0), idot(a9, b1), idot(a9, b2), idot(a9, b3));
      let a10 = sa[arow_at + 10u];
      acc10_0 = acc10_0 + vec4<i32>(idot(a10, b0), idot(a10, b1), idot(a10, b2), idot(a10, b3));
      let a11 = sa[arow_at + 11u];
      acc11_0 = acc11_0 + vec4<i32>(idot(a11, b0), idot(a11, b1), idot(a11, b2), idot(a11, b3));
      let a12 = sa[arow_at + 12u];
      acc12_0 = acc12_0 + vec4<i32>(idot(a12, b0), idot(a12, b1), idot(a12, b2), idot(a12, b3));
      let a13 = sa[arow_at + 13u];
      acc13_0 = acc13_0 + vec4<i32>(idot(a13, b0), idot(a13, b1), idot(a13, b2), idot(a13, b3));
      let a14 = sa[arow_at + 14u];
      acc14_0 = acc14_0 + vec4<i32>(idot(a14, b0), idot(a14, b1), idot(a14, b2), idot(a14, b3));
      let a15 = sa[arow_at + 15u];
      acc15_0 = acc15_0 + vec4<i32>(idot(a15, b0), idot(a15, b1), idot(a15, b2), idot(a15, b3));
    }
    {
      let bcol = 448u + lid.x * 4u;
      let b0 = sb[bcol];
      let b1 = sb[bcol + 1u];
      let b2 = sb[bcol + 2u];
      let b3 = sb[bcol + 3u];
      let arow_at = 896u + lid.y * 16u;
      let a0 = sa[arow_at];
      acc0_0 = acc0_0 + vec4<i32>(idot(a0, b0), idot(a0, b1), idot(a0, b2), idot(a0, b3));
      let a1 = sa[arow_at + 1u];
      acc1_0 = acc1_0 + vec4<i32>(idot(a1, b0), idot(a1, b1), idot(a1, b2), idot(a1, b3));
      let a2 = sa[arow_at + 2u];
      acc2_0 = acc2_0 + vec4<i32>(idot(a2, b0), idot(a2, b1), idot(a2, b2), idot(a2, b3));
      let a3 = sa[arow_at + 3u];
      acc3_0 = acc3_0 + vec4<i32>(idot(a3, b0), idot(a3, b1), idot(a3, b2), idot(a3, b3));
      let a4 = sa[arow_at + 4u];
      acc4_0 = acc4_0 + vec4<i32>(idot(a4, b0), idot(a4, b1), idot(a4, b2), idot(a4, b3));
      let a5 = sa[arow_at + 5u];
      acc5_0 = acc5_0 + vec4<i32>(idot(a5, b0), idot(a5, b1), idot(a5, b2), idot(a5, b3));
      let a6 = sa[arow_at + 6u];
      acc6_0 = acc6_0 + vec4<i32>(idot(a6, b0), idot(a6, b1), idot(a6, b2), idot(a6, b3));
      let a7 = sa[arow_at + 7u];
      acc7_0 = acc7_0 + vec4<i32>(idot(a7, b0), idot(a7, b1), idot(a7, b2), idot(a7, b3));
      let a8 = sa[arow_at + 8u];
      acc8_0 = acc8_0 + vec4<i32>(idot(a8, b0), idot(a8, b1), idot(a8, b2), idot(a8, b3));
      let a9 = sa[arow_at + 9u];
      acc9_0 = acc9_0 + vec4<i32>(idot(a9, b0), idot(a9, b1), idot(a9, b2), idot(a9, b3));
      let a10 = sa[arow_at + 10u];
      acc10_0 = acc10_0 + vec4<i32>(idot(a10, b0), idot(a10, b1), idot(a10, b2), idot(a10, b3));
      let a11 = sa[arow_at + 11u];
      acc11_0 = acc11_0 + vec4<i32>(idot(a11, b0), idot(a11, b1), idot(a11, b2), idot(a11, b3));
      let a12 = sa[arow_at + 12u];
      acc12_0 = acc12_0 + vec4<i32>(idot(a12, b0), idot(a12, b1), idot(a12, b2), idot(a12, b3));
      let a13 = sa[arow_at + 13u];
      acc13_0 = acc13_0 + vec4<i32>(idot(a13, b0), idot(a13, b1), idot(a13, b2), idot(a13, b3));
      let a14 = sa[arow_at + 14u];
      acc14_0 = acc14_0 + vec4<i32>(idot(a14, b0), idot(a14, b1), idot(a14, b2), idot(a14, b3));
      let a15 = sa[arow_at + 15u];
      acc15_0 = acc15_0 + vec4<i32>(idot(a15, b0), idot(a15, b1), idot(a15, b2), idot(a15, b3));
    }
      workgroupBarrier();
    }
    // Flush to f32 exactly at the group boundary (MUST: single rounding of fma; xs is not
    // applied until the end)
    let ws0 = vec4<f32>(wscale[wsb0 + gi], wscale[wsb1 + gi], wscale[wsb2 + gi], wscale[wsb3 + gi]);
    accf0_0 = fma(vec4<f32>(acc0_0), ws0, accf0_0);
    accf1_0 = fma(vec4<f32>(acc1_0), ws0, accf1_0);
    accf2_0 = fma(vec4<f32>(acc2_0), ws0, accf2_0);
    accf3_0 = fma(vec4<f32>(acc3_0), ws0, accf3_0);
    accf4_0 = fma(vec4<f32>(acc4_0), ws0, accf4_0);
    accf5_0 = fma(vec4<f32>(acc5_0), ws0, accf5_0);
    accf6_0 = fma(vec4<f32>(acc6_0), ws0, accf6_0);
    accf7_0 = fma(vec4<f32>(acc7_0), ws0, accf7_0);
    accf8_0 = fma(vec4<f32>(acc8_0), ws0, accf8_0);
    accf9_0 = fma(vec4<f32>(acc9_0), ws0, accf9_0);
    accf10_0 = fma(vec4<f32>(acc10_0), ws0, accf10_0);
    accf11_0 = fma(vec4<f32>(acc11_0), ws0, accf11_0);
    accf12_0 = fma(vec4<f32>(acc12_0), ws0, accf12_0);
    accf13_0 = fma(vec4<f32>(acc13_0), ws0, accf13_0);
    accf14_0 = fma(vec4<f32>(acc14_0), ws0, accf14_0);
    accf15_0 = fma(vec4<f32>(acc15_0), ws0, accf15_0);
  }
  let n4 = dims.n / 4u;
  let ocq0 = ocol / 4u;
  if (ocq0 < n4) {
    // n % 4 == 0 and ocq0 < n4, so oc + 3 < n (bias is always f32 — ADR 0006)
    let oc = ocq0 * 4u;
    let biasv = vec4<f32>(bias[oc], bias[oc + 1u], bias[oc + 2u], bias[oc + 3u]);
    // MUST: xs goes to the final fma (single rounding) — wscale was folded at the group boundaries
    if (orow0 < dims.m) {
      out[orow0 * n4 + ocq0] = fma(accf0_0, vec4<f32>(xscale[orow0]), biasv);
    }
    if (orow1 < dims.m) {
      out[orow1 * n4 + ocq0] = fma(accf1_0, vec4<f32>(xscale[orow1]), biasv);
    }
    if (orow2 < dims.m) {
      out[orow2 * n4 + ocq0] = fma(accf2_0, vec4<f32>(xscale[orow2]), biasv);
    }
    if (orow3 < dims.m) {
      out[orow3 * n4 + ocq0] = fma(accf3_0, vec4<f32>(xscale[orow3]), biasv);
    }
    if (orow4 < dims.m) {
      out[orow4 * n4 + ocq0] = fma(accf4_0, vec4<f32>(xscale[orow4]), biasv);
    }
    if (orow5 < dims.m) {
      out[orow5 * n4 + ocq0] = fma(accf5_0, vec4<f32>(xscale[orow5]), biasv);
    }
    if (orow6 < dims.m) {
      out[orow6 * n4 + ocq0] = fma(accf6_0, vec4<f32>(xscale[orow6]), biasv);
    }
    if (orow7 < dims.m) {
      out[orow7 * n4 + ocq0] = fma(accf7_0, vec4<f32>(xscale[orow7]), biasv);
    }
    if (orow8 < dims.m) {
      out[orow8 * n4 + ocq0] = fma(accf8_0, vec4<f32>(xscale[orow8]), biasv);
    }
    if (orow9 < dims.m) {
      out[orow9 * n4 + ocq0] = fma(accf9_0, vec4<f32>(xscale[orow9]), biasv);
    }
    if (orow10 < dims.m) {
      out[orow10 * n4 + ocq0] = fma(accf10_0, vec4<f32>(xscale[orow10]), biasv);
    }
    if (orow11 < dims.m) {
      out[orow11 * n4 + ocq0] = fma(accf11_0, vec4<f32>(xscale[orow11]), biasv);
    }
    if (orow12 < dims.m) {
      out[orow12 * n4 + ocq0] = fma(accf12_0, vec4<f32>(xscale[orow12]), biasv);
    }
    if (orow13 < dims.m) {
      out[orow13 * n4 + ocq0] = fma(accf13_0, vec4<f32>(xscale[orow13]), biasv);
    }
    if (orow14 < dims.m) {
      out[orow14 * n4 + ocq0] = fma(accf14_0, vec4<f32>(xscale[orow14]), biasv);
    }
    if (orow15 < dims.m) {
      out[orow15 * n4 + ocq0] = fma(accf15_0, vec4<f32>(xscale[orow15]), biasv);
    }
  }
}
