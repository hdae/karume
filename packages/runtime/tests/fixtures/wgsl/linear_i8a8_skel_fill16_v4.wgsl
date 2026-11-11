// karume linear (x[m,k] · wᵀ[k,n] + bias[n], integer dot product of per-token i8 activations × i8 weights, tile 128x64 / 16x4 per thread / wg 16x8 / K 32 / vec4 skeleton + vec4)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> xq: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> wscale: array<f32>;
@group(0) @binding(6) var<storage, read> xscale: array<f32>;

// integer dot product (WGSL language extension packed_4x8_integer_dot_product); returns **the same integer** as the emulated variant
fn idot(a: u32, b: u32) -> i32 {
  return dot4I8Packed(a, b);
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
  // k % 16 == 0 on this form, so a row is whole vec4s (16 i8 each)
  let k16 = dims.k / 16u;
  // x tile assignment (vec4 fill: 128 rows × 2 quads, 2 slots)
  let aq = tid % 2u;
  let ar = tid / 2u;
  let sa_at = aq * 512u + ar;
  let arow0 = wid.y * 128u + ar;
  let arow_base0 = arow0 * k16;
  let arow1 = wid.y * 128u + ar + 64u;
  let arow_base1 = arow1 * k16;
  // W tile assignment (vec4 fill: 64 output channels × 2 quads, 1 slots)
  let wq4 = tid % 2u;
  let wc = tid / 2u;
  let sb_at = wq4 * 256u + wc;
  let wcol0 = wid.x * 64u + wc;
  let wrow_base0 = wcol0 * k16;
  // The loop condition is uniform (dims is a uniform buffer) — needed for the inner
  // workgroupBarrier to meet WGSL's uniformity requirement
  let tiles = (k4 + 7u) / 8u;
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
  for (var t = 0u; t < tiles; t = t + 1u) {
    let aquad = t * 2u + aq;
    var av0 = vec4<u32>(0u);
    if (arow0 < dims.m && aquad < k16) {
      av0 = xq[arow_base0 + aquad];
    }
    sa[sa_at] = av0.x;
    sa[sa_at + 128u] = av0.y;
    sa[sa_at + 256u] = av0.z;
    sa[sa_at + 384u] = av0.w;
    var av1 = vec4<u32>(0u);
    if (arow1 < dims.m && aquad < k16) {
      av1 = xq[arow_base1 + aquad];
    }
    sa[sa_at + 64u] = av1.x;
    sa[sa_at + 192u] = av1.y;
    sa[sa_at + 320u] = av1.z;
    sa[sa_at + 448u] = av1.w;
    let wquad = t * 2u + wq4;
    var wv0 = vec4<u32>(0u);
    if (wcol0 < dims.n && wquad < k16) {
      wv0 = w[wrow_base0 + wquad];
    }
    sb[sb_at] = wv0.x;
    sb[sb_at + 64u] = wv0.y;
    sb[sb_at + 128u] = wv0.z;
    sb[sb_at + 192u] = wv0.w;
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
  let n4 = dims.n / 4u;
  let ocq0 = wid.x * 16u + lid.x * 1u;
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
  if (ocq0 < n4) {
    // n % 4 == 0 and ocq0 < n4, so oc + 3 < n (bias is always f32 — ADR 0006)
    let oc = ocq0 * 4u;
    let ws = vec4<f32>(wscale[oc], wscale[oc + 1u], wscale[oc + 2u], wscale[oc + 3u]);
    let biasv = vec4<f32>(bias[oc], bias[oc + 1u], bias[oc + 2u], bias[oc + 3u]);
    // MUST: fold xs·wscale into one f32 first; the multiply-add is fma (single rounding)
    if (orow0 < dims.m) {
      out[orow0 * n4 + ocq0] = fma(vec4<f32>(acc0_0), xscale[orow0] * ws, biasv);
    }
    if (orow1 < dims.m) {
      out[orow1 * n4 + ocq0] = fma(vec4<f32>(acc1_0), xscale[orow1] * ws, biasv);
    }
    if (orow2 < dims.m) {
      out[orow2 * n4 + ocq0] = fma(vec4<f32>(acc2_0), xscale[orow2] * ws, biasv);
    }
    if (orow3 < dims.m) {
      out[orow3 * n4 + ocq0] = fma(vec4<f32>(acc3_0), xscale[orow3] * ws, biasv);
    }
    if (orow4 < dims.m) {
      out[orow4 * n4 + ocq0] = fma(vec4<f32>(acc4_0), xscale[orow4] * ws, biasv);
    }
    if (orow5 < dims.m) {
      out[orow5 * n4 + ocq0] = fma(vec4<f32>(acc5_0), xscale[orow5] * ws, biasv);
    }
    if (orow6 < dims.m) {
      out[orow6 * n4 + ocq0] = fma(vec4<f32>(acc6_0), xscale[orow6] * ws, biasv);
    }
    if (orow7 < dims.m) {
      out[orow7 * n4 + ocq0] = fma(vec4<f32>(acc7_0), xscale[orow7] * ws, biasv);
    }
    if (orow8 < dims.m) {
      out[orow8 * n4 + ocq0] = fma(vec4<f32>(acc8_0), xscale[orow8] * ws, biasv);
    }
    if (orow9 < dims.m) {
      out[orow9 * n4 + ocq0] = fma(vec4<f32>(acc9_0), xscale[orow9] * ws, biasv);
    }
    if (orow10 < dims.m) {
      out[orow10 * n4 + ocq0] = fma(vec4<f32>(acc10_0), xscale[orow10] * ws, biasv);
    }
    if (orow11 < dims.m) {
      out[orow11 * n4 + ocq0] = fma(vec4<f32>(acc11_0), xscale[orow11] * ws, biasv);
    }
    if (orow12 < dims.m) {
      out[orow12 * n4 + ocq0] = fma(vec4<f32>(acc12_0), xscale[orow12] * ws, biasv);
    }
    if (orow13 < dims.m) {
      out[orow13 * n4 + ocq0] = fma(vec4<f32>(acc13_0), xscale[orow13] * ws, biasv);
    }
    if (orow14 < dims.m) {
      out[orow14 * n4 + ocq0] = fma(vec4<f32>(acc14_0), xscale[orow14] * ws, biasv);
    }
    if (orow15 < dims.m) {
      out[orow15 * n4 + ocq0] = fma(vec4<f32>(acc15_0), xscale[orow15] * ws, biasv);
    }
  }
}
