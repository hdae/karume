// karume linear (x[m,k] · wᵀ[k,n] + bias[n], f32, register tile 64x32 / 4x4 per thread / wg 8x16 + vec4)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<vec4<f32>>;

// Shared A tile (64 rows × K 16, scalar storage) and
// shared B tile (K 16 × 8 column quads, column direction grouped into vec4)
var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<vec4<f32>, 128>;

@compute @workgroup_size(8, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 8u + lid.x;
  let k4 = dims.k / 4u;
  let n4 = dims.n / 4u;
  // A tile assignment (64 rows × 4 quads over 128 threads in 2 rounds)
  let ar = tid / 4u;
  let aq = tid % 4u;
  let arow0 = wid.y * 64u + ar;
  let arow_base0 = arow0 * k4;
  let sa_base0 = aq * 256u + ar;
  let arow1 = arow0 + 32u;
  let arow_base1 = arow_base0 + 32u * k4;
  let sa_base1 = sa_base0 + 32u;
  // W tile assignment (32 output channels × 4 quads over 128 threads in 1 rounds)
  let wc0 = tid / 4u;
  let wq = tid % 4u;
  let wcol0 = wid.x * 32u + wc0;
  let wrow_base0 = wcol0 * dims.k;
  // Placed transposed on the shared-memory side
  // (column quad = wc / 4, component = wc % 4)
  let wsq0 = wc0 / 4u;
  let wsl0 = wc0 % 4u;
  let sb_base0 = (wq * 4u) * 8u + wsq0;
  // The loop condition is uniform (dims is a uniform buffer) — needed for the inner workgroupBarrier
  // to meet WGSL's uniformity requirement
  let tiles = (dims.k + 15u) / 16u;
  var acc0_0 = vec4<f32>(0.0);
  var acc1_0 = vec4<f32>(0.0);
  var acc2_0 = vec4<f32>(0.0);
  var acc3_0 = vec4<f32>(0.0);
  for (var t = 0u; t < tiles; t = t + 1u) {
    // Fill out-of-range entries with 0. They do not contribute to the inner product, so
    // remainder shapes give the same result
    let ak0 = t * 16u + aq * 4u;
    var av0 = vec4<f32>(0.0);
    if (arow0 < dims.m && ak0 < dims.k) {
      av0 = x[arow_base0 + t * 4u + aq];
    }
    sa[sa_base0] = av0.x;
    sa[sa_base0 + 64u] = av0.y;
    sa[sa_base0 + 128u] = av0.z;
    sa[sa_base0 + 192u] = av0.w;
    var av1 = vec4<f32>(0.0);
    if (arow1 < dims.m && ak0 < dims.k) {
      av1 = x[arow_base1 + t * 4u + aq];
    }
    sa[sa_base1] = av1.x;
    sa[sa_base1 + 64u] = av1.y;
    sa[sa_base1 + 128u] = av1.z;
    sa[sa_base1 + 192u] = av1.w;
    let wk0 = t * 16u + wq * 4u;
    var wv0 = vec4<f32>(0.0);
    if (wcol0 < dims.n && wk0 < dims.k) {
      wv0 = w[(wrow_base0 + wk0) >> 2u];
    }
    switch wsl0 {
      case 0u: {
        sb[sb_base0].x = wv0.x;
        sb[sb_base0 + 8u].x = wv0.y;
        sb[sb_base0 + 16u].x = wv0.z;
        sb[sb_base0 + 24u].x = wv0.w;
      }
      case 1u: {
        sb[sb_base0].y = wv0.x;
        sb[sb_base0 + 8u].y = wv0.y;
        sb[sb_base0 + 16u].y = wv0.z;
        sb[sb_base0 + 24u].y = wv0.w;
      }
      case 2u: {
        sb[sb_base0].z = wv0.x;
        sb[sb_base0 + 8u].z = wv0.y;
        sb[sb_base0 + 16u].z = wv0.z;
        sb[sb_base0 + 24u].z = wv0.w;
      }
      default: {
        sb[sb_base0].w = wv0.x;
        sb[sb_base0 + 8u].w = wv0.y;
        sb[sb_base0 + 16u].w = wv0.z;
        sb[sb_base0 + 24u].w = wv0.w;
      }
    }
    workgroupBarrier();
    // 5 shared loads (1 vec4 of B + 4 scalars of A) give 16 MACs.
    // The reduction is sequential in ascending k, and the addition order per output element matches
    // the 16×16 kernel with 1 output per thread exactly.
    {
      let kk = 0u;
      let bv0 = sb[kk * 8u + lid.x];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 4u + 0u] * bv0;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 4u + 1u] * bv0;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 4u + 2u] * bv0;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 4u + 3u] * bv0;
    }
    {
      let kk = 1u;
      let bv0 = sb[kk * 8u + lid.x];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 4u + 0u] * bv0;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 4u + 1u] * bv0;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 4u + 2u] * bv0;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 4u + 3u] * bv0;
    }
    {
      let kk = 2u;
      let bv0 = sb[kk * 8u + lid.x];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 4u + 0u] * bv0;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 4u + 1u] * bv0;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 4u + 2u] * bv0;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 4u + 3u] * bv0;
    }
    {
      let kk = 3u;
      let bv0 = sb[kk * 8u + lid.x];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 4u + 0u] * bv0;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 4u + 1u] * bv0;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 4u + 2u] * bv0;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 4u + 3u] * bv0;
    }
    {
      let kk = 4u;
      let bv0 = sb[kk * 8u + lid.x];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 4u + 0u] * bv0;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 4u + 1u] * bv0;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 4u + 2u] * bv0;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 4u + 3u] * bv0;
    }
    {
      let kk = 5u;
      let bv0 = sb[kk * 8u + lid.x];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 4u + 0u] * bv0;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 4u + 1u] * bv0;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 4u + 2u] * bv0;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 4u + 3u] * bv0;
    }
    {
      let kk = 6u;
      let bv0 = sb[kk * 8u + lid.x];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 4u + 0u] * bv0;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 4u + 1u] * bv0;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 4u + 2u] * bv0;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 4u + 3u] * bv0;
    }
    {
      let kk = 7u;
      let bv0 = sb[kk * 8u + lid.x];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 4u + 0u] * bv0;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 4u + 1u] * bv0;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 4u + 2u] * bv0;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 4u + 3u] * bv0;
    }
    {
      let kk = 8u;
      let bv0 = sb[kk * 8u + lid.x];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 4u + 0u] * bv0;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 4u + 1u] * bv0;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 4u + 2u] * bv0;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 4u + 3u] * bv0;
    }
    {
      let kk = 9u;
      let bv0 = sb[kk * 8u + lid.x];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 4u + 0u] * bv0;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 4u + 1u] * bv0;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 4u + 2u] * bv0;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 4u + 3u] * bv0;
    }
    {
      let kk = 10u;
      let bv0 = sb[kk * 8u + lid.x];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 4u + 0u] * bv0;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 4u + 1u] * bv0;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 4u + 2u] * bv0;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 4u + 3u] * bv0;
    }
    {
      let kk = 11u;
      let bv0 = sb[kk * 8u + lid.x];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 4u + 0u] * bv0;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 4u + 1u] * bv0;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 4u + 2u] * bv0;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 4u + 3u] * bv0;
    }
    {
      let kk = 12u;
      let bv0 = sb[kk * 8u + lid.x];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 4u + 0u] * bv0;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 4u + 1u] * bv0;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 4u + 2u] * bv0;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 4u + 3u] * bv0;
    }
    {
      let kk = 13u;
      let bv0 = sb[kk * 8u + lid.x];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 4u + 0u] * bv0;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 4u + 1u] * bv0;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 4u + 2u] * bv0;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 4u + 3u] * bv0;
    }
    {
      let kk = 14u;
      let bv0 = sb[kk * 8u + lid.x];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 4u + 0u] * bv0;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 4u + 1u] * bv0;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 4u + 2u] * bv0;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 4u + 3u] * bv0;
    }
    {
      let kk = 15u;
      let bv0 = sb[kk * 8u + lid.x];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 4u + 0u] * bv0;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 4u + 1u] * bv0;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 4u + 2u] * bv0;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 4u + 3u] * bv0;
    }
    workgroupBarrier();
  }
  let ocq0 = wid.x * 8u + lid.x;
  let orow0 = wid.y * 64u + lid.y * 4u;
  let orow1 = orow0 + 1u;
  let orow2 = orow0 + 2u;
  let orow3 = orow0 + 3u;
  if (ocq0 < n4) {
    // n % 4 == 0 and ocq0 < n4, so oc + 3 < n (bias is always f32 — ADR 0006)
    let oc = ocq0 * 4u;
    let biasv = vec4<f32>(bias[oc], bias[oc + 1u], bias[oc + 2u], bias[oc + 3u]);
    if (orow0 < dims.m) {
      out[orow0 * n4 + ocq0] = acc0_0 + biasv;
    }
    if (orow1 < dims.m) {
      out[orow1 * n4 + ocq0] = acc1_0 + biasv;
    }
    if (orow2 < dims.m) {
      out[orow2 * n4 + ocq0] = acc2_0 + biasv;
    }
    if (orow3 < dims.m) {
      out[orow3 * n4 + ocq0] = acc3_0 + biasv;
    }
  }
}
