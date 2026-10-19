// karume linear (x[m,k] · wᵀ[k,n] + bias[n], f32, register tile 16x16 / 1x4 per thread / wg 4x16 + vec4)
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

// Shared A tile (16 rows × K 16, scalar storage) and
// shared B tile (K 16 × 4 column quads, column direction grouped into vec4)
var<workgroup> sa: array<f32, 256>;
var<workgroup> sb: array<vec4<f32>, 64>;

@compute @workgroup_size(4, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 4u + lid.x;
  let k4 = dims.k / 4u;
  let n4 = dims.n / 4u;
  // A tile assignment (16 rows × 4 quads over 64 threads in 1 rounds)
  let ar = tid / 4u;
  let aq = tid % 4u;
  let arow0 = wid.y * 16u + ar;
  let arow_base0 = arow0 * k4;
  let sa_base0 = ar * 16u + aq * 4u;
  // W tile assignment (16 output channels × 4 quads over 64 threads in 1 rounds)
  let wc0 = tid / 4u;
  let wq = tid % 4u;
  let wcol0 = wid.x * 16u + wc0;
  let wrow_base0 = wcol0 * dims.k;
  // Placed transposed on the shared-memory side
  // (column quad = wc / 4, component = wc % 4)
  let wsq0 = wc0 / 4u;
  let wsl0 = wc0 % 4u;
  let sb_base0 = (wq * 4u) * 4u + wsq0;
  // The loop condition is uniform (dims is a uniform buffer) — needed for the inner workgroupBarrier
  // to meet WGSL's uniformity requirement
  let tiles = (dims.k + 15u) / 16u;
  var acc0_0 = vec4<f32>(0.0);
  for (var t = 0u; t < tiles; t = t + 1u) {
    // Fill out-of-range entries with 0. They do not contribute to the inner product, so
    // remainder shapes give the same result
    let ak0 = t * 16u + aq * 4u;
    var av0 = vec4<f32>(0.0);
    if (arow0 < dims.m && ak0 < dims.k) {
      av0 = x[arow_base0 + t * 4u + aq];
    }
    sa[sa_base0] = av0.x;
    sa[sa_base0 + 1u] = av0.y;
    sa[sa_base0 + 2u] = av0.z;
    sa[sa_base0 + 3u] = av0.w;
    let wk0 = t * 16u + wq * 4u;
    var wv0 = vec4<f32>(0.0);
    if (wcol0 < dims.n && wk0 < dims.k) {
      wv0 = w[(wrow_base0 + wk0) >> 2u];
    }
    switch wsl0 {
      case 0u: {
        sb[sb_base0].x = wv0.x;
        sb[sb_base0 + 4u].x = wv0.y;
        sb[sb_base0 + 8u].x = wv0.z;
        sb[sb_base0 + 12u].x = wv0.w;
      }
      case 1u: {
        sb[sb_base0].y = wv0.x;
        sb[sb_base0 + 4u].y = wv0.y;
        sb[sb_base0 + 8u].y = wv0.z;
        sb[sb_base0 + 12u].y = wv0.w;
      }
      case 2u: {
        sb[sb_base0].z = wv0.x;
        sb[sb_base0 + 4u].z = wv0.y;
        sb[sb_base0 + 8u].z = wv0.z;
        sb[sb_base0 + 12u].z = wv0.w;
      }
      default: {
        sb[sb_base0].w = wv0.x;
        sb[sb_base0 + 4u].w = wv0.y;
        sb[sb_base0 + 8u].w = wv0.z;
        sb[sb_base0 + 12u].w = wv0.w;
      }
    }
    workgroupBarrier();
    // 2 shared loads (1 vec4 of B + 1 scalars of A) give 4 MACs.
    // The reduction is sequential in ascending k, and the addition order per output element matches
    // the 16×16 kernel with 1 output per thread exactly.
    for (var kk = 0u; kk < 16u; kk = kk + 1u) {
      let bv0 = sb[kk * 4u + lid.x];
      acc0_0 = acc0_0 + sa[(lid.y * 1u + 0u) * 16u + kk] * bv0;
    }
    workgroupBarrier();
  }
  let ocq0 = wid.x * 4u + lid.x;
  let orow0 = wid.y * 16u + lid.y * 1u;
  if (ocq0 < n4) {
    // n % 4 == 0 and ocq0 < n4, so oc + 3 < n (bias is always f32 — ADR 0006)
    let oc = ocq0 * 4u;
    let biasv = vec4<f32>(bias[oc], bias[oc + 1u], bias[oc + 2u], bias[oc + 3u]);
    if (orow0 < dims.m) {
      out[orow0 * n4 + ocq0] = acc0_0 + biasv;
    }
  }
}
