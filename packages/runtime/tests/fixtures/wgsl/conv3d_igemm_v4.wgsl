// karume conv3d (x[Cin,T,H,W] * W[Cout,Cin,Kt,Kh,Kw] + b[Cout], f32, implicit GEMM レジスタ 64x128 タイル / 1 スレッド 8x8 / wg 16x8 + vec4)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
  channels_in: u32,
  time_in: u32,
  height_in: u32,
  width_in: u32,
  height_out: u32,
  width_out: u32,
  kernel_t: u32,
  kernel_h: u32,
  kernel_w: u32,
  stride_t: u32,
  stride_h: u32,
  stride_w: u32,
  padding_t: u32,
  padding_h: u32,
  padding_w: u32,
  dilation_t: u32,
  dilation_h: u32,
  dilation_w: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<vec4<f32>>;

// Xcol[k][n] = x[ic, ot·st − pt + kt·dt, oy·sh − ph + kh·dh, ox·sw − pw + kw·dw] (0 out of range).
// n = (ot·Hout + oy)·Wout + ox. Unbatched, so there is no batch base
fn xcol(ic: u32, kt: i32, ky: i32, kx: i32, n: u32) -> f32 {
  let plane = dims.height_out * dims.width_out;
  let pixel = n % plane;
  let it = i32((n / plane) * dims.stride_t) + kt;
  let iy = i32((pixel / dims.width_out) * dims.stride_h) + ky;
  let ix = i32((pixel % dims.width_out) * dims.stride_w) + kx;
  if (it < 0 || u32(it) >= dims.time_in || iy < 0 || u32(iy) >= dims.height_in ||
      ix < 0 || u32(ix) >= dims.width_in) {
    return 0.0;
  }
  return x[((ic * dims.time_in + u32(it)) * dims.height_in + u32(iy)) * dims.width_in + u32(ix)];
}

// Shared A tile (64 rows × K 16, scalar storage) and
// shared B tile (K 16 × 32 column quads, column direction grouped into vec4)
var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<vec4<f32>, 512>;

@compute @workgroup_size(16, 8)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 16u + lid.x;
  let n4 = dims.n / 4u;
  // A tile assignment (64 rows × 4 quads over 128 threads in 2 rounds)
  let ar = tid / 4u;
  let aq = tid % 4u;
  let arow0 = wid.y * 64u + ar;
  let arow_base0 = arow0 * dims.k;
  let sa_base0 = ar * 16u + aq * 4u;
  let arow1 = arow0 + 32u;
  let arow_base1 = arow_base0 + 32u * dims.k;
  let sa_base1 = sa_base0 + 512u;
  // B tile assignment (K 16 rows × 32 column quads over 128 threads in 4 rounds)
  let bk0 = tid / 32u;
  let bcq = tid % 32u;
  let bc4 = wid.x * 32u + bcq;
  let bk1 = bk0 + 4u;
  let bk2 = bk0 + 8u;
  let bk3 = bk0 + 12u;
  // Invariant in the K tile loop (the step for splitting flat k into (ic, kt, kh, kw)と、quad の先頭 n を (ot, 画素) へ割る刻み)
  let khw = dims.kernel_h * dims.kernel_w;
  let kthw = dims.kernel_t * khw;
  let hw_out = dims.height_out * dims.width_out;
  // The loop condition is uniform (dims is a uniform buffer) — needed for the inner workgroupBarrier
  // to meet WGSL's uniformity requirement
  let tiles = (dims.k + 15u) / 16u;
  let bias0 = wid.y * 64u + lid.y * 8u;
  var acc0_0 = vec4<f32>(bias[min(bias0, dims.m - 1u)]);
  var acc0_1 = vec4<f32>(bias[min(bias0, dims.m - 1u)]);
  var acc1_0 = vec4<f32>(bias[min(bias0 + 1u, dims.m - 1u)]);
  var acc1_1 = vec4<f32>(bias[min(bias0 + 1u, dims.m - 1u)]);
  var acc2_0 = vec4<f32>(bias[min(bias0 + 2u, dims.m - 1u)]);
  var acc2_1 = vec4<f32>(bias[min(bias0 + 2u, dims.m - 1u)]);
  var acc3_0 = vec4<f32>(bias[min(bias0 + 3u, dims.m - 1u)]);
  var acc3_1 = vec4<f32>(bias[min(bias0 + 3u, dims.m - 1u)]);
  var acc4_0 = vec4<f32>(bias[min(bias0 + 4u, dims.m - 1u)]);
  var acc4_1 = vec4<f32>(bias[min(bias0 + 4u, dims.m - 1u)]);
  var acc5_0 = vec4<f32>(bias[min(bias0 + 5u, dims.m - 1u)]);
  var acc5_1 = vec4<f32>(bias[min(bias0 + 5u, dims.m - 1u)]);
  var acc6_0 = vec4<f32>(bias[min(bias0 + 6u, dims.m - 1u)]);
  var acc6_1 = vec4<f32>(bias[min(bias0 + 6u, dims.m - 1u)]);
  var acc7_0 = vec4<f32>(bias[min(bias0 + 7u, dims.m - 1u)]);
  var acc7_1 = vec4<f32>(bias[min(bias0 + 7u, dims.m - 1u)]);
  for (var t = 0u; t < tiles; t = t + 1u) {
    // Fill out-of-range entries with 0. They do not contribute to the inner product, so a K
    // remainder gives the same result
    let ak0 = t * 16u + aq * 4u;
    var av0 = vec4<f32>(0.0);
    if (arow0 < dims.m && ak0 < dims.k) {
      av0 = w[(arow_base0 + ak0) >> 2u];
    }
    sa[sa_base0] = av0.x;
    sa[sa_base0 + 1u] = av0.y;
    sa[sa_base0 + 2u] = av0.z;
    sa[sa_base0 + 3u] = av0.w;
    var av1 = vec4<f32>(0.0);
    if (arow1 < dims.m && ak0 < dims.k) {
      av1 = w[(arow_base1 + ak0) >> 2u];
    }
    sa[sa_base1] = av1.x;
    sa[sa_base1 + 1u] = av1.y;
    sa[sa_base1 + 2u] = av1.z;
    sa[sa_base1 + 3u] = av1.w;
    let brow0 = t * 16u + bk0;
    var bv4_0 = vec4<f32>(0.0);
    if (brow0 < dims.k && bc4 < n4) {
      let ic = brow0 / kthw;
      let kr = brow0 % kthw;
      let kt = i32((kr / khw) * dims.dilation_t) - i32(dims.padding_t);
      let ks = kr % khw;
      let ky = i32((ks / dims.kernel_w) * dims.dilation_h) - i32(dims.padding_h);
      let kx = i32((ks % dims.kernel_w) * dims.dilation_w) - i32(dims.padding_w);
      // The 4 columns of the quad are contiguous ox in the same frame and output row (the v4
      // condition), so x can be read contiguously too
      let n0 = bc4 * 4u;
      let pixel0 = n0 % hw_out;
      let it = i32((n0 / hw_out) * dims.stride_t) + kt;
      let iy = i32((pixel0 / dims.width_out) * dims.stride_h) + ky;
      let ix0 = i32((pixel0 % dims.width_out) * dims.stride_w) + kx;
      if (it >= 0 && u32(it) < dims.time_in && iy >= 0 && u32(iy) < dims.height_in &&
          ix0 >= 0 && u32(ix0) + 3u < dims.width_in) {
        let base = ((ic * dims.time_in + u32(it)) * dims.height_in + u32(iy)) * dims.width_in +
          u32(ix0);
        bv4_0 = vec4<f32>(x[base], x[base + 1u], x[base + 2u], x[base + 3u]);
      } else {
        // Only time edges, image edges and the padding area come here (0 out of range — the MUST of
        // xcol)
        bv4_0 = vec4<f32>(
          xcol(ic, kt, ky, kx, n0),
          xcol(ic, kt, ky, kx, n0 + 1u),
          xcol(ic, kt, ky, kx, n0 + 2u),
          xcol(ic, kt, ky, kx, n0 + 3u),
        );
      }
    }
    sb[bk0 * 32u + bcq] = bv4_0;
    let brow1 = t * 16u + bk1;
    var bv4_1 = vec4<f32>(0.0);
    if (brow1 < dims.k && bc4 < n4) {
      let ic = brow1 / kthw;
      let kr = brow1 % kthw;
      let kt = i32((kr / khw) * dims.dilation_t) - i32(dims.padding_t);
      let ks = kr % khw;
      let ky = i32((ks / dims.kernel_w) * dims.dilation_h) - i32(dims.padding_h);
      let kx = i32((ks % dims.kernel_w) * dims.dilation_w) - i32(dims.padding_w);
      // The 4 columns of the quad are contiguous ox in the same frame and output row (the v4
      // condition), so x can be read contiguously too
      let n0 = bc4 * 4u;
      let pixel0 = n0 % hw_out;
      let it = i32((n0 / hw_out) * dims.stride_t) + kt;
      let iy = i32((pixel0 / dims.width_out) * dims.stride_h) + ky;
      let ix0 = i32((pixel0 % dims.width_out) * dims.stride_w) + kx;
      if (it >= 0 && u32(it) < dims.time_in && iy >= 0 && u32(iy) < dims.height_in &&
          ix0 >= 0 && u32(ix0) + 3u < dims.width_in) {
        let base = ((ic * dims.time_in + u32(it)) * dims.height_in + u32(iy)) * dims.width_in +
          u32(ix0);
        bv4_1 = vec4<f32>(x[base], x[base + 1u], x[base + 2u], x[base + 3u]);
      } else {
        // Only time edges, image edges and the padding area come here (0 out of range — the MUST of
        // xcol)
        bv4_1 = vec4<f32>(
          xcol(ic, kt, ky, kx, n0),
          xcol(ic, kt, ky, kx, n0 + 1u),
          xcol(ic, kt, ky, kx, n0 + 2u),
          xcol(ic, kt, ky, kx, n0 + 3u),
        );
      }
    }
    sb[bk1 * 32u + bcq] = bv4_1;
    let brow2 = t * 16u + bk2;
    var bv4_2 = vec4<f32>(0.0);
    if (brow2 < dims.k && bc4 < n4) {
      let ic = brow2 / kthw;
      let kr = brow2 % kthw;
      let kt = i32((kr / khw) * dims.dilation_t) - i32(dims.padding_t);
      let ks = kr % khw;
      let ky = i32((ks / dims.kernel_w) * dims.dilation_h) - i32(dims.padding_h);
      let kx = i32((ks % dims.kernel_w) * dims.dilation_w) - i32(dims.padding_w);
      // The 4 columns of the quad are contiguous ox in the same frame and output row (the v4
      // condition), so x can be read contiguously too
      let n0 = bc4 * 4u;
      let pixel0 = n0 % hw_out;
      let it = i32((n0 / hw_out) * dims.stride_t) + kt;
      let iy = i32((pixel0 / dims.width_out) * dims.stride_h) + ky;
      let ix0 = i32((pixel0 % dims.width_out) * dims.stride_w) + kx;
      if (it >= 0 && u32(it) < dims.time_in && iy >= 0 && u32(iy) < dims.height_in &&
          ix0 >= 0 && u32(ix0) + 3u < dims.width_in) {
        let base = ((ic * dims.time_in + u32(it)) * dims.height_in + u32(iy)) * dims.width_in +
          u32(ix0);
        bv4_2 = vec4<f32>(x[base], x[base + 1u], x[base + 2u], x[base + 3u]);
      } else {
        // Only time edges, image edges and the padding area come here (0 out of range — the MUST of
        // xcol)
        bv4_2 = vec4<f32>(
          xcol(ic, kt, ky, kx, n0),
          xcol(ic, kt, ky, kx, n0 + 1u),
          xcol(ic, kt, ky, kx, n0 + 2u),
          xcol(ic, kt, ky, kx, n0 + 3u),
        );
      }
    }
    sb[bk2 * 32u + bcq] = bv4_2;
    let brow3 = t * 16u + bk3;
    var bv4_3 = vec4<f32>(0.0);
    if (brow3 < dims.k && bc4 < n4) {
      let ic = brow3 / kthw;
      let kr = brow3 % kthw;
      let kt = i32((kr / khw) * dims.dilation_t) - i32(dims.padding_t);
      let ks = kr % khw;
      let ky = i32((ks / dims.kernel_w) * dims.dilation_h) - i32(dims.padding_h);
      let kx = i32((ks % dims.kernel_w) * dims.dilation_w) - i32(dims.padding_w);
      // The 4 columns of the quad are contiguous ox in the same frame and output row (the v4
      // condition), so x can be read contiguously too
      let n0 = bc4 * 4u;
      let pixel0 = n0 % hw_out;
      let it = i32((n0 / hw_out) * dims.stride_t) + kt;
      let iy = i32((pixel0 / dims.width_out) * dims.stride_h) + ky;
      let ix0 = i32((pixel0 % dims.width_out) * dims.stride_w) + kx;
      if (it >= 0 && u32(it) < dims.time_in && iy >= 0 && u32(iy) < dims.height_in &&
          ix0 >= 0 && u32(ix0) + 3u < dims.width_in) {
        let base = ((ic * dims.time_in + u32(it)) * dims.height_in + u32(iy)) * dims.width_in +
          u32(ix0);
        bv4_3 = vec4<f32>(x[base], x[base + 1u], x[base + 2u], x[base + 3u]);
      } else {
        // Only time edges, image edges and the padding area come here (0 out of range — the MUST of
        // xcol)
        bv4_3 = vec4<f32>(
          xcol(ic, kt, ky, kx, n0),
          xcol(ic, kt, ky, kx, n0 + 1u),
          xcol(ic, kt, ky, kx, n0 + 2u),
          xcol(ic, kt, ky, kx, n0 + 3u),
        );
      }
    }
    sb[bk3 * 32u + bcq] = bv4_3;
    workgroupBarrier();
    // 10 shared loads (2 vec4 of B + 8 scalars of A) give 64 MACs.
    // The reduction is sequential in ascending k, and the addition order per output element matches
    // the 16×16 kernel with 1 output per thread exactly.
    for (var kk = 0u; kk < 16u; kk = kk + 1u) {
      let bv0 = sb[kk * 32u + lid.x * 2u];
      let bv1 = sb[kk * 32u + lid.x * 2u + 1u];
      acc0_0 = acc0_0 + sa[(lid.y * 8u + 0u) * 16u + kk] * bv0;
      acc0_1 = acc0_1 + sa[(lid.y * 8u + 0u) * 16u + kk] * bv1;
      acc1_0 = acc1_0 + sa[(lid.y * 8u + 1u) * 16u + kk] * bv0;
      acc1_1 = acc1_1 + sa[(lid.y * 8u + 1u) * 16u + kk] * bv1;
      acc2_0 = acc2_0 + sa[(lid.y * 8u + 2u) * 16u + kk] * bv0;
      acc2_1 = acc2_1 + sa[(lid.y * 8u + 2u) * 16u + kk] * bv1;
      acc3_0 = acc3_0 + sa[(lid.y * 8u + 3u) * 16u + kk] * bv0;
      acc3_1 = acc3_1 + sa[(lid.y * 8u + 3u) * 16u + kk] * bv1;
      acc4_0 = acc4_0 + sa[(lid.y * 8u + 4u) * 16u + kk] * bv0;
      acc4_1 = acc4_1 + sa[(lid.y * 8u + 4u) * 16u + kk] * bv1;
      acc5_0 = acc5_0 + sa[(lid.y * 8u + 5u) * 16u + kk] * bv0;
      acc5_1 = acc5_1 + sa[(lid.y * 8u + 5u) * 16u + kk] * bv1;
      acc6_0 = acc6_0 + sa[(lid.y * 8u + 6u) * 16u + kk] * bv0;
      acc6_1 = acc6_1 + sa[(lid.y * 8u + 6u) * 16u + kk] * bv1;
      acc7_0 = acc7_0 + sa[(lid.y * 8u + 7u) * 16u + kk] * bv0;
      acc7_1 = acc7_1 + sa[(lid.y * 8u + 7u) * 16u + kk] * bv1;
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
    if (orow0 < dims.m) {
      out[orow0 * n4 + ocq0] = acc0_0;
    }
    if (orow1 < dims.m) {
      out[orow1 * n4 + ocq0] = acc1_0;
    }
    if (orow2 < dims.m) {
      out[orow2 * n4 + ocq0] = acc2_0;
    }
    if (orow3 < dims.m) {
      out[orow3 * n4 + ocq0] = acc3_0;
    }
    if (orow4 < dims.m) {
      out[orow4 * n4 + ocq0] = acc4_0;
    }
    if (orow5 < dims.m) {
      out[orow5 * n4 + ocq0] = acc5_0;
    }
    if (orow6 < dims.m) {
      out[orow6 * n4 + ocq0] = acc6_0;
    }
    if (orow7 < dims.m) {
      out[orow7 * n4 + ocq0] = acc7_0;
    }
  }
  if (ocq1 < n4) {
    if (orow0 < dims.m) {
      out[orow0 * n4 + ocq1] = acc0_1;
    }
    if (orow1 < dims.m) {
      out[orow1 * n4 + ocq1] = acc1_1;
    }
    if (orow2 < dims.m) {
      out[orow2 * n4 + ocq1] = acc2_1;
    }
    if (orow3 < dims.m) {
      out[orow3 * n4 + ocq1] = acc3_1;
    }
    if (orow4 < dims.m) {
      out[orow4 * n4 + ocq1] = acc4_1;
    }
    if (orow5 < dims.m) {
      out[orow5 * n4 + ocq1] = acc5_1;
    }
    if (orow6 < dims.m) {
      out[orow6 * n4 + ocq1] = acc6_1;
    }
    if (orow7 < dims.m) {
      out[orow7 * n4 + ocq1] = acc7_1;
    }
  }
}
