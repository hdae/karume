// karume conv2d (x[B,Cin,H,W] * W[Cout,Cin,Kh,Kw] + b[Cout], f32, implicit GEMM register tile 64x128 / 8x8 per thread / wg 16x8 + vec4)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
  channels_in: u32,
  height_in: u32,
  width_in: u32,
  width_out: u32,
  kernel_h: u32,
  kernel_w: u32,
  stride_h: u32,
  stride_w: u32,
  padding_h: u32,
  padding_w: u32,
  dilation_h: u32,
  dilation_w: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<vec4<f32>>;

// One element of the input row whose first element is x[rb] (0 out of range)
fn xcol_row(rb: u32, ix: i32) -> f32 {
  if (ix < 0 || u32(ix) >= dims.width_in) {
    return 0.0;
  }
  return x[rb + u32(ix)];
}
// The quad shares the output row (the v4 condition): one row check, then one check per column.
// xc = b·Cin + ic (the batch is the dispatch's z axis, so the caller adds it)
fn xcol4(xc: u32, iy: i32, ix0: i32) -> vec4<f32> {
  if (iy < 0 || u32(iy) >= dims.height_in) {
    return vec4<f32>(0.0);
  }
  let rb = (xc * dims.height_in + u32(iy)) * dims.width_in;
  let e0 = xcol_row(rb, ix0);
  let e1 = xcol_row(rb, ix0 + 1);
  let e2 = xcol_row(rb, ix0 + 2);
  let e3 = xcol_row(rb, ix0 + 3);
  return vec4<f32>(e0, e1, e2, e3);
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
  let sa_base0 = aq * 256u + ar;
  let arow1 = arow0 + 32u;
  let arow_base1 = arow_base0 + 32u * dims.k;
  let sa_base1 = sa_base0 + 32u;
  // The batch is uniform per workgroup
  // (one z index = the output plane [Cout, Hout·Wout] of one batch)
  let xbase = wid.z * dims.channels_in;
  let cbase = wid.z * dims.m * n4;
  // B tile assignment (K 16 rows × 32 column quads over 128 threads in 4 rounds)
  let bk0 = tid / 32u;
  let bcq = tid % 32u;
  let bc4 = wid.x * 32u + bcq;
  let bk1 = bk0 + 4u;
  let bk2 = bk0 + 8u;
  let bk3 = bk0 + 12u;
  // Invariant in the K tile loop (the step for splitting flat k into (ic, kh, kw))
  let khw = dims.kernel_h * dims.kernel_w;
  // The quad's column origin, invariant in the K tile loop
  let n0q = bc4 * 4u;
  let oy_q = i32((n0q / dims.width_out) * dims.stride_h);
  let ox_q = i32((n0q % dims.width_out) * dims.stride_w);
  // One K tile advances flat k by 16: the step as mixed-radix digits (ic, kh, kw)
  let step_ic = 16u / khw;
  let step_r = 16u % khw;
  let step_kh = step_r / dims.kernel_w;
  let step_kw = step_r % dims.kernel_w;
  // Each slot's (ic, kh, kw) at K tile 0 — the divisions the original form does every tile
  var c_ic0 = bk0 / khw;
  var c_kh0 = (bk0 % khw) / dims.kernel_w;
  var c_kw0 = (bk0 % khw) % dims.kernel_w;
  var c_ic1 = bk1 / khw;
  var c_kh1 = (bk1 % khw) / dims.kernel_w;
  var c_kw1 = (bk1 % khw) % dims.kernel_w;
  var c_ic2 = bk2 / khw;
  var c_kh2 = (bk2 % khw) / dims.kernel_w;
  var c_kw2 = (bk2 % khw) % dims.kernel_w;
  var c_ic3 = bk3 / khw;
  var c_kh3 = (bk3 % khw) / dims.kernel_w;
  var c_kw3 = (bk3 % khw) % dims.kernel_w;
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
    sa[sa_base0 + 64u] = av0.y;
    sa[sa_base0 + 128u] = av0.z;
    sa[sa_base0 + 192u] = av0.w;
    var av1 = vec4<f32>(0.0);
    if (arow1 < dims.m && ak0 < dims.k) {
      av1 = w[(arow_base1 + ak0) >> 2u];
    }
    sa[sa_base1] = av1.x;
    sa[sa_base1 + 64u] = av1.y;
    sa[sa_base1 + 128u] = av1.z;
    sa[sa_base1 + 192u] = av1.w;
    let brow0 = t * 16u + bk0;
    var bv4_0 = vec4<f32>(0.0);
    if (brow0 < dims.k && bc4 < n4) {
      let ic = c_ic0;
      let ky = i32(c_kh0 * dims.dilation_h) - i32(dims.padding_h);
      let kx = i32(c_kw0 * dims.dilation_w) - i32(dims.padding_w);
      let xc = xbase + ic;
      // The 4 columns of the quad are contiguous ox in the same output row (the v4 condition), so x
      // can be read contiguously too
      let iy = oy_q + ky;
      let ix0 = ox_q + kx;
      if (iy >= 0 && u32(iy) < dims.height_in && ix0 >= 0 && u32(ix0) + 3u < dims.width_in) {
        let base = (xc * dims.height_in + u32(iy)) * dims.width_in + u32(ix0);
        bv4_0 = vec4<f32>(x[base], x[base + 1u], x[base + 2u], x[base + 3u]);
      } else {
        // Only image edges and the padding area come here (0 out of range — the MUST of xcol)
        bv4_0 = xcol4(xc, iy, ix0);
      }
    }
    sb[bk0 * 32u + bcq] = bv4_0;
    c_kw0 = c_kw0 + step_kw;
    if (c_kw0 >= dims.kernel_w) { c_kw0 = c_kw0 - dims.kernel_w; c_kh0 = c_kh0 + 1u; }
    c_kh0 = c_kh0 + step_kh;
    if (c_kh0 >= dims.kernel_h) { c_kh0 = c_kh0 - dims.kernel_h; c_ic0 = c_ic0 + 1u; }
    c_ic0 = c_ic0 + step_ic;
    let brow1 = t * 16u + bk1;
    var bv4_1 = vec4<f32>(0.0);
    if (brow1 < dims.k && bc4 < n4) {
      let ic = c_ic1;
      let ky = i32(c_kh1 * dims.dilation_h) - i32(dims.padding_h);
      let kx = i32(c_kw1 * dims.dilation_w) - i32(dims.padding_w);
      let xc = xbase + ic;
      // The 4 columns of the quad are contiguous ox in the same output row (the v4 condition), so x
      // can be read contiguously too
      let iy = oy_q + ky;
      let ix0 = ox_q + kx;
      if (iy >= 0 && u32(iy) < dims.height_in && ix0 >= 0 && u32(ix0) + 3u < dims.width_in) {
        let base = (xc * dims.height_in + u32(iy)) * dims.width_in + u32(ix0);
        bv4_1 = vec4<f32>(x[base], x[base + 1u], x[base + 2u], x[base + 3u]);
      } else {
        // Only image edges and the padding area come here (0 out of range — the MUST of xcol)
        bv4_1 = xcol4(xc, iy, ix0);
      }
    }
    sb[bk1 * 32u + bcq] = bv4_1;
    c_kw1 = c_kw1 + step_kw;
    if (c_kw1 >= dims.kernel_w) { c_kw1 = c_kw1 - dims.kernel_w; c_kh1 = c_kh1 + 1u; }
    c_kh1 = c_kh1 + step_kh;
    if (c_kh1 >= dims.kernel_h) { c_kh1 = c_kh1 - dims.kernel_h; c_ic1 = c_ic1 + 1u; }
    c_ic1 = c_ic1 + step_ic;
    let brow2 = t * 16u + bk2;
    var bv4_2 = vec4<f32>(0.0);
    if (brow2 < dims.k && bc4 < n4) {
      let ic = c_ic2;
      let ky = i32(c_kh2 * dims.dilation_h) - i32(dims.padding_h);
      let kx = i32(c_kw2 * dims.dilation_w) - i32(dims.padding_w);
      let xc = xbase + ic;
      // The 4 columns of the quad are contiguous ox in the same output row (the v4 condition), so x
      // can be read contiguously too
      let iy = oy_q + ky;
      let ix0 = ox_q + kx;
      if (iy >= 0 && u32(iy) < dims.height_in && ix0 >= 0 && u32(ix0) + 3u < dims.width_in) {
        let base = (xc * dims.height_in + u32(iy)) * dims.width_in + u32(ix0);
        bv4_2 = vec4<f32>(x[base], x[base + 1u], x[base + 2u], x[base + 3u]);
      } else {
        // Only image edges and the padding area come here (0 out of range — the MUST of xcol)
        bv4_2 = xcol4(xc, iy, ix0);
      }
    }
    sb[bk2 * 32u + bcq] = bv4_2;
    c_kw2 = c_kw2 + step_kw;
    if (c_kw2 >= dims.kernel_w) { c_kw2 = c_kw2 - dims.kernel_w; c_kh2 = c_kh2 + 1u; }
    c_kh2 = c_kh2 + step_kh;
    if (c_kh2 >= dims.kernel_h) { c_kh2 = c_kh2 - dims.kernel_h; c_ic2 = c_ic2 + 1u; }
    c_ic2 = c_ic2 + step_ic;
    let brow3 = t * 16u + bk3;
    var bv4_3 = vec4<f32>(0.0);
    if (brow3 < dims.k && bc4 < n4) {
      let ic = c_ic3;
      let ky = i32(c_kh3 * dims.dilation_h) - i32(dims.padding_h);
      let kx = i32(c_kw3 * dims.dilation_w) - i32(dims.padding_w);
      let xc = xbase + ic;
      // The 4 columns of the quad are contiguous ox in the same output row (the v4 condition), so x
      // can be read contiguously too
      let iy = oy_q + ky;
      let ix0 = ox_q + kx;
      if (iy >= 0 && u32(iy) < dims.height_in && ix0 >= 0 && u32(ix0) + 3u < dims.width_in) {
        let base = (xc * dims.height_in + u32(iy)) * dims.width_in + u32(ix0);
        bv4_3 = vec4<f32>(x[base], x[base + 1u], x[base + 2u], x[base + 3u]);
      } else {
        // Only image edges and the padding area come here (0 out of range — the MUST of xcol)
        bv4_3 = xcol4(xc, iy, ix0);
      }
    }
    sb[bk3 * 32u + bcq] = bv4_3;
    c_kw3 = c_kw3 + step_kw;
    if (c_kw3 >= dims.kernel_w) { c_kw3 = c_kw3 - dims.kernel_w; c_kh3 = c_kh3 + 1u; }
    c_kh3 = c_kh3 + step_kh;
    if (c_kh3 >= dims.kernel_h) { c_kh3 = c_kh3 - dims.kernel_h; c_ic3 = c_ic3 + 1u; }
    c_ic3 = c_ic3 + step_ic;
    workgroupBarrier();
    // 10 shared loads (2 vec4 of B + 8 scalars of A) give 64 MACs.
    // The reduction is sequential in ascending k, and the addition order per output element matches
    // the 16×16 kernel with 1 output per thread exactly.
    {
      let kk = 0u;
      let bv0 = sb[kk * 32u + lid.x];
      let bv1 = sb[kk * 32u + lid.x + 16u];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 8u + 0u] * bv0;
      acc0_1 = acc0_1 + sa[kk * 64u + lid.y * 8u + 0u] * bv1;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 8u + 1u] * bv0;
      acc1_1 = acc1_1 + sa[kk * 64u + lid.y * 8u + 1u] * bv1;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 8u + 2u] * bv0;
      acc2_1 = acc2_1 + sa[kk * 64u + lid.y * 8u + 2u] * bv1;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 8u + 3u] * bv0;
      acc3_1 = acc3_1 + sa[kk * 64u + lid.y * 8u + 3u] * bv1;
      acc4_0 = acc4_0 + sa[kk * 64u + lid.y * 8u + 4u] * bv0;
      acc4_1 = acc4_1 + sa[kk * 64u + lid.y * 8u + 4u] * bv1;
      acc5_0 = acc5_0 + sa[kk * 64u + lid.y * 8u + 5u] * bv0;
      acc5_1 = acc5_1 + sa[kk * 64u + lid.y * 8u + 5u] * bv1;
      acc6_0 = acc6_0 + sa[kk * 64u + lid.y * 8u + 6u] * bv0;
      acc6_1 = acc6_1 + sa[kk * 64u + lid.y * 8u + 6u] * bv1;
      acc7_0 = acc7_0 + sa[kk * 64u + lid.y * 8u + 7u] * bv0;
      acc7_1 = acc7_1 + sa[kk * 64u + lid.y * 8u + 7u] * bv1;
    }
    {
      let kk = 1u;
      let bv0 = sb[kk * 32u + lid.x];
      let bv1 = sb[kk * 32u + lid.x + 16u];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 8u + 0u] * bv0;
      acc0_1 = acc0_1 + sa[kk * 64u + lid.y * 8u + 0u] * bv1;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 8u + 1u] * bv0;
      acc1_1 = acc1_1 + sa[kk * 64u + lid.y * 8u + 1u] * bv1;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 8u + 2u] * bv0;
      acc2_1 = acc2_1 + sa[kk * 64u + lid.y * 8u + 2u] * bv1;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 8u + 3u] * bv0;
      acc3_1 = acc3_1 + sa[kk * 64u + lid.y * 8u + 3u] * bv1;
      acc4_0 = acc4_0 + sa[kk * 64u + lid.y * 8u + 4u] * bv0;
      acc4_1 = acc4_1 + sa[kk * 64u + lid.y * 8u + 4u] * bv1;
      acc5_0 = acc5_0 + sa[kk * 64u + lid.y * 8u + 5u] * bv0;
      acc5_1 = acc5_1 + sa[kk * 64u + lid.y * 8u + 5u] * bv1;
      acc6_0 = acc6_0 + sa[kk * 64u + lid.y * 8u + 6u] * bv0;
      acc6_1 = acc6_1 + sa[kk * 64u + lid.y * 8u + 6u] * bv1;
      acc7_0 = acc7_0 + sa[kk * 64u + lid.y * 8u + 7u] * bv0;
      acc7_1 = acc7_1 + sa[kk * 64u + lid.y * 8u + 7u] * bv1;
    }
    {
      let kk = 2u;
      let bv0 = sb[kk * 32u + lid.x];
      let bv1 = sb[kk * 32u + lid.x + 16u];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 8u + 0u] * bv0;
      acc0_1 = acc0_1 + sa[kk * 64u + lid.y * 8u + 0u] * bv1;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 8u + 1u] * bv0;
      acc1_1 = acc1_1 + sa[kk * 64u + lid.y * 8u + 1u] * bv1;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 8u + 2u] * bv0;
      acc2_1 = acc2_1 + sa[kk * 64u + lid.y * 8u + 2u] * bv1;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 8u + 3u] * bv0;
      acc3_1 = acc3_1 + sa[kk * 64u + lid.y * 8u + 3u] * bv1;
      acc4_0 = acc4_0 + sa[kk * 64u + lid.y * 8u + 4u] * bv0;
      acc4_1 = acc4_1 + sa[kk * 64u + lid.y * 8u + 4u] * bv1;
      acc5_0 = acc5_0 + sa[kk * 64u + lid.y * 8u + 5u] * bv0;
      acc5_1 = acc5_1 + sa[kk * 64u + lid.y * 8u + 5u] * bv1;
      acc6_0 = acc6_0 + sa[kk * 64u + lid.y * 8u + 6u] * bv0;
      acc6_1 = acc6_1 + sa[kk * 64u + lid.y * 8u + 6u] * bv1;
      acc7_0 = acc7_0 + sa[kk * 64u + lid.y * 8u + 7u] * bv0;
      acc7_1 = acc7_1 + sa[kk * 64u + lid.y * 8u + 7u] * bv1;
    }
    {
      let kk = 3u;
      let bv0 = sb[kk * 32u + lid.x];
      let bv1 = sb[kk * 32u + lid.x + 16u];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 8u + 0u] * bv0;
      acc0_1 = acc0_1 + sa[kk * 64u + lid.y * 8u + 0u] * bv1;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 8u + 1u] * bv0;
      acc1_1 = acc1_1 + sa[kk * 64u + lid.y * 8u + 1u] * bv1;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 8u + 2u] * bv0;
      acc2_1 = acc2_1 + sa[kk * 64u + lid.y * 8u + 2u] * bv1;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 8u + 3u] * bv0;
      acc3_1 = acc3_1 + sa[kk * 64u + lid.y * 8u + 3u] * bv1;
      acc4_0 = acc4_0 + sa[kk * 64u + lid.y * 8u + 4u] * bv0;
      acc4_1 = acc4_1 + sa[kk * 64u + lid.y * 8u + 4u] * bv1;
      acc5_0 = acc5_0 + sa[kk * 64u + lid.y * 8u + 5u] * bv0;
      acc5_1 = acc5_1 + sa[kk * 64u + lid.y * 8u + 5u] * bv1;
      acc6_0 = acc6_0 + sa[kk * 64u + lid.y * 8u + 6u] * bv0;
      acc6_1 = acc6_1 + sa[kk * 64u + lid.y * 8u + 6u] * bv1;
      acc7_0 = acc7_0 + sa[kk * 64u + lid.y * 8u + 7u] * bv0;
      acc7_1 = acc7_1 + sa[kk * 64u + lid.y * 8u + 7u] * bv1;
    }
    {
      let kk = 4u;
      let bv0 = sb[kk * 32u + lid.x];
      let bv1 = sb[kk * 32u + lid.x + 16u];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 8u + 0u] * bv0;
      acc0_1 = acc0_1 + sa[kk * 64u + lid.y * 8u + 0u] * bv1;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 8u + 1u] * bv0;
      acc1_1 = acc1_1 + sa[kk * 64u + lid.y * 8u + 1u] * bv1;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 8u + 2u] * bv0;
      acc2_1 = acc2_1 + sa[kk * 64u + lid.y * 8u + 2u] * bv1;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 8u + 3u] * bv0;
      acc3_1 = acc3_1 + sa[kk * 64u + lid.y * 8u + 3u] * bv1;
      acc4_0 = acc4_0 + sa[kk * 64u + lid.y * 8u + 4u] * bv0;
      acc4_1 = acc4_1 + sa[kk * 64u + lid.y * 8u + 4u] * bv1;
      acc5_0 = acc5_0 + sa[kk * 64u + lid.y * 8u + 5u] * bv0;
      acc5_1 = acc5_1 + sa[kk * 64u + lid.y * 8u + 5u] * bv1;
      acc6_0 = acc6_0 + sa[kk * 64u + lid.y * 8u + 6u] * bv0;
      acc6_1 = acc6_1 + sa[kk * 64u + lid.y * 8u + 6u] * bv1;
      acc7_0 = acc7_0 + sa[kk * 64u + lid.y * 8u + 7u] * bv0;
      acc7_1 = acc7_1 + sa[kk * 64u + lid.y * 8u + 7u] * bv1;
    }
    {
      let kk = 5u;
      let bv0 = sb[kk * 32u + lid.x];
      let bv1 = sb[kk * 32u + lid.x + 16u];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 8u + 0u] * bv0;
      acc0_1 = acc0_1 + sa[kk * 64u + lid.y * 8u + 0u] * bv1;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 8u + 1u] * bv0;
      acc1_1 = acc1_1 + sa[kk * 64u + lid.y * 8u + 1u] * bv1;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 8u + 2u] * bv0;
      acc2_1 = acc2_1 + sa[kk * 64u + lid.y * 8u + 2u] * bv1;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 8u + 3u] * bv0;
      acc3_1 = acc3_1 + sa[kk * 64u + lid.y * 8u + 3u] * bv1;
      acc4_0 = acc4_0 + sa[kk * 64u + lid.y * 8u + 4u] * bv0;
      acc4_1 = acc4_1 + sa[kk * 64u + lid.y * 8u + 4u] * bv1;
      acc5_0 = acc5_0 + sa[kk * 64u + lid.y * 8u + 5u] * bv0;
      acc5_1 = acc5_1 + sa[kk * 64u + lid.y * 8u + 5u] * bv1;
      acc6_0 = acc6_0 + sa[kk * 64u + lid.y * 8u + 6u] * bv0;
      acc6_1 = acc6_1 + sa[kk * 64u + lid.y * 8u + 6u] * bv1;
      acc7_0 = acc7_0 + sa[kk * 64u + lid.y * 8u + 7u] * bv0;
      acc7_1 = acc7_1 + sa[kk * 64u + lid.y * 8u + 7u] * bv1;
    }
    {
      let kk = 6u;
      let bv0 = sb[kk * 32u + lid.x];
      let bv1 = sb[kk * 32u + lid.x + 16u];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 8u + 0u] * bv0;
      acc0_1 = acc0_1 + sa[kk * 64u + lid.y * 8u + 0u] * bv1;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 8u + 1u] * bv0;
      acc1_1 = acc1_1 + sa[kk * 64u + lid.y * 8u + 1u] * bv1;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 8u + 2u] * bv0;
      acc2_1 = acc2_1 + sa[kk * 64u + lid.y * 8u + 2u] * bv1;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 8u + 3u] * bv0;
      acc3_1 = acc3_1 + sa[kk * 64u + lid.y * 8u + 3u] * bv1;
      acc4_0 = acc4_0 + sa[kk * 64u + lid.y * 8u + 4u] * bv0;
      acc4_1 = acc4_1 + sa[kk * 64u + lid.y * 8u + 4u] * bv1;
      acc5_0 = acc5_0 + sa[kk * 64u + lid.y * 8u + 5u] * bv0;
      acc5_1 = acc5_1 + sa[kk * 64u + lid.y * 8u + 5u] * bv1;
      acc6_0 = acc6_0 + sa[kk * 64u + lid.y * 8u + 6u] * bv0;
      acc6_1 = acc6_1 + sa[kk * 64u + lid.y * 8u + 6u] * bv1;
      acc7_0 = acc7_0 + sa[kk * 64u + lid.y * 8u + 7u] * bv0;
      acc7_1 = acc7_1 + sa[kk * 64u + lid.y * 8u + 7u] * bv1;
    }
    {
      let kk = 7u;
      let bv0 = sb[kk * 32u + lid.x];
      let bv1 = sb[kk * 32u + lid.x + 16u];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 8u + 0u] * bv0;
      acc0_1 = acc0_1 + sa[kk * 64u + lid.y * 8u + 0u] * bv1;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 8u + 1u] * bv0;
      acc1_1 = acc1_1 + sa[kk * 64u + lid.y * 8u + 1u] * bv1;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 8u + 2u] * bv0;
      acc2_1 = acc2_1 + sa[kk * 64u + lid.y * 8u + 2u] * bv1;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 8u + 3u] * bv0;
      acc3_1 = acc3_1 + sa[kk * 64u + lid.y * 8u + 3u] * bv1;
      acc4_0 = acc4_0 + sa[kk * 64u + lid.y * 8u + 4u] * bv0;
      acc4_1 = acc4_1 + sa[kk * 64u + lid.y * 8u + 4u] * bv1;
      acc5_0 = acc5_0 + sa[kk * 64u + lid.y * 8u + 5u] * bv0;
      acc5_1 = acc5_1 + sa[kk * 64u + lid.y * 8u + 5u] * bv1;
      acc6_0 = acc6_0 + sa[kk * 64u + lid.y * 8u + 6u] * bv0;
      acc6_1 = acc6_1 + sa[kk * 64u + lid.y * 8u + 6u] * bv1;
      acc7_0 = acc7_0 + sa[kk * 64u + lid.y * 8u + 7u] * bv0;
      acc7_1 = acc7_1 + sa[kk * 64u + lid.y * 8u + 7u] * bv1;
    }
    {
      let kk = 8u;
      let bv0 = sb[kk * 32u + lid.x];
      let bv1 = sb[kk * 32u + lid.x + 16u];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 8u + 0u] * bv0;
      acc0_1 = acc0_1 + sa[kk * 64u + lid.y * 8u + 0u] * bv1;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 8u + 1u] * bv0;
      acc1_1 = acc1_1 + sa[kk * 64u + lid.y * 8u + 1u] * bv1;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 8u + 2u] * bv0;
      acc2_1 = acc2_1 + sa[kk * 64u + lid.y * 8u + 2u] * bv1;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 8u + 3u] * bv0;
      acc3_1 = acc3_1 + sa[kk * 64u + lid.y * 8u + 3u] * bv1;
      acc4_0 = acc4_0 + sa[kk * 64u + lid.y * 8u + 4u] * bv0;
      acc4_1 = acc4_1 + sa[kk * 64u + lid.y * 8u + 4u] * bv1;
      acc5_0 = acc5_0 + sa[kk * 64u + lid.y * 8u + 5u] * bv0;
      acc5_1 = acc5_1 + sa[kk * 64u + lid.y * 8u + 5u] * bv1;
      acc6_0 = acc6_0 + sa[kk * 64u + lid.y * 8u + 6u] * bv0;
      acc6_1 = acc6_1 + sa[kk * 64u + lid.y * 8u + 6u] * bv1;
      acc7_0 = acc7_0 + sa[kk * 64u + lid.y * 8u + 7u] * bv0;
      acc7_1 = acc7_1 + sa[kk * 64u + lid.y * 8u + 7u] * bv1;
    }
    {
      let kk = 9u;
      let bv0 = sb[kk * 32u + lid.x];
      let bv1 = sb[kk * 32u + lid.x + 16u];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 8u + 0u] * bv0;
      acc0_1 = acc0_1 + sa[kk * 64u + lid.y * 8u + 0u] * bv1;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 8u + 1u] * bv0;
      acc1_1 = acc1_1 + sa[kk * 64u + lid.y * 8u + 1u] * bv1;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 8u + 2u] * bv0;
      acc2_1 = acc2_1 + sa[kk * 64u + lid.y * 8u + 2u] * bv1;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 8u + 3u] * bv0;
      acc3_1 = acc3_1 + sa[kk * 64u + lid.y * 8u + 3u] * bv1;
      acc4_0 = acc4_0 + sa[kk * 64u + lid.y * 8u + 4u] * bv0;
      acc4_1 = acc4_1 + sa[kk * 64u + lid.y * 8u + 4u] * bv1;
      acc5_0 = acc5_0 + sa[kk * 64u + lid.y * 8u + 5u] * bv0;
      acc5_1 = acc5_1 + sa[kk * 64u + lid.y * 8u + 5u] * bv1;
      acc6_0 = acc6_0 + sa[kk * 64u + lid.y * 8u + 6u] * bv0;
      acc6_1 = acc6_1 + sa[kk * 64u + lid.y * 8u + 6u] * bv1;
      acc7_0 = acc7_0 + sa[kk * 64u + lid.y * 8u + 7u] * bv0;
      acc7_1 = acc7_1 + sa[kk * 64u + lid.y * 8u + 7u] * bv1;
    }
    {
      let kk = 10u;
      let bv0 = sb[kk * 32u + lid.x];
      let bv1 = sb[kk * 32u + lid.x + 16u];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 8u + 0u] * bv0;
      acc0_1 = acc0_1 + sa[kk * 64u + lid.y * 8u + 0u] * bv1;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 8u + 1u] * bv0;
      acc1_1 = acc1_1 + sa[kk * 64u + lid.y * 8u + 1u] * bv1;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 8u + 2u] * bv0;
      acc2_1 = acc2_1 + sa[kk * 64u + lid.y * 8u + 2u] * bv1;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 8u + 3u] * bv0;
      acc3_1 = acc3_1 + sa[kk * 64u + lid.y * 8u + 3u] * bv1;
      acc4_0 = acc4_0 + sa[kk * 64u + lid.y * 8u + 4u] * bv0;
      acc4_1 = acc4_1 + sa[kk * 64u + lid.y * 8u + 4u] * bv1;
      acc5_0 = acc5_0 + sa[kk * 64u + lid.y * 8u + 5u] * bv0;
      acc5_1 = acc5_1 + sa[kk * 64u + lid.y * 8u + 5u] * bv1;
      acc6_0 = acc6_0 + sa[kk * 64u + lid.y * 8u + 6u] * bv0;
      acc6_1 = acc6_1 + sa[kk * 64u + lid.y * 8u + 6u] * bv1;
      acc7_0 = acc7_0 + sa[kk * 64u + lid.y * 8u + 7u] * bv0;
      acc7_1 = acc7_1 + sa[kk * 64u + lid.y * 8u + 7u] * bv1;
    }
    {
      let kk = 11u;
      let bv0 = sb[kk * 32u + lid.x];
      let bv1 = sb[kk * 32u + lid.x + 16u];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 8u + 0u] * bv0;
      acc0_1 = acc0_1 + sa[kk * 64u + lid.y * 8u + 0u] * bv1;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 8u + 1u] * bv0;
      acc1_1 = acc1_1 + sa[kk * 64u + lid.y * 8u + 1u] * bv1;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 8u + 2u] * bv0;
      acc2_1 = acc2_1 + sa[kk * 64u + lid.y * 8u + 2u] * bv1;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 8u + 3u] * bv0;
      acc3_1 = acc3_1 + sa[kk * 64u + lid.y * 8u + 3u] * bv1;
      acc4_0 = acc4_0 + sa[kk * 64u + lid.y * 8u + 4u] * bv0;
      acc4_1 = acc4_1 + sa[kk * 64u + lid.y * 8u + 4u] * bv1;
      acc5_0 = acc5_0 + sa[kk * 64u + lid.y * 8u + 5u] * bv0;
      acc5_1 = acc5_1 + sa[kk * 64u + lid.y * 8u + 5u] * bv1;
      acc6_0 = acc6_0 + sa[kk * 64u + lid.y * 8u + 6u] * bv0;
      acc6_1 = acc6_1 + sa[kk * 64u + lid.y * 8u + 6u] * bv1;
      acc7_0 = acc7_0 + sa[kk * 64u + lid.y * 8u + 7u] * bv0;
      acc7_1 = acc7_1 + sa[kk * 64u + lid.y * 8u + 7u] * bv1;
    }
    {
      let kk = 12u;
      let bv0 = sb[kk * 32u + lid.x];
      let bv1 = sb[kk * 32u + lid.x + 16u];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 8u + 0u] * bv0;
      acc0_1 = acc0_1 + sa[kk * 64u + lid.y * 8u + 0u] * bv1;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 8u + 1u] * bv0;
      acc1_1 = acc1_1 + sa[kk * 64u + lid.y * 8u + 1u] * bv1;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 8u + 2u] * bv0;
      acc2_1 = acc2_1 + sa[kk * 64u + lid.y * 8u + 2u] * bv1;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 8u + 3u] * bv0;
      acc3_1 = acc3_1 + sa[kk * 64u + lid.y * 8u + 3u] * bv1;
      acc4_0 = acc4_0 + sa[kk * 64u + lid.y * 8u + 4u] * bv0;
      acc4_1 = acc4_1 + sa[kk * 64u + lid.y * 8u + 4u] * bv1;
      acc5_0 = acc5_0 + sa[kk * 64u + lid.y * 8u + 5u] * bv0;
      acc5_1 = acc5_1 + sa[kk * 64u + lid.y * 8u + 5u] * bv1;
      acc6_0 = acc6_0 + sa[kk * 64u + lid.y * 8u + 6u] * bv0;
      acc6_1 = acc6_1 + sa[kk * 64u + lid.y * 8u + 6u] * bv1;
      acc7_0 = acc7_0 + sa[kk * 64u + lid.y * 8u + 7u] * bv0;
      acc7_1 = acc7_1 + sa[kk * 64u + lid.y * 8u + 7u] * bv1;
    }
    {
      let kk = 13u;
      let bv0 = sb[kk * 32u + lid.x];
      let bv1 = sb[kk * 32u + lid.x + 16u];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 8u + 0u] * bv0;
      acc0_1 = acc0_1 + sa[kk * 64u + lid.y * 8u + 0u] * bv1;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 8u + 1u] * bv0;
      acc1_1 = acc1_1 + sa[kk * 64u + lid.y * 8u + 1u] * bv1;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 8u + 2u] * bv0;
      acc2_1 = acc2_1 + sa[kk * 64u + lid.y * 8u + 2u] * bv1;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 8u + 3u] * bv0;
      acc3_1 = acc3_1 + sa[kk * 64u + lid.y * 8u + 3u] * bv1;
      acc4_0 = acc4_0 + sa[kk * 64u + lid.y * 8u + 4u] * bv0;
      acc4_1 = acc4_1 + sa[kk * 64u + lid.y * 8u + 4u] * bv1;
      acc5_0 = acc5_0 + sa[kk * 64u + lid.y * 8u + 5u] * bv0;
      acc5_1 = acc5_1 + sa[kk * 64u + lid.y * 8u + 5u] * bv1;
      acc6_0 = acc6_0 + sa[kk * 64u + lid.y * 8u + 6u] * bv0;
      acc6_1 = acc6_1 + sa[kk * 64u + lid.y * 8u + 6u] * bv1;
      acc7_0 = acc7_0 + sa[kk * 64u + lid.y * 8u + 7u] * bv0;
      acc7_1 = acc7_1 + sa[kk * 64u + lid.y * 8u + 7u] * bv1;
    }
    {
      let kk = 14u;
      let bv0 = sb[kk * 32u + lid.x];
      let bv1 = sb[kk * 32u + lid.x + 16u];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 8u + 0u] * bv0;
      acc0_1 = acc0_1 + sa[kk * 64u + lid.y * 8u + 0u] * bv1;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 8u + 1u] * bv0;
      acc1_1 = acc1_1 + sa[kk * 64u + lid.y * 8u + 1u] * bv1;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 8u + 2u] * bv0;
      acc2_1 = acc2_1 + sa[kk * 64u + lid.y * 8u + 2u] * bv1;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 8u + 3u] * bv0;
      acc3_1 = acc3_1 + sa[kk * 64u + lid.y * 8u + 3u] * bv1;
      acc4_0 = acc4_0 + sa[kk * 64u + lid.y * 8u + 4u] * bv0;
      acc4_1 = acc4_1 + sa[kk * 64u + lid.y * 8u + 4u] * bv1;
      acc5_0 = acc5_0 + sa[kk * 64u + lid.y * 8u + 5u] * bv0;
      acc5_1 = acc5_1 + sa[kk * 64u + lid.y * 8u + 5u] * bv1;
      acc6_0 = acc6_0 + sa[kk * 64u + lid.y * 8u + 6u] * bv0;
      acc6_1 = acc6_1 + sa[kk * 64u + lid.y * 8u + 6u] * bv1;
      acc7_0 = acc7_0 + sa[kk * 64u + lid.y * 8u + 7u] * bv0;
      acc7_1 = acc7_1 + sa[kk * 64u + lid.y * 8u + 7u] * bv1;
    }
    {
      let kk = 15u;
      let bv0 = sb[kk * 32u + lid.x];
      let bv1 = sb[kk * 32u + lid.x + 16u];
      acc0_0 = acc0_0 + sa[kk * 64u + lid.y * 8u + 0u] * bv0;
      acc0_1 = acc0_1 + sa[kk * 64u + lid.y * 8u + 0u] * bv1;
      acc1_0 = acc1_0 + sa[kk * 64u + lid.y * 8u + 1u] * bv0;
      acc1_1 = acc1_1 + sa[kk * 64u + lid.y * 8u + 1u] * bv1;
      acc2_0 = acc2_0 + sa[kk * 64u + lid.y * 8u + 2u] * bv0;
      acc2_1 = acc2_1 + sa[kk * 64u + lid.y * 8u + 2u] * bv1;
      acc3_0 = acc3_0 + sa[kk * 64u + lid.y * 8u + 3u] * bv0;
      acc3_1 = acc3_1 + sa[kk * 64u + lid.y * 8u + 3u] * bv1;
      acc4_0 = acc4_0 + sa[kk * 64u + lid.y * 8u + 4u] * bv0;
      acc4_1 = acc4_1 + sa[kk * 64u + lid.y * 8u + 4u] * bv1;
      acc5_0 = acc5_0 + sa[kk * 64u + lid.y * 8u + 5u] * bv0;
      acc5_1 = acc5_1 + sa[kk * 64u + lid.y * 8u + 5u] * bv1;
      acc6_0 = acc6_0 + sa[kk * 64u + lid.y * 8u + 6u] * bv0;
      acc6_1 = acc6_1 + sa[kk * 64u + lid.y * 8u + 6u] * bv1;
      acc7_0 = acc7_0 + sa[kk * 64u + lid.y * 8u + 7u] * bv0;
      acc7_1 = acc7_1 + sa[kk * 64u + lid.y * 8u + 7u] * bv1;
    }
    workgroupBarrier();
  }
  let ocq0 = wid.x * 32u + lid.x;
  let ocq1 = ocq0 + 16u;
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
      out[cbase + orow0 * n4 + ocq0] = acc0_0;
    }
    if (orow1 < dims.m) {
      out[cbase + orow1 * n4 + ocq0] = acc1_0;
    }
    if (orow2 < dims.m) {
      out[cbase + orow2 * n4 + ocq0] = acc2_0;
    }
    if (orow3 < dims.m) {
      out[cbase + orow3 * n4 + ocq0] = acc3_0;
    }
    if (orow4 < dims.m) {
      out[cbase + orow4 * n4 + ocq0] = acc4_0;
    }
    if (orow5 < dims.m) {
      out[cbase + orow5 * n4 + ocq0] = acc5_0;
    }
    if (orow6 < dims.m) {
      out[cbase + orow6 * n4 + ocq0] = acc6_0;
    }
    if (orow7 < dims.m) {
      out[cbase + orow7 * n4 + ocq0] = acc7_0;
    }
  }
  if (ocq1 < n4) {
    if (orow0 < dims.m) {
      out[cbase + orow0 * n4 + ocq1] = acc0_1;
    }
    if (orow1 < dims.m) {
      out[cbase + orow1 * n4 + ocq1] = acc1_1;
    }
    if (orow2 < dims.m) {
      out[cbase + orow2 * n4 + ocq1] = acc2_1;
    }
    if (orow3 < dims.m) {
      out[cbase + orow3 * n4 + ocq1] = acc3_1;
    }
    if (orow4 < dims.m) {
      out[cbase + orow4 * n4 + ocq1] = acc4_1;
    }
    if (orow5 < dims.m) {
      out[cbase + orow5 * n4 + ocq1] = acc5_1;
    }
    if (orow6 < dims.m) {
      out[cbase + orow6 * n4 + ocq1] = acc6_1;
    }
    if (orow7 < dims.m) {
      out[cbase + orow7 * n4 + ocq1] = acc7_1;
    }
  }
}
