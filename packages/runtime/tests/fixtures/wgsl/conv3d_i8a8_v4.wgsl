// karume conv3d (x[Cin,T,H,W] * W[Cout,Cin,Kt,Kh,Kw] + b[Cout], integer dot product of per-pixel i8 activations × i8 weights, tile 128x64 / 8x8 per thread / wg 8x16 / K 16 + vec4)
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
@group(0) @binding(1) var<storage, read> xq: array<u32>;
@group(0) @binding(2) var<storage, read> w: array<u32>; // tap-major [Cout][Kt·Kh·Kw][Cin], four channels per word
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> wscale: array<f32>;
@group(0) @binding(6) var<storage, read> xscale: array<f32>;

// integer dot product (WGSL language extension packed_4x8_integer_dot_product); returns **the same integer** as the emulated variant
fn idot(a: u32, b: u32) -> i32 {
  return dot4I8Packed(a, b);
}

// Input pixel of output pixel n under the tap offsets (the dilated tap minus the padding); −1 out of
// range. n = (ot·Hout + oy)·Wout + ox; the pixel is (it·Hin + iy)·Win + ix of the transposed
// [T·H·W][Cin] activation
fn xpix(n: u32, kt: i32, ky: i32, kx: i32) -> i32 {
  let plane = dims.height_out * dims.width_out;
  let pixel = n % plane;
  let it = i32((n / plane) * dims.stride_t) + kt;
  let iy = i32((pixel / dims.width_out) * dims.stride_h) + ky;
  let ix = i32((pixel % dims.width_out) * dims.stride_w) + kx;
  if (it < 0 || u32(it) >= dims.time_in || iy < 0 || u32(iy) >= dims.height_in ||
      ix < 0 || u32(ix) >= dims.width_in) {
    return -1;
  }
  return i32((u32(it) * dims.height_in + u32(iy)) * dims.width_in + u32(ix));
}

// Shared tiles are laid out [pack][row] / [pack][column] (the i8a8 family's layout)
var<workgroup> sa: array<u32, 512>;
var<workgroup> sb: array<u32, 256>;

@compute @workgroup_size(8, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 8u + lid.x;
  // Eligibility requires Cin % 4 == 0, so a pixel's channels start on a word boundary
  let cin4 = dims.channels_in / 4u;
  let khw = dims.kernel_h * dims.kernel_w;
  let taps = dims.kernel_t * khw;
  // W tile assignment (128 output channels × 4 packs over 128 threads in 4 rounds). abase is the word number of the row's first pack
  // (tap-major [Cout][taps][Cin], four channels per word)
  let ar = tid / 4u;
  let ap = tid % 4u;
  let sa_at = ap * 128u + ar;
  let arow0 = wid.y * 128u + ar;
  let abase0 = arow0 * taps * cin4;
  let arow1 = arow0 + 32u;
  let abase1 = arow1 * taps * cin4;
  let arow2 = arow0 + 64u;
  let abase2 = arow2 * taps * cin4;
  let arow3 = arow0 + 96u;
  let abase3 = arow3 * taps * cin4;
  // x tile assignment (64 output pixels × 4 packs over 128 threads in 2 rounds)
  let wc = tid / 4u;
  let wp = tid % 4u;
  let sb_at = wp * 64u + wc;
  let wcol0 = wid.x * 64u + wc;
  let wcol1 = wcol0 + 32u;
  // Output assignment is **outside the loop** (the flush at every tap looks up per-column pixel
  // scales)
  let ocol = wid.x * 64u + lid.x * 8u;
  let orow0 = wid.y * 128u + lid.y * 8u;
  let orow1 = orow0 + 1u;
  let orow2 = orow0 + 2u;
  let orow3 = orow0 + 3u;
  let orow4 = orow0 + 4u;
  let orow5 = orow0 + 5u;
  let orow6 = orow0 + 6u;
  let orow7 = orow0 + 7u;
  // f32 accumulator (written only at tap boundaries — Kt·Kh·Kw roundings)
  var accf0_0 = vec4<f32>(0.0);
  var accf0_1 = vec4<f32>(0.0);
  var accf1_0 = vec4<f32>(0.0);
  var accf1_1 = vec4<f32>(0.0);
  var accf2_0 = vec4<f32>(0.0);
  var accf2_1 = vec4<f32>(0.0);
  var accf3_0 = vec4<f32>(0.0);
  var accf3_1 = vec4<f32>(0.0);
  var accf4_0 = vec4<f32>(0.0);
  var accf4_1 = vec4<f32>(0.0);
  var accf5_0 = vec4<f32>(0.0);
  var accf5_1 = vec4<f32>(0.0);
  var accf6_0 = vec4<f32>(0.0);
  var accf6_1 = vec4<f32>(0.0);
  var accf7_0 = vec4<f32>(0.0);
  var accf7_1 = vec4<f32>(0.0);
  // The loop bounds are uniform (dims is a uniform buffer) — needed for the inner
  // workgroupBarrier to meet WGSL's uniformity requirement
  let tiles = (cin4 + 3u) / 4u;
  for (var tap = 0u; tap < taps; tap = tap + 1u) {
    // The tap's offsets into the input (dilated, minus the padding — the decomposition of
    // src/kernels/gemm.ts conv3dKDecode, with ci taken out of the flat k)
    let kt = i32((tap / khw) * dims.dilation_t) - i32(dims.padding_t);
    let ks = tap % khw;
    let ky = i32((ks / dims.kernel_w) * dims.dilation_h) - i32(dims.padding_h);
    let kx = i32((ks % dims.kernel_w) * dims.dilation_w) - i32(dims.padding_w);
    var bpix0 = -1;
    if (wcol0 < dims.n) {
      bpix0 = xpix(wcol0, kt, ky, kx);
    }
    var bpix1 = -1;
    if (wcol1 < dims.n) {
      bpix1 = xpix(wcol1, kt, ky, kx);
    }
    // The i32 accumulator is reset per tap (|acci| ≤ Cin·127·128 — limit: CONV3D_I8A8_MAX_CHANNELS_IN)
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
      // A side: one word of the row's pack at this tap (tap-major [Cout][taps][Cin])
      let apack = t * 4u + ap;
      var av0 = 0u;
      if (arow0 < dims.m && apack < cin4) {
        av0 = w[abase0 + tap * cin4 + apack];
      }
      sa[sa_at] = av0;
      var av1 = 0u;
      if (arow1 < dims.m && apack < cin4) {
        av1 = w[abase1 + tap * cin4 + apack];
      }
      sa[sa_at + 32u] = av1;
      var av2 = 0u;
      if (arow2 < dims.m && apack < cin4) {
        av2 = w[abase2 + tap * cin4 + apack];
      }
      sa[sa_at + 64u] = av2;
      var av3 = 0u;
      if (arow3 < dims.m && apack < cin4) {
        av3 = w[abase3 + tap * cin4 + apack];
      }
      sa[sa_at + 96u] = av3;
      // B side: one word of the input pixel's packed channels (0 out of range — the MUST of the
      // out-of-range tap)
      let wpack = t * 4u + wp;
      var bv0 = 0u;
      if (bpix0 >= 0 && wpack < cin4) {
        bv0 = xq[u32(bpix0) * cin4 + wpack];
      }
      sb[sb_at] = bv0;
      var bv1 = 0u;
      if (bpix1 >= 0 && wpack < cin4) {
        bv1 = xq[u32(bpix1) * cin4 + wpack];
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
    // Flush to f32 exactly at the tap boundary with the input pixel's scale per column (MUST:
    // single rounding of fma; wscale is not applied until the end)
    var xs0 = 0.0;
    if (ocol < dims.n) {
      let pix0 = xpix(ocol, kt, ky, kx);
      if (pix0 >= 0) {
        xs0 = xscale[u32(pix0)];
      }
    }
    var xs1 = 0.0;
    if (ocol + 1u < dims.n) {
      let pix1 = xpix(ocol + 1u, kt, ky, kx);
      if (pix1 >= 0) {
        xs1 = xscale[u32(pix1)];
      }
    }
    var xs2 = 0.0;
    if (ocol + 2u < dims.n) {
      let pix2 = xpix(ocol + 2u, kt, ky, kx);
      if (pix2 >= 0) {
        xs2 = xscale[u32(pix2)];
      }
    }
    var xs3 = 0.0;
    if (ocol + 3u < dims.n) {
      let pix3 = xpix(ocol + 3u, kt, ky, kx);
      if (pix3 >= 0) {
        xs3 = xscale[u32(pix3)];
      }
    }
    var xs4 = 0.0;
    if (ocol + 4u < dims.n) {
      let pix4 = xpix(ocol + 4u, kt, ky, kx);
      if (pix4 >= 0) {
        xs4 = xscale[u32(pix4)];
      }
    }
    var xs5 = 0.0;
    if (ocol + 5u < dims.n) {
      let pix5 = xpix(ocol + 5u, kt, ky, kx);
      if (pix5 >= 0) {
        xs5 = xscale[u32(pix5)];
      }
    }
    var xs6 = 0.0;
    if (ocol + 6u < dims.n) {
      let pix6 = xpix(ocol + 6u, kt, ky, kx);
      if (pix6 >= 0) {
        xs6 = xscale[u32(pix6)];
      }
    }
    var xs7 = 0.0;
    if (ocol + 7u < dims.n) {
      let pix7 = xpix(ocol + 7u, kt, ky, kx);
      if (pix7 >= 0) {
        xs7 = xscale[u32(pix7)];
      }
    }
    let xsv0 = vec4<f32>(xs0, xs1, xs2, xs3);
    accf0_0 = fma(vec4<f32>(acc0_0), xsv0, accf0_0);
    accf1_0 = fma(vec4<f32>(acc1_0), xsv0, accf1_0);
    accf2_0 = fma(vec4<f32>(acc2_0), xsv0, accf2_0);
    accf3_0 = fma(vec4<f32>(acc3_0), xsv0, accf3_0);
    accf4_0 = fma(vec4<f32>(acc4_0), xsv0, accf4_0);
    accf5_0 = fma(vec4<f32>(acc5_0), xsv0, accf5_0);
    accf6_0 = fma(vec4<f32>(acc6_0), xsv0, accf6_0);
    accf7_0 = fma(vec4<f32>(acc7_0), xsv0, accf7_0);
    let xsv1 = vec4<f32>(xs4, xs5, xs6, xs7);
    accf0_1 = fma(vec4<f32>(acc0_1), xsv1, accf0_1);
    accf1_1 = fma(vec4<f32>(acc1_1), xsv1, accf1_1);
    accf2_1 = fma(vec4<f32>(acc2_1), xsv1, accf2_1);
    accf3_1 = fma(vec4<f32>(acc3_1), xsv1, accf3_1);
    accf4_1 = fma(vec4<f32>(acc4_1), xsv1, accf4_1);
    accf5_1 = fma(vec4<f32>(acc5_1), xsv1, accf5_1);
    accf6_1 = fma(vec4<f32>(acc6_1), xsv1, accf6_1);
    accf7_1 = fma(vec4<f32>(acc7_1), xsv1, accf7_1);
  }
  let n4 = dims.n / 4u;
  let ocq0 = ocol / 4u;
  let ocq1 = ocq0 + 1u;
  if (ocq0 < n4) {
    // n % 4 == 0 and ocq0 < n4, so the four pixels of the word are in range (bias is
    // always f32 — ADR 0006)
    // MUST: wscale goes to the final fma (single rounding) — xs was folded at the tap boundaries
    if (orow0 < dims.m) {
      out[orow0 * n4 + ocq0] = fma(accf0_0, vec4<f32>(wscale[orow0]), vec4<f32>(bias[orow0]));
    }
    if (orow1 < dims.m) {
      out[orow1 * n4 + ocq0] = fma(accf1_0, vec4<f32>(wscale[orow1]), vec4<f32>(bias[orow1]));
    }
    if (orow2 < dims.m) {
      out[orow2 * n4 + ocq0] = fma(accf2_0, vec4<f32>(wscale[orow2]), vec4<f32>(bias[orow2]));
    }
    if (orow3 < dims.m) {
      out[orow3 * n4 + ocq0] = fma(accf3_0, vec4<f32>(wscale[orow3]), vec4<f32>(bias[orow3]));
    }
    if (orow4 < dims.m) {
      out[orow4 * n4 + ocq0] = fma(accf4_0, vec4<f32>(wscale[orow4]), vec4<f32>(bias[orow4]));
    }
    if (orow5 < dims.m) {
      out[orow5 * n4 + ocq0] = fma(accf5_0, vec4<f32>(wscale[orow5]), vec4<f32>(bias[orow5]));
    }
    if (orow6 < dims.m) {
      out[orow6 * n4 + ocq0] = fma(accf6_0, vec4<f32>(wscale[orow6]), vec4<f32>(bias[orow6]));
    }
    if (orow7 < dims.m) {
      out[orow7 * n4 + ocq0] = fma(accf7_0, vec4<f32>(wscale[orow7]), vec4<f32>(bias[orow7]));
    }
  }
  if (ocq1 < n4) {
    // n % 4 == 0 and ocq1 < n4, so the four pixels of the word are in range (bias is
    // always f32 — ADR 0006)
    // MUST: wscale goes to the final fma (single rounding) — xs was folded at the tap boundaries
    if (orow0 < dims.m) {
      out[orow0 * n4 + ocq1] = fma(accf0_1, vec4<f32>(wscale[orow0]), vec4<f32>(bias[orow0]));
    }
    if (orow1 < dims.m) {
      out[orow1 * n4 + ocq1] = fma(accf1_1, vec4<f32>(wscale[orow1]), vec4<f32>(bias[orow1]));
    }
    if (orow2 < dims.m) {
      out[orow2 * n4 + ocq1] = fma(accf2_1, vec4<f32>(wscale[orow2]), vec4<f32>(bias[orow2]));
    }
    if (orow3 < dims.m) {
      out[orow3 * n4 + ocq1] = fma(accf3_1, vec4<f32>(wscale[orow3]), vec4<f32>(bias[orow3]));
    }
    if (orow4 < dims.m) {
      out[orow4 * n4 + ocq1] = fma(accf4_1, vec4<f32>(wscale[orow4]), vec4<f32>(bias[orow4]));
    }
    if (orow5 < dims.m) {
      out[orow5 * n4 + ocq1] = fma(accf5_1, vec4<f32>(wscale[orow5]), vec4<f32>(bias[orow5]));
    }
    if (orow6 < dims.m) {
      out[orow6 * n4 + ocq1] = fma(accf6_1, vec4<f32>(wscale[orow6]), vec4<f32>(bias[orow6]));
    }
    if (orow7 < dims.m) {
      out[orow7 * n4 + ocq1] = fma(accf7_1, vec4<f32>(wscale[orow7]), vec4<f32>(bias[orow7]));
    }
  }
}
