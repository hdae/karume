// karume conv2d (x[B,Cin,H,W] * W[Cout,Cin,Kh,Kw] + b[Cout], f32, implicit GEMM レジスタ 32x64 タイル / 1 スレッド 8x4 / wg 16x4)
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
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

// Xcol[k][n] = x[b, ic, oy·sh − ph + kh·dh, ox·sw − pw + kw·dw]（範囲外は 0）。
// xc = b·Cin + ic（バッチは dispatch の z 軸なので呼び出し側で足す）
fn xcol(xc: u32, ky: i32, kx: i32, n: u32) -> f32 {
  let iy = i32((n / dims.width_out) * dims.stride_h) + ky;
  let ix = i32((n % dims.width_out) * dims.stride_w) + kx;
  if (iy < 0 || u32(iy) >= dims.height_in || ix < 0 || u32(ix) >= dims.width_in) {
    return 0.0;
  }
  return x[(xc * dims.height_in + u32(iy)) * dims.width_in + u32(ix)];
}

// 共有 A タイル（32 行 × K 16・スカラ格納）と
// 共有 B タイル（K 16 × 列 quad 16・列方向を vec4 に束ねた形）
var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 4)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 16u + lid.x;
  // A タイルの担当（32 行 × 4 quad を 64 スレッドで 2 巡）
  let ar = tid / 4u;
  let aq = tid % 4u;
  let arow0 = wid.y * 32u + ar;
  let arow_base0 = arow0 * dims.k;
  let sa_base0 = ar * 16u + aq * 4u;
  let arow1 = arow0 + 16u;
  let arow_base1 = arow_base0 + 16u * dims.k;
  let sa_base1 = sa_base0 + 256u;
  // バッチは workgroup 単位で一様（z 軸 1 つが 1 バッチの出力平面 [Cout, Hout·Wout]）
  let xbase = wid.z * dims.channels_in;
  let cbase = wid.z * dims.m * dims.n;
  // B タイルの担当（K 16 行 × 列 quad 16 を 64 スレッドで 4 巡）
  let bk0 = tid / 16u;
  let bcq = tid % 16u;
  let bcol = wid.x * 64u + bcq * 4u;
  let bk1 = bk0 + 4u;
  let bk2 = bk0 + 8u;
  let bk3 = bk0 + 12u;
  // K タイルループ不変（平坦 k を (ic, kh, kw) へ割るための刻み）
  let khw = dims.kernel_h * dims.kernel_w;
  // ループ条件は uniform（dims は uniform バッファ）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすために必要
  let tiles = (dims.k + 15u) / 16u;
  let bias0 = wid.y * 32u + lid.y * 8u;
  var acc0_0 = vec4<f32>(bias[bias0]);
  var acc1_0 = vec4<f32>(bias[bias0 + 1u]);
  var acc2_0 = vec4<f32>(bias[bias0 + 2u]);
  var acc3_0 = vec4<f32>(bias[bias0 + 3u]);
  var acc4_0 = vec4<f32>(bias[bias0 + 4u]);
  var acc5_0 = vec4<f32>(bias[bias0 + 5u]);
  var acc6_0 = vec4<f32>(bias[bias0 + 6u]);
  var acc7_0 = vec4<f32>(bias[bias0 + 7u]);
  for (var t = 0u; t < tiles; t = t + 1u) {
    // 範囲外は 0 で埋める。内積に寄与しないので K 端数でも結果は変わらない
    let ak0 = t * 16u + aq * 4u;
    var av0 = vec4<f32>(0.0);
    if (arow0 < dims.m) {
      let abase = arow_base0 + ak0;
      if (ak0 < dims.k) {
        av0.x = w[abase];
      }
      if (ak0 + 1u < dims.k) {
        av0.y = w[abase + 1u];
      }
      if (ak0 + 2u < dims.k) {
        av0.z = w[abase + 2u];
      }
      if (ak0 + 3u < dims.k) {
        av0.w = w[abase + 3u];
      }
    }
    sa[sa_base0] = av0.x;
    sa[sa_base0 + 1u] = av0.y;
    sa[sa_base0 + 2u] = av0.z;
    sa[sa_base0 + 3u] = av0.w;
    var av1 = vec4<f32>(0.0);
    if (arow1 < dims.m) {
      let abase = arow_base1 + ak0;
      if (ak0 < dims.k) {
        av1.x = w[abase];
      }
      if (ak0 + 1u < dims.k) {
        av1.y = w[abase + 1u];
      }
      if (ak0 + 2u < dims.k) {
        av1.z = w[abase + 2u];
      }
      if (ak0 + 3u < dims.k) {
        av1.w = w[abase + 3u];
      }
    }
    sa[sa_base1] = av1.x;
    sa[sa_base1 + 1u] = av1.y;
    sa[sa_base1 + 2u] = av1.z;
    sa[sa_base1 + 3u] = av1.w;
    let brow0 = t * 16u + bk0;
    var bv4_0 = vec4<f32>(0.0);
    if (brow0 < dims.k) {
      let ic = brow0 / khw;
      let kr = brow0 % khw;
      let ky = i32((kr / dims.kernel_w) * dims.dilation_h) - i32(dims.padding_h);
      let kx = i32((kr % dims.kernel_w) * dims.dilation_w) - i32(dims.padding_w);
      let xc = xbase + ic;
      if (bcol < dims.n) {
        bv4_0.x = xcol(xc, ky, kx, bcol);
      }
      if (bcol + 1u < dims.n) {
        bv4_0.y = xcol(xc, ky, kx, bcol + 1u);
      }
      if (bcol + 2u < dims.n) {
        bv4_0.z = xcol(xc, ky, kx, bcol + 2u);
      }
      if (bcol + 3u < dims.n) {
        bv4_0.w = xcol(xc, ky, kx, bcol + 3u);
      }
    }
    sb[bk0 * 16u + bcq] = bv4_0;
    let brow1 = t * 16u + bk1;
    var bv4_1 = vec4<f32>(0.0);
    if (brow1 < dims.k) {
      let ic = brow1 / khw;
      let kr = brow1 % khw;
      let ky = i32((kr / dims.kernel_w) * dims.dilation_h) - i32(dims.padding_h);
      let kx = i32((kr % dims.kernel_w) * dims.dilation_w) - i32(dims.padding_w);
      let xc = xbase + ic;
      if (bcol < dims.n) {
        bv4_1.x = xcol(xc, ky, kx, bcol);
      }
      if (bcol + 1u < dims.n) {
        bv4_1.y = xcol(xc, ky, kx, bcol + 1u);
      }
      if (bcol + 2u < dims.n) {
        bv4_1.z = xcol(xc, ky, kx, bcol + 2u);
      }
      if (bcol + 3u < dims.n) {
        bv4_1.w = xcol(xc, ky, kx, bcol + 3u);
      }
    }
    sb[bk1 * 16u + bcq] = bv4_1;
    let brow2 = t * 16u + bk2;
    var bv4_2 = vec4<f32>(0.0);
    if (brow2 < dims.k) {
      let ic = brow2 / khw;
      let kr = brow2 % khw;
      let ky = i32((kr / dims.kernel_w) * dims.dilation_h) - i32(dims.padding_h);
      let kx = i32((kr % dims.kernel_w) * dims.dilation_w) - i32(dims.padding_w);
      let xc = xbase + ic;
      if (bcol < dims.n) {
        bv4_2.x = xcol(xc, ky, kx, bcol);
      }
      if (bcol + 1u < dims.n) {
        bv4_2.y = xcol(xc, ky, kx, bcol + 1u);
      }
      if (bcol + 2u < dims.n) {
        bv4_2.z = xcol(xc, ky, kx, bcol + 2u);
      }
      if (bcol + 3u < dims.n) {
        bv4_2.w = xcol(xc, ky, kx, bcol + 3u);
      }
    }
    sb[bk2 * 16u + bcq] = bv4_2;
    let brow3 = t * 16u + bk3;
    var bv4_3 = vec4<f32>(0.0);
    if (brow3 < dims.k) {
      let ic = brow3 / khw;
      let kr = brow3 % khw;
      let ky = i32((kr / dims.kernel_w) * dims.dilation_h) - i32(dims.padding_h);
      let kx = i32((kr % dims.kernel_w) * dims.dilation_w) - i32(dims.padding_w);
      let xc = xbase + ic;
      if (bcol < dims.n) {
        bv4_3.x = xcol(xc, ky, kx, bcol);
      }
      if (bcol + 1u < dims.n) {
        bv4_3.y = xcol(xc, ky, kx, bcol + 1u);
      }
      if (bcol + 2u < dims.n) {
        bv4_3.z = xcol(xc, ky, kx, bcol + 2u);
      }
      if (bcol + 3u < dims.n) {
        bv4_3.w = xcol(xc, ky, kx, bcol + 3u);
      }
    }
    sb[bk3 * 16u + bcq] = bv4_3;
    workgroupBarrier();
    // 共有ロード 9 回（B の vec4 1 + A のスカラ 8）で 32 MAC。
    // 縮約は k 昇順の逐次で、1 出力要素あたりの加算順序は 16×16 の 1 スレッド 1 出力と
    // 完全に一致する。
    for (var kk = 0u; kk < 16u; kk = kk + 1u) {
      let bv0 = sb[kk * 16u + lid.x];
      acc0_0 = acc0_0 + sa[(lid.y * 8u + 0u) * 16u + kk] * bv0;
      acc1_0 = acc1_0 + sa[(lid.y * 8u + 1u) * 16u + kk] * bv0;
      acc2_0 = acc2_0 + sa[(lid.y * 8u + 2u) * 16u + kk] * bv0;
      acc3_0 = acc3_0 + sa[(lid.y * 8u + 3u) * 16u + kk] * bv0;
      acc4_0 = acc4_0 + sa[(lid.y * 8u + 4u) * 16u + kk] * bv0;
      acc5_0 = acc5_0 + sa[(lid.y * 8u + 5u) * 16u + kk] * bv0;
      acc6_0 = acc6_0 + sa[(lid.y * 8u + 6u) * 16u + kk] * bv0;
      acc7_0 = acc7_0 + sa[(lid.y * 8u + 7u) * 16u + kk] * bv0;
    }
    workgroupBarrier();
  }
  let ocol = wid.x * 64u + lid.x * 4u;
  let orow0 = wid.y * 32u + lid.y * 8u;
  let orow1 = orow0 + 1u;
  let orow2 = orow0 + 2u;
  let orow3 = orow0 + 3u;
  let orow4 = orow0 + 4u;
  let orow5 = orow0 + 5u;
  let orow6 = orow0 + 6u;
  let orow7 = orow0 + 7u;
  if (orow0 < dims.m) {
    let obase = cbase + orow0 * dims.n;
    if (ocol < dims.n) {
      out[obase + ocol] = acc0_0.x;
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = acc0_0.y;
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = acc0_0.z;
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = acc0_0.w;
    }
  }
  if (orow1 < dims.m) {
    let obase = cbase + orow1 * dims.n;
    if (ocol < dims.n) {
      out[obase + ocol] = acc1_0.x;
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = acc1_0.y;
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = acc1_0.z;
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = acc1_0.w;
    }
  }
  if (orow2 < dims.m) {
    let obase = cbase + orow2 * dims.n;
    if (ocol < dims.n) {
      out[obase + ocol] = acc2_0.x;
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = acc2_0.y;
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = acc2_0.z;
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = acc2_0.w;
    }
  }
  if (orow3 < dims.m) {
    let obase = cbase + orow3 * dims.n;
    if (ocol < dims.n) {
      out[obase + ocol] = acc3_0.x;
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = acc3_0.y;
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = acc3_0.z;
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = acc3_0.w;
    }
  }
  if (orow4 < dims.m) {
    let obase = cbase + orow4 * dims.n;
    if (ocol < dims.n) {
      out[obase + ocol] = acc4_0.x;
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = acc4_0.y;
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = acc4_0.z;
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = acc4_0.w;
    }
  }
  if (orow5 < dims.m) {
    let obase = cbase + orow5 * dims.n;
    if (ocol < dims.n) {
      out[obase + ocol] = acc5_0.x;
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = acc5_0.y;
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = acc5_0.z;
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = acc5_0.w;
    }
  }
  if (orow6 < dims.m) {
    let obase = cbase + orow6 * dims.n;
    if (ocol < dims.n) {
      out[obase + ocol] = acc6_0.x;
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = acc6_0.y;
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = acc6_0.z;
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = acc6_0.w;
    }
  }
  if (orow7 < dims.m) {
    let obase = cbase + orow7 * dims.n;
    if (ocol < dims.n) {
      out[obase + ocol] = acc7_0.x;
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = acc7_0.y;
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = acc7_0.z;
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = acc7_0.w;
    }
  }
}
