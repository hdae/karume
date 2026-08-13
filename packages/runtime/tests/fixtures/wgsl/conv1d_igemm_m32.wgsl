// karume conv1d (x[B,Cin,L] * W[Cout,Cin,K] + b[Cout], f32, implicit GEMM レジスタ 32x128 タイル / 1 スレッド 8x8 / wg 16x4)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
  channels_in: u32,
  length_in: u32,
  kernel: u32,
  stride: u32,
  padding: u32,
  dilation: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

// Xcol[k][n] = x[b, ic, n·stride − padding + k·dilation]（範囲外は 0）。
// kt = k·dilation − padding は K タイル内で不変なので呼び出し側が畳んで渡す。
// xc = b·Cin + ic（バッチは dispatch の z 軸なので呼び出し側で足す）
fn xcol(xc: u32, kt: i32, n: u32) -> f32 {
  let ix = i32(n * dims.stride) + kt;
  if (ix < 0 || u32(ix) >= dims.length_in) {
    return 0.0;
  }
  return x[xc * dims.length_in + u32(ix)];
}

// 共有 A タイル（32 行 × K 16・スカラ格納）と
// 共有 B タイル（K 16 × 列 quad 32・列方向を vec4 に束ねた形）
var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<vec4<f32>, 512>;

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
  // バッチは workgroup 単位で一様（z 軸 1 つが 1 バッチの出力平面 [Cout, Lout]）
  let xbase = wid.z * dims.channels_in;
  let cbase = wid.z * dims.m * dims.n;
  // B タイルの担当（K 16 行 × 列 quad 32 を 64 スレッドで 8 巡）
  let bk0 = tid / 32u;
  let bcq = tid % 32u;
  let bcol = wid.x * 128u + bcq * 4u;
  let bk1 = bk0 + 2u;
  let bk2 = bk0 + 4u;
  let bk3 = bk0 + 6u;
  let bk4 = bk0 + 8u;
  let bk5 = bk0 + 10u;
  let bk6 = bk0 + 12u;
  let bk7 = bk0 + 14u;
  // ループ条件は uniform（dims は uniform バッファ）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすために必要
  let tiles = (dims.k + 15u) / 16u;
  let bias0 = wid.y * 32u + lid.y * 8u;
  var acc0_0 = vec4<f32>(bias[bias0]);
  var acc0_1 = vec4<f32>(bias[bias0]);
  var acc1_0 = vec4<f32>(bias[bias0 + 1u]);
  var acc1_1 = vec4<f32>(bias[bias0 + 1u]);
  var acc2_0 = vec4<f32>(bias[bias0 + 2u]);
  var acc2_1 = vec4<f32>(bias[bias0 + 2u]);
  var acc3_0 = vec4<f32>(bias[bias0 + 3u]);
  var acc3_1 = vec4<f32>(bias[bias0 + 3u]);
  var acc4_0 = vec4<f32>(bias[bias0 + 4u]);
  var acc4_1 = vec4<f32>(bias[bias0 + 4u]);
  var acc5_0 = vec4<f32>(bias[bias0 + 5u]);
  var acc5_1 = vec4<f32>(bias[bias0 + 5u]);
  var acc6_0 = vec4<f32>(bias[bias0 + 6u]);
  var acc6_1 = vec4<f32>(bias[bias0 + 6u]);
  var acc7_0 = vec4<f32>(bias[bias0 + 7u]);
  var acc7_1 = vec4<f32>(bias[bias0 + 7u]);
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
      let ic = brow0 / dims.kernel;
      let kt = i32((brow0 % dims.kernel) * dims.dilation) - i32(dims.padding);
      let xc = xbase + ic;
      if (bcol < dims.n) {
        bv4_0.x = xcol(xc, kt, bcol);
      }
      if (bcol + 1u < dims.n) {
        bv4_0.y = xcol(xc, kt, bcol + 1u);
      }
      if (bcol + 2u < dims.n) {
        bv4_0.z = xcol(xc, kt, bcol + 2u);
      }
      if (bcol + 3u < dims.n) {
        bv4_0.w = xcol(xc, kt, bcol + 3u);
      }
    }
    sb[bk0 * 32u + bcq] = bv4_0;
    let brow1 = t * 16u + bk1;
    var bv4_1 = vec4<f32>(0.0);
    if (brow1 < dims.k) {
      let ic = brow1 / dims.kernel;
      let kt = i32((brow1 % dims.kernel) * dims.dilation) - i32(dims.padding);
      let xc = xbase + ic;
      if (bcol < dims.n) {
        bv4_1.x = xcol(xc, kt, bcol);
      }
      if (bcol + 1u < dims.n) {
        bv4_1.y = xcol(xc, kt, bcol + 1u);
      }
      if (bcol + 2u < dims.n) {
        bv4_1.z = xcol(xc, kt, bcol + 2u);
      }
      if (bcol + 3u < dims.n) {
        bv4_1.w = xcol(xc, kt, bcol + 3u);
      }
    }
    sb[bk1 * 32u + bcq] = bv4_1;
    let brow2 = t * 16u + bk2;
    var bv4_2 = vec4<f32>(0.0);
    if (brow2 < dims.k) {
      let ic = brow2 / dims.kernel;
      let kt = i32((brow2 % dims.kernel) * dims.dilation) - i32(dims.padding);
      let xc = xbase + ic;
      if (bcol < dims.n) {
        bv4_2.x = xcol(xc, kt, bcol);
      }
      if (bcol + 1u < dims.n) {
        bv4_2.y = xcol(xc, kt, bcol + 1u);
      }
      if (bcol + 2u < dims.n) {
        bv4_2.z = xcol(xc, kt, bcol + 2u);
      }
      if (bcol + 3u < dims.n) {
        bv4_2.w = xcol(xc, kt, bcol + 3u);
      }
    }
    sb[bk2 * 32u + bcq] = bv4_2;
    let brow3 = t * 16u + bk3;
    var bv4_3 = vec4<f32>(0.0);
    if (brow3 < dims.k) {
      let ic = brow3 / dims.kernel;
      let kt = i32((brow3 % dims.kernel) * dims.dilation) - i32(dims.padding);
      let xc = xbase + ic;
      if (bcol < dims.n) {
        bv4_3.x = xcol(xc, kt, bcol);
      }
      if (bcol + 1u < dims.n) {
        bv4_3.y = xcol(xc, kt, bcol + 1u);
      }
      if (bcol + 2u < dims.n) {
        bv4_3.z = xcol(xc, kt, bcol + 2u);
      }
      if (bcol + 3u < dims.n) {
        bv4_3.w = xcol(xc, kt, bcol + 3u);
      }
    }
    sb[bk3 * 32u + bcq] = bv4_3;
    let brow4 = t * 16u + bk4;
    var bv4_4 = vec4<f32>(0.0);
    if (brow4 < dims.k) {
      let ic = brow4 / dims.kernel;
      let kt = i32((brow4 % dims.kernel) * dims.dilation) - i32(dims.padding);
      let xc = xbase + ic;
      if (bcol < dims.n) {
        bv4_4.x = xcol(xc, kt, bcol);
      }
      if (bcol + 1u < dims.n) {
        bv4_4.y = xcol(xc, kt, bcol + 1u);
      }
      if (bcol + 2u < dims.n) {
        bv4_4.z = xcol(xc, kt, bcol + 2u);
      }
      if (bcol + 3u < dims.n) {
        bv4_4.w = xcol(xc, kt, bcol + 3u);
      }
    }
    sb[bk4 * 32u + bcq] = bv4_4;
    let brow5 = t * 16u + bk5;
    var bv4_5 = vec4<f32>(0.0);
    if (brow5 < dims.k) {
      let ic = brow5 / dims.kernel;
      let kt = i32((brow5 % dims.kernel) * dims.dilation) - i32(dims.padding);
      let xc = xbase + ic;
      if (bcol < dims.n) {
        bv4_5.x = xcol(xc, kt, bcol);
      }
      if (bcol + 1u < dims.n) {
        bv4_5.y = xcol(xc, kt, bcol + 1u);
      }
      if (bcol + 2u < dims.n) {
        bv4_5.z = xcol(xc, kt, bcol + 2u);
      }
      if (bcol + 3u < dims.n) {
        bv4_5.w = xcol(xc, kt, bcol + 3u);
      }
    }
    sb[bk5 * 32u + bcq] = bv4_5;
    let brow6 = t * 16u + bk6;
    var bv4_6 = vec4<f32>(0.0);
    if (brow6 < dims.k) {
      let ic = brow6 / dims.kernel;
      let kt = i32((brow6 % dims.kernel) * dims.dilation) - i32(dims.padding);
      let xc = xbase + ic;
      if (bcol < dims.n) {
        bv4_6.x = xcol(xc, kt, bcol);
      }
      if (bcol + 1u < dims.n) {
        bv4_6.y = xcol(xc, kt, bcol + 1u);
      }
      if (bcol + 2u < dims.n) {
        bv4_6.z = xcol(xc, kt, bcol + 2u);
      }
      if (bcol + 3u < dims.n) {
        bv4_6.w = xcol(xc, kt, bcol + 3u);
      }
    }
    sb[bk6 * 32u + bcq] = bv4_6;
    let brow7 = t * 16u + bk7;
    var bv4_7 = vec4<f32>(0.0);
    if (brow7 < dims.k) {
      let ic = brow7 / dims.kernel;
      let kt = i32((brow7 % dims.kernel) * dims.dilation) - i32(dims.padding);
      let xc = xbase + ic;
      if (bcol < dims.n) {
        bv4_7.x = xcol(xc, kt, bcol);
      }
      if (bcol + 1u < dims.n) {
        bv4_7.y = xcol(xc, kt, bcol + 1u);
      }
      if (bcol + 2u < dims.n) {
        bv4_7.z = xcol(xc, kt, bcol + 2u);
      }
      if (bcol + 3u < dims.n) {
        bv4_7.w = xcol(xc, kt, bcol + 3u);
      }
    }
    sb[bk7 * 32u + bcq] = bv4_7;
    workgroupBarrier();
    // 共有ロード 10 回（B の vec4 2 + A のスカラ 8）で 64 MAC。
    // 縮約は k 昇順の逐次で、1 出力要素あたりの加算順序は 16×16 の 1 スレッド 1 出力と
    // 完全に一致する。
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
  let ocol = wid.x * 128u + lid.x * 8u;
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
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = acc0_1.x;
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = acc0_1.y;
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = acc0_1.z;
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = acc0_1.w;
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
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = acc1_1.x;
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = acc1_1.y;
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = acc1_1.z;
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = acc1_1.w;
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
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = acc2_1.x;
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = acc2_1.y;
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = acc2_1.z;
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = acc2_1.w;
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
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = acc3_1.x;
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = acc3_1.y;
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = acc3_1.z;
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = acc3_1.w;
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
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = acc4_1.x;
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = acc4_1.y;
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = acc4_1.z;
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = acc4_1.w;
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
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = acc5_1.x;
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = acc5_1.y;
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = acc5_1.z;
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = acc5_1.w;
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
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = acc6_1.x;
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = acc6_1.y;
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = acc6_1.z;
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = acc6_1.w;
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
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = acc7_1.x;
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = acc7_1.y;
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = acc7_1.z;
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = acc7_1.w;
    }
  }
}
