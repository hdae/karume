// karume conv2d (x[B,Cin,H,W] * W[Cout,Cin,Kh,Kw] + b[Cout], f32, implicit GEMM レジスタ 32x64 タイル)
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

@compute @workgroup_size(16, 8)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 16u + lid.x;
  // A タイルの担当（32 行 × 4 quad = 128 スレッド）
  let ar = tid / 4u;
  let aq = tid % 4u;
  let arow = wid.y * 32u + ar;
  let arow_base = arow * dims.k;
  let sa_base = ar * 16u + aq * 4u;
  // バッチは workgroup 単位で一様（z 軸 1 つが 1 バッチの出力平面 [Cout, Hout·Wout]）
  let xbase = wid.z * dims.channels_in;
  let cbase = wid.z * dims.m * dims.n;
  // B タイルの担当（K 16 行 × 列 quad 16 = 256 要素 / 128 スレッド）
  let bkr = tid / 16u;
  let bcq = tid % 16u;
  let bcol = wid.x * 64u + bcq * 4u;
  // K タイルループ不変（平坦 k を (ic, kh, kw) へ割るための刻み）
  let khw = dims.kernel_h * dims.kernel_w;
  // ループ条件は uniform（dims は uniform バッファ）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすために必要
  let tiles = (dims.k + 15u) / 16u;
  let bias0 = wid.y * 32u + lid.y * 4u;
  var acc = array<vec4<f32>, 4>(
    vec4<f32>(bias[bias0]),
    vec4<f32>(bias[bias0 + 1u]),
    vec4<f32>(bias[bias0 + 2u]),
    vec4<f32>(bias[bias0 + 3u]),
  );
  for (var t = 0u; t < tiles; t = t + 1u) {
    // 範囲外は 0 で埋める。内積に寄与しないので K 端数でも結果は変わらない
    let ak0 = t * 16u + aq * 4u;
    var av = vec4<f32>(0.0);
    if (arow < dims.m) {
      let abase = arow_base + ak0;
      if (ak0 < dims.k) {
        av.x = w[abase];
      }
      if (ak0 + 1u < dims.k) {
        av.y = w[abase + 1u];
      }
      if (ak0 + 2u < dims.k) {
        av.z = w[abase + 2u];
      }
      if (ak0 + 3u < dims.k) {
        av.w = w[abase + 3u];
      }
    }
    sa[sa_base] = av.x;
    sa[sa_base + 1u] = av.y;
    sa[sa_base + 2u] = av.z;
    sa[sa_base + 3u] = av.w;
    // 32 行タイルは 128 スレッドなので B タイル 256 要素を 2 パスで埋める
    for (var bp = 0u; bp < 2u; bp = bp + 1u) {
      let bk = bp * 8u + bkr;
      let brow = t * 16u + bk;
      var bv4 = vec4<f32>(0.0);
      if (brow < dims.k) {
        let ic = brow / khw;
        let kr = brow % khw;
        let ky = i32((kr / dims.kernel_w) * dims.dilation_h) - i32(dims.padding_h);
        let kx = i32((kr % dims.kernel_w) * dims.dilation_w) - i32(dims.padding_w);
        let xc = xbase + ic;
        if (bcol < dims.n) {
          bv4.x = xcol(xc, ky, kx, bcol);
        }
        if (bcol + 1u < dims.n) {
          bv4.y = xcol(xc, ky, kx, bcol + 1u);
        }
        if (bcol + 2u < dims.n) {
          bv4.z = xcol(xc, ky, kx, bcol + 2u);
        }
        if (bcol + 3u < dims.n) {
          bv4.w = xcol(xc, ky, kx, bcol + 3u);
        }
      }
      sb[bk * 16u + bcq] = bv4;
    }
    workgroupBarrier();
    // 共有ロード 5 回（B の vec4 1 + A のスカラ 4）で 16 MAC。縮約は k 昇順の逐次で、
    // 1 出力要素あたりの加算順序は 16×16 の 1 スレッド 1 出力と完全に一致する。
    for (var kk = 0u; kk < 16u; kk = kk + 1u) {
      let bv = sb[kk * 16u + lid.x];
      for (var i = 0u; i < 4u; i = i + 1u) {
        acc[i] = acc[i] + sa[(lid.y * 4u + i) * 16u + kk] * bv;
      }
    }
    workgroupBarrier();
  }
  let ocol = wid.x * 64u + lid.x * 4u;
  let orow0 = wid.y * 32u + lid.y * 4u;
  for (var i = 0u; i < 4u; i = i + 1u) {
    let orow = orow0 + i;
    if (orow < dims.m) {
      let obase = cbase + orow * dims.n;
      if (ocol < dims.n) {
        out[obase + ocol] = acc[i].x;
      }
      if (ocol + 1u < dims.n) {
        out[obase + ocol + 1u] = acc[i].y;
      }
      if (ocol + 2u < dims.n) {
        out[obase + ocol + 2u] = acc[i].z;
      }
      if (ocol + 3u < dims.n) {
        out[obase + ocol + 3u] = acc[i].w;
      }
    }
  }
}
