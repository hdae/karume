// karume conv2d (x[B,Cin,H,W] * W[Cout,Cin,Kh,Kw] + b[Cout], f32, 重み f16 格納, implicit GEMM レジスタ 32x64 タイル + vec4)
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
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<vec4<f32>>;

// f16 格納の quad 展開: 要素 i..i+3 = w[i / 2] と w[i / 2 + 1] の 2 語
fn dequant4(i: u32) -> vec4<f32> {
  let lo = unpack2x16float(w[i >> 1u]);
  let hi = unpack2x16float(w[(i >> 1u) + 1u]);
  return vec4<f32>(lo.x, lo.y, hi.x, hi.y);
}

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
  let n4 = dims.n / 4u;
  // A タイルの担当（32 行 × 4 quad = 128 スレッド）
  let ar = tid / 4u;
  let aq = tid % 4u;
  let arow = wid.y * 32u + ar;
  let arow_base = arow * dims.k;
  let sa_base = ar * 16u + aq * 4u;
  // バッチは workgroup 単位で一様（z 軸 1 つが 1 バッチの出力平面 [Cout, Hout·Wout]）
  let xbase = wid.z * dims.channels_in;
  let cbase = wid.z * dims.m * n4;
  // B タイルの担当（K 16 行 × 列 quad 16 = 256 要素 / 128 スレッド）
  let bkr = tid / 16u;
  let bcq = tid % 16u;
  let bc4 = wid.x * 16u + bcq;
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
    if (arow < dims.m && ak0 < dims.k) {
      av = dequant4(arow_base + ak0);
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
      if (brow < dims.k && bc4 < n4) {
        let ic = brow / khw;
        let kr = brow % khw;
        let ky = i32((kr / dims.kernel_w) * dims.dilation_h) - i32(dims.padding_h);
        let kx = i32((kr % dims.kernel_w) * dims.dilation_w) - i32(dims.padding_w);
        let xc = xbase + ic;
        // quad の 4 列は同じ出力行の連続 ox（v4 の条件）なので x 側も連続に読める
        let n0 = bc4 * 4u;
        let iy = i32((n0 / dims.width_out) * dims.stride_h) + ky;
        let ix0 = i32((n0 % dims.width_out) * dims.stride_w) + kx;
        if (iy >= 0 && u32(iy) < dims.height_in && ix0 >= 0 && u32(ix0) + 3u < dims.width_in) {
          let base = (xc * dims.height_in + u32(iy)) * dims.width_in + u32(ix0);
          bv4 = vec4<f32>(x[base], x[base + 1u], x[base + 2u], x[base + 3u]);
        } else {
          // 画像端と padding 域だけがここに来る（範囲外は 0 — xcol の MUST）
          bv4 = vec4<f32>(
            xcol(xc, ky, kx, n0),
            xcol(xc, ky, kx, n0 + 1u),
            xcol(xc, ky, kx, n0 + 2u),
            xcol(xc, ky, kx, n0 + 3u),
          );
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
  let ocq = wid.x * 16u + lid.x;
  let orow0 = wid.y * 32u + lid.y * 4u;
  if (ocq < n4) {
    for (var i = 0u; i < 4u; i = i + 1u) {
      let orow = orow0 + i;
      if (orow < dims.m) {
        out[cbase + orow * n4 + ocq] = acc[i];
      }
    }
  }
}
