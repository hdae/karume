// karume linear (x[m,k] · wᵀ[k,n] + bias[n], f32, 重み i8 格納, レジスタ 64x64 タイル / 1 スレッド 8x4 / wg 16x8)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@group(0) @binding(5) var<storage, read> wscale: array<f32>;

// i8 格納の展開: 要素 i = f32(unpack4xI8(w[i / 4])[i % 4]) · scale
// （平坦添字で語と位置を割る。scale は出力チャネルごと — ADR 0019）
fn dequant(i: u32, scale: f32) -> f32 {
  return f32(unpack4xI8(w[i >> 2u])[i & 3u]) * scale;
}

// 共有 A タイル（64 行 × K 16・スカラ格納）と
// 共有 B タイル（K 16 × 列 quad 16・列方向を vec4 に束ねた形）
var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 8)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 16u + lid.x;
  // A タイルの担当（64 行 × 4 quad を 128 スレッドで 2 巡）
  let ar = tid / 4u;
  let aq = tid % 4u;
  let arow0 = wid.y * 64u + ar;
  let arow_base0 = arow0 * dims.k;
  let sa_base0 = ar * 16u + aq * 4u;
  let arow1 = arow0 + 32u;
  let arow_base1 = arow_base0 + 32u * dims.k;
  let sa_base1 = sa_base0 + 512u;
  // W タイルの担当（64 出力チャネル × 4 quad を 128 スレッドで 2 巡）
  let wc0 = tid / 4u;
  let wq = tid % 4u;
  let wcol0 = wid.x * 64u + wc0;
  // 出力チャネルの scale はループ不変 — 重みの要素ごとに引き直さない（ADR 0019）
  let wscale_v = wscale[wcol0];
  let wrow_base0 = wcol0 * dims.k;
  // 共有メモリ側で転置して置く（列 quad = wc / 4・成分 = wc % 4）
  let wsq0 = wc0 / 4u;
  let wsl0 = wc0 % 4u;
  let sb_base0 = (wq * 4u) * 16u + wsq0;
  let wc1 = wc0 + 32u;
  let wcol1 = wcol0 + 32u;
  // 出力チャネルの scale はループ不変 — 重みの要素ごとに引き直さない（ADR 0019）
  let wscale_v1 = wscale[wcol1];
  let wrow_base1 = wcol1 * dims.k;
  let wsq1 = wc1 / 4u;
  let wsl1 = wc1 % 4u;
  let sb_base1 = (wq * 4u) * 16u + wsq1;
  // ループ条件は uniform（dims は uniform バッファ）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすために必要
  let tiles = (dims.k + 15u) / 16u;
  var acc0_0 = vec4<f32>(0.0);
  var acc1_0 = vec4<f32>(0.0);
  var acc2_0 = vec4<f32>(0.0);
  var acc3_0 = vec4<f32>(0.0);
  var acc4_0 = vec4<f32>(0.0);
  var acc5_0 = vec4<f32>(0.0);
  var acc6_0 = vec4<f32>(0.0);
  var acc7_0 = vec4<f32>(0.0);
  for (var t = 0u; t < tiles; t = t + 1u) {
    // 範囲外は 0 で埋める。内積に寄与しないので端数 shape でも結果は変わらない
    let ak0 = t * 16u + aq * 4u;
    var av0 = vec4<f32>(0.0);
    if (arow0 < dims.m) {
      if (ak0 < dims.k) {
        av0.x = x[arow_base0 + ak0];
      }
      if (ak0 + 1u < dims.k) {
        av0.y = x[arow_base0 + ak0 + 1u];
      }
      if (ak0 + 2u < dims.k) {
        av0.z = x[arow_base0 + ak0 + 2u];
      }
      if (ak0 + 3u < dims.k) {
        av0.w = x[arow_base0 + ak0 + 3u];
      }
    }
    sa[sa_base0] = av0.x;
    sa[sa_base0 + 1u] = av0.y;
    sa[sa_base0 + 2u] = av0.z;
    sa[sa_base0 + 3u] = av0.w;
    var av1 = vec4<f32>(0.0);
    if (arow1 < dims.m) {
      if (ak0 < dims.k) {
        av1.x = x[arow_base1 + ak0];
      }
      if (ak0 + 1u < dims.k) {
        av1.y = x[arow_base1 + ak0 + 1u];
      }
      if (ak0 + 2u < dims.k) {
        av1.z = x[arow_base1 + ak0 + 2u];
      }
      if (ak0 + 3u < dims.k) {
        av1.w = x[arow_base1 + ak0 + 3u];
      }
    }
    sa[sa_base1] = av1.x;
    sa[sa_base1 + 1u] = av1.y;
    sa[sa_base1 + 2u] = av1.z;
    sa[sa_base1 + 3u] = av1.w;
    let wk0 = t * 16u + wq * 4u;
    var wv0 = vec4<f32>(0.0);
    if (wcol0 < dims.n) {
      let wbase = wrow_base0 + wk0;
      if (wk0 < dims.k) {
        wv0.x = dequant(wbase, wscale_v);
      }
      if (wk0 + 1u < dims.k) {
        wv0.y = dequant(wbase + 1u, wscale_v);
      }
      if (wk0 + 2u < dims.k) {
        wv0.z = dequant(wbase + 2u, wscale_v);
      }
      if (wk0 + 3u < dims.k) {
        wv0.w = dequant(wbase + 3u, wscale_v);
      }
    }
    switch wsl0 {
      case 0u: {
        sb[sb_base0].x = wv0.x;
        sb[sb_base0 + 16u].x = wv0.y;
        sb[sb_base0 + 32u].x = wv0.z;
        sb[sb_base0 + 48u].x = wv0.w;
      }
      case 1u: {
        sb[sb_base0].y = wv0.x;
        sb[sb_base0 + 16u].y = wv0.y;
        sb[sb_base0 + 32u].y = wv0.z;
        sb[sb_base0 + 48u].y = wv0.w;
      }
      case 2u: {
        sb[sb_base0].z = wv0.x;
        sb[sb_base0 + 16u].z = wv0.y;
        sb[sb_base0 + 32u].z = wv0.z;
        sb[sb_base0 + 48u].z = wv0.w;
      }
      default: {
        sb[sb_base0].w = wv0.x;
        sb[sb_base0 + 16u].w = wv0.y;
        sb[sb_base0 + 32u].w = wv0.z;
        sb[sb_base0 + 48u].w = wv0.w;
      }
    }
    var wv1 = vec4<f32>(0.0);
    if (wcol1 < dims.n) {
      let wbase = wrow_base1 + wk0;
      if (wk0 < dims.k) {
        wv1.x = dequant(wbase, wscale_v1);
      }
      if (wk0 + 1u < dims.k) {
        wv1.y = dequant(wbase + 1u, wscale_v1);
      }
      if (wk0 + 2u < dims.k) {
        wv1.z = dequant(wbase + 2u, wscale_v1);
      }
      if (wk0 + 3u < dims.k) {
        wv1.w = dequant(wbase + 3u, wscale_v1);
      }
    }
    switch wsl1 {
      case 0u: {
        sb[sb_base1].x = wv1.x;
        sb[sb_base1 + 16u].x = wv1.y;
        sb[sb_base1 + 32u].x = wv1.z;
        sb[sb_base1 + 48u].x = wv1.w;
      }
      case 1u: {
        sb[sb_base1].y = wv1.x;
        sb[sb_base1 + 16u].y = wv1.y;
        sb[sb_base1 + 32u].y = wv1.z;
        sb[sb_base1 + 48u].y = wv1.w;
      }
      case 2u: {
        sb[sb_base1].z = wv1.x;
        sb[sb_base1 + 16u].z = wv1.y;
        sb[sb_base1 + 32u].z = wv1.z;
        sb[sb_base1 + 48u].z = wv1.w;
      }
      default: {
        sb[sb_base1].w = wv1.x;
        sb[sb_base1 + 16u].w = wv1.y;
        sb[sb_base1 + 32u].w = wv1.z;
        sb[sb_base1 + 48u].w = wv1.w;
      }
    }
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
  let orow0 = wid.y * 64u + lid.y * 8u;
  let orow1 = orow0 + 1u;
  let orow2 = orow0 + 2u;
  let orow3 = orow0 + 3u;
  let orow4 = orow0 + 4u;
  let orow5 = orow0 + 5u;
  let orow6 = orow0 + 6u;
  let orow7 = orow0 + 7u;
  if (orow0 < dims.m) {
    let obase = orow0 * dims.n;
    if (ocol < dims.n) {
      out[obase + ocol] = acc0_0.x + bias[ocol];
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = acc0_0.y + bias[ocol + 1u];
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = acc0_0.z + bias[ocol + 2u];
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = acc0_0.w + bias[ocol + 3u];
    }
  }
  if (orow1 < dims.m) {
    let obase = orow1 * dims.n;
    if (ocol < dims.n) {
      out[obase + ocol] = acc1_0.x + bias[ocol];
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = acc1_0.y + bias[ocol + 1u];
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = acc1_0.z + bias[ocol + 2u];
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = acc1_0.w + bias[ocol + 3u];
    }
  }
  if (orow2 < dims.m) {
    let obase = orow2 * dims.n;
    if (ocol < dims.n) {
      out[obase + ocol] = acc2_0.x + bias[ocol];
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = acc2_0.y + bias[ocol + 1u];
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = acc2_0.z + bias[ocol + 2u];
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = acc2_0.w + bias[ocol + 3u];
    }
  }
  if (orow3 < dims.m) {
    let obase = orow3 * dims.n;
    if (ocol < dims.n) {
      out[obase + ocol] = acc3_0.x + bias[ocol];
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = acc3_0.y + bias[ocol + 1u];
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = acc3_0.z + bias[ocol + 2u];
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = acc3_0.w + bias[ocol + 3u];
    }
  }
  if (orow4 < dims.m) {
    let obase = orow4 * dims.n;
    if (ocol < dims.n) {
      out[obase + ocol] = acc4_0.x + bias[ocol];
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = acc4_0.y + bias[ocol + 1u];
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = acc4_0.z + bias[ocol + 2u];
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = acc4_0.w + bias[ocol + 3u];
    }
  }
  if (orow5 < dims.m) {
    let obase = orow5 * dims.n;
    if (ocol < dims.n) {
      out[obase + ocol] = acc5_0.x + bias[ocol];
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = acc5_0.y + bias[ocol + 1u];
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = acc5_0.z + bias[ocol + 2u];
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = acc5_0.w + bias[ocol + 3u];
    }
  }
  if (orow6 < dims.m) {
    let obase = orow6 * dims.n;
    if (ocol < dims.n) {
      out[obase + ocol] = acc6_0.x + bias[ocol];
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = acc6_0.y + bias[ocol + 1u];
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = acc6_0.z + bias[ocol + 2u];
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = acc6_0.w + bias[ocol + 3u];
    }
  }
  if (orow7 < dims.m) {
    let obase = orow7 * dims.n;
    if (ocol < dims.n) {
      out[obase + ocol] = acc7_0.x + bias[ocol];
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = acc7_0.y + bias[ocol + 1u];
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = acc7_0.z + bias[ocol + 2u];
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = acc7_0.w + bias[ocol + 3u];
    }
  }
}
