// karume linear (x[m,k] · wᵀ[k,n] + bias[n], f32, 重み i4 格納, レジスタ 128x128 タイル / 1 スレッド 8x8 / wg 16x16)
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

// i4 格納の展開: 要素 i = f32(i32(nibble) − 8) · scale
// （1 語 = 8 要素。平坦添字で語 i/8・バイト (i/2)%4・nibble i%2 を割る — ADR 0069）
fn dequant(i: u32, scale: f32) -> f32 {
  let byte = unpack4xU8(w[i >> 3u])[(i >> 1u) & 3u];
  let nibble = select(byte & 0xFu, byte >> 4u, (i & 1u) == 1u);
  return f32(i32(nibble) - 8) * scale;
}

// 共有 A タイル（128 行 × K 16・スカラ格納）と
// 共有 B タイル（K 16 × 列 quad 32・列方向を vec4 に束ねた形）
var<workgroup> sa: array<f32, 2048>;
var<workgroup> sb: array<vec4<f32>, 512>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 16u + lid.x;
  // A タイルの担当（128 行 × 4 quad を 256 スレッドで 2 巡）
  let ar = tid / 4u;
  let aq = tid % 4u;
  let arow0 = wid.y * 128u + ar;
  let arow_base0 = arow0 * dims.k;
  let sa_base0 = ar * 16u + aq * 4u;
  let arow1 = arow0 + 64u;
  let arow_base1 = arow_base0 + 64u * dims.k;
  let sa_base1 = sa_base0 + 1024u;
  // W タイルの担当（128 出力チャネル × 4 quad を 256 スレッドで 2 巡）
  let wc0 = tid / 4u;
  let wq = tid % 4u;
  let wcol0 = wid.x * 128u + wc0;
  let wrow_base0 = wcol0 * dims.k;
  // 共有メモリ側で転置して置く（列 quad = wc / 4・成分 = wc % 4）
  let wsq0 = wc0 / 4u;
  let wsl0 = wc0 % 4u;
  let sb_base0 = (wq * 4u) * 32u + wsq0;
  let wc1 = wc0 + 64u;
  let wcol1 = wcol0 + 64u;
  let wrow_base1 = wcol1 * dims.k;
  let wsq1 = wc1 / 4u;
  let wsl1 = wc1 % 4u;
  let sb_base1 = (wq * 4u) * 32u + wsq1;
  // ループ条件は uniform（dims は uniform バッファ）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすために必要
  let tiles = (dims.k + 15u) / 16u;
  var acc0_0 = vec4<f32>(0.0);
  var acc0_1 = vec4<f32>(0.0);
  var acc1_0 = vec4<f32>(0.0);
  var acc1_1 = vec4<f32>(0.0);
  var acc2_0 = vec4<f32>(0.0);
  var acc2_1 = vec4<f32>(0.0);
  var acc3_0 = vec4<f32>(0.0);
  var acc3_1 = vec4<f32>(0.0);
  var acc4_0 = vec4<f32>(0.0);
  var acc4_1 = vec4<f32>(0.0);
  var acc5_0 = vec4<f32>(0.0);
  var acc5_1 = vec4<f32>(0.0);
  var acc6_0 = vec4<f32>(0.0);
  var acc6_1 = vec4<f32>(0.0);
  var acc7_0 = vec4<f32>(0.0);
  var acc7_1 = vec4<f32>(0.0);
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
      let wgs0 = wscale[wcol0 * (dims.k >> 5u) + (wk0 >> 5u)];
      if (wk0 < dims.k) {
        wv0.x = dequant(wbase, wgs0);
      }
      if (wk0 + 1u < dims.k) {
        wv0.y = dequant(wbase + 1u, wgs0);
      }
      if (wk0 + 2u < dims.k) {
        wv0.z = dequant(wbase + 2u, wgs0);
      }
      if (wk0 + 3u < dims.k) {
        wv0.w = dequant(wbase + 3u, wgs0);
      }
    }
    switch wsl0 {
      case 0u: {
        sb[sb_base0].x = wv0.x;
        sb[sb_base0 + 32u].x = wv0.y;
        sb[sb_base0 + 64u].x = wv0.z;
        sb[sb_base0 + 96u].x = wv0.w;
      }
      case 1u: {
        sb[sb_base0].y = wv0.x;
        sb[sb_base0 + 32u].y = wv0.y;
        sb[sb_base0 + 64u].y = wv0.z;
        sb[sb_base0 + 96u].y = wv0.w;
      }
      case 2u: {
        sb[sb_base0].z = wv0.x;
        sb[sb_base0 + 32u].z = wv0.y;
        sb[sb_base0 + 64u].z = wv0.z;
        sb[sb_base0 + 96u].z = wv0.w;
      }
      default: {
        sb[sb_base0].w = wv0.x;
        sb[sb_base0 + 32u].w = wv0.y;
        sb[sb_base0 + 64u].w = wv0.z;
        sb[sb_base0 + 96u].w = wv0.w;
      }
    }
    var wv1 = vec4<f32>(0.0);
    if (wcol1 < dims.n) {
      let wbase = wrow_base1 + wk0;
      let wgs1 = wscale[wcol1 * (dims.k >> 5u) + (wk0 >> 5u)];
      if (wk0 < dims.k) {
        wv1.x = dequant(wbase, wgs1);
      }
      if (wk0 + 1u < dims.k) {
        wv1.y = dequant(wbase + 1u, wgs1);
      }
      if (wk0 + 2u < dims.k) {
        wv1.z = dequant(wbase + 2u, wgs1);
      }
      if (wk0 + 3u < dims.k) {
        wv1.w = dequant(wbase + 3u, wgs1);
      }
    }
    switch wsl1 {
      case 0u: {
        sb[sb_base1].x = wv1.x;
        sb[sb_base1 + 32u].x = wv1.y;
        sb[sb_base1 + 64u].x = wv1.z;
        sb[sb_base1 + 96u].x = wv1.w;
      }
      case 1u: {
        sb[sb_base1].y = wv1.x;
        sb[sb_base1 + 32u].y = wv1.y;
        sb[sb_base1 + 64u].y = wv1.z;
        sb[sb_base1 + 96u].y = wv1.w;
      }
      case 2u: {
        sb[sb_base1].z = wv1.x;
        sb[sb_base1 + 32u].z = wv1.y;
        sb[sb_base1 + 64u].z = wv1.z;
        sb[sb_base1 + 96u].z = wv1.w;
      }
      default: {
        sb[sb_base1].w = wv1.x;
        sb[sb_base1 + 32u].w = wv1.y;
        sb[sb_base1 + 64u].w = wv1.z;
        sb[sb_base1 + 96u].w = wv1.w;
      }
    }
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
  let orow0 = wid.y * 128u + lid.y * 8u;
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
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = acc0_1.x + bias[ocol + 4u];
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = acc0_1.y + bias[ocol + 5u];
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = acc0_1.z + bias[ocol + 6u];
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = acc0_1.w + bias[ocol + 7u];
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
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = acc1_1.x + bias[ocol + 4u];
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = acc1_1.y + bias[ocol + 5u];
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = acc1_1.z + bias[ocol + 6u];
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = acc1_1.w + bias[ocol + 7u];
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
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = acc2_1.x + bias[ocol + 4u];
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = acc2_1.y + bias[ocol + 5u];
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = acc2_1.z + bias[ocol + 6u];
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = acc2_1.w + bias[ocol + 7u];
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
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = acc3_1.x + bias[ocol + 4u];
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = acc3_1.y + bias[ocol + 5u];
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = acc3_1.z + bias[ocol + 6u];
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = acc3_1.w + bias[ocol + 7u];
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
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = acc4_1.x + bias[ocol + 4u];
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = acc4_1.y + bias[ocol + 5u];
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = acc4_1.z + bias[ocol + 6u];
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = acc4_1.w + bias[ocol + 7u];
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
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = acc5_1.x + bias[ocol + 4u];
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = acc5_1.y + bias[ocol + 5u];
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = acc5_1.z + bias[ocol + 6u];
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = acc5_1.w + bias[ocol + 7u];
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
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = acc6_1.x + bias[ocol + 4u];
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = acc6_1.y + bias[ocol + 5u];
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = acc6_1.z + bias[ocol + 6u];
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = acc6_1.w + bias[ocol + 7u];
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
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = acc7_1.x + bias[ocol + 4u];
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = acc7_1.y + bias[ocol + 5u];
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = acc7_1.z + bias[ocol + 6u];
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = acc7_1.w + bias[ocol + 7u];
    }
  }
}
