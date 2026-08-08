// karume linear (x[m,k] · wᵀ[k,n] + bias[n], f32, レジスタ 64x64 タイル)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

// 共有 A タイル（64 行 × K 16・スカラ格納）と
// 共有 B タイル（K 16 × 列 quad 16・列方向を vec4 に束ねた形）
var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 16u + lid.x;
  // A タイルの担当（64 行 × 4 quad = 256 スレッド）
  let ar = tid / 4u;
  let aq = tid % 4u;
  let arow = wid.y * 64u + ar;
  let arow_base = arow * dims.k;
  let sa_base = ar * 16u + aq * 4u;
  // W タイルの担当（64 出力チャネル × 4 quad = 256 スレッド）
  let wc = tid / 4u;
  let wq = tid % 4u;
  let wcol = wid.x * 64u + wc;
  let wrow_base = wcol * dims.k;
  // 共有メモリ側で転置して置く（列 quad = wc / 4・成分 = wc % 4）
  let wsq = wc / 4u;
  let wsl = wc % 4u;
  let sb_base = (wq * 4u) * 16u + wsq;
  // ループ条件は uniform（dims は uniform バッファ）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすために必要
  let tiles = (dims.k + 15u) / 16u;
  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);
  for (var t = 0u; t < tiles; t = t + 1u) {
    // 範囲外は 0 で埋める。内積に寄与しないので端数 shape でも結果は変わらない
    let ak0 = t * 16u + aq * 4u;
    var av = vec4<f32>(0.0);
    if (arow < dims.m) {
      if (ak0 < dims.k) {
        av.x = x[arow_base + ak0];
      }
      if (ak0 + 1u < dims.k) {
        av.y = x[arow_base + ak0 + 1u];
      }
      if (ak0 + 2u < dims.k) {
        av.z = x[arow_base + ak0 + 2u];
      }
      if (ak0 + 3u < dims.k) {
        av.w = x[arow_base + ak0 + 3u];
      }
    }
    sa[sa_base] = av.x;
    sa[sa_base + 1u] = av.y;
    sa[sa_base + 2u] = av.z;
    sa[sa_base + 3u] = av.w;
    let wk0 = t * 16u + wq * 4u;
    var wv = vec4<f32>(0.0);
    if (wcol < dims.n) {
      let wbase = wrow_base + wk0;
      if (wk0 < dims.k) {
        wv.x = w[wbase];
      }
      if (wk0 + 1u < dims.k) {
        wv.y = w[wbase + 1u];
      }
      if (wk0 + 2u < dims.k) {
        wv.z = w[wbase + 2u];
      }
      if (wk0 + 3u < dims.k) {
        wv.w = w[wbase + 3u];
      }
    }
    switch wsl {
      case 0u: {
        sb[sb_base].x = wv.x;
        sb[sb_base + 16u].x = wv.y;
        sb[sb_base + 32u].x = wv.z;
        sb[sb_base + 48u].x = wv.w;
      }
      case 1u: {
        sb[sb_base].y = wv.x;
        sb[sb_base + 16u].y = wv.y;
        sb[sb_base + 32u].y = wv.z;
        sb[sb_base + 48u].y = wv.w;
      }
      case 2u: {
        sb[sb_base].z = wv.x;
        sb[sb_base + 16u].z = wv.y;
        sb[sb_base + 32u].z = wv.z;
        sb[sb_base + 48u].z = wv.w;
      }
      default: {
        sb[sb_base].w = wv.x;
        sb[sb_base + 16u].w = wv.y;
        sb[sb_base + 32u].w = wv.z;
        sb[sb_base + 48u].w = wv.w;
      }
    }
    workgroupBarrier();
    // 共有ロード 5 回（B の vec4 1 + A のスカラ 4）で 16 MAC。縮約は k 昇順の逐次で、
    // 1 出力要素あたりの加算順序は 16×16 の 1 スレッド 1 出力と完全に一致する。
    for (var kk = 0u; kk < 16u; kk = kk + 1u) {
      let bv = sb[kk * 16u + lid.x];
      acc0 = acc0 + sa[(lid.y * 4u + 0u) * 16u + kk] * bv;
      acc1 = acc1 + sa[(lid.y * 4u + 1u) * 16u + kk] * bv;
      acc2 = acc2 + sa[(lid.y * 4u + 2u) * 16u + kk] * bv;
      acc3 = acc3 + sa[(lid.y * 4u + 3u) * 16u + kk] * bv;
    }
    workgroupBarrier();
  }
  let ocol = wid.x * 64u + lid.x * 4u;
  let orow0 = wid.y * 64u + lid.y * 4u;
  if (orow0 < dims.m) {
    let obase = orow0 * dims.n;
    if (ocol < dims.n) {
      out[obase + ocol] = acc0.x + bias[ocol];
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = acc0.y + bias[ocol + 1u];
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = acc0.z + bias[ocol + 2u];
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = acc0.w + bias[ocol + 3u];
    }
  }
  let orow1 = orow0 + 1u;
  if (orow1 < dims.m) {
    let obase = orow1 * dims.n;
    if (ocol < dims.n) {
      out[obase + ocol] = acc1.x + bias[ocol];
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = acc1.y + bias[ocol + 1u];
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = acc1.z + bias[ocol + 2u];
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = acc1.w + bias[ocol + 3u];
    }
  }
  let orow2 = orow0 + 2u;
  if (orow2 < dims.m) {
    let obase = orow2 * dims.n;
    if (ocol < dims.n) {
      out[obase + ocol] = acc2.x + bias[ocol];
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = acc2.y + bias[ocol + 1u];
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = acc2.z + bias[ocol + 2u];
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = acc2.w + bias[ocol + 3u];
    }
  }
  let orow3 = orow0 + 3u;
  if (orow3 < dims.m) {
    let obase = orow3 * dims.n;
    if (ocol < dims.n) {
      out[obase + ocol] = acc3.x + bias[ocol];
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = acc3.y + bias[ocol + 1u];
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = acc3.z + bias[ocol + 2u];
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = acc3.w + bias[ocol + 3u];
    }
  }
}
