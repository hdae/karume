// karume linear (x[m,k] · wᵀ[k,n] + bias[n], f32, 重み f16 格納, f16 タイル計算, レジスタ 64x64 タイル + vec4)
enable f16;
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<vec4<f32>>;

// f16 格納の quad 展開: 要素 i..i+3 = w[i / 2] と w[i / 2 + 1] の 2 語
fn dequant4(i: u32) -> vec4<f32> {
  let lo = unpack2x16float(w[i >> 1u]);
  let hi = unpack2x16float(w[(i >> 1u) + 1u]);
  return vec4<f32>(lo.x, lo.y, hi.x, hi.y);
}

// 共有 A タイル（64 行 × K 16・スカラ格納）と
// 共有 B タイル（K 16 × 列 quad 16・列方向を vec4 に束ねた形）。f16 変種は共有バイトが半分
// （8192 B → 4096 B / WG）— 期待利得はこの 1 機序に全て乗る
var<workgroup> sa: array<f16, 1024>;
var<workgroup> sb: array<vec4<f16>, 256>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 16u + lid.x;
  let k4 = dims.k / 4u;
  let n4 = dims.n / 4u;
  // A タイルの担当（64 行 × 4 quad = 256 スレッド）
  let ar = tid / 4u;
  let aq = tid % 4u;
  let arow = wid.y * 64u + ar;
  let arow_base = arow * k4;
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
    if (arow < dims.m && ak0 < dims.k) {
      av = x[arow_base + t * 4u + aq];
    }
    sa[sa_base] = f16(av.x);
    sa[sa_base + 1u] = f16(av.y);
    sa[sa_base + 2u] = f16(av.z);
    sa[sa_base + 3u] = f16(av.w);
    let wk0 = t * 16u + wq * 4u;
    var wv = vec4<f32>(0.0);
    if (wcol < dims.n && wk0 < dims.k) {
      wv = dequant4(wrow_base + wk0);
    }
    switch wsl {
      case 0u: {
        sb[sb_base].x = f16(wv.x);
        sb[sb_base + 16u].x = f16(wv.y);
        sb[sb_base + 32u].x = f16(wv.z);
        sb[sb_base + 48u].x = f16(wv.w);
      }
      case 1u: {
        sb[sb_base].y = f16(wv.x);
        sb[sb_base + 16u].y = f16(wv.y);
        sb[sb_base + 32u].y = f16(wv.z);
        sb[sb_base + 48u].y = f16(wv.w);
      }
      case 2u: {
        sb[sb_base].z = f16(wv.x);
        sb[sb_base + 16u].z = f16(wv.y);
        sb[sb_base + 32u].z = f16(wv.z);
        sb[sb_base + 48u].z = f16(wv.w);
      }
      default: {
        sb[sb_base].w = f16(wv.x);
        sb[sb_base + 16u].w = f16(wv.y);
        sb[sb_base + 32u].w = f16(wv.z);
        sb[sb_base + 48u].w = f16(wv.w);
      }
    }
    workgroupBarrier();
    // 共有ロード 5 回（B の vec4 1 + A のスカラ 4）で 16 MAC。縮約は k 昇順の逐次で、
    // 1 出力要素あたりの加算順序は 16×16 の 1 スレッド 1 出力と完全に一致する。
    // MUST: f32 への拡幅は**レジスタロード時に 1 回**（8 回 / 16 MAC）。MAC ごとに
    // f32(av * bv) と書くと変換が 16 回に増え、しかも積が f16 精度に落ちる（ADR 0028 の丸め列 2）。
    // f16 → f32 の拡幅は厳密なので、値は「入力を f16 に丸めた f32 変種」と 1 ビットも違わない。
    for (var kk = 0u; kk < 16u; kk = kk + 1u) {
      let bv = vec4<f32>(sb[kk * 16u + lid.x]);
      acc0 = acc0 + f32(sa[(lid.y * 4u + 0u) * 16u + kk]) * bv;
      acc1 = acc1 + f32(sa[(lid.y * 4u + 1u) * 16u + kk]) * bv;
      acc2 = acc2 + f32(sa[(lid.y * 4u + 2u) * 16u + kk]) * bv;
      acc3 = acc3 + f32(sa[(lid.y * 4u + 3u) * 16u + kk]) * bv;
    }
    workgroupBarrier();
  }
  let ocq = wid.x * 16u + lid.x;
  let orow0 = wid.y * 64u + lid.y * 4u;
  if (ocq < n4) {
    // n % 4 == 0 かつ ocq < n4 なので oc + 3 < n（bias は常に f32 — ADR 0006）
    let oc = ocq * 4u;
    let biasv = vec4<f32>(bias[oc], bias[oc + 1u], bias[oc + 2u], bias[oc + 3u]);
    if (orow0 < dims.m) {
      out[orow0 * n4 + ocq] = acc0 + biasv;
    }
    let orow1 = orow0 + 1u;
    if (orow1 < dims.m) {
      out[orow1 * n4 + ocq] = acc1 + biasv;
    }
    let orow2 = orow0 + 2u;
    if (orow2 < dims.m) {
      out[orow2 * n4 + ocq] = acc2 + biasv;
    }
    let orow3 = orow0 + 3u;
    if (orow3 < dims.m) {
      out[orow3 * n4 + ocq] = acc3 + biasv;
    }
  }
}
