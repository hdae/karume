// karume linear (x[m,k] · wᵀ[k,n] + bias[n], f32, 重み f16 格納, f16 タイル計算, レジスタ 64x64 タイル / 1 スレッド 8x4 / wg 16x8 + vec4)
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

@compute @workgroup_size(16, 8)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 16u + lid.x;
  let k4 = dims.k / 4u;
  let n4 = dims.n / 4u;
  // A タイルの担当（64 行 × 4 quad を 128 スレッドで 2 巡）
  let ar = tid / 4u;
  let aq = tid % 4u;
  let arow0 = wid.y * 64u + ar;
  let arow_base0 = arow0 * k4;
  let sa_base0 = ar * 16u + aq * 4u;
  let arow1 = arow0 + 32u;
  let arow_base1 = arow_base0 + 32u * k4;
  let sa_base1 = sa_base0 + 512u;
  // W タイルの担当（64 出力チャネル × 4 quad を 128 スレッドで 2 巡）
  let wc0 = tid / 4u;
  let wq = tid % 4u;
  let wcol0 = wid.x * 64u + wc0;
  let wrow_base0 = wcol0 * dims.k;
  // 共有メモリ側で転置して置く（列 quad = wc / 4・成分 = wc % 4）
  let wsq0 = wc0 / 4u;
  let wsl0 = wc0 % 4u;
  let sb_base0 = (wq * 4u) * 16u + wsq0;
  let wc1 = wc0 + 32u;
  let wcol1 = wcol0 + 32u;
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
    if (arow0 < dims.m && ak0 < dims.k) {
      av0 = x[arow_base0 + t * 4u + aq];
    }
    sa[sa_base0] = f16(av0.x);
    sa[sa_base0 + 1u] = f16(av0.y);
    sa[sa_base0 + 2u] = f16(av0.z);
    sa[sa_base0 + 3u] = f16(av0.w);
    var av1 = vec4<f32>(0.0);
    if (arow1 < dims.m && ak0 < dims.k) {
      av1 = x[arow_base1 + t * 4u + aq];
    }
    sa[sa_base1] = f16(av1.x);
    sa[sa_base1 + 1u] = f16(av1.y);
    sa[sa_base1 + 2u] = f16(av1.z);
    sa[sa_base1 + 3u] = f16(av1.w);
    let wk0 = t * 16u + wq * 4u;
    var wv0 = vec4<f32>(0.0);
    if (wcol0 < dims.n && wk0 < dims.k) {
      wv0 = dequant4(wrow_base0 + wk0);
    }
    switch wsl0 {
      case 0u: {
        sb[sb_base0].x = f16(wv0.x);
        sb[sb_base0 + 16u].x = f16(wv0.y);
        sb[sb_base0 + 32u].x = f16(wv0.z);
        sb[sb_base0 + 48u].x = f16(wv0.w);
      }
      case 1u: {
        sb[sb_base0].y = f16(wv0.x);
        sb[sb_base0 + 16u].y = f16(wv0.y);
        sb[sb_base0 + 32u].y = f16(wv0.z);
        sb[sb_base0 + 48u].y = f16(wv0.w);
      }
      case 2u: {
        sb[sb_base0].z = f16(wv0.x);
        sb[sb_base0 + 16u].z = f16(wv0.y);
        sb[sb_base0 + 32u].z = f16(wv0.z);
        sb[sb_base0 + 48u].z = f16(wv0.w);
      }
      default: {
        sb[sb_base0].w = f16(wv0.x);
        sb[sb_base0 + 16u].w = f16(wv0.y);
        sb[sb_base0 + 32u].w = f16(wv0.z);
        sb[sb_base0 + 48u].w = f16(wv0.w);
      }
    }
    var wv1 = vec4<f32>(0.0);
    if (wcol1 < dims.n && wk0 < dims.k) {
      wv1 = dequant4(wrow_base1 + wk0);
    }
    switch wsl1 {
      case 0u: {
        sb[sb_base1].x = f16(wv1.x);
        sb[sb_base1 + 16u].x = f16(wv1.y);
        sb[sb_base1 + 32u].x = f16(wv1.z);
        sb[sb_base1 + 48u].x = f16(wv1.w);
      }
      case 1u: {
        sb[sb_base1].y = f16(wv1.x);
        sb[sb_base1 + 16u].y = f16(wv1.y);
        sb[sb_base1 + 32u].y = f16(wv1.z);
        sb[sb_base1 + 48u].y = f16(wv1.w);
      }
      case 2u: {
        sb[sb_base1].z = f16(wv1.x);
        sb[sb_base1 + 16u].z = f16(wv1.y);
        sb[sb_base1 + 32u].z = f16(wv1.z);
        sb[sb_base1 + 48u].z = f16(wv1.w);
      }
      default: {
        sb[sb_base1].w = f16(wv1.x);
        sb[sb_base1 + 16u].w = f16(wv1.y);
        sb[sb_base1 + 32u].w = f16(wv1.z);
        sb[sb_base1 + 48u].w = f16(wv1.w);
      }
    }
    workgroupBarrier();
    // 共有ロード 9 回（B の vec4 1 + A のスカラ 8）で 32 MAC。
    // 縮約は k 昇順の逐次で、1 出力要素あたりの加算順序は 16×16 の 1 スレッド 1 出力と
    // 完全に一致する。
    // MUST: f32 への拡幅は**レジスタロード時に 1 回**（9 回 / 32 MAC）。MAC ごとに
    // f32(av * bv) と書くと変換が 32 回に増え、しかも積が f16 精度に落ちる（ADR 0028 の丸め列 2）。
    // f16 → f32 の拡幅は厳密なので、値は「入力を f16 に丸めた f32 変種」と 1 ビットも違わない。
    for (var kk = 0u; kk < 16u; kk = kk + 1u) {
      let bv0 = vec4<f32>(sb[kk * 16u + lid.x]);
      acc0_0 = acc0_0 + f32(sa[(lid.y * 8u + 0u) * 16u + kk]) * bv0;
      acc1_0 = acc1_0 + f32(sa[(lid.y * 8u + 1u) * 16u + kk]) * bv0;
      acc2_0 = acc2_0 + f32(sa[(lid.y * 8u + 2u) * 16u + kk]) * bv0;
      acc3_0 = acc3_0 + f32(sa[(lid.y * 8u + 3u) * 16u + kk]) * bv0;
      acc4_0 = acc4_0 + f32(sa[(lid.y * 8u + 4u) * 16u + kk]) * bv0;
      acc5_0 = acc5_0 + f32(sa[(lid.y * 8u + 5u) * 16u + kk]) * bv0;
      acc6_0 = acc6_0 + f32(sa[(lid.y * 8u + 6u) * 16u + kk]) * bv0;
      acc7_0 = acc7_0 + f32(sa[(lid.y * 8u + 7u) * 16u + kk]) * bv0;
    }
    workgroupBarrier();
  }
  let ocq0 = wid.x * 16u + lid.x;
  let orow0 = wid.y * 64u + lid.y * 8u;
  let orow1 = orow0 + 1u;
  let orow2 = orow0 + 2u;
  let orow3 = orow0 + 3u;
  let orow4 = orow0 + 4u;
  let orow5 = orow0 + 5u;
  let orow6 = orow0 + 6u;
  let orow7 = orow0 + 7u;
  if (ocq0 < n4) {
    // n % 4 == 0 かつ ocq0 < n4 なので oc + 3 < n（bias は常に f32 — ADR 0006）
    let oc = ocq0 * 4u;
    let biasv = vec4<f32>(bias[oc], bias[oc + 1u], bias[oc + 2u], bias[oc + 3u]);
    if (orow0 < dims.m) {
      out[orow0 * n4 + ocq0] = acc0_0 + biasv;
    }
    if (orow1 < dims.m) {
      out[orow1 * n4 + ocq0] = acc1_0 + biasv;
    }
    if (orow2 < dims.m) {
      out[orow2 * n4 + ocq0] = acc2_0 + biasv;
    }
    if (orow3 < dims.m) {
      out[orow3 * n4 + ocq0] = acc3_0 + biasv;
    }
    if (orow4 < dims.m) {
      out[orow4 * n4 + ocq0] = acc4_0 + biasv;
    }
    if (orow5 < dims.m) {
      out[orow5 * n4 + ocq0] = acc5_0 + biasv;
    }
    if (orow6 < dims.m) {
      out[orow6 * n4 + ocq0] = acc6_0 + biasv;
    }
    if (orow7 < dims.m) {
      out[orow7 * n4 + ocq0] = acc7_0 + biasv;
    }
  }
}
