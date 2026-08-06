// karume linear (x[m,k] · wᵀ[k,n] + bias[n], f32, 重み f16 格納, f16 タイル計算, レジスタ 64x64 タイル)
enable f16;
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

// f16 格納の展開: 要素 i = unpack2x16float(w[i / 2])[i % 2]（平坦添字の偶奇で対を選ぶ）
fn dequant(i: u32) -> f32 {
  let pair = unpack2x16float(w[i >> 1u]);
  return select(pair.x, pair.y, (i & 1u) == 1u);
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
  var acc = array<vec4<f32>, 4>(
    vec4<f32>(0.0),
    vec4<f32>(0.0),
    vec4<f32>(0.0),
    vec4<f32>(0.0),
  );
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
    sa[sa_base] = f16(av.x);
    sa[sa_base + 1u] = f16(av.y);
    sa[sa_base + 2u] = f16(av.z);
    sa[sa_base + 3u] = f16(av.w);
    let wk0 = t * 16u + wq * 4u;
    var wv = vec4<f32>(0.0);
    if (wcol < dims.n) {
      let wbase = wrow_base + wk0;
      if (wk0 < dims.k) {
        wv.x = dequant(wbase);
      }
      if (wk0 + 1u < dims.k) {
        wv.y = dequant(wbase + 1u);
      }
      if (wk0 + 2u < dims.k) {
        wv.z = dequant(wbase + 2u);
      }
      if (wk0 + 3u < dims.k) {
        wv.w = dequant(wbase + 3u);
      }
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
      for (var i = 0u; i < 4u; i = i + 1u) {
        acc[i] = acc[i] + f32(sa[(lid.y * 4u + i) * 16u + kk]) * bv;
      }
    }
    workgroupBarrier();
  }
  let ocol = wid.x * 64u + lid.x * 4u;
  let orow0 = wid.y * 64u + lid.y * 4u;
  for (var i = 0u; i < 4u; i = i + 1u) {
    let orow = orow0 + i;
    if (orow < dims.m) {
      let obase = orow * dims.n;
      if (ocol < dims.n) {
        out[obase + ocol] = acc[i].x + bias[ocol];
      }
      if (ocol + 1u < dims.n) {
        out[obase + ocol + 1u] = acc[i].y + bias[ocol + 1u];
      }
      if (ocol + 2u < dims.n) {
        out[obase + ocol + 2u] = acc[i].z + bias[ocol + 2u];
      }
      if (ocol + 3u < dims.n) {
        out[obase + ocol + 3u] = acc[i].w + bias[ocol + 3u];
      }
    }
  }
}
