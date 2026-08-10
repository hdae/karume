// karume attention_qk (S[b,m,n] = (q·scale)[b,m,d] · (k·scale)[b,n,d]ᵀ, f32, f16 タイル計算・S も f16, レジスタ 128x128 タイル / 1 スレッド 8x8 / wg 16x16)
enable f16;
struct Dims {
  m: u32,
  n: u32,
  k: u32,
  scale: f32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read_write> s: array<f16>;

// 共有 A タイル（128 行 × K 16・スカラ格納）と
// 共有 B タイル（K 16 × 列 quad 32・列方向を vec4 に束ねた形）。f16 変種は共有バイトが半分
// （16384 B → 8192 B / WG）— 期待利得はこの 1 機序に全て乗る
var<workgroup> sa: array<f16, 2048>;
var<workgroup> sb: array<vec4<f16>, 512>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 16u + lid.x;
  // バッチは workgroup 単位で一様（z 軸 1 つが 1 バッチのタイル）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすための前提
  let abase = wid.z * dims.m * dims.k;
  let bbase = wid.z * dims.n * dims.k;
  let cbase = wid.z * dims.m * dims.n;
  // A タイルの担当（128 行 × 4 quad を 256 スレッドで 2 巡）
  let ar = tid / 4u;
  let aq = tid % 4u;
  let arow0 = wid.y * 128u + ar;
  let arow_base0 = abase + arow0 * dims.k;
  let sa_base0 = ar * 16u + aq * 4u;
  let arow1 = arow0 + 64u;
  let arow_base1 = arow_base0 + 64u * dims.k;
  let sa_base1 = sa_base0 + 1024u;
  // k タイルの担当（128 列（N）× 4 quad を 256 スレッドで 2 巡）。
  // k は [N,D] のまま読み、**共有メモリ側で転置して置く**（linear の重み読みと同じ構造）
  let wc0 = tid / 4u;
  let wq = tid % 4u;
  let wcol0 = wid.x * 128u + wc0;
  let krow_base0 = bbase + wcol0 * dims.k;
  let wsq0 = wc0 / 4u;
  let wsl0 = wc0 % 4u;
  let sb_base0 = (wq * 4u) * 32u + wsq0;
  let wc1 = wc0 + 64u;
  let wcol1 = wcol0 + 64u;
  let krow_base1 = krow_base0 + 64u * dims.k;
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
        av0.x = q[arow_base0 + ak0] * dims.scale;
      }
      if (ak0 + 1u < dims.k) {
        av0.y = q[arow_base0 + ak0 + 1u] * dims.scale;
      }
      if (ak0 + 2u < dims.k) {
        av0.z = q[arow_base0 + ak0 + 2u] * dims.scale;
      }
      if (ak0 + 3u < dims.k) {
        av0.w = q[arow_base0 + ak0 + 3u] * dims.scale;
      }
    }
    sa[sa_base0] = f16(av0.x);
    sa[sa_base0 + 1u] = f16(av0.y);
    sa[sa_base0 + 2u] = f16(av0.z);
    sa[sa_base0 + 3u] = f16(av0.w);
    var av1 = vec4<f32>(0.0);
    if (arow1 < dims.m) {
      if (ak0 < dims.k) {
        av1.x = q[arow_base1 + ak0] * dims.scale;
      }
      if (ak0 + 1u < dims.k) {
        av1.y = q[arow_base1 + ak0 + 1u] * dims.scale;
      }
      if (ak0 + 2u < dims.k) {
        av1.z = q[arow_base1 + ak0 + 2u] * dims.scale;
      }
      if (ak0 + 3u < dims.k) {
        av1.w = q[arow_base1 + ak0 + 3u] * dims.scale;
      }
    }
    sa[sa_base1] = f16(av1.x);
    sa[sa_base1 + 1u] = f16(av1.y);
    sa[sa_base1 + 2u] = f16(av1.z);
    sa[sa_base1 + 3u] = f16(av1.w);
    let wk0 = t * 16u + wq * 4u;
    var wv0 = vec4<f32>(0.0);
    if (wcol0 < dims.n) {
      if (wk0 < dims.k) {
        wv0.x = k[krow_base0 + wk0];
      }
      if (wk0 + 1u < dims.k) {
        wv0.y = k[krow_base0 + wk0 + 1u];
      }
      if (wk0 + 2u < dims.k) {
        wv0.z = k[krow_base0 + wk0 + 2u];
      }
      if (wk0 + 3u < dims.k) {
        wv0.w = k[krow_base0 + wk0 + 3u];
      }
    }
    // 半スケール契約（ADR 0023）: scale は q 側と k 側の**両方**へ掛ける。範囲外は 0 のままで
    // 0 · scale = 0 なので端数タイルの結論は変わらない
    wv0 = wv0 * dims.scale;
    switch wsl0 {
      case 0u: {
        sb[sb_base0].x = f16(wv0.x);
        sb[sb_base0 + 32u].x = f16(wv0.y);
        sb[sb_base0 + 64u].x = f16(wv0.z);
        sb[sb_base0 + 96u].x = f16(wv0.w);
      }
      case 1u: {
        sb[sb_base0].y = f16(wv0.x);
        sb[sb_base0 + 32u].y = f16(wv0.y);
        sb[sb_base0 + 64u].y = f16(wv0.z);
        sb[sb_base0 + 96u].y = f16(wv0.w);
      }
      case 2u: {
        sb[sb_base0].z = f16(wv0.x);
        sb[sb_base0 + 32u].z = f16(wv0.y);
        sb[sb_base0 + 64u].z = f16(wv0.z);
        sb[sb_base0 + 96u].z = f16(wv0.w);
      }
      default: {
        sb[sb_base0].w = f16(wv0.x);
        sb[sb_base0 + 32u].w = f16(wv0.y);
        sb[sb_base0 + 64u].w = f16(wv0.z);
        sb[sb_base0 + 96u].w = f16(wv0.w);
      }
    }
    var wv1 = vec4<f32>(0.0);
    if (wcol1 < dims.n) {
      if (wk0 < dims.k) {
        wv1.x = k[krow_base1 + wk0];
      }
      if (wk0 + 1u < dims.k) {
        wv1.y = k[krow_base1 + wk0 + 1u];
      }
      if (wk0 + 2u < dims.k) {
        wv1.z = k[krow_base1 + wk0 + 2u];
      }
      if (wk0 + 3u < dims.k) {
        wv1.w = k[krow_base1 + wk0 + 3u];
      }
    }
    // 半スケール契約（ADR 0023）: scale は q 側と k 側の**両方**へ掛ける。範囲外は 0 のままで
    // 0 · scale = 0 なので端数タイルの結論は変わらない
    wv1 = wv1 * dims.scale;
    switch wsl1 {
      case 0u: {
        sb[sb_base1].x = f16(wv1.x);
        sb[sb_base1 + 32u].x = f16(wv1.y);
        sb[sb_base1 + 64u].x = f16(wv1.z);
        sb[sb_base1 + 96u].x = f16(wv1.w);
      }
      case 1u: {
        sb[sb_base1].y = f16(wv1.x);
        sb[sb_base1 + 32u].y = f16(wv1.y);
        sb[sb_base1 + 64u].y = f16(wv1.z);
        sb[sb_base1 + 96u].y = f16(wv1.w);
      }
      case 2u: {
        sb[sb_base1].z = f16(wv1.x);
        sb[sb_base1 + 32u].z = f16(wv1.y);
        sb[sb_base1 + 64u].z = f16(wv1.z);
        sb[sb_base1 + 96u].z = f16(wv1.w);
      }
      default: {
        sb[sb_base1].w = f16(wv1.x);
        sb[sb_base1 + 32u].w = f16(wv1.y);
        sb[sb_base1 + 64u].w = f16(wv1.z);
        sb[sb_base1 + 96u].w = f16(wv1.w);
      }
    }
    workgroupBarrier();
    // 共有ロード 10 回（B の vec4 2 + A のスカラ 8）で 64 MAC。
    // 縮約は k 昇順の逐次で、1 出力要素あたりの加算順序は 16×16 の 1 スレッド 1 出力と
    // 完全に一致する。
    // MUST: f32 への拡幅は**レジスタロード時に 1 回**（10 回 / 64 MAC）。MAC ごとに
    // f32(av * bv) と書くと変換が 64 回に増え、しかも積が f16 精度に落ちる（ADR 0028 の丸め列 2）。
    // f16 → f32 の拡幅は厳密なので、値は「入力を f16 に丸めた f32 変種」と 1 ビットも違わない。
    for (var kk = 0u; kk < 16u; kk = kk + 1u) {
      let bv0 = vec4<f32>(sb[kk * 32u + lid.x * 2u]);
      let bv1 = vec4<f32>(sb[kk * 32u + lid.x * 2u + 1u]);
      acc0_0 = acc0_0 + f32(sa[(lid.y * 8u + 0u) * 16u + kk]) * bv0;
      acc0_1 = acc0_1 + f32(sa[(lid.y * 8u + 0u) * 16u + kk]) * bv1;
      acc1_0 = acc1_0 + f32(sa[(lid.y * 8u + 1u) * 16u + kk]) * bv0;
      acc1_1 = acc1_1 + f32(sa[(lid.y * 8u + 1u) * 16u + kk]) * bv1;
      acc2_0 = acc2_0 + f32(sa[(lid.y * 8u + 2u) * 16u + kk]) * bv0;
      acc2_1 = acc2_1 + f32(sa[(lid.y * 8u + 2u) * 16u + kk]) * bv1;
      acc3_0 = acc3_0 + f32(sa[(lid.y * 8u + 3u) * 16u + kk]) * bv0;
      acc3_1 = acc3_1 + f32(sa[(lid.y * 8u + 3u) * 16u + kk]) * bv1;
      acc4_0 = acc4_0 + f32(sa[(lid.y * 8u + 4u) * 16u + kk]) * bv0;
      acc4_1 = acc4_1 + f32(sa[(lid.y * 8u + 4u) * 16u + kk]) * bv1;
      acc5_0 = acc5_0 + f32(sa[(lid.y * 8u + 5u) * 16u + kk]) * bv0;
      acc5_1 = acc5_1 + f32(sa[(lid.y * 8u + 5u) * 16u + kk]) * bv1;
      acc6_0 = acc6_0 + f32(sa[(lid.y * 8u + 6u) * 16u + kk]) * bv0;
      acc6_1 = acc6_1 + f32(sa[(lid.y * 8u + 6u) * 16u + kk]) * bv1;
      acc7_0 = acc7_0 + f32(sa[(lid.y * 8u + 7u) * 16u + kk]) * bv0;
      acc7_1 = acc7_1 + f32(sa[(lid.y * 8u + 7u) * 16u + kk]) * bv1;
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
    let obase = cbase + orow0 * dims.n;
    if (ocol < dims.n) {
      s[obase + ocol] = f16(acc0_0.x);
    }
    if (ocol + 1u < dims.n) {
      s[obase + ocol + 1u] = f16(acc0_0.y);
    }
    if (ocol + 2u < dims.n) {
      s[obase + ocol + 2u] = f16(acc0_0.z);
    }
    if (ocol + 3u < dims.n) {
      s[obase + ocol + 3u] = f16(acc0_0.w);
    }
    if (ocol + 4u < dims.n) {
      s[obase + ocol + 4u] = f16(acc0_1.x);
    }
    if (ocol + 5u < dims.n) {
      s[obase + ocol + 5u] = f16(acc0_1.y);
    }
    if (ocol + 6u < dims.n) {
      s[obase + ocol + 6u] = f16(acc0_1.z);
    }
    if (ocol + 7u < dims.n) {
      s[obase + ocol + 7u] = f16(acc0_1.w);
    }
  }
  if (orow1 < dims.m) {
    let obase = cbase + orow1 * dims.n;
    if (ocol < dims.n) {
      s[obase + ocol] = f16(acc1_0.x);
    }
    if (ocol + 1u < dims.n) {
      s[obase + ocol + 1u] = f16(acc1_0.y);
    }
    if (ocol + 2u < dims.n) {
      s[obase + ocol + 2u] = f16(acc1_0.z);
    }
    if (ocol + 3u < dims.n) {
      s[obase + ocol + 3u] = f16(acc1_0.w);
    }
    if (ocol + 4u < dims.n) {
      s[obase + ocol + 4u] = f16(acc1_1.x);
    }
    if (ocol + 5u < dims.n) {
      s[obase + ocol + 5u] = f16(acc1_1.y);
    }
    if (ocol + 6u < dims.n) {
      s[obase + ocol + 6u] = f16(acc1_1.z);
    }
    if (ocol + 7u < dims.n) {
      s[obase + ocol + 7u] = f16(acc1_1.w);
    }
  }
  if (orow2 < dims.m) {
    let obase = cbase + orow2 * dims.n;
    if (ocol < dims.n) {
      s[obase + ocol] = f16(acc2_0.x);
    }
    if (ocol + 1u < dims.n) {
      s[obase + ocol + 1u] = f16(acc2_0.y);
    }
    if (ocol + 2u < dims.n) {
      s[obase + ocol + 2u] = f16(acc2_0.z);
    }
    if (ocol + 3u < dims.n) {
      s[obase + ocol + 3u] = f16(acc2_0.w);
    }
    if (ocol + 4u < dims.n) {
      s[obase + ocol + 4u] = f16(acc2_1.x);
    }
    if (ocol + 5u < dims.n) {
      s[obase + ocol + 5u] = f16(acc2_1.y);
    }
    if (ocol + 6u < dims.n) {
      s[obase + ocol + 6u] = f16(acc2_1.z);
    }
    if (ocol + 7u < dims.n) {
      s[obase + ocol + 7u] = f16(acc2_1.w);
    }
  }
  if (orow3 < dims.m) {
    let obase = cbase + orow3 * dims.n;
    if (ocol < dims.n) {
      s[obase + ocol] = f16(acc3_0.x);
    }
    if (ocol + 1u < dims.n) {
      s[obase + ocol + 1u] = f16(acc3_0.y);
    }
    if (ocol + 2u < dims.n) {
      s[obase + ocol + 2u] = f16(acc3_0.z);
    }
    if (ocol + 3u < dims.n) {
      s[obase + ocol + 3u] = f16(acc3_0.w);
    }
    if (ocol + 4u < dims.n) {
      s[obase + ocol + 4u] = f16(acc3_1.x);
    }
    if (ocol + 5u < dims.n) {
      s[obase + ocol + 5u] = f16(acc3_1.y);
    }
    if (ocol + 6u < dims.n) {
      s[obase + ocol + 6u] = f16(acc3_1.z);
    }
    if (ocol + 7u < dims.n) {
      s[obase + ocol + 7u] = f16(acc3_1.w);
    }
  }
  if (orow4 < dims.m) {
    let obase = cbase + orow4 * dims.n;
    if (ocol < dims.n) {
      s[obase + ocol] = f16(acc4_0.x);
    }
    if (ocol + 1u < dims.n) {
      s[obase + ocol + 1u] = f16(acc4_0.y);
    }
    if (ocol + 2u < dims.n) {
      s[obase + ocol + 2u] = f16(acc4_0.z);
    }
    if (ocol + 3u < dims.n) {
      s[obase + ocol + 3u] = f16(acc4_0.w);
    }
    if (ocol + 4u < dims.n) {
      s[obase + ocol + 4u] = f16(acc4_1.x);
    }
    if (ocol + 5u < dims.n) {
      s[obase + ocol + 5u] = f16(acc4_1.y);
    }
    if (ocol + 6u < dims.n) {
      s[obase + ocol + 6u] = f16(acc4_1.z);
    }
    if (ocol + 7u < dims.n) {
      s[obase + ocol + 7u] = f16(acc4_1.w);
    }
  }
  if (orow5 < dims.m) {
    let obase = cbase + orow5 * dims.n;
    if (ocol < dims.n) {
      s[obase + ocol] = f16(acc5_0.x);
    }
    if (ocol + 1u < dims.n) {
      s[obase + ocol + 1u] = f16(acc5_0.y);
    }
    if (ocol + 2u < dims.n) {
      s[obase + ocol + 2u] = f16(acc5_0.z);
    }
    if (ocol + 3u < dims.n) {
      s[obase + ocol + 3u] = f16(acc5_0.w);
    }
    if (ocol + 4u < dims.n) {
      s[obase + ocol + 4u] = f16(acc5_1.x);
    }
    if (ocol + 5u < dims.n) {
      s[obase + ocol + 5u] = f16(acc5_1.y);
    }
    if (ocol + 6u < dims.n) {
      s[obase + ocol + 6u] = f16(acc5_1.z);
    }
    if (ocol + 7u < dims.n) {
      s[obase + ocol + 7u] = f16(acc5_1.w);
    }
  }
  if (orow6 < dims.m) {
    let obase = cbase + orow6 * dims.n;
    if (ocol < dims.n) {
      s[obase + ocol] = f16(acc6_0.x);
    }
    if (ocol + 1u < dims.n) {
      s[obase + ocol + 1u] = f16(acc6_0.y);
    }
    if (ocol + 2u < dims.n) {
      s[obase + ocol + 2u] = f16(acc6_0.z);
    }
    if (ocol + 3u < dims.n) {
      s[obase + ocol + 3u] = f16(acc6_0.w);
    }
    if (ocol + 4u < dims.n) {
      s[obase + ocol + 4u] = f16(acc6_1.x);
    }
    if (ocol + 5u < dims.n) {
      s[obase + ocol + 5u] = f16(acc6_1.y);
    }
    if (ocol + 6u < dims.n) {
      s[obase + ocol + 6u] = f16(acc6_1.z);
    }
    if (ocol + 7u < dims.n) {
      s[obase + ocol + 7u] = f16(acc6_1.w);
    }
  }
  if (orow7 < dims.m) {
    let obase = cbase + orow7 * dims.n;
    if (ocol < dims.n) {
      s[obase + ocol] = f16(acc7_0.x);
    }
    if (ocol + 1u < dims.n) {
      s[obase + ocol + 1u] = f16(acc7_0.y);
    }
    if (ocol + 2u < dims.n) {
      s[obase + ocol + 2u] = f16(acc7_0.z);
    }
    if (ocol + 3u < dims.n) {
      s[obase + ocol + 3u] = f16(acc7_0.w);
    }
    if (ocol + 4u < dims.n) {
      s[obase + ocol + 4u] = f16(acc7_1.x);
    }
    if (ocol + 5u < dims.n) {
      s[obase + ocol + 5u] = f16(acc7_1.y);
    }
    if (ocol + 6u < dims.n) {
      s[obase + ocol + 6u] = f16(acc7_1.z);
    }
    if (ocol + 7u < dims.n) {
      s[obase + ocol + 7u] = f16(acc7_1.w);
    }
  }
}
