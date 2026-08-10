// karume attention_qk (S[b,m,n] = (q·scale)[b,m,d] · (k·scale)[b,n,d]ᵀ, f32, レジスタ 64x64 タイル / 1 スレッド 8x4 / wg 16x8)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
  scale: f32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read_write> s: array<f32>;

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
  // バッチは workgroup 単位で一様（z 軸 1 つが 1 バッチのタイル）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすための前提
  let abase = wid.z * dims.m * dims.k;
  let bbase = wid.z * dims.n * dims.k;
  let cbase = wid.z * dims.m * dims.n;
  // A タイルの担当（64 行 × 4 quad を 128 スレッドで 2 巡）
  let ar = tid / 4u;
  let aq = tid % 4u;
  let arow0 = wid.y * 64u + ar;
  let arow_base0 = abase + arow0 * dims.k;
  let sa_base0 = ar * 16u + aq * 4u;
  let arow1 = arow0 + 32u;
  let arow_base1 = arow_base0 + 32u * dims.k;
  let sa_base1 = sa_base0 + 512u;
  // k タイルの担当（64 列（N）× 4 quad を 128 スレッドで 2 巡）。
  // k は [N,D] のまま読み、**共有メモリ側で転置して置く**（linear の重み読みと同じ構造）
  let wc0 = tid / 4u;
  let wq = tid % 4u;
  let wcol0 = wid.x * 64u + wc0;
  let krow_base0 = bbase + wcol0 * dims.k;
  let wsq0 = wc0 / 4u;
  let wsl0 = wc0 % 4u;
  let sb_base0 = (wq * 4u) * 16u + wsq0;
  let wc1 = wc0 + 32u;
  let wcol1 = wcol0 + 32u;
  let krow_base1 = krow_base0 + 32u * dims.k;
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
    sa[sa_base0] = av0.x;
    sa[sa_base0 + 1u] = av0.y;
    sa[sa_base0 + 2u] = av0.z;
    sa[sa_base0 + 3u] = av0.w;
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
    sa[sa_base1] = av1.x;
    sa[sa_base1 + 1u] = av1.y;
    sa[sa_base1 + 2u] = av1.z;
    sa[sa_base1 + 3u] = av1.w;
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
    let obase = cbase + orow0 * dims.n;
    if (ocol < dims.n) {
      s[obase + ocol] = acc0_0.x;
    }
    if (ocol + 1u < dims.n) {
      s[obase + ocol + 1u] = acc0_0.y;
    }
    if (ocol + 2u < dims.n) {
      s[obase + ocol + 2u] = acc0_0.z;
    }
    if (ocol + 3u < dims.n) {
      s[obase + ocol + 3u] = acc0_0.w;
    }
  }
  if (orow1 < dims.m) {
    let obase = cbase + orow1 * dims.n;
    if (ocol < dims.n) {
      s[obase + ocol] = acc1_0.x;
    }
    if (ocol + 1u < dims.n) {
      s[obase + ocol + 1u] = acc1_0.y;
    }
    if (ocol + 2u < dims.n) {
      s[obase + ocol + 2u] = acc1_0.z;
    }
    if (ocol + 3u < dims.n) {
      s[obase + ocol + 3u] = acc1_0.w;
    }
  }
  if (orow2 < dims.m) {
    let obase = cbase + orow2 * dims.n;
    if (ocol < dims.n) {
      s[obase + ocol] = acc2_0.x;
    }
    if (ocol + 1u < dims.n) {
      s[obase + ocol + 1u] = acc2_0.y;
    }
    if (ocol + 2u < dims.n) {
      s[obase + ocol + 2u] = acc2_0.z;
    }
    if (ocol + 3u < dims.n) {
      s[obase + ocol + 3u] = acc2_0.w;
    }
  }
  if (orow3 < dims.m) {
    let obase = cbase + orow3 * dims.n;
    if (ocol < dims.n) {
      s[obase + ocol] = acc3_0.x;
    }
    if (ocol + 1u < dims.n) {
      s[obase + ocol + 1u] = acc3_0.y;
    }
    if (ocol + 2u < dims.n) {
      s[obase + ocol + 2u] = acc3_0.z;
    }
    if (ocol + 3u < dims.n) {
      s[obase + ocol + 3u] = acc3_0.w;
    }
  }
  if (orow4 < dims.m) {
    let obase = cbase + orow4 * dims.n;
    if (ocol < dims.n) {
      s[obase + ocol] = acc4_0.x;
    }
    if (ocol + 1u < dims.n) {
      s[obase + ocol + 1u] = acc4_0.y;
    }
    if (ocol + 2u < dims.n) {
      s[obase + ocol + 2u] = acc4_0.z;
    }
    if (ocol + 3u < dims.n) {
      s[obase + ocol + 3u] = acc4_0.w;
    }
  }
  if (orow5 < dims.m) {
    let obase = cbase + orow5 * dims.n;
    if (ocol < dims.n) {
      s[obase + ocol] = acc5_0.x;
    }
    if (ocol + 1u < dims.n) {
      s[obase + ocol + 1u] = acc5_0.y;
    }
    if (ocol + 2u < dims.n) {
      s[obase + ocol + 2u] = acc5_0.z;
    }
    if (ocol + 3u < dims.n) {
      s[obase + ocol + 3u] = acc5_0.w;
    }
  }
  if (orow6 < dims.m) {
    let obase = cbase + orow6 * dims.n;
    if (ocol < dims.n) {
      s[obase + ocol] = acc6_0.x;
    }
    if (ocol + 1u < dims.n) {
      s[obase + ocol + 1u] = acc6_0.y;
    }
    if (ocol + 2u < dims.n) {
      s[obase + ocol + 2u] = acc6_0.z;
    }
    if (ocol + 3u < dims.n) {
      s[obase + ocol + 3u] = acc6_0.w;
    }
  }
  if (orow7 < dims.m) {
    let obase = cbase + orow7 * dims.n;
    if (ocol < dims.n) {
      s[obase + ocol] = acc7_0.x;
    }
    if (ocol + 1u < dims.n) {
      s[obase + ocol + 1u] = acc7_0.y;
    }
    if (ocol + 2u < dims.n) {
      s[obase + ocol + 2u] = acc7_0.z;
    }
    if (ocol + 3u < dims.n) {
      s[obase + ocol + 3u] = acc7_0.w;
    }
  }
}
