// karume attention_qk (S[b,m,n] = (q·scale)[b,m,d] · (k·scale)[b,n,d]ᵀ, f32, GQA（k は kv-head へ整数除算で写す）, レジスタ 128x128 タイル / 1 スレッド 8x8 / wg 16x16 + vec4)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
  scale: f32,
  kv_repeat: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> k: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> s: array<vec4<f32>>;

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
  let k4 = dims.k / 4u;
  let n4 = dims.n / 4u;
  // バッチは workgroup 単位で一様（z 軸 1 つが 1 バッチのタイル）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすための前提。MUST: base は quad 単位（要素単位で組むと隣のバッチを読む）
  // GQA（ADR 0067 決定 2）: K / V だけ kv-head へ写す（wid.z / r = b·Hkv + h/r）。
  // q / S / O / 行統計は q-head のまま
  let abase = wid.z * dims.m * k4;
  let bbase = (wid.z / dims.kv_repeat) * dims.n * dims.k;
  let cbase = wid.z * dims.m * n4;
  // A タイルの担当（128 行 × 4 quad を 256 スレッドで 2 巡）
  let ar = tid / 4u;
  let aq = tid % 4u;
  let arow0 = wid.y * 128u + ar;
  let arow_base0 = abase + arow0 * k4;
  let sa_base0 = ar * 16u + aq * 4u;
  let arow1 = arow0 + 64u;
  let arow_base1 = arow_base0 + 64u * k4;
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
    if (arow0 < dims.m && ak0 < dims.k) {
      let raw0 = q[arow_base0 + t * 4u + aq];
      av0 = vec4<f32>(
        raw0.x * dims.scale,
        raw0.y * dims.scale,
        raw0.z * dims.scale,
        raw0.w * dims.scale,
      );
    }
    sa[sa_base0] = av0.x;
    sa[sa_base0 + 1u] = av0.y;
    sa[sa_base0 + 2u] = av0.z;
    sa[sa_base0 + 3u] = av0.w;
    var av1 = vec4<f32>(0.0);
    if (arow1 < dims.m && ak0 < dims.k) {
      let raw1 = q[arow_base1 + t * 4u + aq];
      av1 = vec4<f32>(
        raw1.x * dims.scale,
        raw1.y * dims.scale,
        raw1.z * dims.scale,
        raw1.w * dims.scale,
      );
    }
    sa[sa_base1] = av1.x;
    sa[sa_base1 + 1u] = av1.y;
    sa[sa_base1 + 2u] = av1.z;
    sa[sa_base1 + 3u] = av1.w;
    let wk0 = t * 16u + wq * 4u;
    var wv0 = vec4<f32>(0.0);
    if (wcol0 < dims.n && wk0 < dims.k) {
      wv0 = k[(krow_base0 + wk0) >> 2u];
    }
    // 半スケール契約（ADR 0023）: scale は q 側と k 側の**両方**へ掛ける。範囲外は 0 のままで
    // 0 · scale = 0 なので端数タイルの結論は変わらない
    wv0 = wv0 * dims.scale;
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
    if (wcol1 < dims.n && wk0 < dims.k) {
      wv1 = k[(krow_base1 + wk0) >> 2u];
    }
    // 半スケール契約（ADR 0023）: scale は q 側と k 側の**両方**へ掛ける。範囲外は 0 のままで
    // 0 · scale = 0 なので端数タイルの結論は変わらない
    wv1 = wv1 * dims.scale;
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
  let ocq0 = wid.x * 32u + lid.x * 2u;
  let ocq1 = ocq0 + 1u;
  let orow0 = wid.y * 128u + lid.y * 8u;
  let orow1 = orow0 + 1u;
  let orow2 = orow0 + 2u;
  let orow3 = orow0 + 3u;
  let orow4 = orow0 + 4u;
  let orow5 = orow0 + 5u;
  let orow6 = orow0 + 6u;
  let orow7 = orow0 + 7u;
  if (ocq0 < n4) {
    if (orow0 < dims.m) {
      s[cbase + orow0 * n4 + ocq0] = acc0_0;
    }
    if (orow1 < dims.m) {
      s[cbase + orow1 * n4 + ocq0] = acc1_0;
    }
    if (orow2 < dims.m) {
      s[cbase + orow2 * n4 + ocq0] = acc2_0;
    }
    if (orow3 < dims.m) {
      s[cbase + orow3 * n4 + ocq0] = acc3_0;
    }
    if (orow4 < dims.m) {
      s[cbase + orow4 * n4 + ocq0] = acc4_0;
    }
    if (orow5 < dims.m) {
      s[cbase + orow5 * n4 + ocq0] = acc5_0;
    }
    if (orow6 < dims.m) {
      s[cbase + orow6 * n4 + ocq0] = acc6_0;
    }
    if (orow7 < dims.m) {
      s[cbase + orow7 * n4 + ocq0] = acc7_0;
    }
  }
  if (ocq1 < n4) {
    if (orow0 < dims.m) {
      s[cbase + orow0 * n4 + ocq1] = acc0_1;
    }
    if (orow1 < dims.m) {
      s[cbase + orow1 * n4 + ocq1] = acc1_1;
    }
    if (orow2 < dims.m) {
      s[cbase + orow2 * n4 + ocq1] = acc2_1;
    }
    if (orow3 < dims.m) {
      s[cbase + orow3 * n4 + ocq1] = acc3_1;
    }
    if (orow4 < dims.m) {
      s[cbase + orow4 * n4 + ocq1] = acc4_1;
    }
    if (orow5 < dims.m) {
      s[cbase + orow5 * n4 + ocq1] = acc5_1;
    }
    if (orow6 < dims.m) {
      s[cbase + orow6 * n4 + ocq1] = acc6_1;
    }
    if (orow7 < dims.m) {
      s[cbase + orow7 * n4 + ocq1] = acc7_1;
    }
  }
}
