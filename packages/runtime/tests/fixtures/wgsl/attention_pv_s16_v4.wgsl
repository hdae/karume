// karume attention_pv (O[b,m,d] = P[b,m,n] · v[b,n,d], P = exp(S − m)·inv は非実体化, f32, S は f16 格納（pack2x16float）, レジスタ 128x128 タイル / 1 スレッド 8x8 / wg 16x16 + vec4)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> s: array<u32>;
@group(0) @binding(2) var<storage, read> v: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> stats: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<vec4<f32>>;

// f16 格納の quad 読み: quad q = s[2q] と s[2q + 1] の 2 語（unpack は厳密）
fn score_quad(q: u32) -> vec4<f32> {
  let w = q * 2u;
  let lo = unpack2x16float(s[w]);
  let hi = unpack2x16float(s[w + 1u]);
  return vec4<f32>(lo.x, lo.y, hi.x, hi.y);
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
  let k4 = dims.k / 4u;
  let n4 = dims.n / 4u;
  // バッチは workgroup 単位で一様（z 軸 1 つが 1 バッチのタイル）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすための前提。MUST: base は quad 単位（要素単位で組むと隣のバッチを読む）
  let abase = wid.z * dims.m * k4;
  let bbase = wid.z * dims.k * n4;
  let cbase = wid.z * dims.m * n4;
  let rbase = wid.z * dims.m;
  // A タイルの担当（128 行 × 4 quad を 256 スレッドで 2 巡）
  let ar = tid / 4u;
  let aq = tid % 4u;
  let arow0 = wid.y * 128u + ar;
  let arow_base0 = abase + arow0 * k4;
  let sa_base0 = ar * 16u + aq * 4u;
  let arow1 = arow0 + 64u;
  let arow_base1 = arow_base0 + 64u * k4;
  let sa_base1 = sa_base0 + 1024u;
  // 端タイルでは arow >= m がありうるので添字を 0 へ倒す（読んだ値は arow < m の枝でしか
  // 使われない）。範囲外の stats を読むこと自体は WGSL の境界付きアクセスで安全
  let stat_at0 = select(0u, (rbase + arow0) * 2u, arow0 < dims.m);
  let row_max0 = stats[stat_at0];
  let row_inv0 = stats[stat_at0 + 1u];
  let stat_at1 = select(0u, (rbase + arow1) * 2u, arow1 < dims.m);
  let row_max1 = stats[stat_at1];
  let row_inv1 = stats[stat_at1 + 1u];
  // B タイルの担当（K 16 行 × 列 quad 32 を 256 スレッドで 2 巡）
  let bk0 = tid / 32u;
  let bcq = tid % 32u;
  let bc4 = wid.x * 32u + bcq;
  let bk1 = bk0 + 8u;
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
      let raw0 = score_quad(arow_base0 + t * 4u + aq);
      av0 = vec4<f32>(
        exp(raw0.x - row_max0) * row_inv0,
        exp(raw0.y - row_max0) * row_inv0,
        exp(raw0.z - row_max0) * row_inv0,
        exp(raw0.w - row_max0) * row_inv0,
      );
    }
    sa[sa_base0] = av0.x;
    sa[sa_base0 + 1u] = av0.y;
    sa[sa_base0 + 2u] = av0.z;
    sa[sa_base0 + 3u] = av0.w;
    var av1 = vec4<f32>(0.0);
    if (arow1 < dims.m && ak0 < dims.k) {
      let raw1 = score_quad(arow_base1 + t * 4u + aq);
      av1 = vec4<f32>(
        exp(raw1.x - row_max1) * row_inv1,
        exp(raw1.y - row_max1) * row_inv1,
        exp(raw1.z - row_max1) * row_inv1,
        exp(raw1.w - row_max1) * row_inv1,
      );
    }
    sa[sa_base1] = av1.x;
    sa[sa_base1 + 1u] = av1.y;
    sa[sa_base1 + 2u] = av1.z;
    sa[sa_base1 + 3u] = av1.w;
    let brow0 = t * 16u + bk0;
    var bv4_0 = vec4<f32>(0.0);
    if (brow0 < dims.k && bc4 < n4) {
      bv4_0 = v[bbase + brow0 * n4 + bc4];
    }
    sb[bk0 * 32u + bcq] = bv4_0;
    let brow1 = t * 16u + bk1;
    var bv4_1 = vec4<f32>(0.0);
    if (brow1 < dims.k && bc4 < n4) {
      bv4_1 = v[bbase + brow1 * n4 + bc4];
    }
    sb[bk1 * 32u + bcq] = bv4_1;
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
      o[cbase + orow0 * n4 + ocq0] = acc0_0;
    }
    if (orow1 < dims.m) {
      o[cbase + orow1 * n4 + ocq0] = acc1_0;
    }
    if (orow2 < dims.m) {
      o[cbase + orow2 * n4 + ocq0] = acc2_0;
    }
    if (orow3 < dims.m) {
      o[cbase + orow3 * n4 + ocq0] = acc3_0;
    }
    if (orow4 < dims.m) {
      o[cbase + orow4 * n4 + ocq0] = acc4_0;
    }
    if (orow5 < dims.m) {
      o[cbase + orow5 * n4 + ocq0] = acc5_0;
    }
    if (orow6 < dims.m) {
      o[cbase + orow6 * n4 + ocq0] = acc6_0;
    }
    if (orow7 < dims.m) {
      o[cbase + orow7 * n4 + ocq0] = acc7_0;
    }
  }
  if (ocq1 < n4) {
    if (orow0 < dims.m) {
      o[cbase + orow0 * n4 + ocq1] = acc0_1;
    }
    if (orow1 < dims.m) {
      o[cbase + orow1 * n4 + ocq1] = acc1_1;
    }
    if (orow2 < dims.m) {
      o[cbase + orow2 * n4 + ocq1] = acc2_1;
    }
    if (orow3 < dims.m) {
      o[cbase + orow3 * n4 + ocq1] = acc3_1;
    }
    if (orow4 < dims.m) {
      o[cbase + orow4 * n4 + ocq1] = acc4_1;
    }
    if (orow5 < dims.m) {
      o[cbase + orow5 * n4 + ocq1] = acc5_1;
    }
    if (orow6 < dims.m) {
      o[cbase + orow6 * n4 + ocq1] = acc6_1;
    }
    if (orow7 < dims.m) {
      o[cbase + orow7 * n4 + ocq1] = acc7_1;
    }
  }
}
