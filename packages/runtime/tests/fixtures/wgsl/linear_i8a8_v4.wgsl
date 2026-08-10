// karume linear (x[m,k] · wᵀ[k,n] + bias[n], 活性 per-token i8 × 重み i8 の整数内積, タイル 128x64 / 1 スレッド 8x8 / wg 8x16 / K 16 + vec4)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> xq: array<u32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> wscale: array<f32>;
@group(0) @binding(6) var<storage, read> xscale: array<f32>;

// 整数内積（WGSL 言語拡張 packed_4x8_integer_dot_product）。エミュ変種と**同じ整数**を返す
fn idot(a: u32, b: u32) -> i32 {
  return dot4I8Packed(a, b);
}

// 共有タイルは **[pack][行] / [pack][列]** 配置（プロトタイプの [行][pack] からの組み替え —
// 内側の読みが語ストライド 4 になりバンク衝突が 8-way から 2-way に減る）
var<workgroup> sa: array<u32, 512>;
var<workgroup> sb: array<u32, 256>;

@compute @workgroup_size(8, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 8u + lid.x;
  // 適格条件で k % 4 == 0 なので、行頭は必ず語境界に来る（i8 ペイロードは平坦添字 4 詰め）
  let k4 = dims.k / 4u;
  // x タイルの担当（128 行 × 4 pack を 128 スレッドで 4 巡）
  let ar = tid / 4u;
  let ap = tid % 4u;
  let sa_at = ap * 128u + ar;
  let arow0 = wid.y * 128u + ar;
  let arow_base0 = arow0 * k4;
  let arow1 = arow0 + 32u;
  let arow_base1 = arow1 * k4;
  let arow2 = arow0 + 64u;
  let arow_base2 = arow2 * k4;
  let arow3 = arow0 + 96u;
  let arow_base3 = arow3 * k4;
  // W タイルの担当（64 出力チャネル × 4 pack を 128 スレッドで 2 巡）。
  // 重みは [n,k] のまま読む（連続方向が k = パック方向なので**転置が要らない**）
  let wc = tid / 4u;
  let wp = tid % 4u;
  let sb_at = wp * 64u + wc;
  let wcol0 = wid.x * 64u + wc;
  let wrow_base0 = wcol0 * k4;
  let wcol1 = wcol0 + 32u;
  let wrow_base1 = wcol1 * k4;
  // ループ条件は uniform（dims は uniform バッファ）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすために必要
  let tiles = (k4 + 3u) / 4u;
  var acc0_0 = vec4<i32>(0);
  var acc0_1 = vec4<i32>(0);
  var acc1_0 = vec4<i32>(0);
  var acc1_1 = vec4<i32>(0);
  var acc2_0 = vec4<i32>(0);
  var acc2_1 = vec4<i32>(0);
  var acc3_0 = vec4<i32>(0);
  var acc3_1 = vec4<i32>(0);
  var acc4_0 = vec4<i32>(0);
  var acc4_1 = vec4<i32>(0);
  var acc5_0 = vec4<i32>(0);
  var acc5_1 = vec4<i32>(0);
  var acc6_0 = vec4<i32>(0);
  var acc6_1 = vec4<i32>(0);
  var acc7_0 = vec4<i32>(0);
  var acc7_1 = vec4<i32>(0);
  for (var t = 0u; t < tiles; t = t + 1u) {
    // 範囲外は 0 で埋める（dot4I8Packed(0, x) == 0 なので K 端数でも結果は厳密）
    let apack = t * 4u + ap;
    var av0 = 0u;
    if (arow0 < dims.m && apack < k4) {
      av0 = xq[arow_base0 + apack];
    }
    sa[sa_at] = av0;
    var av1 = 0u;
    if (arow1 < dims.m && apack < k4) {
      av1 = xq[arow_base1 + apack];
    }
    sa[sa_at + 32u] = av1;
    var av2 = 0u;
    if (arow2 < dims.m && apack < k4) {
      av2 = xq[arow_base2 + apack];
    }
    sa[sa_at + 64u] = av2;
    var av3 = 0u;
    if (arow3 < dims.m && apack < k4) {
      av3 = xq[arow_base3 + apack];
    }
    sa[sa_at + 96u] = av3;
    let wpack = t * 4u + wp;
    var wv0 = 0u;
    if (wcol0 < dims.n && wpack < k4) {
      wv0 = w[wrow_base0 + wpack];
    }
    sb[sb_at] = wv0;
    var wv1 = 0u;
    if (wcol1 < dims.n && wpack < k4) {
      wv1 = w[wrow_base1 + wpack];
    }
    sb[sb_at + 32u] = wv1;
    workgroupBarrier();
    // 共有ロード 16 回（B の 8 語 + A の 8 語）で 64 個の整数内積 = 256 MAC。
    // 縮約は i32 の厳密加算なので**順序に依存しない**（f32 骨格と違い加算順は数値契約に無い）
    for (var p = 0u; p < 4u; p = p + 1u) {
      let bcol = p * 64u + lid.x * 8u;
      let b0 = sb[bcol];
      let b1 = sb[bcol + 1u];
      let b2 = sb[bcol + 2u];
      let b3 = sb[bcol + 3u];
      let b4 = sb[bcol + 4u];
      let b5 = sb[bcol + 5u];
      let b6 = sb[bcol + 6u];
      let b7 = sb[bcol + 7u];
      let arow_at = p * 128u + lid.y * 8u;
      let a0 = sa[arow_at];
      acc0_0 = acc0_0 + vec4<i32>(idot(a0, b0), idot(a0, b1), idot(a0, b2), idot(a0, b3));
      acc0_1 = acc0_1 + vec4<i32>(idot(a0, b4), idot(a0, b5), idot(a0, b6), idot(a0, b7));
      let a1 = sa[arow_at + 1u];
      acc1_0 = acc1_0 + vec4<i32>(idot(a1, b0), idot(a1, b1), idot(a1, b2), idot(a1, b3));
      acc1_1 = acc1_1 + vec4<i32>(idot(a1, b4), idot(a1, b5), idot(a1, b6), idot(a1, b7));
      let a2 = sa[arow_at + 2u];
      acc2_0 = acc2_0 + vec4<i32>(idot(a2, b0), idot(a2, b1), idot(a2, b2), idot(a2, b3));
      acc2_1 = acc2_1 + vec4<i32>(idot(a2, b4), idot(a2, b5), idot(a2, b6), idot(a2, b7));
      let a3 = sa[arow_at + 3u];
      acc3_0 = acc3_0 + vec4<i32>(idot(a3, b0), idot(a3, b1), idot(a3, b2), idot(a3, b3));
      acc3_1 = acc3_1 + vec4<i32>(idot(a3, b4), idot(a3, b5), idot(a3, b6), idot(a3, b7));
      let a4 = sa[arow_at + 4u];
      acc4_0 = acc4_0 + vec4<i32>(idot(a4, b0), idot(a4, b1), idot(a4, b2), idot(a4, b3));
      acc4_1 = acc4_1 + vec4<i32>(idot(a4, b4), idot(a4, b5), idot(a4, b6), idot(a4, b7));
      let a5 = sa[arow_at + 5u];
      acc5_0 = acc5_0 + vec4<i32>(idot(a5, b0), idot(a5, b1), idot(a5, b2), idot(a5, b3));
      acc5_1 = acc5_1 + vec4<i32>(idot(a5, b4), idot(a5, b5), idot(a5, b6), idot(a5, b7));
      let a6 = sa[arow_at + 6u];
      acc6_0 = acc6_0 + vec4<i32>(idot(a6, b0), idot(a6, b1), idot(a6, b2), idot(a6, b3));
      acc6_1 = acc6_1 + vec4<i32>(idot(a6, b4), idot(a6, b5), idot(a6, b6), idot(a6, b7));
      let a7 = sa[arow_at + 7u];
      acc7_0 = acc7_0 + vec4<i32>(idot(a7, b0), idot(a7, b1), idot(a7, b2), idot(a7, b3));
      acc7_1 = acc7_1 + vec4<i32>(idot(a7, b4), idot(a7, b5), idot(a7, b6), idot(a7, b7));
    }
    workgroupBarrier();
  }
  let n4 = dims.n / 4u;
  let ocq0 = wid.x * 16u + lid.x * 2u;
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
    // n % 4 == 0 かつ ocq0 < n4 なので oc + 3 < n（bias は常に f32 — ADR 0006）
    let oc = ocq0 * 4u;
    let ws = vec4<f32>(wscale[oc], wscale[oc + 1u], wscale[oc + 2u], wscale[oc + 3u]);
    let biasv = vec4<f32>(bias[oc], bias[oc + 1u], bias[oc + 2u], bias[oc + 3u]);
    // MUST: xs·wscale を先に 1 つの f32 へ畳み、積和は fma（単一丸め）
    if (orow0 < dims.m) {
      out[orow0 * n4 + ocq0] = fma(vec4<f32>(acc0_0), xscale[orow0] * ws, biasv);
    }
    if (orow1 < dims.m) {
      out[orow1 * n4 + ocq0] = fma(vec4<f32>(acc1_0), xscale[orow1] * ws, biasv);
    }
    if (orow2 < dims.m) {
      out[orow2 * n4 + ocq0] = fma(vec4<f32>(acc2_0), xscale[orow2] * ws, biasv);
    }
    if (orow3 < dims.m) {
      out[orow3 * n4 + ocq0] = fma(vec4<f32>(acc3_0), xscale[orow3] * ws, biasv);
    }
    if (orow4 < dims.m) {
      out[orow4 * n4 + ocq0] = fma(vec4<f32>(acc4_0), xscale[orow4] * ws, biasv);
    }
    if (orow5 < dims.m) {
      out[orow5 * n4 + ocq0] = fma(vec4<f32>(acc5_0), xscale[orow5] * ws, biasv);
    }
    if (orow6 < dims.m) {
      out[orow6 * n4 + ocq0] = fma(vec4<f32>(acc6_0), xscale[orow6] * ws, biasv);
    }
    if (orow7 < dims.m) {
      out[orow7 * n4 + ocq0] = fma(vec4<f32>(acc7_0), xscale[orow7] * ws, biasv);
    }
  }
  if (ocq1 < n4) {
    // n % 4 == 0 かつ ocq1 < n4 なので oc + 3 < n（bias は常に f32 — ADR 0006）
    let oc = ocq1 * 4u;
    let ws = vec4<f32>(wscale[oc], wscale[oc + 1u], wscale[oc + 2u], wscale[oc + 3u]);
    let biasv = vec4<f32>(bias[oc], bias[oc + 1u], bias[oc + 2u], bias[oc + 3u]);
    // MUST: xs·wscale を先に 1 つの f32 へ畳み、積和は fma（単一丸め）
    if (orow0 < dims.m) {
      out[orow0 * n4 + ocq1] = fma(vec4<f32>(acc0_1), xscale[orow0] * ws, biasv);
    }
    if (orow1 < dims.m) {
      out[orow1 * n4 + ocq1] = fma(vec4<f32>(acc1_1), xscale[orow1] * ws, biasv);
    }
    if (orow2 < dims.m) {
      out[orow2 * n4 + ocq1] = fma(vec4<f32>(acc2_1), xscale[orow2] * ws, biasv);
    }
    if (orow3 < dims.m) {
      out[orow3 * n4 + ocq1] = fma(vec4<f32>(acc3_1), xscale[orow3] * ws, biasv);
    }
    if (orow4 < dims.m) {
      out[orow4 * n4 + ocq1] = fma(vec4<f32>(acc4_1), xscale[orow4] * ws, biasv);
    }
    if (orow5 < dims.m) {
      out[orow5 * n4 + ocq1] = fma(vec4<f32>(acc5_1), xscale[orow5] * ws, biasv);
    }
    if (orow6 < dims.m) {
      out[orow6 * n4 + ocq1] = fma(vec4<f32>(acc6_1), xscale[orow6] * ws, biasv);
    }
    if (orow7 < dims.m) {
      out[orow7 * n4 + ocq1] = fma(vec4<f32>(acc7_1), xscale[orow7] * ws, biasv);
    }
  }
}
