// karume attention_qk (S[b,m,n] = q[b,m,d] · k[b,n,d]ᵀ, q/k とも per-token i8 の整数内積, 半スケールは dequant 側, 行窓 a, タイル 128x64 / 1 スレッド 8x8 / wg 8x16 / K 16 + vec4)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
  scale: f32,
  row_offset: u32,
  rows_full: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> qq: array<u32>;
@group(0) @binding(2) var<storage, read> kq: array<u32>;
@group(0) @binding(3) var<storage, read_write> s: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> qscale: array<f32>;
@group(0) @binding(5) var<storage, read> kscale: array<f32>;

// 整数内積（WGSL 言語拡張 packed_4x8_integer_dot_product）。エミュ変種と**同じ整数**を返す
fn idot(a: u32, b: u32) -> i32 {
  return dot4I8Packed(a, b);
}

// 共有タイルは **[pack][行] / [pack][列]** 配置（linear の i8a8 と同じ — 内側の読みが
// 語ストライド 4 になりバンク衝突が 8-way から 2-way に減る）
var<workgroup> sa: array<u32, 512>;
var<workgroup> sb: array<u32, 256>;

@compute @workgroup_size(8, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 8u + lid.x;
  // 適格条件で D % 4 == 0 なので、行頭は必ず語境界に来る（i8 ペイロードは平坦添字 4 詰め）
  let k4 = dims.k / 4u;
  let n4 = dims.n / 4u;
  // B と H を畳んだバッチ軸（workgroup 単位で一様 — 内側の workgroupBarrier が WGSL の
  // 一様性要件を満たすための前提）。MUST: q / k のペイロードは **pack 単位**、scale は行数
  // 単位、S は要素（v4 では quad）単位で数える — 単位を取り違えると B·H ≥ 2 で
  // 隣の head を読み書きする（例外なしの誤値で、B=H=1 のテストには出ない）
  let qbase = wid.z * dims.rows_full * k4 + dims.row_offset * k4;
  let kbase = wid.z * dims.n * k4;
  let qsbase = wid.z * dims.rows_full + dims.row_offset;
  let ksbase = wid.z * dims.n;
  let sbase = wid.z * dims.m * n4;
  // q タイルの担当（128 行 × 4 pack を 128 スレッドで 4 巡）
  let ar = tid / 4u;
  let ap = tid % 4u;
  let sa_at = ap * 128u + ar;
  let arow0 = wid.y * 128u + ar;
  let arow_base0 = qbase + arow0 * k4;
  let arow1 = arow0 + 32u;
  let arow_base1 = qbase + arow1 * k4;
  let arow2 = arow0 + 64u;
  let arow_base2 = qbase + arow2 * k4;
  let arow3 = arow0 + 96u;
  let arow_base3 = qbase + arow3 * k4;
  // B タイルの担当（64 列 × 4 pack を 128 スレッドで 2 巡）。
  // k は [N,D] のまま読む（連続方向が D = パック方向なので**転置が要らない**）
  let wc = tid / 4u;
  let wp = tid % 4u;
  let sb_at = wp * 64u + wc;
  let wcol0 = wid.x * 64u + wc;
  let krow_base0 = kbase + wcol0 * k4;
  let wcol1 = wcol0 + 32u;
  let krow_base1 = kbase + wcol1 * k4;
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
      av0 = qq[arow_base0 + apack];
    }
    sa[sa_at] = av0;
    var av1 = 0u;
    if (arow1 < dims.m && apack < k4) {
      av1 = qq[arow_base1 + apack];
    }
    sa[sa_at + 32u] = av1;
    var av2 = 0u;
    if (arow2 < dims.m && apack < k4) {
      av2 = qq[arow_base2 + apack];
    }
    sa[sa_at + 64u] = av2;
    var av3 = 0u;
    if (arow3 < dims.m && apack < k4) {
      av3 = qq[arow_base3 + apack];
    }
    sa[sa_at + 96u] = av3;
    let kpack = t * 4u + wp;
    var k0 = 0u;
    if (wcol0 < dims.n && kpack < k4) {
      k0 = kq[krow_base0 + kpack];
    }
    sb[sb_at] = k0;
    var k1 = 0u;
    if (wcol1 < dims.n && kpack < k4) {
      k1 = kq[krow_base1 + kpack];
    }
    sb[sb_at + 32u] = k1;
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
    // n % 4 == 0 かつ ocq0 < n4 なので oc + 3 < n。
    // MUST: 半スケールは dequant 側で q / k の**両方**へ（列側はここで 1 度だけ）
    let oc = ocq0 * 4u;
    let ks = vec4<f32>(
      kscale[ksbase + oc],
      kscale[ksbase + oc + 1u],
      kscale[ksbase + oc + 2u],
      kscale[ksbase + oc + 3u],
    ) * dims.scale;
    // MUST: qs'·ks' を先に 1 つの f32 へ畳んでから f32(acc) に掛ける（bias が無いので fma は無い）
    if (orow0 < dims.m) {
      s[sbase + orow0 * n4 + ocq0] = vec4<f32>(acc0_0) * ((qscale[qsbase + orow0] * dims.scale) * ks);
    }
    if (orow1 < dims.m) {
      s[sbase + orow1 * n4 + ocq0] = vec4<f32>(acc1_0) * ((qscale[qsbase + orow1] * dims.scale) * ks);
    }
    if (orow2 < dims.m) {
      s[sbase + orow2 * n4 + ocq0] = vec4<f32>(acc2_0) * ((qscale[qsbase + orow2] * dims.scale) * ks);
    }
    if (orow3 < dims.m) {
      s[sbase + orow3 * n4 + ocq0] = vec4<f32>(acc3_0) * ((qscale[qsbase + orow3] * dims.scale) * ks);
    }
    if (orow4 < dims.m) {
      s[sbase + orow4 * n4 + ocq0] = vec4<f32>(acc4_0) * ((qscale[qsbase + orow4] * dims.scale) * ks);
    }
    if (orow5 < dims.m) {
      s[sbase + orow5 * n4 + ocq0] = vec4<f32>(acc5_0) * ((qscale[qsbase + orow5] * dims.scale) * ks);
    }
    if (orow6 < dims.m) {
      s[sbase + orow6 * n4 + ocq0] = vec4<f32>(acc6_0) * ((qscale[qsbase + orow6] * dims.scale) * ks);
    }
    if (orow7 < dims.m) {
      s[sbase + orow7 * n4 + ocq0] = vec4<f32>(acc7_0) * ((qscale[qsbase + orow7] * dims.scale) * ks);
    }
  }
  if (ocq1 < n4) {
    // n % 4 == 0 かつ ocq1 < n4 なので oc + 3 < n。
    // MUST: 半スケールは dequant 側で q / k の**両方**へ（列側はここで 1 度だけ）
    let oc = ocq1 * 4u;
    let ks = vec4<f32>(
      kscale[ksbase + oc],
      kscale[ksbase + oc + 1u],
      kscale[ksbase + oc + 2u],
      kscale[ksbase + oc + 3u],
    ) * dims.scale;
    // MUST: qs'·ks' を先に 1 つの f32 へ畳んでから f32(acc) に掛ける（bias が無いので fma は無い）
    if (orow0 < dims.m) {
      s[sbase + orow0 * n4 + ocq1] = vec4<f32>(acc0_1) * ((qscale[qsbase + orow0] * dims.scale) * ks);
    }
    if (orow1 < dims.m) {
      s[sbase + orow1 * n4 + ocq1] = vec4<f32>(acc1_1) * ((qscale[qsbase + orow1] * dims.scale) * ks);
    }
    if (orow2 < dims.m) {
      s[sbase + orow2 * n4 + ocq1] = vec4<f32>(acc2_1) * ((qscale[qsbase + orow2] * dims.scale) * ks);
    }
    if (orow3 < dims.m) {
      s[sbase + orow3 * n4 + ocq1] = vec4<f32>(acc3_1) * ((qscale[qsbase + orow3] * dims.scale) * ks);
    }
    if (orow4 < dims.m) {
      s[sbase + orow4 * n4 + ocq1] = vec4<f32>(acc4_1) * ((qscale[qsbase + orow4] * dims.scale) * ks);
    }
    if (orow5 < dims.m) {
      s[sbase + orow5 * n4 + ocq1] = vec4<f32>(acc5_1) * ((qscale[qsbase + orow5] * dims.scale) * ks);
    }
    if (orow6 < dims.m) {
      s[sbase + orow6 * n4 + ocq1] = vec4<f32>(acc6_1) * ((qscale[qsbase + orow6] * dims.scale) * ks);
    }
    if (orow7 < dims.m) {
      s[sbase + orow7 * n4 + ocq1] = vec4<f32>(acc7_1) * ((qscale[qsbase + orow7] * dims.scale) * ks);
    }
  }
}
