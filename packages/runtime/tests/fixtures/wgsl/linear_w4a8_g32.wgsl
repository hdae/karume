// karume linear (x[m,k] · wᵀ[k,n] + bias[n], 活性 per-token i8 × 重み i4 群 32 の整数内積, タイル 128x64 / 1 スレッド 8x8 / wg 8x16 / K 16)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> xq: array<u32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read> wscale: array<f32>;
@group(0) @binding(6) var<storage, read> xscale: array<f32>;

// 整数内積（WGSL 言語拡張 packed_4x8_integer_dot_product）。エミュ変種と**同じ整数**を返す
fn idot(a: u32, b: u32) -> i32 {
  return dot4I8Packed(a, b);
}

// i4 格納の整数レーン展開: 平坦要素 i..i+3（i は 4 の倍数）→ i8 レーン 4 詰めの u32。
// 1 語 = 8 要素なので 4 要素は語の上下 16bit のどちらか（(i >> 2) & 1 が選ぶ）。復元は
// q = u − 8 の**整数のまま**（scale は group 境界の f32 flush が掛ける — ADR 0069 決定 4）
fn i4lanes(i: u32) -> u32 {
  let half = (w[i >> 3u] >> (((i >> 2u) & 1u) * 16u)) & 0xFFFFu;
  let u = vec4<u32>(half, half >> 4u, half >> 8u, half >> 12u) & vec4<u32>(0xFu);
  return pack4xI8(vec4<i32>(u) - vec4<i32>(8));
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
  let wrow_base0 = wcol0 * dims.k;
  let wcol1 = wcol0 + 32u;
  let wrow_base1 = wcol1 * dims.k;
  // 出力の担当は**ループの外**（group 境界の flush が列ごとの scale を引く）
  let ocol = wid.x * 64u + lid.x * 8u;
  let orow0 = wid.y * 128u + lid.y * 8u;
  let orow1 = orow0 + 1u;
  let orow2 = orow0 + 2u;
  let orow3 = orow0 + 3u;
  let orow4 = orow0 + 4u;
  let orow5 = orow0 + 5u;
  let orow6 = orow0 + 6u;
  let orow7 = orow0 + 7u;
  // group scale は [n, k/g] の平坦。行内 group 数と列ごとの行頭はループ不変なので巻き上げる
  let groups = dims.k >> 5u;
  let wsb0 = ocol * groups;
  let wsb1 = (ocol + 1u) * groups;
  let wsb2 = (ocol + 2u) * groups;
  let wsb3 = (ocol + 3u) * groups;
  let wsb4 = (ocol + 4u) * groups;
  let wsb5 = (ocol + 5u) * groups;
  let wsb6 = (ocol + 6u) * groups;
  let wsb7 = (ocol + 7u) * groups;
  // f32 accumulator（group 境界でだけ書かれる — 丸めは k/g 回）
  var accf0_0 = vec4<f32>(0.0);
  var accf0_1 = vec4<f32>(0.0);
  var accf1_0 = vec4<f32>(0.0);
  var accf1_1 = vec4<f32>(0.0);
  var accf2_0 = vec4<f32>(0.0);
  var accf2_1 = vec4<f32>(0.0);
  var accf3_0 = vec4<f32>(0.0);
  var accf3_1 = vec4<f32>(0.0);
  var accf4_0 = vec4<f32>(0.0);
  var accf4_1 = vec4<f32>(0.0);
  var accf5_0 = vec4<f32>(0.0);
  var accf5_1 = vec4<f32>(0.0);
  var accf6_0 = vec4<f32>(0.0);
  var accf6_1 = vec4<f32>(0.0);
  var accf7_0 = vec4<f32>(0.0);
  var accf7_1 = vec4<f32>(0.0);
  // ループ条件は uniform（dims は uniform バッファ）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすために必要
  for (var gi = 0u; gi < groups; gi = gi + 1u) {
    // i32 accumulator は group ごとに初期化（|acci| ≤ g·127·8 — 門は LINEAR_W4A8_MAX_GROUP）
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
    // 1 group = K タイルちょうど 2 枚（g % tileK == 0 の門）。t は通し番号
    let gbase = gi * 2u;
    for (var gt = 0u; gt < 2u; gt = gt + 1u) {
      let t = gbase + gt;
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
    // i4 は 1 語 8 要素なので pack の添字は平坦要素（k % g == 0 ∧ g % tileK == 0 で K 端数なし）
    let welem = t * 16u + wp * 4u;
    var wv0 = 0u;
    if (wcol0 < dims.n) {
      wv0 = i4lanes(wrow_base0 + welem);
    }
    sb[sb_at] = wv0;
    var wv1 = 0u;
    if (wcol1 < dims.n) {
      wv1 = i4lanes(wrow_base1 + welem);
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
    // group 境界ちょうどで f32 へ flush（MUST: fma の単一丸め・xs は最後まで掛けない）
    let ws0 = vec4<f32>(wscale[wsb0 + gi], wscale[wsb1 + gi], wscale[wsb2 + gi], wscale[wsb3 + gi]);
    accf0_0 = fma(vec4<f32>(acc0_0), ws0, accf0_0);
    accf1_0 = fma(vec4<f32>(acc1_0), ws0, accf1_0);
    accf2_0 = fma(vec4<f32>(acc2_0), ws0, accf2_0);
    accf3_0 = fma(vec4<f32>(acc3_0), ws0, accf3_0);
    accf4_0 = fma(vec4<f32>(acc4_0), ws0, accf4_0);
    accf5_0 = fma(vec4<f32>(acc5_0), ws0, accf5_0);
    accf6_0 = fma(vec4<f32>(acc6_0), ws0, accf6_0);
    accf7_0 = fma(vec4<f32>(acc7_0), ws0, accf7_0);
    let ws1 = vec4<f32>(wscale[wsb4 + gi], wscale[wsb5 + gi], wscale[wsb6 + gi], wscale[wsb7 + gi]);
    accf0_1 = fma(vec4<f32>(acc0_1), ws1, accf0_1);
    accf1_1 = fma(vec4<f32>(acc1_1), ws1, accf1_1);
    accf2_1 = fma(vec4<f32>(acc2_1), ws1, accf2_1);
    accf3_1 = fma(vec4<f32>(acc3_1), ws1, accf3_1);
    accf4_1 = fma(vec4<f32>(acc4_1), ws1, accf4_1);
    accf5_1 = fma(vec4<f32>(acc5_1), ws1, accf5_1);
    accf6_1 = fma(vec4<f32>(acc6_1), ws1, accf6_1);
    accf7_1 = fma(vec4<f32>(acc7_1), ws1, accf7_1);
  }
  if (orow0 < dims.m) {
    let obase = orow0 * dims.n;
    // MUST: xs は最後の fma（単一丸め）へ回す — wscale は group 境界で畳み済み
    let xs = xscale[orow0];
    if (ocol < dims.n) {
      out[obase + ocol] = fma(accf0_0.x, xs, bias[ocol]);
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = fma(accf0_0.y, xs, bias[ocol + 1u]);
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = fma(accf0_0.z, xs, bias[ocol + 2u]);
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = fma(accf0_0.w, xs, bias[ocol + 3u]);
    }
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = fma(accf0_1.x, xs, bias[ocol + 4u]);
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = fma(accf0_1.y, xs, bias[ocol + 5u]);
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = fma(accf0_1.z, xs, bias[ocol + 6u]);
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = fma(accf0_1.w, xs, bias[ocol + 7u]);
    }
  }
  if (orow1 < dims.m) {
    let obase = orow1 * dims.n;
    // MUST: xs は最後の fma（単一丸め）へ回す — wscale は group 境界で畳み済み
    let xs = xscale[orow1];
    if (ocol < dims.n) {
      out[obase + ocol] = fma(accf1_0.x, xs, bias[ocol]);
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = fma(accf1_0.y, xs, bias[ocol + 1u]);
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = fma(accf1_0.z, xs, bias[ocol + 2u]);
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = fma(accf1_0.w, xs, bias[ocol + 3u]);
    }
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = fma(accf1_1.x, xs, bias[ocol + 4u]);
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = fma(accf1_1.y, xs, bias[ocol + 5u]);
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = fma(accf1_1.z, xs, bias[ocol + 6u]);
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = fma(accf1_1.w, xs, bias[ocol + 7u]);
    }
  }
  if (orow2 < dims.m) {
    let obase = orow2 * dims.n;
    // MUST: xs は最後の fma（単一丸め）へ回す — wscale は group 境界で畳み済み
    let xs = xscale[orow2];
    if (ocol < dims.n) {
      out[obase + ocol] = fma(accf2_0.x, xs, bias[ocol]);
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = fma(accf2_0.y, xs, bias[ocol + 1u]);
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = fma(accf2_0.z, xs, bias[ocol + 2u]);
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = fma(accf2_0.w, xs, bias[ocol + 3u]);
    }
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = fma(accf2_1.x, xs, bias[ocol + 4u]);
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = fma(accf2_1.y, xs, bias[ocol + 5u]);
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = fma(accf2_1.z, xs, bias[ocol + 6u]);
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = fma(accf2_1.w, xs, bias[ocol + 7u]);
    }
  }
  if (orow3 < dims.m) {
    let obase = orow3 * dims.n;
    // MUST: xs は最後の fma（単一丸め）へ回す — wscale は group 境界で畳み済み
    let xs = xscale[orow3];
    if (ocol < dims.n) {
      out[obase + ocol] = fma(accf3_0.x, xs, bias[ocol]);
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = fma(accf3_0.y, xs, bias[ocol + 1u]);
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = fma(accf3_0.z, xs, bias[ocol + 2u]);
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = fma(accf3_0.w, xs, bias[ocol + 3u]);
    }
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = fma(accf3_1.x, xs, bias[ocol + 4u]);
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = fma(accf3_1.y, xs, bias[ocol + 5u]);
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = fma(accf3_1.z, xs, bias[ocol + 6u]);
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = fma(accf3_1.w, xs, bias[ocol + 7u]);
    }
  }
  if (orow4 < dims.m) {
    let obase = orow4 * dims.n;
    // MUST: xs は最後の fma（単一丸め）へ回す — wscale は group 境界で畳み済み
    let xs = xscale[orow4];
    if (ocol < dims.n) {
      out[obase + ocol] = fma(accf4_0.x, xs, bias[ocol]);
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = fma(accf4_0.y, xs, bias[ocol + 1u]);
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = fma(accf4_0.z, xs, bias[ocol + 2u]);
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = fma(accf4_0.w, xs, bias[ocol + 3u]);
    }
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = fma(accf4_1.x, xs, bias[ocol + 4u]);
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = fma(accf4_1.y, xs, bias[ocol + 5u]);
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = fma(accf4_1.z, xs, bias[ocol + 6u]);
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = fma(accf4_1.w, xs, bias[ocol + 7u]);
    }
  }
  if (orow5 < dims.m) {
    let obase = orow5 * dims.n;
    // MUST: xs は最後の fma（単一丸め）へ回す — wscale は group 境界で畳み済み
    let xs = xscale[orow5];
    if (ocol < dims.n) {
      out[obase + ocol] = fma(accf5_0.x, xs, bias[ocol]);
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = fma(accf5_0.y, xs, bias[ocol + 1u]);
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = fma(accf5_0.z, xs, bias[ocol + 2u]);
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = fma(accf5_0.w, xs, bias[ocol + 3u]);
    }
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = fma(accf5_1.x, xs, bias[ocol + 4u]);
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = fma(accf5_1.y, xs, bias[ocol + 5u]);
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = fma(accf5_1.z, xs, bias[ocol + 6u]);
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = fma(accf5_1.w, xs, bias[ocol + 7u]);
    }
  }
  if (orow6 < dims.m) {
    let obase = orow6 * dims.n;
    // MUST: xs は最後の fma（単一丸め）へ回す — wscale は group 境界で畳み済み
    let xs = xscale[orow6];
    if (ocol < dims.n) {
      out[obase + ocol] = fma(accf6_0.x, xs, bias[ocol]);
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = fma(accf6_0.y, xs, bias[ocol + 1u]);
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = fma(accf6_0.z, xs, bias[ocol + 2u]);
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = fma(accf6_0.w, xs, bias[ocol + 3u]);
    }
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = fma(accf6_1.x, xs, bias[ocol + 4u]);
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = fma(accf6_1.y, xs, bias[ocol + 5u]);
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = fma(accf6_1.z, xs, bias[ocol + 6u]);
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = fma(accf6_1.w, xs, bias[ocol + 7u]);
    }
  }
  if (orow7 < dims.m) {
    let obase = orow7 * dims.n;
    // MUST: xs は最後の fma（単一丸め）へ回す — wscale は group 境界で畳み済み
    let xs = xscale[orow7];
    if (ocol < dims.n) {
      out[obase + ocol] = fma(accf7_0.x, xs, bias[ocol]);
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = fma(accf7_0.y, xs, bias[ocol + 1u]);
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = fma(accf7_0.z, xs, bias[ocol + 2u]);
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = fma(accf7_0.w, xs, bias[ocol + 3u]);
    }
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = fma(accf7_1.x, xs, bias[ocol + 4u]);
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = fma(accf7_1.y, xs, bias[ocol + 5u]);
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = fma(accf7_1.z, xs, bias[ocol + 6u]);
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = fma(accf7_1.w, xs, bias[ocol + 7u]);
    }
  }
}
