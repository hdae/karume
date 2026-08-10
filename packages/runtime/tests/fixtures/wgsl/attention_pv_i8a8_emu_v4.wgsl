// karume attention_pv (O[b,m,d] = P[b,m,n] · v[b,n,d], P̃ = round(127·exp(S − m)) は非実体化・エミュ, v は per-column i8（Vᵀ 連続）, タイル 64x128 / 1 スレッド 8x8 / wg 16x8 / K 16 + vec4)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
// 適格条件で N % 4 == 0 なので、S の行頭は常に quad 境界に来る（P̃ の 4 詰めと同じ刻み）
@group(0) @binding(1) var<storage, read> s: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vq: array<u32>;
@group(0) @binding(3) var<storage, read> stats: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> vscale: array<f32>;

// 整数内積のエミュレーション（core WGSL のみ）。dot4I8Packed と**同じ整数**を返す —
// 整数加算は結合的かつ厳密で、中間値も最大 4·127² = 64,516 なので巻き戻らない
fn idot(a: u32, b: u32) -> i32 {
  return dot(unpack4xI8(a), unpack4xI8(b));
}

// 共有タイルは **[pack][行] / [pack][列]** 配置（①QK / linear の i8a8 と同じ — 内側の読みが
// 語ストライド 4 になりバンク衝突が 8-way から 2-way に減る）
var<workgroup> sa: array<u32, 256>;
var<workgroup> sb: array<u32, 512>;

@compute @workgroup_size(16, 8)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 16u + lid.x;
  // 縮約軸 N は S 側では quad 単位・vq 側では pack 単位（同じ 4 要素の刻み）
  let k4 = dims.k / 4u;
  let n4 = dims.n / 4u;
  // B と H を畳んだバッチ軸（workgroup 単位で一様 — 内側の workgroupBarrier が WGSL の
  // 一様性要件を満たすための前提）。MUST: S は quad 単位、vq は pack 単位、行統計は行あたり
  // 2 語、vscale は D 本、O は要素（v4 では quad）単位で数える — 単位を
  // 取り違えると B·H ≥ 2 で隣の head を読み書きする（例外なしの誤値で B=H=1 では出ない）
  let sbase = wid.z * dims.m * k4;
  let vbase = wid.z * dims.n * k4;
  let rbase = wid.z * dims.m;
  let vsbase = wid.z * dims.n;
  let obase = wid.z * dims.m * n4;
  // P̃ タイルの担当（64 行 × 4 pack を 128 スレッドで 2 巡）
  let ar = tid / 4u;
  let ap = tid % 4u;
  let sa_at = ap * 64u + ar;
  let arow0 = wid.y * 64u + ar;
  let arow_base0 = sbase + arow0 * k4;
  let arow1 = arow0 + 32u;
  let arow_base1 = sbase + arow1 * k4;
  // 行の最大 m（K タイルループ不変なので 1 度だけ引く）。端タイルでは arow >= M がありうるので
  // 添字を 0 へ倒す（読んだ値は arow < M の枝でしか使われない）
  let stat_at0 = select(0u, (rbase + arow0) * 2u, arow0 < dims.m);
  let row_max0 = stats[stat_at0];
  let stat_at1 = select(0u, (rbase + arow1) * 2u, arow1 < dims.m);
  let row_max1 = stats[stat_at1];
  // B タイルの担当（128 列 × 4 pack を 128 スレッドで 4 巡）。
  // 列は D。vq は [D,N] の N 連続（= パック方向）なので**転置が要らない**
  let wc = tid / 4u;
  let wp = tid % 4u;
  let sb_at = wp * 128u + wc;
  let wcol0 = wid.x * 128u + wc;
  let vrow_base0 = vbase + wcol0 * k4;
  let wcol1 = wcol0 + 32u;
  let vrow_base1 = vbase + wcol1 * k4;
  let wcol2 = wcol0 + 64u;
  let vrow_base2 = vbase + wcol2 * k4;
  let wcol3 = wcol0 + 96u;
  let vrow_base3 = vbase + wcol3 * k4;
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
    // 範囲外は 0 で埋める（qP = 0 は内積に寄与しないので K 端数でも結果は厳密）
    let apack = t * 4u + ap;
    var av0 = 0u;
    if (arow0 < dims.m && apack < k4) {
      let raw0 = s[arow_base0 + apack];
      // P̃ の scale は **1/127 固定**（行内 max が exp(0) = 1 で構造的 — amax は取らない）。
      // 除算が 1 つも無いので WGSL の 2.5 ULP 問題の外側にいる
      let p0 = vec4<f32>(
        round(exp(raw0.x - row_max0) * 127.0),
        round(exp(raw0.y - row_max0) * 127.0),
        round(exp(raw0.z - row_max0) * 127.0),
        round(exp(raw0.w - row_max0) * 127.0),
      );
      av0 = pack4xI8(vec4<i32>(p0));
    }
    sa[sa_at] = av0;
    var av1 = 0u;
    if (arow1 < dims.m && apack < k4) {
      let raw1 = s[arow_base1 + apack];
      // P̃ の scale は **1/127 固定**（行内 max が exp(0) = 1 で構造的 — amax は取らない）。
      // 除算が 1 つも無いので WGSL の 2.5 ULP 問題の外側にいる
      let p1 = vec4<f32>(
        round(exp(raw1.x - row_max1) * 127.0),
        round(exp(raw1.y - row_max1) * 127.0),
        round(exp(raw1.z - row_max1) * 127.0),
        round(exp(raw1.w - row_max1) * 127.0),
      );
      av1 = pack4xI8(vec4<i32>(p1));
    }
    sa[sa_at + 32u] = av1;
    let vpack = t * 4u + wp;
    var v0 = 0u;
    if (wcol0 < dims.n && vpack < k4) {
      v0 = vq[vrow_base0 + vpack];
    }
    sb[sb_at] = v0;
    var v1 = 0u;
    if (wcol1 < dims.n && vpack < k4) {
      v1 = vq[vrow_base1 + vpack];
    }
    sb[sb_at + 32u] = v1;
    var v2 = 0u;
    if (wcol2 < dims.n && vpack < k4) {
      v2 = vq[vrow_base2 + vpack];
    }
    sb[sb_at + 64u] = v2;
    var v3 = 0u;
    if (wcol3 < dims.n && vpack < k4) {
      v3 = vq[vrow_base3 + vpack];
    }
    sb[sb_at + 96u] = v3;
    workgroupBarrier();
    // 共有ロード 16 回（B の 8 語 + A の 8 語）で 64 個の整数内積 = 256 MAC。
    // 縮約は i32 の厳密加算なので**順序に依存しない**（f32 骨格と違い加算順は数値契約に無い）
    for (var p = 0u; p < 4u; p = p + 1u) {
      let bcol = p * 128u + lid.x * 8u;
      let b0 = sb[bcol];
      let b1 = sb[bcol + 1u];
      let b2 = sb[bcol + 2u];
      let b3 = sb[bcol + 3u];
      let b4 = sb[bcol + 4u];
      let b5 = sb[bcol + 5u];
      let b6 = sb[bcol + 6u];
      let b7 = sb[bcol + 7u];
      let arow_at = p * 64u + lid.y * 8u;
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
  let ocq0 = wid.x * 32u + lid.x * 2u;
  let ocq1 = ocq0 + 1u;
  let orow0 = wid.y * 64u + lid.y * 8u;
  let orow1 = orow0 + 1u;
  let orow2 = orow0 + 2u;
  let orow3 = orow0 + 3u;
  let orow4 = orow0 + 4u;
  let orow5 = orow0 + 5u;
  let orow6 = orow0 + 6u;
  let orow7 = orow0 + 7u;
  if (ocq0 < n4) {
    // D % 4 == 0 かつ ocq0 < n4 なので oc + 3 < D。
    // MUST: 列 scale は Vᵀ の**行** = (b, h, d)（= V の per-column scale）。行 scale の側と
    // 取り違えても例外は出ない
    let oc = ocq0 * 4u;
    let vs = vec4<f32>(
      vscale[vsbase + oc],
      vscale[vsbase + oc + 1u],
      vscale[vsbase + oc + 2u],
      vscale[vsbase + oc + 3u],
    );
    // MUST: prow·vs を先に 1 つの f32 へ畳んでから f32(acc) に掛ける（bias が無いので fma は無い）
    if (orow0 < dims.m) {
      let prow = stats[(rbase + orow0) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow0 * n4 + ocq0] = vec4<f32>(acc0_0) * (prow * vs);
    }
    if (orow1 < dims.m) {
      let prow = stats[(rbase + orow1) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow1 * n4 + ocq0] = vec4<f32>(acc1_0) * (prow * vs);
    }
    if (orow2 < dims.m) {
      let prow = stats[(rbase + orow2) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow2 * n4 + ocq0] = vec4<f32>(acc2_0) * (prow * vs);
    }
    if (orow3 < dims.m) {
      let prow = stats[(rbase + orow3) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow3 * n4 + ocq0] = vec4<f32>(acc3_0) * (prow * vs);
    }
    if (orow4 < dims.m) {
      let prow = stats[(rbase + orow4) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow4 * n4 + ocq0] = vec4<f32>(acc4_0) * (prow * vs);
    }
    if (orow5 < dims.m) {
      let prow = stats[(rbase + orow5) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow5 * n4 + ocq0] = vec4<f32>(acc5_0) * (prow * vs);
    }
    if (orow6 < dims.m) {
      let prow = stats[(rbase + orow6) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow6 * n4 + ocq0] = vec4<f32>(acc6_0) * (prow * vs);
    }
    if (orow7 < dims.m) {
      let prow = stats[(rbase + orow7) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow7 * n4 + ocq0] = vec4<f32>(acc7_0) * (prow * vs);
    }
  }
  if (ocq1 < n4) {
    // D % 4 == 0 かつ ocq1 < n4 なので oc + 3 < D。
    // MUST: 列 scale は Vᵀ の**行** = (b, h, d)（= V の per-column scale）。行 scale の側と
    // 取り違えても例外は出ない
    let oc = ocq1 * 4u;
    let vs = vec4<f32>(
      vscale[vsbase + oc],
      vscale[vsbase + oc + 1u],
      vscale[vsbase + oc + 2u],
      vscale[vsbase + oc + 3u],
    );
    // MUST: prow·vs を先に 1 つの f32 へ畳んでから f32(acc) に掛ける（bias が無いので fma は無い）
    if (orow0 < dims.m) {
      let prow = stats[(rbase + orow0) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow0 * n4 + ocq1] = vec4<f32>(acc0_1) * (prow * vs);
    }
    if (orow1 < dims.m) {
      let prow = stats[(rbase + orow1) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow1 * n4 + ocq1] = vec4<f32>(acc1_1) * (prow * vs);
    }
    if (orow2 < dims.m) {
      let prow = stats[(rbase + orow2) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow2 * n4 + ocq1] = vec4<f32>(acc2_1) * (prow * vs);
    }
    if (orow3 < dims.m) {
      let prow = stats[(rbase + orow3) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow3 * n4 + ocq1] = vec4<f32>(acc3_1) * (prow * vs);
    }
    if (orow4 < dims.m) {
      let prow = stats[(rbase + orow4) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow4 * n4 + ocq1] = vec4<f32>(acc4_1) * (prow * vs);
    }
    if (orow5 < dims.m) {
      let prow = stats[(rbase + orow5) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow5 * n4 + ocq1] = vec4<f32>(acc5_1) * (prow * vs);
    }
    if (orow6 < dims.m) {
      let prow = stats[(rbase + orow6) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow6 * n4 + ocq1] = vec4<f32>(acc6_1) * (prow * vs);
    }
    if (orow7 < dims.m) {
      let prow = stats[(rbase + orow7) * 2u + 1u] * 0.007874015748031496;
      o[obase + orow7 * n4 + ocq1] = vec4<f32>(acc7_1) * (prow * vs);
    }
  }
}
