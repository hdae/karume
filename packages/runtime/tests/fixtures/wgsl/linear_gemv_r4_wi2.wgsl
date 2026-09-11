// karume linear gemv rows (M≥2: out[m,n] = x[m,k] · wᵀ[n,k] + bias[n], f32, 重み i2 格納, 32 列 / wg, 語 4 本先読み, 4 行 / スレッド)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
// 行頭が 16 B 整列なのは k % 64 == 0 から（適格判定が保証する）
@group(0) @binding(2) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read> wscale: array<f32>;
fn linear_i2_word(pwa: vec4<u32>, xqa: u32, wscale_v: f32, initial: f32) -> f32 {
  var acc = initial;
    let qa_0 = (pwa.x >> 0u) & 255u;
    let da_0 = vec4<f32>(vec4<i32>(vec4<u32>(qa_0, qa_0 >> 2u, qa_0 >> 4u, qa_0 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xa_0_0 = x[xqa + 0u];
    acc = acc + xa_0_0.x * da_0.x;
    acc = acc + xa_0_0.y * da_0.y;
    acc = acc + xa_0_0.z * da_0.z;
    acc = acc + xa_0_0.w * da_0.w;
    let qa_1 = (pwa.x >> 8u) & 255u;
    let da_1 = vec4<f32>(vec4<i32>(vec4<u32>(qa_1, qa_1 >> 2u, qa_1 >> 4u, qa_1 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xa_1_0 = x[xqa + 1u];
    acc = acc + xa_1_0.x * da_1.x;
    acc = acc + xa_1_0.y * da_1.y;
    acc = acc + xa_1_0.z * da_1.z;
    acc = acc + xa_1_0.w * da_1.w;
    let qa_2 = (pwa.x >> 16u) & 255u;
    let da_2 = vec4<f32>(vec4<i32>(vec4<u32>(qa_2, qa_2 >> 2u, qa_2 >> 4u, qa_2 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xa_2_0 = x[xqa + 2u];
    acc = acc + xa_2_0.x * da_2.x;
    acc = acc + xa_2_0.y * da_2.y;
    acc = acc + xa_2_0.z * da_2.z;
    acc = acc + xa_2_0.w * da_2.w;
    let qa_3 = (pwa.x >> 24u) & 255u;
    let da_3 = vec4<f32>(vec4<i32>(vec4<u32>(qa_3, qa_3 >> 2u, qa_3 >> 4u, qa_3 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xa_3_0 = x[xqa + 3u];
    acc = acc + xa_3_0.x * da_3.x;
    acc = acc + xa_3_0.y * da_3.y;
    acc = acc + xa_3_0.z * da_3.z;
    acc = acc + xa_3_0.w * da_3.w;
    let qa_4 = (pwa.y >> 0u) & 255u;
    let da_4 = vec4<f32>(vec4<i32>(vec4<u32>(qa_4, qa_4 >> 2u, qa_4 >> 4u, qa_4 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xa_4_0 = x[xqa + 4u];
    acc = acc + xa_4_0.x * da_4.x;
    acc = acc + xa_4_0.y * da_4.y;
    acc = acc + xa_4_0.z * da_4.z;
    acc = acc + xa_4_0.w * da_4.w;
    let qa_5 = (pwa.y >> 8u) & 255u;
    let da_5 = vec4<f32>(vec4<i32>(vec4<u32>(qa_5, qa_5 >> 2u, qa_5 >> 4u, qa_5 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xa_5_0 = x[xqa + 5u];
    acc = acc + xa_5_0.x * da_5.x;
    acc = acc + xa_5_0.y * da_5.y;
    acc = acc + xa_5_0.z * da_5.z;
    acc = acc + xa_5_0.w * da_5.w;
    let qa_6 = (pwa.y >> 16u) & 255u;
    let da_6 = vec4<f32>(vec4<i32>(vec4<u32>(qa_6, qa_6 >> 2u, qa_6 >> 4u, qa_6 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xa_6_0 = x[xqa + 6u];
    acc = acc + xa_6_0.x * da_6.x;
    acc = acc + xa_6_0.y * da_6.y;
    acc = acc + xa_6_0.z * da_6.z;
    acc = acc + xa_6_0.w * da_6.w;
    let qa_7 = (pwa.y >> 24u) & 255u;
    let da_7 = vec4<f32>(vec4<i32>(vec4<u32>(qa_7, qa_7 >> 2u, qa_7 >> 4u, qa_7 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xa_7_0 = x[xqa + 7u];
    acc = acc + xa_7_0.x * da_7.x;
    acc = acc + xa_7_0.y * da_7.y;
    acc = acc + xa_7_0.z * da_7.z;
    acc = acc + xa_7_0.w * da_7.w;
    let qa_8 = (pwa.z >> 0u) & 255u;
    let da_8 = vec4<f32>(vec4<i32>(vec4<u32>(qa_8, qa_8 >> 2u, qa_8 >> 4u, qa_8 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xa_8_0 = x[xqa + 8u];
    acc = acc + xa_8_0.x * da_8.x;
    acc = acc + xa_8_0.y * da_8.y;
    acc = acc + xa_8_0.z * da_8.z;
    acc = acc + xa_8_0.w * da_8.w;
    let qa_9 = (pwa.z >> 8u) & 255u;
    let da_9 = vec4<f32>(vec4<i32>(vec4<u32>(qa_9, qa_9 >> 2u, qa_9 >> 4u, qa_9 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xa_9_0 = x[xqa + 9u];
    acc = acc + xa_9_0.x * da_9.x;
    acc = acc + xa_9_0.y * da_9.y;
    acc = acc + xa_9_0.z * da_9.z;
    acc = acc + xa_9_0.w * da_9.w;
    let qa_10 = (pwa.z >> 16u) & 255u;
    let da_10 = vec4<f32>(vec4<i32>(vec4<u32>(qa_10, qa_10 >> 2u, qa_10 >> 4u, qa_10 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xa_10_0 = x[xqa + 10u];
    acc = acc + xa_10_0.x * da_10.x;
    acc = acc + xa_10_0.y * da_10.y;
    acc = acc + xa_10_0.z * da_10.z;
    acc = acc + xa_10_0.w * da_10.w;
    let qa_11 = (pwa.z >> 24u) & 255u;
    let da_11 = vec4<f32>(vec4<i32>(vec4<u32>(qa_11, qa_11 >> 2u, qa_11 >> 4u, qa_11 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xa_11_0 = x[xqa + 11u];
    acc = acc + xa_11_0.x * da_11.x;
    acc = acc + xa_11_0.y * da_11.y;
    acc = acc + xa_11_0.z * da_11.z;
    acc = acc + xa_11_0.w * da_11.w;
    let qa_12 = (pwa.w >> 0u) & 255u;
    let da_12 = vec4<f32>(vec4<i32>(vec4<u32>(qa_12, qa_12 >> 2u, qa_12 >> 4u, qa_12 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xa_12_0 = x[xqa + 12u];
    acc = acc + xa_12_0.x * da_12.x;
    acc = acc + xa_12_0.y * da_12.y;
    acc = acc + xa_12_0.z * da_12.z;
    acc = acc + xa_12_0.w * da_12.w;
    let qa_13 = (pwa.w >> 8u) & 255u;
    let da_13 = vec4<f32>(vec4<i32>(vec4<u32>(qa_13, qa_13 >> 2u, qa_13 >> 4u, qa_13 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xa_13_0 = x[xqa + 13u];
    acc = acc + xa_13_0.x * da_13.x;
    acc = acc + xa_13_0.y * da_13.y;
    acc = acc + xa_13_0.z * da_13.z;
    acc = acc + xa_13_0.w * da_13.w;
    let qa_14 = (pwa.w >> 16u) & 255u;
    let da_14 = vec4<f32>(vec4<i32>(vec4<u32>(qa_14, qa_14 >> 2u, qa_14 >> 4u, qa_14 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xa_14_0 = x[xqa + 14u];
    acc = acc + xa_14_0.x * da_14.x;
    acc = acc + xa_14_0.y * da_14.y;
    acc = acc + xa_14_0.z * da_14.z;
    acc = acc + xa_14_0.w * da_14.w;
    let qa_15 = (pwa.w >> 24u) & 255u;
    let da_15 = vec4<f32>(vec4<i32>(vec4<u32>(qa_15, qa_15 >> 2u, qa_15 >> 4u, qa_15 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xa_15_0 = x[xqa + 15u];
    acc = acc + xa_15_0.x * da_15.x;
    acc = acc + xa_15_0.y * da_15.y;
    acc = acc + xa_15_0.z * da_15.z;
    acc = acc + xa_15_0.w * da_15.w;
  return acc;
}


@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  // 共有メモリも barrier も持たないので、端の workgroup は早期 return してよい
  if (col >= dims.n) {
    return;
  }
  let units = dims.k / 64u;
  let row_base = col * units;
  // 行ブロックの先頭行と、行ごとの x の先頭（quad 単位）。m を超える行は最終行を読み直す
  // （読みを範囲内に保つだけで、書き戻しは行数の内側だけ）
  let kq = dims.k / 4u;
  let row0 = gid.y * 4u;
  let xr0 = min(row0 + 0u, dims.m - 1u) * kq;
  let xr1 = min(row0 + 1u, dims.m - 1u) * kq;
  let xr2 = min(row0 + 2u, dims.m - 1u) * kq;
  let xr3 = min(row0 + 3u, dims.m - 1u) * kq;
  // 出力チャネルの scale はループ不変 — 重みの要素ごとに引き直さない（ADR 0019）
  let wscale_v = wscale[col];
  var acc0 = 0.0;
  var acc1 = 0.0;
  var acc2 = 0.0;
  var acc3 = 0.0;
  var unit = 0u;
  // 先読みぶんの重み語を**先に**全て発行してから積和へ入る（メモリ並列度）。語の処理順は
  // 昇順のままなので、1 出力要素あたりの加算順序は先読み本数によらず同じ
  for (; unit + 4u <= units; unit = unit + 4u) {
    let unit0 = unit + 0u;
    let pw0 = w[row_base + unit0];
    let xq0 = unit0 * 16u;
    let unit1 = unit + 1u;
    let pw1 = w[row_base + unit1];
    let xq1 = unit1 * 16u;
    let unit2 = unit + 2u;
    let pw2 = w[row_base + unit2];
    let xq2 = unit2 * 16u;
    let unit3 = unit + 3u;
    let pw3 = w[row_base + unit3];
    let xq3 = unit3 * 16u;
    acc0 = linear_i2_word(pw0, xr0 + xq0, wscale_v, acc0);
    acc1 = linear_i2_word(pw0, xr1 + xq0, wscale_v, acc1);
    acc2 = linear_i2_word(pw0, xr2 + xq0, wscale_v, acc2);
    acc3 = linear_i2_word(pw0, xr3 + xq0, wscale_v, acc3);
    acc0 = linear_i2_word(pw1, xr0 + xq1, wscale_v, acc0);
    acc1 = linear_i2_word(pw1, xr1 + xq1, wscale_v, acc1);
    acc2 = linear_i2_word(pw1, xr2 + xq1, wscale_v, acc2);
    acc3 = linear_i2_word(pw1, xr3 + xq1, wscale_v, acc3);
    acc0 = linear_i2_word(pw2, xr0 + xq2, wscale_v, acc0);
    acc1 = linear_i2_word(pw2, xr1 + xq2, wscale_v, acc1);
    acc2 = linear_i2_word(pw2, xr2 + xq2, wscale_v, acc2);
    acc3 = linear_i2_word(pw2, xr3 + xq2, wscale_v, acc3);
    acc0 = linear_i2_word(pw3, xr0 + xq3, wscale_v, acc0);
    acc1 = linear_i2_word(pw3, xr1 + xq3, wscale_v, acc1);
    acc2 = linear_i2_word(pw3, xr2 + xq3, wscale_v, acc2);
    acc3 = linear_i2_word(pw3, xr3 + xq3, wscale_v, acc3);
  }
  // 端数の語（units % 4 本）— 上と同じ順序を 1 語ずつ辿る
  for (; unit < units; unit = unit + 1u) {
    let unitt = unit;
    let pwt = w[row_base + unitt];
    let xqt = unitt * 16u;
    acc0 = linear_i2_word(pwt, xr0 + xqt, wscale_v, acc0);
    acc1 = linear_i2_word(pwt, xr1 + xqt, wscale_v, acc1);
    acc2 = linear_i2_word(pwt, xr2 + xqt, wscale_v, acc2);
    acc3 = linear_i2_word(pwt, xr3 + xqt, wscale_v, acc3);
  }
  if (row0 + 0u < dims.m) {
    out[(row0 + 0u) * dims.n + col] = acc0 + bias[col];
  }
  if (row0 + 1u < dims.m) {
    out[(row0 + 1u) * dims.n + col] = acc1 + bias[col];
  }
  if (row0 + 2u < dims.m) {
    out[(row0 + 2u) * dims.n + col] = acc2 + bias[col];
  }
  if (row0 + 3u < dims.m) {
    out[(row0 + 3u) * dims.n + col] = acc3 + bias[col];
  }
}
