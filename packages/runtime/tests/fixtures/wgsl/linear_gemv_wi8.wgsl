// karume linear gemv (M=1: out[n] = x[k] · wᵀ[n,k] + bias[n], f32, 重み i8 格納, 32 列 / wg, 語 4 本先読み)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
// 行頭が 16 B 整列なのは k % 16 == 0 から（適格判定が保証する）
@group(0) @binding(2) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read> wscale: array<f32>;

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  // 共有メモリも barrier も持たないので、端の workgroup は早期 return してよい
  if (col >= dims.n) {
    return;
  }
  let units = dims.k / 16u;
  let row_base = col * units;
  // 出力チャネルの scale はループ不変 — 重みの要素ごとに引き直さない（ADR 0019）
  let wscale_v = wscale[col];
  var acc = 0.0;
  var unit = 0u;
  // 先読みぶんの重み語を**先に**全て発行してから積和へ入る（メモリ並列度）。語の処理順は
  // 昇順のままなので、1 出力要素あたりの加算順序は先読み本数によらず同じ
  for (; unit + 4u <= units; unit = unit + 4u) {
    let unit0 = unit + 0u;
    let pw0 = w[row_base + unit0];
    let xq0 = unit0 * 4u;
    let unit1 = unit + 1u;
    let pw1 = w[row_base + unit1];
    let xq1 = unit1 * 4u;
    let unit2 = unit + 2u;
    let pw2 = w[row_base + unit2];
    let xq2 = unit2 * 4u;
    let unit3 = unit + 3u;
    let pw3 = w[row_base + unit3];
    let xq3 = unit3 * 4u;
    let b0_0 = unpack4xI8(pw0.x);
    let xa0_0 = x[xq0 + 0u];
    acc = acc + xa0_0.x * (f32(b0_0.x) * wscale_v);
    acc = acc + xa0_0.y * (f32(b0_0.y) * wscale_v);
    acc = acc + xa0_0.z * (f32(b0_0.z) * wscale_v);
    acc = acc + xa0_0.w * (f32(b0_0.w) * wscale_v);
    let b0_1 = unpack4xI8(pw0.y);
    let xa0_1 = x[xq0 + 1u];
    acc = acc + xa0_1.x * (f32(b0_1.x) * wscale_v);
    acc = acc + xa0_1.y * (f32(b0_1.y) * wscale_v);
    acc = acc + xa0_1.z * (f32(b0_1.z) * wscale_v);
    acc = acc + xa0_1.w * (f32(b0_1.w) * wscale_v);
    let b0_2 = unpack4xI8(pw0.z);
    let xa0_2 = x[xq0 + 2u];
    acc = acc + xa0_2.x * (f32(b0_2.x) * wscale_v);
    acc = acc + xa0_2.y * (f32(b0_2.y) * wscale_v);
    acc = acc + xa0_2.z * (f32(b0_2.z) * wscale_v);
    acc = acc + xa0_2.w * (f32(b0_2.w) * wscale_v);
    let b0_3 = unpack4xI8(pw0.w);
    let xa0_3 = x[xq0 + 3u];
    acc = acc + xa0_3.x * (f32(b0_3.x) * wscale_v);
    acc = acc + xa0_3.y * (f32(b0_3.y) * wscale_v);
    acc = acc + xa0_3.z * (f32(b0_3.z) * wscale_v);
    acc = acc + xa0_3.w * (f32(b0_3.w) * wscale_v);
    let b1_0 = unpack4xI8(pw1.x);
    let xa1_0 = x[xq1 + 0u];
    acc = acc + xa1_0.x * (f32(b1_0.x) * wscale_v);
    acc = acc + xa1_0.y * (f32(b1_0.y) * wscale_v);
    acc = acc + xa1_0.z * (f32(b1_0.z) * wscale_v);
    acc = acc + xa1_0.w * (f32(b1_0.w) * wscale_v);
    let b1_1 = unpack4xI8(pw1.y);
    let xa1_1 = x[xq1 + 1u];
    acc = acc + xa1_1.x * (f32(b1_1.x) * wscale_v);
    acc = acc + xa1_1.y * (f32(b1_1.y) * wscale_v);
    acc = acc + xa1_1.z * (f32(b1_1.z) * wscale_v);
    acc = acc + xa1_1.w * (f32(b1_1.w) * wscale_v);
    let b1_2 = unpack4xI8(pw1.z);
    let xa1_2 = x[xq1 + 2u];
    acc = acc + xa1_2.x * (f32(b1_2.x) * wscale_v);
    acc = acc + xa1_2.y * (f32(b1_2.y) * wscale_v);
    acc = acc + xa1_2.z * (f32(b1_2.z) * wscale_v);
    acc = acc + xa1_2.w * (f32(b1_2.w) * wscale_v);
    let b1_3 = unpack4xI8(pw1.w);
    let xa1_3 = x[xq1 + 3u];
    acc = acc + xa1_3.x * (f32(b1_3.x) * wscale_v);
    acc = acc + xa1_3.y * (f32(b1_3.y) * wscale_v);
    acc = acc + xa1_3.z * (f32(b1_3.z) * wscale_v);
    acc = acc + xa1_3.w * (f32(b1_3.w) * wscale_v);
    let b2_0 = unpack4xI8(pw2.x);
    let xa2_0 = x[xq2 + 0u];
    acc = acc + xa2_0.x * (f32(b2_0.x) * wscale_v);
    acc = acc + xa2_0.y * (f32(b2_0.y) * wscale_v);
    acc = acc + xa2_0.z * (f32(b2_0.z) * wscale_v);
    acc = acc + xa2_0.w * (f32(b2_0.w) * wscale_v);
    let b2_1 = unpack4xI8(pw2.y);
    let xa2_1 = x[xq2 + 1u];
    acc = acc + xa2_1.x * (f32(b2_1.x) * wscale_v);
    acc = acc + xa2_1.y * (f32(b2_1.y) * wscale_v);
    acc = acc + xa2_1.z * (f32(b2_1.z) * wscale_v);
    acc = acc + xa2_1.w * (f32(b2_1.w) * wscale_v);
    let b2_2 = unpack4xI8(pw2.z);
    let xa2_2 = x[xq2 + 2u];
    acc = acc + xa2_2.x * (f32(b2_2.x) * wscale_v);
    acc = acc + xa2_2.y * (f32(b2_2.y) * wscale_v);
    acc = acc + xa2_2.z * (f32(b2_2.z) * wscale_v);
    acc = acc + xa2_2.w * (f32(b2_2.w) * wscale_v);
    let b2_3 = unpack4xI8(pw2.w);
    let xa2_3 = x[xq2 + 3u];
    acc = acc + xa2_3.x * (f32(b2_3.x) * wscale_v);
    acc = acc + xa2_3.y * (f32(b2_3.y) * wscale_v);
    acc = acc + xa2_3.z * (f32(b2_3.z) * wscale_v);
    acc = acc + xa2_3.w * (f32(b2_3.w) * wscale_v);
    let b3_0 = unpack4xI8(pw3.x);
    let xa3_0 = x[xq3 + 0u];
    acc = acc + xa3_0.x * (f32(b3_0.x) * wscale_v);
    acc = acc + xa3_0.y * (f32(b3_0.y) * wscale_v);
    acc = acc + xa3_0.z * (f32(b3_0.z) * wscale_v);
    acc = acc + xa3_0.w * (f32(b3_0.w) * wscale_v);
    let b3_1 = unpack4xI8(pw3.y);
    let xa3_1 = x[xq3 + 1u];
    acc = acc + xa3_1.x * (f32(b3_1.x) * wscale_v);
    acc = acc + xa3_1.y * (f32(b3_1.y) * wscale_v);
    acc = acc + xa3_1.z * (f32(b3_1.z) * wscale_v);
    acc = acc + xa3_1.w * (f32(b3_1.w) * wscale_v);
    let b3_2 = unpack4xI8(pw3.z);
    let xa3_2 = x[xq3 + 2u];
    acc = acc + xa3_2.x * (f32(b3_2.x) * wscale_v);
    acc = acc + xa3_2.y * (f32(b3_2.y) * wscale_v);
    acc = acc + xa3_2.z * (f32(b3_2.z) * wscale_v);
    acc = acc + xa3_2.w * (f32(b3_2.w) * wscale_v);
    let b3_3 = unpack4xI8(pw3.w);
    let xa3_3 = x[xq3 + 3u];
    acc = acc + xa3_3.x * (f32(b3_3.x) * wscale_v);
    acc = acc + xa3_3.y * (f32(b3_3.y) * wscale_v);
    acc = acc + xa3_3.z * (f32(b3_3.z) * wscale_v);
    acc = acc + xa3_3.w * (f32(b3_3.w) * wscale_v);
  }
  // 端数の語（units % 4 本）— 上と同じ順序を 1 語ずつ辿る
  for (; unit < units; unit = unit + 1u) {
    let unitt = unit;
    let pwt = w[row_base + unitt];
    let xqt = unitt * 4u;
    let bt_0 = unpack4xI8(pwt.x);
    let xat_0 = x[xqt + 0u];
    acc = acc + xat_0.x * (f32(bt_0.x) * wscale_v);
    acc = acc + xat_0.y * (f32(bt_0.y) * wscale_v);
    acc = acc + xat_0.z * (f32(bt_0.z) * wscale_v);
    acc = acc + xat_0.w * (f32(bt_0.w) * wscale_v);
    let bt_1 = unpack4xI8(pwt.y);
    let xat_1 = x[xqt + 1u];
    acc = acc + xat_1.x * (f32(bt_1.x) * wscale_v);
    acc = acc + xat_1.y * (f32(bt_1.y) * wscale_v);
    acc = acc + xat_1.z * (f32(bt_1.z) * wscale_v);
    acc = acc + xat_1.w * (f32(bt_1.w) * wscale_v);
    let bt_2 = unpack4xI8(pwt.z);
    let xat_2 = x[xqt + 2u];
    acc = acc + xat_2.x * (f32(bt_2.x) * wscale_v);
    acc = acc + xat_2.y * (f32(bt_2.y) * wscale_v);
    acc = acc + xat_2.z * (f32(bt_2.z) * wscale_v);
    acc = acc + xat_2.w * (f32(bt_2.w) * wscale_v);
    let bt_3 = unpack4xI8(pwt.w);
    let xat_3 = x[xqt + 3u];
    acc = acc + xat_3.x * (f32(bt_3.x) * wscale_v);
    acc = acc + xat_3.y * (f32(bt_3.y) * wscale_v);
    acc = acc + xat_3.z * (f32(bt_3.z) * wscale_v);
    acc = acc + xat_3.w * (f32(bt_3.w) * wscale_v);
  }
  out[col] = acc + bias[col];
}
