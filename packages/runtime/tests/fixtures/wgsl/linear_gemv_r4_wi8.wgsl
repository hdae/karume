// karume linear gemv rows (M≥2: out[m,n] = x[m,k] · wᵀ[n,k] + bias[n], f32, 重み i8 格納, 32 列 / wg, 語 4 本先読み, 4 行 / スレッド)
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
fn linear_i8_word(pwa: vec4<u32>, xqa: u32, wscale_v: f32, initial: f32) -> f32 {
  var acc = initial;
    let ba_0 = unpack4xI8(pwa.x);
    let xaa_0 = x[xqa + 0u];
    acc = acc + xaa_0.x * (f32(ba_0.x) * wscale_v);
    acc = acc + xaa_0.y * (f32(ba_0.y) * wscale_v);
    acc = acc + xaa_0.z * (f32(ba_0.z) * wscale_v);
    acc = acc + xaa_0.w * (f32(ba_0.w) * wscale_v);
    let ba_1 = unpack4xI8(pwa.y);
    let xaa_1 = x[xqa + 1u];
    acc = acc + xaa_1.x * (f32(ba_1.x) * wscale_v);
    acc = acc + xaa_1.y * (f32(ba_1.y) * wscale_v);
    acc = acc + xaa_1.z * (f32(ba_1.z) * wscale_v);
    acc = acc + xaa_1.w * (f32(ba_1.w) * wscale_v);
    let ba_2 = unpack4xI8(pwa.z);
    let xaa_2 = x[xqa + 2u];
    acc = acc + xaa_2.x * (f32(ba_2.x) * wscale_v);
    acc = acc + xaa_2.y * (f32(ba_2.y) * wscale_v);
    acc = acc + xaa_2.z * (f32(ba_2.z) * wscale_v);
    acc = acc + xaa_2.w * (f32(ba_2.w) * wscale_v);
    let ba_3 = unpack4xI8(pwa.w);
    let xaa_3 = x[xqa + 3u];
    acc = acc + xaa_3.x * (f32(ba_3.x) * wscale_v);
    acc = acc + xaa_3.y * (f32(ba_3.y) * wscale_v);
    acc = acc + xaa_3.z * (f32(ba_3.z) * wscale_v);
    acc = acc + xaa_3.w * (f32(ba_3.w) * wscale_v);
  return acc;
}


@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  // 共有メモリも barrier も持たないので、端の workgroup は早期 return してよい
  if (col >= dims.n) {
    return;
  }
  let units = dims.k / 16u;
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
    acc0 = linear_i8_word(pw0, xr0 + xq0, wscale_v, acc0);
    acc1 = linear_i8_word(pw0, xr1 + xq0, wscale_v, acc1);
    acc2 = linear_i8_word(pw0, xr2 + xq0, wscale_v, acc2);
    acc3 = linear_i8_word(pw0, xr3 + xq0, wscale_v, acc3);
    acc0 = linear_i8_word(pw1, xr0 + xq1, wscale_v, acc0);
    acc1 = linear_i8_word(pw1, xr1 + xq1, wscale_v, acc1);
    acc2 = linear_i8_word(pw1, xr2 + xq1, wscale_v, acc2);
    acc3 = linear_i8_word(pw1, xr3 + xq1, wscale_v, acc3);
    acc0 = linear_i8_word(pw2, xr0 + xq2, wscale_v, acc0);
    acc1 = linear_i8_word(pw2, xr1 + xq2, wscale_v, acc1);
    acc2 = linear_i8_word(pw2, xr2 + xq2, wscale_v, acc2);
    acc3 = linear_i8_word(pw2, xr3 + xq2, wscale_v, acc3);
    acc0 = linear_i8_word(pw3, xr0 + xq3, wscale_v, acc0);
    acc1 = linear_i8_word(pw3, xr1 + xq3, wscale_v, acc1);
    acc2 = linear_i8_word(pw3, xr2 + xq3, wscale_v, acc2);
    acc3 = linear_i8_word(pw3, xr3 + xq3, wscale_v, acc3);
  }
  // 端数の語（units % 4 本）— 上と同じ順序を 1 語ずつ辿る
  for (; unit < units; unit = unit + 1u) {
    let unitt = unit;
    let pwt = w[row_base + unitt];
    let xqt = unitt * 4u;
    acc0 = linear_i8_word(pwt, xr0 + xqt, wscale_v, acc0);
    acc1 = linear_i8_word(pwt, xr1 + xqt, wscale_v, acc1);
    acc2 = linear_i8_word(pwt, xr2 + xqt, wscale_v, acc2);
    acc3 = linear_i8_word(pwt, xr3 + xqt, wscale_v, acc3);
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
