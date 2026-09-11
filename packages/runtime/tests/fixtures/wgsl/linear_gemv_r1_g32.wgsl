// karume linear gemv rows (M≥2: out[m,n] = x[m,k] · wᵀ[n,k] + bias[n], f32, 重み i4 格納, 32 列 / wg, 語 4 本先読み, 1 行 / スレッド)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
// 行頭が 16 B 整列なのは k % 32 == 0 から（適格判定が保証する）
@group(0) @binding(2) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read> wscale: array<f32>;
fn linear_i4_word(pwa: vec4<u32>, xqa: u32, wsa: f32, initial: f32) -> f32 {
  var acc = initial;
    let ba_0 = unpack4xU8(pwa.x);
    let xaa_0 = x[xqa + 0u];
    let xba_0 = x[xqa + 1u];
    acc = acc + xaa_0.x * (f32(i32(ba_0.x & 0xFu) - 8) * wsa);
    acc = acc + xaa_0.y * (f32(i32(ba_0.x >> 4u) - 8) * wsa);
    acc = acc + xaa_0.z * (f32(i32(ba_0.y & 0xFu) - 8) * wsa);
    acc = acc + xaa_0.w * (f32(i32(ba_0.y >> 4u) - 8) * wsa);
    acc = acc + xba_0.x * (f32(i32(ba_0.z & 0xFu) - 8) * wsa);
    acc = acc + xba_0.y * (f32(i32(ba_0.z >> 4u) - 8) * wsa);
    acc = acc + xba_0.z * (f32(i32(ba_0.w & 0xFu) - 8) * wsa);
    acc = acc + xba_0.w * (f32(i32(ba_0.w >> 4u) - 8) * wsa);
    let ba_1 = unpack4xU8(pwa.y);
    let xaa_1 = x[xqa + 2u];
    let xba_1 = x[xqa + 3u];
    acc = acc + xaa_1.x * (f32(i32(ba_1.x & 0xFu) - 8) * wsa);
    acc = acc + xaa_1.y * (f32(i32(ba_1.x >> 4u) - 8) * wsa);
    acc = acc + xaa_1.z * (f32(i32(ba_1.y & 0xFu) - 8) * wsa);
    acc = acc + xaa_1.w * (f32(i32(ba_1.y >> 4u) - 8) * wsa);
    acc = acc + xba_1.x * (f32(i32(ba_1.z & 0xFu) - 8) * wsa);
    acc = acc + xba_1.y * (f32(i32(ba_1.z >> 4u) - 8) * wsa);
    acc = acc + xba_1.z * (f32(i32(ba_1.w & 0xFu) - 8) * wsa);
    acc = acc + xba_1.w * (f32(i32(ba_1.w >> 4u) - 8) * wsa);
    let ba_2 = unpack4xU8(pwa.z);
    let xaa_2 = x[xqa + 4u];
    let xba_2 = x[xqa + 5u];
    acc = acc + xaa_2.x * (f32(i32(ba_2.x & 0xFu) - 8) * wsa);
    acc = acc + xaa_2.y * (f32(i32(ba_2.x >> 4u) - 8) * wsa);
    acc = acc + xaa_2.z * (f32(i32(ba_2.y & 0xFu) - 8) * wsa);
    acc = acc + xaa_2.w * (f32(i32(ba_2.y >> 4u) - 8) * wsa);
    acc = acc + xba_2.x * (f32(i32(ba_2.z & 0xFu) - 8) * wsa);
    acc = acc + xba_2.y * (f32(i32(ba_2.z >> 4u) - 8) * wsa);
    acc = acc + xba_2.z * (f32(i32(ba_2.w & 0xFu) - 8) * wsa);
    acc = acc + xba_2.w * (f32(i32(ba_2.w >> 4u) - 8) * wsa);
    let ba_3 = unpack4xU8(pwa.w);
    let xaa_3 = x[xqa + 6u];
    let xba_3 = x[xqa + 7u];
    acc = acc + xaa_3.x * (f32(i32(ba_3.x & 0xFu) - 8) * wsa);
    acc = acc + xaa_3.y * (f32(i32(ba_3.x >> 4u) - 8) * wsa);
    acc = acc + xaa_3.z * (f32(i32(ba_3.y & 0xFu) - 8) * wsa);
    acc = acc + xaa_3.w * (f32(i32(ba_3.y >> 4u) - 8) * wsa);
    acc = acc + xba_3.x * (f32(i32(ba_3.z & 0xFu) - 8) * wsa);
    acc = acc + xba_3.y * (f32(i32(ba_3.z >> 4u) - 8) * wsa);
    acc = acc + xba_3.z * (f32(i32(ba_3.w & 0xFu) - 8) * wsa);
    acc = acc + xba_3.w * (f32(i32(ba_3.w >> 4u) - 8) * wsa);
  return acc;
}


@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  // 共有メモリも barrier も持たないので、端の workgroup は早期 return してよい
  if (col >= dims.n) {
    return;
  }
  let units = dims.k / 32u;
  let row_base = col * units;
  // 行ブロックの先頭行と、行ごとの x の先頭（quad 単位）。m を超える行は最終行を読み直す
  // （読みを範囲内に保つだけで、書き戻しは行数の内側だけ）
  let kq = dims.k / 4u;
  let row0 = gid.y * 1u;
  let xr0 = min(row0 + 0u, dims.m - 1u) * kq;
  let scale_base = col * (dims.k >> 5u);
  var acc0 = 0.0;
  var unit = 0u;
  // 先読みぶんの重み語を**先に**全て発行してから積和へ入る（メモリ並列度）。語の処理順は
  // 昇順のままなので、1 出力要素あたりの加算順序は先読み本数によらず同じ
  for (; unit + 4u <= units; unit = unit + 4u) {
    let unit0 = unit + 0u;
    let pw0 = w[row_base + unit0];
    let ws0 = wscale[scale_base + ((unit0 * 32u) >> 5u)];
    let xq0 = unit0 * 8u;
    let unit1 = unit + 1u;
    let pw1 = w[row_base + unit1];
    let ws1 = wscale[scale_base + ((unit1 * 32u) >> 5u)];
    let xq1 = unit1 * 8u;
    let unit2 = unit + 2u;
    let pw2 = w[row_base + unit2];
    let ws2 = wscale[scale_base + ((unit2 * 32u) >> 5u)];
    let xq2 = unit2 * 8u;
    let unit3 = unit + 3u;
    let pw3 = w[row_base + unit3];
    let ws3 = wscale[scale_base + ((unit3 * 32u) >> 5u)];
    let xq3 = unit3 * 8u;
    acc0 = linear_i4_word(pw0, xr0 + xq0, ws0, acc0);
    acc0 = linear_i4_word(pw1, xr0 + xq1, ws1, acc0);
    acc0 = linear_i4_word(pw2, xr0 + xq2, ws2, acc0);
    acc0 = linear_i4_word(pw3, xr0 + xq3, ws3, acc0);
  }
  // 端数の語（units % 4 本）— 上と同じ順序を 1 語ずつ辿る
  for (; unit < units; unit = unit + 1u) {
    let unitt = unit;
    let pwt = w[row_base + unitt];
    let wst = wscale[scale_base + ((unitt * 32u) >> 5u)];
    let xqt = unitt * 8u;
    acc0 = linear_i4_word(pwt, xr0 + xqt, wst, acc0);
  }
  if (row0 + 0u < dims.m) {
    out[(row0 + 0u) * dims.n + col] = acc0 + bias[col];
  }
}
