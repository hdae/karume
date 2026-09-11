// karume linear gemv (M=1: out[n] = x[k] · wᵀ[n,k] + bias[n], f32, 32 列 / wg, 語 4 本先読み)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
// 行頭が 16 B 整列なのは k % 4 == 0 から（適格判定が保証する）
@group(0) @binding(2) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;


@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  // 共有メモリも barrier も持たないので、端の workgroup は早期 return してよい
  if (col >= dims.n) {
    return;
  }
  let units = dims.k / 4u;
  let row_base = col * units;
  var acc = 0.0;
  var unit = 0u;
  // 先読みぶんの重み語を**先に**全て発行してから積和へ入る（メモリ並列度）。語の処理順は
  // 昇順のままなので、1 出力要素あたりの加算順序は先読み本数によらず同じ
  for (; unit + 4u <= units; unit = unit + 4u) {
    let unit0 = unit + 0u;
    let pw0 = w[row_base + unit0];
    let xq0 = unit0 * 1u;
    let unit1 = unit + 1u;
    let pw1 = w[row_base + unit1];
    let xq1 = unit1 * 1u;
    let unit2 = unit + 2u;
    let pw2 = w[row_base + unit2];
    let xq2 = unit2 * 1u;
    let unit3 = unit + 3u;
    let pw3 = w[row_base + unit3];
    let xq3 = unit3 * 1u;
    let xf0 = x[xq0];
    acc = acc + xf0.x * pw0.x;
    acc = acc + xf0.y * pw0.y;
    acc = acc + xf0.z * pw0.z;
    acc = acc + xf0.w * pw0.w;
    let xf1 = x[xq1];
    acc = acc + xf1.x * pw1.x;
    acc = acc + xf1.y * pw1.y;
    acc = acc + xf1.z * pw1.z;
    acc = acc + xf1.w * pw1.w;
    let xf2 = x[xq2];
    acc = acc + xf2.x * pw2.x;
    acc = acc + xf2.y * pw2.y;
    acc = acc + xf2.z * pw2.z;
    acc = acc + xf2.w * pw2.w;
    let xf3 = x[xq3];
    acc = acc + xf3.x * pw3.x;
    acc = acc + xf3.y * pw3.y;
    acc = acc + xf3.z * pw3.z;
    acc = acc + xf3.w * pw3.w;
  }
  // 端数の語（units % 4 本）— 上と同じ順序を 1 語ずつ辿る
  for (; unit < units; unit = unit + 1u) {
    let unitt = unit;
    let pwt = w[row_base + unitt];
    let xqt = unitt * 1u;
    let xft = x[xqt];
    acc = acc + xft.x * pwt.x;
    acc = acc + xft.y * pwt.y;
    acc = acc + xft.z * pwt.z;
    acc = acc + xft.w * pwt.w;
  }
  out[col] = acc + bias[col];
}
