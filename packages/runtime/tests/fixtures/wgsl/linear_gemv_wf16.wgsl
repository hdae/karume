// karume linear gemv (M=1: out[n] = x[k] · wᵀ[n,k] + bias[n], f32, 重み f16 格納, 32 列 / wg, 語 4 本先読み)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
// 行頭が 16 B 整列なのは k % 8 == 0 から（適格判定が保証する）
@group(0) @binding(2) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;


@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  // 共有メモリも barrier も持たないので、端の workgroup は早期 return してよい
  if (col >= dims.n) {
    return;
  }
  let units = dims.k / 8u;
  let row_base = col * units;
  var acc = 0.0;
  var unit = 0u;
  // 先読みぶんの重み語を**先に**全て発行してから積和へ入る（メモリ並列度）。語の処理順は
  // 昇順のままなので、1 出力要素あたりの加算順序は先読み本数によらず同じ
  for (; unit + 4u <= units; unit = unit + 4u) {
    let unit0 = unit + 0u;
    let pw0 = w[row_base + unit0];
    let xq0 = unit0 * 2u;
    let unit1 = unit + 1u;
    let pw1 = w[row_base + unit1];
    let xq1 = unit1 * 2u;
    let unit2 = unit + 2u;
    let pw2 = w[row_base + unit2];
    let xq2 = unit2 * 2u;
    let unit3 = unit + 3u;
    let pw3 = w[row_base + unit3];
    let xq3 = unit3 * 2u;
    let h0_0 = unpack2x16float(pw0.x);
    let x0_0 = x[xq0 + 0u];
    acc = acc + x0_0.x * h0_0.x;
    acc = acc + x0_0.y * h0_0.y;
    let h0_1 = unpack2x16float(pw0.y);
    let x0_1 = x[xq0 + 0u];
    acc = acc + x0_1.z * h0_1.x;
    acc = acc + x0_1.w * h0_1.y;
    let h0_2 = unpack2x16float(pw0.z);
    let x0_2 = x[xq0 + 1u];
    acc = acc + x0_2.x * h0_2.x;
    acc = acc + x0_2.y * h0_2.y;
    let h0_3 = unpack2x16float(pw0.w);
    let x0_3 = x[xq0 + 1u];
    acc = acc + x0_3.z * h0_3.x;
    acc = acc + x0_3.w * h0_3.y;
    let h1_0 = unpack2x16float(pw1.x);
    let x1_0 = x[xq1 + 0u];
    acc = acc + x1_0.x * h1_0.x;
    acc = acc + x1_0.y * h1_0.y;
    let h1_1 = unpack2x16float(pw1.y);
    let x1_1 = x[xq1 + 0u];
    acc = acc + x1_1.z * h1_1.x;
    acc = acc + x1_1.w * h1_1.y;
    let h1_2 = unpack2x16float(pw1.z);
    let x1_2 = x[xq1 + 1u];
    acc = acc + x1_2.x * h1_2.x;
    acc = acc + x1_2.y * h1_2.y;
    let h1_3 = unpack2x16float(pw1.w);
    let x1_3 = x[xq1 + 1u];
    acc = acc + x1_3.z * h1_3.x;
    acc = acc + x1_3.w * h1_3.y;
    let h2_0 = unpack2x16float(pw2.x);
    let x2_0 = x[xq2 + 0u];
    acc = acc + x2_0.x * h2_0.x;
    acc = acc + x2_0.y * h2_0.y;
    let h2_1 = unpack2x16float(pw2.y);
    let x2_1 = x[xq2 + 0u];
    acc = acc + x2_1.z * h2_1.x;
    acc = acc + x2_1.w * h2_1.y;
    let h2_2 = unpack2x16float(pw2.z);
    let x2_2 = x[xq2 + 1u];
    acc = acc + x2_2.x * h2_2.x;
    acc = acc + x2_2.y * h2_2.y;
    let h2_3 = unpack2x16float(pw2.w);
    let x2_3 = x[xq2 + 1u];
    acc = acc + x2_3.z * h2_3.x;
    acc = acc + x2_3.w * h2_3.y;
    let h3_0 = unpack2x16float(pw3.x);
    let x3_0 = x[xq3 + 0u];
    acc = acc + x3_0.x * h3_0.x;
    acc = acc + x3_0.y * h3_0.y;
    let h3_1 = unpack2x16float(pw3.y);
    let x3_1 = x[xq3 + 0u];
    acc = acc + x3_1.z * h3_1.x;
    acc = acc + x3_1.w * h3_1.y;
    let h3_2 = unpack2x16float(pw3.z);
    let x3_2 = x[xq3 + 1u];
    acc = acc + x3_2.x * h3_2.x;
    acc = acc + x3_2.y * h3_2.y;
    let h3_3 = unpack2x16float(pw3.w);
    let x3_3 = x[xq3 + 1u];
    acc = acc + x3_3.z * h3_3.x;
    acc = acc + x3_3.w * h3_3.y;
  }
  // 端数の語（units % 4 本）— 上と同じ順序を 1 語ずつ辿る
  for (; unit < units; unit = unit + 1u) {
    let unitt = unit;
    let pwt = w[row_base + unitt];
    let xqt = unitt * 2u;
    let ht_0 = unpack2x16float(pwt.x);
    let xt_0 = x[xqt + 0u];
    acc = acc + xt_0.x * ht_0.x;
    acc = acc + xt_0.y * ht_0.y;
    let ht_1 = unpack2x16float(pwt.y);
    let xt_1 = x[xqt + 0u];
    acc = acc + xt_1.z * ht_1.x;
    acc = acc + xt_1.w * ht_1.y;
    let ht_2 = unpack2x16float(pwt.z);
    let xt_2 = x[xqt + 1u];
    acc = acc + xt_2.x * ht_2.x;
    acc = acc + xt_2.y * ht_2.y;
    let ht_3 = unpack2x16float(pwt.w);
    let xt_3 = x[xqt + 1u];
    acc = acc + xt_3.z * ht_3.x;
    acc = acc + xt_3.w * ht_3.y;
  }
  out[col] = acc + bias[col];
}
