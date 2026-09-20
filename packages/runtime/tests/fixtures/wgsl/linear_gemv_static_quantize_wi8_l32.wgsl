// karume linear gemv K parallel (i8, 32 lanes/output)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
  rounding_mask: u32,
  srq: array<vec4<u32>, 65>,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
// 行頭が 16 B 整列なのは k % 16 == 0 から（適格判定が保証する）
@group(0) @binding(2) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<u32>;
@group(0) @binding(5) var<storage, read> wscale: array<f32>;
fn srqWord(index: u32) -> u32 { return dims.srq[index >> 2u][index & 3u]; }
fn quantize(bits: u32) -> u32 {
  if (srqWord(258u) != 0u) { return bits; }
  let magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u) { return bits | 0x00400000u; }
  var lo = 0u;
  lo += select(0u, 64u, magnitude >= srqWord(lo + 64u));
  lo += select(0u, 32u, magnitude >= srqWord(lo + 32u));
  lo += select(0u, 16u, magnitude >= srqWord(lo + 16u));
  lo += select(0u, 8u, magnitude >= srqWord(lo + 8u));
  lo += select(0u, 4u, magnitude >= srqWord(lo + 4u));
  lo += select(0u, 2u, magnitude >= srqWord(lo + 2u));
  lo += select(0u, 1u, magnitude >= srqWord(lo + 1u));
  lo += select(0u, 1u, magnitude >= srqWord(128u));
  let sign = bits & 0x80000000u;
  let level = min(lo, select(127u, 128u, sign != 0u));
  return srqWord(129u + level) | sign;
}

var<workgroup> partial: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let lane = lid % 32u;
  let col = wg.x * 4u + lid / 32u;
  var acc = 0.0;
  // 端の列も部分和0を書き、全threadが同じbarrierを通る。
  if (col < dims.n) {
    let units = dims.k / 16u;
    let row_base = col * units;
  // 出力チャネルの scale はループ不変 — 重みの要素ごとに引き直さない（ADR 0019）
  let wscale_v = wscale[col];
    for (var unit = lane; unit < units; unit += 32u) {
    let unitt = unit;
    let pwt = w[row_base + unitt];
    let xqt = wg.y * (dims.k / 4u) + unitt * 4u;
    let bt_0 = unpack4xI8(pwt.x);
    let xat_0 = x[xqt + 0u];
    acc = fma(xat_0.x, (f32(bt_0.x) * wscale_v), acc);
    acc = fma(xat_0.y, (f32(bt_0.y) * wscale_v), acc);
    acc = fma(xat_0.z, (f32(bt_0.z) * wscale_v), acc);
    acc = fma(xat_0.w, (f32(bt_0.w) * wscale_v), acc);
    let bt_1 = unpack4xI8(pwt.y);
    let xat_1 = x[xqt + 1u];
    acc = fma(xat_1.x, (f32(bt_1.x) * wscale_v), acc);
    acc = fma(xat_1.y, (f32(bt_1.y) * wscale_v), acc);
    acc = fma(xat_1.z, (f32(bt_1.z) * wscale_v), acc);
    acc = fma(xat_1.w, (f32(bt_1.w) * wscale_v), acc);
    let bt_2 = unpack4xI8(pwt.z);
    let xat_2 = x[xqt + 2u];
    acc = fma(xat_2.x, (f32(bt_2.x) * wscale_v), acc);
    acc = fma(xat_2.y, (f32(bt_2.y) * wscale_v), acc);
    acc = fma(xat_2.z, (f32(bt_2.z) * wscale_v), acc);
    acc = fma(xat_2.w, (f32(bt_2.w) * wscale_v), acc);
    let bt_3 = unpack4xI8(pwt.w);
    let xat_3 = x[xqt + 3u];
    acc = fma(xat_3.x, (f32(bt_3.x) * wscale_v), acc);
    acc = fma(xat_3.y, (f32(bt_3.y) * wscale_v), acc);
    acc = fma(xat_3.z, (f32(bt_3.z) * wscale_v), acc);
    acc = fma(xat_3.w, (f32(bt_3.w) * wscale_v), acc);
    }
  }
  partial[lid] = acc;
  workgroupBarrier();
  for (var width = 16u; width > 0u; width /= 2u) {
    if (lane < width) {
      partial[lid] = partial[lid] + partial[lid + width];
    }
    workgroupBarrier();
  }
  if (col < dims.n && lane == 0u) {
    out[wg.y * dims.n + col] = quantize(bitcast<u32>(partial[lid] + bias[col]) ^ dims.rounding_mask);
  }
}
