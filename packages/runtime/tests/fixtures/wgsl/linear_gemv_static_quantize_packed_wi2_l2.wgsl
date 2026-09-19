// karume linear gemv K parallel (i2, 2 lanes/output, packed int8 活性)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
  rounding_mask: u32,
  srq: array<vec4<u32>, 65>,
  x_scale: f32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<vec4<u32>>;
// 行頭が 16 B 整列なのは k % 64 == 0 から（適格判定が保証する）
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
  let lane = lid % 2u;
  let col = wg.x * 64u + lid / 2u;
  var acc = 0.0;
  // 端の列も部分和0を書き、全threadが同じbarrierを通る。
  if (col < dims.n) {
    let units = dims.k / 64u;
    let row_base = col * units;
  // 出力チャネルの scale はループ不変 — 重みの要素ごとに引き直さない（ADR 0019）
  let wscale_v = wscale[col];
    for (var unit = lane; unit < units; unit += 2u) {
    let unitt = unit;
    let pwt = w[row_base + unitt];
    let xqt = wg.y * (dims.k / 16u) + unitt * 4u;
    let qt_0 = (pwt.x >> 0u) & 255u;
    let dt_0 = vec4<f32>(vec4<i32>(vec4<u32>(qt_0, qt_0 >> 2u, qt_0 >> 4u, qt_0 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xpt_0 = x[xqt + 0u];
    let xt_0_0 = vec4<f32>(unpack4xI8(xpt_0.x)) * dims.x_scale;
    acc = acc + xt_0_0.x * dt_0.x;
    acc = acc + xt_0_0.y * dt_0.y;
    acc = acc + xt_0_0.z * dt_0.z;
    acc = acc + xt_0_0.w * dt_0.w;
    let qt_1 = (pwt.x >> 8u) & 255u;
    let dt_1 = vec4<f32>(vec4<i32>(vec4<u32>(qt_1, qt_1 >> 2u, qt_1 >> 4u, qt_1 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xt_1_0 = vec4<f32>(unpack4xI8(xpt_0.y)) * dims.x_scale;
    acc = acc + xt_1_0.x * dt_1.x;
    acc = acc + xt_1_0.y * dt_1.y;
    acc = acc + xt_1_0.z * dt_1.z;
    acc = acc + xt_1_0.w * dt_1.w;
    let qt_2 = (pwt.x >> 16u) & 255u;
    let dt_2 = vec4<f32>(vec4<i32>(vec4<u32>(qt_2, qt_2 >> 2u, qt_2 >> 4u, qt_2 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xt_2_0 = vec4<f32>(unpack4xI8(xpt_0.z)) * dims.x_scale;
    acc = acc + xt_2_0.x * dt_2.x;
    acc = acc + xt_2_0.y * dt_2.y;
    acc = acc + xt_2_0.z * dt_2.z;
    acc = acc + xt_2_0.w * dt_2.w;
    let qt_3 = (pwt.x >> 24u) & 255u;
    let dt_3 = vec4<f32>(vec4<i32>(vec4<u32>(qt_3, qt_3 >> 2u, qt_3 >> 4u, qt_3 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xt_3_0 = vec4<f32>(unpack4xI8(xpt_0.w)) * dims.x_scale;
    acc = acc + xt_3_0.x * dt_3.x;
    acc = acc + xt_3_0.y * dt_3.y;
    acc = acc + xt_3_0.z * dt_3.z;
    acc = acc + xt_3_0.w * dt_3.w;
    let qt_4 = (pwt.y >> 0u) & 255u;
    let dt_4 = vec4<f32>(vec4<i32>(vec4<u32>(qt_4, qt_4 >> 2u, qt_4 >> 4u, qt_4 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xpt_1 = x[xqt + 1u];
    let xt_4_0 = vec4<f32>(unpack4xI8(xpt_1.x)) * dims.x_scale;
    acc = acc + xt_4_0.x * dt_4.x;
    acc = acc + xt_4_0.y * dt_4.y;
    acc = acc + xt_4_0.z * dt_4.z;
    acc = acc + xt_4_0.w * dt_4.w;
    let qt_5 = (pwt.y >> 8u) & 255u;
    let dt_5 = vec4<f32>(vec4<i32>(vec4<u32>(qt_5, qt_5 >> 2u, qt_5 >> 4u, qt_5 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xt_5_0 = vec4<f32>(unpack4xI8(xpt_1.y)) * dims.x_scale;
    acc = acc + xt_5_0.x * dt_5.x;
    acc = acc + xt_5_0.y * dt_5.y;
    acc = acc + xt_5_0.z * dt_5.z;
    acc = acc + xt_5_0.w * dt_5.w;
    let qt_6 = (pwt.y >> 16u) & 255u;
    let dt_6 = vec4<f32>(vec4<i32>(vec4<u32>(qt_6, qt_6 >> 2u, qt_6 >> 4u, qt_6 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xt_6_0 = vec4<f32>(unpack4xI8(xpt_1.z)) * dims.x_scale;
    acc = acc + xt_6_0.x * dt_6.x;
    acc = acc + xt_6_0.y * dt_6.y;
    acc = acc + xt_6_0.z * dt_6.z;
    acc = acc + xt_6_0.w * dt_6.w;
    let qt_7 = (pwt.y >> 24u) & 255u;
    let dt_7 = vec4<f32>(vec4<i32>(vec4<u32>(qt_7, qt_7 >> 2u, qt_7 >> 4u, qt_7 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xt_7_0 = vec4<f32>(unpack4xI8(xpt_1.w)) * dims.x_scale;
    acc = acc + xt_7_0.x * dt_7.x;
    acc = acc + xt_7_0.y * dt_7.y;
    acc = acc + xt_7_0.z * dt_7.z;
    acc = acc + xt_7_0.w * dt_7.w;
    let qt_8 = (pwt.z >> 0u) & 255u;
    let dt_8 = vec4<f32>(vec4<i32>(vec4<u32>(qt_8, qt_8 >> 2u, qt_8 >> 4u, qt_8 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xpt_2 = x[xqt + 2u];
    let xt_8_0 = vec4<f32>(unpack4xI8(xpt_2.x)) * dims.x_scale;
    acc = acc + xt_8_0.x * dt_8.x;
    acc = acc + xt_8_0.y * dt_8.y;
    acc = acc + xt_8_0.z * dt_8.z;
    acc = acc + xt_8_0.w * dt_8.w;
    let qt_9 = (pwt.z >> 8u) & 255u;
    let dt_9 = vec4<f32>(vec4<i32>(vec4<u32>(qt_9, qt_9 >> 2u, qt_9 >> 4u, qt_9 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xt_9_0 = vec4<f32>(unpack4xI8(xpt_2.y)) * dims.x_scale;
    acc = acc + xt_9_0.x * dt_9.x;
    acc = acc + xt_9_0.y * dt_9.y;
    acc = acc + xt_9_0.z * dt_9.z;
    acc = acc + xt_9_0.w * dt_9.w;
    let qt_10 = (pwt.z >> 16u) & 255u;
    let dt_10 = vec4<f32>(vec4<i32>(vec4<u32>(qt_10, qt_10 >> 2u, qt_10 >> 4u, qt_10 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xt_10_0 = vec4<f32>(unpack4xI8(xpt_2.z)) * dims.x_scale;
    acc = acc + xt_10_0.x * dt_10.x;
    acc = acc + xt_10_0.y * dt_10.y;
    acc = acc + xt_10_0.z * dt_10.z;
    acc = acc + xt_10_0.w * dt_10.w;
    let qt_11 = (pwt.z >> 24u) & 255u;
    let dt_11 = vec4<f32>(vec4<i32>(vec4<u32>(qt_11, qt_11 >> 2u, qt_11 >> 4u, qt_11 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xt_11_0 = vec4<f32>(unpack4xI8(xpt_2.w)) * dims.x_scale;
    acc = acc + xt_11_0.x * dt_11.x;
    acc = acc + xt_11_0.y * dt_11.y;
    acc = acc + xt_11_0.z * dt_11.z;
    acc = acc + xt_11_0.w * dt_11.w;
    let qt_12 = (pwt.w >> 0u) & 255u;
    let dt_12 = vec4<f32>(vec4<i32>(vec4<u32>(qt_12, qt_12 >> 2u, qt_12 >> 4u, qt_12 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xpt_3 = x[xqt + 3u];
    let xt_12_0 = vec4<f32>(unpack4xI8(xpt_3.x)) * dims.x_scale;
    acc = acc + xt_12_0.x * dt_12.x;
    acc = acc + xt_12_0.y * dt_12.y;
    acc = acc + xt_12_0.z * dt_12.z;
    acc = acc + xt_12_0.w * dt_12.w;
    let qt_13 = (pwt.w >> 8u) & 255u;
    let dt_13 = vec4<f32>(vec4<i32>(vec4<u32>(qt_13, qt_13 >> 2u, qt_13 >> 4u, qt_13 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xt_13_0 = vec4<f32>(unpack4xI8(xpt_3.y)) * dims.x_scale;
    acc = acc + xt_13_0.x * dt_13.x;
    acc = acc + xt_13_0.y * dt_13.y;
    acc = acc + xt_13_0.z * dt_13.z;
    acc = acc + xt_13_0.w * dt_13.w;
    let qt_14 = (pwt.w >> 16u) & 255u;
    let dt_14 = vec4<f32>(vec4<i32>(vec4<u32>(qt_14, qt_14 >> 2u, qt_14 >> 4u, qt_14 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xt_14_0 = vec4<f32>(unpack4xI8(xpt_3.z)) * dims.x_scale;
    acc = acc + xt_14_0.x * dt_14.x;
    acc = acc + xt_14_0.y * dt_14.y;
    acc = acc + xt_14_0.z * dt_14.z;
    acc = acc + xt_14_0.w * dt_14.w;
    let qt_15 = (pwt.w >> 24u) & 255u;
    let dt_15 = vec4<f32>(vec4<i32>(vec4<u32>(qt_15, qt_15 >> 2u, qt_15 >> 4u, qt_15 >> 6u) & vec4<u32>(3u)) - vec4<i32>(2)) * wscale_v;
    let xt_15_0 = vec4<f32>(unpack4xI8(xpt_3.w)) * dims.x_scale;
    acc = acc + xt_15_0.x * dt_15.x;
    acc = acc + xt_15_0.y * dt_15.y;
    acc = acc + xt_15_0.z * dt_15.z;
    acc = acc + xt_15_0.w * dt_15.w;
    }
  }
  partial[lid] = acc;
  workgroupBarrier();
  for (var width = 1u; width > 0u; width /= 2u) {
    if (lane < width) {
      partial[lid] = partial[lid] + partial[lid + width];
    }
    workgroupBarrier();
  }
  if (col < dims.n && lane == 0u) {
    out[wg.y * dims.n + col] = quantize(bitcast<u32>(partial[lid] + bias[col]) ^ dims.rounding_mask);
  }
}
