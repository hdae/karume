// karume linear gemv K parallel (i4, 4 lanes/output, packed int8 活性)
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
// 行頭が 16 B 整列なのは k % 32 == 0 から（適格判定が保証する）
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
  let lane = lid % 4u;
  let col = wg.x * 32u + lid / 4u;
  var acc = 0.0;
  // 端の列も部分和0を書き、全threadが同じbarrierを通る。
  if (col < dims.n) {
    let units = dims.k / 32u;
    let row_base = col * units;
  let scale_base = col * (dims.k >> 9u);
    for (var unit = lane; unit < units; unit += 4u) {
    let unitt = unit;
    let pwt = w[row_base + unitt];
    let wst = wscale[scale_base + ((unitt * 32u) >> 9u)];
    let xqt = wg.y * (dims.k / 16u) + unitt * 2u;
    let bt_0 = unpack4xU8(pwt.x);
    let xpt_0 = x[xqt + 0u];
    let xat_0 = vec4<f32>(unpack4xI8(xpt_0.x)) * dims.x_scale;
    let xbt_0 = vec4<f32>(unpack4xI8(xpt_0.y)) * dims.x_scale;
    acc = acc + xat_0.x * (f32(i32(bt_0.x & 0xFu) - 8) * wst);
    acc = acc + xat_0.y * (f32(i32(bt_0.x >> 4u) - 8) * wst);
    acc = acc + xat_0.z * (f32(i32(bt_0.y & 0xFu) - 8) * wst);
    acc = acc + xat_0.w * (f32(i32(bt_0.y >> 4u) - 8) * wst);
    acc = acc + xbt_0.x * (f32(i32(bt_0.z & 0xFu) - 8) * wst);
    acc = acc + xbt_0.y * (f32(i32(bt_0.z >> 4u) - 8) * wst);
    acc = acc + xbt_0.z * (f32(i32(bt_0.w & 0xFu) - 8) * wst);
    acc = acc + xbt_0.w * (f32(i32(bt_0.w >> 4u) - 8) * wst);
    let bt_1 = unpack4xU8(pwt.y);
    let xat_1 = vec4<f32>(unpack4xI8(xpt_0.z)) * dims.x_scale;
    let xbt_1 = vec4<f32>(unpack4xI8(xpt_0.w)) * dims.x_scale;
    acc = acc + xat_1.x * (f32(i32(bt_1.x & 0xFu) - 8) * wst);
    acc = acc + xat_1.y * (f32(i32(bt_1.x >> 4u) - 8) * wst);
    acc = acc + xat_1.z * (f32(i32(bt_1.y & 0xFu) - 8) * wst);
    acc = acc + xat_1.w * (f32(i32(bt_1.y >> 4u) - 8) * wst);
    acc = acc + xbt_1.x * (f32(i32(bt_1.z & 0xFu) - 8) * wst);
    acc = acc + xbt_1.y * (f32(i32(bt_1.z >> 4u) - 8) * wst);
    acc = acc + xbt_1.z * (f32(i32(bt_1.w & 0xFu) - 8) * wst);
    acc = acc + xbt_1.w * (f32(i32(bt_1.w >> 4u) - 8) * wst);
    let bt_2 = unpack4xU8(pwt.z);
    let xpt_1 = x[xqt + 1u];
    let xat_2 = vec4<f32>(unpack4xI8(xpt_1.x)) * dims.x_scale;
    let xbt_2 = vec4<f32>(unpack4xI8(xpt_1.y)) * dims.x_scale;
    acc = acc + xat_2.x * (f32(i32(bt_2.x & 0xFu) - 8) * wst);
    acc = acc + xat_2.y * (f32(i32(bt_2.x >> 4u) - 8) * wst);
    acc = acc + xat_2.z * (f32(i32(bt_2.y & 0xFu) - 8) * wst);
    acc = acc + xat_2.w * (f32(i32(bt_2.y >> 4u) - 8) * wst);
    acc = acc + xbt_2.x * (f32(i32(bt_2.z & 0xFu) - 8) * wst);
    acc = acc + xbt_2.y * (f32(i32(bt_2.z >> 4u) - 8) * wst);
    acc = acc + xbt_2.z * (f32(i32(bt_2.w & 0xFu) - 8) * wst);
    acc = acc + xbt_2.w * (f32(i32(bt_2.w >> 4u) - 8) * wst);
    let bt_3 = unpack4xU8(pwt.w);
    let xat_3 = vec4<f32>(unpack4xI8(xpt_1.z)) * dims.x_scale;
    let xbt_3 = vec4<f32>(unpack4xI8(xpt_1.w)) * dims.x_scale;
    acc = acc + xat_3.x * (f32(i32(bt_3.x & 0xFu) - 8) * wst);
    acc = acc + xat_3.y * (f32(i32(bt_3.x >> 4u) - 8) * wst);
    acc = acc + xat_3.z * (f32(i32(bt_3.y & 0xFu) - 8) * wst);
    acc = acc + xat_3.w * (f32(i32(bt_3.y >> 4u) - 8) * wst);
    acc = acc + xbt_3.x * (f32(i32(bt_3.z & 0xFu) - 8) * wst);
    acc = acc + xbt_3.y * (f32(i32(bt_3.z >> 4u) - 8) * wst);
    acc = acc + xbt_3.z * (f32(i32(bt_3.w & 0xFu) - 8) * wst);
    acc = acc + xbt_3.w * (f32(i32(bt_3.w >> 4u) - 8) * wst);
    }
  }
  partial[lid] = acc;
  workgroupBarrier();
  for (var width = 2u; width > 0u; width /= 2u) {
    if (lane < width) {
      partial[lid] = partial[lid] + partial[lid + width];
    }
    workgroupBarrier();
  }
  if (col < dims.n && lane == 0u) {
    out[wg.y * dims.n + col] = quantize(bitcast<u32>(partial[lid] + bias[col]) ^ dims.rounding_mask);
  }
}
