// karume linear gemv K parallel (i4, 4 lanes/output)
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
    let xqt = wg.y * (dims.k / 4u) + unitt * 8u;
    let bt_0 = unpack4xU8(pwt.x);
    let xat_0 = x[xqt + 0u];
    let xbt_0 = x[xqt + 1u];
    acc = fma(xat_0.x, (f32(i32(bt_0.x & 0xFu) - 8) * wst), acc);
    acc = fma(xat_0.y, (f32(i32(bt_0.x >> 4u) - 8) * wst), acc);
    acc = fma(xat_0.z, (f32(i32(bt_0.y & 0xFu) - 8) * wst), acc);
    acc = fma(xat_0.w, (f32(i32(bt_0.y >> 4u) - 8) * wst), acc);
    acc = fma(xbt_0.x, (f32(i32(bt_0.z & 0xFu) - 8) * wst), acc);
    acc = fma(xbt_0.y, (f32(i32(bt_0.z >> 4u) - 8) * wst), acc);
    acc = fma(xbt_0.z, (f32(i32(bt_0.w & 0xFu) - 8) * wst), acc);
    acc = fma(xbt_0.w, (f32(i32(bt_0.w >> 4u) - 8) * wst), acc);
    let bt_1 = unpack4xU8(pwt.y);
    let xat_1 = x[xqt + 2u];
    let xbt_1 = x[xqt + 3u];
    acc = fma(xat_1.x, (f32(i32(bt_1.x & 0xFu) - 8) * wst), acc);
    acc = fma(xat_1.y, (f32(i32(bt_1.x >> 4u) - 8) * wst), acc);
    acc = fma(xat_1.z, (f32(i32(bt_1.y & 0xFu) - 8) * wst), acc);
    acc = fma(xat_1.w, (f32(i32(bt_1.y >> 4u) - 8) * wst), acc);
    acc = fma(xbt_1.x, (f32(i32(bt_1.z & 0xFu) - 8) * wst), acc);
    acc = fma(xbt_1.y, (f32(i32(bt_1.z >> 4u) - 8) * wst), acc);
    acc = fma(xbt_1.z, (f32(i32(bt_1.w & 0xFu) - 8) * wst), acc);
    acc = fma(xbt_1.w, (f32(i32(bt_1.w >> 4u) - 8) * wst), acc);
    let bt_2 = unpack4xU8(pwt.z);
    let xat_2 = x[xqt + 4u];
    let xbt_2 = x[xqt + 5u];
    acc = fma(xat_2.x, (f32(i32(bt_2.x & 0xFu) - 8) * wst), acc);
    acc = fma(xat_2.y, (f32(i32(bt_2.x >> 4u) - 8) * wst), acc);
    acc = fma(xat_2.z, (f32(i32(bt_2.y & 0xFu) - 8) * wst), acc);
    acc = fma(xat_2.w, (f32(i32(bt_2.y >> 4u) - 8) * wst), acc);
    acc = fma(xbt_2.x, (f32(i32(bt_2.z & 0xFu) - 8) * wst), acc);
    acc = fma(xbt_2.y, (f32(i32(bt_2.z >> 4u) - 8) * wst), acc);
    acc = fma(xbt_2.z, (f32(i32(bt_2.w & 0xFu) - 8) * wst), acc);
    acc = fma(xbt_2.w, (f32(i32(bt_2.w >> 4u) - 8) * wst), acc);
    let bt_3 = unpack4xU8(pwt.w);
    let xat_3 = x[xqt + 6u];
    let xbt_3 = x[xqt + 7u];
    acc = fma(xat_3.x, (f32(i32(bt_3.x & 0xFu) - 8) * wst), acc);
    acc = fma(xat_3.y, (f32(i32(bt_3.x >> 4u) - 8) * wst), acc);
    acc = fma(xat_3.z, (f32(i32(bt_3.y & 0xFu) - 8) * wst), acc);
    acc = fma(xat_3.w, (f32(i32(bt_3.y >> 4u) - 8) * wst), acc);
    acc = fma(xbt_3.x, (f32(i32(bt_3.z & 0xFu) - 8) * wst), acc);
    acc = fma(xbt_3.y, (f32(i32(bt_3.z >> 4u) - 8) * wst), acc);
    acc = fma(xbt_3.z, (f32(i32(bt_3.w & 0xFu) - 8) * wst), acc);
    acc = fma(xbt_3.w, (f32(i32(bt_3.w >> 4u) - 8) * wst), acc);
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
    out[wg.y * dims.n + col] = partial[lid] + bias[col];
  }
}
