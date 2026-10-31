// karume linear gemv K parallel (i4, 32 lanes/output, packed int8 activations, staged activations)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
  x_scale: f32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<vec4<u32>>;
// Row starts are 16 B aligned because k % 32 == 0 (guaranteed by the eligibility check)
@group(0) @binding(2) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read> wscale: array<f32>;
var<workgroup> partial: array<f32, 128>;
var<workgroup> sx: array<vec4<u32>, 64u>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let lane = lid % 32u;
  let col = wg.x * 4u + lid / 32u;
  var acc = 0.0;
  // Edge columns load their share of the staged activation and pass every barrier; their hoisted
  // reads take the last column, and their sums are never stored.
  let units = dims.k / 32u;
  let col_read = min(col, dims.n - 1u);
  let row_base = col_read * units;
  let scale_base = col_read * (dims.k >> 11u);
  let sx_base = lane * 2u;
  let chunks = (units + 31u) / 32u;
  for (var chunk = 0u; chunk < chunks; chunk += 1u) {
    // Chunk c stages the activation words (packed) of weight words c·32 … c·32 + 31.
    for (var q = lid; q < 64u; q += 128u) {
      let v = chunk * 64u + q;
      var staged_quad = vec4<u32>(0u);
      if (v < dims.k / 16u) {
        staged_quad = x[wg.y * (dims.k / 16u) + v];
      }
      sx[(q) ^ (((q) >> 3u) & 7u)] = staged_quad;
    }
    workgroupBarrier();
    let unit = chunk * 32u + lane;
    if (col < dims.n && unit < units) {
    let unitt = unit;
    let pwt = w[row_base + unitt];
    let wst = wscale[scale_base + ((unitt * 32u) >> 11u)];
    let bt_0 = unpack4xU8(pwt.x);
    let xpt_0 = sx[(sx_base + 0u) ^ (((sx_base + 0u) >> 3u) & 7u)];
    let xat_0 = vec4<f32>(unpack4xI8(xpt_0.x)) * dims.x_scale;
    let xbt_0 = vec4<f32>(unpack4xI8(xpt_0.y)) * dims.x_scale;
    acc = fma(xat_0.x, (f32(i32(bt_0.x & 0xFu) - 8) * wst), acc);
    acc = fma(xat_0.y, (f32(i32(bt_0.x >> 4u) - 8) * wst), acc);
    acc = fma(xat_0.z, (f32(i32(bt_0.y & 0xFu) - 8) * wst), acc);
    acc = fma(xat_0.w, (f32(i32(bt_0.y >> 4u) - 8) * wst), acc);
    acc = fma(xbt_0.x, (f32(i32(bt_0.z & 0xFu) - 8) * wst), acc);
    acc = fma(xbt_0.y, (f32(i32(bt_0.z >> 4u) - 8) * wst), acc);
    acc = fma(xbt_0.z, (f32(i32(bt_0.w & 0xFu) - 8) * wst), acc);
    acc = fma(xbt_0.w, (f32(i32(bt_0.w >> 4u) - 8) * wst), acc);
    let bt_1 = unpack4xU8(pwt.y);
    let xat_1 = vec4<f32>(unpack4xI8(xpt_0.z)) * dims.x_scale;
    let xbt_1 = vec4<f32>(unpack4xI8(xpt_0.w)) * dims.x_scale;
    acc = fma(xat_1.x, (f32(i32(bt_1.x & 0xFu) - 8) * wst), acc);
    acc = fma(xat_1.y, (f32(i32(bt_1.x >> 4u) - 8) * wst), acc);
    acc = fma(xat_1.z, (f32(i32(bt_1.y & 0xFu) - 8) * wst), acc);
    acc = fma(xat_1.w, (f32(i32(bt_1.y >> 4u) - 8) * wst), acc);
    acc = fma(xbt_1.x, (f32(i32(bt_1.z & 0xFu) - 8) * wst), acc);
    acc = fma(xbt_1.y, (f32(i32(bt_1.z >> 4u) - 8) * wst), acc);
    acc = fma(xbt_1.z, (f32(i32(bt_1.w & 0xFu) - 8) * wst), acc);
    acc = fma(xbt_1.w, (f32(i32(bt_1.w >> 4u) - 8) * wst), acc);
    let bt_2 = unpack4xU8(pwt.z);
    let xpt_1 = sx[(sx_base + 1u) ^ (((sx_base + 1u) >> 3u) & 7u)];
    let xat_2 = vec4<f32>(unpack4xI8(xpt_1.x)) * dims.x_scale;
    let xbt_2 = vec4<f32>(unpack4xI8(xpt_1.y)) * dims.x_scale;
    acc = fma(xat_2.x, (f32(i32(bt_2.x & 0xFu) - 8) * wst), acc);
    acc = fma(xat_2.y, (f32(i32(bt_2.x >> 4u) - 8) * wst), acc);
    acc = fma(xat_2.z, (f32(i32(bt_2.y & 0xFu) - 8) * wst), acc);
    acc = fma(xat_2.w, (f32(i32(bt_2.y >> 4u) - 8) * wst), acc);
    acc = fma(xbt_2.x, (f32(i32(bt_2.z & 0xFu) - 8) * wst), acc);
    acc = fma(xbt_2.y, (f32(i32(bt_2.z >> 4u) - 8) * wst), acc);
    acc = fma(xbt_2.z, (f32(i32(bt_2.w & 0xFu) - 8) * wst), acc);
    acc = fma(xbt_2.w, (f32(i32(bt_2.w >> 4u) - 8) * wst), acc);
    let bt_3 = unpack4xU8(pwt.w);
    let xat_3 = vec4<f32>(unpack4xI8(xpt_1.z)) * dims.x_scale;
    let xbt_3 = vec4<f32>(unpack4xI8(xpt_1.w)) * dims.x_scale;
    acc = fma(xat_3.x, (f32(i32(bt_3.x & 0xFu) - 8) * wst), acc);
    acc = fma(xat_3.y, (f32(i32(bt_3.x >> 4u) - 8) * wst), acc);
    acc = fma(xat_3.z, (f32(i32(bt_3.y & 0xFu) - 8) * wst), acc);
    acc = fma(xat_3.w, (f32(i32(bt_3.y >> 4u) - 8) * wst), acc);
    acc = fma(xbt_3.x, (f32(i32(bt_3.z & 0xFu) - 8) * wst), acc);
    acc = fma(xbt_3.y, (f32(i32(bt_3.z >> 4u) - 8) * wst), acc);
    acc = fma(xbt_3.z, (f32(i32(bt_3.w & 0xFu) - 8) * wst), acc);
    acc = fma(xbt_3.w, (f32(i32(bt_3.w >> 4u) - 8) * wst), acc);
    }
    workgroupBarrier();
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
    out[wg.y * dims.n + col] = partial[lid] + bias[col];
  }
}
