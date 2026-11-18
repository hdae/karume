// karume linear gemv K parallel (bf16, 16 lanes/output)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
// Row starts are 16 B aligned because k % 8 == 0 (guaranteed by the eligibility check)
@group(0) @binding(2) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

var<workgroup> partial: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let lane = lid % 16u;
  let col = wg.x * 8u + lid / 16u;
  var acc = 0.0;
  // Edge columns also write a partial sum of 0, so every thread passes the same barriers.
  if (col < dims.n) {
    let units = dims.k / 8u;
    let row_base = col * units;
    for (var unit = lane; unit < units; unit += 16u) {
    let unitt = unit;
    let pwt = w[row_base + unitt];
    let xqt = wg.y * (dims.k / 4u) + unitt * 2u;
    let bt_0 = pwt.x;
    let xt_0 = x[xqt + 0u];
    acc = fma(xt_0.x, bitcast<f32>(bt_0 << 16u), acc);
    acc = fma(xt_0.y, bitcast<f32>(bt_0 & 0xFFFF0000u), acc);
    let bt_1 = pwt.y;
    let xt_1 = x[xqt + 0u];
    acc = fma(xt_1.z, bitcast<f32>(bt_1 << 16u), acc);
    acc = fma(xt_1.w, bitcast<f32>(bt_1 & 0xFFFF0000u), acc);
    let bt_2 = pwt.z;
    let xt_2 = x[xqt + 1u];
    acc = fma(xt_2.x, bitcast<f32>(bt_2 << 16u), acc);
    acc = fma(xt_2.y, bitcast<f32>(bt_2 & 0xFFFF0000u), acc);
    let bt_3 = pwt.w;
    let xt_3 = x[xqt + 1u];
    acc = fma(xt_3.z, bitcast<f32>(bt_3 << 16u), acc);
    acc = fma(xt_3.w, bitcast<f32>(bt_3 & 0xFFFF0000u), acc);
    }
  }
  partial[lid] = acc;
  workgroupBarrier();
  for (var width = 8u; width > 0u; width /= 2u) {
    if (lane < width) {
      partial[lid] = partial[lid] + partial[lid + width];
    }
    workgroupBarrier();
  }
  if (col < dims.n && lane == 0u) {
    out[wg.y * dims.n + col] = partial[lid] + bias[col];
  }
}
