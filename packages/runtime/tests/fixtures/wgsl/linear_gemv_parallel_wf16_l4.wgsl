// karume linear gemv K parallel (f16, 4 lanes/output)
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
  let lane = lid % 4u;
  let col = wg.x * 32u + lid / 4u;
  var acc = 0.0;
  // Edge columns also write a partial sum of 0, so every thread passes the same barriers.
  if (col < dims.n) {
    let units = dims.k / 8u;
    let row_base = col * units;
    for (var unit = lane; unit < units; unit += 4u) {
    let unitt = unit;
    let pwt = w[row_base + unitt];
    let xqt = wg.y * (dims.k / 4u) + unitt * 2u;
    let ht_0 = unpack2x16float(pwt.x);
    let xt_0 = x[xqt + 0u];
    acc = fma(xt_0.x, ht_0.x, acc);
    acc = fma(xt_0.y, ht_0.y, acc);
    let ht_1 = unpack2x16float(pwt.y);
    let xt_1 = x[xqt + 0u];
    acc = fma(xt_1.z, ht_1.x, acc);
    acc = fma(xt_1.w, ht_1.y, acc);
    let ht_2 = unpack2x16float(pwt.z);
    let xt_2 = x[xqt + 1u];
    acc = fma(xt_2.x, ht_2.x, acc);
    acc = fma(xt_2.y, ht_2.y, acc);
    let ht_3 = unpack2x16float(pwt.w);
    let xt_3 = x[xqt + 1u];
    acc = fma(xt_3.z, ht_3.x, acc);
    acc = fma(xt_3.w, ht_3.y, acc);
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
