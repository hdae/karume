// karume linear gemv K parallel (f32, 8 lanes/output)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
// Row starts are 16 B aligned because k % 4 == 0 (guaranteed by the eligibility check)
@group(0) @binding(2) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

var<workgroup> partial: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let lane = lid % 8u;
  let col = wg.x * 16u + lid / 8u;
  var acc = 0.0;
  // Edge columns also write a partial sum of 0, so every thread passes the same barriers.
  if (col < dims.n) {
    let units = dims.k / 4u;
    let row_base = col * units;
    for (var unit = lane; unit < units; unit += 8u) {
    let unitt = unit;
    let pwt = w[row_base + unitt];
    let xqt = wg.y * (dims.k / 4u) + unitt * 1u;
    let xft = x[xqt];
    acc = fma(xft.x, pwt.x, acc);
    acc = fma(xft.y, pwt.y, acc);
    acc = fma(xft.z, pwt.z, acc);
    acc = fma(xft.w, pwt.w, acc);
    }
  }
  partial[lid] = acc;
  workgroupBarrier();
  for (var width = 4u; width > 0u; width /= 2u) {
    if (lane < width) {
      partial[lid] = partial[lid] + partial[lid + width];
    }
    workgroupBarrier();
  }
  if (col < dims.n && lane == 0u) {
    out[wg.y * dims.n + col] = partial[lid] + bias[col];
  }
}
