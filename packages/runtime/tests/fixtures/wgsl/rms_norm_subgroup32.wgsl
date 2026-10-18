enable subgroups, subgroup_size_control;
struct Params {
  rows: u32,
  dim: u32,
  eps: f32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> scratch: array<f32, 8>;
@compute @workgroup_size(256) @subgroup_size(32)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
  @builtin(subgroup_invocation_id) sub: u32,
  @builtin(subgroup_id) sg: u32,
  @builtin(num_subgroups) numsg: u32,
) {
  let lid = lid3.x;
  let dim = params.dim;
  let scale = 1.0 / f32(dim);
  var row = wid.x;
  while (row < params.rows) {
    let base = row * dim;
    var acc = 0.0;
    var i = lid;
    while (i < dim) {
      let v = x[base + i];
      acc = acc + v * v;
      i = i + 256u;
    }
    let subtotal = subgroupAdd(acc);
    if (subgroupElect()) { scratch[sg] = subtotal; }
    workgroupBarrier();
    // Do not assume a mapping between local_invocation_id and the position within a subgroup.
    // Every subgroup reads the 8 partial sums in the same order and hands the total to all lanes.
    var across = 0.0;
    if (sub < numsg) { across = scratch[sub]; }
    let sum = subgroupAdd(across);
    let inv = inverseSqrt(sum * scale + params.eps);
    // Finish reading before the next grid-stride row overwrites the shared partial sums.
    workgroupBarrier();
    var o = lid;
    while (o < dim) {
      out[base + o] = x[base + o] * inv * weight[o];
      o = o + 256u;
    }
    row = row + nwg.x;
  }
}
