// karume group_norm totals (pass 0: mean, pass 1: 1 / sqrt(variance + eps); f32, fixed tree)
struct Params {
  rows: u32,
  chunks: u32,
  phase: u32,
  inv_length: f32,
  eps: f32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> partial: array<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  var row = wid.x;
  while (row < params.rows) {
    let base = row * params.chunks;
    var acc = 0.0;
    var j = lid;
    while (j < params.chunks) {
      acc = acc + partial[base + j];
      j = j + 256u;
    }
    scratch[lid] = acc;
    workgroupBarrier();
    var stride = 128u;
    while (stride > 0u) {
      if (lid < stride) {
        scratch[lid] = scratch[lid] + scratch[lid + stride];
      }
      workgroupBarrier();
      stride = stride / 2u;
    }
    if (lid == 0u) {
      if (params.phase == 0u) {
        stats[2u * row] = scratch[0u] * params.inv_length;
      } else {
        stats[2u * row + 1u] = 1.0 / sqrt(scratch[0u] * params.inv_length + params.eps);
      }
    }
    // make every read of scratch[0] finish before the next row overwrites scratch[lid]
    workgroupBarrier();
    row = row + nwg.x;
  }
}
