// karume group_norm partial sums (pass 0: x, pass 1: (x - mean)^2; f32, 1024-element chunks, fixed tree)
struct Params {
  rows: u32,
  length: u32,
  chunks: u32,
  phase: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read_write> partial: array<f32>;

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let total = params.rows * params.chunks;
  var p = wid.x;
  while (p < total) {
    let row = p / params.chunks;
    let begin = (p - row * params.chunks) * 1024u;
    let base = row * params.length;
    var mean = 0.0;
    if (params.phase == 1u) {
      mean = stats[2u * row];
    }
    var acc = 0.0;
    for (var k = 0u; k < 4u; k = k + 1u) {
      let i = begin + k * 256u + lid;
      if (i < params.length) {
        let v = x[base + i];
        if (params.phase == 0u) {
          acc = acc + v;
        } else {
          let d = v - mean;
          acc = acc + d * d;
        }
      }
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
      partial[p] = scratch[0u];
    }
    // make every read of scratch[0] finish before the next chunk overwrites scratch[lid]
    workgroupBarrier();
    p = p + nwg.x;
  }
}
