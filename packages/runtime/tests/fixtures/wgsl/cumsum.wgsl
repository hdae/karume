// karume cumsum (last dim, f32, sequential per row)
struct Params {
  rows: u32,
  dim: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let stride = nwg.x * 256u;
  var row = gid.x;
  while (row < params.rows) {
    let base = row * params.dim;
    var acc = 0.0;
    var j = 0u;
    while (j < params.dim) {
      acc = acc + x[base + j];
      out[base + j] = acc;
      j = j + 1u;
    }
    row = row + stride;
  }
}
