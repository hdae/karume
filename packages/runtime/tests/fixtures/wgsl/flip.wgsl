// karume flip (static axis, f32, full-write)
struct Params {
  n: u32,
  len: u32,
  inner: u32,
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
  var i = gid.x;
  while (i < params.n) {
    let tail = i % params.inner;
    let rest = i / params.inner;
    let axis = rest % params.len;
    let head = rest / params.len;
    let src = (head * params.len + (params.len - 1u - axis)) * params.inner + tail;
    out[i] = x[src];
    i = i + stride;
  }
}
