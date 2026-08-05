// karume elementwise ge_scalar (rank 2, generated)
@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read> in0: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let n = params[0u];
  let s0 = bitcast<f32>(params[5u]);
  let stride = nwg.x * 256u;
  var i = gid.x;
  while (i < n) {
    var rem = i;
    let c1 = rem % params[2u]; rem = rem / params[2u];
    let c0 = rem;
    let v0 = in0[c0 * params[3u] + c1 * params[4u]];
    out[i] = select(0u, 1u, v0 >= s0);
    i = i + stride;
  }
}
