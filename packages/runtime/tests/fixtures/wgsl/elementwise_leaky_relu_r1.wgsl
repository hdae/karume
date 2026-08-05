// karume elementwise leaky_relu (rank 1, generated)
@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read> in0: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let n = params[0u];
  let s0 = bitcast<f32>(params[3u]);
  let stride = nwg.x * 256u;
  var i = gid.x;
  while (i < n) {
    var rem = i;
    let c0 = rem;
    let v0 = in0[c0 * params[2u]];
    out[i] = select(s0 * v0, v0, v0 >= 0.0);
    i = i + stride;
  }
}
