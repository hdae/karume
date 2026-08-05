// karume elementwise clamp_min (rank 1, generated)
@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read> in0: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

fn is_nan_bits(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

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
    out[i] = select(select(v0, s0, v0 < s0), v0, is_nan_bits(v0));
    i = i + stride;
  }
}
