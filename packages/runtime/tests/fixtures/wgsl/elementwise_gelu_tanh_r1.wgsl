// karume elementwise gelu_tanh (rank 1, generated)
@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read> in0: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

fn is_nan_bits(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

fn tanh_stable(x: f32) -> f32 {
  let lo = select(x, -9.5, x < -9.5);
  let t = select(lo, 9.5, x > 9.5);
  return select(tanh(t), x, is_nan_bits(x));
}

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let n = params[0u];
  let stride = nwg.x * 256u;
  var i = gid.x;
  while (i < n) {
    var rem = i;
    let c0 = rem;
    let v0 = in0[c0 * params[2u]];
    out[i] = 0.5 * v0 * (1.0 + tanh_stable(0.7978845608028654 * (v0 + 0.044715 * v0 * v0 * v0)));
    i = i + stride;
  }
}
