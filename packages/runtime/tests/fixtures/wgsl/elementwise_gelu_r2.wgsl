// karume elementwise gelu (rank 2, generated)
@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read> in0: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

fn erf_approx(x: f32) -> f32 {
  let s = sign(x);
  let a = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  var p = 1.061405429;
  p = -1.453152027 + p * t;
  p = 1.421413741 + p * t;
  p = -0.284496736 + p * t;
  p = 0.254829592 + p * t;
  return s * (1.0 - p * t * exp(-a * a));
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
    let c1 = rem % params[2u]; rem = rem / params[2u];
    let c0 = rem;
    let v0 = in0[c0 * params[3u] + c1 * params[4u]];
    out[i] = 0.5 * v0 * (1.0 + erf_approx(v0 * 0.7071067811865476));
    i = i + stride;
  }
}
