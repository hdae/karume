// karume elementwise log1p (rank 1, generated)
@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read> in0: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

fn log1p_series(x: f32) -> f32 {
  // 係数は 1/k（Horner 形）。切り替え点より外は log(1+x) をそのまま使う。
  let series = x * (1.0 - x * (0.5 - x * (0.3333333333333333 - x * (0.25 - x * (0.2 - x * (0.16666666666666666 - x * (0.14285714285714285 - x * (0.125 - x * (0.1111111111111111 - x * (0.1))))))))));
  return select(log(1.0 + x), series, abs(x) < 0.25);
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
    out[i] = log1p_series(v0);
    i = i + stride;
  }
}
