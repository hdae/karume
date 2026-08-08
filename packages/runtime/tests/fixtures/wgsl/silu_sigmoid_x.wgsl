// karume SiLU (sigmoid-x, f32, staged sigmoid, full-write)
struct Params {
  n: u32,
  reserved0: u32,
  reserved1: u32,
  reserved2: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
var<workgroup> sigmoid_bits: array<u32, 256>;

fn sigmoid_stable(x: f32) -> f32 {
  let t = exp(-abs(x));
  return select(1.0 / (1.0 + t), t / (1.0 + t), x < 0.0);
}

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let blocks = params.n / 256u +
    select(0u, 1u, params.n % 256u != 0u);
  var block = wid.x;
  while (block < blocks) {
    let i = block * 256u + lid;
    if (i < params.n) {
      let x_for_sigmoid = x[i];
      sigmoid_bits[lid] = bitcast<u32>(sigmoid_stable(x_for_sigmoid));
    }
    workgroupBarrier();
    if (i < params.n) {
      let x_for_mul = x[i];
      let sigmoid_after_store = bitcast<f32>(sigmoid_bits[lid]);
      out[i] = sigmoid_after_store * x_for_mul;
    }
    workgroupBarrier();
    block = block + nwg.x;
  }
}
