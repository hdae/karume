// karume gated residual (residual + y * where(mask, gate_c, gate_g), f32, staged product)
struct Params {
  n: u32,
  dim: u32,
  reserved0: u32,
  reserved1: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> mask: array<u32>;
@group(0) @binding(2) var<storage, read> gate_condition: array<f32>;
@group(0) @binding(3) var<storage, read> gate_generation: array<f32>;
@group(0) @binding(4) var<storage, read> y: array<f32>;
@group(0) @binding(5) var<storage, read> residual: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
var<workgroup> products: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let stride = nwg.x * 256u;
  var base = wid.x * 256u;
  while (base < params.n) {
    let i = base + lid;
    if (i < params.n) {
      let row = i / params.dim;
      let o = i - row * params.dim;
      let gate = select(gate_generation[o], gate_condition[o], mask[row] != 0u);
      products[lid] = bitcast<u32>(y[i] * gate);
    }
    workgroupBarrier();
    if (i < params.n) {
      out[i] = residual[i] + bitcast<f32>(products[lid]);
    }
    workgroupBarrier();
    base = base + stride;
  }
}
