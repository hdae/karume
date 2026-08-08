// karume RoPE (half split, f32, full-write)
struct Params {
  n: u32,
  sequence: u32,
  head_dim: u32,
  half_dim: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> cos_table: array<f32>;
@group(0) @binding(3) var<storage, read> sin_table: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
var<workgroup> products: array<vec2<u32>, 256>;

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
      let d = i % params.head_dim;
      let row = i / params.head_dim;
      let token = row % params.sequence;
      let row_base = i - d;
      var rotated_bits: u32;
      if (d < params.half_dim) {
        rotated_bits = bitcast<u32>(x[row_base + d + params.half_dim]) ^ 0x80000000u;
      } else {
        rotated_bits = bitcast<u32>(x[row_base + d - params.half_dim]);
      }
      let table_index = token * params.head_dim + d;
      let direct_bits = bitcast<u32>(x[i] * cos_table[table_index]);
      let cross_bits = bitcast<u32>(bitcast<f32>(rotated_bits) * sin_table[table_index]);
      products[lid] = vec2<u32>(direct_bits, cross_bits);
    } else {
      products[lid] = vec2<u32>(0u);
    }
    workgroupBarrier();
    if (i < params.n) {
      let pair = products[lid];
      out[i] = bitcast<f32>(pair.x) + bitcast<f32>(pair.y);
    }
    workgroupBarrier();
    base = base + stride;
  }
}
