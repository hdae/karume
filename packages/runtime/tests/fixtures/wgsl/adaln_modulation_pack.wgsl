// karume adaLN modulation pack (4 vectors -> one table, raw u32 copy, full-write)
struct Params {
  dim: u32,
  reserved0: u32,
  reserved1: u32,
  reserved2: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> scale_condition: array<u32>;
@group(0) @binding(2) var<storage, read> scale_generation: array<u32>;
@group(0) @binding(3) var<storage, read> shift_condition: array<u32>;
@group(0) @binding(4) var<storage, read> shift_generation: array<u32>;
@group(0) @binding(5) var<storage, read_write> modulation_table: array<u32>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let dim = params.dim;
  let stride = nwg.x * 256u;
  var o = gid.x;
  while (o < dim) {
    modulation_table[o] = scale_condition[o];
    modulation_table[dim + o] = scale_generation[o];
    modulation_table[2u * dim + o] = shift_condition[o];
    modulation_table[3u * dim + o] = shift_generation[o];
    o = o + stride;
  }
}
