// karume upsample2x (NCHW nearest, f32 bits, full-write)
struct Params {
  n: u32,
  width: u32,
  out_width: u32,
  reserved: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let stride = nwg.x * 256u;
  var i = gid.x;
  while (i < params.n) {
    let col = i % params.width;
    let row = i / params.width;
    let base = 2u * row * params.out_width + 2u * col;
    let value = x[i];
    out[base] = value;
    out[base + 1u] = value;
    out[base + params.out_width] = value;
    out[base + params.out_width + 1u] = value;
    i = i + stride;
  }
}
