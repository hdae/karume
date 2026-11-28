// karume linear split-K, pass 2 (the K-slice partial sums added in slice order, then + bias[n], f32, grid-stride / wg 256)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
  slices: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> partials: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let total = dims.m * dims.n;
  let stride = nwg.x * 256u;
  var i = gid.x;
  while (i < total) {
    var acc = partials[i];
    for (var slice = 1u; slice < dims.slices; slice = slice + 1u) {
      acc = acc + partials[slice * total + i];
    }
    // The bias is added once, last (not as the accumulator's initial value).
    out[i] = acc + bias[i % dims.n];
    i = i + stride;
  }
}
