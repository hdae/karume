// karume group_norm apply ((x - mean) * rstd * weight[c] + bias[c]; f32, grid-stride, full-write)
struct Params {
  count: u32,
  length: u32,
  inner: u32,
  groups: u32,
  per_group: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read> weight: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let step = nwg.x * 256u;
  var e = wid.x * 256u + lid3.x;
  while (e < params.count) {
    let row = e / params.length;
    let o = e - row * params.length;
    let c = (row % params.groups) * params.per_group + o / params.inner;
    out[e] = (x[e] - stats[2u * row]) * stats[2u * row + 1u] * weight[c] + bias[c];
    e = e + step;
  }
}
