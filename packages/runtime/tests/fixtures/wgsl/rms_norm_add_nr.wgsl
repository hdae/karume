// karume rms_norm (last dim, weight only, f32, sum of squares in 1 pass; weight, x, residual prefetched, 6 elements per thread)
struct Params {
  rows: u32,
  dim: u32,
  eps: f32,
  rounding_mask: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

var<workgroup> scratch: array<f32, 256>;

@group(0) @binding(3) var<storage, read> residual: array<f32>;
@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let dim = params.dim;
  let scale = 1.0 / f32(dim);
  var wv: array<f32, 6>;
  for (var t = 0u; t < 6u; t = t + 1u) { wv[t] = weight[lid + t * 256u]; }
  var row = wid.x;
  while (row < params.rows) {
    let base = row * dim;
    var xv: array<f32, 6>;
    for (var t = 0u; t < 6u; t = t + 1u) { xv[t] = x[base + lid + t * 256u]; }
    var rv: array<f32, 6>;
    for (var t = 0u; t < 6u; t = t + 1u) { rv[t] = residual[base + lid + t * 256u]; }

    // sum of squares (1 pass)
    var acc = 0.0;
    var i = lid;
    while (i < dim) {
      let v = x[base + i];
      acc = acc + v * v;
      i = i + 256u;
    }
    scratch[lid] = acc;
    workgroupBarrier();
    var stride = 128u;
    while (stride > 0u) {
      if (lid < stride) {
        scratch[lid] = scratch[lid] + scratch[lid + stride];
      }
      workgroupBarrier();
      stride = stride / 2u;
    }
    let inv = inverseSqrt(scratch[0u] * scale + params.eps);
    // make every read of scratch[0] finish before the next row overwrites scratch[lid]
    workgroupBarrier();

    for (var t = 0u; t < 6u; t = t + 1u) {
      let o = lid + t * 256u;
      out[base + o] = bitcast<f32>(bitcast<u32>(xv[t] * inv * wv[t]) ^ params.rounding_mask) + rv[t];
    }
    row = row + nwg.x;
  }
}
