// karume adaLN (layer_norm -> 1 + scale -> mul -> add, f32, staged modulation)
struct Params {
  rows: u32,
  dim: u32,
  eps: f32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> scale_vec: array<f32>;
@group(0) @binding(5) var<storage, read> one: array<f32>;
@group(0) @binding(6) var<storage, read> shift: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;

var<workgroup> scratch: array<f32, 256>;
var<workgroup> staged: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let dim = params.dim;
  let scale = 1.0 / f32(dim);
  // NOTE: the ceiling is not written as (dim + 255) / 256 — it overflows at dim = u32::MAX.
  let blocks = dim / 256u +
    select(0u, 1u, dim % 256u != 0u);
  var row = wid.x;
  while (row < params.rows) {
    let base = row * dim;

    // ① sum → mean
    var acc = 0.0;
    var i = lid;
    while (i < dim) {
      acc = acc + x[base + i];
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
    let mean = scratch[0u] * scale;
    // make every read of scratch finish before ② overwrites it
    workgroupBarrier();

    // ② sum of squared deviations → population variance (correction = 0)
    var sq = 0.0;
    var j = lid;
    while (j < dim) {
      let d = x[base + j] - mean;
      sq = sq + d * d;
      j = j + 256u;
    }
    scratch[lid] = sq;
    workgroupBarrier();
    var stride2 = 128u;
    while (stride2 > 0u) {
      if (lid < stride2) {
        scratch[lid] = scratch[lid] + scratch[lid + stride2];
      }
      workgroupBarrier();
      stride2 = stride2 / 2u;
    }
    let inv = 1.0 / sqrt(scratch[0u] * scale + params.eps);
    // make every read of scratch[0] finish before the next row overwrites scratch[lid]
    workgroupBarrier();

    var block = 0u;
    while (block < blocks) {
      let o = block * 256u + lid;
      // ③ materialize the modulation factor 1 + scale (where the add node writes to storage in the
      // plain sequence)
      if (o < dim) {
        staged[lid] = bitcast<u32>(scale_vec[o] + one[0u]);
      }
      workgroupBarrier();
      // ④ normalisation (the same expression as the plain layer_norm) → multiply. Rounded at the
      // same point as the plain mul.
      if (o < dim) {
        let modulation = bitcast<f32>(staged[lid]);
        staged[lid] = bitcast<u32>(((x[base + o] - mean) * inv * weight[o] + bias[o]) * modulation);
      }
      workgroupBarrier();
      // ⑤ add shift (the plain add node)
      if (o < dim) {
        out[base + o] = bitcast<f32>(staged[lid]) + shift[o];
      }
      workgroupBarrier();
      block = block + 1u;
    }
    row = row + nwg.x;
  }
}
