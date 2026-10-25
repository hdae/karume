// karume rms_norm → RoPE (pair rotation, token-major, f32, full-write) — the normalised row is never stored
struct Params {
  rows: u32,
  dim: u32,
  eps: f32,
  head_dim: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> cos_table: array<f32>;
@group(0) @binding(4) var<storage, read> sin_table: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

var<workgroup> scratch: array<f32, 256>;
var<workgroup> products: array<vec2<u32>, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let dim = params.dim;
  let scale = 1.0 / f32(dim);
  var row = wid.x;
  while (row < params.rows) {
    let base = row * dim;

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

    // RoPE on the normalised row, 256 elements per chunk (a workgroup-uniform loop: the
    // barriers below need every invocation). Every element of this row belongs to token `row`.
    let token = row;
    var chunk = 0u;
    while (chunk < dim) {
      let o = chunk + lid;
      let i = base + o;
      if (o < dim) {
        let d = o % params.head_dim;
      var rotated_bits: u32;
      if ((i & 1u) == 0u) {
        rotated_bits = bitcast<u32>((x[base + o + 1u] * inv * weight[o + 1u])) ^ 0x80000000u;
      } else {
        rotated_bits = bitcast<u32>((x[base + o - 1u] * inv * weight[o - 1u]));
      }
      let table_index = token * params.head_dim + d;
      let direct_bits = bitcast<u32>((x[base + o] * inv * weight[o]) * cos_table[table_index]);
      let cross_bits = bitcast<u32>(bitcast<f32>(rotated_bits) * sin_table[table_index]);
      products[lid] = vec2<u32>(direct_bits, cross_bits);
      } else {
        products[lid] = vec2<u32>(0u);
      }
      workgroupBarrier();
      if (o < dim) {
        let pair = products[lid];
        out[i] = bitcast<f32>(pair.x) + bitcast<f32>(pair.y);
      }
      workgroupBarrier();
      chunk = chunk + 256u;
    }
    row = row + nwg.x;
  }
}
