// karume argmax (last dim, f32>i32, min-index tie-break, -inf identity)
struct Params {
  rows: u32,
  dim: u32,
  neg_inf: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<i32>;

fn is_nan_bits(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

fn argmax_beats(vb: f32, ib: u32, va: f32, ia: u32) -> bool {
  let na = is_nan_bits(va);
  let nb = is_nan_bits(vb);
  if (na != nb) {
    return nb;
  }
  if (na) {
    return ib < ia;
  }
  return vb > va || (vb == va && ib < ia);
}

var<workgroup> scratch_value: array<f32, 256>;
var<workgroup> scratch_index: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let dim = params.dim;
  let neg_inf = bitcast<f32>(params.neg_inf);
  var row = wid.x;
  while (row < params.rows) {
    let base = row * dim;
    // identity は値も index も最弱（番兵 dim = 候補なし）— 全 -inf 行でも最初の実要素が
    // index 比較で勝つので、結果は決定的に最小 index になる
    var best = neg_inf;
    var best_at = dim;
    var i = lid;
    while (i < dim) {
      let v = x[base + i];
      if (argmax_beats(v, i, best, best_at)) {
        best = v;
        best_at = i;
      }
      i = i + 256u;
    }
    scratch_value[lid] = best;
    scratch_index[lid] = best_at;
    workgroupBarrier();
    var stride = 128u;
    while (stride > 0u) {
      if (lid < stride) {
        let other = scratch_value[lid + stride];
        let other_at = scratch_index[lid + stride];
        if (argmax_beats(other, other_at, scratch_value[lid], scratch_index[lid])) {
          scratch_value[lid] = other;
          scratch_index[lid] = other_at;
        }
      }
      workgroupBarrier();
      stride = stride / 2u;
    }
    if (lid == 0u) {
      out[row] = i32(scratch_index[0u]);
    }
    // 次の行が scratch を上書きする前に scratch[0] の読み終わりを揃える
    workgroupBarrier();
    row = row + nwg.x;
  }
}
