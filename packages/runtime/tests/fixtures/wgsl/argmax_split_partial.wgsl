// karume argmax split partial (last dim, f32>i32, min-index tie-break, -inf identity)
struct Params {
  rows: u32,
  dim: u32,
  groups: u32,
  neg_inf: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<u32>;
@group(0) @binding(2) var<storage, read_write> partial: array<u32>;

fn f32_rank_key(bits: u32) -> u32 {
  let magnitude = bits & 0x7fffffffu;
  let zeroed = select(bits, 0u, magnitude == 0u);
  let ordered = select(zeroed ^ 0x80000000u, ~zeroed, (zeroed & 0x80000000u) != 0u);
  return select(ordered, 0xffffffffu, magnitude > 0x7f800000u);
}

fn rank_key_beats(kb: u32, ib: u32, ka: u32, ia: u32) -> bool {
  return kb > ka || (kb == ka && ib < ia);
}

var<workgroup> scratch_value: array<u32, 256>;
var<workgroup> scratch_index: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let dim = params.dim;
  let neg_inf = f32_rank_key(params.neg_inf);
  // x 軸 = 区間（1 workgroup = 1 区間・上限超過は fail loudly）・y 軸 = 行（grid-stride）
  let group = wid.x;
  let start = group * 4096u;
  let end = min(start + 4096u, dim);
  var row = wid.y;
  while (row < params.rows) {
    let base = row * dim;
    var best = neg_inf;
    var best_at = dim;
    var i = start + lid;
    while (i < end) {
      let v = f32_rank_key(x[base + i]);
      if (rank_key_beats(v, i, best, best_at)) {
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
        if (rank_key_beats(other, other_at, scratch_value[lid], scratch_index[lid])) {
          scratch_value[lid] = other;
          scratch_index[lid] = other_at;
        }
      }
      workgroupBarrier();
      stride = stride / 2u;
    }
    if (lid == 0u) {
      let at = (row * params.groups + group) * 2u;
      partial[at] = scratch_value[0u];
      partial[at + 1u] = scratch_index[0u];
    }
    workgroupBarrier();
    row = row + nwg.y;
  }
}
