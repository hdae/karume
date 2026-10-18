// karume topk (last dim, f32 values + i32 indices, descending, min-index tie-break, -inf identity, k=1)
struct Params {
  rows: u32,
  dim: u32,
  neg_inf: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<u32>;
@group(0) @binding(2) var<storage, read_write> values: array<u32>;
@group(0) @binding(3) var<storage, read_write> indices: array<i32>;

fn f32_rank_key(bits: u32) -> u32 {
  let magnitude = bits & 0x7fffffffu;
  let zeroed = select(bits, 0u, magnitude == 0u);
  let ordered = select(zeroed ^ 0x80000000u, ~zeroed, (zeroed & 0x80000000u) != 0u);
  return select(ordered, 0xffffffffu, magnitude > 0x7f800000u);
}

fn rank_key_beats(kb: u32, ib: u32, ka: u32, ia: u32) -> bool {
  return kb > ka || (kb == ka && ib < ia);
}

var<workgroup> cand_value: array<u32, 32>;
var<workgroup> cand_index: array<u32, 32>;
var<workgroup> head_value: array<u32, 32>;
var<workgroup> head_index: array<u32, 32>;

@compute @workgroup_size(32)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let dim = params.dim;
  let neg_inf = f32_rank_key(params.neg_inf);
  let block = lid * 1u;
  var row = wid.x;
  while (row < params.rows) {
    let base = row * dim;
    // phase 1: lane-local top-k. The identity is the weakest in both value and index (sentinel
    // dim = no candidate), so a real element always wins — even an all -inf row gives k entries
    // from the smallest index
    for (var s = 0u; s < 1u; s = s + 1u) {
      cand_value[block + s] = neg_inf;
      cand_index[block + s] = dim;
    }
    var i = lid;
    while (i < dim) {
      let v = f32_rank_key(x[base + i]);
      // a candidate that cannot beat the tail (the weakest) is dropped here. Only a winner goes on
      // to the insertion that keeps the descending order
      if (rank_key_beats(v, i, cand_value[block + 0u], cand_index[block + 0u])) {
        var s = 0u;
        while (s > 0u && rank_key_beats(v, i, cand_value[block + s - 1u], cand_index[block + s - 1u])) {
          cand_value[block + s] = cand_value[block + s - 1u];
          cand_index[block + s] = cand_index[block + s - 1u];
          s = s - 1u;
        }
        cand_value[block + s] = v;
        cand_index[block + s] = i;
      }
      i = i + 32u;
    }
    // phase 2: merge. The maximum of the lanes' heads = the maximum of the remaining set (each
    // block is descending), so k rounds give values descending, ties by ascending index. The
    // cursor advances at most k-1 times, so block + cursor always stays inside the block
    var cursor = 0u;
    for (var r = 0u; r < 1u; r = r + 1u) {
      head_value[lid] = cand_value[block + cursor];
      head_index[lid] = cand_index[block + cursor];
      workgroupBarrier();
      var stride = 16u;
      while (stride > 0u) {
        if (lid < stride) {
          let other = head_value[lid + stride];
          let other_at = head_index[lid + stride];
          if (rank_key_beats(other, other_at, head_value[lid], head_index[lid])) {
            head_value[lid] = other;
            head_index[lid] = other_at;
          }
        }
        workgroupBarrier();
        stride = stride / 2u;
      }
      let won = head_index[0u];
      if (lid == 0u) {
        values[row * 1u + r] = x[base + won];
        indices[row * 1u + r] = i32(won);
      }
      // the scan splits work by i ≡ lid (mod 32), so the winning element's owner is won % 32
      if (won % 32u == lid) {
        cursor = cursor + 1u;
      }
      // make every read of won finish before the next round overwrites head
      workgroupBarrier();
    }
    row = row + nwg.x;
  }
}
