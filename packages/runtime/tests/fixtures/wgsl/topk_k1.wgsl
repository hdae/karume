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
    // 相 1: レーン局所 top-k。identity は値も index も最弱（番兵 dim = 候補なし）なので、
    // 実要素は必ず勝つ — 全 -inf 行でも答えが最小 index から k 本に決まる
    for (var s = 0u; s < 1u; s = s + 1u) {
      cand_value[block + s] = neg_inf;
      cand_index[block + s] = dim;
    }
    var i = lid;
    while (i < dim) {
      let v = f32_rank_key(x[base + i]);
      // 末尾（最弱）に勝てない候補はここで捨てる。勝つ候補だけが降順を保つ挿入へ進む
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
    // 相 2: merge。各レーンの先頭同士の最大元 = 残り集合の最大元（各ブロックが降順）なので、
    // k ラウンド回せば値降順・同値なら index 昇順で出る。カーソルは k-1 回しか進まないので
    // block + cursor は常にブロック内
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
      // 走査は i ≡ lid (mod 32) の分担なので、勝った要素の持ち主は won % 32 で決まる
      if (won % 32u == lid) {
        cursor = cursor + 1u;
      }
      // 次ラウンドが head を上書きする前に won の読み終わりを揃える
      workgroupBarrier();
    }
    row = row + nwg.x;
  }
}
