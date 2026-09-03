// karume attention_state_pv (states 形の O = P @ V, f32, KV 並列縮約, sliding window, GQA)
struct Params {
  rows_block: u32,
  row_offset: u32,
  chunk_rows: u32,
  depth: u32,
  kv_repeat: u32,
  window: u32,
  capacity: u32,
  col_cap: u32,
  neg_inf: u32,
  scale: f32,
}
struct Lengths {
  past: u32,
  query: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read> ins_v: array<f32>;
@group(0) @binding(4) var<storage, read> slot_v: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<uniform> lengths: Lengths;

fn slot_row(col: u32) -> u32 {
  return col % params.window;
}

fn column_base(past: u32) -> u32 {
  return past - min(past, params.window - 1u);
}

fn live_columns(past: u32, query: u32) -> u32 {
  return min(past, params.window - 1u) + query;
}

fn effective_rows(query: u32) -> u32 {
  if (query <= params.row_offset) {
    return 0u;
  }
  return min(params.rows_block, query - params.row_offset);
}

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let d = wid.x * 16u + lid.x;
  let local_row = wid.y;
  let lane = lid.y;
  let z = wid.z;
  let in_depth = d < params.depth;
  let at = (z * params.chunk_rows + params.row_offset + local_row) * params.depth + d;
  // pad 行（row ≥ Q）は live を走査せず厳密 0（③ と同じ契約）。局所行は workgroup 一様なので
  // barrier の手前で返してよい
  if (local_row >= effective_rows(lengths.query)) {
    if (in_depth) {
      out[at] = 0.0;
    }
    return;
  }
  let past = lengths.past;
  let live = live_columns(past, lengths.query);
  let base_col = column_base(past);
  let kv_plane = z / params.kv_repeat;
  let s_row = z * params.rows_block + local_row;
  let s_base = s_row * params.col_cap;
  let amax = stats[s_row * 2u];
  let inv = stats[s_row * 2u + 1u];
  // レーンごとの部分和（col 昇順・stride KV_LANES）。d が範囲外のレーンは走査せず 0 を寄与する
  // （barrier に参加させるため return しない）
  var acc = 0.0;
  if (in_depth) {
    for (var cl = lane; cl < live; cl = cl + 16u) {
      let col = base_col + cl;
      let p = exp(s[s_base + cl] - amax) * inv;
      var value = 0.0;
      if (col < past) {
        value = slot_v[(kv_plane * params.capacity + slot_row(col)) * params.depth + d];
      } else {
        value = ins_v[(kv_plane * params.chunk_rows + (col - past)) * params.depth + d];
      }
      acc = acc + p * value;
    }
  }
  scratch[lane * 16u + lid.x] = acc;
  workgroupBarrier();
  // 固定順の木縮約（stride 8 → 4 → 2 → 1）— 決定性の根拠
  var stride = 8u;
  while (stride > 0u) {
    if (lane < stride) {
      let mine = lane * 16u + lid.x;
      scratch[mine] = scratch[mine] + scratch[(lane + stride) * 16u + lid.x];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (lane == 0u && in_depth) {
    out[at] = scratch[lid.x];
  }
}
