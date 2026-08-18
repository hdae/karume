// karume attention_state_qk (states 形の S 実体化, f32, sliding window, GQA)
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
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> ins_k: array<f32>;
@group(0) @binding(3) var<storage, read> slot_k: array<f32>;
@group(0) @binding(4) var<storage, read_write> s: array<f32>;
@group(0) @binding(5) var<uniform> lengths: Lengths;

fn slot_row(col: u32) -> u32 {
  return col % params.window;
}

fn column_base(past: u32) -> u32 {
  return past - min(past, params.window - 1u);
}

fn live_columns(past: u32, query: u32) -> u32 {
  return min(past, params.window - 1u) + query;
}

fn in_window(col: u32, limit: u32) -> bool {
  return col <= limit && (limit - col) < params.window;
}

fn effective_rows(query: u32) -> u32 {
  if (query <= params.row_offset) {
    return 0u;
  }
  return min(params.rows_block, query - params.row_offset);
}

fn score_slot(q_base: u32, k_base: u32) -> f32 {
  var acc = 0.0;
  for (var d = 0u; d < params.depth; d = d + 1u) {
    acc = acc + (q[q_base + d] * params.scale) * (slot_k[k_base + d] * params.scale);
  }
  return acc;
}

fn score_ins(q_base: u32, k_base: u32) -> f32 {
  var acc = 0.0;
  for (var d = 0u; d < params.depth; d = d + 1u) {
    acc = acc + (q[q_base + d] * params.scale) * (ins_k[k_base + d] * params.scale);
  }
  return acc;
}

@compute @workgroup_size(16, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let past = lengths.past;
  let live = live_columns(past, lengths.query);
  let local_row = gid.y;
  let cl = gid.x;
  // 端数タイルの空振り。live より右（[live, col_cap)）は残骸のまま残すのが正で、
  // 読者（②③）が live で切ることと対になっている。行は**有効行まで**（pad 行の S は
  // 誰も読まない — ③ が 0 を書いて返す）ので、仕事量が Q に比例する
  if (local_row >= effective_rows(lengths.query) || cl >= live) {
    return;
  }
  let z = gid.z;
  let col = column_base(past) + cl;
  let row = params.row_offset + local_row;
  let q_base = (z * params.chunk_rows + row) * params.depth;
  let kv_plane = z / params.kv_repeat;
  // 述語外は -inf。live 範囲は**述語外でも必ず書く**（書かないと ② が前回の残骸を食う）
  var value = bitcast<f32>(params.neg_inf);
  if (in_window(col, past + row)) {
    if (col < past) {
      // past（col < P）はスロットから。物理行は読み書き同式の slot_row
      value = score_slot(q_base, (kv_plane * params.capacity + slot_row(col)) * params.depth);
    } else {
      // current（col ≥ P）は今 step の ins の行 col − P から
      value = score_ins(q_base, (kv_plane * params.chunk_rows + (col - past)) * params.depth);
    }
  }
  s[(z * params.rows_block + local_row) * params.col_cap + cl] = value;
}
