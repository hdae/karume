// karume attention_state_qk (states 形の S 実体化, f32, D 並列縮約, readonly = past のみ, sliding window, GQA)
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
@group(0) @binding(2) var<storage, read> slot_k: array<f32>;
@group(0) @binding(3) var<storage, read_write> s: array<f32>;
@group(0) @binding(4) var<uniform> lengths: Lengths;

fn slot_row(col: u32) -> u32 {
  return col % params.capacity;
}

fn column_base(past: u32) -> u32 {
  return past - min(past, params.window);
}

fn live_columns(past: u32, query: u32) -> u32 {
  return min(past, params.window);
}

fn in_window(col: u32, past: u32) -> bool {
  return col < past && (past - 1u - col) < params.window;
}

fn effective_rows(query: u32) -> u32 {
  if (query <= params.row_offset) {
    return 0u;
  }
  return min(params.rows_block, query - params.row_offset);
}

fn score_slot(q_base: u32, k_base: u32, lane: u32) -> f32 {
  var acc = 0.0;
  for (var d = lane; d < params.depth; d = d + 16u) {
    acc = acc + (q[q_base + d] * params.scale) * (slot_k[k_base + d] * params.scale);
  }
  return acc;
}

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let local_row = wid.y;
  if (local_row >= effective_rows(lengths.query)) {
    return;
  }
  let past = lengths.past;
  let live = live_columns(past, lengths.query);
  let cl = wid.x * 16u + lid.x;
  let lane = lid.y;
  let z = wid.z;
  let col = column_base(past) + cl;
  let row = params.row_offset + local_row;
  let q_base = (z * params.chunk_rows + row) * params.depth;
  let kv_plane = z / params.kv_repeat;
  // 読むのはスロット（past）だけ。端数タイル（cl ≥ live）は内積を回さず **return もしない**
  // （下の barrier は一様制御流の中だけ）
  let inside = cl < live && in_window(col, past);
  var acc = 0.0;
  if (inside) {
    acc = score_slot(q_base, (kv_plane * params.capacity + slot_row(col)) * params.depth, lane);
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
  // 述語外は -inf。live 範囲は**述語外でも必ず書く**（書かないと ② が前回の残骸を食う）
  if (lane == 0u && cl < live) {
    var value = bitcast<f32>(params.neg_inf);
    if (inside) {
      value = scratch[lid.x];
    }
    s[(z * params.rows_block + local_row) * params.col_cap + cl] = value;
  }
}
