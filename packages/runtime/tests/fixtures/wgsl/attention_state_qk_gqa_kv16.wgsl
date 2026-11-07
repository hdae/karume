// karume attention_state_qk (states-form S materialisation, f32, GQA, KV in f16 storage (unpack2x16float))
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
@group(0) @binding(3) var<storage, read> slot_k: array<u32>;
@group(0) @binding(4) var<storage, read_write> s: array<f32>;
@group(0) @binding(5) var<uniform> lengths: Lengths;

fn slot_row(col: u32) -> u32 {
  return col;
}

fn column_base(past: u32) -> u32 {
  return 0u;
}

fn live_columns(past: u32, query: u32) -> u32 {
  return past + query;
}

fn in_window(col: u32, limit: u32) -> bool {
  return col <= limit;
}

fn effective_rows(query: u32) -> u32 {
  if (query <= params.row_offset) {
    return 0u;
  }
  return min(params.rows_block, query - params.row_offset);
}

// Decode of f16 storage: element i = unpack2x16float(slot_k[i / 2])[i % 2] (unpack is exact)
fn slot_k_at(i: u32) -> f32 {
  let pair = unpack2x16float(slot_k[i >> 1u]);
  return select(pair.x, pair.y, (i & 1u) == 1u);
}

fn score_slot(q_base: u32, k_base: u32) -> f32 {
  var acc = 0.0;
  for (var d = 0u; d < params.depth; d = d + 1u) {
    acc = acc + (q[q_base + d] * params.scale) * (slot_k_at(k_base + d) * params.scale);
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
  // Empty run of the edge tile. Leaving the leftovers right of live ([live, col_cap)) in place is
  // correct and pairs with the readers (②③) cutting at live. Rows go **up to the valid rows** (no
  // one reads the S of pad rows — ③ writes 0 and returns), so the work is proportional to Q
  if (local_row >= effective_rows(lengths.query) || cl >= live) {
    return;
  }
  let z = gid.z;
  let col = column_base(past) + cl;
  let row = params.row_offset + local_row;
  let q_base = (z * params.chunk_rows + row) * params.depth;
  let kv_plane = z / params.kv_repeat;
  // -inf outside the predicate. The live range is **always written, even outside the predicate**
  // (otherwise ② consumes the previous leftovers)
  var value = bitcast<f32>(params.neg_inf);
  if (in_window(col, past + row)) {
    if (col < past) {
      // past (col < P) comes from the slot. The physical row is slot_row (same formula for reads
      // and writes)
      value = score_slot(q_base, (kv_plane * params.capacity + slot_row(col)) * params.depth);
    } else {
      // current (col ≥ P) comes from row col − P of this step's ins
      value = score_ins(q_base, (kv_plane * params.chunk_rows + (col - past)) * params.depth);
    }
  }
  s[(z * params.rows_block + local_row) * params.col_cap + cl] = value;
}
