// karume attention_state_qk (states-form S materialisation, f32, D-parallel reduction, sliding window, GQA)
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
  return col % params.capacity;
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

fn score_slot(q_base: u32, k_base: u32, lane: u32) -> f32 {
  var acc = 0.0;
  for (var d = lane; d < params.depth; d = d + 16u) {
    acc = acc + (q[q_base + d] * params.scale) * (slot_k[k_base + d] * params.scale);
  }
  return acc;
}

fn score_ins(q_base: u32, k_base: u32, lane: u32) -> f32 {
  var acc = 0.0;
  for (var d = lane; d < params.depth; d = d + 16u) {
    acc = acc + (q[q_base + d] * params.scale) * (ins_k[k_base + d] * params.scale);
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
  // No one reads the S of pad rows (③ writes 0 and returns), so return without writing a word. The
  // local row is workgroup-uniform, so returning before the barrier is allowed
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
  // Edge-tile columns (cl ≥ live) and columns outside the predicate do not run the dot product.
  // **No return** — this branch is not workgroup-uniform, and the barrier below can sit only inside
  // uniform control flow (contributes 0.0)
  let inside = cl < live && in_window(col, past + row);
  var acc = 0.0;
  if (inside) {
    if (col < past) {
      // past (col < P) comes from the slot. The physical row is slot_row (same formula for reads
      // and writes)
      acc = score_slot(q_base, (kv_plane * params.capacity + slot_row(col)) * params.depth, lane);
    } else {
      // current (col ≥ P) comes from row col − P of this step's ins
      acc = score_ins(q_base, (kv_plane * params.chunk_rows + (col - past)) * params.depth, lane);
    }
  }
  scratch[lane * 16u + lid.x] = acc;
  workgroupBarrier();
  // Fixed-order tree reduction (stride 8 → 4 → 2 → 1) — the basis of determinism
  var stride = 8u;
  while (stride > 0u) {
    if (lane < stride) {
      let mine = lane * 16u + lid.x;
      scratch[mine] = scratch[mine] + scratch[(lane + stride) * 16u + lid.x];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  // -inf outside the predicate. The live range is **always written, even outside the predicate**
  // (otherwise ② consumes the previous leftovers)
  if (lane == 0u && cl < live) {
    var value = bitcast<f32>(params.neg_inf);
    if (inside) {
      value = scratch[lid.x];
    }
    s[(z * params.rows_block + local_row) * params.col_cap + cl] = value;
  }
}
