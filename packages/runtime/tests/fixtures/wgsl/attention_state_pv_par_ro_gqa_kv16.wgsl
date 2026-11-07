// karume attention_state_pv (states-form O = P @ V, f32, KV-parallel reduction, readonly = past only, GQA, KV in f16 storage (unpack2x16float))
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
@group(0) @binding(3) var<storage, read> slot_v: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> lengths: Lengths;

fn slot_row(col: u32) -> u32 {
  return col;
}

fn column_base(past: u32) -> u32 {
  return 0u;
}

fn live_columns(past: u32, query: u32) -> u32 {
  return past;
}

fn effective_rows(query: u32) -> u32 {
  if (query <= params.row_offset) {
    return 0u;
  }
  return min(params.rows_block, query - params.row_offset);
}

// Decode of f16 storage: element i = unpack2x16float(slot_v[i / 2])[i % 2] (unpack is exact)
fn slot_v_at(i: u32) -> f32 {
  let pair = unpack2x16float(slot_v[i >> 1u]);
  return select(pair.x, pair.y, (i & 1u) == 1u);
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
  if (local_row >= effective_rows(lengths.query)) {
    if (lane == 0u && in_depth) {
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
  // Per-lane partial sum (ascending col, stride KV_LANES). Only the slot (past) is read
  var acc = 0.0;
  if (in_depth) {
    for (var cl = lane; cl < live; cl = cl + 16u) {
      let col = base_col + cl;
      let p = exp(s[s_base + cl] - amax) * inv;
      acc = acc + p * slot_v_at((kv_plane * params.capacity + slot_row(col)) * params.depth + d);
    }
  }
  scratch[lane * 16u + lid.x] = acc;
  workgroupBarrier();
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
