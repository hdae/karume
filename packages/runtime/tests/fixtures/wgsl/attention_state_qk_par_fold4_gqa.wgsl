// karume attention_state_qk (states-form S materialisation, f32, D-parallel reduction, 4 query heads per workgroup, GQA)
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

// The partial sum of head f, lane l, column x lives at (f * D_LANES + l) * FOLD_TILE_X + x
var<workgroup> scratch: array<f32, 512>;

@compute @workgroup_size(8, 16)
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
  let cl = wid.x * 8u + lid.x;
  let lane = lid.y;
  // The heads z .. z + 3 of this workgroup (z = the group's first head). 4 divides
  // kv_repeat (checked by the host), so the kv plane of z is the kv plane of every one of them
  let z = wid.z * 4u;
  let col = column_base(past) + cl;
  let row = params.row_offset + local_row;
  let q_base0 = ((z + 0u) * params.chunk_rows + row) * params.depth;
  let q_base1 = ((z + 1u) * params.chunk_rows + row) * params.depth;
  let q_base2 = ((z + 2u) * params.chunk_rows + row) * params.depth;
  let q_base3 = ((z + 3u) * params.chunk_rows + row) * params.depth;
  let kv_plane = z / params.kv_repeat;
  // Edge-tile columns (cl ≥ live) and columns outside the predicate do not run the dot product.
  // **No return** — this branch is not workgroup-uniform, and the barrier below can sit only inside
  // uniform control flow (contributes 0.0)
  let inside = cl < live && in_window(col, past + row);
  var acc0 = 0.0;
  var acc1 = 0.0;
  var acc2 = 0.0;
  var acc3 = 0.0;
  if (inside) {
    // Each K element is loaded once and multiplied into the 4 heads. Per head the terms and
    // their order are ①''s (d ascending from lane, step D_LANES — the same one-term expression)
    if (col < past) {
      // past (col < P) comes from the slot. The physical row is slot_row (same formula for reads
      // and writes)
      let k_base = (kv_plane * params.capacity + slot_row(col)) * params.depth;
      for (var d = lane; d < params.depth; d = d + 16u) {
        let kd = slot_k[k_base + d] * params.scale;
        acc0 = acc0 + (q[q_base0 + d] * params.scale) * kd;
        acc1 = acc1 + (q[q_base1 + d] * params.scale) * kd;
        acc2 = acc2 + (q[q_base2 + d] * params.scale) * kd;
        acc3 = acc3 + (q[q_base3 + d] * params.scale) * kd;
      }
    } else {
      // current (col ≥ P) comes from row col − P of this step's ins
      let k_base = (kv_plane * params.chunk_rows + (col - past)) * params.depth;
      for (var d = lane; d < params.depth; d = d + 16u) {
        let kd = ins_k[k_base + d] * params.scale;
        acc0 = acc0 + (q[q_base0 + d] * params.scale) * kd;
        acc1 = acc1 + (q[q_base1 + d] * params.scale) * kd;
        acc2 = acc2 + (q[q_base2 + d] * params.scale) * kd;
        acc3 = acc3 + (q[q_base3 + d] * params.scale) * kd;
      }
    }
  }
  scratch[(0u + lane) * 8u + lid.x] = acc0;
  scratch[(16u + lane) * 8u + lid.x] = acc1;
  scratch[(32u + lane) * 8u + lid.x] = acc2;
  scratch[(48u + lane) * 8u + lid.x] = acc3;
  workgroupBarrier();
  // Fixed-order tree reduction per head (stride 8 → 4 → 2 → 1) — the same tree as ①'
  var stride = 8u;
  while (stride > 0u) {
    if (lane < stride) {
      let mine0 = (0u + lane) * 8u + lid.x;
      scratch[mine0] = scratch[mine0] + scratch[mine0 + stride * 8u];
      let mine1 = (16u + lane) * 8u + lid.x;
      scratch[mine1] = scratch[mine1] + scratch[mine1 + stride * 8u];
      let mine2 = (32u + lane) * 8u + lid.x;
      scratch[mine2] = scratch[mine2] + scratch[mine2 + stride * 8u];
      let mine3 = (48u + lane) * 8u + lid.x;
      scratch[mine3] = scratch[mine3] + scratch[mine3 + stride * 8u];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  // -inf outside the predicate. The live range is **always written, even outside the predicate**
  // (otherwise ② consumes the previous leftovers). Lane f writes head f's word, the root of head f's
  // tree (its lane-0 slot)
  if (lane < 4u && cl < live) {
    var value = bitcast<f32>(params.neg_inf);
    if (inside) {
      value = scratch[lane * 128u + lid.x];
    }
    s[((z + lane) * params.rows_block + local_row) * params.col_cap + cl] = value;
  }
}
