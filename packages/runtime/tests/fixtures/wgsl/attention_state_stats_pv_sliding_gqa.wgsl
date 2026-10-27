// karume attention_state_stats_pv (states-form O = P @ V, f32, KV-parallel reduction, sliding window, GQA)
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
@group(0) @binding(3) var<storage, read> ins_v: array<f32>;
@group(0) @binding(4) var<storage, read> slot_v: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<uniform> lengths: Lengths;

fn slot_row(col: u32) -> u32 {
  return col % params.capacity;
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

fn is_nan_bits(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}
fn nan_max(a: f32, b: f32) -> f32 {
  return select(select(max(a, b), b, is_nan_bits(b)), a, is_nan_bits(a));
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
  // Pad rows (row ≥ Q) do not scan live and are exactly 0 (the same contract as ③). The local row is
  // workgroup-uniform, so returning before the barrier is allowed
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
  // Row statistics use the same indices and summation order as the original 256 lanes. A mapping
  // separate from the PV KV lanes.
  let flat = lid.y * 16u + lid.x;
  let neg_inf = bitcast<f32>(params.neg_inf);
  var hi = neg_inf;
  for (var i = flat; i < live; i = i + 256u) {
    hi = nan_max(hi, s[s_base + i]);
  }
  scratch[flat] = hi;
  workgroupBarrier();
  for (var step = 128u; step > 0u; step = step / 2u) {
    if (flat < step) {
      scratch[flat] = nan_max(scratch[flat], scratch[flat + step]);
    }
    workgroupBarrier();
  }
  let maximum = scratch[0u];
  workgroupBarrier();
  var sum = 0.0;
  if (maximum != neg_inf) {
    for (var i = flat; i < live; i = i + 256u) {
      sum = sum + exp(s[s_base + i] - maximum);
    }
  }
  scratch[flat] = sum;
  workgroupBarrier();
  for (var step = 128u; step > 0u; step = step / 2u) {
    if (flat < step) {
      scratch[flat] = scratch[flat] + scratch[flat + step];
    }
    workgroupBarrier();
  }
  var amax = 0.0;
  var inv = 0.0;
  if (maximum != neg_inf) {
    amax = maximum;
    inv = 1.0 / scratch[0u];
  }
  // Reuse scratch for the PV partial sums only after the denominator has been read.
  workgroupBarrier();
  // Per-lane partial sum (ascending col, stride KV_LANES). A lane whose d is out of range does not
  // scan and contributes 0 (it does not return, so that it takes part in the barrier)
  var acc = 0.0;
  if (in_depth) {
    // Batches of 8 columns of this lane: the V loads of a batch are issued before its
    // multiply-adds (the single-column loop waited on one V load per dependent step). The additions
    // stay one per column in ascending col, so the sum is the one the single-column loop builds
    var cl = lane;
    while (cl + 112u < live) {
      let col0 = base_col + cl;
      var v0 = 0.0;
      if (col0 < past) {
        v0 = slot_v[(kv_plane * params.capacity + slot_row(col0)) * params.depth + d];
      } else {
        v0 = ins_v[(kv_plane * params.chunk_rows + (col0 - past)) * params.depth + d];
      }
      let col1 = base_col + cl + 16u;
      var v1 = 0.0;
      if (col1 < past) {
        v1 = slot_v[(kv_plane * params.capacity + slot_row(col1)) * params.depth + d];
      } else {
        v1 = ins_v[(kv_plane * params.chunk_rows + (col1 - past)) * params.depth + d];
      }
      let col2 = base_col + cl + 32u;
      var v2 = 0.0;
      if (col2 < past) {
        v2 = slot_v[(kv_plane * params.capacity + slot_row(col2)) * params.depth + d];
      } else {
        v2 = ins_v[(kv_plane * params.chunk_rows + (col2 - past)) * params.depth + d];
      }
      let col3 = base_col + cl + 48u;
      var v3 = 0.0;
      if (col3 < past) {
        v3 = slot_v[(kv_plane * params.capacity + slot_row(col3)) * params.depth + d];
      } else {
        v3 = ins_v[(kv_plane * params.chunk_rows + (col3 - past)) * params.depth + d];
      }
      let col4 = base_col + cl + 64u;
      var v4 = 0.0;
      if (col4 < past) {
        v4 = slot_v[(kv_plane * params.capacity + slot_row(col4)) * params.depth + d];
      } else {
        v4 = ins_v[(kv_plane * params.chunk_rows + (col4 - past)) * params.depth + d];
      }
      let col5 = base_col + cl + 80u;
      var v5 = 0.0;
      if (col5 < past) {
        v5 = slot_v[(kv_plane * params.capacity + slot_row(col5)) * params.depth + d];
      } else {
        v5 = ins_v[(kv_plane * params.chunk_rows + (col5 - past)) * params.depth + d];
      }
      let col6 = base_col + cl + 96u;
      var v6 = 0.0;
      if (col6 < past) {
        v6 = slot_v[(kv_plane * params.capacity + slot_row(col6)) * params.depth + d];
      } else {
        v6 = ins_v[(kv_plane * params.chunk_rows + (col6 - past)) * params.depth + d];
      }
      let col7 = base_col + cl + 112u;
      var v7 = 0.0;
      if (col7 < past) {
        v7 = slot_v[(kv_plane * params.capacity + slot_row(col7)) * params.depth + d];
      } else {
        v7 = ins_v[(kv_plane * params.chunk_rows + (col7 - past)) * params.depth + d];
      }
      acc = acc + (exp(s[s_base + cl] - amax) * inv) * v0;
      acc = acc + (exp(s[s_base + cl + 16u] - amax) * inv) * v1;
      acc = acc + (exp(s[s_base + cl + 32u] - amax) * inv) * v2;
      acc = acc + (exp(s[s_base + cl + 48u] - amax) * inv) * v3;
      acc = acc + (exp(s[s_base + cl + 64u] - amax) * inv) * v4;
      acc = acc + (exp(s[s_base + cl + 80u] - amax) * inv) * v5;
      acc = acc + (exp(s[s_base + cl + 96u] - amax) * inv) * v6;
      acc = acc + (exp(s[s_base + cl + 112u] - amax) * inv) * v7;
      cl = cl + 128u;
    }
    // The last columns of this lane (fewer than 8), one at a time
    while (cl < live) {
      let col0 = base_col + cl;
      var v0 = 0.0;
      if (col0 < past) {
        v0 = slot_v[(kv_plane * params.capacity + slot_row(col0)) * params.depth + d];
      } else {
        v0 = ins_v[(kv_plane * params.chunk_rows + (col0 - past)) * params.depth + d];
      }
      acc = acc + (exp(s[s_base + cl] - amax) * inv) * v0;
      cl = cl + 16u;
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
  if (lane == 0u && in_depth) {
    out[at] = scratch[lid.x];
  }
}
