// karume attention_state_pv (states 形の O = P @ V, f32, GQA)
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
  return col;
}

fn column_base(past: u32) -> u32 {
  return 0u;
}

fn live_columns(past: u32, query: u32) -> u32 {
  return past + query;
}

fn effective_rows(query: u32) -> u32 {
  if (query <= params.row_offset) {
    return 0u;
  }
  return min(params.rows_block, query - params.row_offset);
}

@compute @workgroup_size(16, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let d = gid.x;
  let local_row = gid.y;
  if (d >= params.depth || local_row >= params.rows_block) {
    return;
  }
  let z = gid.z;
  let at = (z * params.chunk_rows + params.row_offset + local_row) * params.depth + d;
  // pad 行（row ≥ Q）: live を走査せず**厳密 0** を書いて返す（full-write は保つ = ADR 0066
  // 追記 6 の「値が契約上無意味」を 0 で固定）。空行 ⊂ pad 行なので ADR 0067 決定 6 の
  // 「空行 → 厳密 0」はこの分岐が構造的に包含し、非有限 V による 0·NaN の穴も同時に閉じる
  if (local_row >= effective_rows(lengths.query)) {
    out[at] = 0.0;
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
  // 縮約は col 昇順の逐次（決定性）。述語外は S が -inf なので p = 0 が**厳密**に出る
  var acc = 0.0;
  for (var cl = 0u; cl < live; cl = cl + 1u) {
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
  out[at] = acc;
}
