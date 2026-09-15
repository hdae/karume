// karume attention_state_stats_pv (states 形の O = P @ V, f32, KV 並列縮約, GQA)
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
  // pad 行（row ≥ Q）は live を走査せず厳密 0（③ と同じ契約）。局所行は workgroup 一様なので
  // barrier の手前で返してよい
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
  // 行統計は元の 256 lane と同じ添字・加算順。PV の KV lane とは別の写像。
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
  // 分母を読み終えてから scratch を PV の部分和へ使い回す。
  workgroupBarrier();
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
