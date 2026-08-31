// karume attention_state_stats (states 形の行統計 m = amax(S) と inv = 1/Σexp(S - m), f32, sliding window)
struct Params {
  batch_heads: u32,
  rows_block: u32,
  row_offset: u32,
  col_cap: u32,
  window: u32,
  neg_inf: u32,
}
struct Lengths {
  past: u32,
  query: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;
@group(0) @binding(3) var<uniform> lengths: Lengths;

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

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let neg_inf = bitcast<f32>(params.neg_inf);
  let live = live_columns(lengths.past, lengths.query);
  // 有効行は各 z 平面の**前詰め** rows 本。total = 0 ならループへ入らないので rows での
  // 除算・剰余は 0 除算にならない（ホストも 0 なら dispatch を積まない）
  let rows = effective_rows(lengths.query);
  let total = params.batch_heads * rows;
  var index = wid.x;
  while (index < total) {
    let row = (index / rows) * params.rows_block + index % rows;
    let base = row * params.col_cap;

    // ① 行の最大値。identity は **-inf**（有限 sentinel は MUST NOT — ADR 0067 決定 6）
    var hi = neg_inf;
    var i = lid;
    while (i < live) {
      hi = nan_max(hi, s[base + i]);
      i = i + 256u;
    }
    scratch[lid] = hi;
    workgroupBarrier();
    var stride = 128u;
    while (stride > 0u) {
      if (lid < stride) {
        scratch[lid] = nan_max(scratch[lid], scratch[lid + stride]);
      }
      workgroupBarrier();
      stride = stride / 2u;
    }
    let amax = scratch[0u];
    // scratch の読み終わりを揃えてから ② で上書きする
    workgroupBarrier();

    // ② Σ exp(S - amax)。**空行（amax == -inf）は 1 度も回さない** —
    // exp(-inf - (-inf)) = exp(NaN) = NaN が分母へ入る
    let empty = amax == neg_inf;
    var acc = 0.0;
    if (!empty) {
      var j = lid;
      while (j < live) {
        acc = acc + exp(s[base + j] - amax);
        j = j + 256u;
      }
    }
    scratch[lid] = acc;
    workgroupBarrier();
    var stride2 = 128u;
    while (stride2 > 0u) {
      if (lid < stride2) {
        scratch[lid] = scratch[lid] + scratch[lid + stride2];
      }
      workgroupBarrier();
      stride2 = stride2 / 2u;
    }
    if (lid == 0u) {
      // 空行は (0.0, 0.0)。③ の exp(-inf - 0) * 0 = 0 で出力が**厳密 0** になる
      var m = 0.0;
      var inv = 0.0;
      if (!empty) {
        m = amax;
        inv = 1.0 / scratch[0u];
      }
      stats[row * 2u] = m;
      stats[row * 2u + 1u] = inv;
    }
    // 次の行が scratch[lid] を上書きする前に scratch[0] の読み終わりを揃える
    workgroupBarrier();
    index = index + nwg.x;
  }
}
