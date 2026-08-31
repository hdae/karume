// karume attention_stats (行ごとの m = amax(S) と inv = 1/Σexp(S - m), f32)
struct Params {
  rows: u32,
  dim: u32,
  neg_inf: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;

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
  let dim = params.dim;
  let neg_inf = bitcast<f32>(params.neg_inf);
  var row = wid.x;
  while (row < params.rows) {
    let base = row * dim;

    // ① 行の最大値（safe-softmax の減算項 — identity は -inf）
    var hi = neg_inf;
    var i = lid;
    while (i < dim) {
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
    let row_max = scratch[0u];
    // 全要素 -inf の行（全マスク行）— 減算項を 0 にして NaN を作らない（safe_softmax と同形）
    let empty = row_max == neg_inf;
    let amax = select(row_max, 0.0, empty);
    // scratch の読み終わりを揃えてから ② で上書きする
    workgroupBarrier();

    // ② Σ exp(S - amax)（最大要素が exp(0) = 1 を出すので分母は必ず 1 以上）
    var acc = 0.0;
    var j = lid;
    while (j < dim) {
      acc = acc + exp(s[base + j] - amax);
      j = j + 256u;
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
    // MUST: 逆数はここで作る（③ で割り算に戻すと softmax のパス③と演算が変わる）
    let inv = 1.0 / scratch[0u];
    if (lid == 0u) {
      // 空行は (0.0, 0.0) — ③ の exp(S - 0) = exp(-inf) = 0 に inv = 0 が掛かり出力が厳密 0
      stats[row * 2u] = amax;
      stats[row * 2u + 1u] = select(inv, 0.0, empty);
    }
    // 次の行が scratch[lid] を上書きする前に scratch[0] の読み終わりを揃える
    workgroupBarrier();
    row = row + nwg.x;
  }
}
