// karume attention_stats (per-row m = amax(S) and inv = 1/Σexp(S - m), f32, S in f16 storage (pack2x16float))
struct Params {
  rows: u32,
  dim: u32,
  neg_inf: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> s: array<u32>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;

// Decode of f16 storage: element i = unpack2x16float(s[i / 2])[i % 2]
// (the parity of the flat index picks the half of the pair)
fn score_at(i: u32) -> f32 {
  let pair = unpack2x16float(s[i >> 1u]);
  return select(pair.x, pair.y, (i & 1u) == 1u);
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
  let dim = params.dim;
  let neg_inf = bitcast<f32>(params.neg_inf);
  var row = wid.x;
  while (row < params.rows) {
    let base = row * dim;

    // ① Row maximum (the subtracted term of safe-softmax — identity is -inf)
    var hi = neg_inf;
    var i = lid;
    while (i < dim) {
      hi = nan_max(hi, score_at(base + i));
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
    // A row with every element -inf (no key visible) — make the subtracted term 0 so no NaN is
    // produced (same form as safe_softmax)
    let empty = row_max == neg_inf;
    let amax = select(row_max, 0.0, empty);
    // Make sure scratch has been read before ② overwrites it
    workgroupBarrier();

    // ② Σ exp(S - amax) (the largest element gives exp(0) = 1, so the denominator is at least 1)
    var acc = 0.0;
    var j = lid;
    while (j < dim) {
      acc = acc + exp(score_at(base + j) - amax);
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
    // MUST: the reciprocal is built here (going back to a division in ③ would change the operation
    // versus softmax pass ③)
    let inv = 1.0 / scratch[0u];
    if (lid == 0u) {
      // An empty row is (0.0, 0.0) — ③ multiplies exp(S - 0) = exp(-inf) = 0 by inv = 0, so the
      // output is exactly 0
      stats[row * 2u] = amax;
      stats[row * 2u + 1u] = select(inv, 0.0, empty);
    }
    // Make sure scratch[0] has been read before the next row overwrites scratch[lid]
    workgroupBarrier();
    row = row + nwg.x;
  }
}
