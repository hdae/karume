// karume softmax (last dim, f32, safe-softmax: exp(x - amax))
struct Params {
  rows: u32,
  dim: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

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
  var row = wid.x;
  while (row < params.rows) {
    let base = row * dim;

    // ① row maximum (the subtrahend of safe-softmax)
    var hi = -3.402823466e38;
    var i = lid;
    while (i < dim) {
      hi = nan_max(hi, x[base + i]);
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
    // make every read of scratch finish before ② overwrites it
    workgroupBarrier();

    // ② Σ exp(x - amax) (the largest element yields exp(0) = 1, so the denominator is always ≥ 1)
    var acc = 0.0;
    var j = lid;
    while (j < dim) {
      acc = acc + exp(x[base + j] - amax);
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
    let inv = 1.0 / scratch[0u];
    // make every read of scratch[0] finish before the next row overwrites scratch[lid]
    workgroupBarrier();

    // ③ write-out (exp takes the same argument as in ② — it returns the same value, so it is
    // deterministic)
    var o = lid;
    while (o < dim) {
      out[base + o] = exp(x[base + o] - amax) * inv;
      o = o + 256u;
    }
    row = row + nwg.x;
  }
}
