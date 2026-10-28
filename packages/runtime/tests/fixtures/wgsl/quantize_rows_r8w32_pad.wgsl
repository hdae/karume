// karume quantize_rows (per-row per-token symmetric i8: s = max(amax/127, tiny), q = clamp(round(x/s), ±127)) — 8 rows × 32 lanes / workgroup, rows of any length (the last word zero-padded)
struct Params {
  rows: u32,
  dim: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> xq: array<u32>;
@group(0) @binding(3) var<storage, read_write> xs: array<f32>;

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
  // Rows of any length: ceil(dim / 4) words per row, the bytes from element dim on are 0
  let quads = (dim + 3u) / 4u;
  // Lane groups: width 32 × 8 groups. lane = position in a group, slot = group
  let lane = lid % 32u;
  let slot = lid / 32u;
  var rowBase = wid.x * 8u;
  while (rowBase < params.rows) {
    let row = rowBase + slot;
    // A group without a row at the tail does not read or write and only joins the barriers (the
    // row loop is workgroup-uniform)
    let has_row = row < params.rows;
    let base = row * dim;

    // ① Absolute maximum of the row (NaN propagates by a bit-pattern check — plain max swallows it)
    var acc = 0.0;
    if (has_row) {
      var i = lane;
      while (i < dim) {
        acc = nan_max(acc, abs(x[base + i]));
        i = i + 32u;
      }
    }
    scratch[lid] = acc;
    workgroupBarrier();
    var stride = 16u;
    while (stride > 0u) {
      if (lane < stride) {
        scratch[lid] = nan_max(scratch[lid], scratch[lid + stride]);
      }
      workgroupBarrier();
      stride = stride / 2u;
    }
    let amax = scratch[slot * 32u];
    // An all-zero row is exact: s = tiny → q = 0 → 0·tiny = 0. max swallows NaN, so the outer
    // select passes it.
    // MUST: **multiplication by 1/127**, not division by 127 (only multiplication is correctly
    // rounded — the MUST above)
    let s = select(max(amax * 0.007874015748031496, 1.1754943508222875e-38), amax, is_nan_bits(amax));
    if (has_row && lane == 0u) {
      xs[row] = s;
    }

    // ② Quantize each 4 consecutive elements (quad) and pack them into one word
    if (has_row) {
      let qbase = row * quads;
      var q = lane;
      while (q < quads) {
        let e = base + q * 4u;
        // In the last word the elements from dim on are padding: clamped reads, bytes forced to 0
        let last = base + dim - 1u;
        let inside = vec4<u32>(0u, 1u, 2u, 3u) + q * 4u < vec4<u32>(dim);
        let v = vec4<f32>(x[e], x[min(e + 1u, last)], x[min(e + 2u, last)], x[min(e + 3u, last)]) / s;
        let r = select(vec4<f32>(0.0), clamp(round(v), vec4<f32>(-127.0), vec4<f32>(127.0)), inside);
        xq[qbase + q] = pack4xI8(vec4<i32>(r));
        q = q + 32u;
      }
    }

    // Make sure scratch has been read before the next row overwrites scratch[lid]
    workgroupBarrier();
    rowBase = rowBase + nwg.x * 8u;
  }
}
