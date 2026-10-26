// karume quantize_rows channel-major (x[C][P] → xq[P][C/4], xs[P]; 16 pixels × 16 lanes, 16-word blocks)
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

var<workgroup> partial: array<f32, 256>;
var<workgroup> staged: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let px = lid % 16u;
  let lane = lid / 16u;
  let pixelCount = params.rows;
  let dim = params.dim;
  // The caller guarantees dim % 4 == 0 (the i8 payload packs 4 per word)
  let quads = dim / 4u;
  var block = wid.x * 16u;
  while (block < pixelCount) {
    let p = block + px;
    let inside = p < pixelCount;

    // ① Absolute maximum of the pixel over its channels (NaN propagates by a bit-pattern check)
    var acc = 0.0;
    if (inside) {
      var c = lane;
      while (c < dim) {
        acc = nan_max(acc, abs(x[c * pixelCount + p]));
        c = c + 16u;
      }
    }
    partial[lid] = acc;
    workgroupBarrier();
    var amax = partial[px];
    for (var l = 1u; l < 16u; l = l + 1u) {
      amax = nan_max(amax, partial[l * 16u + px]);
    }
    // MUST: the same scale text as the row form (multiplication by 1/127, the tiny floor, NaN kept)
    let s = select(max(amax * 0.007874015748031496, 1.1754943508222875e-38), amax, is_nan_bits(amax));
    if (inside && lane == 0u) {
      xs[p] = s;
    }

    // ② Quantise 16 words per pixel at a time into the stage, then write each pixel's words
    // contiguously
    var w0 = 0u;
    while (w0 < quads) {
      var j = lane;
      while (j < 16u) {
        let q = w0 + j;
        if (inside && q < quads) {
          let c = q * 4u;
          let v = vec4<f32>(
            x[c * pixelCount + p],
            x[(c + 1u) * pixelCount + p],
            x[(c + 2u) * pixelCount + p],
            x[(c + 3u) * pixelCount + p],
          ) / s;
          let r = clamp(round(v), vec4<f32>(-127.0), vec4<f32>(127.0));
          staged[px * 16u + j] = pack4xI8(vec4<i32>(r));
        }
        j = j + 16u;
      }
      workgroupBarrier();
      var t = lid;
      while (t < 256u) {
        let wp = block + t / 16u;
        let wq = w0 + t % 16u;
        if (wp < pixelCount && wq < quads) {
          xq[wp * quads + wq] = staged[t];
        }
        t = t + 256u;
      }
      // The stage and the partials are rewritten by the next block
      workgroupBarrier();
      w0 = w0 + 16u;
    }
    block = block + nwg.x * 16u;
  }
}
