// karume quantize_rows with a gelu prologue (per-row per-token symmetric i8 of y = gelu(x); y is never stored) — the row in registers, 8 vec4 per lane
struct Params {
  rows: u32,
  dim: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> xq: array<u32>;
@group(0) @binding(3) var<storage, read_write> xs: array<f32>;

fn erf_approx(x: f32) -> f32 {
  let s = sign(x);
  let a = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  var p = 1.061405429;
  p = -1.453152027 + p * t;
  p = 1.421413741 + p * t;
  p = -0.284496736 + p * t;
  p = 0.254829592 + p * t;
  return s * (1.0 - p * t * exp(-a * a));
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
  // The caller guarantees dim % 4 == 0 and dim <= 8192 (quantizeRowsPrologueEligible)
  let quads = params.dim / 4u;
  var row = wid.x;
  while (row < params.rows) {
    let base = row * quads;

    // ① y = gelu(x) once per element, kept in registers, and the absolute maximum of y (NaN
    // propagates by a bit-pattern check — plain max swallows it)
    var acc = 0.0;
    let q0 = lid + 0u;
    var y0 = vec4<f32>(0.0);
    if (q0 < quads) {
      let x0 = x[base + q0];
      y0 = vec4<f32>(0.5 * x0.x * (1.0 + erf_approx(x0.x * 0.7071067811865476)), 0.5 * x0.y * (1.0 + erf_approx(x0.y * 0.7071067811865476)), 0.5 * x0.z * (1.0 + erf_approx(x0.z * 0.7071067811865476)), 0.5 * x0.w * (1.0 + erf_approx(x0.w * 0.7071067811865476)));
      acc = nan_max(acc, nan_max(nan_max(abs(y0.x), abs(y0.y)), nan_max(abs(y0.z), abs(y0.w))));
    }
    let q1 = lid + 256u;
    var y1 = vec4<f32>(0.0);
    if (q1 < quads) {
      let x1 = x[base + q1];
      y1 = vec4<f32>(0.5 * x1.x * (1.0 + erf_approx(x1.x * 0.7071067811865476)), 0.5 * x1.y * (1.0 + erf_approx(x1.y * 0.7071067811865476)), 0.5 * x1.z * (1.0 + erf_approx(x1.z * 0.7071067811865476)), 0.5 * x1.w * (1.0 + erf_approx(x1.w * 0.7071067811865476)));
      acc = nan_max(acc, nan_max(nan_max(abs(y1.x), abs(y1.y)), nan_max(abs(y1.z), abs(y1.w))));
    }
    let q2 = lid + 512u;
    var y2 = vec4<f32>(0.0);
    if (q2 < quads) {
      let x2 = x[base + q2];
      y2 = vec4<f32>(0.5 * x2.x * (1.0 + erf_approx(x2.x * 0.7071067811865476)), 0.5 * x2.y * (1.0 + erf_approx(x2.y * 0.7071067811865476)), 0.5 * x2.z * (1.0 + erf_approx(x2.z * 0.7071067811865476)), 0.5 * x2.w * (1.0 + erf_approx(x2.w * 0.7071067811865476)));
      acc = nan_max(acc, nan_max(nan_max(abs(y2.x), abs(y2.y)), nan_max(abs(y2.z), abs(y2.w))));
    }
    let q3 = lid + 768u;
    var y3 = vec4<f32>(0.0);
    if (q3 < quads) {
      let x3 = x[base + q3];
      y3 = vec4<f32>(0.5 * x3.x * (1.0 + erf_approx(x3.x * 0.7071067811865476)), 0.5 * x3.y * (1.0 + erf_approx(x3.y * 0.7071067811865476)), 0.5 * x3.z * (1.0 + erf_approx(x3.z * 0.7071067811865476)), 0.5 * x3.w * (1.0 + erf_approx(x3.w * 0.7071067811865476)));
      acc = nan_max(acc, nan_max(nan_max(abs(y3.x), abs(y3.y)), nan_max(abs(y3.z), abs(y3.w))));
    }
    let q4 = lid + 1024u;
    var y4 = vec4<f32>(0.0);
    if (q4 < quads) {
      let x4 = x[base + q4];
      y4 = vec4<f32>(0.5 * x4.x * (1.0 + erf_approx(x4.x * 0.7071067811865476)), 0.5 * x4.y * (1.0 + erf_approx(x4.y * 0.7071067811865476)), 0.5 * x4.z * (1.0 + erf_approx(x4.z * 0.7071067811865476)), 0.5 * x4.w * (1.0 + erf_approx(x4.w * 0.7071067811865476)));
      acc = nan_max(acc, nan_max(nan_max(abs(y4.x), abs(y4.y)), nan_max(abs(y4.z), abs(y4.w))));
    }
    let q5 = lid + 1280u;
    var y5 = vec4<f32>(0.0);
    if (q5 < quads) {
      let x5 = x[base + q5];
      y5 = vec4<f32>(0.5 * x5.x * (1.0 + erf_approx(x5.x * 0.7071067811865476)), 0.5 * x5.y * (1.0 + erf_approx(x5.y * 0.7071067811865476)), 0.5 * x5.z * (1.0 + erf_approx(x5.z * 0.7071067811865476)), 0.5 * x5.w * (1.0 + erf_approx(x5.w * 0.7071067811865476)));
      acc = nan_max(acc, nan_max(nan_max(abs(y5.x), abs(y5.y)), nan_max(abs(y5.z), abs(y5.w))));
    }
    let q6 = lid + 1536u;
    var y6 = vec4<f32>(0.0);
    if (q6 < quads) {
      let x6 = x[base + q6];
      y6 = vec4<f32>(0.5 * x6.x * (1.0 + erf_approx(x6.x * 0.7071067811865476)), 0.5 * x6.y * (1.0 + erf_approx(x6.y * 0.7071067811865476)), 0.5 * x6.z * (1.0 + erf_approx(x6.z * 0.7071067811865476)), 0.5 * x6.w * (1.0 + erf_approx(x6.w * 0.7071067811865476)));
      acc = nan_max(acc, nan_max(nan_max(abs(y6.x), abs(y6.y)), nan_max(abs(y6.z), abs(y6.w))));
    }
    let q7 = lid + 1792u;
    var y7 = vec4<f32>(0.0);
    if (q7 < quads) {
      let x7 = x[base + q7];
      y7 = vec4<f32>(0.5 * x7.x * (1.0 + erf_approx(x7.x * 0.7071067811865476)), 0.5 * x7.y * (1.0 + erf_approx(x7.y * 0.7071067811865476)), 0.5 * x7.z * (1.0 + erf_approx(x7.z * 0.7071067811865476)), 0.5 * x7.w * (1.0 + erf_approx(x7.w * 0.7071067811865476)));
      acc = nan_max(acc, nan_max(nan_max(abs(y7.x), abs(y7.y)), nan_max(abs(y7.z), abs(y7.w))));
    }
    scratch[lid] = acc;
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
    // MUST: **multiplication by 1/127** (the MUST above), and NaN passes through the outer select
    let s = select(max(amax * 0.007874015748031496, 1.1754943508222875e-38), amax, is_nan_bits(amax));
    if (lid == 0u) {
      xs[row] = s;
    }

    // ② Quantize the kept quads (the plain form's pass ② on y)
    let qbase = row * quads;
    if (q0 < quads) {
      let r0 = clamp(round(y0 / s), vec4<f32>(-127.0), vec4<f32>(127.0));
      xq[qbase + q0] = pack4xI8(vec4<i32>(r0));
    }
    if (q1 < quads) {
      let r1 = clamp(round(y1 / s), vec4<f32>(-127.0), vec4<f32>(127.0));
      xq[qbase + q1] = pack4xI8(vec4<i32>(r1));
    }
    if (q2 < quads) {
      let r2 = clamp(round(y2 / s), vec4<f32>(-127.0), vec4<f32>(127.0));
      xq[qbase + q2] = pack4xI8(vec4<i32>(r2));
    }
    if (q3 < quads) {
      let r3 = clamp(round(y3 / s), vec4<f32>(-127.0), vec4<f32>(127.0));
      xq[qbase + q3] = pack4xI8(vec4<i32>(r3));
    }
    if (q4 < quads) {
      let r4 = clamp(round(y4 / s), vec4<f32>(-127.0), vec4<f32>(127.0));
      xq[qbase + q4] = pack4xI8(vec4<i32>(r4));
    }
    if (q5 < quads) {
      let r5 = clamp(round(y5 / s), vec4<f32>(-127.0), vec4<f32>(127.0));
      xq[qbase + q5] = pack4xI8(vec4<i32>(r5));
    }
    if (q6 < quads) {
      let r6 = clamp(round(y6 / s), vec4<f32>(-127.0), vec4<f32>(127.0));
      xq[qbase + q6] = pack4xI8(vec4<i32>(r6));
    }
    if (q7 < quads) {
      let r7 = clamp(round(y7 / s), vec4<f32>(-127.0), vec4<f32>(127.0));
      xq[qbase + q7] = pack4xI8(vec4<i32>(r7));
    }

    // Make sure scratch[0] has been read before the next row overwrites scratch[lid]
    workgroupBarrier();
    row = row + nwg.x;
  }
}
