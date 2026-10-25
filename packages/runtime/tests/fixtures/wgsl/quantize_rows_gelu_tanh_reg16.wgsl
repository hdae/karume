// karume quantize_rows with a gelu_tanh prologue (per-row per-token symmetric i8 of y = gelu_tanh(x); y is never stored) — the row in registers, 16 vec4 per lane
struct Params {
  rows: u32,
  dim: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> xq: array<u32>;
@group(0) @binding(3) var<storage, read_write> xs: array<f32>;

fn is_nan_bits(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

fn tanh_stable(x: f32) -> f32 {
  let lo = select(x, -9.5, x < -9.5);
  let t = select(lo, 9.5, x > 9.5);
  return select(tanh(t), x, is_nan_bits(x));
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
  // The caller guarantees dim % 4 == 0 and dim <= 16384 (quantizeRowsPrologueEligible)
  let quads = params.dim / 4u;
  var row = wid.x;
  while (row < params.rows) {
    let base = row * quads;

    // ① y = gelu_tanh(x) once per element, kept in registers, and the absolute maximum of y (NaN
    // propagates by a bit-pattern check — plain max swallows it)
    var acc = 0.0;
    let q0 = lid + 0u;
    var y0 = vec4<f32>(0.0);
    if (q0 < quads) {
      let x0 = x[base + q0];
      y0 = vec4<f32>(0.5 * x0.x * (1.0 + tanh_stable(0.7978845608028654 * (x0.x + 0.044715 * x0.x * x0.x * x0.x))), 0.5 * x0.y * (1.0 + tanh_stable(0.7978845608028654 * (x0.y + 0.044715 * x0.y * x0.y * x0.y))), 0.5 * x0.z * (1.0 + tanh_stable(0.7978845608028654 * (x0.z + 0.044715 * x0.z * x0.z * x0.z))), 0.5 * x0.w * (1.0 + tanh_stable(0.7978845608028654 * (x0.w + 0.044715 * x0.w * x0.w * x0.w))));
      acc = nan_max(acc, nan_max(nan_max(abs(y0.x), abs(y0.y)), nan_max(abs(y0.z), abs(y0.w))));
    }
    let q1 = lid + 256u;
    var y1 = vec4<f32>(0.0);
    if (q1 < quads) {
      let x1 = x[base + q1];
      y1 = vec4<f32>(0.5 * x1.x * (1.0 + tanh_stable(0.7978845608028654 * (x1.x + 0.044715 * x1.x * x1.x * x1.x))), 0.5 * x1.y * (1.0 + tanh_stable(0.7978845608028654 * (x1.y + 0.044715 * x1.y * x1.y * x1.y))), 0.5 * x1.z * (1.0 + tanh_stable(0.7978845608028654 * (x1.z + 0.044715 * x1.z * x1.z * x1.z))), 0.5 * x1.w * (1.0 + tanh_stable(0.7978845608028654 * (x1.w + 0.044715 * x1.w * x1.w * x1.w))));
      acc = nan_max(acc, nan_max(nan_max(abs(y1.x), abs(y1.y)), nan_max(abs(y1.z), abs(y1.w))));
    }
    let q2 = lid + 512u;
    var y2 = vec4<f32>(0.0);
    if (q2 < quads) {
      let x2 = x[base + q2];
      y2 = vec4<f32>(0.5 * x2.x * (1.0 + tanh_stable(0.7978845608028654 * (x2.x + 0.044715 * x2.x * x2.x * x2.x))), 0.5 * x2.y * (1.0 + tanh_stable(0.7978845608028654 * (x2.y + 0.044715 * x2.y * x2.y * x2.y))), 0.5 * x2.z * (1.0 + tanh_stable(0.7978845608028654 * (x2.z + 0.044715 * x2.z * x2.z * x2.z))), 0.5 * x2.w * (1.0 + tanh_stable(0.7978845608028654 * (x2.w + 0.044715 * x2.w * x2.w * x2.w))));
      acc = nan_max(acc, nan_max(nan_max(abs(y2.x), abs(y2.y)), nan_max(abs(y2.z), abs(y2.w))));
    }
    let q3 = lid + 768u;
    var y3 = vec4<f32>(0.0);
    if (q3 < quads) {
      let x3 = x[base + q3];
      y3 = vec4<f32>(0.5 * x3.x * (1.0 + tanh_stable(0.7978845608028654 * (x3.x + 0.044715 * x3.x * x3.x * x3.x))), 0.5 * x3.y * (1.0 + tanh_stable(0.7978845608028654 * (x3.y + 0.044715 * x3.y * x3.y * x3.y))), 0.5 * x3.z * (1.0 + tanh_stable(0.7978845608028654 * (x3.z + 0.044715 * x3.z * x3.z * x3.z))), 0.5 * x3.w * (1.0 + tanh_stable(0.7978845608028654 * (x3.w + 0.044715 * x3.w * x3.w * x3.w))));
      acc = nan_max(acc, nan_max(nan_max(abs(y3.x), abs(y3.y)), nan_max(abs(y3.z), abs(y3.w))));
    }
    let q4 = lid + 1024u;
    var y4 = vec4<f32>(0.0);
    if (q4 < quads) {
      let x4 = x[base + q4];
      y4 = vec4<f32>(0.5 * x4.x * (1.0 + tanh_stable(0.7978845608028654 * (x4.x + 0.044715 * x4.x * x4.x * x4.x))), 0.5 * x4.y * (1.0 + tanh_stable(0.7978845608028654 * (x4.y + 0.044715 * x4.y * x4.y * x4.y))), 0.5 * x4.z * (1.0 + tanh_stable(0.7978845608028654 * (x4.z + 0.044715 * x4.z * x4.z * x4.z))), 0.5 * x4.w * (1.0 + tanh_stable(0.7978845608028654 * (x4.w + 0.044715 * x4.w * x4.w * x4.w))));
      acc = nan_max(acc, nan_max(nan_max(abs(y4.x), abs(y4.y)), nan_max(abs(y4.z), abs(y4.w))));
    }
    let q5 = lid + 1280u;
    var y5 = vec4<f32>(0.0);
    if (q5 < quads) {
      let x5 = x[base + q5];
      y5 = vec4<f32>(0.5 * x5.x * (1.0 + tanh_stable(0.7978845608028654 * (x5.x + 0.044715 * x5.x * x5.x * x5.x))), 0.5 * x5.y * (1.0 + tanh_stable(0.7978845608028654 * (x5.y + 0.044715 * x5.y * x5.y * x5.y))), 0.5 * x5.z * (1.0 + tanh_stable(0.7978845608028654 * (x5.z + 0.044715 * x5.z * x5.z * x5.z))), 0.5 * x5.w * (1.0 + tanh_stable(0.7978845608028654 * (x5.w + 0.044715 * x5.w * x5.w * x5.w))));
      acc = nan_max(acc, nan_max(nan_max(abs(y5.x), abs(y5.y)), nan_max(abs(y5.z), abs(y5.w))));
    }
    let q6 = lid + 1536u;
    var y6 = vec4<f32>(0.0);
    if (q6 < quads) {
      let x6 = x[base + q6];
      y6 = vec4<f32>(0.5 * x6.x * (1.0 + tanh_stable(0.7978845608028654 * (x6.x + 0.044715 * x6.x * x6.x * x6.x))), 0.5 * x6.y * (1.0 + tanh_stable(0.7978845608028654 * (x6.y + 0.044715 * x6.y * x6.y * x6.y))), 0.5 * x6.z * (1.0 + tanh_stable(0.7978845608028654 * (x6.z + 0.044715 * x6.z * x6.z * x6.z))), 0.5 * x6.w * (1.0 + tanh_stable(0.7978845608028654 * (x6.w + 0.044715 * x6.w * x6.w * x6.w))));
      acc = nan_max(acc, nan_max(nan_max(abs(y6.x), abs(y6.y)), nan_max(abs(y6.z), abs(y6.w))));
    }
    let q7 = lid + 1792u;
    var y7 = vec4<f32>(0.0);
    if (q7 < quads) {
      let x7 = x[base + q7];
      y7 = vec4<f32>(0.5 * x7.x * (1.0 + tanh_stable(0.7978845608028654 * (x7.x + 0.044715 * x7.x * x7.x * x7.x))), 0.5 * x7.y * (1.0 + tanh_stable(0.7978845608028654 * (x7.y + 0.044715 * x7.y * x7.y * x7.y))), 0.5 * x7.z * (1.0 + tanh_stable(0.7978845608028654 * (x7.z + 0.044715 * x7.z * x7.z * x7.z))), 0.5 * x7.w * (1.0 + tanh_stable(0.7978845608028654 * (x7.w + 0.044715 * x7.w * x7.w * x7.w))));
      acc = nan_max(acc, nan_max(nan_max(abs(y7.x), abs(y7.y)), nan_max(abs(y7.z), abs(y7.w))));
    }
    let q8 = lid + 2048u;
    var y8 = vec4<f32>(0.0);
    if (q8 < quads) {
      let x8 = x[base + q8];
      y8 = vec4<f32>(0.5 * x8.x * (1.0 + tanh_stable(0.7978845608028654 * (x8.x + 0.044715 * x8.x * x8.x * x8.x))), 0.5 * x8.y * (1.0 + tanh_stable(0.7978845608028654 * (x8.y + 0.044715 * x8.y * x8.y * x8.y))), 0.5 * x8.z * (1.0 + tanh_stable(0.7978845608028654 * (x8.z + 0.044715 * x8.z * x8.z * x8.z))), 0.5 * x8.w * (1.0 + tanh_stable(0.7978845608028654 * (x8.w + 0.044715 * x8.w * x8.w * x8.w))));
      acc = nan_max(acc, nan_max(nan_max(abs(y8.x), abs(y8.y)), nan_max(abs(y8.z), abs(y8.w))));
    }
    let q9 = lid + 2304u;
    var y9 = vec4<f32>(0.0);
    if (q9 < quads) {
      let x9 = x[base + q9];
      y9 = vec4<f32>(0.5 * x9.x * (1.0 + tanh_stable(0.7978845608028654 * (x9.x + 0.044715 * x9.x * x9.x * x9.x))), 0.5 * x9.y * (1.0 + tanh_stable(0.7978845608028654 * (x9.y + 0.044715 * x9.y * x9.y * x9.y))), 0.5 * x9.z * (1.0 + tanh_stable(0.7978845608028654 * (x9.z + 0.044715 * x9.z * x9.z * x9.z))), 0.5 * x9.w * (1.0 + tanh_stable(0.7978845608028654 * (x9.w + 0.044715 * x9.w * x9.w * x9.w))));
      acc = nan_max(acc, nan_max(nan_max(abs(y9.x), abs(y9.y)), nan_max(abs(y9.z), abs(y9.w))));
    }
    let q10 = lid + 2560u;
    var y10 = vec4<f32>(0.0);
    if (q10 < quads) {
      let x10 = x[base + q10];
      y10 = vec4<f32>(0.5 * x10.x * (1.0 + tanh_stable(0.7978845608028654 * (x10.x + 0.044715 * x10.x * x10.x * x10.x))), 0.5 * x10.y * (1.0 + tanh_stable(0.7978845608028654 * (x10.y + 0.044715 * x10.y * x10.y * x10.y))), 0.5 * x10.z * (1.0 + tanh_stable(0.7978845608028654 * (x10.z + 0.044715 * x10.z * x10.z * x10.z))), 0.5 * x10.w * (1.0 + tanh_stable(0.7978845608028654 * (x10.w + 0.044715 * x10.w * x10.w * x10.w))));
      acc = nan_max(acc, nan_max(nan_max(abs(y10.x), abs(y10.y)), nan_max(abs(y10.z), abs(y10.w))));
    }
    let q11 = lid + 2816u;
    var y11 = vec4<f32>(0.0);
    if (q11 < quads) {
      let x11 = x[base + q11];
      y11 = vec4<f32>(0.5 * x11.x * (1.0 + tanh_stable(0.7978845608028654 * (x11.x + 0.044715 * x11.x * x11.x * x11.x))), 0.5 * x11.y * (1.0 + tanh_stable(0.7978845608028654 * (x11.y + 0.044715 * x11.y * x11.y * x11.y))), 0.5 * x11.z * (1.0 + tanh_stable(0.7978845608028654 * (x11.z + 0.044715 * x11.z * x11.z * x11.z))), 0.5 * x11.w * (1.0 + tanh_stable(0.7978845608028654 * (x11.w + 0.044715 * x11.w * x11.w * x11.w))));
      acc = nan_max(acc, nan_max(nan_max(abs(y11.x), abs(y11.y)), nan_max(abs(y11.z), abs(y11.w))));
    }
    let q12 = lid + 3072u;
    var y12 = vec4<f32>(0.0);
    if (q12 < quads) {
      let x12 = x[base + q12];
      y12 = vec4<f32>(0.5 * x12.x * (1.0 + tanh_stable(0.7978845608028654 * (x12.x + 0.044715 * x12.x * x12.x * x12.x))), 0.5 * x12.y * (1.0 + tanh_stable(0.7978845608028654 * (x12.y + 0.044715 * x12.y * x12.y * x12.y))), 0.5 * x12.z * (1.0 + tanh_stable(0.7978845608028654 * (x12.z + 0.044715 * x12.z * x12.z * x12.z))), 0.5 * x12.w * (1.0 + tanh_stable(0.7978845608028654 * (x12.w + 0.044715 * x12.w * x12.w * x12.w))));
      acc = nan_max(acc, nan_max(nan_max(abs(y12.x), abs(y12.y)), nan_max(abs(y12.z), abs(y12.w))));
    }
    let q13 = lid + 3328u;
    var y13 = vec4<f32>(0.0);
    if (q13 < quads) {
      let x13 = x[base + q13];
      y13 = vec4<f32>(0.5 * x13.x * (1.0 + tanh_stable(0.7978845608028654 * (x13.x + 0.044715 * x13.x * x13.x * x13.x))), 0.5 * x13.y * (1.0 + tanh_stable(0.7978845608028654 * (x13.y + 0.044715 * x13.y * x13.y * x13.y))), 0.5 * x13.z * (1.0 + tanh_stable(0.7978845608028654 * (x13.z + 0.044715 * x13.z * x13.z * x13.z))), 0.5 * x13.w * (1.0 + tanh_stable(0.7978845608028654 * (x13.w + 0.044715 * x13.w * x13.w * x13.w))));
      acc = nan_max(acc, nan_max(nan_max(abs(y13.x), abs(y13.y)), nan_max(abs(y13.z), abs(y13.w))));
    }
    let q14 = lid + 3584u;
    var y14 = vec4<f32>(0.0);
    if (q14 < quads) {
      let x14 = x[base + q14];
      y14 = vec4<f32>(0.5 * x14.x * (1.0 + tanh_stable(0.7978845608028654 * (x14.x + 0.044715 * x14.x * x14.x * x14.x))), 0.5 * x14.y * (1.0 + tanh_stable(0.7978845608028654 * (x14.y + 0.044715 * x14.y * x14.y * x14.y))), 0.5 * x14.z * (1.0 + tanh_stable(0.7978845608028654 * (x14.z + 0.044715 * x14.z * x14.z * x14.z))), 0.5 * x14.w * (1.0 + tanh_stable(0.7978845608028654 * (x14.w + 0.044715 * x14.w * x14.w * x14.w))));
      acc = nan_max(acc, nan_max(nan_max(abs(y14.x), abs(y14.y)), nan_max(abs(y14.z), abs(y14.w))));
    }
    let q15 = lid + 3840u;
    var y15 = vec4<f32>(0.0);
    if (q15 < quads) {
      let x15 = x[base + q15];
      y15 = vec4<f32>(0.5 * x15.x * (1.0 + tanh_stable(0.7978845608028654 * (x15.x + 0.044715 * x15.x * x15.x * x15.x))), 0.5 * x15.y * (1.0 + tanh_stable(0.7978845608028654 * (x15.y + 0.044715 * x15.y * x15.y * x15.y))), 0.5 * x15.z * (1.0 + tanh_stable(0.7978845608028654 * (x15.z + 0.044715 * x15.z * x15.z * x15.z))), 0.5 * x15.w * (1.0 + tanh_stable(0.7978845608028654 * (x15.w + 0.044715 * x15.w * x15.w * x15.w))));
      acc = nan_max(acc, nan_max(nan_max(abs(y15.x), abs(y15.y)), nan_max(abs(y15.z), abs(y15.w))));
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
    if (q8 < quads) {
      let r8 = clamp(round(y8 / s), vec4<f32>(-127.0), vec4<f32>(127.0));
      xq[qbase + q8] = pack4xI8(vec4<i32>(r8));
    }
    if (q9 < quads) {
      let r9 = clamp(round(y9 / s), vec4<f32>(-127.0), vec4<f32>(127.0));
      xq[qbase + q9] = pack4xI8(vec4<i32>(r9));
    }
    if (q10 < quads) {
      let r10 = clamp(round(y10 / s), vec4<f32>(-127.0), vec4<f32>(127.0));
      xq[qbase + q10] = pack4xI8(vec4<i32>(r10));
    }
    if (q11 < quads) {
      let r11 = clamp(round(y11 / s), vec4<f32>(-127.0), vec4<f32>(127.0));
      xq[qbase + q11] = pack4xI8(vec4<i32>(r11));
    }
    if (q12 < quads) {
      let r12 = clamp(round(y12 / s), vec4<f32>(-127.0), vec4<f32>(127.0));
      xq[qbase + q12] = pack4xI8(vec4<i32>(r12));
    }
    if (q13 < quads) {
      let r13 = clamp(round(y13 / s), vec4<f32>(-127.0), vec4<f32>(127.0));
      xq[qbase + q13] = pack4xI8(vec4<i32>(r13));
    }
    if (q14 < quads) {
      let r14 = clamp(round(y14 / s), vec4<f32>(-127.0), vec4<f32>(127.0));
      xq[qbase + q14] = pack4xI8(vec4<i32>(r14));
    }
    if (q15 < quads) {
      let r15 = clamp(round(y15 / s), vec4<f32>(-127.0), vec4<f32>(127.0));
      xq[qbase + q15] = pack4xI8(vec4<i32>(r15));
    }

    // Make sure scratch[0] has been read before the next row overwrites scratch[lid]
    workgroupBarrier();
    row = row + nwg.x;
  }
}
