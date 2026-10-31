// karume gelu_tanh -> mul -> static_quantize (gelu-other, f32, SRQ v2, u32 out)
struct Params {
  n: u32,
  mask_act: u32,
  mask_mul: u32,
  reserved: u32,
  srq: array<vec4<u32>, 65>,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read> other: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;

fn is_nan_bits(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

fn tanh_stable(x: f32) -> f32 {
  let lo = select(x, -9.5, x < -9.5);
  let t = select(lo, 9.5, x > 9.5);
  return select(tanh(t), x, is_nan_bits(x));
}

fn word(index: u32) -> u32 { return params.srq[index >> 2u][index & 3u]; }
fn quantize(bits: u32) -> u32 {
  if (word(258u) != 0u) { return bits; }
  let magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u) { return bits | 0x00400000u; }
  let sign = bits & 0x80000000u;
  let cap = select(127u, 128u, sign != 0u);
  let c_guess = min(u32(bitcast<f32>(magnitude) / bitcast<f32>(word(130u)) + 0.5), 128u);
  let off_guess = ((c_guess > 0u) & (magnitude < word(c_guess))) |
    ((c_guess < 128u) & (magnitude >= word(c_guess + 1u)));
  // The output word of the guess loads with the check (it is discarded when the check fails)
  let guessed = word(129u + min(c_guess, cap));
  if (!off_guess) { return guessed | sign; }
  var lo = 0u;
  var hi = 128u;
  while (lo < hi) {
    let mid = (lo + hi) >> 1u;
    if (magnitude >= word(1u + mid)) { lo = mid + 1u; }
    else { hi = mid; }
  }
  let level = min(lo, cap);
  return word(129u + level) | sign;
}
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * 128u;
  var i = gid.x;
  while (i < params.n) {
    let v0 = gate[i];
    let v1 = other[i];
    // Rounding barriers: the two values the plain chain stores (masks are 0 at run time)
    let act = bitcast<f32>(bitcast<u32>(0.5 * v0 * (1.0 + tanh_stable(0.7978845608028654 * (v0 + 0.044715 * v0 * v0 * v0)))) ^ params.mask_act);
    out[i] = quantize(bitcast<u32>(act * v1) ^ params.mask_mul);
    i = i + stride;
  }
}
