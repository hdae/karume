// karume static_quantize: 固定 f32 scale の SRQ（ADR 0097）
@group(0) @binding(0) var<uniform> params: array<vec4<u32>, 65>;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
fn word(index: u32) -> u32 { return params[index >> 2u][index & 3u]; }
fn quantize(bits: u32) -> u32 {
  if (word(258u) != 0u) { return bits; }
  let magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u) { return bits | 0x00400000u; }
  var lo = 0u;
  var hi = 128u;
  while (lo < hi) {
    let mid = (lo + hi) >> 1u;
    if (magnitude >= word(1u + mid)) { lo = mid + 1u; }
    else { hi = mid; }
  }
  let sign = bits & 0x80000000u;
  let level = min(lo, select(127u, 128u, sign != 0u));
  return word(129u + level) | sign;
}
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let stride = groups.x * 128u;
  var i = gid.x;
  while (i < word(0u)) {
    output[i] = quantize(input[i]);
    i += stride;
  }
}
