// karume static_quantize packed: 固定 f32 scale の SRQ を int8 コード 4 個 / u32 語で書く（ADR 0105）
@group(0) @binding(0) var<uniform> params: array<vec4<u32>, 65>;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
fn word(index: u32) -> u32 { return params[index >> 2u][index & 3u]; }
fn quantizeCode(bits: u32) -> u32 {
  let magnitude = bits & 0x7fffffffu;
  var lo = 0u;
  var hi = 128u;
  while (lo < hi) {
    let mid = (lo + hi) >> 1u;
    if (magnitude >= word(1u + mid)) { lo = mid + 1u; }
    else { hi = mid; }
  }
  let sign = bits & 0x80000000u;
  let level = min(lo, select(127u, 128u, sign != 0u));
  // 負は 2 の補数の低位 8 bit（level 0 の符号は席が無く +0 へ落ちる）
  return select(level, (0u - level) & 255u, sign != 0u);
}
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let stride = groups.x * 128u;
  let words = word(0u) >> 2u;
  var i = gid.x;
  while (i < words) {
    let base = i << 2u;
    output[i] = quantizeCode(input[base]) | (quantizeCode(input[base + 1u]) << 8u) |
      (quantizeCode(input[base + 2u]) << 16u) | (quantizeCode(input[base + 3u]) << 24u);
    i += stride;
  }
}
