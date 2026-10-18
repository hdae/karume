// karume static_quantize packed: SRQ with a fixed f32 scale written as 4 int8 codes per u32 word (ADR 0105)
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
  // Negatives are the low 8 bits of the two's complement (level 0 has no sign slot and falls to +0)
  return select(level, (0u - level) & 255u, sign != 0u);
}
// Code storage for one tile (128 elements)
var<workgroup> codes: array<u32, 128>;
@compute @workgroup_size(128)
fn main(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(num_workgroups) groups: vec3<u32>,
) {
  let count = word(0u);
  let tiles = (count + 127u) / 128u;
  // Tile selection depends on workgroup_id alone = the barrier below is uniform even at the edge
  for (var tile = wg.x; tile < tiles; tile += groups.x) {
    let index = tile * 128u + lid;
    var code = 0u;
    if (index < count) { code = quantizeCode(input[index]); }
    codes[lid] = code;
    workgroupBarrier();
    // The element count is a multiple of 4 (the params check), so a word is either entirely in range
    // or entirely out of range
    if (lid < 32u) {
      let base = lid * 4u;
      if (tile * 128u + base < count) {
        output[tile * 32u + lid] = codes[base] |
          (codes[base + 1u] << 8u) | (codes[base + 2u] << 16u) | (codes[base + 3u] << 24u);
      }
    }
    // Every thread finishes the reads above before the next tile overwrites codes (WAR)
    workgroupBarrier();
  }
}
