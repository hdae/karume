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
// 1 タイル（128 要素）ぶんのコード置き場
var<workgroup> codes: array<u32, 128>;
@compute @workgroup_size(128)
fn main(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(num_workgroups) groups: vec3<u32>,
) {
  let count = word(0u);
  let tiles = (count + 127u) / 128u;
  // タイルの選択は workgroup_id だけで決まる = 下の barrier は端でも workgroup 一様
  for (var tile = wg.x; tile < tiles; tile += groups.x) {
    let index = tile * 128u + lid;
    var code = 0u;
    if (index < count) { code = quantizeCode(input[index]); }
    codes[lid] = code;
    workgroupBarrier();
    // 要素数は 4 の倍数（params の門）なので、1 語は全要素が範囲内か全要素が範囲外のどちらか
    if (lid < 32u) {
      let base = lid * 4u;
      if (tile * 128u + base < count) {
        output[tile * 32u + lid] = codes[base] |
          (codes[base + 1u] << 8u) | (codes[base + 2u] << 16u) | (codes[base + 3u] << 24u);
      }
    }
    // 次のタイルが codes を上書きする前に、上の読みを全スレッドが終える（WAR）
    workgroupBarrier();
  }
}
