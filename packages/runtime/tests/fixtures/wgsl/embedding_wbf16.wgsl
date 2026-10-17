// karume embedding (out[..., h] = weight[index[...], h], f32 / 添字 i32, 重み bf16 格納)
struct Dims {
  n: u32,
  hidden: u32,
  vocab: u32,
  oob: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read> index: array<i32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

// bf16 格納の展開: 要素 i = weight[i / 2] の下位 / 上位 16 bit を f32 の上位へ（平坦添字の偶奇で選ぶ）
fn dequant(i: u32) -> f32 {
  let word = weight[i >> 1u];
  return bitcast<f32>(select(word << 16u, word & 0xFFFF0000u, (i & 1u) == 1u));
}

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let stride = nwg.x * 256u;
  var i = gid.x;
  while (i < dims.n) {
    let row = i / dims.hidden;
    let col = i % dims.hidden;
    let pick = index[row];
    // 契約外の添字は別の行を返さず NaN で汚染する（カーネル doc の裁定）
    if (pick < 0 || u32(pick) >= dims.vocab) {
      out[i] = bitcast<f32>(dims.oob);
    } else {
      out[i] = dequant(u32(pick) * dims.hidden + col);
    }
    i = i + stride;
  }
}
