// karume embedding (out[..., h] = weight[index[...], h], f32 / 添字 i32, 重み i8 格納)
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

@group(0) @binding(4) var<storage, read> wscale: array<f32>;

// i8 格納の展開: 要素 i = f32(unpack4xI8(weight[i / 4])[i % 4]) · scale
// （平坦添字で語と位置を割る。scale は出力チャネルごと — ADR 0019）
fn dequant(i: u32, scale: f32) -> f32 {
  return f32(unpack4xI8(weight[i >> 2u])[i & 3u]) * scale;
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
      // 出力チャネルの scale はループ不変 — 重みの要素ごとに引き直さない（ADR 0019）
      let wscale_v = wscale[u32(pick)];
      out[i] = dequant(u32(pick) * dims.hidden + col, wscale_v);
    }
    i = i + stride;
  }
}
