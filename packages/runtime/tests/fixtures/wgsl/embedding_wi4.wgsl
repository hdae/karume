// karume embedding (out[..., h] = weight[index[...], h], f32 / 添字 i32, 重み i4 格納)
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

// i4 格納の展開: 要素 i = f32(i32(nibble) − 8) · scale
// （1 語 = 8 要素。平坦添字で語 i/8・バイト (i/2)%4・nibble i%2 を割る — ADR 0069）
fn dequant(i: u32, scale: f32) -> f32 {
  let byte = unpack4xU8(weight[i >> 3u])[(i >> 1u) & 3u];
  let nibble = select(byte & 0xFu, byte >> 4u, (i & 1u) == 1u);
  return f32(i32(nibble) - 8) * scale;
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
      // group scale は量子化軸（列）依存 — 1 スレッド 1 要素なので巻き上げず要素ごとに引く
      let wscale_v = wscale[u32(pick) * (dims.hidden >> 5u) + (col >> 5u)];
      out[i] = dequant(u32(pick) * dims.hidden + col, wscale_v);
    }
    i = i + stride;
  }
}
