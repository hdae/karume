// karume gather (out[..., j] = src[..., index[..., j]], 最終次元固定, f32 / 添字 i32)
struct Dims {
  n: u32,
  cols: u32,
  src_cols: u32,
  oob: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> index: array<i32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let stride = nwg.x * 256u;
  var i = gid.x;
  while (i < dims.n) {
    let row = i / dims.cols;
    let pick = index[i];
    // 契約外の添字は別の要素を返さず NaN で汚染する（カーネル doc の裁定）
    if (pick < 0 || u32(pick) >= dims.src_cols) {
      out[i] = bitcast<f32>(dims.oob);
    } else {
      out[i] = src[row * dims.src_cols + u32(pick)];
    }
    i = i + stride;
  }
}
