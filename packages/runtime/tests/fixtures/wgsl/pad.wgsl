// karume pad (last dim, constant 0, f32, full-write)
struct Params {
  n: u32,
  out_len: u32,
  in_len: u32,
  left: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let stride = nwg.x * 256u;
  var i = gid.x;
  while (i < params.n) {
    let col = i % params.out_len;
    let row = i / params.out_len;
    // 範囲外にも必ず 0 を書く（未書き込みを 1 要素も残さない）
    var value = 0.0;
    if (col >= params.left && col < params.left + params.in_len) {
      value = x[row * params.in_len + (col - params.left)];
    }
    out[i] = value;
    i = i + stride;
  }
}
