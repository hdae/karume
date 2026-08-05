// karume masked_fill (out = select(x, value, mask), rank 4, f32 / mask bool)
@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> mask: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let n = params[0u];
  let fill = bitcast<f32>(params[9u]);
  let stride = nwg.x * 256u;
  var i = gid.x;
  while (i < n) {
    var rem = i;
    let c3 = rem % params[4u]; rem = rem / params[4u];
    let c2 = rem % params[3u]; rem = rem / params[3u];
    let c1 = rem % params[2u]; rem = rem / params[2u];
    let c0 = rem;
    let mask_index = c0 * params[5u] + c1 * params[6u] + c2 * params[7u] + c3 * params[8u];
    // bool の格納は u32 の 0 / 1（ADR 0009）— 真の位置だけを埋め値に差し替える
    out[i] = select(x[i], fill, mask[mask_index] != 0u);
    i = i + stride;
  }
}
