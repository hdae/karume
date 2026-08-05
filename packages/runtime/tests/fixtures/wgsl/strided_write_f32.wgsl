// karume strided write (rank 4, f32, generated)
@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let n = params[0u];
  let stride = nwg.x * 256u;
  var i = gid.x;
  while (i < n) {
    var rem = i;
    let c3 = rem % params[4u]; rem = rem / params[4u];
    let c2 = rem % params[3u]; rem = rem / params[3u];
    let c1 = rem % params[2u]; rem = rem / params[2u];
    let c0 = rem;
    let out_index = params[9u] + c0 * params[5u] + c1 * params[6u] + c2 * params[7u] + c3 * params[8u];
    out[out_index] = src[i];
    i = i + stride;
  }
}
