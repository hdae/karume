// karume axis reduce sum (non-last dim, bool>i32, generated)
struct Params {
  out_count: u32,
  axis_len: u32,
  inner: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<i32>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let inner = params.inner;
  let axis_len = params.axis_len;
  let stride = nwg.x * 256u;
  var t = gid.x;
  while (t < params.out_count) {
    // output index t = outer·inner + i; back to the base address with only the reduced axis removed
    let base = (t / inner) * axis_len * inner + (t % inner);
    var acc: array<i32, 9>;
    for (var m = 0u; m < 256u; m = m + 1u) {
      // reproduce the leaf order of the row reduce (bit reversal) in one thread
      let slot = reverseBits(m) >> 24u;
      var v = 0i;
      var c = slot;
      while (c < axis_len) {
        v = v + select(0i, 1i, x[base + c * inner] != 0u);
        c = c + 256u;
      }
      var k = 0u;
      while ((m & (1u << k)) != 0u) {
        v = acc[k] + v;
        k = k + 1u;
      }
      acc[k] = v;
    }
    out[t] = acc[8u];
    t = t + stride;
  }
}
