// karume axis reduce sum (non-last dim, bool>i32, split 16, generated)
struct Params {
  out_count: u32,
  axis_len: u32,
  inner: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<i32>;

var<workgroup> roots: array<i32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let o = lid.x % 16u;
  let s = lid.x / 16u;
  let inner = params.inner;
  let axis_len = params.axis_len;
  var tb = wid.x * 16u;
  while (tb < params.out_count) {
    let t = tb + o;
    var root = 0i;
    if (t < params.out_count) {
      // output index t = outer·inner + i; back to the base address with only the reduced axis removed
      let base = (t / inner) * axis_len * inner + (t % inner);
      var acc: array<i32, 5>;
      for (var j = 0u; j < 16u; j = j + 1u) {
        // the row reduce's leaf order (bit reversal), restricted to this subtree's aligned range
        let slot = reverseBits(s * 16u + j) >> 24u;
        var v = 0i;
        var c = slot;
        while (c < axis_len) {
          v = v + select(0i, 1i, x[base + c * inner] != 0u);
          c = c + 256u;
        }
        var k = 0u;
        while ((j & (1u << k)) != 0u) {
          v = acc[k] + v;
          k = k + 1u;
        }
        acc[k] = v;
      }
      root = acc[4u];
    }
    roots[lid.x] = root;
    workgroupBarrier();
    if (s % 2u == 0u) {
      roots[lid.x] = roots[lid.x] + roots[lid.x + 16u];
    }
    workgroupBarrier();
    if (s % 4u == 0u) {
      roots[lid.x] = roots[lid.x] + roots[lid.x + 32u];
    }
    workgroupBarrier();
    if (s % 8u == 0u) {
      roots[lid.x] = roots[lid.x] + roots[lid.x + 64u];
    }
    workgroupBarrier();
    if (s % 16u == 0u) {
      roots[lid.x] = roots[lid.x] + roots[lid.x + 128u];
    }
    workgroupBarrier();
    if (s == 0u && t < params.out_count) {
      out[t] = roots[o];
    }
    tb = tb + nwg.x * 16u;
  }
}
