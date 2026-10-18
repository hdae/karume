// karume linear gemv (M=1: out[n] = x[k] · wᵀ[n,k] + bias[n], f32, 32 cols / wg, 4 words prefetched)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
// Row starts are 16 B aligned because k % 4 == 0 (guaranteed by the eligibility check)
@group(0) @binding(2) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;


@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  // No shared memory and no barrier, so edge workgroups may return early
  if (col >= dims.n) {
    return;
  }
  let units = dims.k / 4u;
  let row_base = col * units;
  var acc = 0.0;
  var unit = 0u;
  // Issue all prefetched weight words **first**, then do the multiply-adds (memory parallelism).
  // Words are still processed in ascending order, so the addition order per output element is the
  // same whatever the prefetch count
  for (; unit + 4u <= units; unit = unit + 4u) {
    let unit0 = unit + 0u;
    let pw0 = w[row_base + unit0];
    let xq0 = unit0 * 1u;
    let unit1 = unit + 1u;
    let pw1 = w[row_base + unit1];
    let xq1 = unit1 * 1u;
    let unit2 = unit + 2u;
    let pw2 = w[row_base + unit2];
    let xq2 = unit2 * 1u;
    let unit3 = unit + 3u;
    let pw3 = w[row_base + unit3];
    let xq3 = unit3 * 1u;
    let xf0 = x[xq0];
    acc = acc + xf0.x * pw0.x;
    acc = acc + xf0.y * pw0.y;
    acc = acc + xf0.z * pw0.z;
    acc = acc + xf0.w * pw0.w;
    let xf1 = x[xq1];
    acc = acc + xf1.x * pw1.x;
    acc = acc + xf1.y * pw1.y;
    acc = acc + xf1.z * pw1.z;
    acc = acc + xf1.w * pw1.w;
    let xf2 = x[xq2];
    acc = acc + xf2.x * pw2.x;
    acc = acc + xf2.y * pw2.y;
    acc = acc + xf2.z * pw2.z;
    acc = acc + xf2.w * pw2.w;
    let xf3 = x[xq3];
    acc = acc + xf3.x * pw3.x;
    acc = acc + xf3.y * pw3.y;
    acc = acc + xf3.z * pw3.z;
    acc = acc + xf3.w * pw3.w;
  }
  // Remainder words (units % 4 of them) — walk the same order one word at a time
  for (; unit < units; unit = unit + 1u) {
    let unitt = unit;
    let pwt = w[row_base + unitt];
    let xqt = unitt * 1u;
    let xft = x[xqt];
    acc = acc + xft.x * pwt.x;
    acc = acc + xft.y * pwt.y;
    acc = acc + xft.z * pwt.z;
    acc = acc + xft.w * pwt.w;
  }
  out[col] = acc + bias[col];
}
