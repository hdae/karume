// karume linear gemv K parallel (i8, 4 lanes/output, packed int8 activations, staged activations)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
  rounding_mask: u32,
  srq: array<vec4<u32>, 65>,
  x_scale: f32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<vec4<u32>>;
// Row starts are 16 B aligned because k % 16 == 0 (guaranteed by the eligibility check)
@group(0) @binding(2) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<u32>;
@group(0) @binding(5) var<storage, read> wscale: array<f32>;
fn srqWord(index: u32) -> u32 { return dims.srq[index >> 2u][index & 3u]; }
fn quantize(bits: u32) -> u32 {
  if (srqWord(258u) != 0u) { return bits; }
  let magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u) { return bits | 0x00400000u; }
  var lo = 0u;
  lo += select(0u, 64u, magnitude >= srqWord(lo + 64u));
  lo += select(0u, 32u, magnitude >= srqWord(lo + 32u));
  lo += select(0u, 16u, magnitude >= srqWord(lo + 16u));
  lo += select(0u, 8u, magnitude >= srqWord(lo + 8u));
  lo += select(0u, 4u, magnitude >= srqWord(lo + 4u));
  lo += select(0u, 2u, magnitude >= srqWord(lo + 2u));
  lo += select(0u, 1u, magnitude >= srqWord(lo + 1u));
  lo += select(0u, 1u, magnitude >= srqWord(128u));
  let sign = bits & 0x80000000u;
  let level = min(lo, select(127u, 128u, sign != 0u));
  return srqWord(129u + level) | sign;
}

var<workgroup> partial: array<f32, 128>;
var<workgroup> sx: array<vec4<u32>, 4u>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let lane = lid % 4u;
  let col = wg.x * 32u + lid / 4u;
  var acc = 0.0;
  // Edge columns load their share of the staged activation and pass every barrier; their hoisted
  // reads take the last column, and their sums are never stored.
  let units = dims.k / 16u;
  let col_read = min(col, dims.n - 1u);
  let row_base = col_read * units;
  // the output channel's scale is loop-invariant — not re-read per weight element (ADR 0019)
  let wscale_v = wscale[col_read];
  let sx_base = lane * 1u;
  let chunks = (units + 3u) / 4u;
  for (var chunk = 0u; chunk < chunks; chunk += 1u) {
    // Chunk c stages the activation words (packed) of weight words c·4 … c·4 + 3.
    for (var q = lid; q < 4u; q += 128u) {
      let v = chunk * 4u + q;
      var staged_quad = vec4<u32>(0u);
      if (v < dims.k / 16u) {
        staged_quad = x[wg.y * (dims.k / 16u) + v];
      }
      sx[(q) ^ (((q) >> 3u) & 7u)] = staged_quad;
    }
    workgroupBarrier();
    let unit = chunk * 4u + lane;
    if (col < dims.n && unit < units) {
    let unitt = unit;
    let pwt = w[row_base + unitt];
    let bt_0 = unpack4xI8(pwt.x);
    let xpt_0 = sx[(sx_base + 0u) ^ (((sx_base + 0u) >> 3u) & 7u)];
    let xat_0 = vec4<f32>(unpack4xI8(xpt_0.x)) * dims.x_scale;
    acc = fma(xat_0.x, (f32(bt_0.x) * wscale_v), acc);
    acc = fma(xat_0.y, (f32(bt_0.y) * wscale_v), acc);
    acc = fma(xat_0.z, (f32(bt_0.z) * wscale_v), acc);
    acc = fma(xat_0.w, (f32(bt_0.w) * wscale_v), acc);
    let bt_1 = unpack4xI8(pwt.y);
    let xat_1 = vec4<f32>(unpack4xI8(xpt_0.y)) * dims.x_scale;
    acc = fma(xat_1.x, (f32(bt_1.x) * wscale_v), acc);
    acc = fma(xat_1.y, (f32(bt_1.y) * wscale_v), acc);
    acc = fma(xat_1.z, (f32(bt_1.z) * wscale_v), acc);
    acc = fma(xat_1.w, (f32(bt_1.w) * wscale_v), acc);
    let bt_2 = unpack4xI8(pwt.z);
    let xat_2 = vec4<f32>(unpack4xI8(xpt_0.z)) * dims.x_scale;
    acc = fma(xat_2.x, (f32(bt_2.x) * wscale_v), acc);
    acc = fma(xat_2.y, (f32(bt_2.y) * wscale_v), acc);
    acc = fma(xat_2.z, (f32(bt_2.z) * wscale_v), acc);
    acc = fma(xat_2.w, (f32(bt_2.w) * wscale_v), acc);
    let bt_3 = unpack4xI8(pwt.w);
    let xat_3 = vec4<f32>(unpack4xI8(xpt_0.w)) * dims.x_scale;
    acc = fma(xat_3.x, (f32(bt_3.x) * wscale_v), acc);
    acc = fma(xat_3.y, (f32(bt_3.y) * wscale_v), acc);
    acc = fma(xat_3.z, (f32(bt_3.z) * wscale_v), acc);
    acc = fma(xat_3.w, (f32(bt_3.w) * wscale_v), acc);
    }
    workgroupBarrier();
  }
  partial[lid] = acc;
  workgroupBarrier();
  for (var width = 2u; width > 0u; width /= 2u) {
    if (lane < width) {
      partial[lid] = partial[lid] + partial[lid + width];
    }
    workgroupBarrier();
  }
  if (col < dims.n && lane == 0u) {
    out[wg.y * dims.n + col] = quantize(bitcast<u32>(partial[lid] + bias[col]) ^ dims.rounding_mask);
  }
}
