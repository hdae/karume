// karume linear gemv K parallel (bf16, 32 lanes/output, staged activations)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
// Row starts are 16 B aligned because k % 8 == 0 (guaranteed by the eligibility check)
@group(0) @binding(2) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

var<workgroup> partial: array<f32, 128>;
var<workgroup> sx: array<vec4<f32>, 64u>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let lane = lid % 32u;
  let col = wg.x * 4u + lid / 32u;
  var acc = 0.0;
  // Edge columns load their share of the staged activation and pass every barrier; their hoisted
  // reads take the last column, and their sums are never stored.
  let units = dims.k / 8u;
  let col_read = min(col, dims.n - 1u);
  let row_base = col_read * units;
  let sx_base = lane * 2u;
  let chunks = (units + 31u) / 32u;
  for (var chunk = 0u; chunk < chunks; chunk += 1u) {
    // Chunk c stages the activation quads of words c·32 … c·32 + 31.
    for (var q = lid; q < 64u; q += 128u) {
      let v = chunk * 64u + q;
      var staged_quad = vec4<f32>(0.0);
      if (v < dims.k / 4u) {
        staged_quad = x[wg.y * (dims.k / 4u) + v];
      }
      sx[(q) ^ (((q) >> 3u) & 7u)] = staged_quad;
    }
    workgroupBarrier();
    let unit = chunk * 32u + lane;
    if (col < dims.n && unit < units) {
    let unitt = unit;
    let pwt = w[row_base + unitt];
    let bt_0 = pwt.x;
    let xt_0 = sx[(sx_base + 0u) ^ (((sx_base + 0u) >> 3u) & 7u)];
    acc = fma(xt_0.x, bitcast<f32>(bt_0 << 16u), acc);
    acc = fma(xt_0.y, bitcast<f32>(bt_0 & 0xFFFF0000u), acc);
    let bt_1 = pwt.y;
    let xt_1 = sx[(sx_base + 0u) ^ (((sx_base + 0u) >> 3u) & 7u)];
    acc = fma(xt_1.z, bitcast<f32>(bt_1 << 16u), acc);
    acc = fma(xt_1.w, bitcast<f32>(bt_1 & 0xFFFF0000u), acc);
    let bt_2 = pwt.z;
    let xt_2 = sx[(sx_base + 1u) ^ (((sx_base + 1u) >> 3u) & 7u)];
    acc = fma(xt_2.x, bitcast<f32>(bt_2 << 16u), acc);
    acc = fma(xt_2.y, bitcast<f32>(bt_2 & 0xFFFF0000u), acc);
    let bt_3 = pwt.w;
    let xt_3 = sx[(sx_base + 1u) ^ (((sx_base + 1u) >> 3u) & 7u)];
    acc = fma(xt_3.z, bitcast<f32>(bt_3 << 16u), acc);
    acc = fma(xt_3.w, bitcast<f32>(bt_3 & 0xFFFF0000u), acc);
    }
    workgroupBarrier();
  }
  partial[lid] = acc;
  workgroupBarrier();
  for (var width = 16u; width > 0u; width /= 2u) {
    if (lane < width) {
      partial[lid] = partial[lid] + partial[lid + width];
    }
    workgroupBarrier();
  }
  if (col < dims.n && lane == 0u) {
    out[wg.y * dims.n + col] = partial[lid] + bias[col];
  }
}
