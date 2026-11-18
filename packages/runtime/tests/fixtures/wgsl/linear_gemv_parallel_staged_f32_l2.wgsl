// karume linear gemv K parallel (f32, 2 lanes/output, staged activations)
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

var<workgroup> partial: array<f32, 128>;
var<workgroup> sx: array<vec4<f32>, 2u>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let lane = lid % 2u;
  let col = wg.x * 64u + lid / 2u;
  var acc = 0.0;
  // Edge columns load their share of the staged activation and pass every barrier; their hoisted
  // reads take the last column, and their sums are never stored.
  let units = dims.k / 4u;
  let col_read = min(col, dims.n - 1u);
  let row_base = col_read * units;
  let sx_base = lane * 1u;
  let chunks = (units + 1u) / 2u;
  for (var chunk = 0u; chunk < chunks; chunk += 1u) {
    // Chunk c stages the activation quads of words c·2 … c·2 + 1.
    for (var q = lid; q < 2u; q += 128u) {
      let v = chunk * 2u + q;
      var staged_quad = vec4<f32>(0.0);
      if (v < dims.k / 4u) {
        staged_quad = x[wg.y * (dims.k / 4u) + v];
      }
      sx[(q) ^ (((q) >> 3u) & 7u)] = staged_quad;
    }
    workgroupBarrier();
    let unit = chunk * 2u + lane;
    if (col < dims.n && unit < units) {
    let unitt = unit;
    let pwt = w[row_base + unitt];
    let xft = sx[(sx_base + 0u) ^ (((sx_base + 0u) >> 3u) & 7u)];
    acc = fma(xft.x, pwt.x, acc);
    acc = fma(xft.y, pwt.y, acc);
    acc = fma(xft.z, pwt.z, acc);
    acc = fma(xft.w, pwt.w, acc);
    }
    workgroupBarrier();
  }
  partial[lid] = acc;
  workgroupBarrier();
  for (var width = 1u; width > 0u; width /= 2u) {
    if (lane < width) {
      partial[lid] = partial[lid] + partial[lid + width];
    }
    workgroupBarrier();
  }
  if (col < dims.n && lane == 0u) {
    out[wg.y * dims.n + col] = partial[lid] + bias[col];
  }
}
