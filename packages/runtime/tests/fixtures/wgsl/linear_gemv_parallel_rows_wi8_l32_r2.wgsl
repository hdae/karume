// karume linear gemv K parallel rows (i8, 32 lanes/output, 2 rows / thread, staged activations)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
// Row starts are 16 B aligned because k % 16 == 0 (guaranteed by the eligibility check)
@group(0) @binding(2) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read> wscale: array<f32>;
var<workgroup> partial: array<f32, 256>;
var<workgroup> sx: array<vec4<f32>, 256u>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let lane = lid % 32u;
  let col = wg.x * 4u + lid / 32u;
  // The first row of the row block and the start of x per row (in quads). Rows beyond m re-read the
  // last row (only to keep reads in range; write-back is only inside the row count)
  let kq = dims.k / 4u;
  let row0 = wg.y * 2u;
  let xr0 = min(row0 + 0u, dims.m - 1u) * kq;
  let xr1 = min(row0 + 1u, dims.m - 1u) * kq;
  var acc0 = 0.0;
  var acc1 = 0.0;
  // Edge columns load their share of the staged activation and pass every barrier; their hoisted
  // reads take the last column, and their sums are never stored.
  let units = dims.k / 16u;
  let col_read = min(col, dims.n - 1u);
  let row_base = col_read * units;
  // the output channel's scale is loop-invariant — not re-read per weight element (ADR 0019)
  let wscale_v = wscale[col_read];
  let sx_base = lane * 4u;
  let chunks = (units + 31u) / 32u;
  for (var chunk = 0u; chunk < chunks; chunk += 1u) {
    // Chunk c stages, per row, the activation quads of words c·32 … c·32 + 31.
    for (var q = lid; q < 128u; q += 128u) {
      let v = chunk * 128u + q;
      var staged_quad0 = vec4<f32>(0.0);
      if (v < kq) {
        staged_quad0 = x[xr0 + v];
      }
      sx[(0u + q) ^ (((0u + q) >> 3u) & 7u)] = staged_quad0;
      var staged_quad1 = vec4<f32>(0.0);
      if (v < kq) {
        staged_quad1 = x[xr1 + v];
      }
      sx[(128u + q) ^ (((128u + q) >> 3u) & 7u)] = staged_quad1;
    }
    workgroupBarrier();
    let unit = chunk * 32u + lane;
    if (col < dims.n && unit < units) {
    let unitt = unit;
    let pwt = w[row_base + unitt];
    let bt_0 = unpack4xI8(pwt.x);
    let dt_0_0 = (f32(bt_0.x) * wscale_v);
    let dt_0_1 = (f32(bt_0.y) * wscale_v);
    let dt_0_2 = (f32(bt_0.z) * wscale_v);
    let dt_0_3 = (f32(bt_0.w) * wscale_v);
    let xat_0_0 = sx[(0u + sx_base + 0u) ^ (((0u + sx_base + 0u) >> 3u) & 7u)];
    acc0 = fma(xat_0_0.x, dt_0_0, acc0);
    acc0 = fma(xat_0_0.y, dt_0_1, acc0);
    acc0 = fma(xat_0_0.z, dt_0_2, acc0);
    acc0 = fma(xat_0_0.w, dt_0_3, acc0);
    let xat_0_1 = sx[(128u + sx_base + 0u) ^ (((128u + sx_base + 0u) >> 3u) & 7u)];
    acc1 = fma(xat_0_1.x, dt_0_0, acc1);
    acc1 = fma(xat_0_1.y, dt_0_1, acc1);
    acc1 = fma(xat_0_1.z, dt_0_2, acc1);
    acc1 = fma(xat_0_1.w, dt_0_3, acc1);
    let bt_1 = unpack4xI8(pwt.y);
    let dt_1_0 = (f32(bt_1.x) * wscale_v);
    let dt_1_1 = (f32(bt_1.y) * wscale_v);
    let dt_1_2 = (f32(bt_1.z) * wscale_v);
    let dt_1_3 = (f32(bt_1.w) * wscale_v);
    let xat_1_0 = sx[(0u + sx_base + 1u) ^ (((0u + sx_base + 1u) >> 3u) & 7u)];
    acc0 = fma(xat_1_0.x, dt_1_0, acc0);
    acc0 = fma(xat_1_0.y, dt_1_1, acc0);
    acc0 = fma(xat_1_0.z, dt_1_2, acc0);
    acc0 = fma(xat_1_0.w, dt_1_3, acc0);
    let xat_1_1 = sx[(128u + sx_base + 1u) ^ (((128u + sx_base + 1u) >> 3u) & 7u)];
    acc1 = fma(xat_1_1.x, dt_1_0, acc1);
    acc1 = fma(xat_1_1.y, dt_1_1, acc1);
    acc1 = fma(xat_1_1.z, dt_1_2, acc1);
    acc1 = fma(xat_1_1.w, dt_1_3, acc1);
    let bt_2 = unpack4xI8(pwt.z);
    let dt_2_0 = (f32(bt_2.x) * wscale_v);
    let dt_2_1 = (f32(bt_2.y) * wscale_v);
    let dt_2_2 = (f32(bt_2.z) * wscale_v);
    let dt_2_3 = (f32(bt_2.w) * wscale_v);
    let xat_2_0 = sx[(0u + sx_base + 2u) ^ (((0u + sx_base + 2u) >> 3u) & 7u)];
    acc0 = fma(xat_2_0.x, dt_2_0, acc0);
    acc0 = fma(xat_2_0.y, dt_2_1, acc0);
    acc0 = fma(xat_2_0.z, dt_2_2, acc0);
    acc0 = fma(xat_2_0.w, dt_2_3, acc0);
    let xat_2_1 = sx[(128u + sx_base + 2u) ^ (((128u + sx_base + 2u) >> 3u) & 7u)];
    acc1 = fma(xat_2_1.x, dt_2_0, acc1);
    acc1 = fma(xat_2_1.y, dt_2_1, acc1);
    acc1 = fma(xat_2_1.z, dt_2_2, acc1);
    acc1 = fma(xat_2_1.w, dt_2_3, acc1);
    let bt_3 = unpack4xI8(pwt.w);
    let dt_3_0 = (f32(bt_3.x) * wscale_v);
    let dt_3_1 = (f32(bt_3.y) * wscale_v);
    let dt_3_2 = (f32(bt_3.z) * wscale_v);
    let dt_3_3 = (f32(bt_3.w) * wscale_v);
    let xat_3_0 = sx[(0u + sx_base + 3u) ^ (((0u + sx_base + 3u) >> 3u) & 7u)];
    acc0 = fma(xat_3_0.x, dt_3_0, acc0);
    acc0 = fma(xat_3_0.y, dt_3_1, acc0);
    acc0 = fma(xat_3_0.z, dt_3_2, acc0);
    acc0 = fma(xat_3_0.w, dt_3_3, acc0);
    let xat_3_1 = sx[(128u + sx_base + 3u) ^ (((128u + sx_base + 3u) >> 3u) & 7u)];
    acc1 = fma(xat_3_1.x, dt_3_0, acc1);
    acc1 = fma(xat_3_1.y, dt_3_1, acc1);
    acc1 = fma(xat_3_1.z, dt_3_2, acc1);
    acc1 = fma(xat_3_1.w, dt_3_3, acc1);
    }
    workgroupBarrier();
  }
  partial[lid] = acc0;
  partial[128u + lid] = acc1;
  workgroupBarrier();
  for (var width = 16u; width > 0u; width /= 2u) {
    if (lane < width) {
      partial[lid] = partial[lid] + partial[lid + width];
      partial[128u + lid] = partial[128u + lid] + partial[128u + lid + width];
    }
    workgroupBarrier();
  }
  if (col < dims.n && lane == 0u) {
    if (row0 + 0u < dims.m) {
      out[(row0 + 0u) * dims.n + col] = partial[lid] + bias[col];
    }
    if (row0 + 1u < dims.m) {
      out[(row0 + 1u) * dims.n + col] = partial[128u + lid] + bias[col];
    }
  }
}
