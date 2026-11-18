// karume linear gemv K parallel rows (i4, 4 lanes/output, 4 rows / thread, staged activations)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
// Row starts are 16 B aligned because k % 32 == 0 (guaranteed by the eligibility check)
@group(0) @binding(2) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read> wscale: array<f32>;
var<workgroup> partial: array<f32, 512>;
var<workgroup> sx: array<vec4<f32>, 128u>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let lane = lid % 4u;
  let col = wg.x * 32u + lid / 4u;
  // The first row of the row block and the start of x per row (in quads). Rows beyond m re-read the
  // last row (only to keep reads in range; write-back is only inside the row count)
  let kq = dims.k / 4u;
  let row0 = wg.y * 4u;
  let xr0 = min(row0 + 0u, dims.m - 1u) * kq;
  let xr1 = min(row0 + 1u, dims.m - 1u) * kq;
  let xr2 = min(row0 + 2u, dims.m - 1u) * kq;
  let xr3 = min(row0 + 3u, dims.m - 1u) * kq;
  var acc0 = 0.0;
  var acc1 = 0.0;
  var acc2 = 0.0;
  var acc3 = 0.0;
  // Edge columns load their share of the staged activation and pass every barrier; their hoisted
  // reads take the last column, and their sums are never stored.
  let units = dims.k / 32u;
  let col_read = min(col, dims.n - 1u);
  let row_base = col_read * units;
  let scale_base = col_read * (dims.k >> 5u);
  let sx_base = lane * 8u;
  let chunks = (units + 3u) / 4u;
  for (var chunk = 0u; chunk < chunks; chunk += 1u) {
    // Chunk c stages, per row, the activation quads of words c·4 … c·4 + 3.
    for (var q = lid; q < 32u; q += 128u) {
      let v = chunk * 32u + q;
      var staged_quad0 = vec4<f32>(0.0);
      if (v < kq) {
        staged_quad0 = x[xr0 + v];
      }
      sx[(0u + q) ^ (((0u + q) >> 3u) & 7u)] = staged_quad0;
      var staged_quad1 = vec4<f32>(0.0);
      if (v < kq) {
        staged_quad1 = x[xr1 + v];
      }
      sx[(32u + q) ^ (((32u + q) >> 3u) & 7u)] = staged_quad1;
      var staged_quad2 = vec4<f32>(0.0);
      if (v < kq) {
        staged_quad2 = x[xr2 + v];
      }
      sx[(64u + q) ^ (((64u + q) >> 3u) & 7u)] = staged_quad2;
      var staged_quad3 = vec4<f32>(0.0);
      if (v < kq) {
        staged_quad3 = x[xr3 + v];
      }
      sx[(96u + q) ^ (((96u + q) >> 3u) & 7u)] = staged_quad3;
    }
    workgroupBarrier();
    let unit = chunk * 4u + lane;
    if (col < dims.n && unit < units) {
    let unitt = unit;
    let pwt = w[row_base + unitt];
    let wst = wscale[scale_base + ((unitt * 32u) >> 5u)];
    let bt_0 = unpack4xU8(pwt.x);
    let dt_0_0 = (f32(i32(bt_0.x & 0xFu) - 8) * wst);
    let dt_0_1 = (f32(i32(bt_0.x >> 4u) - 8) * wst);
    let dt_0_2 = (f32(i32(bt_0.y & 0xFu) - 8) * wst);
    let dt_0_3 = (f32(i32(bt_0.y >> 4u) - 8) * wst);
    let dt_0_4 = (f32(i32(bt_0.z & 0xFu) - 8) * wst);
    let dt_0_5 = (f32(i32(bt_0.z >> 4u) - 8) * wst);
    let dt_0_6 = (f32(i32(bt_0.w & 0xFu) - 8) * wst);
    let dt_0_7 = (f32(i32(bt_0.w >> 4u) - 8) * wst);
    let xat_0_0 = sx[(0u + sx_base + 0u) ^ (((0u + sx_base + 0u) >> 3u) & 7u)];
    let xbt_0_0 = sx[(0u + sx_base + 1u) ^ (((0u + sx_base + 1u) >> 3u) & 7u)];
    acc0 = fma(xat_0_0.x, dt_0_0, acc0);
    acc0 = fma(xat_0_0.y, dt_0_1, acc0);
    acc0 = fma(xat_0_0.z, dt_0_2, acc0);
    acc0 = fma(xat_0_0.w, dt_0_3, acc0);
    acc0 = fma(xbt_0_0.x, dt_0_4, acc0);
    acc0 = fma(xbt_0_0.y, dt_0_5, acc0);
    acc0 = fma(xbt_0_0.z, dt_0_6, acc0);
    acc0 = fma(xbt_0_0.w, dt_0_7, acc0);
    let xat_0_1 = sx[(32u + sx_base + 0u) ^ (((32u + sx_base + 0u) >> 3u) & 7u)];
    let xbt_0_1 = sx[(32u + sx_base + 1u) ^ (((32u + sx_base + 1u) >> 3u) & 7u)];
    acc1 = fma(xat_0_1.x, dt_0_0, acc1);
    acc1 = fma(xat_0_1.y, dt_0_1, acc1);
    acc1 = fma(xat_0_1.z, dt_0_2, acc1);
    acc1 = fma(xat_0_1.w, dt_0_3, acc1);
    acc1 = fma(xbt_0_1.x, dt_0_4, acc1);
    acc1 = fma(xbt_0_1.y, dt_0_5, acc1);
    acc1 = fma(xbt_0_1.z, dt_0_6, acc1);
    acc1 = fma(xbt_0_1.w, dt_0_7, acc1);
    let xat_0_2 = sx[(64u + sx_base + 0u) ^ (((64u + sx_base + 0u) >> 3u) & 7u)];
    let xbt_0_2 = sx[(64u + sx_base + 1u) ^ (((64u + sx_base + 1u) >> 3u) & 7u)];
    acc2 = fma(xat_0_2.x, dt_0_0, acc2);
    acc2 = fma(xat_0_2.y, dt_0_1, acc2);
    acc2 = fma(xat_0_2.z, dt_0_2, acc2);
    acc2 = fma(xat_0_2.w, dt_0_3, acc2);
    acc2 = fma(xbt_0_2.x, dt_0_4, acc2);
    acc2 = fma(xbt_0_2.y, dt_0_5, acc2);
    acc2 = fma(xbt_0_2.z, dt_0_6, acc2);
    acc2 = fma(xbt_0_2.w, dt_0_7, acc2);
    let xat_0_3 = sx[(96u + sx_base + 0u) ^ (((96u + sx_base + 0u) >> 3u) & 7u)];
    let xbt_0_3 = sx[(96u + sx_base + 1u) ^ (((96u + sx_base + 1u) >> 3u) & 7u)];
    acc3 = fma(xat_0_3.x, dt_0_0, acc3);
    acc3 = fma(xat_0_3.y, dt_0_1, acc3);
    acc3 = fma(xat_0_3.z, dt_0_2, acc3);
    acc3 = fma(xat_0_3.w, dt_0_3, acc3);
    acc3 = fma(xbt_0_3.x, dt_0_4, acc3);
    acc3 = fma(xbt_0_3.y, dt_0_5, acc3);
    acc3 = fma(xbt_0_3.z, dt_0_6, acc3);
    acc3 = fma(xbt_0_3.w, dt_0_7, acc3);
    let bt_1 = unpack4xU8(pwt.y);
    let dt_1_0 = (f32(i32(bt_1.x & 0xFu) - 8) * wst);
    let dt_1_1 = (f32(i32(bt_1.x >> 4u) - 8) * wst);
    let dt_1_2 = (f32(i32(bt_1.y & 0xFu) - 8) * wst);
    let dt_1_3 = (f32(i32(bt_1.y >> 4u) - 8) * wst);
    let dt_1_4 = (f32(i32(bt_1.z & 0xFu) - 8) * wst);
    let dt_1_5 = (f32(i32(bt_1.z >> 4u) - 8) * wst);
    let dt_1_6 = (f32(i32(bt_1.w & 0xFu) - 8) * wst);
    let dt_1_7 = (f32(i32(bt_1.w >> 4u) - 8) * wst);
    let xat_1_0 = sx[(0u + sx_base + 2u) ^ (((0u + sx_base + 2u) >> 3u) & 7u)];
    let xbt_1_0 = sx[(0u + sx_base + 3u) ^ (((0u + sx_base + 3u) >> 3u) & 7u)];
    acc0 = fma(xat_1_0.x, dt_1_0, acc0);
    acc0 = fma(xat_1_0.y, dt_1_1, acc0);
    acc0 = fma(xat_1_0.z, dt_1_2, acc0);
    acc0 = fma(xat_1_0.w, dt_1_3, acc0);
    acc0 = fma(xbt_1_0.x, dt_1_4, acc0);
    acc0 = fma(xbt_1_0.y, dt_1_5, acc0);
    acc0 = fma(xbt_1_0.z, dt_1_6, acc0);
    acc0 = fma(xbt_1_0.w, dt_1_7, acc0);
    let xat_1_1 = sx[(32u + sx_base + 2u) ^ (((32u + sx_base + 2u) >> 3u) & 7u)];
    let xbt_1_1 = sx[(32u + sx_base + 3u) ^ (((32u + sx_base + 3u) >> 3u) & 7u)];
    acc1 = fma(xat_1_1.x, dt_1_0, acc1);
    acc1 = fma(xat_1_1.y, dt_1_1, acc1);
    acc1 = fma(xat_1_1.z, dt_1_2, acc1);
    acc1 = fma(xat_1_1.w, dt_1_3, acc1);
    acc1 = fma(xbt_1_1.x, dt_1_4, acc1);
    acc1 = fma(xbt_1_1.y, dt_1_5, acc1);
    acc1 = fma(xbt_1_1.z, dt_1_6, acc1);
    acc1 = fma(xbt_1_1.w, dt_1_7, acc1);
    let xat_1_2 = sx[(64u + sx_base + 2u) ^ (((64u + sx_base + 2u) >> 3u) & 7u)];
    let xbt_1_2 = sx[(64u + sx_base + 3u) ^ (((64u + sx_base + 3u) >> 3u) & 7u)];
    acc2 = fma(xat_1_2.x, dt_1_0, acc2);
    acc2 = fma(xat_1_2.y, dt_1_1, acc2);
    acc2 = fma(xat_1_2.z, dt_1_2, acc2);
    acc2 = fma(xat_1_2.w, dt_1_3, acc2);
    acc2 = fma(xbt_1_2.x, dt_1_4, acc2);
    acc2 = fma(xbt_1_2.y, dt_1_5, acc2);
    acc2 = fma(xbt_1_2.z, dt_1_6, acc2);
    acc2 = fma(xbt_1_2.w, dt_1_7, acc2);
    let xat_1_3 = sx[(96u + sx_base + 2u) ^ (((96u + sx_base + 2u) >> 3u) & 7u)];
    let xbt_1_3 = sx[(96u + sx_base + 3u) ^ (((96u + sx_base + 3u) >> 3u) & 7u)];
    acc3 = fma(xat_1_3.x, dt_1_0, acc3);
    acc3 = fma(xat_1_3.y, dt_1_1, acc3);
    acc3 = fma(xat_1_3.z, dt_1_2, acc3);
    acc3 = fma(xat_1_3.w, dt_1_3, acc3);
    acc3 = fma(xbt_1_3.x, dt_1_4, acc3);
    acc3 = fma(xbt_1_3.y, dt_1_5, acc3);
    acc3 = fma(xbt_1_3.z, dt_1_6, acc3);
    acc3 = fma(xbt_1_3.w, dt_1_7, acc3);
    let bt_2 = unpack4xU8(pwt.z);
    let dt_2_0 = (f32(i32(bt_2.x & 0xFu) - 8) * wst);
    let dt_2_1 = (f32(i32(bt_2.x >> 4u) - 8) * wst);
    let dt_2_2 = (f32(i32(bt_2.y & 0xFu) - 8) * wst);
    let dt_2_3 = (f32(i32(bt_2.y >> 4u) - 8) * wst);
    let dt_2_4 = (f32(i32(bt_2.z & 0xFu) - 8) * wst);
    let dt_2_5 = (f32(i32(bt_2.z >> 4u) - 8) * wst);
    let dt_2_6 = (f32(i32(bt_2.w & 0xFu) - 8) * wst);
    let dt_2_7 = (f32(i32(bt_2.w >> 4u) - 8) * wst);
    let xat_2_0 = sx[(0u + sx_base + 4u) ^ (((0u + sx_base + 4u) >> 3u) & 7u)];
    let xbt_2_0 = sx[(0u + sx_base + 5u) ^ (((0u + sx_base + 5u) >> 3u) & 7u)];
    acc0 = fma(xat_2_0.x, dt_2_0, acc0);
    acc0 = fma(xat_2_0.y, dt_2_1, acc0);
    acc0 = fma(xat_2_0.z, dt_2_2, acc0);
    acc0 = fma(xat_2_0.w, dt_2_3, acc0);
    acc0 = fma(xbt_2_0.x, dt_2_4, acc0);
    acc0 = fma(xbt_2_0.y, dt_2_5, acc0);
    acc0 = fma(xbt_2_0.z, dt_2_6, acc0);
    acc0 = fma(xbt_2_0.w, dt_2_7, acc0);
    let xat_2_1 = sx[(32u + sx_base + 4u) ^ (((32u + sx_base + 4u) >> 3u) & 7u)];
    let xbt_2_1 = sx[(32u + sx_base + 5u) ^ (((32u + sx_base + 5u) >> 3u) & 7u)];
    acc1 = fma(xat_2_1.x, dt_2_0, acc1);
    acc1 = fma(xat_2_1.y, dt_2_1, acc1);
    acc1 = fma(xat_2_1.z, dt_2_2, acc1);
    acc1 = fma(xat_2_1.w, dt_2_3, acc1);
    acc1 = fma(xbt_2_1.x, dt_2_4, acc1);
    acc1 = fma(xbt_2_1.y, dt_2_5, acc1);
    acc1 = fma(xbt_2_1.z, dt_2_6, acc1);
    acc1 = fma(xbt_2_1.w, dt_2_7, acc1);
    let xat_2_2 = sx[(64u + sx_base + 4u) ^ (((64u + sx_base + 4u) >> 3u) & 7u)];
    let xbt_2_2 = sx[(64u + sx_base + 5u) ^ (((64u + sx_base + 5u) >> 3u) & 7u)];
    acc2 = fma(xat_2_2.x, dt_2_0, acc2);
    acc2 = fma(xat_2_2.y, dt_2_1, acc2);
    acc2 = fma(xat_2_2.z, dt_2_2, acc2);
    acc2 = fma(xat_2_2.w, dt_2_3, acc2);
    acc2 = fma(xbt_2_2.x, dt_2_4, acc2);
    acc2 = fma(xbt_2_2.y, dt_2_5, acc2);
    acc2 = fma(xbt_2_2.z, dt_2_6, acc2);
    acc2 = fma(xbt_2_2.w, dt_2_7, acc2);
    let xat_2_3 = sx[(96u + sx_base + 4u) ^ (((96u + sx_base + 4u) >> 3u) & 7u)];
    let xbt_2_3 = sx[(96u + sx_base + 5u) ^ (((96u + sx_base + 5u) >> 3u) & 7u)];
    acc3 = fma(xat_2_3.x, dt_2_0, acc3);
    acc3 = fma(xat_2_3.y, dt_2_1, acc3);
    acc3 = fma(xat_2_3.z, dt_2_2, acc3);
    acc3 = fma(xat_2_3.w, dt_2_3, acc3);
    acc3 = fma(xbt_2_3.x, dt_2_4, acc3);
    acc3 = fma(xbt_2_3.y, dt_2_5, acc3);
    acc3 = fma(xbt_2_3.z, dt_2_6, acc3);
    acc3 = fma(xbt_2_3.w, dt_2_7, acc3);
    let bt_3 = unpack4xU8(pwt.w);
    let dt_3_0 = (f32(i32(bt_3.x & 0xFu) - 8) * wst);
    let dt_3_1 = (f32(i32(bt_3.x >> 4u) - 8) * wst);
    let dt_3_2 = (f32(i32(bt_3.y & 0xFu) - 8) * wst);
    let dt_3_3 = (f32(i32(bt_3.y >> 4u) - 8) * wst);
    let dt_3_4 = (f32(i32(bt_3.z & 0xFu) - 8) * wst);
    let dt_3_5 = (f32(i32(bt_3.z >> 4u) - 8) * wst);
    let dt_3_6 = (f32(i32(bt_3.w & 0xFu) - 8) * wst);
    let dt_3_7 = (f32(i32(bt_3.w >> 4u) - 8) * wst);
    let xat_3_0 = sx[(0u + sx_base + 6u) ^ (((0u + sx_base + 6u) >> 3u) & 7u)];
    let xbt_3_0 = sx[(0u + sx_base + 7u) ^ (((0u + sx_base + 7u) >> 3u) & 7u)];
    acc0 = fma(xat_3_0.x, dt_3_0, acc0);
    acc0 = fma(xat_3_0.y, dt_3_1, acc0);
    acc0 = fma(xat_3_0.z, dt_3_2, acc0);
    acc0 = fma(xat_3_0.w, dt_3_3, acc0);
    acc0 = fma(xbt_3_0.x, dt_3_4, acc0);
    acc0 = fma(xbt_3_0.y, dt_3_5, acc0);
    acc0 = fma(xbt_3_0.z, dt_3_6, acc0);
    acc0 = fma(xbt_3_0.w, dt_3_7, acc0);
    let xat_3_1 = sx[(32u + sx_base + 6u) ^ (((32u + sx_base + 6u) >> 3u) & 7u)];
    let xbt_3_1 = sx[(32u + sx_base + 7u) ^ (((32u + sx_base + 7u) >> 3u) & 7u)];
    acc1 = fma(xat_3_1.x, dt_3_0, acc1);
    acc1 = fma(xat_3_1.y, dt_3_1, acc1);
    acc1 = fma(xat_3_1.z, dt_3_2, acc1);
    acc1 = fma(xat_3_1.w, dt_3_3, acc1);
    acc1 = fma(xbt_3_1.x, dt_3_4, acc1);
    acc1 = fma(xbt_3_1.y, dt_3_5, acc1);
    acc1 = fma(xbt_3_1.z, dt_3_6, acc1);
    acc1 = fma(xbt_3_1.w, dt_3_7, acc1);
    let xat_3_2 = sx[(64u + sx_base + 6u) ^ (((64u + sx_base + 6u) >> 3u) & 7u)];
    let xbt_3_2 = sx[(64u + sx_base + 7u) ^ (((64u + sx_base + 7u) >> 3u) & 7u)];
    acc2 = fma(xat_3_2.x, dt_3_0, acc2);
    acc2 = fma(xat_3_2.y, dt_3_1, acc2);
    acc2 = fma(xat_3_2.z, dt_3_2, acc2);
    acc2 = fma(xat_3_2.w, dt_3_3, acc2);
    acc2 = fma(xbt_3_2.x, dt_3_4, acc2);
    acc2 = fma(xbt_3_2.y, dt_3_5, acc2);
    acc2 = fma(xbt_3_2.z, dt_3_6, acc2);
    acc2 = fma(xbt_3_2.w, dt_3_7, acc2);
    let xat_3_3 = sx[(96u + sx_base + 6u) ^ (((96u + sx_base + 6u) >> 3u) & 7u)];
    let xbt_3_3 = sx[(96u + sx_base + 7u) ^ (((96u + sx_base + 7u) >> 3u) & 7u)];
    acc3 = fma(xat_3_3.x, dt_3_0, acc3);
    acc3 = fma(xat_3_3.y, dt_3_1, acc3);
    acc3 = fma(xat_3_3.z, dt_3_2, acc3);
    acc3 = fma(xat_3_3.w, dt_3_3, acc3);
    acc3 = fma(xbt_3_3.x, dt_3_4, acc3);
    acc3 = fma(xbt_3_3.y, dt_3_5, acc3);
    acc3 = fma(xbt_3_3.z, dt_3_6, acc3);
    acc3 = fma(xbt_3_3.w, dt_3_7, acc3);
    }
    workgroupBarrier();
  }
  partial[lid] = acc0;
  partial[128u + lid] = acc1;
  partial[256u + lid] = acc2;
  partial[384u + lid] = acc3;
  workgroupBarrier();
  for (var width = 2u; width > 0u; width /= 2u) {
    if (lane < width) {
      partial[lid] = partial[lid] + partial[lid + width];
      partial[128u + lid] = partial[128u + lid] + partial[128u + lid + width];
      partial[256u + lid] = partial[256u + lid] + partial[256u + lid + width];
      partial[384u + lid] = partial[384u + lid] + partial[384u + lid + width];
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
    if (row0 + 2u < dims.m) {
      out[(row0 + 2u) * dims.n + col] = partial[256u + lid] + bias[col];
    }
    if (row0 + 3u < dims.m) {
      out[(row0 + 3u) * dims.n + col] = partial[384u + lid] + bias[col];
    }
  }
}
