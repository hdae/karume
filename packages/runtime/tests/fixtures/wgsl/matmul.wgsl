// karume matmul (a[m,k] · b[k,n], f32, register tile 128x128 / 8x8 per thread / wg 16x16)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

// Shared A tile (128 rows × K 16, scalar storage) and
// shared B tile (K 16 × 32 column quads, column direction grouped into vec4)
var<workgroup> sa: array<f32, 2048>;
var<workgroup> sb: array<vec4<f32>, 512>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 16u + lid.x;
  // A tile assignment (128 rows × 4 quads over 256 threads in 2 rounds)
  let ar = tid / 4u;
  let aq = tid % 4u;
  let arow0 = wid.y * 128u + ar;
  let arow_base0 = arow0 * dims.k;
  let sa_base0 = ar * 16u + aq * 4u;
  let arow1 = arow0 + 64u;
  let arow_base1 = arow_base0 + 64u * dims.k;
  let sa_base1 = sa_base0 + 1024u;
  // B tile assignment (K 16 rows × 32 column quads over 256 threads in 2 rounds)
  let bk0 = tid / 32u;
  let bcq = tid % 32u;
  let bcol = wid.x * 128u + bcq * 4u;
  let bk1 = bk0 + 8u;
  // The loop condition is uniform (dims is a uniform buffer) — needed for the inner workgroupBarrier
  // to meet WGSL's uniformity requirement
  let tiles = (dims.k + 15u) / 16u;
  var acc0_0 = vec4<f32>(0.0);
  var acc0_1 = vec4<f32>(0.0);
  var acc1_0 = vec4<f32>(0.0);
  var acc1_1 = vec4<f32>(0.0);
  var acc2_0 = vec4<f32>(0.0);
  var acc2_1 = vec4<f32>(0.0);
  var acc3_0 = vec4<f32>(0.0);
  var acc3_1 = vec4<f32>(0.0);
  var acc4_0 = vec4<f32>(0.0);
  var acc4_1 = vec4<f32>(0.0);
  var acc5_0 = vec4<f32>(0.0);
  var acc5_1 = vec4<f32>(0.0);
  var acc6_0 = vec4<f32>(0.0);
  var acc6_1 = vec4<f32>(0.0);
  var acc7_0 = vec4<f32>(0.0);
  var acc7_1 = vec4<f32>(0.0);
  for (var t = 0u; t < tiles; t = t + 1u) {
    // Fill out-of-range entries with 0. They do not contribute to the inner product, so
    // remainder shapes give the same result
    let ak0 = t * 16u + aq * 4u;
    var av0 = vec4<f32>(0.0);
    if (arow0 < dims.m) {
      if (ak0 < dims.k) {
        av0.x = a[arow_base0 + ak0];
      }
      if (ak0 + 1u < dims.k) {
        av0.y = a[arow_base0 + ak0 + 1u];
      }
      if (ak0 + 2u < dims.k) {
        av0.z = a[arow_base0 + ak0 + 2u];
      }
      if (ak0 + 3u < dims.k) {
        av0.w = a[arow_base0 + ak0 + 3u];
      }
    }
    sa[sa_base0] = av0.x;
    sa[sa_base0 + 1u] = av0.y;
    sa[sa_base0 + 2u] = av0.z;
    sa[sa_base0 + 3u] = av0.w;
    var av1 = vec4<f32>(0.0);
    if (arow1 < dims.m) {
      if (ak0 < dims.k) {
        av1.x = a[arow_base1 + ak0];
      }
      if (ak0 + 1u < dims.k) {
        av1.y = a[arow_base1 + ak0 + 1u];
      }
      if (ak0 + 2u < dims.k) {
        av1.z = a[arow_base1 + ak0 + 2u];
      }
      if (ak0 + 3u < dims.k) {
        av1.w = a[arow_base1 + ak0 + 3u];
      }
    }
    sa[sa_base1] = av1.x;
    sa[sa_base1 + 1u] = av1.y;
    sa[sa_base1 + 2u] = av1.z;
    sa[sa_base1 + 3u] = av1.w;
    let brow0 = t * 16u + bk0;
    var bv4_0 = vec4<f32>(0.0);
    if (brow0 < dims.k) {
      let brow_base = brow0 * dims.n;
      if (bcol < dims.n) {
        bv4_0.x = b[brow_base + bcol];
      }
      if (bcol + 1u < dims.n) {
        bv4_0.y = b[brow_base + bcol + 1u];
      }
      if (bcol + 2u < dims.n) {
        bv4_0.z = b[brow_base + bcol + 2u];
      }
      if (bcol + 3u < dims.n) {
        bv4_0.w = b[brow_base + bcol + 3u];
      }
    }
    sb[bk0 * 32u + bcq] = bv4_0;
    let brow1 = t * 16u + bk1;
    var bv4_1 = vec4<f32>(0.0);
    if (brow1 < dims.k) {
      let brow_base = brow1 * dims.n;
      if (bcol < dims.n) {
        bv4_1.x = b[brow_base + bcol];
      }
      if (bcol + 1u < dims.n) {
        bv4_1.y = b[brow_base + bcol + 1u];
      }
      if (bcol + 2u < dims.n) {
        bv4_1.z = b[brow_base + bcol + 2u];
      }
      if (bcol + 3u < dims.n) {
        bv4_1.w = b[brow_base + bcol + 3u];
      }
    }
    sb[bk1 * 32u + bcq] = bv4_1;
    workgroupBarrier();
    // 10 shared loads (2 vec4 of B + 8 scalars of A) give 64 MACs.
    // The reduction is sequential in ascending k, and the addition order per output element matches
    // the 16×16 kernel with 1 output per thread exactly.
    for (var kk = 0u; kk < 16u; kk = kk + 1u) {
      let bv0 = sb[kk * 32u + lid.x * 2u];
      let bv1 = sb[kk * 32u + lid.x * 2u + 1u];
      acc0_0 = acc0_0 + sa[(lid.y * 8u + 0u) * 16u + kk] * bv0;
      acc0_1 = acc0_1 + sa[(lid.y * 8u + 0u) * 16u + kk] * bv1;
      acc1_0 = acc1_0 + sa[(lid.y * 8u + 1u) * 16u + kk] * bv0;
      acc1_1 = acc1_1 + sa[(lid.y * 8u + 1u) * 16u + kk] * bv1;
      acc2_0 = acc2_0 + sa[(lid.y * 8u + 2u) * 16u + kk] * bv0;
      acc2_1 = acc2_1 + sa[(lid.y * 8u + 2u) * 16u + kk] * bv1;
      acc3_0 = acc3_0 + sa[(lid.y * 8u + 3u) * 16u + kk] * bv0;
      acc3_1 = acc3_1 + sa[(lid.y * 8u + 3u) * 16u + kk] * bv1;
      acc4_0 = acc4_0 + sa[(lid.y * 8u + 4u) * 16u + kk] * bv0;
      acc4_1 = acc4_1 + sa[(lid.y * 8u + 4u) * 16u + kk] * bv1;
      acc5_0 = acc5_0 + sa[(lid.y * 8u + 5u) * 16u + kk] * bv0;
      acc5_1 = acc5_1 + sa[(lid.y * 8u + 5u) * 16u + kk] * bv1;
      acc6_0 = acc6_0 + sa[(lid.y * 8u + 6u) * 16u + kk] * bv0;
      acc6_1 = acc6_1 + sa[(lid.y * 8u + 6u) * 16u + kk] * bv1;
      acc7_0 = acc7_0 + sa[(lid.y * 8u + 7u) * 16u + kk] * bv0;
      acc7_1 = acc7_1 + sa[(lid.y * 8u + 7u) * 16u + kk] * bv1;
    }
    workgroupBarrier();
  }
  let ocol = wid.x * 128u + lid.x * 8u;
  let orow0 = wid.y * 128u + lid.y * 8u;
  let orow1 = orow0 + 1u;
  let orow2 = orow0 + 2u;
  let orow3 = orow0 + 3u;
  let orow4 = orow0 + 4u;
  let orow5 = orow0 + 5u;
  let orow6 = orow0 + 6u;
  let orow7 = orow0 + 7u;
  if (orow0 < dims.m) {
    let obase = orow0 * dims.n;
    if (ocol < dims.n) {
      c[obase + ocol] = acc0_0.x;
    }
    if (ocol + 1u < dims.n) {
      c[obase + ocol + 1u] = acc0_0.y;
    }
    if (ocol + 2u < dims.n) {
      c[obase + ocol + 2u] = acc0_0.z;
    }
    if (ocol + 3u < dims.n) {
      c[obase + ocol + 3u] = acc0_0.w;
    }
    if (ocol + 4u < dims.n) {
      c[obase + ocol + 4u] = acc0_1.x;
    }
    if (ocol + 5u < dims.n) {
      c[obase + ocol + 5u] = acc0_1.y;
    }
    if (ocol + 6u < dims.n) {
      c[obase + ocol + 6u] = acc0_1.z;
    }
    if (ocol + 7u < dims.n) {
      c[obase + ocol + 7u] = acc0_1.w;
    }
  }
  if (orow1 < dims.m) {
    let obase = orow1 * dims.n;
    if (ocol < dims.n) {
      c[obase + ocol] = acc1_0.x;
    }
    if (ocol + 1u < dims.n) {
      c[obase + ocol + 1u] = acc1_0.y;
    }
    if (ocol + 2u < dims.n) {
      c[obase + ocol + 2u] = acc1_0.z;
    }
    if (ocol + 3u < dims.n) {
      c[obase + ocol + 3u] = acc1_0.w;
    }
    if (ocol + 4u < dims.n) {
      c[obase + ocol + 4u] = acc1_1.x;
    }
    if (ocol + 5u < dims.n) {
      c[obase + ocol + 5u] = acc1_1.y;
    }
    if (ocol + 6u < dims.n) {
      c[obase + ocol + 6u] = acc1_1.z;
    }
    if (ocol + 7u < dims.n) {
      c[obase + ocol + 7u] = acc1_1.w;
    }
  }
  if (orow2 < dims.m) {
    let obase = orow2 * dims.n;
    if (ocol < dims.n) {
      c[obase + ocol] = acc2_0.x;
    }
    if (ocol + 1u < dims.n) {
      c[obase + ocol + 1u] = acc2_0.y;
    }
    if (ocol + 2u < dims.n) {
      c[obase + ocol + 2u] = acc2_0.z;
    }
    if (ocol + 3u < dims.n) {
      c[obase + ocol + 3u] = acc2_0.w;
    }
    if (ocol + 4u < dims.n) {
      c[obase + ocol + 4u] = acc2_1.x;
    }
    if (ocol + 5u < dims.n) {
      c[obase + ocol + 5u] = acc2_1.y;
    }
    if (ocol + 6u < dims.n) {
      c[obase + ocol + 6u] = acc2_1.z;
    }
    if (ocol + 7u < dims.n) {
      c[obase + ocol + 7u] = acc2_1.w;
    }
  }
  if (orow3 < dims.m) {
    let obase = orow3 * dims.n;
    if (ocol < dims.n) {
      c[obase + ocol] = acc3_0.x;
    }
    if (ocol + 1u < dims.n) {
      c[obase + ocol + 1u] = acc3_0.y;
    }
    if (ocol + 2u < dims.n) {
      c[obase + ocol + 2u] = acc3_0.z;
    }
    if (ocol + 3u < dims.n) {
      c[obase + ocol + 3u] = acc3_0.w;
    }
    if (ocol + 4u < dims.n) {
      c[obase + ocol + 4u] = acc3_1.x;
    }
    if (ocol + 5u < dims.n) {
      c[obase + ocol + 5u] = acc3_1.y;
    }
    if (ocol + 6u < dims.n) {
      c[obase + ocol + 6u] = acc3_1.z;
    }
    if (ocol + 7u < dims.n) {
      c[obase + ocol + 7u] = acc3_1.w;
    }
  }
  if (orow4 < dims.m) {
    let obase = orow4 * dims.n;
    if (ocol < dims.n) {
      c[obase + ocol] = acc4_0.x;
    }
    if (ocol + 1u < dims.n) {
      c[obase + ocol + 1u] = acc4_0.y;
    }
    if (ocol + 2u < dims.n) {
      c[obase + ocol + 2u] = acc4_0.z;
    }
    if (ocol + 3u < dims.n) {
      c[obase + ocol + 3u] = acc4_0.w;
    }
    if (ocol + 4u < dims.n) {
      c[obase + ocol + 4u] = acc4_1.x;
    }
    if (ocol + 5u < dims.n) {
      c[obase + ocol + 5u] = acc4_1.y;
    }
    if (ocol + 6u < dims.n) {
      c[obase + ocol + 6u] = acc4_1.z;
    }
    if (ocol + 7u < dims.n) {
      c[obase + ocol + 7u] = acc4_1.w;
    }
  }
  if (orow5 < dims.m) {
    let obase = orow5 * dims.n;
    if (ocol < dims.n) {
      c[obase + ocol] = acc5_0.x;
    }
    if (ocol + 1u < dims.n) {
      c[obase + ocol + 1u] = acc5_0.y;
    }
    if (ocol + 2u < dims.n) {
      c[obase + ocol + 2u] = acc5_0.z;
    }
    if (ocol + 3u < dims.n) {
      c[obase + ocol + 3u] = acc5_0.w;
    }
    if (ocol + 4u < dims.n) {
      c[obase + ocol + 4u] = acc5_1.x;
    }
    if (ocol + 5u < dims.n) {
      c[obase + ocol + 5u] = acc5_1.y;
    }
    if (ocol + 6u < dims.n) {
      c[obase + ocol + 6u] = acc5_1.z;
    }
    if (ocol + 7u < dims.n) {
      c[obase + ocol + 7u] = acc5_1.w;
    }
  }
  if (orow6 < dims.m) {
    let obase = orow6 * dims.n;
    if (ocol < dims.n) {
      c[obase + ocol] = acc6_0.x;
    }
    if (ocol + 1u < dims.n) {
      c[obase + ocol + 1u] = acc6_0.y;
    }
    if (ocol + 2u < dims.n) {
      c[obase + ocol + 2u] = acc6_0.z;
    }
    if (ocol + 3u < dims.n) {
      c[obase + ocol + 3u] = acc6_0.w;
    }
    if (ocol + 4u < dims.n) {
      c[obase + ocol + 4u] = acc6_1.x;
    }
    if (ocol + 5u < dims.n) {
      c[obase + ocol + 5u] = acc6_1.y;
    }
    if (ocol + 6u < dims.n) {
      c[obase + ocol + 6u] = acc6_1.z;
    }
    if (ocol + 7u < dims.n) {
      c[obase + ocol + 7u] = acc6_1.w;
    }
  }
  if (orow7 < dims.m) {
    let obase = orow7 * dims.n;
    if (ocol < dims.n) {
      c[obase + ocol] = acc7_0.x;
    }
    if (ocol + 1u < dims.n) {
      c[obase + ocol + 1u] = acc7_0.y;
    }
    if (ocol + 2u < dims.n) {
      c[obase + ocol + 2u] = acc7_0.z;
    }
    if (ocol + 3u < dims.n) {
      c[obase + ocol + 3u] = acc7_0.w;
    }
    if (ocol + 4u < dims.n) {
      c[obase + ocol + 4u] = acc7_1.x;
    }
    if (ocol + 5u < dims.n) {
      c[obase + ocol + 5u] = acc7_1.y;
    }
    if (ocol + 6u < dims.n) {
      c[obase + ocol + 6u] = acc7_1.z;
    }
    if (ocol + 7u < dims.n) {
      c[obase + ocol + 7u] = acc7_1.w;
    }
  }
}
