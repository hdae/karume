// karume attention_state_qk (states-shaped S materialisation, f32, shared K row tiles of the GEMM skeleton, sliding window, GQA, register tile 128x128 / 8x8 per thread / wg 16x16)
struct Lengths {
  past: u32,
  query: u32,
}
struct Dims {
  m: u32,
  n: u32,
  k: u32,
  row_offset: u32,
  chunk_rows: u32,
  kv_repeat: u32,
  window: u32,
  capacity: u32,
  neg_inf: u32,
  scale: f32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> ins_k: array<f32>;
@group(0) @binding(3) var<storage, read> slot_k: array<f32>;
@group(0) @binding(4) var<storage, read_write> s: array<f32>;
@group(0) @binding(5) var<uniform> lengths: Lengths;

fn slot_row(col: u32) -> u32 {
  return col % dims.capacity;
}

fn column_base(past: u32) -> u32 {
  return past - min(past, dims.window - 1u);
}

fn live_columns(past: u32, query: u32) -> u32 {
  return min(past, dims.window - 1u) + query;
}

fn in_window(col: u32, limit: u32) -> bool {
  return col <= limit && (limit - col) < dims.window;
}

fn effective_rows(query: u32) -> u32 {
  if (query <= dims.row_offset) {
    return 0u;
  }
  return min(dims.m, query - dims.row_offset);
}

// The 2 sources of K (the same split as score_slot / score_ins of ①). Row starts are already folded
// per column, so this is only the choice of binding
fn k_read(from_slot: bool, index: u32) -> f32 {
  if (from_slot) {
    return slot_k[index];
  }
  return ins_k[index];
}

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
  // The logical lengths are uniform (owned by the context), so workgroup-uniform — the premise for the
  // inner workgroupBarrier to meet WGSL's uniformity requirement
  let past = lengths.past;
  let live = live_columns(past, lengths.query);
  let base_col = column_base(past);
  let neg_inf = bitcast<f32>(dims.neg_inf);
  let z = wid.z;
  let kv_plane = z / dims.kv_repeat;
  // A (q) is read from row row_offset of the whole chunk [B·H, M, D], and S is written
  // block-relative from row 0 (the same as ①'s index (z · rows_block + local row) · col_cap + cl)
  let abase = (z * dims.chunk_rows + dims.row_offset) * dims.k;
  let cbase = z * dims.m * dims.n;
  // A tile assignment (128 rows × 4 quads over 256 threads in 2 rounds)
  let ar = tid / 4u;
  let aq = tid % 4u;
  let arow0 = wid.y * 128u + ar;
  let arow_base0 = abase + arow0 * dims.k;
  let sa_base0 = ar * 16u + aq * 4u;
  let arow1 = arow0 + 64u;
  let arow_base1 = arow_base0 + 64u * dims.k;
  let sa_base1 = sa_base0 + 1024u;
  // K tile assignment (128 columns (logical col) × 4 quads over 256 threads in 2 rounds).
  // K is read as [row, D] as is and **placed transposed on the shared-memory side** (the same
  // structure as fused attention's k read)
  let wc0 = tid / 4u;
  let wq = tid % 4u;
  let wcol0 = wid.x * 128u + wc0;
  let wsq0 = wc0 / 4u;
  let wsl0 = wc0 % 4u;
  let sb_base0 = (wq * 4u) * 32u + wsq0;
  let wc1 = wc0 + 64u;
  let wcol1 = wcol0 + 64u;
  let wsq1 = wc1 / 4u;
  let wsl1 = wc1 % 4u;
  let sb_base1 = (wq * 4u) * 32u + wsq1;
  // K row start and source per column (invariant in the K tile loop, so folded once here). Same
  // expression as ① — col < P from the slot (physical row via slot_row, the same expression for
  // read and write), the rest from ins at col − P
  let kcol0 = base_col + wcol0;
  let kpast0 = kcol0 < past;
  var krow_base0 = 0u;
  if (kpast0) {
    krow_base0 = (kv_plane * dims.capacity + slot_row(kcol0)) * dims.k;
  } else {
    krow_base0 = (kv_plane * dims.chunk_rows + (kcol0 - past)) * dims.k;
  }
  let kcol1 = base_col + wcol1;
  let kpast1 = kcol1 < past;
  var krow_base1 = 0u;
  if (kpast1) {
    krow_base1 = (kv_plane * dims.capacity + slot_row(kcol1)) * dims.k;
  } else {
    krow_base1 = (kv_plane * dims.chunk_rows + (kcol1 - past)) * dims.k;
  }
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
        av0.x = q[arow_base0 + ak0] * dims.scale;
      }
      if (ak0 + 1u < dims.k) {
        av0.y = q[arow_base0 + ak0 + 1u] * dims.scale;
      }
      if (ak0 + 2u < dims.k) {
        av0.z = q[arow_base0 + ak0 + 2u] * dims.scale;
      }
      if (ak0 + 3u < dims.k) {
        av0.w = q[arow_base0 + ak0 + 3u] * dims.scale;
      }
    }
    sa[sa_base0] = av0.x;
    sa[sa_base0 + 1u] = av0.y;
    sa[sa_base0 + 2u] = av0.z;
    sa[sa_base0 + 3u] = av0.w;
    var av1 = vec4<f32>(0.0);
    if (arow1 < dims.m) {
      if (ak0 < dims.k) {
        av1.x = q[arow_base1 + ak0] * dims.scale;
      }
      if (ak0 + 1u < dims.k) {
        av1.y = q[arow_base1 + ak0 + 1u] * dims.scale;
      }
      if (ak0 + 2u < dims.k) {
        av1.z = q[arow_base1 + ak0 + 2u] * dims.scale;
      }
      if (ak0 + 3u < dims.k) {
        av1.w = q[arow_base1 + ak0 + 3u] * dims.scale;
      }
    }
    sa[sa_base1] = av1.x;
    sa[sa_base1 + 1u] = av1.y;
    sa[sa_base1 + 2u] = av1.z;
    sa[sa_base1 + 3u] = av1.w;
    let wk0 = t * 16u + wq * 4u;
    var wv0 = vec4<f32>(0.0);
    if (wcol0 < live) {
      if (wk0 < dims.k) {
        wv0.x = k_read(kpast0, krow_base0 + wk0);
      }
      if (wk0 + 1u < dims.k) {
        wv0.y = k_read(kpast0, krow_base0 + wk0 + 1u);
      }
      if (wk0 + 2u < dims.k) {
        wv0.z = k_read(kpast0, krow_base0 + wk0 + 2u);
      }
      if (wk0 + 3u < dims.k) {
        wv0.w = k_read(kpast0, krow_base0 + wk0 + 3u);
      }
    }
    // Half-scale contract (same as ①): scale is applied to **both** the q side and the k side
    wv0 = wv0 * dims.scale;
    switch wsl0 {
      case 0u: {
        sb[sb_base0].x = wv0.x;
        sb[sb_base0 + 32u].x = wv0.y;
        sb[sb_base0 + 64u].x = wv0.z;
        sb[sb_base0 + 96u].x = wv0.w;
      }
      case 1u: {
        sb[sb_base0].y = wv0.x;
        sb[sb_base0 + 32u].y = wv0.y;
        sb[sb_base0 + 64u].y = wv0.z;
        sb[sb_base0 + 96u].y = wv0.w;
      }
      case 2u: {
        sb[sb_base0].z = wv0.x;
        sb[sb_base0 + 32u].z = wv0.y;
        sb[sb_base0 + 64u].z = wv0.z;
        sb[sb_base0 + 96u].z = wv0.w;
      }
      default: {
        sb[sb_base0].w = wv0.x;
        sb[sb_base0 + 32u].w = wv0.y;
        sb[sb_base0 + 64u].w = wv0.z;
        sb[sb_base0 + 96u].w = wv0.w;
      }
    }
    var wv1 = vec4<f32>(0.0);
    if (wcol1 < live) {
      if (wk0 < dims.k) {
        wv1.x = k_read(kpast1, krow_base1 + wk0);
      }
      if (wk0 + 1u < dims.k) {
        wv1.y = k_read(kpast1, krow_base1 + wk0 + 1u);
      }
      if (wk0 + 2u < dims.k) {
        wv1.z = k_read(kpast1, krow_base1 + wk0 + 2u);
      }
      if (wk0 + 3u < dims.k) {
        wv1.w = k_read(kpast1, krow_base1 + wk0 + 3u);
      }
    }
    // Half-scale contract (same as ①): scale is applied to **both** the q side and the k side
    wv1 = wv1 * dims.scale;
    switch wsl1 {
      case 0u: {
        sb[sb_base1].x = wv1.x;
        sb[sb_base1 + 32u].x = wv1.y;
        sb[sb_base1 + 64u].x = wv1.z;
        sb[sb_base1 + 96u].x = wv1.w;
      }
      case 1u: {
        sb[sb_base1].y = wv1.x;
        sb[sb_base1 + 32u].y = wv1.y;
        sb[sb_base1 + 64u].y = wv1.z;
        sb[sb_base1 + 96u].y = wv1.w;
      }
      case 2u: {
        sb[sb_base1].z = wv1.x;
        sb[sb_base1 + 32u].z = wv1.y;
        sb[sb_base1 + 64u].z = wv1.z;
        sb[sb_base1 + 96u].z = wv1.w;
      }
      default: {
        sb[sb_base1].w = wv1.x;
        sb[sb_base1 + 32u].w = wv1.y;
        sb[sb_base1 + 64u].w = wv1.z;
        sb[sb_base1 + 96u].w = wv1.w;
      }
    }
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
  // Rows up to the **valid rows** (nobody reads S of pad rows — ③ writes 0 and returns).
  // Columns up to live (leaving the leftovers in [live, col_cap) untouched is correct, as in ①)
  let rows_live = effective_rows(lengths.query);
  let ocol = wid.x * 128u + lid.x * 8u;
  let orow0 = wid.y * 128u + lid.y * 8u;
  let orow1 = orow0 + 1u;
  let orow2 = orow0 + 2u;
  let orow3 = orow0 + 3u;
  let orow4 = orow0 + 4u;
  let orow5 = orow0 + 5u;
  let orow6 = orow0 + 6u;
  let orow7 = orow0 + 7u;
  if (orow0 < rows_live) {
    let obase = cbase + orow0 * dims.n;
    let limit = past + dims.row_offset + orow0;
    if (ocol < live) {
      s[obase + ocol] = select(neg_inf, acc0_0.x, in_window(base_col + ocol, limit));
    }
    if (ocol + 1u < live) {
      s[obase + ocol + 1u] = select(neg_inf, acc0_0.y, in_window(base_col + ocol + 1u, limit));
    }
    if (ocol + 2u < live) {
      s[obase + ocol + 2u] = select(neg_inf, acc0_0.z, in_window(base_col + ocol + 2u, limit));
    }
    if (ocol + 3u < live) {
      s[obase + ocol + 3u] = select(neg_inf, acc0_0.w, in_window(base_col + ocol + 3u, limit));
    }
    if (ocol + 4u < live) {
      s[obase + ocol + 4u] = select(neg_inf, acc0_1.x, in_window(base_col + ocol + 4u, limit));
    }
    if (ocol + 5u < live) {
      s[obase + ocol + 5u] = select(neg_inf, acc0_1.y, in_window(base_col + ocol + 5u, limit));
    }
    if (ocol + 6u < live) {
      s[obase + ocol + 6u] = select(neg_inf, acc0_1.z, in_window(base_col + ocol + 6u, limit));
    }
    if (ocol + 7u < live) {
      s[obase + ocol + 7u] = select(neg_inf, acc0_1.w, in_window(base_col + ocol + 7u, limit));
    }
  }
  if (orow1 < rows_live) {
    let obase = cbase + orow1 * dims.n;
    let limit = past + dims.row_offset + orow1;
    if (ocol < live) {
      s[obase + ocol] = select(neg_inf, acc1_0.x, in_window(base_col + ocol, limit));
    }
    if (ocol + 1u < live) {
      s[obase + ocol + 1u] = select(neg_inf, acc1_0.y, in_window(base_col + ocol + 1u, limit));
    }
    if (ocol + 2u < live) {
      s[obase + ocol + 2u] = select(neg_inf, acc1_0.z, in_window(base_col + ocol + 2u, limit));
    }
    if (ocol + 3u < live) {
      s[obase + ocol + 3u] = select(neg_inf, acc1_0.w, in_window(base_col + ocol + 3u, limit));
    }
    if (ocol + 4u < live) {
      s[obase + ocol + 4u] = select(neg_inf, acc1_1.x, in_window(base_col + ocol + 4u, limit));
    }
    if (ocol + 5u < live) {
      s[obase + ocol + 5u] = select(neg_inf, acc1_1.y, in_window(base_col + ocol + 5u, limit));
    }
    if (ocol + 6u < live) {
      s[obase + ocol + 6u] = select(neg_inf, acc1_1.z, in_window(base_col + ocol + 6u, limit));
    }
    if (ocol + 7u < live) {
      s[obase + ocol + 7u] = select(neg_inf, acc1_1.w, in_window(base_col + ocol + 7u, limit));
    }
  }
  if (orow2 < rows_live) {
    let obase = cbase + orow2 * dims.n;
    let limit = past + dims.row_offset + orow2;
    if (ocol < live) {
      s[obase + ocol] = select(neg_inf, acc2_0.x, in_window(base_col + ocol, limit));
    }
    if (ocol + 1u < live) {
      s[obase + ocol + 1u] = select(neg_inf, acc2_0.y, in_window(base_col + ocol + 1u, limit));
    }
    if (ocol + 2u < live) {
      s[obase + ocol + 2u] = select(neg_inf, acc2_0.z, in_window(base_col + ocol + 2u, limit));
    }
    if (ocol + 3u < live) {
      s[obase + ocol + 3u] = select(neg_inf, acc2_0.w, in_window(base_col + ocol + 3u, limit));
    }
    if (ocol + 4u < live) {
      s[obase + ocol + 4u] = select(neg_inf, acc2_1.x, in_window(base_col + ocol + 4u, limit));
    }
    if (ocol + 5u < live) {
      s[obase + ocol + 5u] = select(neg_inf, acc2_1.y, in_window(base_col + ocol + 5u, limit));
    }
    if (ocol + 6u < live) {
      s[obase + ocol + 6u] = select(neg_inf, acc2_1.z, in_window(base_col + ocol + 6u, limit));
    }
    if (ocol + 7u < live) {
      s[obase + ocol + 7u] = select(neg_inf, acc2_1.w, in_window(base_col + ocol + 7u, limit));
    }
  }
  if (orow3 < rows_live) {
    let obase = cbase + orow3 * dims.n;
    let limit = past + dims.row_offset + orow3;
    if (ocol < live) {
      s[obase + ocol] = select(neg_inf, acc3_0.x, in_window(base_col + ocol, limit));
    }
    if (ocol + 1u < live) {
      s[obase + ocol + 1u] = select(neg_inf, acc3_0.y, in_window(base_col + ocol + 1u, limit));
    }
    if (ocol + 2u < live) {
      s[obase + ocol + 2u] = select(neg_inf, acc3_0.z, in_window(base_col + ocol + 2u, limit));
    }
    if (ocol + 3u < live) {
      s[obase + ocol + 3u] = select(neg_inf, acc3_0.w, in_window(base_col + ocol + 3u, limit));
    }
    if (ocol + 4u < live) {
      s[obase + ocol + 4u] = select(neg_inf, acc3_1.x, in_window(base_col + ocol + 4u, limit));
    }
    if (ocol + 5u < live) {
      s[obase + ocol + 5u] = select(neg_inf, acc3_1.y, in_window(base_col + ocol + 5u, limit));
    }
    if (ocol + 6u < live) {
      s[obase + ocol + 6u] = select(neg_inf, acc3_1.z, in_window(base_col + ocol + 6u, limit));
    }
    if (ocol + 7u < live) {
      s[obase + ocol + 7u] = select(neg_inf, acc3_1.w, in_window(base_col + ocol + 7u, limit));
    }
  }
  if (orow4 < rows_live) {
    let obase = cbase + orow4 * dims.n;
    let limit = past + dims.row_offset + orow4;
    if (ocol < live) {
      s[obase + ocol] = select(neg_inf, acc4_0.x, in_window(base_col + ocol, limit));
    }
    if (ocol + 1u < live) {
      s[obase + ocol + 1u] = select(neg_inf, acc4_0.y, in_window(base_col + ocol + 1u, limit));
    }
    if (ocol + 2u < live) {
      s[obase + ocol + 2u] = select(neg_inf, acc4_0.z, in_window(base_col + ocol + 2u, limit));
    }
    if (ocol + 3u < live) {
      s[obase + ocol + 3u] = select(neg_inf, acc4_0.w, in_window(base_col + ocol + 3u, limit));
    }
    if (ocol + 4u < live) {
      s[obase + ocol + 4u] = select(neg_inf, acc4_1.x, in_window(base_col + ocol + 4u, limit));
    }
    if (ocol + 5u < live) {
      s[obase + ocol + 5u] = select(neg_inf, acc4_1.y, in_window(base_col + ocol + 5u, limit));
    }
    if (ocol + 6u < live) {
      s[obase + ocol + 6u] = select(neg_inf, acc4_1.z, in_window(base_col + ocol + 6u, limit));
    }
    if (ocol + 7u < live) {
      s[obase + ocol + 7u] = select(neg_inf, acc4_1.w, in_window(base_col + ocol + 7u, limit));
    }
  }
  if (orow5 < rows_live) {
    let obase = cbase + orow5 * dims.n;
    let limit = past + dims.row_offset + orow5;
    if (ocol < live) {
      s[obase + ocol] = select(neg_inf, acc5_0.x, in_window(base_col + ocol, limit));
    }
    if (ocol + 1u < live) {
      s[obase + ocol + 1u] = select(neg_inf, acc5_0.y, in_window(base_col + ocol + 1u, limit));
    }
    if (ocol + 2u < live) {
      s[obase + ocol + 2u] = select(neg_inf, acc5_0.z, in_window(base_col + ocol + 2u, limit));
    }
    if (ocol + 3u < live) {
      s[obase + ocol + 3u] = select(neg_inf, acc5_0.w, in_window(base_col + ocol + 3u, limit));
    }
    if (ocol + 4u < live) {
      s[obase + ocol + 4u] = select(neg_inf, acc5_1.x, in_window(base_col + ocol + 4u, limit));
    }
    if (ocol + 5u < live) {
      s[obase + ocol + 5u] = select(neg_inf, acc5_1.y, in_window(base_col + ocol + 5u, limit));
    }
    if (ocol + 6u < live) {
      s[obase + ocol + 6u] = select(neg_inf, acc5_1.z, in_window(base_col + ocol + 6u, limit));
    }
    if (ocol + 7u < live) {
      s[obase + ocol + 7u] = select(neg_inf, acc5_1.w, in_window(base_col + ocol + 7u, limit));
    }
  }
  if (orow6 < rows_live) {
    let obase = cbase + orow6 * dims.n;
    let limit = past + dims.row_offset + orow6;
    if (ocol < live) {
      s[obase + ocol] = select(neg_inf, acc6_0.x, in_window(base_col + ocol, limit));
    }
    if (ocol + 1u < live) {
      s[obase + ocol + 1u] = select(neg_inf, acc6_0.y, in_window(base_col + ocol + 1u, limit));
    }
    if (ocol + 2u < live) {
      s[obase + ocol + 2u] = select(neg_inf, acc6_0.z, in_window(base_col + ocol + 2u, limit));
    }
    if (ocol + 3u < live) {
      s[obase + ocol + 3u] = select(neg_inf, acc6_0.w, in_window(base_col + ocol + 3u, limit));
    }
    if (ocol + 4u < live) {
      s[obase + ocol + 4u] = select(neg_inf, acc6_1.x, in_window(base_col + ocol + 4u, limit));
    }
    if (ocol + 5u < live) {
      s[obase + ocol + 5u] = select(neg_inf, acc6_1.y, in_window(base_col + ocol + 5u, limit));
    }
    if (ocol + 6u < live) {
      s[obase + ocol + 6u] = select(neg_inf, acc6_1.z, in_window(base_col + ocol + 6u, limit));
    }
    if (ocol + 7u < live) {
      s[obase + ocol + 7u] = select(neg_inf, acc6_1.w, in_window(base_col + ocol + 7u, limit));
    }
  }
  if (orow7 < rows_live) {
    let obase = cbase + orow7 * dims.n;
    let limit = past + dims.row_offset + orow7;
    if (ocol < live) {
      s[obase + ocol] = select(neg_inf, acc7_0.x, in_window(base_col + ocol, limit));
    }
    if (ocol + 1u < live) {
      s[obase + ocol + 1u] = select(neg_inf, acc7_0.y, in_window(base_col + ocol + 1u, limit));
    }
    if (ocol + 2u < live) {
      s[obase + ocol + 2u] = select(neg_inf, acc7_0.z, in_window(base_col + ocol + 2u, limit));
    }
    if (ocol + 3u < live) {
      s[obase + ocol + 3u] = select(neg_inf, acc7_0.w, in_window(base_col + ocol + 3u, limit));
    }
    if (ocol + 4u < live) {
      s[obase + ocol + 4u] = select(neg_inf, acc7_1.x, in_window(base_col + ocol + 4u, limit));
    }
    if (ocol + 5u < live) {
      s[obase + ocol + 5u] = select(neg_inf, acc7_1.y, in_window(base_col + ocol + 5u, limit));
    }
    if (ocol + 6u < live) {
      s[obase + ocol + 6u] = select(neg_inf, acc7_1.z, in_window(base_col + ocol + 6u, limit));
    }
    if (ocol + 7u < live) {
      s[obase + ocol + 7u] = select(neg_inf, acc7_1.w, in_window(base_col + ocol + 7u, limit));
    }
  }
}
