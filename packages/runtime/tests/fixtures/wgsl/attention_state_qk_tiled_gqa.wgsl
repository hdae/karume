// karume attention_state_qk (states-shaped S materialisation, f32, shared K row tiles of the GEMM skeleton, GQA, レジスタ 16x16 タイル / 1 スレッド 1x4 / wg 4x16)
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
  return col;
}

fn column_base(past: u32) -> u32 {
  return 0u;
}

fn live_columns(past: u32, query: u32) -> u32 {
  return past + query;
}

fn in_window(col: u32, limit: u32) -> bool {
  return col <= limit;
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

// Shared A tile (16 rows × K 16, scalar storage) and
// shared B tile (K 16 × 4 column quads, column direction grouped into vec4)
var<workgroup> sa: array<f32, 256>;
var<workgroup> sb: array<vec4<f32>, 64>;

@compute @workgroup_size(4, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 4u + lid.x;
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
  // A tile assignment (16 rows × 4 quads over 64 threads in 1 rounds)
  let ar = tid / 4u;
  let aq = tid % 4u;
  let arow0 = wid.y * 16u + ar;
  let arow_base0 = abase + arow0 * dims.k;
  let sa_base0 = ar * 16u + aq * 4u;
  // K tile assignment (16 columns (logical col) × 4 quads over 64 threads in 1 rounds).
  // K is read as [row, D] as is and **placed transposed on the shared-memory side** (the same
  // structure as fused attention's k read)
  let wc0 = tid / 4u;
  let wq = tid % 4u;
  let wcol0 = wid.x * 16u + wc0;
  let wsq0 = wc0 / 4u;
  let wsl0 = wc0 % 4u;
  let sb_base0 = (wq * 4u) * 4u + wsq0;
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
  // The loop condition is uniform (dims is a uniform buffer) — needed for the inner workgroupBarrier
  // to meet WGSL's uniformity requirement
  let tiles = (dims.k + 15u) / 16u;
  var acc0_0 = vec4<f32>(0.0);
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
        sb[sb_base0 + 4u].x = wv0.y;
        sb[sb_base0 + 8u].x = wv0.z;
        sb[sb_base0 + 12u].x = wv0.w;
      }
      case 1u: {
        sb[sb_base0].y = wv0.x;
        sb[sb_base0 + 4u].y = wv0.y;
        sb[sb_base0 + 8u].y = wv0.z;
        sb[sb_base0 + 12u].y = wv0.w;
      }
      case 2u: {
        sb[sb_base0].z = wv0.x;
        sb[sb_base0 + 4u].z = wv0.y;
        sb[sb_base0 + 8u].z = wv0.z;
        sb[sb_base0 + 12u].z = wv0.w;
      }
      default: {
        sb[sb_base0].w = wv0.x;
        sb[sb_base0 + 4u].w = wv0.y;
        sb[sb_base0 + 8u].w = wv0.z;
        sb[sb_base0 + 12u].w = wv0.w;
      }
    }
    workgroupBarrier();
    // 2 shared loads (1 vec4 of B + 1 scalars of A) give 4 MACs.
    // The reduction is sequential in ascending k, and the addition order per output element matches
    // the 16×16 kernel with 1 output per thread exactly.
    for (var kk = 0u; kk < 16u; kk = kk + 1u) {
      let bv0 = sb[kk * 4u + lid.x];
      acc0_0 = acc0_0 + sa[(lid.y * 1u + 0u) * 16u + kk] * bv0;
    }
    workgroupBarrier();
  }
  // Rows up to the **valid rows** (nobody reads S of pad rows — ③ writes 0 and returns).
  // Columns up to live (leaving the leftovers in [live, col_cap) untouched is correct, as in ①)
  let rows_live = effective_rows(lengths.query);
  let ocol = wid.x * 16u + lid.x * 4u;
  let orow0 = wid.y * 16u + lid.y * 1u;
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
  }
}
