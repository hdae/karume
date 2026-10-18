// karume attention_state_pv (states-shaped O = P @ V, P = exp(S − m)·inv not materialised, f32, shared V row tiles of the GEMM skeleton, GQA, レジスタ 16x16 タイル / 1 スレッド 1x4 / wg 4x16)
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
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read> ins_v: array<f32>;
@group(0) @binding(4) var<storage, read> slot_v: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<uniform> lengths: Lengths;

fn slot_row(col: u32) -> u32 {
  return col;
}

fn column_base(past: u32) -> u32 {
  return 0u;
}

fn live_columns(past: u32, query: u32) -> u32 {
  return past + query;
}

fn effective_rows(query: u32) -> u32 {
  if (query <= dims.row_offset) {
    return 0u;
  }
  return min(dims.m, query - dims.row_offset);
}

// The 2 sources of V (the same split as slot_v / ins_v of ③). The fill side builds the row start
// per K tile
fn v_read(from_slot: bool, index: u32) -> f32 {
  if (from_slot) {
    return slot_v[index];
  }
  return ins_v[index];
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
  let rows_live = effective_rows(lengths.query);
  let z = wid.z;
  let kv_plane = z / dims.kv_repeat;
  // A (S) and the row statistics are block-relative (the same as ③'s (z · rows_block + local row) ·
  // col_cap), and O is written from row row_offset of the whole chunk [B·H, M, D]
  let abase = z * dims.m * dims.k;
  let cbase = (z * dims.chunk_rows + dims.row_offset) * dims.n;
  let rbase = z * dims.m;
  // A tile assignment (16 rows × 4 quads over 64 threads in 1 rounds)
  let ar = tid / 4u;
  let aq = tid % 4u;
  let arow0 = wid.y * 16u + ar;
  let arow_base0 = abase + arow0 * dims.k;
  let sa_base0 = ar * 16u + aq * 4u;
  // On edge tiles arow >= m can happen, so the index falls back to 0 (the value read is used only in the
  // arow < m branch). Reading stats out of range is itself safe under WGSL's bounded access
  let stat_at0 = select(0u, (rbase + arow0) * 2u, arow0 < rows_live);
  let row_max0 = stats[stat_at0];
  let row_inv0 = stats[stat_at0 + 1u];
  // B tile assignment (K 16 rows × 4 column quads over 64 threads in 1 rounds)
  let bk0 = tid / 4u;
  let bcq = tid % 4u;
  let bcol = wid.x * 16u + bcq * 4u;
  // The K loop goes up to the live columns. **A row tile with not a single valid row does not
  // iterate at all** (the counterpart of ③'s pad rows not scanning live — work ∝ Q). It derives
  // from wid.y, so it is workgroup-uniform and does not break the uniformity requirement of the
  // inner workgroupBarrier
  let k_live = select(0u, live, wid.y * 16u < rows_live);
  // The loop condition is uniform (dims is a uniform buffer) — needed for the inner workgroupBarrier
  // to meet WGSL's uniformity requirement
  let tiles = (k_live + 15u) / 16u;
  var acc0_0 = vec4<f32>(0.0);
  for (var t = 0u; t < tiles; t = t + 1u) {
    // Fill out-of-range entries with 0. They do not contribute to the inner product, so
    // remainder shapes give the same result
    let ak0 = t * 16u + aq * 4u;
    var av0 = vec4<f32>(0.0);
    if (arow0 < rows_live) {
      if (ak0 < live) {
        av0.x = exp(s[arow_base0 + ak0] - row_max0) * row_inv0;
      }
      if (ak0 + 1u < live) {
        av0.y = exp(s[arow_base0 + ak0 + 1u] - row_max0) * row_inv0;
      }
      if (ak0 + 2u < live) {
        av0.z = exp(s[arow_base0 + ak0 + 2u] - row_max0) * row_inv0;
      }
      if (ak0 + 3u < live) {
        av0.w = exp(s[arow_base0 + ak0 + 3u] - row_max0) * row_inv0;
      }
    }
    sa[sa_base0] = av0.x;
    sa[sa_base0 + 1u] = av0.y;
    sa[sa_base0 + 2u] = av0.z;
    sa[sa_base0 + 3u] = av0.w;
    let brow0 = t * 16u + bk0;
    var bv4_0 = vec4<f32>(0.0);
    if (brow0 < live) {
      // V row start of column col (same expression as ③). It changes per K tile, so it cannot be
      // folded into the prologue
      let vcol0 = base_col + brow0;
      let vpast0 = vcol0 < past;
      var vrow_base0 = 0u;
      if (vpast0) {
        vrow_base0 = (kv_plane * dims.capacity + slot_row(vcol0)) * dims.n;
      } else {
        vrow_base0 = (kv_plane * dims.chunk_rows + (vcol0 - past)) * dims.n;
      }
      if (bcol < dims.n) {
        bv4_0.x = v_read(vpast0, vrow_base0 + bcol);
      }
      if (bcol + 1u < dims.n) {
        bv4_0.y = v_read(vpast0, vrow_base0 + bcol + 1u);
      }
      if (bcol + 2u < dims.n) {
        bv4_0.z = v_read(vpast0, vrow_base0 + bcol + 2u);
      }
      if (bcol + 3u < dims.n) {
        bv4_0.w = v_read(vpast0, vrow_base0 + bcol + 3u);
      }
    }
    sb[bk0 * 4u + bcq] = bv4_0;
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
  // All rows of rows_block (full-write, as in ③). Pad rows get **exactly 0**, not acc
  let ocol = wid.x * 16u + lid.x * 4u;
  let orow0 = wid.y * 16u + lid.y * 1u;
  if (orow0 < dims.m) {
    let obase = cbase + orow0 * dims.n;
    let live_row0 = orow0 < rows_live;
    if (ocol < dims.n) {
      out[obase + ocol] = select(0.0, acc0_0.x, live_row0);
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = select(0.0, acc0_0.y, live_row0);
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = select(0.0, acc0_0.z, live_row0);
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = select(0.0, acc0_0.w, live_row0);
    }
  }
}
