// karume attention_state_pv (states 形の O = P @ V, P = exp(S − m)·inv は非実体化, f32, GEMM 骨格の V 行タイル共有, sliding window, GQA, レジスタ 128x128 タイル / 1 スレッド 8x8 / wg 16x16)
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
  return col % dims.capacity;
}

fn column_base(past: u32) -> u32 {
  return past - min(past, dims.window - 1u);
}

fn live_columns(past: u32, query: u32) -> u32 {
  return min(past, dims.window - 1u) + query;
}

fn effective_rows(query: u32) -> u32 {
  if (query <= dims.row_offset) {
    return 0u;
  }
  return min(dims.m, query - dims.row_offset);
}

// V の 2 源（③ の slot_v / ins_v と同じ踏み分け）。行頭は充填側が K タイルごとに作る
fn v_read(from_slot: bool, index: u32) -> f32 {
  if (from_slot) {
    return slot_v[index];
  }
  return ins_v[index];
}

// 共有 A タイル（128 行 × K 16・スカラ格納）と
// 共有 B タイル（K 16 × 列 quad 32・列方向を vec4 に束ねた形）
var<workgroup> sa: array<f32, 2048>;
var<workgroup> sb: array<vec4<f32>, 512>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 16u + lid.x;
  // 論理長は uniform（context 所有）なので workgroup 一様 — 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすための前提
  let past = lengths.past;
  let live = live_columns(past, lengths.query);
  let base_col = column_base(past);
  let rows_live = effective_rows(lengths.query);
  let z = wid.z;
  let kv_plane = z / dims.kv_repeat;
  // A（S）と行統計はブロック相対（③ の (z · rows_block + 局所行) · col_cap と同じ）、
  // O は chunk 全体 [B·H, M, D] の row_offset 行目から書く
  let abase = z * dims.m * dims.k;
  let cbase = (z * dims.chunk_rows + dims.row_offset) * dims.n;
  let rbase = z * dims.m;
  // A タイルの担当（128 行 × 4 quad を 256 スレッドで 2 巡）
  let ar = tid / 4u;
  let aq = tid % 4u;
  let arow0 = wid.y * 128u + ar;
  let arow_base0 = abase + arow0 * dims.k;
  let sa_base0 = ar * 16u + aq * 4u;
  let arow1 = arow0 + 64u;
  let arow_base1 = arow_base0 + 64u * dims.k;
  let sa_base1 = sa_base0 + 1024u;
  // 端タイルでは arow >= m がありうるので添字を 0 へ倒す（読んだ値は arow < m の枝でしか
  // 使われない）。範囲外の stats を読むこと自体は WGSL の境界付きアクセスで安全
  let stat_at0 = select(0u, (rbase + arow0) * 2u, arow0 < rows_live);
  let row_max0 = stats[stat_at0];
  let row_inv0 = stats[stat_at0 + 1u];
  let stat_at1 = select(0u, (rbase + arow1) * 2u, arow1 < rows_live);
  let row_max1 = stats[stat_at1];
  let row_inv1 = stats[stat_at1 + 1u];
  // B タイルの担当（K 16 行 × 列 quad 32 を 256 スレッドで 2 巡）
  let bk0 = tid / 32u;
  let bcq = tid % 32u;
  let bcol = wid.x * 128u + bcq * 4u;
  let bk1 = bk0 + 8u;
  // K ループは live 列まで。**有効行を 1 行も含まない行タイルは 1 周も回さない**（③ の
  // pad 行が live を走査しないことの写し — 仕事量 ∝ Q）。wid.y 由来なので workgroup 一様で、
  // 内側の workgroupBarrier の一様性要件を壊さない
  let k_live = select(0u, live, wid.y * 128u < rows_live);
  // ループ条件は uniform（dims は uniform バッファ）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすために必要
  let tiles = (k_live + 15u) / 16u;
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
    // 範囲外は 0 で埋める。内積に寄与しないので端数 shape でも結果は変わらない
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
    var av1 = vec4<f32>(0.0);
    if (arow1 < rows_live) {
      if (ak0 < live) {
        av1.x = exp(s[arow_base1 + ak0] - row_max1) * row_inv1;
      }
      if (ak0 + 1u < live) {
        av1.y = exp(s[arow_base1 + ak0 + 1u] - row_max1) * row_inv1;
      }
      if (ak0 + 2u < live) {
        av1.z = exp(s[arow_base1 + ak0 + 2u] - row_max1) * row_inv1;
      }
      if (ak0 + 3u < live) {
        av1.w = exp(s[arow_base1 + ak0 + 3u] - row_max1) * row_inv1;
      }
    }
    sa[sa_base1] = av1.x;
    sa[sa_base1 + 1u] = av1.y;
    sa[sa_base1 + 2u] = av1.z;
    sa[sa_base1 + 3u] = av1.w;
    let brow0 = t * 16u + bk0;
    var bv4_0 = vec4<f32>(0.0);
    if (brow0 < live) {
      // 列 col の V 行頭（③ と同式）。K タイルごとに変わるので prologue へは畳めない
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
    sb[bk0 * 32u + bcq] = bv4_0;
    let brow1 = t * 16u + bk1;
    var bv4_1 = vec4<f32>(0.0);
    if (brow1 < live) {
      // 列 col の V 行頭（③ と同式）。K タイルごとに変わるので prologue へは畳めない
      let vcol1 = base_col + brow1;
      let vpast1 = vcol1 < past;
      var vrow_base1 = 0u;
      if (vpast1) {
        vrow_base1 = (kv_plane * dims.capacity + slot_row(vcol1)) * dims.n;
      } else {
        vrow_base1 = (kv_plane * dims.chunk_rows + (vcol1 - past)) * dims.n;
      }
      if (bcol < dims.n) {
        bv4_1.x = v_read(vpast1, vrow_base1 + bcol);
      }
      if (bcol + 1u < dims.n) {
        bv4_1.y = v_read(vpast1, vrow_base1 + bcol + 1u);
      }
      if (bcol + 2u < dims.n) {
        bv4_1.z = v_read(vpast1, vrow_base1 + bcol + 2u);
      }
      if (bcol + 3u < dims.n) {
        bv4_1.w = v_read(vpast1, vrow_base1 + bcol + 3u);
      }
    }
    sb[bk1 * 32u + bcq] = bv4_1;
    workgroupBarrier();
    // 共有ロード 10 回（B の vec4 2 + A のスカラ 8）で 64 MAC。
    // 縮約は k 昇順の逐次で、1 出力要素あたりの加算順序は 16×16 の 1 スレッド 1 出力と
    // 完全に一致する。
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
  // 行は rows_block 全て（③ と同じ full-write）。pad 行は acc ではなく**厳密 0**
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
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = select(0.0, acc0_1.x, live_row0);
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = select(0.0, acc0_1.y, live_row0);
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = select(0.0, acc0_1.z, live_row0);
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = select(0.0, acc0_1.w, live_row0);
    }
  }
  if (orow1 < dims.m) {
    let obase = cbase + orow1 * dims.n;
    let live_row1 = orow1 < rows_live;
    if (ocol < dims.n) {
      out[obase + ocol] = select(0.0, acc1_0.x, live_row1);
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = select(0.0, acc1_0.y, live_row1);
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = select(0.0, acc1_0.z, live_row1);
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = select(0.0, acc1_0.w, live_row1);
    }
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = select(0.0, acc1_1.x, live_row1);
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = select(0.0, acc1_1.y, live_row1);
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = select(0.0, acc1_1.z, live_row1);
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = select(0.0, acc1_1.w, live_row1);
    }
  }
  if (orow2 < dims.m) {
    let obase = cbase + orow2 * dims.n;
    let live_row2 = orow2 < rows_live;
    if (ocol < dims.n) {
      out[obase + ocol] = select(0.0, acc2_0.x, live_row2);
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = select(0.0, acc2_0.y, live_row2);
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = select(0.0, acc2_0.z, live_row2);
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = select(0.0, acc2_0.w, live_row2);
    }
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = select(0.0, acc2_1.x, live_row2);
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = select(0.0, acc2_1.y, live_row2);
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = select(0.0, acc2_1.z, live_row2);
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = select(0.0, acc2_1.w, live_row2);
    }
  }
  if (orow3 < dims.m) {
    let obase = cbase + orow3 * dims.n;
    let live_row3 = orow3 < rows_live;
    if (ocol < dims.n) {
      out[obase + ocol] = select(0.0, acc3_0.x, live_row3);
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = select(0.0, acc3_0.y, live_row3);
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = select(0.0, acc3_0.z, live_row3);
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = select(0.0, acc3_0.w, live_row3);
    }
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = select(0.0, acc3_1.x, live_row3);
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = select(0.0, acc3_1.y, live_row3);
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = select(0.0, acc3_1.z, live_row3);
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = select(0.0, acc3_1.w, live_row3);
    }
  }
  if (orow4 < dims.m) {
    let obase = cbase + orow4 * dims.n;
    let live_row4 = orow4 < rows_live;
    if (ocol < dims.n) {
      out[obase + ocol] = select(0.0, acc4_0.x, live_row4);
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = select(0.0, acc4_0.y, live_row4);
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = select(0.0, acc4_0.z, live_row4);
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = select(0.0, acc4_0.w, live_row4);
    }
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = select(0.0, acc4_1.x, live_row4);
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = select(0.0, acc4_1.y, live_row4);
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = select(0.0, acc4_1.z, live_row4);
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = select(0.0, acc4_1.w, live_row4);
    }
  }
  if (orow5 < dims.m) {
    let obase = cbase + orow5 * dims.n;
    let live_row5 = orow5 < rows_live;
    if (ocol < dims.n) {
      out[obase + ocol] = select(0.0, acc5_0.x, live_row5);
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = select(0.0, acc5_0.y, live_row5);
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = select(0.0, acc5_0.z, live_row5);
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = select(0.0, acc5_0.w, live_row5);
    }
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = select(0.0, acc5_1.x, live_row5);
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = select(0.0, acc5_1.y, live_row5);
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = select(0.0, acc5_1.z, live_row5);
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = select(0.0, acc5_1.w, live_row5);
    }
  }
  if (orow6 < dims.m) {
    let obase = cbase + orow6 * dims.n;
    let live_row6 = orow6 < rows_live;
    if (ocol < dims.n) {
      out[obase + ocol] = select(0.0, acc6_0.x, live_row6);
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = select(0.0, acc6_0.y, live_row6);
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = select(0.0, acc6_0.z, live_row6);
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = select(0.0, acc6_0.w, live_row6);
    }
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = select(0.0, acc6_1.x, live_row6);
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = select(0.0, acc6_1.y, live_row6);
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = select(0.0, acc6_1.z, live_row6);
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = select(0.0, acc6_1.w, live_row6);
    }
  }
  if (orow7 < dims.m) {
    let obase = cbase + orow7 * dims.n;
    let live_row7 = orow7 < rows_live;
    if (ocol < dims.n) {
      out[obase + ocol] = select(0.0, acc7_0.x, live_row7);
    }
    if (ocol + 1u < dims.n) {
      out[obase + ocol + 1u] = select(0.0, acc7_0.y, live_row7);
    }
    if (ocol + 2u < dims.n) {
      out[obase + ocol + 2u] = select(0.0, acc7_0.z, live_row7);
    }
    if (ocol + 3u < dims.n) {
      out[obase + ocol + 3u] = select(0.0, acc7_0.w, live_row7);
    }
    if (ocol + 4u < dims.n) {
      out[obase + ocol + 4u] = select(0.0, acc7_1.x, live_row7);
    }
    if (ocol + 5u < dims.n) {
      out[obase + ocol + 5u] = select(0.0, acc7_1.y, live_row7);
    }
    if (ocol + 6u < dims.n) {
      out[obase + ocol + 6u] = select(0.0, acc7_1.z, live_row7);
    }
    if (ocol + 7u < dims.n) {
      out[obase + ocol + 7u] = select(0.0, acc7_1.w, live_row7);
    }
  }
}
