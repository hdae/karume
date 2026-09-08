// karume attention_state_qk (states 形の S 実体化, f32, GEMM 骨格の K 行タイル共有, sliding window, レジスタ 16x16 タイル / 1 スレッド 1x4 / wg 4x16)
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

// K の 2 源（① の score_slot / score_ins と同じ踏み分け）。行頭は列ごとに畳んであるので、
// ここは束縛の選択だけ
fn k_read(from_slot: bool, index: u32) -> f32 {
  if (from_slot) {
    return slot_k[index];
  }
  return ins_k[index];
}

// 共有 A タイル（16 行 × K 16・スカラ格納）と
// 共有 B タイル（K 16 × 列 quad 4・列方向を vec4 に束ねた形）
var<workgroup> sa: array<f32, 256>;
var<workgroup> sb: array<vec4<f32>, 64>;

@compute @workgroup_size(4, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 4u + lid.x;
  // 論理長は uniform（context 所有）なので workgroup 一様 — 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすための前提
  let past = lengths.past;
  let live = live_columns(past, lengths.query);
  let base_col = column_base(past);
  let neg_inf = bitcast<f32>(dims.neg_inf);
  let z = wid.z;
  let kv_plane = z;
  // A（q）は chunk 全体 [B·H, M, D] の row_offset 行目から読み、S はブロック相対で 0 行目から
  // 書く（① の添字 (z · rows_block + 局所行) · col_cap + cl と同じ）
  let abase = (z * dims.chunk_rows + dims.row_offset) * dims.k;
  let cbase = z * dims.m * dims.n;
  // A タイルの担当（16 行 × 4 quad を 64 スレッドで 1 巡）
  let ar = tid / 4u;
  let aq = tid % 4u;
  let arow0 = wid.y * 16u + ar;
  let arow_base0 = abase + arow0 * dims.k;
  let sa_base0 = ar * 16u + aq * 4u;
  // K タイルの担当（16 列（論理 col）× 4 quad を 64 スレッドで 1 巡）。
  // K は [行, D] のまま読み、**共有メモリ側で転置して置く**（融合 attention の k 読みと同じ構造）
  let wc0 = tid / 4u;
  let wq = tid % 4u;
  let wcol0 = wid.x * 16u + wc0;
  let wsq0 = wc0 / 4u;
  let wsl0 = wc0 % 4u;
  let sb_base0 = (wq * 4u) * 4u + wsq0;
  // 列ごとの K 行頭と出どころ（K タイルループ不変なのでここで 1 度だけ畳む）。
  // 式は ① と同一 — col < P はスロット（物理行は読み書き同式の slot_row）・以降は ins の col − P
  let kcol0 = base_col + wcol0;
  let kpast0 = kcol0 < past;
  var krow_base0 = 0u;
  if (kpast0) {
    krow_base0 = (kv_plane * dims.capacity + slot_row(kcol0)) * dims.k;
  } else {
    krow_base0 = (kv_plane * dims.chunk_rows + (kcol0 - past)) * dims.k;
  }
  // ループ条件は uniform（dims は uniform バッファ）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすために必要
  let tiles = (dims.k + 15u) / 16u;
  var acc0_0 = vec4<f32>(0.0);
  for (var t = 0u; t < tiles; t = t + 1u) {
    // 範囲外は 0 で埋める。内積に寄与しないので端数 shape でも結果は変わらない
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
    // 半スケール契約（① と同じ）: scale は q 側と k 側の**両方**へ掛ける
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
    // 共有ロード 2 回（B の vec4 1 + A のスカラ 1）で 4 MAC。
    // 縮約は k 昇順の逐次で、1 出力要素あたりの加算順序は 16×16 の 1 スレッド 1 出力と
    // 完全に一致する。
    for (var kk = 0u; kk < 16u; kk = kk + 1u) {
      let bv0 = sb[kk * 4u + lid.x];
      acc0_0 = acc0_0 + sa[(lid.y * 1u + 0u) * 16u + kk] * bv0;
    }
    workgroupBarrier();
  }
  // 行は**有効行**まで（pad 行の S は誰も読まない — ③ が 0 を書いて返す）。列は live まで
  // （[live, col_cap) の残骸は ① と同じく触らないのが正）
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
