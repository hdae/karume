// karume matmul (a[m,k] · b[k,n], f32, レジスタ 64x64 タイル / 1 スレッド 8x4 / wg 16x8 + vec4)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> b: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> c: array<vec4<f32>>;

// 共有 A タイル（64 行 × K 16・スカラ格納）と
// 共有 B タイル（K 16 × 列 quad 16・列方向を vec4 に束ねた形）
var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 8)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 16u + lid.x;
  let k4 = dims.k / 4u;
  let n4 = dims.n / 4u;
  // A タイルの担当（64 行 × 4 quad を 128 スレッドで 2 巡）
  let ar = tid / 4u;
  let aq = tid % 4u;
  let arow0 = wid.y * 64u + ar;
  let arow_base0 = arow0 * k4;
  let sa_base0 = ar * 16u + aq * 4u;
  let arow1 = arow0 + 32u;
  let arow_base1 = arow_base0 + 32u * k4;
  let sa_base1 = sa_base0 + 512u;
  // B タイルの担当（K 16 行 × 列 quad 16 を 128 スレッドで 2 巡）
  let bk0 = tid / 16u;
  let bcq = tid % 16u;
  let bc4 = wid.x * 16u + bcq;
  let bk1 = bk0 + 8u;
  // ループ条件は uniform（dims は uniform バッファ）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすために必要
  let tiles = (dims.k + 15u) / 16u;
  var acc0_0 = vec4<f32>(0.0);
  var acc1_0 = vec4<f32>(0.0);
  var acc2_0 = vec4<f32>(0.0);
  var acc3_0 = vec4<f32>(0.0);
  var acc4_0 = vec4<f32>(0.0);
  var acc5_0 = vec4<f32>(0.0);
  var acc6_0 = vec4<f32>(0.0);
  var acc7_0 = vec4<f32>(0.0);
  for (var t = 0u; t < tiles; t = t + 1u) {
    // 範囲外は 0 で埋める。内積に寄与しないので端数 shape でも結果は変わらない
    let ak0 = t * 16u + aq * 4u;
    var av0 = vec4<f32>(0.0);
    if (arow0 < dims.m && ak0 < dims.k) {
      av0 = a[arow_base0 + t * 4u + aq];
    }
    sa[sa_base0] = av0.x;
    sa[sa_base0 + 1u] = av0.y;
    sa[sa_base0 + 2u] = av0.z;
    sa[sa_base0 + 3u] = av0.w;
    var av1 = vec4<f32>(0.0);
    if (arow1 < dims.m && ak0 < dims.k) {
      av1 = a[arow_base1 + t * 4u + aq];
    }
    sa[sa_base1] = av1.x;
    sa[sa_base1 + 1u] = av1.y;
    sa[sa_base1 + 2u] = av1.z;
    sa[sa_base1 + 3u] = av1.w;
    let brow0 = t * 16u + bk0;
    var bv4_0 = vec4<f32>(0.0);
    if (brow0 < dims.k && bc4 < n4) {
      bv4_0 = b[brow0 * n4 + bc4];
    }
    sb[bk0 * 16u + bcq] = bv4_0;
    let brow1 = t * 16u + bk1;
    var bv4_1 = vec4<f32>(0.0);
    if (brow1 < dims.k && bc4 < n4) {
      bv4_1 = b[brow1 * n4 + bc4];
    }
    sb[bk1 * 16u + bcq] = bv4_1;
    workgroupBarrier();
    // 共有ロード 9 回（B の vec4 1 + A のスカラ 8）で 32 MAC。
    // 縮約は k 昇順の逐次で、1 出力要素あたりの加算順序は 16×16 の 1 スレッド 1 出力と
    // 完全に一致する。
    for (var kk = 0u; kk < 16u; kk = kk + 1u) {
      let bv0 = sb[kk * 16u + lid.x];
      acc0_0 = acc0_0 + sa[(lid.y * 8u + 0u) * 16u + kk] * bv0;
      acc1_0 = acc1_0 + sa[(lid.y * 8u + 1u) * 16u + kk] * bv0;
      acc2_0 = acc2_0 + sa[(lid.y * 8u + 2u) * 16u + kk] * bv0;
      acc3_0 = acc3_0 + sa[(lid.y * 8u + 3u) * 16u + kk] * bv0;
      acc4_0 = acc4_0 + sa[(lid.y * 8u + 4u) * 16u + kk] * bv0;
      acc5_0 = acc5_0 + sa[(lid.y * 8u + 5u) * 16u + kk] * bv0;
      acc6_0 = acc6_0 + sa[(lid.y * 8u + 6u) * 16u + kk] * bv0;
      acc7_0 = acc7_0 + sa[(lid.y * 8u + 7u) * 16u + kk] * bv0;
    }
    workgroupBarrier();
  }
  let ocq0 = wid.x * 16u + lid.x;
  let orow0 = wid.y * 64u + lid.y * 8u;
  let orow1 = orow0 + 1u;
  let orow2 = orow0 + 2u;
  let orow3 = orow0 + 3u;
  let orow4 = orow0 + 4u;
  let orow5 = orow0 + 5u;
  let orow6 = orow0 + 6u;
  let orow7 = orow0 + 7u;
  if (ocq0 < n4) {
    if (orow0 < dims.m) {
      c[orow0 * n4 + ocq0] = acc0_0;
    }
    if (orow1 < dims.m) {
      c[orow1 * n4 + ocq0] = acc1_0;
    }
    if (orow2 < dims.m) {
      c[orow2 * n4 + ocq0] = acc2_0;
    }
    if (orow3 < dims.m) {
      c[orow3 * n4 + ocq0] = acc3_0;
    }
    if (orow4 < dims.m) {
      c[orow4 * n4 + ocq0] = acc4_0;
    }
    if (orow5 < dims.m) {
      c[orow5 * n4 + ocq0] = acc5_0;
    }
    if (orow6 < dims.m) {
      c[orow6 * n4 + ocq0] = acc6_0;
    }
    if (orow7 < dims.m) {
      c[orow7 * n4 + ocq0] = acc7_0;
    }
  }
}
