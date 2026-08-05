// karume bmm (a[b,m,k] · b[b,k,n], f32, レジスタ 64x64 タイル)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

// 共有 A タイル（64 行 × K 16・スカラ格納）と
// 共有 B タイル（K 16 × 列 quad 16・列方向を vec4 に束ねた形）
var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 16u + lid.x;
  // バッチは workgroup 単位で一様（z 軸 1 つが 1 バッチのタイル）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすための前提
  let abase = wid.z * dims.m * dims.k;
  let bbase = wid.z * dims.k * dims.n;
  let cbase = wid.z * dims.m * dims.n;
  // A タイルの担当（64 行 × 4 quad = 256 スレッド）
  let ar = tid / 4u;
  let aq = tid % 4u;
  let arow = wid.y * 64u + ar;
  let arow_base = abase + arow * dims.k;
  let sa_base = ar * 16u + aq * 4u;
  // B タイルの担当（K 16 行 × 列 quad 16 = 256 スレッド）
  let bk = tid / 16u;
  let bcq = tid % 16u;
  let bcol = wid.x * 64u + bcq * 4u;
  // ループ条件は uniform（dims は uniform バッファ）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすために必要
  let tiles = (dims.k + 15u) / 16u;
  var acc = array<vec4<f32>, 4>(
    vec4<f32>(0.0),
    vec4<f32>(0.0),
    vec4<f32>(0.0),
    vec4<f32>(0.0),
  );
  for (var t = 0u; t < tiles; t = t + 1u) {
    // 範囲外は 0 で埋める。内積に寄与しないので端数 shape でも結果は変わらない
    let ak0 = t * 16u + aq * 4u;
    var av = vec4<f32>(0.0);
    if (arow < dims.m) {
      if (ak0 < dims.k) {
        av.x = a[arow_base + ak0];
      }
      if (ak0 + 1u < dims.k) {
        av.y = a[arow_base + ak0 + 1u];
      }
      if (ak0 + 2u < dims.k) {
        av.z = a[arow_base + ak0 + 2u];
      }
      if (ak0 + 3u < dims.k) {
        av.w = a[arow_base + ak0 + 3u];
      }
    }
    sa[sa_base] = av.x;
    sa[sa_base + 1u] = av.y;
    sa[sa_base + 2u] = av.z;
    sa[sa_base + 3u] = av.w;
    let brow = t * 16u + bk;
    var bv4 = vec4<f32>(0.0);
    if (brow < dims.k) {
      let brow_base = bbase + brow * dims.n;
      if (bcol < dims.n) {
        bv4.x = b[brow_base + bcol];
      }
      if (bcol + 1u < dims.n) {
        bv4.y = b[brow_base + bcol + 1u];
      }
      if (bcol + 2u < dims.n) {
        bv4.z = b[brow_base + bcol + 2u];
      }
      if (bcol + 3u < dims.n) {
        bv4.w = b[brow_base + bcol + 3u];
      }
    }
    sb[bk * 16u + bcq] = bv4;
    workgroupBarrier();
    // 共有ロード 5 回（B の vec4 1 + A のスカラ 4）で 16 MAC。縮約は k 昇順の逐次で、
    // 1 出力要素あたりの加算順序は 16×16 の 1 スレッド 1 出力と完全に一致する。
    for (var kk = 0u; kk < 16u; kk = kk + 1u) {
      let bv = sb[kk * 16u + lid.x];
      for (var i = 0u; i < 4u; i = i + 1u) {
        acc[i] = acc[i] + sa[(lid.y * 4u + i) * 16u + kk] * bv;
      }
    }
    workgroupBarrier();
  }
  let ocol = wid.x * 64u + lid.x * 4u;
  let orow0 = wid.y * 64u + lid.y * 4u;
  for (var i = 0u; i < 4u; i = i + 1u) {
    let orow = orow0 + i;
    if (orow < dims.m) {
      let obase = cbase + orow * dims.n;
      if (ocol < dims.n) {
        c[obase + ocol] = acc[i].x;
      }
      if (ocol + 1u < dims.n) {
        c[obase + ocol + 1u] = acc[i].y;
      }
      if (ocol + 2u < dims.n) {
        c[obase + ocol + 2u] = acc[i].z;
      }
      if (ocol + 3u < dims.n) {
        c[obase + ocol + 3u] = acc[i].w;
      }
    }
  }
}
