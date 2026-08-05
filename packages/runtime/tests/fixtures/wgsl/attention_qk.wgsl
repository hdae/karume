// karume attention_qk (S[b,m,n] = (q·scale)[b,m,d] · (k·scale)[b,n,d]ᵀ, f32, レジスタ 64x64 タイル)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
  scale: f32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read_write> s: array<f32>;

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
  let bbase = wid.z * dims.n * dims.k;
  let cbase = wid.z * dims.m * dims.n;
  // A タイルの担当（64 行 × 4 quad = 256 スレッド）
  let ar = tid / 4u;
  let aq = tid % 4u;
  let arow = wid.y * 64u + ar;
  let arow_base = abase + arow * dims.k;
  let sa_base = ar * 16u + aq * 4u;
  // k タイルの担当（64 列（N）× 4 quad = 256 スレッド）。
  // k は [N,D] のまま読み、**共有メモリ側で転置して置く**（linear の重み読みと同じ構造）
  let wc = tid / 4u;
  let wq = tid % 4u;
  let wcol = wid.x * 64u + wc;
  let krow_base = bbase + wcol * dims.k;
  let wsq = wc / 4u;
  let wsl = wc % 4u;
  let sb_base = (wq * 4u) * 16u + wsq;
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
        av.x = q[arow_base + ak0] * dims.scale;
      }
      if (ak0 + 1u < dims.k) {
        av.y = q[arow_base + ak0 + 1u] * dims.scale;
      }
      if (ak0 + 2u < dims.k) {
        av.z = q[arow_base + ak0 + 2u] * dims.scale;
      }
      if (ak0 + 3u < dims.k) {
        av.w = q[arow_base + ak0 + 3u] * dims.scale;
      }
    }
    sa[sa_base] = av.x;
    sa[sa_base + 1u] = av.y;
    sa[sa_base + 2u] = av.z;
    sa[sa_base + 3u] = av.w;
    let wk0 = t * 16u + wq * 4u;
    var wv = vec4<f32>(0.0);
    if (wcol < dims.n) {
      if (wk0 < dims.k) {
        wv.x = k[krow_base + wk0];
      }
      if (wk0 + 1u < dims.k) {
        wv.y = k[krow_base + wk0 + 1u];
      }
      if (wk0 + 2u < dims.k) {
        wv.z = k[krow_base + wk0 + 2u];
      }
      if (wk0 + 3u < dims.k) {
        wv.w = k[krow_base + wk0 + 3u];
      }
    }
    // 半スケール契約（ADR 0023）: scale は q 側と k 側の**両方**へ掛ける。範囲外は 0 のままで
    // 0 · scale = 0 なので端数タイルの結論は変わらない
    wv = wv * dims.scale;
    sb[sb_base][wsl] = wv.x;
    sb[sb_base + 16u][wsl] = wv.y;
    sb[sb_base + 32u][wsl] = wv.z;
    sb[sb_base + 48u][wsl] = wv.w;
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
        s[obase + ocol] = acc[i].x;
      }
      if (ocol + 1u < dims.n) {
        s[obase + ocol + 1u] = acc[i].y;
      }
      if (ocol + 2u < dims.n) {
        s[obase + ocol + 2u] = acc[i].z;
      }
      if (ocol + 3u < dims.n) {
        s[obase + ocol + 3u] = acc[i].w;
      }
    }
  }
}
