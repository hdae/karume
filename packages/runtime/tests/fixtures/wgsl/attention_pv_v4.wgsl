// karume attention_pv (O[b,m,d] = P[b,m,n] · v[b,n,d], P = exp(S − m)·inv は非実体化, f32, レジスタ 64x64 タイル + vec4)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> s: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> v: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> stats: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<vec4<f32>>;

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
  let k4 = dims.k / 4u;
  let n4 = dims.n / 4u;
  // バッチは workgroup 単位で一様（z 軸 1 つが 1 バッチのタイル）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすための前提。MUST: base は quad 単位（要素単位で組むと隣のバッチを読む）
  let abase = wid.z * dims.m * k4;
  let bbase = wid.z * dims.k * n4;
  let cbase = wid.z * dims.m * n4;
  let rbase = wid.z * dims.m;
  // A タイルの担当（64 行 × 4 quad = 256 スレッド）
  let ar = tid / 4u;
  let aq = tid % 4u;
  let arow = wid.y * 64u + ar;
  let arow_base = abase + arow * k4;
  let sa_base = ar * 16u + aq * 4u;
  // 端タイルでは arow >= m がありうるので添字を 0 へ倒す（読んだ値は arow < m の枝でしか
  // 使われない）。範囲外の stats を読むこと自体は WGSL の境界付きアクセスで安全
  let stat_at = select(0u, (rbase + arow) * 2u, arow < dims.m);
  let row_max = stats[stat_at];
  let row_inv = stats[stat_at + 1u];
  // B タイルの担当（K 16 行 × 列 quad 16 = 256 スレッド）
  let bk = tid / 16u;
  let bcq = tid % 16u;
  let bc4 = wid.x * 16u + bcq;
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
    if (arow < dims.m && ak0 < dims.k) {
      let raw = s[arow_base + t * 4u + aq];
      av = vec4<f32>(
        exp(raw.x - row_max) * row_inv,
        exp(raw.y - row_max) * row_inv,
        exp(raw.z - row_max) * row_inv,
        exp(raw.w - row_max) * row_inv,
      );
    }
    sa[sa_base] = av.x;
    sa[sa_base + 1u] = av.y;
    sa[sa_base + 2u] = av.z;
    sa[sa_base + 3u] = av.w;
    let brow = t * 16u + bk;
    var bv4 = vec4<f32>(0.0);
    if (brow < dims.k && bc4 < n4) {
      bv4 = v[bbase + brow * n4 + bc4];
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
  let ocq = wid.x * 16u + lid.x;
  let orow0 = wid.y * 64u + lid.y * 4u;
  if (ocq < n4) {
    for (var i = 0u; i < 4u; i = i + 1u) {
      let orow = orow0 + i;
      if (orow < dims.m) {
        o[cbase + orow * n4 + ocq] = acc[i];
      }
    }
  }
}
