// karume linear (x[m,k] · wᵀ[k,n] + bias[n], 活性 per-token i8 × 重み i8 の整数内積・エミュ, レジスタ 64x64 タイル)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> xq: array<u32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read> wscale: array<f32>;
@group(0) @binding(6) var<storage, read> xscale: array<f32>;

// 整数内積のエミュレーション（core WGSL のみ）。dot4I8Packed と**同じ整数**を返す —
// 整数加算は結合的かつ厳密で、中間値も最大 4·127² = 64,516 なので巻き戻らない
fn idot(a: u32, b: u32) -> i32 {
  return dot(unpack4xI8(a), unpack4xI8(b));
}

// 共有タイルは **[pack][行] / [pack][列]** 配置（プロトタイプの [行][pack] からの組み替え —
// 内側の読みが語ストライド 4 になりバンク衝突が 8-way から 2-way に減る）
var<workgroup> sa: array<u32, 256>;
var<workgroup> sb: array<u32, 256>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 16u + lid.x;
  // 適格条件で k % 4 == 0 なので、行頭は必ず語境界に来る（i8 ペイロードは平坦添字 4 詰め）
  let k4 = dims.k / 4u;
  // x タイルの担当（64 行 × 4 pack = 256 スレッド）
  let ar = tid / 4u;
  let ap = tid % 4u;
  let arow = wid.y * 64u + ar;
  let arow_base = arow * k4;
  // W タイルの担当（64 出力チャネル × 4 pack = 256 スレッド）。
  // 重みは [n,k] のまま読む（連続方向が k = パック方向なので**転置が要らない**）
  let wc = tid / 4u;
  let wp = tid % 4u;
  let wcol = wid.x * 64u + wc;
  let wrow_base = wcol * k4;
  // ループ条件は uniform（dims は uniform バッファ）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすために必要
  let tiles = (k4 + 3u) / 4u;
  var acc = array<vec4<i32>, 4>(
    vec4<i32>(0),
    vec4<i32>(0),
    vec4<i32>(0),
    vec4<i32>(0),
  );
  for (var t = 0u; t < tiles; t = t + 1u) {
    // 範囲外は 0 で埋める（dot4I8Packed(0, x) == 0 なので K 端数でも結果は厳密）
    let apack = t * 4u + ap;
    var av = 0u;
    if (arow < dims.m && apack < k4) {
      av = xq[arow_base + apack];
    }
    sa[ap * 64u + ar] = av;
    let wpack = t * 4u + wp;
    var wv = 0u;
    if (wcol < dims.n && wpack < k4) {
      wv = w[wrow_base + wpack];
    }
    sb[wp * 64u + wc] = wv;
    workgroupBarrier();
    // 共有ロード 5 回（W の 4 語 + x の 1 語）で 16 個の整数内積 = 64 MAC。
    // 縮約は i32 の厳密加算なので**順序に依存しない**（f32 骨格と違い加算順は数値契約に無い）
    for (var p = 0u; p < 4u; p = p + 1u) {
      let bcol = p * 64u + lid.x * 4u;
      let b0 = sb[bcol];
      let b1 = sb[bcol + 1u];
      let b2 = sb[bcol + 2u];
      let b3 = sb[bcol + 3u];
      for (var i = 0u; i < 4u; i = i + 1u) {
        let a = sa[p * 64u + lid.y * 4u + i];
        acc[i] = acc[i] + vec4<i32>(idot(a, b0), idot(a, b1), idot(a, b2), idot(a, b3));
      }
    }
    workgroupBarrier();
  }
  let ocol = wid.x * 64u + lid.x * 4u;
  let orow0 = wid.y * 64u + lid.y * 4u;
  for (var i = 0u; i < 4u; i = i + 1u) {
    let orow = orow0 + i;
    if (orow < dims.m) {
      let obase = orow * dims.n;
      // MUST: xs·wscale を先に 1 つの f32 へ畳み、積和は fma（単一丸め）
      let xs = xscale[orow];
      if (ocol < dims.n) {
        out[obase + ocol] = fma(f32(acc[i].x), xs * wscale[ocol], bias[ocol]);
      }
      if (ocol + 1u < dims.n) {
        out[obase + ocol + 1u] = fma(f32(acc[i].y), xs * wscale[ocol + 1u], bias[ocol + 1u]);
      }
      if (ocol + 2u < dims.n) {
        out[obase + ocol + 2u] = fma(f32(acc[i].z), xs * wscale[ocol + 2u], bias[ocol + 2u]);
      }
      if (ocol + 3u < dims.n) {
        out[obase + ocol + 3u] = fma(f32(acc[i].w), xs * wscale[ocol + 3u], bias[ocol + 3u]);
      }
    }
  }
}
