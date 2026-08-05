// karume attention_qk (S[b,m,n] = q[b,m,d] · k[b,n,d]ᵀ, q/k とも per-token i8 の整数内積・エミュ, 半スケールは dequant 側, レジスタ 64x64 タイル)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
  scale: f32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> qq: array<u32>;
@group(0) @binding(2) var<storage, read> kq: array<u32>;
@group(0) @binding(3) var<storage, read_write> s: array<f32>;
@group(0) @binding(4) var<storage, read> qscale: array<f32>;
@group(0) @binding(5) var<storage, read> kscale: array<f32>;

// 整数内積のエミュレーション（core WGSL のみ）。dot4I8Packed と**同じ整数**を返す —
// 整数加算は結合的かつ厳密で、中間値も最大 4·127² = 64,516 なので巻き戻らない
fn idot(a: u32, b: u32) -> i32 {
  return dot(unpack4xI8(a), unpack4xI8(b));
}

// 共有タイルは **[pack][行] / [pack][列]** 配置（linear の i8a8 と同じ — 内側の読みが
// 語ストライド 4 になりバンク衝突が 8-way から 2-way に減る）
var<workgroup> sa: array<u32, 256>;
var<workgroup> sb: array<u32, 256>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 16u + lid.x;
  // 適格条件で D % 4 == 0 なので、行頭は必ず語境界に来る（i8 ペイロードは平坦添字 4 詰め）
  let k4 = dims.k / 4u;
  // B と H を畳んだバッチ軸（workgroup 単位で一様 — 内側の workgroupBarrier が WGSL の
  // 一様性要件を満たすための前提）。MUST: q / k のペイロードは **pack 単位**、scale は行数
  // 単位、S は要素単位で数える — 単位を取り違えると B·H ≥ 2 で
  // 隣の head を読み書きする（例外なしの誤値で、B=H=1 のテストには出ない）
  let qbase = wid.z * dims.m * k4;
  let kbase = wid.z * dims.n * k4;
  let qsbase = wid.z * dims.m;
  let ksbase = wid.z * dims.n;
  let sbase = wid.z * dims.m * dims.n;
  // q タイルの担当（64 行 × 4 pack = 256 スレッド）
  let ar = tid / 4u;
  let ap = tid % 4u;
  let arow = wid.y * 64u + ar;
  let arow_base = qbase + arow * k4;
  // k タイルの担当（64 列（N）× 4 pack = 256 スレッド）。
  // k は [N,D] のまま読む（連続方向が D = パック方向なので**転置が要らない**）
  let wc = tid / 4u;
  let wp = tid % 4u;
  let wcol = wid.x * 64u + wc;
  let krow_base = kbase + wcol * k4;
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
      av = qq[arow_base + apack];
    }
    sa[ap * 64u + ar] = av;
    let kpack = t * 4u + wp;
    var kv = 0u;
    if (wcol < dims.n && kpack < k4) {
      kv = kq[krow_base + kpack];
    }
    sb[wp * 64u + wc] = kv;
    workgroupBarrier();
    // 共有ロード 5 回（k の 4 語 + q の 1 語）で 16 個の整数内積 = 64 MAC。
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
      let obase = sbase + orow * dims.n;
      // MUST: qs'·ks' を先に 1 つの f32 へ畳んでから f32(acc) に掛ける（bias が無いので fma は無い）
      let qs = qscale[qsbase + orow] * dims.scale;
      if (ocol < dims.n) {
        s[obase + ocol] = f32(acc[i].x) * (qs * (kscale[ksbase + ocol] * dims.scale));
      }
      if (ocol + 1u < dims.n) {
        s[obase + ocol + 1u] = f32(acc[i].y) * (qs * (kscale[ksbase + ocol + 1u] * dims.scale));
      }
      if (ocol + 2u < dims.n) {
        s[obase + ocol + 2u] = f32(acc[i].z) * (qs * (kscale[ksbase + ocol + 2u] * dims.scale));
      }
      if (ocol + 3u < dims.n) {
        s[obase + ocol + 3u] = f32(acc[i].w) * (qs * (kscale[ksbase + ocol + 3u] * dims.scale));
      }
    }
  }
}
