// karume attention_pv (O[b,m,d] = P[b,m,n] · v[b,n,d], P̃ = round(127·exp(S − m)) は非実体化, v は per-column i8（Vᵀ 連続）, S は f16 格納（pack2x16float）, レジスタ 64x64 タイル + vec4)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
// 適格条件で N % 4 == 0 なので、S の行頭は常に quad 境界に来る（P̃ の 4 詰めと同じ刻み）
@group(0) @binding(1) var<storage, read> s: array<u32>;
@group(0) @binding(2) var<storage, read> vq: array<u32>;
@group(0) @binding(3) var<storage, read> stats: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> vscale: array<f32>;

// f16 格納の quad 読み: quad q = s[2q] と s[2q + 1] の 2 語（unpack は厳密）
fn score_quad(q: u32) -> vec4<f32> {
  let w = q * 2u;
  let lo = unpack2x16float(s[w]);
  let hi = unpack2x16float(s[w + 1u]);
  return vec4<f32>(lo.x, lo.y, hi.x, hi.y);
}

// 整数内積（WGSL 言語拡張 packed_4x8_integer_dot_product）。エミュ変種と**同じ整数**を返す
fn idot(a: u32, b: u32) -> i32 {
  return dot4I8Packed(a, b);
}

// 共有タイルは **[pack][行] / [pack][列]** 配置（①QK / linear の i8a8 と同じ — 内側の読みが
// 語ストライド 4 になりバンク衝突が 8-way から 2-way に減る）
var<workgroup> sa: array<u32, 256>;
var<workgroup> sb: array<u32, 256>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 16u + lid.x;
  // 縮約軸 N は S 側では quad 単位・vq 側では pack 単位（同じ 4 要素の刻み）
  let k4 = dims.k / 4u;
  let n4 = dims.n / 4u;
  // B と H を畳んだバッチ軸（workgroup 単位で一様 — 内側の workgroupBarrier が WGSL の
  // 一様性要件を満たすための前提）。MUST: S は quad 単位、vq は pack 単位、行統計は行あたり
  // 2 語、vscale は D 本、O は要素（v4 では quad）単位で数える — 単位を
  // 取り違えると B·H ≥ 2 で隣の head を読み書きする（例外なしの誤値で B=H=1 では出ない）
  let sbase = wid.z * dims.m * k4;
  let vbase = wid.z * dims.n * k4;
  let rbase = wid.z * dims.m;
  let vsbase = wid.z * dims.n;
  let obase = wid.z * dims.m * n4;
  // P̃ タイルの担当（64 行 × 4 pack = 256 スレッド）
  let ar = tid / 4u;
  let ap = tid % 4u;
  let arow = wid.y * 64u + ar;
  let arow_base = sbase + arow * k4;
  // 行の最大 m（K タイルループ不変なので 1 度だけ引く）。端タイルでは arow >= M がありうるので
  // 添字を 0 へ倒す（読んだ値は arow < M の枝でしか使われない）
  let stat_at = select(0u, (rbase + arow) * 2u, arow < dims.m);
  let row_max = stats[stat_at];
  // Vᵀ タイルの担当（64 列（D）× 4 pack = 256 スレッド）。
  // vq は [D,N] の N 連続（= パック方向）なので**転置が要らない**
  let wc = tid / 4u;
  let wp = tid % 4u;
  let wcol = wid.x * 64u + wc;
  let vrow_base = vbase + wcol * k4;
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
    // 範囲外は 0 で埋める（qP = 0 は内積に寄与しないので K 端数でも結果は厳密）
    let apack = t * 4u + ap;
    var av = 0u;
    if (arow < dims.m && apack < k4) {
      let raw = score_quad(arow_base + apack);
      // P̃ の scale は **1/127 固定**（行内 max が exp(0) = 1 で構造的 — amax は取らない）。
      // 除算が 1 つも無いので WGSL の 2.5 ULP 問題の外側にいる
      let p = vec4<f32>(
        round(exp(raw.x - row_max) * 127.0),
        round(exp(raw.y - row_max) * 127.0),
        round(exp(raw.z - row_max) * 127.0),
        round(exp(raw.w - row_max) * 127.0),
      );
      av = pack4xI8(vec4<i32>(p));
    }
    sa[ap * 64u + ar] = av;
    let vpack = t * 4u + wp;
    var vv = 0u;
    if (wcol < dims.n && vpack < k4) {
      vv = vq[vrow_base + vpack];
    }
    sb[wp * 64u + wc] = vv;
    workgroupBarrier();
    // 共有ロード 5 回（v の 4 語 + P̃ の 1 語）で 16 個の整数内積 = 64 MAC。
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
  let ocq = wid.x * 16u + lid.x;
  let orow0 = wid.y * 64u + lid.y * 4u;
  if (ocq < n4) {
    // D % 4 == 0 かつ ocq < n4 なので oc + 3 < D。
    // MUST: 列 scale は Vᵀ の**行** = (b, h, d)（= V の per-column scale）。行 scale の側と
    // 取り違えても例外は出ない
    let oc = ocq * 4u;
    let vs = vec4<f32>(
      vscale[vsbase + oc],
      vscale[vsbase + oc + 1u],
      vscale[vsbase + oc + 2u],
      vscale[vsbase + oc + 3u],
    );
    for (var i = 0u; i < 4u; i = i + 1u) {
      let orow = orow0 + i;
      if (orow < dims.m) {
        // MUST: prow·vs を先に 1 つの f32 へ畳んでから f32(acc) に掛ける（bias が無いので fma は無い）
        let prow = stats[(rbase + orow) * 2u + 1u] * 0.007874015748031496;
        o[obase + orow * n4 + ocq] = vec4<f32>(acc[i]) * (prow * vs);
      }
    }
  }
}
