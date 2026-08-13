// karume conv1d (x[B,Cin,L] * W[Cout,Cin,K] + b[Cout], f32, 重み i8 格納, implicit GEMM レジスタ 32x128 タイル / 1 スレッド 8x8 / wg 16x4 + vec4)
struct Dims {
  m: u32,
  n: u32,
  k: u32,
  channels_in: u32,
  length_in: u32,
  kernel: u32,
  stride: u32,
  padding: u32,
  dilation: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<vec4<f32>>;

@group(0) @binding(5) var<storage, read> wscale: array<f32>;

// i8 格納の quad 展開: 要素 i..i+3 = f32(unpack4xI8(w[i / 4])) · scale（1 語で 4 要素）
// MUST: i は 4 の倍数（v4 経路からのみ呼ぶ）— 語をまたぐと 4 要素が別行に散る
fn dequant4(i: u32, scale: f32) -> vec4<f32> {
  return vec4<f32>(unpack4xI8(w[i >> 2u])) * scale;
}

// Xcol[k][n] = x[b, ic, n·stride − padding + k·dilation]（範囲外は 0）。
// kt = k·dilation − padding は K タイル内で不変なので呼び出し側が畳んで渡す。
// xc = b·Cin + ic（バッチは dispatch の z 軸なので呼び出し側で足す）
fn xcol(xc: u32, kt: i32, n: u32) -> f32 {
  let ix = i32(n * dims.stride) + kt;
  if (ix < 0 || u32(ix) >= dims.length_in) {
    return 0.0;
  }
  return x[xc * dims.length_in + u32(ix)];
}

// 共有 A タイル（32 行 × K 16・スカラ格納）と
// 共有 B タイル（K 16 × 列 quad 32・列方向を vec4 に束ねた形）
var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<vec4<f32>, 512>;

@compute @workgroup_size(16, 4)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 16u + lid.x;
  let n4 = dims.n / 4u;
  // A タイルの担当（32 行 × 4 quad を 64 スレッドで 2 巡）
  let ar = tid / 4u;
  let aq = tid % 4u;
  let arow0 = wid.y * 32u + ar;
  // 出力チャネルの scale はループ不変 — 重みの要素ごとに引き直さない（ADR 0019）
  let wscale_v = wscale[arow0];
  let arow_base0 = arow0 * dims.k;
  let sa_base0 = ar * 16u + aq * 4u;
  let arow1 = arow0 + 16u;
  // 出力チャネルの scale はループ不変 — 重みの要素ごとに引き直さない（ADR 0019）
  let wscale_v1 = wscale[arow1];
  let arow_base1 = arow_base0 + 16u * dims.k;
  let sa_base1 = sa_base0 + 256u;
  // バッチは workgroup 単位で一様（z 軸 1 つが 1 バッチの出力平面 [Cout, Lout]）
  let xbase = wid.z * dims.channels_in;
  let cbase = wid.z * dims.m * n4;
  // B タイルの担当（K 16 行 × 列 quad 32 を 64 スレッドで 8 巡）
  let bk0 = tid / 32u;
  let bcq = tid % 32u;
  let bc4 = wid.x * 32u + bcq;
  let bk1 = bk0 + 2u;
  let bk2 = bk0 + 4u;
  let bk3 = bk0 + 6u;
  let bk4 = bk0 + 8u;
  let bk5 = bk0 + 10u;
  let bk6 = bk0 + 12u;
  let bk7 = bk0 + 14u;
  // ループ条件は uniform（dims は uniform バッファ）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすために必要
  let tiles = (dims.k + 15u) / 16u;
  let bias0 = wid.y * 32u + lid.y * 8u;
  var acc0_0 = vec4<f32>(bias[bias0]);
  var acc0_1 = vec4<f32>(bias[bias0]);
  var acc1_0 = vec4<f32>(bias[bias0 + 1u]);
  var acc1_1 = vec4<f32>(bias[bias0 + 1u]);
  var acc2_0 = vec4<f32>(bias[bias0 + 2u]);
  var acc2_1 = vec4<f32>(bias[bias0 + 2u]);
  var acc3_0 = vec4<f32>(bias[bias0 + 3u]);
  var acc3_1 = vec4<f32>(bias[bias0 + 3u]);
  var acc4_0 = vec4<f32>(bias[bias0 + 4u]);
  var acc4_1 = vec4<f32>(bias[bias0 + 4u]);
  var acc5_0 = vec4<f32>(bias[bias0 + 5u]);
  var acc5_1 = vec4<f32>(bias[bias0 + 5u]);
  var acc6_0 = vec4<f32>(bias[bias0 + 6u]);
  var acc6_1 = vec4<f32>(bias[bias0 + 6u]);
  var acc7_0 = vec4<f32>(bias[bias0 + 7u]);
  var acc7_1 = vec4<f32>(bias[bias0 + 7u]);
  for (var t = 0u; t < tiles; t = t + 1u) {
    // 範囲外は 0 で埋める。内積に寄与しないので K 端数でも結果は変わらない
    let ak0 = t * 16u + aq * 4u;
    var av0 = vec4<f32>(0.0);
    if (arow0 < dims.m && ak0 < dims.k) {
      av0 = dequant4(arow_base0 + ak0, wscale_v);
    }
    sa[sa_base0] = av0.x;
    sa[sa_base0 + 1u] = av0.y;
    sa[sa_base0 + 2u] = av0.z;
    sa[sa_base0 + 3u] = av0.w;
    var av1 = vec4<f32>(0.0);
    if (arow1 < dims.m && ak0 < dims.k) {
      av1 = dequant4(arow_base1 + ak0, wscale_v1);
    }
    sa[sa_base1] = av1.x;
    sa[sa_base1 + 1u] = av1.y;
    sa[sa_base1 + 2u] = av1.z;
    sa[sa_base1 + 3u] = av1.w;
    let brow0 = t * 16u + bk0;
    var bv4_0 = vec4<f32>(0.0);
    if (brow0 < dims.k && bc4 < n4) {
      let ic = brow0 / dims.kernel;
      let kt = i32((brow0 % dims.kernel) * dims.dilation) - i32(dims.padding);
      let xc = xbase + ic;
      // quad の 4 列は連続する ox（v4 の条件 stride == 1）なので x 側も連続に読める
      let n0 = bc4 * 4u;
      let ix0 = i32(n0 * dims.stride) + kt;
      if (ix0 >= 0 && u32(ix0) + 3u < dims.length_in) {
        let base = xc * dims.length_in + u32(ix0);
        bv4_0 = vec4<f32>(x[base], x[base + 1u], x[base + 2u], x[base + 3u]);
      } else {
        // 系列端と padding 域だけがここに来る（範囲外は 0 — xcol の MUST）
        bv4_0 = vec4<f32>(
          xcol(xc, kt, n0),
          xcol(xc, kt, n0 + 1u),
          xcol(xc, kt, n0 + 2u),
          xcol(xc, kt, n0 + 3u),
        );
      }
    }
    sb[bk0 * 32u + bcq] = bv4_0;
    let brow1 = t * 16u + bk1;
    var bv4_1 = vec4<f32>(0.0);
    if (brow1 < dims.k && bc4 < n4) {
      let ic = brow1 / dims.kernel;
      let kt = i32((brow1 % dims.kernel) * dims.dilation) - i32(dims.padding);
      let xc = xbase + ic;
      // quad の 4 列は連続する ox（v4 の条件 stride == 1）なので x 側も連続に読める
      let n0 = bc4 * 4u;
      let ix0 = i32(n0 * dims.stride) + kt;
      if (ix0 >= 0 && u32(ix0) + 3u < dims.length_in) {
        let base = xc * dims.length_in + u32(ix0);
        bv4_1 = vec4<f32>(x[base], x[base + 1u], x[base + 2u], x[base + 3u]);
      } else {
        // 系列端と padding 域だけがここに来る（範囲外は 0 — xcol の MUST）
        bv4_1 = vec4<f32>(
          xcol(xc, kt, n0),
          xcol(xc, kt, n0 + 1u),
          xcol(xc, kt, n0 + 2u),
          xcol(xc, kt, n0 + 3u),
        );
      }
    }
    sb[bk1 * 32u + bcq] = bv4_1;
    let brow2 = t * 16u + bk2;
    var bv4_2 = vec4<f32>(0.0);
    if (brow2 < dims.k && bc4 < n4) {
      let ic = brow2 / dims.kernel;
      let kt = i32((brow2 % dims.kernel) * dims.dilation) - i32(dims.padding);
      let xc = xbase + ic;
      // quad の 4 列は連続する ox（v4 の条件 stride == 1）なので x 側も連続に読める
      let n0 = bc4 * 4u;
      let ix0 = i32(n0 * dims.stride) + kt;
      if (ix0 >= 0 && u32(ix0) + 3u < dims.length_in) {
        let base = xc * dims.length_in + u32(ix0);
        bv4_2 = vec4<f32>(x[base], x[base + 1u], x[base + 2u], x[base + 3u]);
      } else {
        // 系列端と padding 域だけがここに来る（範囲外は 0 — xcol の MUST）
        bv4_2 = vec4<f32>(
          xcol(xc, kt, n0),
          xcol(xc, kt, n0 + 1u),
          xcol(xc, kt, n0 + 2u),
          xcol(xc, kt, n0 + 3u),
        );
      }
    }
    sb[bk2 * 32u + bcq] = bv4_2;
    let brow3 = t * 16u + bk3;
    var bv4_3 = vec4<f32>(0.0);
    if (brow3 < dims.k && bc4 < n4) {
      let ic = brow3 / dims.kernel;
      let kt = i32((brow3 % dims.kernel) * dims.dilation) - i32(dims.padding);
      let xc = xbase + ic;
      // quad の 4 列は連続する ox（v4 の条件 stride == 1）なので x 側も連続に読める
      let n0 = bc4 * 4u;
      let ix0 = i32(n0 * dims.stride) + kt;
      if (ix0 >= 0 && u32(ix0) + 3u < dims.length_in) {
        let base = xc * dims.length_in + u32(ix0);
        bv4_3 = vec4<f32>(x[base], x[base + 1u], x[base + 2u], x[base + 3u]);
      } else {
        // 系列端と padding 域だけがここに来る（範囲外は 0 — xcol の MUST）
        bv4_3 = vec4<f32>(
          xcol(xc, kt, n0),
          xcol(xc, kt, n0 + 1u),
          xcol(xc, kt, n0 + 2u),
          xcol(xc, kt, n0 + 3u),
        );
      }
    }
    sb[bk3 * 32u + bcq] = bv4_3;
    let brow4 = t * 16u + bk4;
    var bv4_4 = vec4<f32>(0.0);
    if (brow4 < dims.k && bc4 < n4) {
      let ic = brow4 / dims.kernel;
      let kt = i32((brow4 % dims.kernel) * dims.dilation) - i32(dims.padding);
      let xc = xbase + ic;
      // quad の 4 列は連続する ox（v4 の条件 stride == 1）なので x 側も連続に読める
      let n0 = bc4 * 4u;
      let ix0 = i32(n0 * dims.stride) + kt;
      if (ix0 >= 0 && u32(ix0) + 3u < dims.length_in) {
        let base = xc * dims.length_in + u32(ix0);
        bv4_4 = vec4<f32>(x[base], x[base + 1u], x[base + 2u], x[base + 3u]);
      } else {
        // 系列端と padding 域だけがここに来る（範囲外は 0 — xcol の MUST）
        bv4_4 = vec4<f32>(
          xcol(xc, kt, n0),
          xcol(xc, kt, n0 + 1u),
          xcol(xc, kt, n0 + 2u),
          xcol(xc, kt, n0 + 3u),
        );
      }
    }
    sb[bk4 * 32u + bcq] = bv4_4;
    let brow5 = t * 16u + bk5;
    var bv4_5 = vec4<f32>(0.0);
    if (brow5 < dims.k && bc4 < n4) {
      let ic = brow5 / dims.kernel;
      let kt = i32((brow5 % dims.kernel) * dims.dilation) - i32(dims.padding);
      let xc = xbase + ic;
      // quad の 4 列は連続する ox（v4 の条件 stride == 1）なので x 側も連続に読める
      let n0 = bc4 * 4u;
      let ix0 = i32(n0 * dims.stride) + kt;
      if (ix0 >= 0 && u32(ix0) + 3u < dims.length_in) {
        let base = xc * dims.length_in + u32(ix0);
        bv4_5 = vec4<f32>(x[base], x[base + 1u], x[base + 2u], x[base + 3u]);
      } else {
        // 系列端と padding 域だけがここに来る（範囲外は 0 — xcol の MUST）
        bv4_5 = vec4<f32>(
          xcol(xc, kt, n0),
          xcol(xc, kt, n0 + 1u),
          xcol(xc, kt, n0 + 2u),
          xcol(xc, kt, n0 + 3u),
        );
      }
    }
    sb[bk5 * 32u + bcq] = bv4_5;
    let brow6 = t * 16u + bk6;
    var bv4_6 = vec4<f32>(0.0);
    if (brow6 < dims.k && bc4 < n4) {
      let ic = brow6 / dims.kernel;
      let kt = i32((brow6 % dims.kernel) * dims.dilation) - i32(dims.padding);
      let xc = xbase + ic;
      // quad の 4 列は連続する ox（v4 の条件 stride == 1）なので x 側も連続に読める
      let n0 = bc4 * 4u;
      let ix0 = i32(n0 * dims.stride) + kt;
      if (ix0 >= 0 && u32(ix0) + 3u < dims.length_in) {
        let base = xc * dims.length_in + u32(ix0);
        bv4_6 = vec4<f32>(x[base], x[base + 1u], x[base + 2u], x[base + 3u]);
      } else {
        // 系列端と padding 域だけがここに来る（範囲外は 0 — xcol の MUST）
        bv4_6 = vec4<f32>(
          xcol(xc, kt, n0),
          xcol(xc, kt, n0 + 1u),
          xcol(xc, kt, n0 + 2u),
          xcol(xc, kt, n0 + 3u),
        );
      }
    }
    sb[bk6 * 32u + bcq] = bv4_6;
    let brow7 = t * 16u + bk7;
    var bv4_7 = vec4<f32>(0.0);
    if (brow7 < dims.k && bc4 < n4) {
      let ic = brow7 / dims.kernel;
      let kt = i32((brow7 % dims.kernel) * dims.dilation) - i32(dims.padding);
      let xc = xbase + ic;
      // quad の 4 列は連続する ox（v4 の条件 stride == 1）なので x 側も連続に読める
      let n0 = bc4 * 4u;
      let ix0 = i32(n0 * dims.stride) + kt;
      if (ix0 >= 0 && u32(ix0) + 3u < dims.length_in) {
        let base = xc * dims.length_in + u32(ix0);
        bv4_7 = vec4<f32>(x[base], x[base + 1u], x[base + 2u], x[base + 3u]);
      } else {
        // 系列端と padding 域だけがここに来る（範囲外は 0 — xcol の MUST）
        bv4_7 = vec4<f32>(
          xcol(xc, kt, n0),
          xcol(xc, kt, n0 + 1u),
          xcol(xc, kt, n0 + 2u),
          xcol(xc, kt, n0 + 3u),
        );
      }
    }
    sb[bk7 * 32u + bcq] = bv4_7;
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
  let ocq0 = wid.x * 32u + lid.x * 2u;
  let ocq1 = ocq0 + 1u;
  let orow0 = wid.y * 32u + lid.y * 8u;
  let orow1 = orow0 + 1u;
  let orow2 = orow0 + 2u;
  let orow3 = orow0 + 3u;
  let orow4 = orow0 + 4u;
  let orow5 = orow0 + 5u;
  let orow6 = orow0 + 6u;
  let orow7 = orow0 + 7u;
  if (ocq0 < n4) {
    if (orow0 < dims.m) {
      out[cbase + orow0 * n4 + ocq0] = acc0_0;
    }
    if (orow1 < dims.m) {
      out[cbase + orow1 * n4 + ocq0] = acc1_0;
    }
    if (orow2 < dims.m) {
      out[cbase + orow2 * n4 + ocq0] = acc2_0;
    }
    if (orow3 < dims.m) {
      out[cbase + orow3 * n4 + ocq0] = acc3_0;
    }
    if (orow4 < dims.m) {
      out[cbase + orow4 * n4 + ocq0] = acc4_0;
    }
    if (orow5 < dims.m) {
      out[cbase + orow5 * n4 + ocq0] = acc5_0;
    }
    if (orow6 < dims.m) {
      out[cbase + orow6 * n4 + ocq0] = acc6_0;
    }
    if (orow7 < dims.m) {
      out[cbase + orow7 * n4 + ocq0] = acc7_0;
    }
  }
  if (ocq1 < n4) {
    if (orow0 < dims.m) {
      out[cbase + orow0 * n4 + ocq1] = acc0_1;
    }
    if (orow1 < dims.m) {
      out[cbase + orow1 * n4 + ocq1] = acc1_1;
    }
    if (orow2 < dims.m) {
      out[cbase + orow2 * n4 + ocq1] = acc2_1;
    }
    if (orow3 < dims.m) {
      out[cbase + orow3 * n4 + ocq1] = acc3_1;
    }
    if (orow4 < dims.m) {
      out[cbase + orow4 * n4 + ocq1] = acc4_1;
    }
    if (orow5 < dims.m) {
      out[cbase + orow5 * n4 + ocq1] = acc5_1;
    }
    if (orow6 < dims.m) {
      out[cbase + orow6 * n4 + ocq1] = acc6_1;
    }
    if (orow7 < dims.m) {
      out[cbase + orow7 * n4 + ocq1] = acc7_1;
    }
  }
}
