/**
 * DCNv2（modulated deformable convolution — op `deform_conv2d`・ADR 0055）の f32 固定カーネル。
 *
 * `out[b, oc, oy, ox] = Σ_{ic,kh,kw} (mask · bilinear(x[b,ic], y, x)) · w[oc,ic,kh,kw] + bias[oc]`、
 * `y = (oy − ph) + kh + off_y` / `x = (ox − pw) + kw + off_x`。stride / dilation / groups /
 * offset_groups は契約に**欄が無い = 1 固定**なので params にも載せない。
 *
 * 1 invocation = **出力 1 要素**の grid-stride で、出力の全要素はちょうど 1 回書かれる
 * （full-write 不変条件 — ADR 0014）。書き込みが排他所有・縮約順が静的なので、offset が
 * 実行時値でも「同一入力 → 同一出力」は保たれる（ADR 0055 決定 3）。
 *
 * MUST: 縮約は `(ic, kh, kw)` 昇順で、**conv2d 直接カーネルと同じ入れ子**（src/kernels/conv2d.ts）。
 * `offset` 全 0・`mask` 全 1 の退化ケースが conv2d とビット一致することが本 op の唯一の
 * A/B オラクル（tests/gpu_deform_conv2d_test.ts）で、ループを組み替えると縮約順が変わって
 * その門が壊れる。**`offset` / `mask` の読みを `ic` ループの外へ巻き上げない**（`offset_groups`
 * が 1 なので意味論的には可能だが、そのためにループを `(kh,kw)` 外・`ic` 内へ組み替えることに
 * なる）。アドレスが `ic` 非依存なのでシェーダコンパイラ側の巻き上げは順序を変えず無害。
 * MUST: `offset` のチャネル並びは **偶数 = y / 奇数 = x**（torchvision 逐語）。入れ替えても
 * 正方カーネル・対称 padding では値が一致しうるので、テストは Kh ≠ Kw の非対称形で固定する。
 * MUST: `mask` は**双線形補間の後・重みの前**に掛ける（`(m · v) · w`）。括り方まで torchvision の
 * 「im2col に `mask · bilinear` を書き、GEMM が `w × col` を取る」形に合わせる。
 * MUST: 境界外は **border clamp ではなくゼロ埋め**の 2 段（中心が `(−1, in)` の外ならタップ
 * 全体 0・内側でも範囲外の隅はその隅だけ 0）。ここは gather / embedding の「範囲外 = NaN 汚染」
 * （docs/limitations.md）とは**別の規約**で、deform の境界外サンプルは正常な意味論。
 * MUST: 範囲判定は**正の形**（`> −1 && < in`）で書き、NaN はその前にビット列で判定して
 * 出力へ伝播させる。負の形（`<= −1 || in <= `）だと NaN が「範囲内」に落ちて
 * `i32(floor(NaN))` の未定義（docs/limitations.md）へ直行し、正の形だけだと NaN が黙って
 * 0 寄与になる（沈黙誤値）。`clamp` に流す案は ADR 0020 が根治した「ドライバの max が
 * NaN を飲む」の再演なので採らない。
 * MUST: grid-stride を使う。実モデル（BiRefNet_HR の `decoder_block1`）の出力は
 * `[1,256,512,512]` = 6,710 万要素で、必要な workgroup 数が 1 次元 dispatch 上限
 * （既定 65535）を 3 桁超える。
 */

import { CodegenError } from "../codegen/errors.ts";

export const DEFORM_CONV2D_WORKGROUP_SIZE = 256;

/** MUST: WGSL を変えたらキーも上げる（パイプラインキャッシュは本文を見ない）。 */
export const DEFORM_CONV2D_KEY =
  `deform_conv2d:v1:nchw:f32:dcnv2:wg${DEFORM_CONV2D_WORKGROUP_SIZE}`;

export const DEFORM_CONV2D_WGSL: string =
  `// karume deform_conv2d (DCNv2: x[B,Cin,H,W] * W[Cout,Cin,Kh,Kw] + b[Cout], f32, 直接畳み込み)
struct Dims {
  n: u32,
  batch: u32,
  channels_in: u32,
  channels_out: u32,
  height_in: u32,
  width_in: u32,
  height_out: u32,
  width_out: u32,
  kernel_h: u32,
  kernel_w: u32,
  padding_h: u32,
  padding_w: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> offsets: array<f32>;
@group(0) @binding(4) var<storage, read> modulator: array<f32>;
@group(0) @binding(5) var<storage, read> bias: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;

// NaN のビット列判定（ADR 0020）— 指数部が全 1 かつ仮数部が非 0。比較演算に寄せると
// ドライバの畳み込みで判定ごと消える。
fn is_nan(v: f32) -> bool {
  return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u;
}

// 入力平面 \`plane\` の双線形サンプル。範囲外はゼロ埋め（4 隅個別）・NaN は伝播。
fn deform_sample(plane: u32, sy: f32, sx: f32) -> f32 {
  if (is_nan(sy) || is_nan(sx)) {
    return bitcast<f32>(0x7fc00000u);
  }
  // 正の形の範囲判定。これを通れば floor(sy) は [-1, H-1] なので i32 変換は必ず定義される。
  let inside = sy > -1.0 && sy < f32(dims.height_in)
    && sx > -1.0 && sx < f32(dims.width_in);
  if (!inside) {
    return 0.0;
  }
  let base_y = floor(sy);
  let base_x = floor(sx);
  let y0 = i32(base_y);
  let x0 = i32(base_x);
  let ly1 = sy - base_y;
  let lx1 = sx - base_x;
  let ly0 = 1.0 - ly1;
  let lx0 = 1.0 - lx1;
  let last_y = i32(dims.height_in) - 1;
  let last_x = i32(dims.width_in) - 1;
  var v00 = 0.0;
  var v01 = 0.0;
  var v10 = 0.0;
  var v11 = 0.0;
  if (y0 >= 0) {
    let row = plane + u32(y0) * dims.width_in;
    if (x0 >= 0) {
      v00 = x[row + u32(x0)];
    }
    if (x0 + 1 <= last_x) {
      v01 = x[row + u32(x0 + 1)];
    }
  }
  if (y0 + 1 <= last_y) {
    let row = plane + u32(y0 + 1) * dims.width_in;
    if (x0 >= 0) {
      v10 = x[row + u32(x0)];
    }
    if (x0 + 1 <= last_x) {
      v11 = x[row + u32(x0 + 1)];
    }
  }
  // 重みと加算順は torchvision 逐語（w1*v1 + w2*v2 + w3*v3 + w4*v4）。
  return ly0 * lx0 * v00 + ly0 * lx1 * v01 + ly1 * lx0 * v10 + ly1 * lx1 * v11;
}

@compute @workgroup_size(${DEFORM_CONV2D_WORKGROUP_SIZE})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let step = nwg.x * ${DEFORM_CONV2D_WORKGROUP_SIZE}u;
  let taps = dims.kernel_h * dims.kernel_w;
  let plane_out = dims.height_out * dims.width_out;
  let plane_in = dims.height_in * dims.width_in;
  var i = gid.x;
  while (i < dims.n) {
    let chunk = dims.channels_out * plane_out;
    let b = i / chunk;
    let rest = i % chunk;
    let oc = rest / plane_out;
    let pixel = rest % plane_out;
    let oy = pixel / dims.width_out;
    let ox = pixel % dims.width_out;
    // 入力側の基準位置は符号付き（padding ぶん負に出る）。stride / dilation は 1 固定。
    let origin_y = i32(oy) - i32(dims.padding_h);
    let origin_x = i32(ox) - i32(dims.padding_w);
    // offset / mask は B と出力画素で引く（offset_groups == 1 なのでグループ軸は無い）。
    let offset_base = b * 2u * taps * plane_out + pixel;
    let mask_base = b * taps * plane_out + pixel;
    var acc = bias[oc];
    for (var ic = 0u; ic < dims.channels_in; ic = ic + 1u) {
      let x_base = (b * dims.channels_in + ic) * plane_in;
      let w_base = (oc * dims.channels_in + ic) * taps;
      for (var kh = 0u; kh < dims.kernel_h; kh = kh + 1u) {
        for (var kw = 0u; kw < dims.kernel_w; kw = kw + 1u) {
          let tap = kh * dims.kernel_w + kw;
          // 偶数チャネル = y / 奇数チャネル = x
          let sy = f32(origin_y + i32(kh)) + offsets[offset_base + 2u * tap * plane_out];
          let sx = f32(origin_x + i32(kw)) + offsets[offset_base + (2u * tap + 1u) * plane_out];
          let m = modulator[mask_base + tap * plane_out];
          acc = acc + (m * deform_sample(x_base, sy, sx)) * w[w_base + tap];
        }
      }
    }
    out[i] = acc;
    i = i + step;
  }
}
`;

/** deform_conv2d の幾何（params の唯一の入力型）。 */
export type DeformConv2dDims = {
  readonly batch: number;
  readonly channelsIn: number;
  readonly channelsOut: number;
  readonly heightIn: number;
  readonly widthIn: number;
  readonly heightOut: number;
  readonly widthOut: number;
  readonly kernelH: number;
  readonly kernelW: number;
  readonly paddingH: number;
  readonly paddingW: number;
};

/**
 * 12 語（48 バイト）の uniform Dims。先頭は出力の全要素数。
 *
 * MUST: カーネル直呼びの経路でも幾何を見る（契約検査と二重だが、conv_transpose1d / conv2d と
 * 同じ二重の門）。空間長 0 の入力は `f32(dims.height_in)` が 0 になり、範囲判定が全タップで
 * false = 出力が bias 一色になる沈黙誤値を作る。
 */
export const deformConv2dParams = (dims: DeformConv2dDims): Uint32Array<ArrayBuffer> => {
  const values = [
    dims.batch,
    dims.channelsIn,
    dims.channelsOut,
    dims.heightIn,
    dims.widthIn,
    dims.heightOut,
    dims.widthOut,
    dims.kernelH,
    dims.kernelW,
    dims.paddingH,
    dims.paddingW,
  ];
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new CodegenError(`deform_conv2d params: 全ての次元は非負整数（${values.join(", ")}）`);
    }
  }
  const positive: readonly (readonly [string, number])[] = [
    ["batch", dims.batch],
    ["channels_in", dims.channelsIn],
    ["channels_out", dims.channelsOut],
    ["height_in", dims.heightIn],
    ["width_in", dims.widthIn],
    ["height_out", dims.heightOut],
    ["width_out", dims.widthOut],
    ["kernel_h", dims.kernelH],
    ["kernel_w", dims.kernelW],
  ];
  for (const [name, value] of positive) {
    if (value < 1) throw new CodegenError(`deform_conv2d params: ${name} は正整数（${value}）`);
  }
  const params = new Uint32Array(12);
  params[0] = dims.batch * dims.channelsOut * dims.heightOut * dims.widthOut;
  values.forEach((value, index) => {
    params[index + 1] = value;
  });
  return params;
};
