/**
 * NCHW の最終 2 空間軸を双線形補間で任意長へ resample する f32 固定カーネル
 * （`align_corners = True` 専業 — op `upsample_bilinear2d`）。
 *
 * 1 invocation = **出力 1 要素**で、入力の 4 近傍を読んで重み付き和を書く（出力駆動）。
 * 出力の全要素はちょうど 1 回書かれる（full-write 不変条件 — ADR 0014）。拡大・縮小の
 * どちらも同じ式で、倍率は非整数でよい（重みは出力位置ごとに違う）。
 *
 * 数値は torch の `aten/src/ATen/native/UpSample.h` に合わせる:
 *
 * - scale = `(in − 1) / (out − 1)`、**出力長 1 の軸は 0**（`area_pixel_compute_scale`）
 * - 源座標 = `scale · 出力添字`（`align_corners` 側は 0.5 のずらしを持たない）
 * - `index0 = trunc(源座標)` / `index1 = index0 + (index0 < in − 1 ? 1 : 0)` /
 *   `λ1 = 源座標 − index0` / `λ0 = 1 − λ1`（`compute_source_index_and_lambda`）
 * - 式木は **H が外・W が内**の入れ子 `λy0·(λx0·v00 + λx1·v01) + λy1·(λx0·v10 + λx1·v11)`
 *   （CUDA `UpSampleBilinear2d.cu` と同形）
 *
 * MUST: 末尾タップの特例（`index1 = index0`）を**整数比較**で書く。f32 の `min` / `max` に
 * 寄せると NaN を飲む実測（ADR 0020 の根治対象と同型）を再演する。
 * MUST: scale は**ホストで割ってから params で運ぶ**。WGSL の f32 除算は 2.5 ULP まで許され
 * （加減乗は正しく丸められる）、シェーダ内で割ると `scale · (out−1)` が入力の末尾添字を
 * わずかに下回りうる — そこが `align_corners` の「出力の端が入力の端と厳密一致」を壊す唯一の
 * 経路で、しかもバックエンドごとに結果が変わる。
 * MUST: 源座標は非負なので `u32(...)` の切り捨てが floor と一致する — `align_corners = False`
 * の座標式（`scale·(i+0.5) − 0.5`）は負値を取りうるのでこの前提が崩れる。**契約に欄が無い**
 * のはそのためで、受理を広げるならこの切り捨てから設計し直すこと。
 * MUST: grid-stride を使う。実モデルの出力は `[1,192,2048,2048]` 級 = 8.05 億要素で、必要な
 * workgroup 数が 1 次元 dispatch 上限（既定 65535）を 3 桁超える。
 */

import { CodegenError } from "../codegen/errors.ts";

export const UPSAMPLE_BILINEAR2D_WORKGROUP_SIZE = 256;

/** MUST: WGSL を変えたらキーも上げる（パイプラインキャッシュは本文を見ない）。 */
export const UPSAMPLE_BILINEAR2D_KEY =
  `upsample_bilinear2d:v1:nchw:f32:align_corners:wg${UPSAMPLE_BILINEAR2D_WORKGROUP_SIZE}`;

export const UPSAMPLE_BILINEAR2D_WGSL: string =
  `// karume upsample_bilinear2d (NCHW bilinear, align_corners=true, f32, full-write)
struct Params {
  n: u32,
  in_h: u32,
  in_w: u32,
  out_h: u32,
  out_w: u32,
  // ホストが f32 で割った (in - 1) / (out - 1)（出力長 1 の軸は 0）
  scale_h: f32,
  scale_w: f32,
  reserved: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(${UPSAMPLE_BILINEAR2D_WORKGROUP_SIZE})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let stride = nwg.x * ${UPSAMPLE_BILINEAR2D_WORKGROUP_SIZE}u;
  let out_hw = params.out_h * params.out_w;
  let in_hw = params.in_h * params.in_w;
  var i = gid.x;
  while (i < params.n) {
    // N と C は補間に関与しないので 1 本の平面添字へ畳む（rank を params に載せない）。
    let ox = i % params.out_w;
    let oy = (i / params.out_w) % params.out_h;
    let plane = i / out_hw;

    let ry = params.scale_h * f32(oy);
    let y0 = u32(ry);
    var y1 = y0;
    if (y0 + 1u < params.in_h) {
      y1 = y0 + 1u;
    }
    let ly1 = ry - f32(y0);
    let ly0 = 1.0 - ly1;

    let rx = params.scale_w * f32(ox);
    let x0 = u32(rx);
    var x1 = x0;
    if (x0 + 1u < params.in_w) {
      x1 = x0 + 1u;
    }
    let lx1 = rx - f32(x0);
    let lx0 = 1.0 - lx1;

    let row0 = plane * in_hw + y0 * params.in_w;
    let row1 = plane * in_hw + y1 * params.in_w;
    out[i] = ly0 * (lx0 * x[row0 + x0] + lx1 * x[row0 + x1])
      + ly1 * (lx0 * x[row1 + x0] + lx1 * x[row1 + x1]);
    i = i + stride;
  }
}
`;

/**
 * torch の `area_pixel_compute_scale`（`align_corners` 側）。**出力長 1 の軸は 0**
 * （そこで `(in−1)/(out−1)` を書くとゼロ除算になり、torch も同じ特例を持つ）。
 *
 * `Math.fround` は f64 除算を f32 へ丸めるが、被除数・除数がともに小さな整数なので
 * 二重丸めは起こらない（除数が 2 冪なら商は f32 で厳密・そうでなければ商は無限 2 進小数なので
 * f32 の丸め境界ちょうどにはならない）。torch の `float(in−1) / float(out−1)` と同じ値。
 */
const alignCornersScale = (lengthIn: number, lengthOut: number): number =>
  lengthOut > 1 ? Math.fround((lengthIn - 1) / (lengthOut - 1)) : 0;

/**
 * 32-byte uniform params（u32 5 語 + f32 2 語 + 予約 1 語 — WGSL の uniform struct は
 * 16 バイト整列 MUST）。`n` は出力の全要素数。
 *
 * MUST: 空間軸の長さ 0 をここで拒否する。0 は scale が `(0−1)/(out−1)` で負になり、
 * 読み出しが入力の外へ出る（GPU では例外なしに隣の値が出る）。
 * MUST: `n % (out_h · out_w)` をここで拒否する。割れない値を通すと最後の不完全な平面の
 * 添字分解が次の平面へずれ、要素数だけでは検出できない沈黙誤値になる（upsample2x と同じ罠）。
 */
export const upsampleBilinear2dParams = (
  n: number,
  heightIn: number,
  widthIn: number,
  heightOut: number,
  widthOut: number,
): Uint32Array<ArrayBuffer> => {
  for (const [name, value] of Object.entries({ n, heightIn, widthIn, heightOut, widthOut })) {
    if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
      throw new CodegenError(`upsample_bilinear2d params: ${name} は u32 の非負整数（${value}）`);
    }
  }
  for (const [name, value] of Object.entries({ heightIn, widthIn, heightOut, widthOut })) {
    if (value === 0) {
      throw new CodegenError(`upsample_bilinear2d params: ${name} は 1 以上`);
    }
  }
  const plane = heightOut * widthOut;
  if (n % plane !== 0) {
    throw new CodegenError(
      `upsample_bilinear2d params: n ${n} が出力平面 ${heightOut}×${widthOut} の整数倍でない`,
    );
  }
  const params = new Uint32Array([n, heightIn, widthIn, heightOut, widthOut, 0, 0, 0]);
  const floats = new Float32Array(params.buffer);
  floats[5] = alignCornersScale(heightIn, heightOut);
  floats[6] = alignCornersScale(widthIn, widthOut);
  return params;
};
