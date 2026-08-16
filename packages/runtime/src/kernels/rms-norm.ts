/**
 * rms_norm（最終次元・weight のみ、f32）の固定カーネル（ADR 0017）。
 * `out[r, i] = x[r, i] · rsqrt(mean_r(x²) + eps) · weight[i]`
 *
 * 入力は先行次元を平坦化した `[rows, dim]` の連続レイアウトとして扱う（契約は最終次元の
 * 正規化のみ — src/ops.ts）。
 *
 * ## layer_norm との差（MUST）
 *
 * **平均を引かない**。layer_norm が「2 パス（平均 → 偏差平方和）」を強いられていたのは
 * 母分散の桁落ちを避けるためで、rms_norm の縮約は**二乗和 1 本**なので 1 パスで足りる
 * （`E[x²] − E[x]²` のような相殺が構造的に起きない）。bias も無い — 足すと layer_norm と
 * 同じ op になる。
 *
 * ## 走査の裁定（性能ではなく正しさの都合）
 *
 * 1 行 = 1 workgroup（256 スレッド）の木構造縮約で、行方向は reduce.ts / layer-norm.ts と
 * 同じ grid-stride。行数が dispatch 上限（仕様既定 65535）を超えても縮退できる
 * （実測: Qwen3 の [1,512,1024] は 512 行だが、DiT の QK ノルムは head 軸ぶん行数が増える）。
 *
 * MUST: params は uniform で渡す。行ループの中に workgroupBarrier があり、ループ条件が
 * workgroup 内で一様である必要がある（storage からのロードは一様性解析で保証されない）。
 * MUST: 逆数平方根は `inverseSqrt`（torch の `rsqrt` と同じ 1 演算）。CPU 参照は
 * `1 / Math.sqrt(…)` の別形で書く — 同じ式を写すと丸めの誤りが両側で相殺する。
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "./params.ts";

export const RMS_NORM_WORKGROUP_SIZE = 256;

/** MUST: WGSL を変えたらキーも上げる（パイプラインキャッシュは本文を見ない）。 */
export const RMS_NORM_KEY = `rms_norm:v1:f32:lastdim:wg${RMS_NORM_WORKGROUP_SIZE}`;

export const RMS_NORM_WGSL: string = `// karume rms_norm (last dim, weight only, f32, 二乗和 1 パス)
struct Params {
  rows: u32,
  dim: u32,
  eps: f32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> scratch: array<f32, ${RMS_NORM_WORKGROUP_SIZE}>;

@compute @workgroup_size(${RMS_NORM_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let dim = params.dim;
  let scale = 1.0 / f32(dim);
  var row = wid.x;
  while (row < params.rows) {
    let base = row * dim;

    // 二乗和（1 パス）
    var acc = 0.0;
    var i = lid;
    while (i < dim) {
      let v = x[base + i];
      acc = acc + v * v;
      i = i + ${RMS_NORM_WORKGROUP_SIZE}u;
    }
    scratch[lid] = acc;
    workgroupBarrier();
    var stride = ${RMS_NORM_WORKGROUP_SIZE / 2}u;
    while (stride > 0u) {
      if (lid < stride) {
        scratch[lid] = scratch[lid] + scratch[lid + stride];
      }
      workgroupBarrier();
      stride = stride / 2u;
    }
    let inv = inverseSqrt(scratch[0u] * scale + params.eps);
    // 次の行が scratch[lid] を上書きする前に scratch[0] の読み終わりを揃える
    workgroupBarrier();

    var o = lid;
    while (o < dim) {
      out[base + o] = x[base + o] * inv * weight[o];
      o = o + ${RMS_NORM_WORKGROUP_SIZE}u;
    }
    row = row + nwg.x;
  }
}
`;

/**
 * uniform の Params（rows / dim / eps。WGSL の uniform struct は 16 バイト整列なので
 * 3 語ぶんの内容でも 16 バイト確保する MUST）。
 *
 * MUST: eps は f32 のビット列として載せる。u32 として書くと指数部が整数値に化けて、
 * 例外なしに「eps ≈ 0」で走る。
 */
export const rmsNormParams = (
  rows: number,
  dim: number,
  eps: number,
): Uint32Array<ArrayBuffer> => {
  assertU32Params("rms_norm params", { rows, dim });
  if (dim < 1) {
    throw new CodegenError(`rms_norm params: dim は正整数（${dim}）`);
  }
  if (!Number.isFinite(eps) || eps <= 0) {
    throw new CodegenError(`rms_norm params: eps は有限の正数（${eps}）`);
  }
  const params = new Uint32Array(4);
  params[0] = rows;
  params[1] = dim;
  new Float32Array(params.buffer)[2] = eps;
  return params;
};
