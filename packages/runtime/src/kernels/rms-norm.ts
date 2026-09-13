/**
 * rms_norm（最終次元・weight のみ、f32）のカーネル（ADR 0017）。
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
 * ## 走査と加算順序
 *
 * 1 行 = 1 workgroup（dim ≤ 128 は 128、それ以外は 256 スレッド）の木構造縮約で、
 * 行方向は reduce.ts / layer-norm.ts と同じ grid-stride。行数が dispatch 上限（仕様既定 65535）を超えても縮退できる
 * （実測: Qwen3 の [1,512,1024] は 512 行だが、DiT の QK ノルムは head 軸ぶん行数が増える）。
 *
 * dim ≤ 128 では 256 版の上半分は二乗和 0 のみ。最初の +0 段だけを省き、
 * 下半分の加算順序と出力式を保つ（ADR 0017 追記 1）。
 *
 * MUST: params は uniform で渡す。行ループの中に workgroupBarrier があり、ループ条件が
 * workgroup 内で一様である必要がある（storage からのロードは一様性解析で保証されない）。
 * MUST: 逆数平方根は `inverseSqrt`（torch の `rsqrt` と同じ 1 演算）。CPU 参照は
 * `1 / Math.sqrt(…)` の別形で書く — 同じ式を写すと丸めの誤りが両側で相殺する。
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";

export const RMS_NORM_WORKGROUP_SIZE = 256;

/** MUST: WGSL を変えたらキーも上げる（異なる本文を同じキーへ混ぜない）。 */
export const RMS_NORM_KEY = `rms_norm:v1:f32:lastdim:wg${RMS_NORM_WORKGROUP_SIZE}`;

/** 後続addの入力順。加算の左右をコード生成でも維持する。 */
export type RmsNormAddOrder = "norm-residual" | "residual-norm";

const rmsNormWgsl = (
  workgroupSize: 128 | 256,
  addOrder?: RmsNormAddOrder,
): string =>
  `// karume rms_norm (last dim, weight only, f32, 二乗和 1 パス)
struct Params {
  rows: u32,
  dim: u32,
  eps: f32,${addOrder === undefined ? "" : "\n  rounding_mask: u32,"}
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(${addOrder === undefined ? 3 : 4}) var<storage, read_write> out: array<f32>;

var<workgroup> scratch: array<f32, ${workgroupSize}>;

${
    addOrder === undefined
      ? ""
      : "@group(0) @binding(3) var<storage, read> residual: array<f32>;\n"
  }@compute @workgroup_size(${workgroupSize})
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
      i = i + ${workgroupSize}u;
    }
    scratch[lid] = acc;
    workgroupBarrier();
    var stride = ${workgroupSize / 2}u;
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
      ${
    addOrder === undefined
      ? "out[base + o] = x[base + o] * inv * weight[o];"
      : addOrder === "norm-residual"
      ? "out[base + o] = bitcast<f32>(bitcast<u32>(x[base + o] * inv * weight[o]) ^ params.rounding_mask) + residual[base + o];"
      : "out[base + o] = residual[base + o] + bitcast<f32>(bitcast<u32>(x[base + o] * inv * weight[o]) ^ params.rounding_mask);"
  }
      o = o + ${workgroupSize}u;
    }
    row = row + nwg.x;
  }
}
`;

// 256 版は既存の本文とキーを維持する。WGSL の純粋な生成だけで GPU には触れない。
export const RMS_NORM_WGSL: string = rmsNormWgsl(RMS_NORM_WORKGROUP_SIZE);
export const RMS_NORM_128_KEY = "rms_norm:v1:f32:lastdim:wg128";
export const RMS_NORM_128_WGSL: string = rmsNormWgsl(128);

/**
 * 任意指定のRMS→add融合（DECIDED: docs/decisions/0099-rms-norm-add-fusion.md）。
 * 直接つなぐと最後の乗算とaddが融合し丸めが変わるため、uniformの0との整数XORを挟む。
 * この値はrmsNormParamsの第4語（0）。定数式のbitcast往復に簡略化しないこと。
 * 実測では参照と一致するが、WGSLは全バックエンドのビット同一を保証しない。
 * workgroup内の縮約とgrid-strideは単体RMSと共有し、公開の既定経路は変更しない。
 */
export const rmsNormAddWgsl = (order: RmsNormAddOrder): string => {
  assertRmsNormAddOrder(order);
  return rmsNormWgsl(RMS_NORM_WORKGROUP_SIZE, order);
};

export const rmsNormAddKey = (order: RmsNormAddOrder): string => {
  assertRmsNormAddOrder(order);
  return `rms_norm_add:v1:${RMS_NORM_KEY}:${order}:xor-round`;
};

const assertRmsNormAddOrder = (order: RmsNormAddOrder): void => {
  if (order !== "norm-residual" && order !== "residual-norm") {
    throw new CodegenError(`rms_norm_add: 未対応の加算順 '${String(order)}'`);
  }
};

/**
 * uniform の Params（rows / dim / eps。WGSL の uniform struct は 16 バイト整列なので
 * 3 語ぶんの内容でも 16 バイト確保する MUST）。
 *
 * MUST: eps は f32 のビット列として載せる。u32 として書くと指数部が整数値に化けて、
 * 例外なしに「eps ≈ 0」で走る。
 *
 * MUST: eps の値域は **f32 として**見る（layer_norm params と同じ二重の門）。f64 で有限正でも
 * `1e-50` は f32 で 0、`1e39` は `+Inf` になり、どちらも f64 のまま計算する CPU 参照と分岐する。
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
  const eps32 = Math.fround(eps);
  if (!Number.isFinite(eps32) || eps32 <= 0) {
    throw new CodegenError(`rms_norm params: eps は f32 として有限の正数（${eps}）`);
  }
  const params = new Uint32Array(4);
  params[0] = rows;
  params[1] = dim;
  new Float32Array(params.buffer)[2] = eps;
  return params;
};
