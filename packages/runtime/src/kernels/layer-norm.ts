/**
 * layer_norm（最終次元・affine あり、f32）の固定カーネル。
 * `out[r, i] = (x[r, i] − mean_r) / sqrt(var_r + eps) · weight[i] + bias[i]`
 *
 * 入力は先行次元を平坦化した `[rows, dim]` の連続レイアウトとして扱う（契約は最終次元の
 * 正規化のみ — src/ops.ts）。
 *
 * ## 分散の規約（MUST）
 *
 * **母分散（N 割り、correction = 0）**。`torch.var` の既定は `correction = 1`（N−1 割り）だが、
 * 正規化層の内部分散は N 割りで、ここを取り違えると次元が小さいほど誤差が拡大する
 * （台帳の反例集「`var` = `sum((x−mean)²)/N` は誤り」の逆方向の罠）。
 *
 * ## 走査の裁定（性能ではなく正しさの都合）
 *
 * 1 行 = 1 workgroup（256 スレッド）の**2 パス**で組む: ① 和 → 平均 ② 偏差平方和 → 分散。
 * `E[x²] − E[x]²` の 1 パス形にすると桁落ちで負の分散が出うる（f32 で `mean²` が大きい行）
 * ので採らない。行方向は reduce.ts と同じ grid-stride で、行数が dispatch 上限（仕様既定
 * 65535）を超えても縮退できる。
 *
 * NOTE: 台帳の「逐次行走査を複製してよいか」の裁定はここでは**逐次にしない**（行長 1024 に
 * 対し 256 スレッドの木構造縮約）。reduce.ts が既に木構造で、同じ形の複製にした方が
 * 「1 行 = 1 workgroup」の不変条件が族として揃う。性能マイルストーンでの置換対象は
 * matmul 族と同じ扱い。
 *
 * MUST: params は uniform で渡す。行ループの中に workgroupBarrier があり、ループ条件が
 * workgroup 内で一様である必要がある（storage からのロードは一様性解析で保証されない）。
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";

export const LAYER_NORM_WORKGROUP_SIZE = 256;

export const LAYER_NORM_KEY = `layer_norm:v1:f32:lastdim:wg${LAYER_NORM_WORKGROUP_SIZE}`;

/**
 * 1 行ぶんの統計（`base` / `mean` / `inv`）を出す本体。**融合カーネルと同一文字列を共有する
 * MUST** — 別々に書くと素の列と融合版で縮約順が割れ、`(x−mean)²` の足し合わせ順が 1 つ違う
 * だけでビット一致が崩れる（`SIGMOID_STABLE_WGSL` を silu.ts と共有しているのと同じ理由）。
 *
 * 前提（利用側が用意する識別子）: `params`（rows / dim / eps）・`x`・`scratch`
 * （`array<f32, {@link LAYER_NORM_WORKGROUP_SIZE}>`）・`lid`・`dim`・`scale`（= 1/dim）と、
 * workgroup 一様な行ループ変数 `row`。barrier を含むのでこのブロックは**一様な制御流**の
 * 中にしか置けない。
 */
export const LAYER_NORM_ROW_STATS_WGSL: string = `    let base = row * dim;

    // ① 和 → 平均
    var acc = 0.0;
    var i = lid;
    while (i < dim) {
      acc = acc + x[base + i];
      i = i + ${LAYER_NORM_WORKGROUP_SIZE}u;
    }
    scratch[lid] = acc;
    workgroupBarrier();
    var stride = ${LAYER_NORM_WORKGROUP_SIZE / 2}u;
    while (stride > 0u) {
      if (lid < stride) {
        scratch[lid] = scratch[lid] + scratch[lid + stride];
      }
      workgroupBarrier();
      stride = stride / 2u;
    }
    let mean = scratch[0u] * scale;
    // scratch の読み終わりを揃えてから ② で上書きする
    workgroupBarrier();

    // ② 偏差平方和 → 母分散（correction = 0）
    var sq = 0.0;
    var j = lid;
    while (j < dim) {
      let d = x[base + j] - mean;
      sq = sq + d * d;
      j = j + ${LAYER_NORM_WORKGROUP_SIZE}u;
    }
    scratch[lid] = sq;
    workgroupBarrier();
    var stride2 = ${LAYER_NORM_WORKGROUP_SIZE / 2}u;
    while (stride2 > 0u) {
      if (lid < stride2) {
        scratch[lid] = scratch[lid] + scratch[lid + stride2];
      }
      workgroupBarrier();
      stride2 = stride2 / 2u;
    }
    let inv = 1.0 / sqrt(scratch[0u] * scale + params.eps);
    // 次の行が scratch[lid] を上書きする前に scratch[0] の読み終わりを揃える
    workgroupBarrier();`;

/**
 * 正規化済み 1 要素の affine（`weight` / `bias` を当てる式）。{@link LAYER_NORM_ROW_STATS_WGSL}
 * と同じ理由で融合カーネルと**同一文字列を共有する MUST**（fma へ縮約するかどうかは式の
 * 書き方で変わる）。行内添字は `base + o`、重み添字は `o` を前提にする。
 */
export const LAYER_NORM_AFFINE_WGSL = "(x[base + o] - mean) * inv * weight[o] + bias[o]";

export const LAYER_NORM_WGSL: string =
  `// karume layer_norm (last dim, affine, f32, 2 パス / 母分散)
struct Params {
  rows: u32,
  dim: u32,
  eps: f32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

var<workgroup> scratch: array<f32, ${LAYER_NORM_WORKGROUP_SIZE}>;

@compute @workgroup_size(${LAYER_NORM_WORKGROUP_SIZE})
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
${LAYER_NORM_ROW_STATS_WGSL}

    var o = lid;
    while (o < dim) {
      out[base + o] = ${LAYER_NORM_AFFINE_WGSL};
      o = o + ${LAYER_NORM_WORKGROUP_SIZE}u;
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
export const layerNormParams = (
  rows: number,
  dim: number,
  eps: number,
): Uint32Array<ArrayBuffer> => {
  assertU32Params("layer_norm params", { rows, dim });
  if (dim < 1) {
    throw new CodegenError(`layer_norm params: dim は正整数（${dim}）`);
  }
  if (!Number.isFinite(eps) || eps <= 0) {
    throw new CodegenError(`layer_norm params: eps は有限の正数（${eps}）`);
  }
  const params = new Uint32Array(4);
  params[0] = rows;
  params[1] = dim;
  new Float32Array(params.buffer)[2] = eps;
  return params;
};
