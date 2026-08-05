/**
 * softmax（最終次元、f32）の固定カーネル。
 *
 * ## safe-softmax（MUST）
 *
 * `exp(x − amax_r)` の形で組む。素朴な `exp(x) / Σexp(x)` は台帳の反例集どおり壊れる:
 *
 * - 大きい正値: `exp(89) > f32 の最大有限値` で `inf`、`inf / inf = NaN`。
 * - 大きい負値: `exp(−104) < f32 の最小非正規数` で全項 0 になり `0 / 0 = NaN`。
 *   masked_fill の埋め値 −3.4028234663852886e+38 を通した行は**必ず**この領域に入る。
 *
 * amax を引いた形なら指数の引数は常に `(−∞, 0]` で、最大要素が必ず `exp(0) = 1` を出すため
 * 分母が 0 にならない。全要素が同じ値（全マスク行を含む）でも `1/dim` の一様分布になり、
 * torch と一致する。
 *
 * 入力は先行次元を平坦化した `[rows, dim]` の連続レイアウトとして扱う。1 行 = 1 workgroup
 * （256 スレッド）の**3 パス**: ① amax ② `Σ exp(x − amax)` ③ 書き出し。②③ で `exp` を
 * 2 度評価するのは、行長ぶんの中間バッファを持たずに済ませるため（正しさ優先 — 性能
 * マイルストーンでの置換対象）。同じ引数の `exp` は同じ値を返すので決定性は保たれる。
 *
 * MUST: params は uniform で渡す（行ループ内に workgroupBarrier があり、ループ条件が
 * workgroup 内で一様である必要がある）。
 * NOTE: WGSL の `max` は NaN 伝播を保証しない（reduce.ts と同じ既知の乖離）。入力に NaN を
 * 含む行の結果は未規定で、M1 の対象は有限値のみ。
 */

import { CodegenError } from "../codegen/errors.ts";

export const SOFTMAX_WORKGROUP_SIZE = 256;

export const SOFTMAX_KEY = `softmax:v1:f32:lastdim:safe:wg${SOFTMAX_WORKGROUP_SIZE}`;

/** f32 の最大有限値。WGSL に無限大リテラルが無いため amax の identity にこれを使う。 */
const F32_MAX = "3.402823466e38";

export const SOFTMAX_WGSL: string = `// karume softmax (last dim, f32, safe-softmax: exp(x - amax))
struct Params {
  rows: u32,
  dim: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

var<workgroup> scratch: array<f32, ${SOFTMAX_WORKGROUP_SIZE}>;

@compute @workgroup_size(${SOFTMAX_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let dim = params.dim;
  var row = wid.x;
  while (row < params.rows) {
    let base = row * dim;

    // ① 行の最大値（safe-softmax の減算項）
    var hi = -${F32_MAX};
    var i = lid;
    while (i < dim) {
      hi = max(hi, x[base + i]);
      i = i + ${SOFTMAX_WORKGROUP_SIZE}u;
    }
    scratch[lid] = hi;
    workgroupBarrier();
    var stride = ${SOFTMAX_WORKGROUP_SIZE / 2}u;
    while (stride > 0u) {
      if (lid < stride) {
        scratch[lid] = max(scratch[lid], scratch[lid + stride]);
      }
      workgroupBarrier();
      stride = stride / 2u;
    }
    let amax = scratch[0u];
    // scratch の読み終わりを揃えてから ② で上書きする
    workgroupBarrier();

    // ② Σ exp(x - amax)（最大要素が exp(0) = 1 を出すので分母は必ず 1 以上）
    var acc = 0.0;
    var j = lid;
    while (j < dim) {
      acc = acc + exp(x[base + j] - amax);
      j = j + ${SOFTMAX_WORKGROUP_SIZE}u;
    }
    scratch[lid] = acc;
    workgroupBarrier();
    var stride2 = ${SOFTMAX_WORKGROUP_SIZE / 2}u;
    while (stride2 > 0u) {
      if (lid < stride2) {
        scratch[lid] = scratch[lid] + scratch[lid + stride2];
      }
      workgroupBarrier();
      stride2 = stride2 / 2u;
    }
    let inv = 1.0 / scratch[0u];
    // 次の行が scratch[lid] を上書きする前に scratch[0] の読み終わりを揃える
    workgroupBarrier();

    // ③ 書き出し（exp は ② と同じ引数 — 同じ値が返るので決定的）
    var o = lid;
    while (o < dim) {
      out[base + o] = exp(x[base + o] - amax) * inv;
      o = o + ${SOFTMAX_WORKGROUP_SIZE}u;
    }
    row = row + nwg.x;
  }
}
`;

/**
 * uniform の Params。WGSL の uniform アドレス空間では struct の整列が 16 バイトになるため、
 * 2 語ぶんの内容でも 16 バイト確保する MUST。
 */
export const softmaxParams = (rows: number, dim: number): Uint32Array<ArrayBuffer> => {
  if (!Number.isSafeInteger(rows) || rows < 0 || !Number.isSafeInteger(dim) || dim < 1) {
    throw new CodegenError(`softmax params: rows は非負整数 / dim は正整数（${rows}, ${dim}）`);
  }
  const params = new Uint32Array(4);
  params[0] = rows;
  params[1] = dim;
  return params;
};
