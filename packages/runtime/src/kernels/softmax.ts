/**
 * softmax / safe_softmax（最終次元、f32）の固定カーネル。
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
 * MUST: ① の行 max は `nan_max`（ビット列 NaN 判定 — ADR 0020）。WGSL の `max` は仕様レベルで
 * NaN を落とす（"Returns e2 if e1 is less than e2, and e1 otherwise"）ため、素の `max` だと
 * **全要素 NaN の行が safe_softmax の空行判定に化けて厳密 0 になり、NaN が黙って消える**
 * （部分 NaN 行は ② の総和経由で従来から伝播していた — 穴は全 NaN 行だけ）。非 NaN 入力では
 * nan_max = 素の max なので既存の数値結果は 1 ビットも動かない（v2）。
 *
 * ## safe_softmax 変種（ADR 0044）
 *
 * 契約は softmax と同一 + 「**行 max が −inf の行は全 0 を書く**」（torch の SDPA 分解が
 * 出す safe-softmax ガードと同じ意味論）。**同じ生成関数から両方を出す** MUST — ②③ の
 * 縮約順序・丸め位置が softmax と 1 語でもずれると、分解経路とのビット同一（parity テスト）
 * が壊れる。変種が足すのは次の 3 点だけ:
 *
 * 1. ① の identity を **−inf** にする（素の softmax は −F32_MAX）。有限要素を 1 つでも持つ
 *    行では max がその要素に支配されるので amax は両者で**同じビット列**になり、違いが出る
 *    のは「全要素 −inf」の行だけ — そこだけ amax が −inf になって判別できる。
 * 2. `empty`（行 max が −inf）なら ② の減算項を 0 に差し替える。`−inf − (−inf) = NaN` を
 *    そもそも作らないための置換で、非 empty 行では `select` が元の値をそのまま返す。
 * 3. ③ の書き出しを `select(…, 0.0, empty)` で包む。empty 行の ② は分母 0（0 除算の結果は
 *    WGSL では不定値）なので、**値を捨てる側で** 0 を確定させる。
 *
 * MUST: 分岐（`if`）で empty 行を早期に書き出さない。`amax` は workgroup メモリ由来の値で、
 * その条件下に `workgroupBarrier` を置くと WGSL の一様性解析に掛かる。`select` なら barrier
 * の位置が両経路で同じまま保たれる。
 * MUST: −inf のビット列は params で運ぶ（gather / embedding の NaN と同じ理由 — 定数式の
 * `bitcast<f32>(0xff800000u)` を「const-expression が inf」としてシェーダ生成エラーにする
 * 実装がありうる）。
 */

import { IS_NAN_BITS_WGSL, NAN_MAX_WGSL } from "../codegen/elementwise.ts";
import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";

export const SOFTMAX_WORKGROUP_SIZE = 256;

// v2: 行 max を nan_max へ（全 NaN 行の沈黙 0 化を塞ぐ — 上の MUST）
export const SOFTMAX_KEY = `softmax:v2:f32:lastdim:safe:wg${SOFTMAX_WORKGROUP_SIZE}`;

export const SAFE_SOFTMAX_KEY =
  `safe_softmax:v2:f32:lastdim:safe:emptyrow0:wg${SOFTMAX_WORKGROUP_SIZE}`;

/** f32 の最大有限値。WGSL に無限大リテラルが無いため amax の identity にこれを使う。 */
const F32_MAX = "3.402823466e38";

/** −inf の f32 ビット列（safe_softmax の params 3 語目）。 */
export const SAFE_SOFTMAX_NEG_INF_BITS = 0xff800000;

/**
 * softmax（`safe` で safe_softmax 変種）の WGSL。正本はこの 1 本で、変種が触るのは
 * 上の doc が列挙した 3 点だけ（共通部を 2 本に複製しない MUST）。
 */
const softmaxWgsl = (safe: boolean): string =>
  `// karume ${safe ? "safe_softmax" : "softmax"} (last dim, f32, safe-softmax: exp(x - amax)${
    safe ? ", 行 max が -inf の行は全 0" : ""
  })
struct Params {
  rows: u32,
  dim: u32,
${safe ? "  neg_inf: u32,\n" : ""}}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

${IS_NAN_BITS_WGSL}

${NAN_MAX_WGSL}

var<workgroup> scratch: array<f32, ${SOFTMAX_WORKGROUP_SIZE}>;

@compute @workgroup_size(${SOFTMAX_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let dim = params.dim;
${safe ? "  let neg_inf = bitcast<f32>(params.neg_inf);\n" : ""}  var row = wid.x;
  while (row < params.rows) {
    let base = row * dim;

    // ① 行の最大値（safe-softmax の減算項${safe ? " — identity は -inf" : ""}）
    var hi = ${safe ? "neg_inf" : `-${F32_MAX}`};
    var i = lid;
    while (i < dim) {
      hi = nan_max(hi, x[base + i]);
      i = i + ${SOFTMAX_WORKGROUP_SIZE}u;
    }
    scratch[lid] = hi;
    workgroupBarrier();
    var stride = ${SOFTMAX_WORKGROUP_SIZE / 2}u;
    while (stride > 0u) {
      if (lid < stride) {
        scratch[lid] = nan_max(scratch[lid], scratch[lid + stride]);
      }
      workgroupBarrier();
      stride = stride / 2u;
    }
${
    safe
      ? `    let row_max = scratch[0u];
    // 全要素 -inf の行（torch のガードが 0 を返す行）— 減算項を 0 にして NaN を作らない
    let empty = row_max == neg_inf;
    let amax = select(row_max, 0.0, empty);
`
      : "    let amax = scratch[0u];\n"
  }    // scratch の読み終わりを揃えてから ② で上書きする
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
      out[base + o] = ${
    safe ? "select(exp(x[base + o] - amax) * inv, 0.0, empty)" : "exp(x[base + o] - amax) * inv"
  };
      o = o + ${SOFTMAX_WORKGROUP_SIZE}u;
    }
    row = row + nwg.x;
  }
}
`;

export const SOFTMAX_WGSL: string = softmaxWgsl(false);

export const SAFE_SOFTMAX_WGSL: string = softmaxWgsl(true);

/**
 * uniform の Params。WGSL の uniform アドレス空間では struct の整列が 16 バイトになるため、
 * 2 語ぶんの内容でも 16 バイト確保する MUST。
 *
 * `safe` のとき 3 語目に −inf のビット列を載せる（safe_softmax は WGSL 側で `bitcast` して
 * ① の identity と空行判定に使う）。
 */
export const softmaxParams = (
  rows: number,
  dim: number,
  safe = false,
): Uint32Array<ArrayBuffer> => {
  assertU32Params("softmax params", { rows, dim });
  if (dim < 1) {
    throw new CodegenError(`softmax params: dim は正整数（${dim}）`);
  }
  const params = new Uint32Array(4);
  params[0] = rows;
  params[1] = dim;
  if (safe) params[2] = SAFE_SOFTMAX_NEG_INF_BITS;
  return params;
};
