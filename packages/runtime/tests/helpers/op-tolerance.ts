/**
 * op 語彙掃引（gpu_ops_test.ts の checkAll）の **op 別 tolerance 表**。
 *
 * 正本 = [docs/research/2026-08-31-op-tolerance-measurement.md](../../../../docs/research/2026-08-31-op-tolerance-measurement.md)
 * §7（実 GPU 実測 + K 依存則 maxAbs ≈ C·eps·√K·max|期待| から導出・全帯とも実測の 4 倍以上の
 * マージンを消費率で機械確認済み）。旧 DEFAULT_TOLERANCE（atol 1e-5 / rtol 1e-3）は掃引の
 * 門としては最悪でも消費率 2.6e-2 しか使われておらず、退行を通す帯だった（同 §2.2）。
 *
 * MUST: **表に無い op は fail loudly**（既定帯へ落とさない）。新 op を足すときは実測して
 * 行を足す — 「黙って汎用帯で通る」形を塞ぐのがこの表の存在理由。ケース単位の例外は
 * `OpCase.tolerance` の上書きで持つ（悪条件入力用の逃げ道 — 表は変えない）。
 *
 * NOTE: 「厳密 (0, 0)」の op のうち GEMM / 縮約系は**このコーパスの入力が 2 進格子上にある**
 * ことに依存した厳密性（研究 §2 の格子論証 — IEEE-754 f32 準拠の実装なら成り立つ）。
 * 一般入力での帯は GEMM / 縮約クラスの値が正で、ビット同一門（gpu_ops_test.ts）が
 * この厳密性を別テストとして固定する。
 */

import type { Tolerance } from "../../src/reference/allclose.ts";

const EXACT: Tolerance = { atol: 0, rtol: 0 };
/** 除算のみの単発 op（実測 1 ULP・仕様 2.5 ULP の 6.4 倍）。 */
const ULP_DIV: Tolerance = { atol: 1e-7, rtol: 2e-6 };
/** 超越関数の単発 op（実測 ≤13 ULP / 仕様 ≤12 ULP の 5 倍 + 絶対床）。 */
const TRANSCENDENTAL: Tolerance = { atol: 2e-6, rtol: 8e-6 };
/** 超越を含む合成 op — 桁落ち位置（gelu 負側テール等）を絶対床で受ける（相対で切らない）。 */
const COMPOSITE: Tolerance = { atol: 4e-6, rtol: 8e-6 };
/** softmax 族（実測 abs 6.0e-8 の 16 倍 — K 非依存を実測で確認済み）。 */
const SOFTMAX: Tolerance = { atol: 1e-6, rtol: 8e-6 };
/** 4 タップ和（bilinear）。 */
const SMALL_REDUCE: Tolerance = { atol: 4e-6, rtol: 4e-6 };
/** 行縮約（K ≤ 16384 の √K 則から）。 */
const ROW_REDUCE: Tolerance = { atol: 4e-5, rtol: 4e-6 };
/** attention（飽和 softmax ケースの abs 6.9e-6 の 6 倍。相対は発散するので絶対床が主）。 */
const ATTENTION: Tolerance = { atol: 4e-5, rtol: 8e-6 };
/**
 * GEMM / conv 族（コーパス最大 K=72 の則 4·eps·√K から。K がこれより桁で大きいケースを
 * 足すときは帯を再導出する）。量子化重み経路の GPU vs 参照突合（gpu_{f16,i8,i4}_weights /
 * skinny / gemv）もこのクラスを明示的に使う。
 */
export const GEMM_TOLERANCE: Tolerance = { atol: 2e-4, rtol: 4e-6 };

const OP_TOLERANCE: Readonly<Record<string, Tolerance>> = {
  // 厳密 — 値を作らない / 正しく丸められる単発演算（f32 出力のもの）
  neg: EXACT,
  abs: EXACT,
  relu: EXACT,
  clamp: EXACT,
  clamp_min: EXACT,
  leaky_relu: EXACT,
  where: EXACT,
  cast: EXACT,
  add: EXACT,
  sub: EXACT,
  mul: EXACT,
  // 厳密 — データ移動のみ
  reshape: EXACT,
  permute: EXACT,
  expand: EXACT,
  slice: EXACT,
  cat: EXACT,
  sym_prefix_slice: EXACT,
  pad: EXACT,
  flip: EXACT,
  gather: EXACT,
  embedding: EXACT,
  masked_fill: EXACT,
  upsample2x: EXACT,
  // 厳密 — 整数 / bool 出力（compareTensors の dtype 分岐でも EXACT になるが、表の全 op
  // 網羅（fail loudly）を保つため行を持つ）
  bitwise_and: EXACT,
  bitwise_not: EXACT,
  ge: EXACT,
  ge_scalar: EXACT,
  gt_scalar: EXACT,
  le_scalar: EXACT,
  // 厳密 — 選択 / 添字 / 逐次丸め一致（cumsum は参照が GPU の逐次丸めを鏡写し）
  amax: EXACT,
  amin: EXACT,
  argmax: EXACT,
  topk: EXACT,
  cumsum: EXACT,
  // ULP 級
  div: ULP_DIV,
  // 超越（単発）
  exp: TRANSCENDENTAL,
  log: TRANSCENDENTAL,
  log1p: TRANSCENDENTAL,
  sqrt: TRANSCENDENTAL,
  sin: TRANSCENDENTAL,
  tanh: TRANSCENDENTAL,
  sigmoid: TRANSCENDENTAL,
  // 合成（桁落ち位置を絶対床で受ける）
  gelu: COMPOSITE,
  gelu_tanh: COMPOSITE,
  silu: COMPOSITE,
  layer_norm: COMPOSITE,
  rms_norm: COMPOSITE,
  gru_scan: COMPOSITE,
  gru_scan_reverse: COMPOSITE,
  softmax: SOFTMAX,
  safe_softmax: SOFTMAX,
  // 縮約 / GEMM
  upsample_bilinear2d: SMALL_REDUCE,
  sum: ROW_REDUCE,
  attention: ATTENTION,
  matmul: GEMM_TOLERANCE,
  bmm: GEMM_TOLERANCE,
  linear: GEMM_TOLERANCE,
  conv1d: GEMM_TOLERANCE,
  conv2d: GEMM_TOLERANCE,
  conv_transpose1d: GEMM_TOLERANCE,
  deform_conv2d: GEMM_TOLERANCE,
};

/**
 * 掃引が使う op 別 tolerance。表に無い op は fail loudly（上の MUST）。
 * i32 / bool 出力は compareTensors が dtype で EXACT を選ぶので、ここは f32 出力の帯だけを持つ。
 */
export const opTolerance = (op: string): Tolerance => {
  if (!Object.hasOwn(OP_TOLERANCE, op)) {
    throw new Error(
      `op-tolerance: op '${op}' の tolerance が表に無い — 実測して行を足すこと` +
        "（docs/research/2026-08-31-op-tolerance-measurement.md §7 の手順・既定帯へは落とさない）",
    );
  }
  return OP_TOLERANCE[op];
};

/**
 * 現行コーパスで **GPU と f64 オラクルがビット同一**であることを固定する op 集合
 * （研究 §7.1 — 入力が 2 進格子上にあり中間和が f32 の厳密整数域に収まるため、IEEE-754 f32
 * 準拠のどの実装でも丸めが一度も起きない）。割れたら tolerance を緩める対象ではなく
 * 「ドライバが f32 未満で計算している / 生成 WGSL が別演算へ落ちた」のシグナル。
 * 特定ドライバで割れる場合は op 単位で門を外して理由を記録する（tanh 飽和カナリアと同じ運用）。
 */
export const BIT_IDENTICAL_OPS: ReadonlySet<string> = new Set([
  "matmul",
  "bmm",
  "linear",
  "conv1d",
  "conv2d",
  "conv_transpose1d",
  "deform_conv2d",
  "sum",
  "upsample_bilinear2d",
]);
