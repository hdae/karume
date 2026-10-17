// The collapse ceilings of the DeBERTa w8a8 mirror gate (output.2 onwards — ADR 0026 addendum
// 2026-10-07) and their judgement. They live outside the e2e test so the asset-free unit test
// (`collapse_ceilings_test.ts`) can import them without registering the GPU cases.

import { AllcloseError } from "../../../runtime/src/reference/allclose.ts";

/** The number of outputs of the full-24layer graph (embedding exit + 24 layer exits). */
export const GRAPH_OUTPUTS = 25;

/** 厳密 tolerance を掛ける出力の本数（output.0 と output.1 — 判別帯が残る 1 層目の出口まで）。 */
export const STRICT_OUTPUTS = 2;

/**
 * output.2 以降に掛ける**崩壊上限**の表 = 出力名 → 相対 RMS 誤差 ‖gpu − golden‖₂ / ‖golden‖₂ の
 * 上限（**この環境で実測から導いた宣言値** — 2026-10-07・RTX 3080 Ti / Ryzen 5 7600・ADR 0026
 * 追記 2026-10-07）。宣言であって環境キー別の行ではない（ADR 0110 決定 4）。
 *
 * 導き方: 出力ごと（観測点 = output.k）に、正当な標本 60 本（4 ケース × 〈GPU a8 1 本 + この機の
 * CPU で採り直した torch 鏡像の別標本 6 本〈AVX512 / AVX2 / DEFAULT × 1 / 6 スレッド〉+ 鏡像の
 * 埋め込み出口に相対 ±2^-23 の乱数を入れた摂動アンサンブル 8 本〉）の最悪 × 2 を、有効数字 2 桁で
 * 切り上げた値。この門は ADR 0110 決定 5 の A/B 門ではない（比較相手は同じ機の参照層ではなく torch
 * 鏡像 golden）が、崩壊上限の導き方だけを決定 5-3（E2E の崩壊上限 = 観測点の実測 × 2 程度）に揃え、
 * 決定 4 の MUST NOT（「実測の 5〜10 倍」を E2E の帯の導出に使う）を踏まない。5-3 の「量子化席なら
 * 理論値との整合を残す」は当てはめない: 活性の丸めは不連続で、上流の 1e-5 級の差が段の ±1 飛びを
 * 起こして数層で飽和する（ADR 0026「検出限界」）ので、output.k の相対 RMS 誤差に使える理論上界が
 * 無い — a8 の数値の正しさは `tests/gpu_i8a8_test.ts` の atol=0 の数値契約が受け持つ。GPU f32 経路の
 * dump は a8 の標本ではないので入れない。最悪は浅い出力ほど小さい（output.2 0.0099 → output.24 0.1620）ので、
 * 1 つのスカラーで掛けると浅い出力の上限が桁違いに緩む — 表にするのはそのため。
 *
 * 再現: `tools/export-recipes/deberta/collapse_ceilings.py` の `derive` が作業 dir の `derive.json`
 * の `ceilings` にこの表を書く（標本の作り方と順序は ADR 0026 追記の「導き直しの手順」）。故障注入は
 * `tools/diag/deberta-collapse.ts` の `faults`（{@link collapseCeilingOf} と {@link judgeCollapse} を
 * そのまま import する）。2026-10-07 の材料は `outputs/diag/deberta-w8a8-2026-10-07/`（git 追跡外）。
 *
 * 数値パリティではない: 受け持つのは非有限（{@link judgeCollapse} が両側・全要素を見る）と、出力
 * 全体が golden と別物になる崩壊（全要素 0 = 1.0・符号反転 ≈ 2.0・全要素 × 2 ≈ 1.0・× 1.5 ≈ 0.5）
 * だけ。中間層の 1 要素の飛びや f32 経路への沈黙フォールバックは捕まえない — 細かな欠陥は
 * output.0 / 1 の厳密 tolerance・census・`tests/gpu_i8a8_test.ts` の atol=0 の数値契約の受け持ち。
 * 上限を広げて通すのではなく、先に i8a8 の scale / accumulator / 適格判定のどれが動いたかを確かめる。
 */
export const COLLAPSE_CEILINGS: Readonly<Record<string, number>> = {
  "output.2": 0.02,
  "output.3": 0.037,
  "output.4": 0.051,
  "output.5": 0.062,
  "output.6": 0.071,
  "output.7": 0.085,
  "output.8": 0.095,
  "output.9": 0.11,
  "output.10": 0.12,
  "output.11": 0.13,
  "output.12": 0.14,
  "output.13": 0.15,
  "output.14": 0.15,
  "output.15": 0.17,
  "output.16": 0.17,
  "output.17": 0.19,
  "output.18": 0.21,
  "output.19": 0.22,
  "output.20": 0.24,
  "output.21": 0.26,
  "output.22": 0.28,
  "output.23": 0.31,
  "output.24": 0.33,
};

/**
 * io の出力名（`output.<k>`）の崩壊上限。表に無い出力は投げる（上限の無い出力を黙って通さない）。
 */
export const collapseCeilingOf = (output: string): number => {
  if (!Object.hasOwn(COLLAPSE_CEILINGS, output)) {
    throw new Error(`${output} の崩壊上限が宣言されていない（COLLAPSE_CEILINGS）`);
  }
  return COLLAPSE_CEILINGS[output];
};

/** 崩壊上限の判定 1 出力ぶん（{@link judgeCollapse}）。 */
export type CollapseReport = {
  readonly pass: boolean;
  /** ‖actual − expected‖₂ / ‖expected‖₂（f64 で積む）。非有限があれば +Inf。 */
  readonly relRms: number;
  /** 記録だけ — 判定には使わない。非有限があれば +Inf。 */
  readonly maxAbs: number;
  /** どちらかの側が NaN / ±Inf だった要素数（全要素を見る）。 */
  readonly nonFiniteCount: number;
};

/**
 * output.2 以降の 1 出力の判定: 非有限 0（両側・全要素）かつ相対 RMS 誤差 ≤ `ceiling`（その出力の
 * 上限 — {@link collapseCeilingOf}）。
 *
 * maxAbs で判定しない理由: 中間層は外れ値チャネル（channel 686・\|ref\| 最大 28.8）の符号の
 * 分岐点で、活性量子化の段の反転が 1 要素を大きく動かす。CPU の torch 鏡像の別標本どうしでも
 * その 1 要素が 7.17 動く（ADR 0026 追記 2026-10-07）。
 */
export const judgeCollapse = (
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  ceiling: number,
): CollapseReport => {
  if (actual.length !== expected.length) {
    throw new AllcloseError(`長さ不一致: actual ${actual.length} vs expected ${expected.length}`);
  }
  let difference = 0;
  let norm = 0;
  let maxAbs = 0;
  let nonFiniteCount = 0;
  for (let index = 0; index < expected.length; index += 1) {
    const x = actual[index];
    const y = expected[index];
    // MUST: 非有限は相対 RMS の大小に任せず数えて落とす — 合否を比較の向き（NaN はどの比較も
    // false）に依存させず、報告にも何が起きたかを出す。
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      nonFiniteCount += 1;
      continue;
    }
    const delta = x - y;
    difference += delta * delta;
    norm += y * y;
    maxAbs = Math.max(maxAbs, Math.abs(delta));
  }
  if (nonFiniteCount > 0) {
    return {
      pass: false,
      relRms: Number.POSITIVE_INFINITY,
      maxAbs: Number.POSITIVE_INFINITY,
      nonFiniteCount,
    };
  }
  const relRms = Math.sqrt(difference) / Math.sqrt(norm);
  // ‖expected‖₂ = 0 は 0/0 = NaN か +Inf になり、`<=` が false を返して落ちる。
  return { pass: relRms <= ceiling, relRms, maxAbs, nonFiniteCount };
};
