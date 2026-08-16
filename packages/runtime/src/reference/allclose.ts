/**
 * 数値一致の判定。GPU 実行と CPU 参照の突合（ADR 0005 の段 2）に使う。
 *
 * MUST: 判定は常に `|x−y| ≤ atol + rtol·|ref|`。相対誤差だけを見るとゼロ近傍の参照値で
 * 発散し、絶対誤差だけを見ると大きい値で無意味に厳しくなる。
 * MUST: NaN / ±Inf はどちらの側にあっても不合格。NaN は全ての比較が false になるため、
 * 素朴な差分判定だと「差 0」に化けて全 NaN 出力が PASS になる。
 */

import type { IrDtype } from "../format/ir.ts";

/** 比較対象の長さ・dtype が違う（取り違え — 許容誤差の問題ではない）。 */
export class AllcloseError extends Error {
  override readonly name = "AllcloseError";
}

export type Tolerance = {
  readonly atol: number;
  readonly rtol: number;
};

/**
 * NOTE: 根拠は薄い。M0 の f32 カーネル（縮約順序が torch と違う）で経験的に通る値を
 * 置いただけで、op ごと・shape ごとの誤差伝播から導いたものではない。torch 由来の
 * ゴールデンが入る M1 で、op 単位に根拠付きの値へ置き換える（ADR 0005 の未決事項）。
 */
export const DEFAULT_TOLERANCE: Tolerance = { atol: 1e-5, rtol: 1e-3 };

/**
 * 整数系（i32 / bool）の突合に使う許容誤差 = 厳密一致（ADR 0009）。
 * 整数演算には丸め差も縮約順序も無いので、1 でもずれたら実装バグ。
 */
export const EXACT_TOLERANCE: Tolerance = { atol: 0, rtol: 0 };

type AllcloseReport = {
  readonly pass: boolean;
  /** 許容誤差を破った要素数。 */
  readonly failCount: number;
  readonly maxAbsError: number;
  readonly maxRelError: number;
  /** maxAbsError（非有限があればその先頭）の位置。 */
  readonly worstIndex: number;
  /** どちらかの側が NaN / ±Inf だった要素数。 */
  readonly nonFiniteCount: number;
};

export const allclose = (
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  tolerance: Tolerance = DEFAULT_TOLERANCE,
): AllcloseReport => {
  if (actual.length !== expected.length) {
    throw new AllcloseError(`長さ不一致: actual ${actual.length} vs expected ${expected.length}`);
  }
  let failCount = 0;
  let maxAbsError = 0;
  let maxRelError = 0;
  let worstIndex = 0;
  let nonFiniteCount = 0;
  for (let i = 0; i < actual.length; i += 1) {
    const x = actual[i];
    const y = expected[i];
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      if (nonFiniteCount === 0) worstIndex = i;
      nonFiniteCount += 1;
      failCount += 1;
      maxAbsError = Number.POSITIVE_INFINITY;
      maxRelError = Number.POSITIVE_INFINITY;
      continue;
    }
    const abs = Math.abs(x - y);
    if (abs > maxAbsError && nonFiniteCount === 0) {
      maxAbsError = abs;
      worstIndex = i;
    }
    maxRelError = Math.max(maxRelError, abs / Math.max(Math.abs(y), 1e-12));
    if (abs > tolerance.atol + tolerance.rtol * Math.abs(y)) {
      failCount += 1;
    }
  }
  return { pass: failCount === 0, failCount, maxAbsError, maxRelError, worstIndex, nonFiniteCount };
};

/** 突合の対象（公開 Tensor と CPU 参照 RefTensor の両方が構造的に当てはまる最小の面）。 */
type NumericTensor = {
  readonly dtype: IrDtype;
  readonly data: ArrayLike<number>;
};

/**
 * dtype に応じて許容誤差を選ぶ突合。f32 は丸め差・縮約順序差が構造的に出るので allclose、
 * i32 / bool は{@link EXACT_TOLERANCE}（厳密一致）。
 *
 * MUST: dtype が食い違ったら比較しない。要素は全型 4 バイトで値も数値として読めてしまうため、
 * 突合が「たまたま通る / たまたま落ちる」になり検証の意味が消える。
 */
export const compareTensors = (
  actual: NumericTensor,
  expected: NumericTensor,
  floatTolerance: Tolerance = DEFAULT_TOLERANCE,
): AllcloseReport => {
  if (actual.dtype !== expected.dtype) {
    throw new AllcloseError(`dtype 不一致: actual ${actual.dtype} vs expected ${expected.dtype}`);
  }
  return allclose(
    actual.data,
    expected.data,
    expected.dtype === "f32" ? floatTolerance : EXACT_TOLERANCE,
  );
};

/** 失敗時のテストメッセージ用（どこがどれだけずれたかを 1 行で出す）。 */
export const formatAllclose = (report: AllcloseReport): string =>
  `fail=${report.failCount} nonFinite=${report.nonFiniteCount} maxAbs=${report.maxAbsError} maxRel=${report.maxRelError} at=${report.worstIndex}`;
