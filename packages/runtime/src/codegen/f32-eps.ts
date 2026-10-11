// 正規化の eps の f32 値域（op 契約〈ops/attrs.ts〉と正規化カーネルの params が共有する唯一の判定）。
//
// 同じ条件を attrs と params が別々に持つと、片側だけ直される（subnormal を params だけが通して
// いた前例がある）ので、判定はここ 1 箇所に置く。例外の種類と文言は呼び手が決める。

/** f32 の最小正規数（2^-126）。 */
const F32_MIN_NORMAL = 2 ** -126;

/**
 * eps が f32 として使えない理由。
 *
 * - `not-positive`: 有限の正数でない（0・負・NaN・±Inf）
 * - `subnormal`: f32 へ丸めると 0 か subnormal になる。FTZ（subnormal を 0 として扱う — WGSL が
 *   許す）の GPU では eps = 0 で走り、全ゼロ行がゼロ除算になる
 * - `overflow`: f32 へ丸めると無限大になる（GPU 側だけ `1/sqrt(… + Inf) = 0` で bias 一色）
 */
export type F32EpsDefect = "not-positive" | "subnormal" | "overflow";

/** eps が「有限かつ f32 の正規数に丸まる」なら `undefined`、そうでなければ理由を返す。 */
export const f32EpsDefect = (value: number): F32EpsDefect | undefined => {
  if (!Number.isFinite(value) || value <= 0) return "not-positive";
  const rounded = Math.fround(value);
  if (rounded < F32_MIN_NORMAL) return "subnormal";
  if (!Number.isFinite(rounded)) return "overflow";
  return undefined;
};
