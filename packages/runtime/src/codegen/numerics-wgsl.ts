/**
 * 数値安定化のために**複数のカーネル族が字面ごと共有する** WGSL 断片の置き場（中立な葉
 * モジュール — codegen/errors.ts・codegen/limits.ts と同じ規律）。
 *
 * MUST: 共有される断片の正本はここ 1 か所（ADR 0020 の NaN 伝播契約は「どのカーネルも同じ
 * 1 本を共有する」ことが成立条件）。書き写すと primitive と融合版で丸め列が割れうる。
 * ここに置くのは elementwise / reduce / attention / GRU / 量子化のどれからも参照されうる
 * 断片だけで、族に閉じた式は各族のモジュールに残す。
 */

/** 整数値も必ず小数リテラルにする（WGSL の AbstractInt に落とさない）。 */
export const f32Literal = (value: number): string =>
  Number.isInteger(value) ? `${value}.0` : `${value}`;

/**
 * MUST: 素朴な `1/(1+exp(-x))` にしない。WGSL は浮動小数のオーバーフロー結果を
 * indeterminate と規定しており、x ≲ -88 で沈黙 NaN になりうる実装が存在する。
 * exp の引数を -|x| に固定すれば結果は常に (0,1] に収まりオーバーフローが構造的に起きない。
 *
 * MUST: SiLU 融合カーネル（src/kernels/silu.ts）はこの本文をそのまま共有する。同じ式を
 * 書き写すと、primitive の sigmoid と融合版で丸め列が割れうる（融合の前提はビット同一）。
 */
export const SIGMOID_STABLE_WGSL = `fn sigmoid_stable(x: f32) -> f32 {
  let t = exp(-abs(x));
  return select(1.0 / (1.0 + t), t / (1.0 + t), x < 0.0);
}`;

/**
 * f32 の NaN を**ビット列**で判定する（符号を落として指数部全 1 + 仮数部非 0）。
 *
 * MUST: 浮動小数の比較で NaN を判定しない。比較単体（`select(0.0, 1.0, NaN < m)`）は仕様
 * どおり false になるのに、**`select(x, m, x < m)` 全体はシェーダコンパイラが `max`
 * イディオムへ畳み、ドライバの `max` が NaN を飲む**（実測・2026-08-02 / 本リポジトリの
 * 検証環境: `clamp_min(NaN, min=0) = 0` / `clamp(NaN, -1, 1) = -1` / `relu(NaN) = 0`）。
 * WGSL には演算の畳み込みを禁じる手段が無いので、**畳み込みの対象にならない整数演算**
 * （`&` と `>`）で判定する。これが {@link nanGuard} を使う唯一の理由。
 */
export const IS_NAN_BITS_WGSL = `fn is_nan_bits(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}`;

/**
 * NaN 伝播する `max`（ADR 0020）。ドライバの `max` は NaN を飲む（WGSL 仕様も「e1 < e2 なら
 * e2、さもなくば e1」で NaN を落とす）ので、縮約で NaN を保存したい側はこれを使う。
 * 非 NaN の 2 引数では素の `max` と同値 — 既存の数値結果は 1 ビットも動かない。
 * 要 {@link IS_NAN_BITS_WGSL}（依存は注入側が並べる）。
 */
export const NAN_MAX_WGSL = `fn nan_max(a: f32, b: f32) -> f32 {
  return select(select(max(a, b), b, is_nan_bits(b)), a, is_nan_bits(a));
}`;

/**
 * NaN 伝播する `min`（{@link NAN_MAX_WGSL} の対）。`amin` の縮約が NaN を保存するために要る。
 * 要 {@link IS_NAN_BITS_WGSL}（依存は注入側が並べる）。
 */
export const NAN_MIN_WGSL = `fn nan_min(a: f32, b: f32) -> f32 {
  return select(select(min(a, b), b, is_nan_bits(b)), a, is_nan_bits(a));
}`;

/**
 * NaN 入力をそのまま返す外殻。中の値式（`finite`）は**非 NaN 経路だけ**を担うので、
 * 既存の式をそのまま包める（= 非 NaN の数値結果がビット単位で動かない）。
 */
export const nanGuard = (a: string, finite: string): string =>
  `select(${finite}, ${a}, is_nan_bits(${a}))`;

/**
 * `tanh` の引数を飽和域で打ち切る閾値。**この値は 2 つの条件の同時成立でしか選べない**
 * （根拠は {@link TANH_STABLE_WGSL} の doc）:
 *
 * 1. f32 の `tanh` がここで既に厳密な ±1.0 に丸まっていること（実測の下限は 9.011）。
 *    下回ると打ち切りが**値を変える** = ビット同一の主張が崩れる。
 * 2. `exp(2·t)` が f32 のオーバーフロー（引数 88.72 = 値 3.40e38）から十分遠いこと。
 *    t = 9.5 なら `exp(19)` = 1.78e8 で 30 桁の余裕がある。
 *
 * 帯 [9.011, 44.36) のほぼ下端寄りを取る — 上へ寄せる利得は無く（打ち切り後の値は
 * どこでも ±1.0）、下端に近いほど条件 2 の余裕が大きい。tests/codegen_wgsl_test.ts が
 * この 2 条件を数値で常設固定する。
 */
export const TANH_SATURATION = 9.5;

/**
 * `tanh` の**飽和打ち切り**版。引数を ±{@link TANH_SATURATION} で頭打ちにしてから組込
 * `tanh` へ渡す。
 *
 * MUST: 素の `tanh(x)` を呼ばない。WGSL は `tanh` の実装を規定しておらず、
 * `(exp(2x) − 1)/(exp(2x) + 1)` で計算する実装（Metal の fast-math 経路 — 実測）は
 * **|x| > 44.36 で exp(2x) が f32 のオーバーフロー**に入り `Inf/Inf` = 沈黙 NaN を返す。
 * `gelu_tanh` の内側引数 `√(2/π)·(x + 0.044715x³)` は前活性 x ≳ 10.05 でそこへ届き、
 * 実モデルの活性（実測最大 11.45）が毎 token 踏む。飽和域を先に潰せば中間の指数が
 * そもそも育たない = オーバーフローが**構造的に**起きない（{@link SIGMOID_STABLE_WGSL}
 * が exp の引数を -|x| に固定しているのと同じ手筋）。
 *
 * MUST: 打ち切りは**非 NaN の値をビット単位で変えない**閾値でだけ許される。f32 の tanh は
 * |x| ≥ 9.011 でちょうど ±1.0 に丸まるので、IEEE 忠実な実装に対しては打ち切りの有無で
 * 結果が 1 ビットも動かない（打ち切られる側は既に ±1.0）。閾値の根拠は
 * {@link TANH_SATURATION}。
 *
 * MUST: NaN 伝播はビット列判定（{@link IS_NAN_BITS_WGSL}）で担い、打ち切りの比較には委ねない。
 * 2 段 select はシェーダコンパイラが `clamp` イディオムへ畳みうるので（機序は IS_NAN_BITS_WGSL の
 * doc）、素朴に書くと `clamp(NaN)` が閾値に化けて **NaN が黙って ±1.0 に飲まれる**。
 *
 * MUST: この本文を書き写さない（{@link SIGMOID_STABLE_WGSL} と同じ理由 — 写すと primitive と
 * 融合版で丸め列が割れうる）。
 *
 * ## 残る懸念（打ち切りを足したことで新たに生じた / 生じなかったもの）
 *
 * 1. 「IEEE 忠実でない tanh 実装が閾値**未満**でも別値を返す」可能性は本修正の射程外。
 *    従来から存在する差で、打ち切りで悪化も改善もしない（閾値未満は式の入力値が不変）。
 * 2. 閾値未満の値はビット同一だが、**式の形が変わったことにシェーダコンパイラが反応して
 *    丸めが動く**理論的余地は残る（WGSL に再結合を禁じる手段が無い以上、原理的に否定
 *    できない）。実測では否定済み — 生成物は WGSL スナップショットで、値は golden 群
 *    （sbv2 の WAV sha256 6 本・gemma4 の系列厳密一致）で固定されており、いずれも
 *    本修正の前後で不変だった。
 * 3. 「中間でオーバーフローするが最終値は有限」という**同じ危険クラス**は横断監査済み
 *    （docs/research/2026-08-31-op-numerics-review.md）。exp 系は全経路が max 減算か -|x|
 *    固定で構造的に安全、softplus は threshold-where 分解が守り、唯一の取り残しだった
 *    src/kernels/gru-scan.ts の組込 `tanh` も本関数の共有で解消した。
 */
export const TANH_STABLE_WGSL = `fn tanh_stable(x: f32) -> f32 {
  let lo = select(x, ${f32Literal(-TANH_SATURATION)}, x < ${f32Literal(-TANH_SATURATION)});
  let t = select(lo, ${f32Literal(TANH_SATURATION)}, x > ${f32Literal(TANH_SATURATION)});
  return ${nanGuard("x", "tanh(t)")};
}`;
