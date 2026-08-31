/**
 * elementwise（unary / binary / where / cast）カーネルの WGSL 生成。
 *
 * MUST: 生成は決定的 — 同一キーからバイト単位で同一の WGSL が出ること。ブラウザの暗黙
 * パイプラインキャッシュのキーは WGSL 文字列そのもので明示的な制御 API が無いため、
 * ゆらぐと同じキーに別カーネルが割り当たる。キーと WGSL は同じ正準化（{@link canonicalize}）
 * から組み立て、生成入力が同じなら両者が必ず同時に決まるようにする。
 * MUST: **要素型はキーに含める**（ADR 0009）。同じ op・同じ rank でも要素型が違えば別カーネル
 * で、載せないと i32 の dispatch に f32 のパイプラインが割り当たり、ビット列の読み替えが
 * 例外なしに通る。
 * MUST: grid-stride ループ前提。1 次元の dispatch 上限（仕様既定 65535）は実モデルで超える。
 *
 * 出力は連続レイアウト、各入力は「右詰め broadcast 済みの stride」で読む（1 の次元は
 * stride 0）。shape は params バッファで渡すので、同じ rank ならパイプラインを使い回せる。
 * attrs のスカラ（clamp の min/max・clamp_min の min・leaky_relu の slope・比較の value）も
 * **params の末尾**で運ぶ — WGSL に焼くと値の種類だけパイプラインが増える
 * （src/kernels/masked-fill.ts と同じ規律）。
 */

import type { IrDtype } from "../format/ir.ts";
import {
  BINARY_OPS,
  type BinaryOpName,
  CAST_OP,
  outputDtypeOf,
  resolveOpContract,
  scalarParamCount,
  slotDtypesOf,
  UNARY_OPS,
  type UnaryOpName,
  WHERE_OP,
} from "../ops.ts";
import { CodegenError } from "./errors.ts";
import { assertU32Params } from "./params.ts";

export const ELEMENTWISE_WORKGROUP_SIZE = 256;

export type ElementwiseSpec =
  | {
    readonly op: UnaryOpName | BinaryOpName | typeof WHERE_OP;
    /** 出力 rank。1 以上（rank 0 のスカラは呼び出し側が rank 1・長さ 1 に正規化する）。 */
    readonly rank: number;
    /**
     * **値スロット**の意味論 dtype。出力 dtype は契約表の写像で決まる（比較 op は bool）。
     * where の条件スロットは常に bool なので、ここに載るのは値の側だけ。
     */
    readonly dtype: IrDtype;
  }
  | {
    readonly op: typeof CAST_OP;
    readonly rank: number;
    /** 変換元の意味論 dtype。 */
    readonly dtype: IrDtype;
    /** 変換先の意味論 dtype。 */
    readonly to: IrDtype;
  };

/**
 * 正準化の結果は kind で判別可能なユニオン。op 名の型は kind で確定するので、
 * 生成側が `op as UnaryOpName` のような取り違えの効かないキャストを書かずに済む
 * （ops.ts の OpContract と同じ方針）。`arity` は入力の個数。
 */
type CanonicalSpec =
  & {
    readonly rank: number;
    /** 値スロットの dtype（キーに載る側）。 */
    readonly from: IrDtype;
    readonly to: IrDtype;
    /** 入力スロットごとの要素型（bindings の `array<T>` に埋まる）。 */
    readonly slots: readonly IrDtype[];
    /** params 末尾に載る f32 スカラの本数。 */
    readonly scalars: number;
  }
  & (
    | { readonly kind: "unary"; readonly op: UnaryOpName; readonly arity: 1 }
    | { readonly kind: "binary"; readonly op: BinaryOpName; readonly arity: 2 }
    | { readonly kind: "cast"; readonly op: typeof CAST_OP; readonly arity: 1 }
    | { readonly kind: "where"; readonly op: typeof WHERE_OP; readonly arity: 3 }
  );

const isUnary = (op: string): op is UnaryOpName => UNARY_OPS.some((name) => name === op);
const isBinary = (op: string): op is BinaryOpName => BINARY_OPS.some((name) => name === op);

/**
 * 生成入力の dtype は契約表で照合する（表を写すと解禁範囲が 2 箇所に割れる）。
 *
 * MUST: 受理集合の**和**ではなくスロット単位で見る。where の和は {bool, f32} なので、
 * 和で見ると条件用の bool が値スロットの生成入力として素通りする。
 */
const assertSlotDtype = (op: string, slot: number, dtype: IrDtype): void => {
  const accept = slotDtypesOf(resolveOpContract(op))[slot];
  if (accept === undefined || !accept.includes(dtype)) {
    throw new CodegenError(
      `elementwise codegen: op '${op}' のスロット ${slot} は dtype '${dtype}' を実行できない`,
    );
  }
};

/** 出力 dtype は契約表の写像から取る（codegen が独自に決めない）— elementwise は単一出力。 */
const outputDtype = (op: string, slotZeroDtype: IrDtype): IrDtype =>
  outputDtypeOf(resolveOpContract(op), 0, slotZeroDtype, "elementwise codegen");

const canonicalize = (spec: ElementwiseSpec): CanonicalSpec => {
  if (!Number.isSafeInteger(spec.rank) || spec.rank < 1) {
    throw new CodegenError(`elementwise codegen: rank は 1 以上の整数（${spec.rank}）`);
  }
  const { op, rank, dtype } = spec;
  if (op === CAST_OP) {
    assertSlotDtype(op, 0, dtype);
    assertSlotDtype(op, 0, spec.to);
    return {
      kind: "cast",
      op,
      rank,
      arity: 1,
      from: dtype,
      to: spec.to,
      slots: [dtype],
      scalars: 0,
    };
  }
  if (op === WHERE_OP) {
    // スロット 0 は条件（bool 固定）、1 / 2 が値。生成入力の dtype は値の側。
    assertSlotDtype(op, 1, dtype);
    return {
      kind: "where",
      op,
      rank,
      arity: 3,
      from: dtype,
      to: outputDtype(op, "bool"),
      slots: ["bool", dtype, dtype],
      scalars: 0,
    };
  }
  if (isUnary(op)) {
    assertSlotDtype(op, 0, dtype);
    return {
      kind: "unary",
      op,
      rank,
      arity: 1,
      from: dtype,
      to: outputDtype(op, dtype),
      slots: [dtype],
      scalars: scalarParamCount(op),
    };
  }
  if (isBinary(op)) {
    assertSlotDtype(op, 0, dtype);
    return {
      kind: "binary",
      op,
      rank,
      arity: 2,
      from: dtype,
      to: outputDtype(op, dtype),
      slots: [dtype, dtype],
      scalars: scalarParamCount(op),
    };
  }
  throw new CodegenError(`elementwise codegen: op '${op}' は elementwise 語彙に無い`);
};

/** 意味論 dtype → WGSL のスカラ型。bool の格納は u32 の 0 / 1（ADR 0009）。 */
const WGSL_SCALAR: Readonly<Record<IrDtype, string>> = {
  f32: "f32",
  i32: "i32",
  bool: "u32",
};

/** dtype ごとのゼロ値リテラル（`!= 0` の真偽化で使う）。 */
const WGSL_ZERO: Readonly<Record<IrDtype, string>> = {
  f32: "0.0",
  i32: "0",
  bool: "0u",
};

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
 * Abramowitz–Stegun 7.1.26。WGSL に erf 組込は無い。
 * gelu は torch 既定の `approximate="none"`（erf 形）を実装するため必要になる。
 * CPU 参照は別式（Chebyshev fit）で erf を出す — 同じ近似式を写すと誤差が相殺して
 * テストが恒真化するため。
 *
 * 誤差の勘定（3 層 — 混同しない）: ①近似式そのものの理論値 ≈1.4e-7（倍精度評価・
 * x≈0.045 で最大）②この f32 Horner + exp を f32 逐次丸めで評価した実測 ≈5.3e-7
 * （x≈0.035 で最大 — CPU の f32 エミュ掃引 2026-08-31）③WGSL 仕様の exp は 3+2|x| ULP まで
 * 許すので、仕様準拠実装の上界はさらに上（a→0 で寄与 ≈3.6e-7）。「1.5e-7」を shader の
 * 保証値として引用しない。gelu 全体では最大絶対 ≈6.8e-7（|gelu|≈4.2 で ≈1.4 ULP）だが、
 * 負側テール（x≈−3.4）は桁落ちで**相対**誤差 ≈2.2e-4 — op 別 tolerance を相対帯で切ると
 * ここが偽陽性になる。
 */
const ERF_FN = `fn erf_approx(x: f32) -> f32 {
  let s = sign(x);
  let a = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  var p = 1.061405429;
  p = -1.453152027 + p * t;
  p = 1.421413741 + p * t;
  p = -0.284496736 + p * t;
  p = 0.254829592 + p * t;
  return s * (1.0 - p * t * exp(-a * a));
}`;

/** 級数に切り替える範囲（`|x| < LOG1P_SERIES_THRESHOLD`）。根拠は {@link LOG1P_FN} の doc。 */
const LOG1P_SERIES_THRESHOLD = 0.25;
/** マクローリン級数の項数（`x^1 … x^n`）。 */
const LOG1P_SERIES_TERMS = 10;

/** 整数値も必ず小数リテラルにする（WGSL の AbstractInt に落とさない）。 */
const f32Literal = (value: number): string => Number.isInteger(value) ? `${value}.0` : `${value}`;

/**
 * `log1p` のマクローリン級数を Horner 形 `x·(1 − x·(1/2 − x·(… ∓ x/n)))` で組み立てる。
 * 係数は `1/k` そのものなので、項数を変えれば式全体が決定的に決まる（手書きの係数表を
 * 持たない = 打ち間違いが構造的に起きない）。
 */
const log1pSeries = (): string => {
  let expr = f32Literal(1 / LOG1P_SERIES_TERMS);
  for (let k = LOG1P_SERIES_TERMS - 1; k >= 1; k -= 1) {
    expr = `${f32Literal(1 / k)} - x * (${expr})`;
  }
  return `x * (${expr})`;
};

/**
 * log1p（WGSL に組込が無い）。**|x| が小さい側は級数、大きい側は `log(1+x)`**。
 *
 * MUST: 全域を `log(1 + x)` で済ませない。f32 では x = 1e-8 の `1 + x` が厳密に 1.0 になり
 * 答えが 0（相対誤差 1.0）になるうえ、WGSL の `log` は仕様上 [0.5, 2.0] の範囲で
 * **絶対**誤差までしか保証しない（相対誤差ではない）ため、x が小さいほど当てにならない。
 * 実測（本リポジトリの検証環境）でも `log(1 + 1e-7)` は真値の約 1.94 倍だった。
 *
 * MUST: `u = fl(1+x)` の丸め残差を `d = u − 1` で取り出す古典的な補正
 * （`log(u)·x/d`）を使わない。**シェーダコンパイラが `(1.0 + x) - 1.0` を代数的に `x` へ
 * 畳む**ことを実測で確認しており（d が常に x と一致した）、補正項が恒等 1 になって素朴形に
 * 退化する。WGSL には演算の再結合を禁じる手段が無いので、この手筋は表現できない。
 *
 * 打ち切り誤差の上界: 項数 n・|x| ≤ a で相対誤差 ≈ `a^n/(n+1)`。a = 0.25 / n = 10 なら
 * 8.7e-8 で f32 の eps（1.19e-7）より下。切り替え点より上は `log` の精度がそのまま出る
 * （a を上げると級数の項数が要り、下げると `log` 側の相対誤差が増える — 両者が釣り合う
 * あたりを取っている）。**実測の最悪相対誤差は 2.7e-7（x = 0.2501、log 側）**で、素朴形の
 * 1.0（x = 1e-8）とは 6 桁以上離れている。乖離は tests/gpu_ops_test.ts が CPU 参照
 * （`Math.log1p`）との突合で常設固定する。
 *
 * NOTE: 定義域は x > −1（softplus 分解形の実引数 `exp(scaled)` は常に正）。x < −1 は
 * `log` の負引数で WGSL 的に indeterminate — torch も NaN なので黙って近似はしない。
 */
const LOG1P_FN = `fn log1p_series(x: f32) -> f32 {
  // 係数は 1/k（Horner 形）。切り替え点より外は log(1+x) をそのまま使う。
  let series = ${log1pSeries()};
  return select(log(1.0 + x), series, abs(x) < ${f32Literal(LOG1P_SERIES_THRESHOLD)});
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
 * NaN 入力をそのまま返す外殻。中の値式（`finite`）は**非 NaN 経路だけ**を担うので、
 * 既存の式をそのまま包める（= 非 NaN の数値結果がビット単位で動かない）。
 */
const nanGuard = (a: string, finite: string): string =>
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

/**
 * 単項の値式。第 2 引数は params 末尾のスカラ（{@link SCALAR_PARAM_ATTRS} の並び）を指す
 * WGSL 名を返す。
 *
 * MUST: `clamp` / `clamp_min` / `relu` は {@link nanGuard} で包む。torch は 3 つとも
 * NaN を伝播するが、素の式は畳み込みで NaN を飲む（機序は {@link IS_NAN_BITS_WGSL}）。
 * MUST: `clamp` は組込の `clamp(x, lo, hi)` を使わない。`lo > hi` の結果が indeterminate
 * である（契約側で min <= max を拒否してはいる）。
 * MUST: `leaky_relu` は `max` / `min` を使わず **select 形**（ADR 0015）。`relu(x) + s·min(x, 0)`
 * のような書き方は `min` が NaN を飲んで伝播が壊れる。select 形は比較が false へ落ちて
 * `s · NaN = NaN` を返すため、**両方の枝に x が現れる**この形だけは畳まれても NaN が残る
 * （tests/gpu_ops_test.ts が実 GPU で固定）。
 */
const UNARY_WGSL: Readonly<
  Record<UnaryOpName, (a: string, scalar: (index: number) => string) => string>
> = {
  neg: (a) => `-${a}`,
  abs: (a) => `abs(${a})`,
  exp: (a) => `exp(${a})`,
  log: (a) => `log(${a})`,
  log1p: (a) => `log1p_series(${a})`,
  sqrt: (a) => `sqrt(${a})`,
  sin: (a) => `sin(${a})`,
  // MUST: 組込 `tanh` の素通しにしない（機序と閾値は {@link TANH_STABLE_WGSL}）。この op は
  // logits softcap（`div` → `tanh` → `mul`）の内側でも踏むので、gelu_tanh と同じ穴が空く。
  tanh: (a) => `tanh_stable(${a})`,
  sigmoid: (a) => `sigmoid_stable(${a})`,
  relu: (a) => nanGuard(a, `max(${a}, 0.0)`),
  // 0.7071067811865476 = 1/√2
  gelu: (a) => `0.5 * ${a} * (1.0 + erf_approx(${a} * 0.7071067811865476))`,
  // torch の approximate="tanh"。0.7978845608028654 = √(2/π)。式そのものが定義なので
  // （erf 形と違い）近似の精度を上げる余地は無い。
  // MUST: 内側は {@link TANH_STABLE_WGSL} を通す。素の `tanh` だと 3 次項が効いて
  // 前活性 x ≳ 10.05 で内側引数が 44.36 を超え、exp 経由実装が沈黙 NaN を返す。
  // NaN の外殻（{@link nanGuard}）は式全体には掛けない — 外側に因子 `x` が残るので
  // `0.5 · NaN · (…)` が NaN のままで、伝播は畳み込みの有無に依らない（erf 形の gelu と同じ扱い）。
  gelu_tanh: (a) =>
    `0.5 * ${a} * (1.0 + tanh_stable(0.7978845608028654 * (${a} + 0.044715 * ${a} * ${a} * ${a})))`,
  // bool は u32 の 0 / 1。`1u - x` にすると格納規約が破れたときに 0/1 の外へ出る。
  bitwise_not: (a) => `select(1u, 0u, ${a} != 0u)`,
  clamp: (a, scalar) =>
    nanGuard(
      a,
      `select(select(${a}, ${scalar(0)}, ${a} < ${scalar(0)}), ${scalar(1)}, ${a} > ${scalar(1)})`,
    ),
  // ADR 0017 の裁定どおり select 形（`max(x, m)` は WGSL が NaN 伝播を保証しない）。
  // NOTE: 「2 段 select なら NaN が伝播する」という ADR 0015 / op-vocabulary.md の当初の
  // 見立ては実測で否定された（畳み込みで `max` に化ける — {@link IS_NAN_BITS_WGSL}）。
  // 伝播を担うのは select の形ではなく、外殻のビット列判定だけ。
  clamp_min: (a, scalar) => nanGuard(a, `select(${a}, ${scalar(0)}, ${a} < ${scalar(0)})`),
  leaky_relu: (a, scalar) => `select(${scalar(0)} * ${a}, ${a}, ${a} >= 0.0)`,
  ge_scalar: (a, scalar) => `select(0u, 1u, ${a} >= ${scalar(0)})`,
  le_scalar: (a, scalar) => `select(0u, 1u, ${a} <= ${scalar(0)})`,
  gt_scalar: (a, scalar) => `select(0u, 1u, ${a} > ${scalar(0)})`,
};

const BINARY_WGSL: Readonly<Record<BinaryOpName, (a: string, b: string) => string>> = {
  add: (a, b) => `${a} + ${b}`,
  sub: (a, b) => `${a} - ${b}`,
  mul: (a, b) => `${a} * ${b}`,
  div: (a, b) => `${a} / ${b}`,
  ge: (a, b) => `select(0u, 1u, ${a} >= ${b})`,
  // 格納規約（0 / 1）が破れても真偽の論理積になる形。`a & b` はビット演算なのでそうならない。
  bitwise_and: (a, b) => `select(0u, 1u, (${a} != 0u) && (${b} != 0u))`,
};

/**
 * cast の値式。bool への変換だけは `x != 0` の真偽化で、それ以外は WGSL の型変換
 * （`i32(f32)` は仕様上 0 方向切り捨て = torch の truncate と一致）。
 */
const castWgsl = (a: string, from: IrDtype, to: IrDtype): string =>
  to === "bool" ? `select(0u, 1u, ${a} != ${WGSL_ZERO[from]})` : `${WGSL_SCALAR[to]}(${a})`;

/** 補助関数の注入順は固定（式の出現順にしない）— 同一キー → バイト同一 WGSL のため。 */
const HELPERS: readonly (readonly [string, string])[] = [
  ["erf_approx", ERF_FN],
  ["is_nan_bits", IS_NAN_BITS_WGSL],
  ["log1p_series", LOG1P_FN],
  ["sigmoid_stable", SIGMOID_STABLE_WGSL],
  ["tanh_stable", TANH_STABLE_WGSL],
];

/**
 * 値式が呼ぶ補助関数を、**補助関数どうしの依存も閉じて**選ぶ（`tanh_stable` が
 * `is_nan_bits` を呼ぶ）。
 *
 * MUST: 依存表を手で持たない。本文の字面が正本で、別に持つ表は被参照側を足したり外したり
 * するたびに黙って腐る。値式の字面だけで絞ると、呼ばれている側が未定義のまま残った WGSL が
 * 出る（コンパイル時に落ちるので沈黙誤値にはならないが、生成の入口で塞ぐ）。
 * 返す順は {@link HELPERS} の並びのままで、選び方は順序に影響しない。
 */
const usedHelpers = (value: string): readonly string[] => {
  const selected = new Set<string>();
  let scope = value;
  for (;;) {
    const added = HELPERS.filter(([name]) => !selected.has(name) && scope.includes(name));
    if (added.length === 0) {
      return HELPERS.filter(([name]) => selected.has(name)).map(([, body]) => body);
    }
    for (const [name, body] of added) {
      selected.add(name);
      scope = `${scope}\n${body}`;
    }
  }
};

/**
 * MUST: WGSL に埋まる生成パラメータは全てキーに載せる。workgroup サイズは
 * `@workgroup_size` と grid-stride の stride 式に埋め込まれるため、載せずに定数を変えると
 * 旧サイズでコンパイル済みのパイプラインが同じキーで再利用され、dispatch 数と食い違う。
 * 要素型も同じ理由でキーに載せる（bindings の `array<T>` に埋まる）。
 *
 * NOTE: 載るのは**値スロットの dtype と出力 dtype**だけで足りる。条件スロット（where の
 * bool）とスカラの本数は op 名から一意に決まるため、独立した軸にならない。
 *
 * v2: clamp / clamp_min / relu に NaN 伝播の外殻が入り WGSL 本文が変わった（IS_NAN_BITS_WGSL）。
 * v3: tanh / gelu_tanh が飽和打ち切りを通るようになった（{@link TANH_STABLE_WGSL}）。
 * 版は **族ごと**に上げる（ADR 0020 決定 3）— op 別の例外表を持つと、次の改版で「どの op が
 * 今どの版か」を二重管理することになる（他の op はキーが変わるだけで生成物は同一）。
 */
export const elementwiseKey = (spec: ElementwiseSpec): string => {
  const { op, rank, from, to } = canonicalize(spec);
  return `ew:v3:${op}:${from}>${to}:r${rank}:wg${ELEMENTWISE_WORKGROUP_SIZE}`;
};

/**
 * params のレイアウト（storage, u32 配列）:
 * `[0]=要素数 n, [1..rank]=出力 dims, 続けて入力ごとに rank 本の stride, 末尾に
 * スカラ attr を f32 のビット列で（{@link SCALAR_PARAM_ATTRS} の並び）`。
 * uniform ではなく storage で渡すのは、この族に workgroupBarrier が無く
 * 一様性解析の制約を受けないため（barrier を持つ reduce / matmul は uniform を使う）。
 */
export const elementwiseWgsl = (spec: ElementwiseSpec): string => {
  const canonical = canonicalize(spec);
  const { op, rank, arity, from, to, slots, scalars } = canonical;
  const strideBase = (input: number): number => 1 + rank + input * rank;
  const scalarAt = (index: number): number => 1 + rank + arity * rank + index;

  const bindings = [`@group(0) @binding(0) var<storage, read> params: array<u32>;`];
  for (let k = 0; k < arity; k += 1) {
    bindings.push(
      `@group(0) @binding(${k + 1}) var<storage, read> in${k}: array<${WGSL_SCALAR[slots[k]]}>;`,
    );
  }
  bindings.push(
    `@group(0) @binding(${arity + 1}) var<storage, read_write> out: array<${WGSL_SCALAR[to]}>;`,
  );

  const decode: string[] = ["    var rem = i;"];
  for (let d = rank - 1; d >= 1; d -= 1) {
    decode.push(`    let c${d} = rem % params[${1 + d}u]; rem = rem / params[${1 + d}u];`);
  }
  decode.push("    let c0 = rem;");

  const loads: string[] = [];
  for (let k = 0; k < arity; k += 1) {
    const terms = Array.from(
      { length: rank },
      (_, d) => `c${d} * params[${strideBase(k) + d}u]`,
    );
    loads.push(`    let v${k} = in${k}[${terms.join(" + ")}];`);
  }

  // スカラは要素ごとに読み直さず、ループの外で 1 度だけ名前に束ねる。
  const constants = Array.from(
    { length: scalars },
    (_, index) => `  let s${index} = bitcast<f32>(params[${scalarAt(index)}u]);`,
  );
  const scalarName = (index: number): string => `s${index}`;

  const value = canonical.kind === "cast"
    ? castWgsl("v0", from, to)
    : canonical.kind === "unary"
    ? UNARY_WGSL[canonical.op]("v0", scalarName)
    : canonical.kind === "binary"
    ? BINARY_WGSL[canonical.op]("v0", "v1")
    // torch の where(cond, a, b) = cond ? a : b。WGSL の select(f, t, cond) は引数順が逆。
    : `select(v2, v1, v0 != 0u)`;
  const used = usedHelpers(value);

  return [
    `// karume elementwise ${op} (rank ${rank}, generated)`,
    ...bindings,
    "",
    ...used.flatMap((fn) => [fn, ""]),
    `@compute @workgroup_size(${ELEMENTWISE_WORKGROUP_SIZE})`,
    "fn main(",
    "  @builtin(global_invocation_id) gid: vec3<u32>,",
    "  @builtin(num_workgroups) nwg: vec3<u32>,",
    ") {",
    "  let n = params[0u];",
    ...constants,
    `  let stride = nwg.x * ${ELEMENTWISE_WORKGROUP_SIZE}u;`,
    "  var i = gid.x;",
    "  while (i < n) {",
    ...decode,
    ...loads,
    `    out[i] = ${value};`,
    "    i = i + stride;",
    "  }",
    "}",
    "",
  ].join("\n");
};

/** 右詰めで rank に揃えた shape（不足ぶんは 1 で埋める）。 */
const padShape = (shape: readonly number[], rank: number): number[] => {
  if (shape.length > rank) {
    throw new CodegenError(
      `elementwise params: 入力 rank ${shape.length} が出力 rank ${rank} を超える`,
    );
  }
  return [...Array<number>(rank - shape.length).fill(1), ...shape];
};

/**
 * params バッファの中身を組み立てる。**生成に使った spec を渡す**のが条件で、rank / 入力の
 * 本数 / スカラの本数の 3 点を同じ正準化（{@link canonicalize}）と照合する。
 *
 * MUST: 3 点照合を外さない。スカラ attr の読み出し位置は WGSL 側が**生成時の `arity`** から
 * `1 + rank + arity·rank + index` で焼くので、ここで書く位置（入力の本数から導く）と食い違うと
 * params 配列の範囲外を読む — WebGPU の境界付きアクセスは例外を出さず、
 * `clamp_min(x, 1e-12)` が別の値の clamp に化けるような沈黙誤値になる。
 * MUST: スカラは f32 へ丸めて載せる。IR の attrs は JSON の f64 なので、丸めずに扱うと
 * CPU 参照（`Math.fround`）と GPU（params の f32 語）で違う値になりうる。
 */
export const elementwiseParams = (
  spec: ElementwiseSpec,
  outShape: readonly number[],
  inputShapes: readonly (readonly number[])[],
  scalars: readonly number[] = [],
): Uint32Array<ArrayBuffer> => {
  const { rank, arity, scalars: scalarCount } = canonicalize(spec);
  if (outShape.length !== rank) {
    throw new CodegenError(
      `elementwise params: 出力 rank ${outShape.length} が生成時の rank ${rank} と違う`,
    );
  }
  if (inputShapes.length !== arity) {
    throw new CodegenError(
      `elementwise params: 入力 ${inputShapes.length} 本が op '${spec.op}' のアリティ ${arity} と違う`,
    );
  }
  if (scalars.length !== scalarCount) {
    throw new CodegenError(
      `elementwise params: スカラ ${scalars.length} 本が op '${spec.op}' の ${scalarCount} 本と違う`,
    );
  }
  const n = outShape.reduce((count, dim) => count * dim, 1);
  assertU32Params("elementwise params", {
    ...Object.fromEntries(outShape.map((dim, d) => [`out_dims[${d}]`, dim])),
    n,
  });
  const params = new Uint32Array(1 + rank + inputShapes.length * rank + scalars.length);
  params[0] = n;
  for (let d = 0; d < rank; d += 1) params[1 + d] = outShape[d];
  inputShapes.forEach((shape, k) => {
    const padded = padShape(shape, rank);
    // 連続 stride を右から積む。1 の次元は stride 0 にして broadcast を吸収する。
    let running = 1;
    const strides = new Array<number>(rank).fill(0);
    for (let d = rank - 1; d >= 0; d -= 1) {
      strides[d] = padded[d] === 1 ? 0 : running;
      running *= padded[d];
    }
    // 長さ 0 の軸を挟むと n が 0 でも左側の stride だけ u32 を超えうる（積は 0 に潰れない）。
    assertU32Params(
      "elementwise params",
      Object.fromEntries(strides.map((stride, d) => [`in${k}_strides[${d}]`, stride])),
    );
    for (let d = 0; d < rank; d += 1) params[1 + rank + k * rank + d] = strides[d];
  });
  const floats = new Float32Array(params.buffer);
  scalars.forEach((value, index) => {
    // MUST: 有限判定は **f32 として**行う（載せ先が f32 語なので、f64 で有限な `1e39` は
    // 黙って `+Inf` に化ける — 契約層 `assertFiniteAttr` と同じ門）。
    if (!Number.isFinite(Math.fround(value))) {
      throw new CodegenError(`elementwise params: スカラ attr が f32 として有限でない（${value}）`);
    }
    floats[1 + rank + inputShapes.length * rank + index] = value;
  });
  return params;
};
