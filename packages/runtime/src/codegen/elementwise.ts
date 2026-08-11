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

/** 出力 dtype は契約表の写像から取る（codegen が独自に決めない）。 */
const outputDtype = (op: string, slotZeroDtype: IrDtype): IrDtype =>
  outputDtypeOf(resolveOpContract(op), slotZeroDtype, "elementwise codegen");

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
 * Abramowitz–Stegun 7.1.26（最大絶対誤差 ≈1.5e-7）。WGSL に erf 組込は無い。
 * gelu は torch 既定の `approximate="none"`（erf 形）を実装するため必要になる。
 * CPU 参照は別式（Chebyshev fit）で erf を出す — 同じ近似式を写すと誤差が相殺して
 * テストが恒真化するため。
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
const IS_NAN_FN = `fn is_nan_bits(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}`;

/**
 * NaN 入力をそのまま返す外殻。中の値式（`finite`）は**非 NaN 経路だけ**を担うので、
 * 既存の式をそのまま包める（= 非 NaN の数値結果がビット単位で動かない）。
 */
const nanGuard = (a: string, finite: string): string =>
  `select(${finite}, ${a}, is_nan_bits(${a}))`;

/**
 * 単項の値式。第 2 引数は params 末尾のスカラ（{@link SCALAR_PARAM_ATTRS} の並び）を指す
 * WGSL 名を返す。
 *
 * MUST: `clamp` / `clamp_min` / `relu` は {@link nanGuard} で包む。torch は 3 つとも
 * NaN を伝播するが、素の式は畳み込みで NaN を飲む（機序は {@link IS_NAN_FN}）。
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
  tanh: (a) => `tanh(${a})`,
  sigmoid: (a) => `sigmoid_stable(${a})`,
  relu: (a) => nanGuard(a, `max(${a}, 0.0)`),
  // 0.7071067811865476 = 1/√2
  gelu: (a) => `0.5 * ${a} * (1.0 + erf_approx(${a} * 0.7071067811865476))`,
  // torch の approximate="tanh"。0.7978845608028654 = √(2/π)。tanh は WGSL 組込なので
  // 補助関数を足さない（erf 形と違い、この式そのものが定義で、近似の精度を上げる余地が無い）。
  // NaN の外殻（{@link nanGuard}）は掛けない — 掛ける理由は `max` / `clamp` イディオムへの
  // 畳み込み（{@link IS_NAN_FN}）だけで、この式にはその形が無い（erf 形の gelu と同じ扱い）。
  gelu_tanh: (a) =>
    `0.5 * ${a} * (1.0 + tanh(0.7978845608028654 * (${a} + 0.044715 * ${a} * ${a} * ${a})))`,
  // bool は u32 の 0 / 1。`1u - x` にすると格納規約が破れたときに 0/1 の外へ出る。
  bitwise_not: (a) => `select(1u, 0u, ${a} != 0u)`,
  clamp: (a, scalar) =>
    nanGuard(
      a,
      `select(select(${a}, ${scalar(0)}, ${a} < ${scalar(0)}), ${scalar(1)}, ${a} > ${scalar(1)})`,
    ),
  // ADR 0017 の裁定どおり select 形（`max(x, m)` は WGSL が NaN 伝播を保証しない）。
  // NOTE: 「2 段 select なら NaN が伝播する」という ADR 0015 / op-vocabulary.md の当初の
  // 見立ては実測で否定された（畳み込みで `max` に化ける — {@link IS_NAN_FN}）。
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
  ["is_nan_bits", IS_NAN_FN],
  ["log1p_series", LOG1P_FN],
  ["sigmoid_stable", SIGMOID_STABLE_WGSL],
];

/**
 * MUST: WGSL に埋まる生成パラメータは全てキーに載せる。workgroup サイズは
 * `@workgroup_size` と grid-stride の stride 式に埋め込まれるため、載せずに定数を変えると
 * 旧サイズでコンパイル済みのパイプラインが同じキーで再利用され、dispatch 数と食い違う。
 * 要素型も同じ理由でキーに載せる（bindings の `array<T>` に埋まる）。
 *
 * NOTE: 載るのは**値スロットの dtype と出力 dtype**だけで足りる。条件スロット（where の
 * bool）とスカラの本数は op 名から一意に決まるため、独立した軸にならない。
 *
 * v2: clamp / clamp_min / relu に NaN 伝播の外殻が入り WGSL 本文が変わった（IS_NAN_FN）。
 * 版は **族ごと**に上げる — op 別の例外表を持つと、次の改版で「どの op が今どの版か」を
 * 二重管理することになる（他の op はキーが変わるだけで生成物は同一）。
 */
export const elementwiseKey = (spec: ElementwiseSpec): string => {
  const { op, rank, from, to } = canonicalize(spec);
  return `ew:v2:${op}:${from}>${to}:r${rank}:wg${ELEMENTWISE_WORKGROUP_SIZE}`;
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
  const used = HELPERS.filter(([name]) => value.includes(name)).map(([, body]) => body);

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
 * params バッファの中身を組み立てる。
 * 出力 shape の rank が生成時の rank と一致していること（呼び出し側の責務）。
 *
 * MUST: スカラは f32 へ丸めて載せる。IR の attrs は JSON の f64 なので、丸めずに扱うと
 * CPU 参照（`Math.fround`）と GPU（params の f32 語）で違う値になりうる。
 */
export const elementwiseParams = (
  outShape: readonly number[],
  inputShapes: readonly (readonly number[])[],
  scalars: readonly number[] = [],
): Uint32Array<ArrayBuffer> => {
  const rank = outShape.length;
  if (rank < 1) throw new CodegenError("elementwise params: 出力 rank は 1 以上");
  const params = new Uint32Array(1 + rank + inputShapes.length * rank + scalars.length);
  params[0] = outShape.reduce((count, dim) => count * dim, 1);
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
    for (let d = 0; d < rank; d += 1) params[1 + rank + k * rank + d] = strides[d];
  });
  const floats = new Float32Array(params.buffer);
  scalars.forEach((value, index) => {
    if (!Number.isFinite(value)) {
      throw new CodegenError(`elementwise params: スカラ attr が有限でない（${value}）`);
    }
    floats[1 + rank + inputShapes.length * rank + index] = value;
  });
  return params;
};
