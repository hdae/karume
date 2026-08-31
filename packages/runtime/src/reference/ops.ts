/**
 * op の CPU 参照実装（素朴・GPU 非依存）。カーネルの数値検証（ADR 0005 の段 2）で
 * GPU 出力と突合するオラクル。
 *
 * 方針: 速度を一切考えない直訳で書く。f32 の elementwise は演算ごとに `Math.fround` して
 * f32 の丸めに寄せ、縮約（matmul / 行 reduce）は f64 で積んで格納時に 1 度だけ丸める
 * （GPU 側は f32 逐次累積なので完全一致はしない — だから allclose で判定する）。
 * i32 / bool は近似の余地が無いので**厳密一致**が期待値（突合側で allclose を使わない）。
 *
 * MUST: 座標→添字の計算を codegen と共有しない。stride の組み立てはまさに検証したい対象で、
 * 同じ実装を呼べばテストは stride バグを検出できなくなる（意図的な二重実装）。
 * MUST: 値式も codegen と別形にする（cast の truncate は WGSL の `i32()` に対し `Math.trunc`、
 * bool 化は `select(…)` に対し比較演算）。同じ式を写すと誤りが両側で相殺する。
 */

import type { IrDtype } from "../format/ir.ts";
import {
  ARGMAX_OP,
  arityFits,
  assertDtype,
  assertSlotDtype,
  attentionScale,
  type BinaryOpName,
  castTargetDtype,
  catDim,
  computeOutputShape,
  conv1dAttrs,
  conv2dAttrs,
  convTranspose1dAttrs,
  deformConv2dAttrs,
  describeArity,
  flipDim,
  type GruScanOpName,
  layerNormAttrs,
  maskedFillValue,
  numel,
  type OpContract,
  outputDtypeOf,
  padAttrs,
  permuteDims,
  reduceDim,
  type ReduceOpName,
  resolveOpContract,
  rmsNormEps,
  scalarParamValues,
  sliceAttrs,
  TOPK_OP,
  topkK,
  type UnaryOpName,
  upsampleBilinear2dAttrs,
} from "../ops.ts";

/** 参照実装に渡された値が契約に合わない（要素数と shape の食い違いなど）。 */
export class ReferenceOpError extends Error {
  override readonly name = "ReferenceOpError";
}

type RefTensorOf<D extends IrDtype, A> = {
  readonly dtype: D;
  readonly shape: readonly number[];
  readonly data: A;
};

/** 公開 Tensor と同じ dtype 判別ユニオン（bool は u32 の 0 / 1 — ADR 0009）。 */
export type RefTensor =
  | RefTensorOf<"f32", Float32Array<ArrayBuffer>>
  | RefTensorOf<"i32", Int32Array<ArrayBuffer>>
  | RefTensorOf<"bool", Uint32Array<ArrayBuffer>>;

/** 渡された TypedArray から dtype を決める（配列型と判別子の対応は 1 箇所に置く）。 */
export const refTensor = (
  shape: readonly number[],
  data: Float32Array<ArrayBuffer> | Int32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>,
): RefTensor => {
  if (data.length !== numel(shape)) {
    throw new ReferenceOpError(`要素数 ${data.length} が shape [${shape.join(",")}] と合わない`);
  }
  if (data instanceof Float32Array) return { dtype: "f32", shape, data };
  if (data instanceof Int32Array) return { dtype: "i32", shape, data };
  return { dtype: "bool", shape, data };
};

/**
 * 数値列を dtype ごとの TypedArray に落とす。i32 は `Int32Array` の格納そのものが
 * 2 の補数ラップになるので、GPU 側（WGSL の i32 演算）と同じ折り返しになる。
 */
const materialize = (
  dtype: IrDtype,
  shape: readonly number[],
  values: readonly number[],
): RefTensor => {
  switch (dtype) {
    case "f32":
      return { dtype, shape, data: Float32Array.from(values) };
    case "i32":
      return { dtype, shape, data: Int32Array.from(values) };
    case "bool":
      return { dtype, shape, data: Uint32Array.from(values) };
  }
};

/**
 * erf のマクローリン級数（f64 で収束するまで加算）。|x| > 4 では erfc(x) < 1.6e-8 で
 * f32 の 1.0 と区別できないため ±1 に丸める。
 *
 * MUST: GPU 側（Abramowitz–Stegun 7.1.26）と同じ式を写さない。同じ近似式なら誤差が相殺し、
 * gelu の突合テストが恒真になる。
 */
const erf = (x: number): number => {
  const a = Math.abs(x);
  if (a > 4) return Math.sign(x);
  let term = a;
  let sum = a;
  for (let n = 1; n < 200; n += 1) {
    term *= -(a * a) / n;
    const contribution = term / (2 * n + 1);
    sum += contribution;
    if (Math.abs(contribution) < 1e-18 * Math.abs(sum)) break;
  }
  return Math.sign(x) * (2 / Math.sqrt(Math.PI)) * sum;
};

/**
 * f32 → f32 の unary（bitwise_not は bool 専業、比較 3 本は bool を返すので別扱い）。
 * 第 2 引数は attrs 由来のスカラ（{@link scalarParamValues} の並び）。
 *
 * MUST: `log1p` は `Math.log1p` を使う（GPU 側の補正式を写さない）。同じ式を書くと
 * 「補正が外れている」ことを突合が検出できなくなる — erf / softmax と同じ規律。
 * MUST: `leaky_relu` / `clamp` も GPU 側の select 形を写さず、比較の向きを逆にした
 * 分岐で書く。NaN では全ての比較が false になるので、どちらの形でも NaN は素通りする
 * （= torch と同じ伝播）。
 */
const UNARY_F32: Readonly<
  Record<
    Exclude<UnaryOpName, "bitwise_not" | "ge_scalar" | "le_scalar" | "gt_scalar">,
    (x: number, scalars: readonly number[]) => number
  >
> = {
  neg: (x) => -x,
  abs: (x) => Math.abs(x),
  exp: (x) => Math.exp(x),
  log: (x) => Math.log(x),
  log1p: (x) => Math.log1p(x),
  sqrt: (x) => Math.sqrt(x),
  // sqrt / tanh と同じく組込どうしで突合する（log1p / erf のように「片側が近似式」では
  // ないので、写し間違いを検出する別式が存在しない）。NOTE: WGSL が `sin` の精度を保証
  // するのは |x| ≤ π の範囲だけで、外側の一致は実装依存 — 突合は allclose で判定する。
  sin: (x) => Math.sin(x),
  tanh: (x) => Math.tanh(x),
  sigmoid: (x) => 1 / (1 + Math.exp(-x)),
  relu: (x) => Math.max(x, 0),
  // torch 既定の gelu（approximate="none"）= 0.5·x·(1 + erf(x/√2))
  gelu: (x) => 0.5 * x * (1 + erf(x / Math.SQRT2)),
  // torch の approximate="tanh"。**ここは GPU 側と同じ式で書く** — erf / log1p と違い、この
  // 式自体が torch の定義であって近似ではないので、別式にすると「別の関数」を突合すること
  // になる（検出したいのは近似の外れではなく実装の取り違え）。0.7978845608028654 = √(2/π)。
  gelu_tanh: (x) => 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))),
  clamp: (x, [min, max]) => (x < min ? min : x > max ? max : x),
  // MUST: 分岐の向きを `x >= min ? x : min` にしない — NaN で false へ落ちて min を返し、
  // torch の `clamp(NaN, min) = NaN` から静かに外れる（この 1 本が向きの検出器）。
  // GPU 側は NaN 伝播をビット列判定の外殻で担う（src/codegen/elementwise.ts）— 参照は
  // その式を写さず独立に書く（log1p / erf と同じ規律）。
  clamp_min: (x, [min]) => (x < min ? min : x),
  leaky_relu: (x, [slope]) => (x < 0 ? slope * x : x),
};

/**
 * {@link UNARY_F32} を attrs スカラ無しで引くときの空表（表は clamp 族と共有しているので
 * 第 2 引数が要る — 呼び分けのために式を写さない）。
 */
const NO_SCALARS: readonly number[] = [];

/** f32 → bool の比較（attrs のスカラと比べる）。bool は u32 の 0 / 1（ADR 0009）。 */
const COMPARE_SCALAR: Readonly<
  Record<"ge_scalar" | "le_scalar" | "gt_scalar", (x: number, value: number) => boolean>
> = {
  ge_scalar: (x, value) => x >= value,
  le_scalar: (x, value) => x <= value,
  gt_scalar: (x, value) => x > value,
};

/** f32 の算術（比較 / 論理は別表 — 出力 dtype が違う）。 */
const BINARY_F32: Readonly<Partial<Record<BinaryOpName, (a: number, b: number) => number>>> = {
  add: (a, b) => a + b,
  sub: (a, b) => a - b,
  mul: (a, b) => a * b,
  div: (a, b) => a / b,
};

/**
 * i32 の二項演算。JS の数値は f64 なので、32bit へ畳む演算を明示する
 * （`Math.imul` / `| 0`）— 素の乗算は 2^53 を超えた時点で静かに精度を失う。
 * 解禁しているのは mul / sub のみ（契約表 — 実測グラフに出る形だけ）。
 */
const BINARY_I32: Readonly<Partial<Record<BinaryOpName, (a: number, b: number) => number>>> = {
  sub: (a, b) => (a - b) | 0,
  mul: (a, b) => Math.imul(a, b),
};

/** f32 × f32 → bool の比較（出力は u32 の 0 / 1 — ADR 0009）。 */
const COMPARE_F32: Readonly<Partial<Record<BinaryOpName, (a: number, b: number) => boolean>>> = {
  ge: (a, b) => a >= b,
};

/**
 * bool × bool → bool の論理演算。
 * MUST: GPU 側（`select` + `!= 0u`）と別形にする — JS の真偽値へ落としてから素の論理演算で
 * 書き、数値化は最後に 1 度だけ行う。
 */
const BINARY_BOOL: Readonly<Partial<Record<BinaryOpName, (a: boolean, b: boolean) => boolean>>> = {
  bitwise_and: (a, b) => a && b,
};

/** dtype ごとの二項演算子（無い組み合わせは契約検査を通っていても参照実装が無い = 内部矛盾）。 */
const binaryEvaluator = (
  op: BinaryOpName,
  dtype: IrDtype,
): (a: number, b: number) => number => {
  if (dtype === "bool") {
    const logical = BINARY_BOOL[op];
    if (logical !== undefined) return (a, b) => (logical(a !== 0, b !== 0) ? 1 : 0);
  } else if (dtype === "i32") {
    const integer = BINARY_I32[op];
    if (integer !== undefined) return integer;
  } else {
    const compare = COMPARE_F32[op];
    if (compare !== undefined) return (a, b) => (compare(a, b) ? 1 : 0);
    const arithmetic = BINARY_F32[op];
    if (arithmetic !== undefined) return (a, b) => Math.fround(arithmetic(a, b));
  }
  throw new ReferenceOpError(`op '${op}' に dtype '${dtype}' の参照実装が無い`);
};

export const referenceUnary = (
  op: UnaryOpName,
  x: RefTensor,
  attrs: Readonly<Record<string, unknown>> = {},
): RefTensor => {
  const contract = resolveOpContract(op);
  assertDtype(contract, x.dtype, "reference");
  const values = new Array<number>(x.data.length);
  if (op === "bitwise_not") {
    // bool（u32 の 0 / 1）の否定。WGSL 側の select とは別形（比較の結果を数値化する）。
    for (let i = 0; i < values.length; i += 1) values[i] = x.data[i] === 0 ? 1 : 0;
    return materialize(x.dtype, [...x.shape], values);
  }
  // MUST: スカラ attr は f32 へ丸めてから使う（GPU 側は params の f32 語で受け取る）。
  const scalars = scalarParamValues(contract, attrs, "reference").map((value) =>
    Math.fround(value)
  );
  const outDtype = outputDtypeOf(contract, 0, x.dtype, "reference");
  if (op === "ge_scalar" || op === "le_scalar" || op === "gt_scalar") {
    const compare = COMPARE_SCALAR[op];
    for (let i = 0; i < values.length; i += 1) values[i] = compare(x.data[i], scalars[0]) ? 1 : 0;
    return materialize(outDtype, [...x.shape], values);
  }
  const apply = UNARY_F32[op];
  for (let i = 0; i < values.length; i += 1) {
    values[i] = Math.fround(apply(x.data[i], scalars));
  }
  return materialize(outDtype, [...x.shape], values);
};

/**
 * 出力の平坦添字 → 入力の平坦添字（右詰め broadcast）。
 *
 * MUST: stride 表を作らない。codegen 側（elementwise.ts の `elementwiseParams`）は
 * 「右から running product を積んで 1 の次元を 0 にする」形で stride を組むので、参照側で
 * 同じ式を書くと stride 導出の誤りが両側で相殺し、GPU との突合テストが恒真化する。
 * ここは別構造 —「出力座標を毎要素 divmod で分解 → 入力 shape を左から Horner 展開で畳む」
 * — で導く。相互検証の根拠は実装の独立性そのものなので、codegen とヘルパも式形も共有しない。
 *
 * 右詰めで入力 rank に載らない先行次元は、入力側では常に座標 0（Horner の畳み込みに入らない）。
 * 長さ 1 の次元も同様に座標 0 へ潰れる（これが broadcast）。
 */
const broadcastIndex = (
  outIndex: number,
  outShape: readonly number[],
  inShape: readonly number[],
): number => {
  const rank = outShape.length;
  const coords = new Array<number>(rank).fill(0);
  let rest = outIndex;
  for (let d = rank - 1; d >= 0; d -= 1) {
    coords[d] = rest % outShape[d];
    rest = Math.floor(rest / outShape[d]);
  }
  const offset = rank - inShape.length;
  let index = 0;
  for (let d = 0; d < inShape.length; d += 1) {
    const extent = inShape[d];
    index = index * extent + (extent === 1 ? 0 : coords[offset + d]);
  }
  return index;
};

export const referenceBinary = (op: BinaryOpName, a: RefTensor, b: RefTensor): RefTensor => {
  const contract = resolveOpContract(op);
  assertDtype(contract, a.dtype, "reference");
  assertDtype(contract, b.dtype, "reference");
  if (a.dtype !== b.dtype) {
    throw new ReferenceOpError(`op '${op}' の入力 dtype が混在（${a.dtype} / ${b.dtype}）`);
  }
  const shape = computeOutputShape(contract, [a.shape, b.shape], "reference")[0];
  const evaluate = binaryEvaluator(op, a.dtype);
  const values = new Array<number>(numel(shape));
  for (let i = 0; i < values.length; i += 1) {
    const va = a.data[broadcastIndex(i, shape, a.shape)];
    const vb = b.data[broadcastIndex(i, shape, b.shape)];
    values[i] = evaluate(va, vb);
  }
  return materialize(outputDtypeOf(contract, 0, a.dtype, "reference"), shape, values);
};

/**
 * where `out = cond ? a : b`（3 者とも右詰め broadcast）。
 *
 * 添字の導出は {@link broadcastIndex} — 「出力座標を毎要素 divmod → 入力 shape を左から
 * Horner」で、codegen の stride 表とは別構造（同ファイル冒頭の MUST）。
 * MUST: 分岐の向きは torch の `where(cond, a, b)`（真なら a）。取り違えても shape も dtype も
 * 変わらないので、値でしか検出できない（テストは a / b を別の値域で埋める）。
 */
export const referenceWhere = (cond: RefTensor, a: RefTensor, b: RefTensor): RefTensor => {
  const contract = resolveOpContract("where");
  assertSlotDtype(contract, 0, cond.dtype, "reference");
  assertSlotDtype(contract, 1, a.dtype, "reference");
  assertSlotDtype(contract, 2, b.dtype, "reference");
  const shape = computeOutputShape(contract, [cond.shape, a.shape, b.shape], "reference")[0];
  const values = new Array<number>(numel(shape));
  for (let i = 0; i < values.length; i += 1) {
    values[i] = cond.data[broadcastIndex(i, shape, cond.shape)] !== 0
      ? a.data[broadcastIndex(i, shape, a.shape)]
      : b.data[broadcastIndex(i, shape, b.shape)];
  }
  return materialize(outputDtypeOf(contract, 0, cond.dtype, "reference"), shape, values);
};

/**
 * 最終次元の前縁和 `out[…, j] = Σ_{i ≤ j} x[…, i]`。
 *
 * MUST: **各出力を独立した内側ループの総和として**計算する（O(n²)）。カーネルは
 * 走査しながら累算器を持ち回る形なので、同じ漸化式で書くと累積方向・初期値・格納位置の
 * 誤りが両側で相殺する。行長は実測 ~10（recon §2）でテストも小さいので、素朴な二重和で
 * 十分速い。
 * MUST: 内側ループは **1 項ごとに `Math.fround` で丸める**（f32 の累算を 1 段ずつ再現する）。
 * JS の数値は f64 なので、まとめて足してから 1 回だけ丸めると参照側だけが f64 精度の
 * 総和になり、行長が伸びるほど GPU との差が「実装バグではなく丸め経路の違い」で開く
 * （行長 300 でオラクルの分解能が許容差を食い潰す）。二重和の**独立性は保ったまま**、
 * 丸め経路だけをカーネルに揃える。
 */
export const referenceCumsum = (
  x: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => {
  const contract = resolveOpContract("cumsum");
  assertDtype(contract, x.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape], "reference", { attrs })[0];
  const dim = shape[shape.length - 1];
  const rows = numel(shape.slice(0, -1));
  const out = new Float32Array(numel(shape));
  for (let row = 0; row < rows; row += 1) {
    const base = row * dim;
    for (let last = 0; last < dim; last += 1) {
      let acc = 0;
      for (let i = 0; i <= last; i += 1) acc = Math.fround(acc + x.data[base + i]);
      out[base + last] = acc;
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * 意味論 dtype の変換。契約（src/ops.ts）どおり
 * **f32 → 整数は 0 方向切り捨て（torch の truncate）**、**x → bool は x != 0** で、
 * bool → 数値は u32 の 0 / 1 をそのまま読む。
 */
export const referenceCast = (x: RefTensor, to: IrDtype): RefTensor => {
  const values = new Array<number>(x.data.length);
  for (let i = 0; i < values.length; i += 1) {
    const v = x.data[i];
    values[i] = to === "bool" ? (v !== 0 ? 1 : 0) : to === "i32" ? Math.trunc(v) : Math.fround(v);
  }
  return materialize(to, [...x.shape], values);
};

/** row-major の C[m,n] = A[m,k] · B[k,n]。f64 で積んで格納時に f32 へ丸める。 */
export const referenceMatmul = (a: RefTensor, b: RefTensor): RefTensor => {
  const contract = resolveOpContract("matmul");
  assertDtype(contract, a.dtype, "reference");
  assertDtype(contract, b.dtype, "reference");
  const shape = computeOutputShape(contract, [a.shape, b.shape], "reference")[0];
  const [m, n] = shape;
  const k = a.shape[1];
  const out = new Float32Array(m * n);
  for (let row = 0; row < m; row += 1) {
    for (let col = 0; col < n; col += 1) {
      let acc = 0;
      for (let i = 0; i < k; i += 1) acc += a.data[row * k + i] * b.data[i * n + col];
      out[row * n + col] = Math.fround(acc);
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * バッチ matmul `C[b,m,n] = A[b,m,k] · B[b,k,n]`。matmul と同じく f64 で積んで格納時に
 * 1 度だけ f32 へ丸める。
 *
 * MUST: バッチのオフセットを「行列 1 枚の要素数」から**毎バッチ独立に**組む。GPU 側は
 * dispatch の z 軸から同じ量を導くので、片方が軸を取り違えれば必ず値が食い違う形状
 * （B / M / K / N を全て違う長さにしたケース）で突合すること。
 */
export const referenceBmm = (a: RefTensor, b: RefTensor): RefTensor => {
  const contract = resolveOpContract("bmm");
  assertDtype(contract, a.dtype, "reference");
  assertDtype(contract, b.dtype, "reference");
  const shape = computeOutputShape(contract, [a.shape, b.shape], "reference")[0];
  const [batch, m, n] = shape;
  const k = a.shape[2];
  const out = new Float32Array(batch * m * n);
  for (let item = 0; item < batch; item += 1) {
    const aBase = item * m * k;
    const bBase = item * k * n;
    const cBase = item * m * n;
    for (let row = 0; row < m; row += 1) {
      for (let col = 0; col < n; col += 1) {
        let acc = 0;
        for (let i = 0; i < k; i += 1) {
          acc += a.data[aBase + row * k + i] * b.data[bBase + i * n + col];
        }
        out[cBase + row * n + col] = Math.fround(acc);
      }
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * 最終次元の gather。`out[..., j] = src[..., index[..., j]]`（先行次元は src と index で一致）。
 *
 * MUST: **範囲外添字は必ず throw**。CPU 参照は契約のオラクルなので、GPU 側の裁定
 * （範囲外は NaN 汚染 — src/kernels/gather.ts）に合わせて緩めない。合わせると「契約違反の
 * グラフが両側で同じ値になる」ので突合が違反を検出できなくなる。
 * MUST: 添字の分解を codegen と別形にする。カーネルは出力の平坦添字を `i / J` で行へ割るが、
 * こちらは行と列の二重ループから平坦添字を組み立てる。
 */
export const referenceGather = (src: RefTensor, index: RefTensor): RefTensor => {
  const contract = resolveOpContract("gather");
  assertSlotDtype(contract, 0, src.dtype, "reference");
  assertSlotDtype(contract, 1, index.dtype, "reference");
  const shape = computeOutputShape(contract, [src.shape, index.shape], "reference")[0];
  const cols = shape[shape.length - 1];
  const srcCols = src.shape[src.shape.length - 1];
  const rows = numel(shape.slice(0, -1));
  const out = new Float32Array(numel(shape));
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const flat = row * cols + col;
      const pick = index.data[flat];
      if (pick < 0 || pick >= srcCols) {
        throw new ReferenceOpError(
          `gather の添字 index[${flat}] = ${pick} が src の最終次元 ${srcCols} の範囲外`,
        );
      }
      out[flat] = src.data[row * srcCols + pick];
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * 1 軸の reduce（keepdim 無し）。縮約軸 `axis` は attrs の `dim`（既定値補完はしない）。
 *
 * 走査は「出力要素ごとに縮約軸を `inner` 送りで舐める」形で、最終次元（`inner == 1`）は
 * 連続走査に退化する。GPU 側は 2 変種に分かれるが、参照は**軸を添字式で扱う 1 本**でよい
 * （f64 で積んで最後に 1 回だけ丸める性質は軸に依らない）。
 *
 * bool 入力の `sum` は**真の個数**（出力 i32 — 契約表の写像）。f64 で数えてから整数配列へ
 * 落とすので、縮約長が 2^24 を超えても丸まらない。
 */
export const referenceRowReduce = (
  op: ReduceOpName,
  x: RefTensor,
  axis: number,
): RefTensor => {
  const contract = resolveOpContract(op);
  assertDtype(contract, x.dtype, "reference");
  const attrs = { dim: axis };
  const shape = computeOutputShape(contract, [x.shape], "reference", { attrs })[0];
  const outDtype = outputDtypeOf(contract, 0, x.dtype, "reference");
  const dim = x.shape[axis];
  const inner = numel(x.shape.slice(axis + 1));
  const count = numel(shape);
  const values = new Array<number>(count);
  for (let index = 0; index < count; index += 1) {
    // 出力添字 index = outer·inner + i。縮約軸だけを外した先頭アドレスへ戻す。
    const base = Math.floor(index / inner) * dim * inner + (index % inner);
    if (op === "sum") {
      let acc = 0;
      for (let i = 0; i < dim; i += 1) {
        const value = x.data[base + i * inner];
        acc += x.dtype === "bool" ? (value !== 0 ? 1 : 0) : value;
      }
      values[index] = outDtype === "f32" ? Math.fround(acc) : acc;
      continue;
    }
    let acc = x.data[base];
    for (let i = 1; i < dim; i += 1) {
      const value = x.data[base + i * inner];
      acc = op === "amax" ? Math.max(acc, value) : Math.min(acc, value);
    }
    values[index] = acc;
  }
  return materialize(outDtype, shape, values);
};

/**
 * 最終次元の argmax（ADR 0068 決定 2）。出力は **i32 の添字**で、shape は入力の最終次元を
 * 1 に潰した形（rank 保存）。
 *
 * 固定挙動（torch 準拠 — 実測 2026-08-17）:
 *
 * - **同値は最小 index**（`[1,3,3,2]` → 1）。
 * - **全 −inf 行は index 0**（`amax` と違い identity が値ではなく添字なので、−inf の行でも
 *   答えが定義される）。
 * - **NaN は最大**（`[1,NaN,3,NaN]` → 1）。NaN が複数あれば最小 index。
 *
 * MUST: GPU 側（src/kernels/argmax.ts）の式を写さない。あちらは「(値, index) 対の辞書式
 * 最大元を木で畳む」形なので、こちらは**先頭から 1 要素ずつ走査して更新する素朴形**で書く
 * （NaN 判定も `Number.isNaN` で、ビット列判定を写さない）。同じ式を書くと木の結合順の
 * 誤りや NaN 判定の抜けが両側で相殺する。
 */
export const referenceArgmax = (x: RefTensor): RefTensor => {
  const contract = resolveOpContract(ARGMAX_OP);
  assertDtype(contract, x.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape], "reference")[0];
  const dim = x.shape[x.shape.length - 1];
  const rows = numel(shape);
  const values = new Array<number>(rows);
  for (let row = 0; row < rows; row += 1) {
    const base = row * dim;
    let bestAt = 0;
    for (let i = 1; i < dim; i += 1) {
      const candidate = x.data[base + i];
      const best = x.data[base + bestAt];
      // NaN は最大（比較演算は NaN で全て false になるので、NaN 側を明示的に拾う）。
      // 既に NaN を掴んでいるなら後続では絶対に更新しない = NaN も最小 index になる。
      if (Number.isNaN(best)) continue;
      if (Number.isNaN(candidate) || candidate > best) bestAt = i;
    }
    values[row] = bestAt;
  }
  return materialize(outputDtypeOf(contract, 0, x.dtype, "reference"), shape, values);
};

/**
 * 最終次元の top-k（ADR 0068 決定 3）。返りは **2 本**（slot 0 = 値 f32 の降順・
 * slot 1 = 添字 i32）で、どちらも `[…, k]`。
 *
 * 固定挙動（実測 2026-08-17 / torch 2.13.0+cpu）:
 *
 * - **値の列は torch とビット一致**（降順・同値の多重度が同じなので、重複だらけの行でも
 *   一致する）。
 * - **添字の列は同値要素で最小 index 優先**。これは torch **準拠ではなく karume の規定**
 *   （torch は同値の順序を保証せず、`topk([5,5,5,5],1)` = 2 に対し `argmax` = 0 で
 *   自己矛盾している — 詳細は src/kernels/topk.ts の NOTE）。k=1 は argmax と同じ答え。
 * - **NaN は最大**（torch も先頭へ出す）。複数あれば最小 index。
 * - **全 −inf 行**でも答えは定義される（identity が値ではなく順位なので、最小 index から
 *   k 本）。
 *
 * MUST: GPU 側（src/kernels/topk.ts）の式を写さない。あちらは「レーン局所 top-k → k ラウンドの
 * トーナメント」なので、こちらは**添字を辞書式順序で並べ替えて先頭 k 本を取る素朴形**で書く
 * （NaN 判定も `Number.isNaN`）。同じ形を書くとマージ境界の誤りや NaN の抜けが両側で相殺する。
 */
export const referenceTopk = (
  x: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): readonly RefTensor[] => {
  const contract = resolveOpContract(TOPK_OP);
  assertDtype(contract, x.dtype, "reference");
  const shapes = computeOutputShape(contract, [x.shape], "reference", { attrs });
  const k = topkK(attrs, "reference");
  const dim = x.shape[x.shape.length - 1];
  const rows = numel(x.shape) / dim;
  const values = new Array<number>(rows * k);
  const indices = new Array<number>(rows * k);
  for (let row = 0; row < rows; row += 1) {
    const base = row * dim;
    // 添字を「値の降順・同値なら添字の昇順」で並べ替える（NaN は最大なので、比較の前に
    // NaN 同士 / NaN 対数値を先に決める）。
    const order = Array.from({ length: dim }, (_, i) => i).sort((a, b) => {
      const va = x.data[base + a];
      const vb = x.data[base + b];
      const na = Number.isNaN(va);
      const nb = Number.isNaN(vb);
      if (na !== nb) return na ? -1 : 1;
      if (!na && va !== vb) return va > vb ? -1 : 1;
      return a - b;
    });
    for (let slot = 0; slot < k; slot += 1) {
      values[row * k + slot] = x.data[base + order[slot]];
      indices[row * k + slot] = order[slot];
    }
  }
  return [
    materialize(outputDtypeOf(contract, 0, x.dtype, "reference"), shapes[0], values),
    materialize(outputDtypeOf(contract, 1, x.dtype, "reference"), shapes[1], indices),
  ];
};

/**
 * 要素順を変えない形の付け替え（ADR 0011 の reshape）。データは連続のまま写す。
 * 実行系はここでバッファを別名化してコピーを出さないが、参照側は素直に写して
 * 「別名化しても値が変わらない」ことのオラクルになる。
 */
export const referenceReshape = (x: RefTensor, shape: readonly number[]): RefTensor => {
  const target = [...shape];
  if (numel(target) !== x.data.length) {
    throw new ReferenceOpError(
      `reshape の要素数が合わない ${x.data.length} → [${target.join(",")}]`,
    );
  }
  return materialize(x.dtype, target, Array.from(x.data));
};

/**
 * 軸の並べ替え。`dims[d]` = 出力の次元 d が取る入力の次元番号。
 *
 * MUST: codegen（strided.ts）と添字計算を共有しない。あちらは「出力座標 → 入力 stride の
 * 内積」で読む gather 形なので、こちらは逆向きの scatter 形 —「**入力**の平坦添字を入力
 * 座標へ分解 → 出力座標へ並べ替え → 左からの Horner で出力添字を畳む」— で書く。
 * 同じ向きで書くと stride 表の組み立て誤りが両側で相殺し、突合テストが恒真化する。
 */
export const referencePermute = (x: RefTensor, dims: readonly number[]): RefTensor => {
  const rank = x.shape.length;
  if (dims.length !== rank || new Set(dims).size !== rank) {
    throw new ReferenceOpError(
      `permute の dims [${dims.join(",")}] が shape [${x.shape.join(",")}] の並べ替えでない`,
    );
  }
  const outShape = dims.map((dim) => {
    const extent = x.shape[dim];
    if (extent === undefined) {
      throw new ReferenceOpError(`permute の dims に軸 ${dim} が入っている（rank ${rank}）`);
    }
    return extent;
  });
  const values = new Array<number>(x.data.length);
  const coords = new Array<number>(rank).fill(0);
  for (let flat = 0; flat < x.data.length; flat += 1) {
    let rest = flat;
    for (let d = rank - 1; d >= 0; d -= 1) {
      coords[d] = rest % x.shape[d];
      rest = Math.floor(rest / x.shape[d]);
    }
    let out = 0;
    for (let d = 0; d < rank; d += 1) out = out * outShape[d] + coords[dims[d]];
    values[out] = x.data[flat];
  }
  return materialize(x.dtype, outShape, values);
};

/**
 * 長さ 1 の次元だけを目標 shape へ複製する（右詰め）。添字の導出は
 * {@link broadcastIndex} — binary elementwise と同じ「出力座標を毎要素 divmod → 入力
 * shape を左から Horner」で、codegen の stride 表とは別構造（同ファイル冒頭の MUST）。
 */
export const referenceExpand = (x: RefTensor, shape: readonly number[]): RefTensor => {
  const target = [...shape];
  if (target.length < x.shape.length) {
    throw new ReferenceOpError(
      `expand は rank を下げられない [${x.shape.join(",")}] → [${target.join(",")}]`,
    );
  }
  const offset = target.length - x.shape.length;
  x.shape.forEach((extent, index) => {
    if (extent !== 1 && extent !== target[offset + index]) {
      throw new ReferenceOpError(
        `expand は長さ 1 でない次元 ${index}（${extent}）を ${target[offset + index]} にできない`,
      );
    }
  });
  const values = new Array<number>(numel(target));
  for (let i = 0; i < values.length; i += 1) {
    values[i] = x.data[broadcastIndex(i, target, x.shape)];
  }
  return materialize(x.dtype, target, values);
};

/**
 * 静的軸・静的範囲の切り出し（ADR 0014）。`dim` 軸の `[start, end)` を取り出す。
 *
 * MUST: codegen（strided.ts の読み族）と添字計算を共有しない。あちらは「出力座標 → 入力
 * stride の内積 + offset」で読む gather 形なので、こちらは逆向きの scatter 形 —
 * 「**入力**の平坦添字を入力座標へ分解 → 窓の外なら捨てる → 出力座標を左からの Horner で
 * 畳む」— で書く（referenceSymPrefixSlice と同じ方針。offset の組み立て誤りを両側で
 * 相殺させないため）。
 */
export const referenceSlice = (
  x: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => {
  const contract = resolveOpContract("slice");
  assertDtype(contract, x.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape], "reference", { attrs })[0];
  const { dim, start } = sliceAttrs(attrs, "reference");
  const rank = x.shape.length;
  const values = new Array<number>(numel(shape));
  const coords = new Array<number>(rank).fill(0);
  for (let flat = 0; flat < x.data.length; flat += 1) {
    let rest = flat;
    for (let d = rank - 1; d >= 0; d -= 1) {
      coords[d] = rest % x.shape[d];
      rest = Math.floor(rest / x.shape[d]);
    }
    const picked = coords[dim] - start;
    if (picked < 0 || picked >= shape[dim]) continue;
    let out = 0;
    for (let d = 0; d < rank; d += 1) out = out * shape[d] + (d === dim ? picked : coords[d]);
    values[out] = x.data[flat];
  }
  return materialize(x.dtype, shape, values);
};

/**
 * 静的軸の連結（ADR 0014）。入力を順に並べて `dim` 軸へ積む。
 *
 * MUST: 出力を**全て埋め切る**ことをオラクル側でも見る（未書き込みの穴が残ったら
 * ReferenceOpError）。GPU 側は入力ごとの部分書きで全域を覆う形なので、参照実装が
 * 「書かれなかった要素を 0 とみなす」と full-write の破れが両側で相殺する。
 * MUST: 添字は入力ごとの走査（scatter 形）で、書き出し位置は**連結軸の座標をずらす**形に
 * 組む — codegen は offset 1 語に畳んだ形なので、同じ式にはしない。
 */
export const referenceCat = (
  inputs: readonly RefTensor[],
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => {
  const contract = resolveOpContract("cat");
  for (const input of inputs) assertDtype(contract, input.dtype, "reference");
  const shape = computeOutputShape(contract, inputs.map((input) => input.shape), "reference", {
    attrs,
  })[0];
  const dim = catDim(attrs, "reference");
  const rank = shape.length;
  const values = new Array<number | undefined>(numel(shape)).fill(undefined);
  const coords = new Array<number>(rank).fill(0);
  let written = 0;
  for (const input of inputs) {
    for (let flat = 0; flat < input.data.length; flat += 1) {
      let rest = flat;
      for (let d = rank - 1; d >= 0; d -= 1) {
        coords[d] = rest % input.shape[d];
        rest = Math.floor(rest / input.shape[d]);
      }
      let out = 0;
      for (let d = 0; d < rank; d += 1) {
        out = out * shape[d] + (d === dim ? coords[d] + written : coords[d]);
      }
      values[out] = input.data[flat];
    }
    written += input.shape[dim];
  }
  const hole = values.findIndex((value) => value === undefined);
  if (hole >= 0) {
    throw new ReferenceOpError(`cat の出力 [${shape.join(",")}] の要素 ${hole} が書かれていない`);
  }
  return materialize(inputs[0].dtype, shape, values as number[]);
};

/**
 * 最終次元の定数 0 埋め（ADR 0014）。
 *
 * MUST: **0 で埋めてから転写する**（scatter 形）。GPU 側は出力 1 要素ごとに「範囲内か」を
 * 判定して転写値か 0 を書く gather 形なので、同じ分岐を写すと境界の off-by-one が両側で
 * 相殺する。ゼロ領域が本当に 0 であること自体もこの形なら値で出る。
 */
export const referencePad = (
  x: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => {
  const contract = resolveOpContract("pad");
  assertDtype(contract, x.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape], "reference", { attrs })[0];
  const { left } = padAttrs(attrs, "reference");
  const lengthIn = x.shape[x.shape.length - 1];
  const lengthOut = shape[shape.length - 1];
  const values = new Array<number>(numel(shape)).fill(0);
  for (let flat = 0; flat < x.data.length; flat += 1) {
    const row = Math.floor(flat / lengthIn);
    const col = flat % lengthIn;
    values[row * lengthOut + col + left] = x.data[flat];
  }
  return materialize(x.dtype, shape, values);
};

/**
 * 静的軸の添字反転（ADR 0014）。
 *
 * MUST: codegen（src/kernels/flip.ts）と向きを揃えない。あちらは出力添字から
 * `len - 1 - c` で入力を読む gather 形なので、こちらは**入力**を走査して反転先へ書く
 * scatter 形にする（軸の取り違えと off-by-one を両側で相殺させないため）。
 */
export const referenceFlip = (
  x: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => {
  const contract = resolveOpContract("flip");
  assertDtype(contract, x.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape], "reference", { attrs })[0];
  const dim = flipDim(attrs, "reference");
  const rank = shape.length;
  const values = new Array<number>(x.data.length);
  const coords = new Array<number>(rank).fill(0);
  for (let flat = 0; flat < x.data.length; flat += 1) {
    let rest = flat;
    for (let d = rank - 1; d >= 0; d -= 1) {
      coords[d] = rest % x.shape[d];
      rest = Math.floor(rest / x.shape[d]);
    }
    let out = 0;
    for (let d = 0; d < rank; d += 1) {
      out = out * shape[d] + (d === dim ? shape[d] - 1 - coords[d] : coords[d]);
    }
    values[out] = x.data[flat];
  }
  return materialize(x.dtype, shape, values);
};

/**
 * 記号 prefix スライス（ADR 0010）— Tmax で焼いた定数 `x` の**各軸の先頭**を `shape` へ
 * 切り出す。prefix 長そのものはランタイム側で `coeff·sym+offset` から決まるので、参照実装は
 * 決まった長さを受け取るだけ。
 *
 * MUST: codegen（strided.ts）と添字計算を共有しない。あちらは「出力座標 → 入力 stride の
 * 内積」で読む gather 形なので、こちらは逆向きの scatter 形 —「**入力**の平坦添字を Tmax 形の
 * 座標へ分解 → prefix の外なら捨てる → 出力座標を左からの Horner で畳む」— で書く
 * （referencePermute と同じ方針。stride 表の組み立て誤りを両側で相殺させないため）。
 */
export const referenceSymPrefixSlice = (x: RefTensor, shape: readonly number[]): RefTensor => {
  const target = [...shape];
  const rank = x.shape.length;
  if (target.length !== rank) {
    throw new ReferenceOpError(
      `sym_prefix_slice は rank を変えない [${x.shape.join(",")}] → [${target.join(",")}]`,
    );
  }
  target.forEach((extent, index) => {
    if (extent < 0 || extent > x.shape[index]) {
      throw new ReferenceOpError(
        `sym_prefix_slice の prefix [${target.join(",")}] が定数 [${
          x.shape.join(",")
        }] に収まらない`,
      );
    }
  });
  const values = new Array<number>(numel(target));
  const coords = new Array<number>(rank).fill(0);
  for (let flat = 0; flat < x.data.length; flat += 1) {
    let rest = flat;
    for (let d = rank - 1; d >= 0; d -= 1) {
      coords[d] = rest % x.shape[d];
      rest = Math.floor(rest / x.shape[d]);
    }
    if (coords.some((coord, d) => coord >= target[d])) continue;
    let out = 0;
    for (let d = 0; d < rank; d += 1) out = out * target[d] + coords[d];
    values[out] = x.data[flat];
  }
  return materialize(x.dtype, target, values);
};

/**
 * linear `out[…, n] = Σ_k x[…, k] · weight[n, k] + bias[n]`（融合 op — ADR 0012）。
 *
 * MUST: 重みの添字を `weight[n * K + k]` の**転置レイアウトのまま**組む。GPU 側も同じ
 * レイアウトを読むが、こちらは「先行次元をタイルではなく素の多重ループで走る」ので、
 * タイル境界・端数の扱いを共有しない（そこが検証したい対象）。
 */
export const referenceLinear = (x: RefTensor, weight: RefTensor, bias: RefTensor): RefTensor => {
  const contract = resolveOpContract("linear");
  for (const input of [x, weight, bias]) assertDtype(contract, input.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape, weight.shape, bias.shape], "reference")[0];
  const outFeatures = weight.shape[0];
  const inFeatures = weight.shape[1];
  const rows = numel(shape.slice(0, -1));
  const out = new Float32Array(numel(shape));
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < outFeatures; col += 1) {
      let acc = 0;
      for (let i = 0; i < inFeatures; i += 1) {
        acc += x.data[row * inFeatures + i] * weight.data[col * inFeatures + i];
      }
      out[row * outFeatures + col] = Math.fround(acc + bias.data[col]);
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * layer_norm（最終次元・affine あり）。分散は**母分散（N 割り、correction = 0）**で、
 * `torch.var` の既定（N−1）ではない（台帳の反例集の該当行）。
 *
 * MUST: 2 パス（平均 → 偏差平方和）で書く。`E[x²] − E[x]²` の 1 パス形は桁落ちで負の分散を
 * 出しうるうえ、GPU 側と同じ近道を写すと誤差が相殺して突合が恒真化する。
 */
export const referenceLayerNorm = (
  x: RefTensor,
  weight: RefTensor,
  bias: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => {
  const contract = resolveOpContract("layer_norm");
  for (const input of [x, weight, bias]) assertDtype(contract, input.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape, weight.shape, bias.shape], "reference", {
    attrs,
  })[0];
  const { eps } = layerNormAttrs(attrs, "reference");
  const dim = shape[shape.length - 1];
  const rows = numel(shape.slice(0, -1));
  const out = new Float32Array(numel(shape));
  for (let row = 0; row < rows; row += 1) {
    const base = row * dim;
    let sum = 0;
    for (let i = 0; i < dim; i += 1) sum += x.data[base + i];
    const mean = sum / dim;
    let squares = 0;
    for (let i = 0; i < dim; i += 1) {
      const deviation = x.data[base + i] - mean;
      squares += deviation * deviation;
    }
    // eps は f32 語で運ばれるスカラ attr なので参照側も f32 へ丸めてから使う（他のスカラと同じ規律）
    const inv = 1 / Math.sqrt(squares / dim + Math.fround(eps));
    for (let i = 0; i < dim; i += 1) {
      out[base + i] = Math.fround((x.data[base + i] - mean) * inv * weight.data[i] + bias.data[i]);
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * rms_norm（最終次元・weight のみ）。`out[r, i] = x[r, i] / sqrt(mean_r(x²) + eps) · weight[i]`
 *
 * MUST: **平均を引かない**（layer_norm を写さない）。rms_norm は二乗の平均そのものを使う
 * ので、偏差平方和にするとゼロ平均でない入力で静かにずれる。
 * MUST: 逆数平方根は `1 / Math.sqrt(…)` で書く（GPU 側は `inverseSqrt` の 1 演算）。同じ
 * 組込みを写すと丸めの誤りが両側で相殺して突合が恒真化する — erf / softmax と同じ規律。
 * MUST: 二乗和は f64 で積んで格納時に 1 度だけ丸める（縮約は matmul / 行 reduce と同じ扱い）。
 */
export const referenceRmsNorm = (
  x: RefTensor,
  weight: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => {
  const contract = resolveOpContract("rms_norm");
  for (const input of [x, weight]) assertDtype(contract, input.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape, weight.shape], "reference", { attrs })[0];
  const eps = rmsNormEps(attrs, "reference");
  const dim = shape[shape.length - 1];
  const rows = numel(shape.slice(0, -1));
  const out = new Float32Array(numel(shape));
  for (let row = 0; row < rows; row += 1) {
    const base = row * dim;
    let squares = 0;
    for (let i = 0; i < dim; i += 1) squares += x.data[base + i] * x.data[base + i];
    // eps は f32 語で運ばれるスカラ attr なので参照側も f32 へ丸めてから使う（layer_norm と同じ）
    const inv = 1 / Math.sqrt(squares / dim + Math.fround(eps));
    for (let i = 0; i < dim; i += 1) {
      out[base + i] = Math.fround(x.data[base + i] * inv * weight.data[i]);
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * softmax / safe_softmax（最終次元、safe-softmax）。
 *
 * MUST: **amax を引いた形**で書く。素朴形は台帳の反例集どおり大入力で壊れ、オラクル側が
 * 同じ壊れ方をすると「カーネルの safe 化が外れている」ことを突合が検出できなくなる。
 *
 * `safe` は ADR 0044 の safe_softmax（行 max が −inf の行に 0 を書く）。有限要素を持つ行の
 * 演算列は素の softmax と**同一のまま**にする MUST — 分けて書くと「変種だけが素朴形へ
 * 退化した」状態を突合が見逃す。
 */
const softmaxLike = (
  x: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
  safe: boolean,
): RefTensor => {
  const contract = resolveOpContract(safe ? "safe_softmax" : "softmax");
  assertDtype(contract, x.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape], "reference", { attrs })[0];
  const dim = shape[shape.length - 1];
  const rows = numel(shape.slice(0, -1));
  const out = new Float32Array(numel(shape));
  for (let row = 0; row < rows; row += 1) {
    const base = row * dim;
    let amax = x.data[base];
    for (let i = 1; i < dim; i += 1) amax = Math.max(amax, x.data[base + i]);
    // 全要素 −inf の行（`out` は 0 初期化済みなのでそのまま次の行へ）。
    if (safe && amax === Number.NEGATIVE_INFINITY) continue;
    let total = 0;
    for (let i = 0; i < dim; i += 1) total += Math.exp(x.data[base + i] - amax);
    for (let i = 0; i < dim; i += 1) {
      out[base + i] = Math.fround(Math.exp(x.data[base + i] - amax) / total);
    }
  }
  return { dtype: "f32", shape, data: out };
};

export const referenceSoftmax = (
  x: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => softmaxLike(x, attrs, false);

/** safe_softmax（ADR 0044）— softmax + 「行 max が −inf の行は全 0」。 */
export const referenceSafeSoftmax = (
  x: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => softmaxLike(x, attrs, true);

/**
 * 融合 attention `out = softmax_lastdim((q·scale) @ (k·scale)ᵀ + mask) @ v`（ADR 0023）。
 *
 * `mask` は省略可能な第 4 入力（`[1,1,M,N]` の加算項を B·H 全体へ broadcast — src/ops.ts）。
 * 加算は **S を f32 へ丸めた後**（GPU 側 ①QK の書き出し epilogue と同じ位置）。
 *
 * MUST: **素直な 3 段**で書く（S を実体化 → safe-softmax で P を実体化 → PV）。GPU 側は
 * P を実体化せず A タイル充填時に `exp(S−m)·inv` を評価する（1 パスの融合）ので、参照が
 * 同じ融合形（online softmax や P 非実体化）で書かれていると、融合の段の誤りが両側で
 * 相殺して突合が検出できなくなる — `referenceSoftmax` の「amax を引く MUST」と同じ規律。
 * MUST: 縮約は f64 で積んで格納時に 1 度だけ f32 へ丸める（`referenceBmm` と同じ）。GPU の
 * f32 逐次累積とはビット一致しないのが**正しい**（だから allclose で判定する）。
 * MUST: `scale` は q と k の**両方**に掛ける（半スケール契約 — src/ops.ts）。片側だけに
 * 掛ける実装と突き合わせると値が √ 倍ずれるので、その誤りはこのオラクルが検出する。
 */
export const referenceAttention = (
  q: RefTensor,
  k: RefTensor,
  v: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
  mask?: RefTensor,
): RefTensor => {
  const contract = resolveOpContract("attention");
  const inputs = mask === undefined ? [q, k, v] : [q, k, v, mask];
  for (const input of inputs) assertDtype(contract, input.dtype, "reference");
  const shape = computeOutputShape(
    contract,
    inputs.map((input) => input.shape),
    "reference",
    { attrs },
  )[0];
  // GPU 側は params の f32 語で運ぶので、オラクルも f32 へ丸めた値を使う（masked_fill と同じ）。
  const scale = Math.fround(attentionScale(attrs, "reference"));
  const [batches, heads, rows, depth] = shape;
  const cols = k.shape[2];
  // GQA の繰り返し数 `r = H / Hkv`（ADR 0067 決定 1 の導出値 — 整除は shape 検査が保証する）。
  const repeat = heads / k.shape[1];
  const out = new Float32Array(numel(shape));
  // f32 の中間（S と P）は「実体化された 3 段」を素直に表す — 段ごとに丸めが 1 回入る。
  const scores = new Float32Array(cols);
  const weights = new Float32Array(cols);
  for (let head = 0; head < batches * heads; head += 1) {
    const qBase = head * rows * depth;
    // MUST: kv-head は**整数除算**で写す（GPU 側 `wid.z / r` と同一の恒等式 — ADR 0067 決定 2。
    // `head = b·H + h` に対し `H = Hkv·r` なら `⌊head/r⌋ = b·Hkv + ⌊h/r⌋` が厳密に成立する）。
    // 剰余（`head % Hkv`）は「Hkv = H かつ B = 1」でだけ一致する別写像で、head 対応も b 項も
    // 壊れる（検出器は tests/helpers/gpu_op_cases.ts の GQA / MQA ケース）。
    const kvBase = Math.floor(head / repeat) * cols * depth;
    for (let row = 0; row < rows; row += 1) {
      // ① S[row, col] = Σ_d (q·scale)[row,d] · (k·scale)[col,d]（+ mask[row,col]）
      for (let col = 0; col < cols; col += 1) {
        let acc = 0;
        for (let d = 0; d < depth; d += 1) {
          acc += (q.data[qBase + row * depth + d] * scale) *
            (k.data[kvBase + col * depth + d] * scale);
        }
        // mask は S を f32 へ丸めた**後**に足す（GPU 側の書き出し epilogue と同じ位置）。
        // mask は [1,1,M,N] なので添字にバッチが入らない = 全 head が同じ行を読む。
        scores[col] = mask === undefined
          ? Math.fround(acc)
          : Math.fround(Math.fround(acc) + mask.data[row * cols + col]);
      }
      // ② safe-softmax（amax を引く形 MUST — referenceSoftmax と同じ）
      let amax = scores[0];
      for (let col = 1; col < cols; col += 1) amax = Math.max(amax, scores[col]);
      let total = 0;
      for (let col = 0; col < cols; col += 1) total += Math.exp(scores[col] - amax);
      for (let col = 0; col < cols; col += 1) {
        weights[col] = Math.fround(Math.exp(scores[col] - amax) / total);
      }
      // ③ O[row, d] = Σ_col P[row,col] · v[col,d]
      for (let d = 0; d < depth; d += 1) {
        let acc = 0;
        for (let col = 0; col < cols; col += 1) {
          acc += weights[col] * v.data[kvBase + col * depth + d];
        }
        out[qBase + row * depth + d] = Math.fround(acc);
      }
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * embedding `out[…, h] = weight[index[…], h]`。
 *
 * MUST: **範囲外添字は必ず throw**（gather と同じ裁定 — GPU 側は NaN 汚染）。オラクルを
 * GPU 側に合わせて緩めると「契約違反のグラフが両側で同じ値になる」ので突合が違反を
 * 検出できなくなる。
 * NOTE: attrs の `padding_idx` は forward に効かないので受け取らない（契約 — src/ops.ts）。
 */
export const referenceEmbedding = (weight: RefTensor, index: RefTensor): RefTensor => {
  const contract = resolveOpContract("embedding");
  assertSlotDtype(contract, 0, weight.dtype, "reference");
  assertSlotDtype(contract, 1, index.dtype, "reference");
  const shape = computeOutputShape(contract, [weight.shape, index.shape], "reference")[0];
  const vocab = weight.shape[0];
  const hidden = weight.shape[1];
  const out = new Float32Array(numel(shape));
  for (let row = 0; row < index.data.length; row += 1) {
    const pick = index.data[row];
    if (pick < 0 || pick >= vocab) {
      throw new ReferenceOpError(
        `embedding の添字 index[${row}] = ${pick} が語彙数 ${vocab} の範囲外`,
      );
    }
    for (let col = 0; col < hidden; col += 1) {
      out[row * hidden + col] = weight.data[pick * hidden + col];
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * masked_fill `out = mask ? value : x`（mask は右詰め broadcast）。
 *
 * 添字の導出は {@link broadcastIndex} — 「出力座標を毎要素 divmod → 入力 shape を左から
 * Horner」で、codegen の stride 表とは別構造（同ファイル冒頭の MUST）。埋め値は GPU 側の
 * f32 語に合わせて `Math.fround` で丸める。
 */
export const referenceMaskedFill = (
  x: RefTensor,
  mask: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => {
  const contract = resolveOpContract("masked_fill");
  assertSlotDtype(contract, 0, x.dtype, "reference");
  assertSlotDtype(contract, 1, mask.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape, mask.shape], "reference")[0];
  const fill = Math.fround(maskedFillValue(attrs, "reference"));
  const out = new Float32Array(numel(shape));
  for (let i = 0; i < out.length; i += 1) {
    out[i] = mask.data[broadcastIndex(i, shape, mask.shape)] !== 0 ? fill : x.data[i];
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * conv1d `out[b, oc, ox] = Σ_{ic,k} x[b, ic, ox·s + k·d − p] · w[oc, ic − g·Cin/groups, k] + b[oc]`
 * （`g` は `oc` が属するグループ — ADR 0015 で groups / dilation を attrs 化）。
 *
 * MUST: padding 域は**読み飛ばす**（0 を足す形にしない）。0 加算に潰すと、範囲外読みの
 * 分岐そのものが消えて添字が負でも通る実装を許してしまう。
 */
export const referenceConv1d = (
  x: RefTensor,
  weight: RefTensor,
  bias: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => {
  const contract = resolveOpContract("conv1d");
  for (const input of [x, weight, bias]) assertDtype(contract, input.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape, weight.shape, bias.shape], "reference", {
    attrs,
  })[0];
  const { stride, padding, dilation, groups } = conv1dAttrs(attrs, "reference");
  const [batch, channelsOut, lengthOut] = shape;
  const channelsIn = x.shape[1];
  const lengthIn = x.shape[2];
  const kernel = weight.shape[2];
  // 契約検査（computeOutputShape）で割り切れは済んでいる。
  const inPerGroup = channelsIn / groups;
  const outPerGroup = channelsOut / groups;
  const out = new Float32Array(numel(shape));
  for (let item = 0; item < batch; item += 1) {
    for (let oc = 0; oc < channelsOut; oc += 1) {
      // 重みの第 2 軸は Cin/groups — 入力チャネルはグループの帯だけを走る
      const icBase = Math.floor(oc / outPerGroup) * inPerGroup;
      for (let ox = 0; ox < lengthOut; ox += 1) {
        let acc = bias.data[oc];
        for (let icRel = 0; icRel < inPerGroup; icRel += 1) {
          for (let k = 0; k < kernel; k += 1) {
            const ix = ox * stride + k * dilation - padding;
            if (ix < 0 || ix >= lengthIn) continue;
            acc += x.data[(item * channelsIn + icBase + icRel) * lengthIn + ix] *
              weight.data[(oc * inPerGroup + icRel) * kernel + k];
          }
        }
        out[(item * channelsOut + oc) * lengthOut + ox] = Math.fround(acc);
      }
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * conv2d `out[b, oc, oy, ox] = Σ_{ic,kh,kw}
 *   x[b, ic, oy·sh + kh·dh − ph, ox·sw + kw·dw − pw] · w[oc, ic − g·Cin/groups, kh, kw] + b[oc]`
 * （`g` は `oc` が属するグループ — ADR 0017）。
 *
 * MUST: 重みは `[Cout, Cin/groups, Kh, Kw]` で、**Kh と Kw の順**も契約。正方カーネル
 * （Kh == Kw）では入れ替えても値が一致するので、テストは Kh ≠ Kw で固定する。
 * MUST: padding 域は**読み飛ばす**（0 を足す形にしない）。0 加算に潰すと、範囲外読みの
 * 分岐そのものが消えて添字が負でも通る実装を許してしまう。
 * MUST: 添字は H / W を独立に組む（`(y·W + x)` の平坦化を先に済ませない）。1 本に潰すと
 * 「H と W を取り違えた stride」が正方入力で一致してしまう。
 */
export const referenceConv2d = (
  x: RefTensor,
  weight: RefTensor,
  bias: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => {
  const contract = resolveOpContract("conv2d");
  for (const input of [x, weight, bias]) assertDtype(contract, input.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape, weight.shape, bias.shape], "reference", {
    attrs,
  })[0];
  const { stride, padding, dilation, groups } = conv2dAttrs(attrs, "reference");
  const [batch, channelsOut, heightOut, widthOut] = shape;
  const channelsIn = x.shape[1];
  const heightIn = x.shape[2];
  const widthIn = x.shape[3];
  const kernelH = weight.shape[2];
  const kernelW = weight.shape[3];
  // 契約検査（computeOutputShape）で割り切れは済んでいる。
  const inPerGroup = channelsIn / groups;
  const outPerGroup = channelsOut / groups;
  const out = new Float32Array(numel(shape));
  for (let item = 0; item < batch; item += 1) {
    for (let oc = 0; oc < channelsOut; oc += 1) {
      // 重みの第 2 軸は Cin/groups — 入力チャネルはグループの帯だけを走る
      const icBase = Math.floor(oc / outPerGroup) * inPerGroup;
      for (let oy = 0; oy < heightOut; oy += 1) {
        for (let ox = 0; ox < widthOut; ox += 1) {
          let acc = bias.data[oc];
          for (let icRel = 0; icRel < inPerGroup; icRel += 1) {
            const plane = (item * channelsIn + icBase + icRel) * heightIn;
            const weightPlane = (oc * inPerGroup + icRel) * kernelH;
            for (let kh = 0; kh < kernelH; kh += 1) {
              const iy = oy * stride[0] + kh * dilation[0] - padding[0];
              if (iy < 0 || iy >= heightIn) continue;
              for (let kw = 0; kw < kernelW; kw += 1) {
                const ix = ox * stride[1] + kw * dilation[1] - padding[1];
                if (ix < 0 || ix >= widthIn) continue;
                acc += x.data[(plane + iy) * widthIn + ix] *
                  weight.data[(weightPlane + kh) * kernelW + kw];
              }
            }
          }
          out[((item * channelsOut + oc) * heightOut + oy) * widthOut + ox] = Math.fround(acc);
        }
      }
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * conv_transpose1d `out[b, oc, ox] = Σ_{ic,k} x[b, ic, (ox + p − k)/s] · w[ic, oc, k] + b[oc]`
 * （`(ox + p − k)` が `s` で割り切れ、商が `[0, L)` に入る `(ic, k)` のみ — ADR 0015）。
 *
 * MUST: 重みは `[Cin, Cout, K]`（conv1d と転置）。
 * MUST: 割り切れない `k` は**読み飛ばす**（そこに入力が無い＝寄与ゼロ）。0 を足す形に
 * 潰すと、割り切れ判定そのものが消えた実装が通ってしまう。
 */
export const referenceConvTranspose1d = (
  x: RefTensor,
  weight: RefTensor,
  bias: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => {
  const contract = resolveOpContract("conv_transpose1d");
  for (const input of [x, weight, bias]) assertDtype(contract, input.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape, weight.shape, bias.shape], "reference", {
    attrs,
  })[0];
  const { stride, padding } = convTranspose1dAttrs(attrs, "reference");
  const [batch, channelsOut, lengthOut] = shape;
  const channelsIn = x.shape[1];
  const lengthIn = x.shape[2];
  const kernel = weight.shape[2];
  const out = new Float32Array(numel(shape));
  for (let item = 0; item < batch; item += 1) {
    for (let oc = 0; oc < channelsOut; oc += 1) {
      for (let ox = 0; ox < lengthOut; ox += 1) {
        let acc = bias.data[oc];
        for (let ic = 0; ic < channelsIn; ic += 1) {
          for (let k = 0; k < kernel; k += 1) {
            const shifted = ox + padding - k;
            if (shifted < 0 || shifted % stride !== 0) continue;
            const ix = shifted / stride;
            if (ix >= lengthIn) continue;
            acc += x.data[(item * channelsIn + ic) * lengthIn + ix] *
              weight.data[(ic * channelsOut + oc) * kernel + k];
          }
        }
        out[(item * channelsOut + oc) * lengthOut + ox] = Math.fround(acc);
      }
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * DCNv2（`torchvision::deform_conv2d`）の CPU 参照 — ADR 0055。
 *
 * `out[b, oc, oy, ox] = Σ_{ic,kh,kw} (m[b, kh·Kw+kw, oy, ox] · bilinear(x[b,ic], y, x)) ·
 *   w[oc, ic, kh, kw] + bias[oc]`、`y = (oy − ph) + kh + off_y` / `x = (ox − pw) + kw + off_x`
 * （stride / dilation / groups / offset_groups は欄が無い = 1 固定）。
 *
 * MUST: 縮約は `(ic, kh, kw)` 昇順で、conv2d 参照と**同じ入れ子**にする。offset を全 0・
 * mask を全 1 にした退化ケースが conv2d と一致することが本 op の主オラクルなので、
 * ループを組み替えると（意味論は同じでも）その門が丸め差で崩れる。
 * MUST: offset / mask の読みを `ic` ループの外へ巻き上げない（同上）。
 * MUST: 源座標は `Math.fround` で f32 へ丸める。**タップの採否**（`(−1, in)` の範囲判定と
 * 4 隅個別のゼロ埋め）は丸め幅で変わりうるので、ここだけは f64 のまま進めない。
 * MUST: 範囲判定は**正の形**（`> −1 && < in`）で書く。NaN は false 側へ落ちるので、
 * NaN だけを先に見て**出力へ伝播**させる（0 に落とすと沈黙誤値になる — ADR 0020 の系譜）。
 * MUST: 境界外は **border clamp ではなくゼロ埋め**（中心が範囲外ならタップ全体 0・内側でも
 * 範囲外の隅はその隅だけ 0）。torchvision の 2 段構えを逐語で写す。
 */
export const referenceDeformConv2d = (
  x: RefTensor,
  weight: RefTensor,
  offset: RefTensor,
  mask: RefTensor,
  bias: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => {
  const contract = resolveOpContract("deform_conv2d");
  for (const input of [x, weight, offset, mask, bias]) {
    assertDtype(contract, input.dtype, "reference");
  }
  const shape = computeOutputShape(
    contract,
    [x.shape, weight.shape, offset.shape, mask.shape, bias.shape],
    "reference",
    { attrs },
  )[0];
  const { padding } = deformConv2dAttrs(attrs, "reference");
  const [batch, channelsOut, heightOut, widthOut] = shape;
  const channelsIn = x.shape[1];
  const heightIn = x.shape[2];
  const widthIn = x.shape[3];
  const kernelH = weight.shape[2];
  const kernelW = weight.shape[3];
  const taps = kernelH * kernelW;
  const out = new Float32Array(numel(shape));
  /** `x[item, channel]` 平面の双線形サンプル（範囲外は 0・NaN は伝播）。 */
  const sample = (plane: number, sourceY: number, sourceX: number): number => {
    if (Number.isNaN(sourceY) || Number.isNaN(sourceX)) return Number.NaN;
    if (
      !(sourceY > -1 && sourceY < heightIn && sourceX > -1 && sourceX < widthIn)
    ) {
      return 0;
    }
    const y0 = Math.floor(sourceY);
    const x0 = Math.floor(sourceX);
    const ly1 = Math.fround(sourceY - y0);
    const lx1 = Math.fround(sourceX - x0);
    const ly0 = 1 - ly1;
    const lx0 = 1 - lx1;
    // 4 隅は**個別に**範囲を見る（clamp ではない）。y0 は範囲判定の帰結で [−1, H−1]。
    const at = (y: number, columnX: number): number =>
      y >= 0 && y < heightIn && columnX >= 0 && columnX < widthIn
        ? x.data[plane + y * widthIn + columnX]
        : 0;
    return ly0 * lx0 * at(y0, x0) + ly0 * lx1 * at(y0, x0 + 1) +
      ly1 * lx0 * at(y0 + 1, x0) + ly1 * lx1 * at(y0 + 1, x0 + 1);
  };
  for (let item = 0; item < batch; item += 1) {
    for (let oc = 0; oc < channelsOut; oc += 1) {
      for (let oy = 0; oy < heightOut; oy += 1) {
        for (let ox = 0; ox < widthOut; ox += 1) {
          let acc = bias.data[oc];
          for (let ic = 0; ic < channelsIn; ic += 1) {
            const plane = (item * channelsIn + ic) * heightIn * widthIn;
            for (let kh = 0; kh < kernelH; kh += 1) {
              for (let kw = 0; kw < kernelW; kw += 1) {
                const tap = kh * kernelW + kw;
                const pixel = oy * widthOut + ox;
                // offset のチャネル並びは **偶数 = y / 奇数 = x**（torchvision 逐語）。
                const offsetBase = (item * 2 * taps + 2 * tap) * heightOut * widthOut;
                const sourceY = Math.fround(
                  oy - padding[0] + kh + offset.data[offsetBase + pixel],
                );
                const sourceX = Math.fround(
                  ox - padding[1] + kw + offset.data[offsetBase + heightOut * widthOut + pixel],
                );
                // modulator は**補間の後**に掛かる（`(m · v) · w` — 括り方まで torchvision と同じ）。
                const modulated = mask.data[(item * taps + tap) * heightOut * widthOut + pixel] *
                  sample(plane, sourceY, sourceX);
                acc += modulated * weight.data[
                  ((oc * channelsIn + ic) * kernelH + kh) * kernelW +
                  kw
                ];
              }
            }
          }
          out[((item * channelsOut + oc) * heightOut + oy) * widthOut + ox] = Math.fround(acc);
        }
      }
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * 1 軸ぶんの双線形タップ表（`align_corners = True`）。
 * torch の `area_pixel_compute_scale` + `compute_source_index_and_lambda` の直訳。
 *
 * MUST: scale と源座標は `Math.fround` で f32 へ丸める。**添字の決定**（`trunc` の結果と
 * 末尾特例）は丸め幅で変わりうるので、ここだけは f64 のまま進めない — 許容差では吸収
 * できない 1 ずれになる。
 */
const bilinearTaps = (
  lengthIn: number,
  lengthOut: number,
): readonly { readonly low: number; readonly high: number; readonly lambda: number }[] => {
  const scale = lengthOut > 1 ? Math.fround((lengthIn - 1) / (lengthOut - 1)) : 0;
  return Array.from({ length: lengthOut }, (_unused, index) => {
    const source = Math.fround(scale * index);
    // 源座標は非負なので floor = trunc（torch の static_cast<int64_t> と同じ結果）。
    const low = Math.floor(source);
    return {
      low,
      // 末尾は 2 タップ目を持たない（`index1 = index0`）。
      high: low + 1 < lengthIn ? low + 1 : low,
      lambda: Math.fround(source - low),
    };
  });
};

/**
 * NCHW の空間 2 軸の双線形 resample（`align_corners = True` 専業 — 第 1 層）。
 *
 * MUST: codegen（src/kernels/upsample-bilinear2d.ts）と添字の畳み方を共有しない。あちらは
 * N·C を 1 本の平面添字へ畳んで平坦添字から座標を復元する形なので、こちらは **4 重ループで
 * 座標を直に回す**（軸の取り違えと平面ずれを両側で相殺させないため）。タップ表を軸ごとに
 * 先に作るのも意図的な別形で、あちらは 1 要素ごとに inline で計算する。
 */
export const referenceUpsampleBilinear2d = (
  x: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => {
  const contract = resolveOpContract("upsample_bilinear2d");
  assertDtype(contract, x.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape], "reference", { attrs })[0];
  upsampleBilinear2dAttrs(attrs, "reference");
  const [batch, channels, heightOut, widthOut] = shape;
  const heightIn = x.shape[2];
  const widthIn = x.shape[3];
  const rows = bilinearTaps(heightIn, heightOut);
  const columns = bilinearTaps(widthIn, widthOut);
  const out = new Float32Array(numel(shape));
  let flat = 0;
  for (let item = 0; item < batch; item += 1) {
    for (let channel = 0; channel < channels; channel += 1) {
      const plane = (item * channels + channel) * heightIn * widthIn;
      for (const row of rows) {
        for (const column of columns) {
          const at = (y: number, index: number): number => x.data[plane + y * widthIn + index];
          // H が外・W が内の入れ子（torch の式木と同じ組み方）。
          const top = (1 - column.lambda) * at(row.low, column.low) +
            column.lambda * at(row.low, column.high);
          const bottom = (1 - column.lambda) * at(row.high, column.low) +
            column.lambda * at(row.high, column.high);
          out[flat] = Math.fround((1 - row.lambda) * top + row.lambda * bottom);
          flat += 1;
        }
      }
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * GRU の隠れ側スキャン（`gru_scan` / `gru_scan_reverse` — 第 2 層・ADR 0056）。
 *
 * `gh = W_hh·h + b_hh` / `r = σ(gh_r + gi_r)` / `z = σ(gh_z + gi_z)` /
 * `n = tanh(gi_n + gh_n·r)` / `h' = (h − n)·z + n`。
 *
 * MUST: 更新式は `(h − n)·z + n` の**逐語**（数学的に同値な `(1 − z)·n + z·h` は f32 で別の
 * 丸め列になる — 実測で 10 万要素中 44,345 件が不一致）。ゲートの足し順と `n` の積の順も
 * torch の分解形に合わせる（ADR 0056 決定 3）。
 * MUST: ステップごとに `Math.fround` で f32 へ落とす。状態が T ステップ運ばれるので、f64 の
 * まま回すと GPU 側との差が T に比例して開き、突合の許容差が形依存になる。
 * MUST: 逆方向は**走査順だけ**を反転し、書き出しは順方向の時間添字（GPU カーネルと同じ契約）。
 * NOTE: 逐次の縮約はカーネルと同じ `k` 昇順だが、こちらは行を GEMM のタイルに割らない素の
 * 内積で、ゲート 3 本も**別々のループ**で回す（添字の畳み方を共有しない — 軸の取り違えを
 * 両側で相殺させないため）。
 */
export const referenceGruScan = (
  name: GruScanOpName,
  gi: RefTensor,
  initial: RefTensor,
  weight: RefTensor,
  bias: RefTensor,
): RefTensor => {
  const contract = resolveOpContract(name);
  for (const input of [gi, initial, weight, bias]) {
    assertDtype(contract, input.dtype, "reference");
  }
  const shape = computeOutputShape(
    contract,
    [gi.shape, initial.shape, weight.shape, bias.shape],
    "reference",
  )[0];
  const [time, batch, hidden] = shape;
  const gates = 3 * hidden;
  const out = new Float32Array(numel(shape));
  const f32 = Math.fround;
  /** ゲート `g`（0 = r / 1 = z / 2 = n）の隠れ側 `Σ_k W_hh[g·H + j, k]·h[k] + b_hh[g·H + j]`。 */
  const hiddenGate = (state: Float32Array, gate: number, unit: number): number => {
    const row = (gate * hidden + unit) * hidden;
    let acc = 0;
    for (let k = 0; k < hidden; k += 1) acc += weight.data[row + k] * state[k];
    // bias は last（GEMM の epilogue 形 — conv 系の bias-first ではない）
    return f32(acc + bias.data[gate * hidden + unit]);
  };
  for (let item = 0; item < batch; item += 1) {
    const state = new Float32Array(hidden);
    for (let unit = 0; unit < hidden; unit += 1) state[unit] = initial.data[item * hidden + unit];
    for (let step = 0; step < time; step += 1) {
      const t = name === "gru_scan_reverse" ? time - 1 - step : step;
      const inputBase = (t * batch + item) * gates;
      const next = new Float32Array(hidden);
      for (let unit = 0; unit < hidden; unit += 1) {
        // ゲートの足し順は**隠れ側が第 1 引数**（torch の分解形の逐語）
        const reset = f32(UNARY_F32.sigmoid(
          f32(hiddenGate(state, 0, unit) + gi.data[inputBase + unit]),
          NO_SCALARS,
        ));
        const update = f32(UNARY_F32.sigmoid(
          f32(hiddenGate(state, 1, unit) + gi.data[inputBase + hidden + unit]),
          NO_SCALARS,
        ));
        // n = tanh(i_n + h_n·r) — 積は `h_n · r`・和は入力側が第 1 引数
        const gated = f32(hiddenGate(state, 2, unit) * reset);
        const candidate = f32(UNARY_F32.tanh(
          f32(gi.data[inputBase + 2 * hidden + unit] + gated),
          NO_SCALARS,
        ));
        // h' = (h − n)·z + n
        const decayed = f32(f32(state[unit] - candidate) * update);
        next[unit] = f32(decayed + candidate);
      }
      for (let unit = 0; unit < hidden; unit += 1) {
        state[unit] = next[unit];
        out[(t * batch + item) * hidden + unit] = next[unit];
      }
    }
  }
  return { dtype: "f32", shape, data: out };
};

/** 単一出力 op のアーム（出力が 1 本であることをアームごとに明示する — ADR 0068 決定 1）。 */
const sole = (tensor: RefTensor): readonly RefTensor[] => [tensor];

/**
 * 契約表の kind で分岐する統一入口（テストがグラフをそのまま辿れるようにするため）。
 * 返りは**出力 slot 昇順の列**で、2 本返すのは `topk`（ADR 0068 決定 3）だけ。
 *
 * `outShape` は「出力の宣言 shape が目標形」の op（reshape / expand — ADR 0011）で必須。
 * 入力からは導けないので、省略されたら黙って推測せず落とす。
 */
export const applyReferenceOpOutputs = (
  op: string,
  inputs: readonly RefTensor[],
  attrs: Readonly<Record<string, unknown>> = {},
  outShape?: readonly number[],
): readonly RefTensor[] => {
  const contract: OpContract = resolveOpContract(op);
  if (!arityFits(contract, inputs.length)) {
    throw new ReferenceOpError(
      `op '${op}' の入力数が ${inputs.length}（契約は ${describeArity(contract)}）`,
    );
  }
  const declared = (): readonly number[] => {
    if (outShape === undefined) {
      throw new ReferenceOpError(`op '${op}' は出力 shape が目標形（outShape が要る）`);
    }
    return outShape;
  };
  switch (contract.kind) {
    case "unary":
      return sole(referenceUnary(contract.name, inputs[0], attrs));
    case "binary":
      return sole(referenceBinary(contract.name, inputs[0], inputs[1]));
    case "where":
      return sole(referenceWhere(inputs[0], inputs[1], inputs[2]));
    case "cumsum":
      return sole(referenceCumsum(inputs[0], attrs));
    case "matmul":
      return sole(referenceMatmul(inputs[0], inputs[1]));
    case "bmm":
      return sole(referenceBmm(inputs[0], inputs[1]));
    case "gather":
      return sole(referenceGather(inputs[0], inputs[1]));
    case "rowReduce":
      return sole(referenceRowReduce(contract.name, inputs[0], reduceDim(attrs, "reference")));
    case "argmax":
      return sole(referenceArgmax(inputs[0]));
    // 唯一の多出力アーム（値 + 添字 — ADR 0068 決定 3）。ここが `sole` を通らないことが
    // そのまま「出力が 2 本」の宣言になる。
    case "topk":
      return referenceTopk(inputs[0], attrs);
    case "cast":
      return sole(referenceCast(inputs[0], castTargetDtype(attrs, "reference")));
    case "reshape":
      return sole(referenceReshape(inputs[0], declared()));
    case "permute":
      return sole(referencePermute(inputs[0], permuteDims(attrs, "reference")));
    case "expand":
      return sole(referenceExpand(inputs[0], declared()));
    case "slice":
      return sole(referenceSlice(inputs[0], attrs));
    case "cat":
      return sole(referenceCat(inputs, attrs));
    case "pad":
      return sole(referencePad(inputs[0], attrs));
    case "flip":
      return sole(referenceFlip(inputs[0], attrs));
    case "symPrefixSlice":
      // prefix 長は束縛（`coeff·sym+offset`）から決まる — 入力からは導けないので、
      // reshape / expand と同じく呼び出し側が解決済みの出力 shape を渡す。
      return sole(referenceSymPrefixSlice(inputs[0], declared()));
    case "linear":
      return sole(referenceLinear(inputs[0], inputs[1], inputs[2]));
    case "layerNorm":
      return sole(referenceLayerNorm(inputs[0], inputs[1], inputs[2], attrs));
    case "rmsNorm":
      return sole(referenceRmsNorm(inputs[0], inputs[1], attrs));
    case "softmax":
      return sole(referenceSoftmax(inputs[0], attrs));
    case "safeSoftmax":
      return sole(referenceSafeSoftmax(inputs[0], attrs));
    case "attention":
      return sole(referenceAttention(inputs[0], inputs[1], inputs[2], attrs, inputs[3]));
    // state を触る op は CPU 参照を持たない（値ではなく context 所有バッファへの effect で、
    // RefTensor の入出力だけでは意味論を表せない — 実装は実行層〈波 D〉が持つ）。
    // MUST: 素通りさせない。0 本の出力列を黙って返すと「参照と GPU が一致した」と見なす
    // parity テストが書けてしまう。
    case "stateAppend":
      throw new ReferenceOpError(
        `op '${op}' は CPU 参照を持たない（state への effect は GenerationContext の担当）`,
      );
    case "embedding":
      return sole(referenceEmbedding(inputs[0], inputs[1]));
    case "maskedFill":
      return sole(referenceMaskedFill(inputs[0], inputs[1], attrs));
    case "conv1d":
      return sole(referenceConv1d(inputs[0], inputs[1], inputs[2], attrs));
    case "conv2d":
      return sole(referenceConv2d(inputs[0], inputs[1], inputs[2], attrs));
    case "convTranspose1d":
      return sole(referenceConvTranspose1d(inputs[0], inputs[1], inputs[2], attrs));
    case "deformConv2d":
      return sole(referenceDeformConv2d(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        attrs,
      ));
    case "upsampleBilinear2d":
      return sole(referenceUpsampleBilinear2d(inputs[0], attrs));
    case "gruScan":
      return sole(referenceGruScan(contract.name, inputs[0], inputs[1], inputs[2], inputs[3]));
  }
};

/**
 * 単一出力 op の統一入口（{@link applyReferenceOpOutputs} の slot 0）。
 *
 * MUST: 多出力 op を黙って slot 0 だけ返す形にしない。`topk` を `applyReferenceOp` で呼ぶと
 * 「値だけが合っていて添字は誰も突き合わせていない」テストが書けてしまうので、本数が違えば
 * 落として複数形の入口へ誘導する。
 */
export const applyReferenceOp = (
  op: string,
  inputs: readonly RefTensor[],
  attrs: Readonly<Record<string, unknown>> = {},
  outShape?: readonly number[],
): RefTensor => {
  const outputs = applyReferenceOpOutputs(op, inputs, attrs, outShape);
  if (outputs.length !== 1) {
    throw new ReferenceOpError(
      `op '${op}' は出力 ${outputs.length} 本（多出力は applyReferenceOpOutputs で受ける）`,
    );
  }
  return outputs[0];
};
