/**
 * reduce（1 軸・keepdim 無し）カーネルの WGSL 生成。**2 変種**を持つ:
 *
 * - **行 reduce**（最終次元）: 入力を先行次元を平坦化した `[rows, dim]` の連続レイアウトとして
 *   扱う。1 行 = 1 workgroup(256) + 256 幅のビット反転二分木。
 * - **軸 reduce**（最終次元以外）: 1 スレッド = 1 出力。縮約軸の送りは `inner`（= 縮約軸より
 *   後ろの次元の積）で、**隣接スレッドが連続アドレスを読む**（コアレス）。
 *
 * MUST: 軸変種の**縮約順序は行 reduce と厳密に一致させる**（設計と記号検証は
 * docs/research/2026-08-04-vae-axis-reduce-recon.md §5.2）。permute + 行 reduce から軸 reduce へ
 * 置き換えても出力がビット同一であることが、この族の数値契約（PNG sha256 門・
 * tests/gpu_reduce_axis_parity_test.ts の atol=0）を成立させている。
 *
 * MUST: 生成は決定的（同一キー → バイト単位同一 WGSL）。キーと WGSL は同じ正準化から作る。
 * MUST: 1 行 = 1 workgroup を基本にしつつ、**行方向も grid-stride** で回す。行数をそのまま
 * dispatch 数にすると 1 次元の workgroup 上限（仕様既定 65535）を実モデルが超える。
 * MUST: 要素型（入力・出力とも）をキーに載せる。`sum` は f32 の総和と **bool の個数
 * カウント（→ i32）** の 2 形を持ち、載せないと片方の dispatch にもう片方のパイプラインが
 * 割り当たってビット列の読み替えが例外なしに通る（ADR 0009）。
 *
 * params を uniform で渡すのは、行ループの中に workgroupBarrier があるため。ループ条件が
 * workgroup 内で一様である必要があり、storage からのロードは一様性解析で保証されない。
 *
 * MUST: `amax` / `amin` は縮約対象に NaN があれば NaN を返す（torch と同じ）。ドライバの
 * `max` / `min` は NaN を飲むので、伝播はビット列判定の {@link IS_NAN_FN} が担う。
 */

import type { IrDtype } from "../format/ir.ts";
import { outputDtypeOf, REDUCE_OPS, type ReduceOpName, resolveOpContract } from "../ops.ts";
import { CodegenError } from "./errors.ts";
import { assertU32Params } from "./params.ts";

const REDUCE_WORKGROUP_SIZE = 256;

export type ReduceSpec = {
  readonly op: ReduceOpName;
  /** 入力の意味論 dtype（出力は契約表の写像で決まる）。 */
  readonly dtype: IrDtype;
};

type CanonicalSpec = {
  readonly op: ReduceOpName;
  readonly from: IrDtype;
  readonly to: IrDtype;
};

const canonicalize = (spec: ReduceSpec): CanonicalSpec => {
  const op = REDUCE_OPS.find((name) => name === spec.op);
  if (op === undefined) {
    throw new CodegenError(`reduce codegen: op '${spec.op}' は reduce 語彙に無い`);
  }
  const contract = resolveOpContract(op);
  if (!contract.dtypes.includes(spec.dtype)) {
    throw new CodegenError(`reduce codegen: op '${op}' は dtype '${spec.dtype}' を実行できない`);
  }
  return { op, from: spec.dtype, to: outputDtypeOf(contract, spec.dtype, "reduce codegen") };
};

/** 意味論 dtype → WGSL のスカラ型。bool の格納は u32 の 0 / 1（ADR 0009）。 */
const WGSL_SCALAR: Readonly<Record<IrDtype, string>> = {
  f32: "f32",
  i32: "i32",
  bool: "u32",
};

/** f32 の最大有限値。WGSL に無限大リテラルが無いため identity にこれを使う。 */
const F32_MAX = "3.402823466e38";

const IDENTITY: Readonly<Record<ReduceOpName, Readonly<Partial<Record<IrDtype, string>>>>> = {
  sum: { f32: "0.0", i32: "0i" },
  amax: { f32: `-${F32_MAX}` },
  amin: { f32: F32_MAX },
};

/**
 * f32 の NaN を**ビット列**で判定する（符号を落として指数部全 1 + 仮数部非 0）。
 *
 * MUST: 浮動小数の比較で NaN を判定しない。比較単体は仕様どおり false になるのに、
 * `select(x, m, x < m)` 全体はシェーダコンパイラが `max` イディオムへ畳み、ドライバの
 * `max` が NaN を飲む（実測・2026-08-02 / 本リポジトリの検証環境）。整数の `&` と `>` は
 * その畳み込みの対象にならない。src/codegen/elementwise.ts と同じ判定式。
 */
const IS_NAN_FN = `fn is_nan_bits(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}`;

/**
 * NaN 伝播する `max` / `min`。**縮約は 2 段（1 スレッドの走査 + workgroup の木）**なので、
 * 伝播は畳み込み関数の側で閉じる（片方の段だけ守っても NaN は identity に飲まれる）。
 * 非 NaN のときは素の `max` / `min` そのままで、既存の数値結果は動かない。
 */
const NAN_PROPAGATING_FN: Readonly<Partial<Record<ReduceOpName, string>>> = {
  amax: `fn nan_max(a: f32, b: f32) -> f32 {
  return select(select(max(a, b), b, is_nan_bits(b)), a, is_nan_bits(a));
}`,
  amin: `fn nan_min(a: f32, b: f32) -> f32 {
  return select(select(min(a, b), b, is_nan_bits(b)), a, is_nan_bits(a));
}`,
};

const COMBINE: Readonly<Record<ReduceOpName, (a: string, b: string) => string>> = {
  sum: (a, b) => `${a} + ${b}`,
  amax: (a, b) => `nan_max(${a}, ${b})`,
  amin: (a, b) => `nan_min(${a}, ${b})`,
};

/**
 * 累算器へ載せるときの読み方。bool 入力は「真の個数」を数えるので 0 / 1 の i32 に落とす
 * （格納規約が破れても真偽で数える形 — `i32(x)` はビット列をそのまま整数にする）。
 */
const load = (expr: string, from: IrDtype): string =>
  from === "bool" ? `select(0i, 1i, ${expr} != 0u)` : expr;

/**
 * v2: amax / amin に NaN 伝播の畳み込み関数が入り WGSL 本文が変わった。版は **族ごと**に
 * 上げる（`sum` はキーが変わるだけで生成物は同一 — op 別の例外表を持たない）。
 */
export const reduceKey = (spec: ReduceSpec): string => {
  const { op, from, to } = canonicalize(spec);
  return `reduce:v2:${op}:${from}>${to}:wg${REDUCE_WORKGROUP_SIZE}`;
};

export const reduceWgsl = (spec: ReduceSpec): string => {
  const { op, from, to } = canonicalize(spec);
  const combine = COMBINE[op];
  const identity = IDENTITY[op][to];
  if (identity === undefined) {
    throw new CodegenError(
      `reduce codegen: op '${op}' は出力 dtype '${to}' の identity を持たない`,
    );
  }
  const accumulator = WGSL_SCALAR[to];
  // 同型なら 1 語で書く（f32 形の生成物はスナップショットでバイト単位に凍結されている）。
  const label = from === to ? from : `${from}>${to}`;
  const propagate = NAN_PROPAGATING_FN[op];
  const helpers = propagate === undefined ? [] : [IS_NAN_FN, propagate];
  return [
    `// karume row reduce ${op} (last dim, ${label}, generated)`,
    "struct Params {",
    "  rows: u32,",
    "  dim: u32,",
    "}",
    "@group(0) @binding(0) var<uniform> params: Params;",
    `@group(0) @binding(1) var<storage, read> x: array<${WGSL_SCALAR[from]}>;`,
    `@group(0) @binding(2) var<storage, read_write> out: array<${accumulator}>;`,
    "",
    ...helpers.flatMap((fn) => [fn, ""]),
    `var<workgroup> scratch: array<${accumulator}, ${REDUCE_WORKGROUP_SIZE}>;`,
    "",
    `@compute @workgroup_size(${REDUCE_WORKGROUP_SIZE})`,
    "fn main(",
    "  @builtin(workgroup_id) wid: vec3<u32>,",
    "  @builtin(local_invocation_id) lid3: vec3<u32>,",
    "  @builtin(num_workgroups) nwg: vec3<u32>,",
    ") {",
    "  let lid = lid3.x;",
    "  let dim = params.dim;",
    "  var row = wid.x;",
    "  while (row < params.rows) {",
    "    let base = row * dim;",
    `    var acc = ${identity};`,
    "    var i = lid;",
    "    while (i < dim) {",
    `      acc = ${combine("acc", load("x[base + i]", from))};`,
    `      i = i + ${REDUCE_WORKGROUP_SIZE}u;`,
    "    }",
    "    scratch[lid] = acc;",
    "    workgroupBarrier();",
    `    var stride = ${REDUCE_WORKGROUP_SIZE / 2}u;`,
    "    while (stride > 0u) {",
    "      if (lid < stride) {",
    `        scratch[lid] = ${combine("scratch[lid]", "scratch[lid + stride]")};`,
    "      }",
    "      workgroupBarrier();",
    "      stride = stride / 2u;",
    "    }",
    "    if (lid == 0u) {",
    "      out[row] = scratch[0u];",
    "    }",
    "    // 次の行が scratch[lid] を上書きする前に scratch[0] の読み終わりを揃える",
    "    workgroupBarrier();",
    "    row = row + nwg.x;",
    "  }",
    "}",
    "",
  ].join("\n");
};

/**
 * uniform の Params。WGSL の uniform アドレス空間では struct の整列が 16 バイトになるため、
 * 2 語ぶんの内容でも 16 バイト確保する MUST（不足すると binding が validation で落ちる）。
 */
export const reduceParams = (rows: number, dim: number): Uint32Array<ArrayBuffer> => {
  assertU32Params("reduce params", { rows, dim });
  const params = new Uint32Array(4);
  params[0] = rows;
  params[1] = dim;
  return params;
};

/** 軸 reduce の並列形（1 スレッド = 1 出力）。行 reduce と同じ 256。 */
export const AXIS_REDUCE_WORKGROUP_SIZE = 256;

/**
 * 軸 reduce の carry-stack の段数。256 回 push した後に有効なのは最上段 1 つだけなので、
 * `log2(256) + 1 = 9` 段要る。
 */
const AXIS_REDUCE_STACK_DEPTH = 9;

/**
 * 軸 reduce 変種のキー。行 reduce（{@link reduceKey}）とは**別のパイプライン**で、既存の
 * キー文字列は 1 バイトも変えない（既定経路の WGSL / キーを動かさないための別変種化）。
 */
export const axisReduceKey = (spec: ReduceSpec): string => {
  const { op, from, to } = canonicalize(spec);
  return `reduce:v2:${op}:${from}>${to}:axis:wg${AXIS_REDUCE_WORKGROUP_SIZE}`;
};

/**
 * 最終次元以外の軸を縮約する変種。1 スレッドが 1 出力を担当し、縮約軸を `inner` 送りで走査する。
 *
 * 縮約順序の再現（**MUST** — ビット同一の根拠）:
 *
 * 1. 行 reduce は「レーン `i` が `x[i], x[i+256], x[i+512], …` を identity から左結合で畳む」
 *    ので、ここでも slot `i` を同じ順序・同じ identity 始まりで作る（`+0.0` の畳み込みを
 *    省略しない — 行 reduce も identity を実際に足すので、`-0.0 + 0.0 = +0.0` の 1 例外まで
 *    同じ挙動になる）。
 * 2. 行 reduce の木は `scratch[lid] = combine(scratch[lid], scratch[lid + stride])`
 *    （stride = 128 → 1）なので、最初に組む対は (0,128) / (1,129) … = **葉の順序がビット反転**。
 *    slot を `reverseBits` の順に carry-stack（2 進カウンタ）へ push すると、この木と
 *    記号式が完全一致する（dim 1〜512 の 21 ケースで検証済み — recon §5.2）。
 * 3. combine の引数順は「先に push した側が第 1 引数」（木の `scratch[lid]` が第 1 引数）。
 *    `amax` / `amin` の NaN 伝播は対称なので値は変わらないが、`sum` の丸めは順序で変わる。
 */
export const axisReduceWgsl = (spec: ReduceSpec): string => {
  const { op, from, to } = canonicalize(spec);
  const combine = COMBINE[op];
  const identity = IDENTITY[op][to];
  if (identity === undefined) {
    throw new CodegenError(
      `reduce codegen: op '${op}' は出力 dtype '${to}' の identity を持たない`,
    );
  }
  const accumulator = WGSL_SCALAR[to];
  const label = from === to ? from : `${from}>${to}`;
  const propagate = NAN_PROPAGATING_FN[op];
  const helpers = propagate === undefined ? [] : [IS_NAN_FN, propagate];
  const size = AXIS_REDUCE_WORKGROUP_SIZE;
  return [
    `// karume axis reduce ${op} (non-last dim, ${label}, generated)`,
    "struct Params {",
    "  out_count: u32,",
    "  axis_len: u32,",
    "  inner: u32,",
    "}",
    "@group(0) @binding(0) var<uniform> params: Params;",
    `@group(0) @binding(1) var<storage, read> x: array<${WGSL_SCALAR[from]}>;`,
    `@group(0) @binding(2) var<storage, read_write> out: array<${accumulator}>;`,
    "",
    ...helpers.flatMap((fn) => [fn, ""]),
    `@compute @workgroup_size(${size})`,
    "fn main(",
    "  @builtin(global_invocation_id) gid: vec3<u32>,",
    "  @builtin(num_workgroups) nwg: vec3<u32>,",
    ") {",
    "  let inner = params.inner;",
    "  let axis_len = params.axis_len;",
    `  let stride = nwg.x * ${size}u;`,
    "  var t = gid.x;",
    "  while (t < params.out_count) {",
    "    // 出力添字 t = outer·inner + i。縮約軸だけを外した先頭アドレスへ戻す。",
    "    let base = (t / inner) * axis_len * inner + (t % inner);",
    `    var acc: array<${accumulator}, ${AXIS_REDUCE_STACK_DEPTH}>;`,
    `    for (var m = 0u; m < ${size}u; m = m + 1u) {`,
    "      // 行 reduce の葉の並び（ビット反転）を 1 スレッドで再現する",
    "      let slot = reverseBits(m) >> 24u;",
    `      var v = ${identity};`,
    "      var c = slot;",
    "      while (c < axis_len) {",
    `        v = ${combine("v", load("x[base + c * inner]", from))};`,
    `        c = c + ${size}u;`,
    "      }",
    "      var k = 0u;",
    "      while ((m & (1u << k)) != 0u) {",
    `        v = ${combine("acc[k]", "v")};`,
    "        k = k + 1u;",
    "      }",
    "      acc[k] = v;",
    "    }",
    `    out[t] = acc[${AXIS_REDUCE_STACK_DEPTH - 1}u];`,
    "    t = t + stride;",
    "  }",
    "}",
    "",
  ].join("\n");
};

/**
 * 軸 reduce の uniform Params（`{out_count, axis_len, inner}`・16 バイト整列）。
 *
 * `inner` は縮約軸より後ろの次元の積。0 になるのは後続の軸に長さ 0 があるときだけで、
 * そのとき `out_count` も 0 なので `t / inner` は 1 度も評価されない（ループに入らない）。
 */
export const axisReduceParams = (
  outCount: number,
  axisLen: number,
  inner: number,
): Uint32Array<ArrayBuffer> => {
  assertU32Params("axis reduce params", {
    out_count: outCount,
    axis_len: axisLen,
    inner,
  });
  const params = new Uint32Array(4);
  params[0] = outCount;
  params[1] = axisLen;
  params[2] = inner;
  return params;
};
