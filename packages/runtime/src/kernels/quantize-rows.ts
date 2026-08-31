/**
 * 活性の **per-token symmetric int8 量子化**（w8a8 実行経路の前段 — ADR 0025 予定 /
 * 設計は docs/research/2026-08-03-dp4a-w8a8-design.md §4.2）。
 *
 * `[rows, dim]` の連続レイアウトを行（= token）ごとに量子化して 2 本の出力を書く:
 *
 * ```
 * s        = max(rowmax(|x|) / 127, f32 tiny)      … xs[row]（per-token scale・f32）
 * q        = clamp(round(x / s), -127, +127)       … xq[row·dim/4 + …]（i8 を 4 詰めた u32）
 * ```
 *
 * 形は行 reduce 族と同型（**1 行 = 1 workgroup（256）・行方向は grid-stride**）で、行を
 * 2 回走る（パス① absmax → パス② 量子化）。行をもう一度グローバルから読むのは、行長ぶんの
 * 共有メモリを持てないため（実測形の最大は 1 行 32KB）。
 *
 * MUST: **±127 に閉じる**（−128 を使わない）。ADR 0019 の重み側と同じ規約で、絶対値最大の
 * 要素が厳密に復元され fake-quant が冪等になる。`pack4xI8Clamp`（[−128,127] へ飽和）には
 * 頼らない — 明示 `clamp(±127)` の後に `pack4xI8` を使う。
 * NOTE（検出限界・故障注入で実測）: この `clamp` の**枝そのものは有限データでは到達しない**。
 * `s = amax/127` なので `|x/s| ≤ 127` が構造的に成り立ち、丸め誤差を足しても 127.5 を超え
 * られないため。つまり「格子を ±128 にする」退行の検出器は clamp ではなく **scale の分母**で、
 * `clamp` の境界だけを ±128 に変えても全テストが緑のまま通る（分母を 128 にすると 6 本が赤）。
 * clamp を残すのは非有限値と将来の `s` 変更に対する防御で、数値契約の担い手ではない。
 * MUST: absmax の NaN 伝播は**ビット列判定**（ADR 0020）。ドライバの `max` は NaN を飲むので、
 * 素の `max` に頼ると NaN 行の scale が有限値になり、行全体の NaN が静かに消える。
 * `s` を作る `max(…, tiny)` も同じ理由で NaN を外殻の `select` で通す。
 * MUST: `dim % 4 == 0` を呼び出し側が保証する（i8 ペイロードは平坦添字 4 詰めで、行頭が
 * 語境界に来る条件そのもの — 適格判定は src/runtime/executor.ts）。ここでは端数を持たない
 * 前提で `dim / 4` 語を全て書く（full-write）。
 *
 * NOTE（実測・2026-08-03 / 本機 Deno 2.9.4 + RTX 3080 Ti）: WGSL の `round` は**偶数丸め**
 * （`round(0.5) = 0` / `round(1.5) = 2` / `round(-0.5) = -0` / `round(2.5) = 2`）で、
 * `torch.round` と一致する。TS 参照は `Math.round`（half-up）ではなく
 * `roundTiesToEven`（src/reference/i8a8.ts）を使う。
 *
 * ## 除算の精度（実測で判明した設計上の制約・2026-08-03）
 *
 * WGSL の f32 除算は**正しく丸められない**（仕様の許容は 2.5 ULP。本機の実測では
 * `a / b` は `a * rcp(b)` に落ち、IEEE の正しい丸めと **200,000 サンプル中 55,605 件**で
 * 1 ULP 割れた）。乗算・加算・`fma` は正しく丸められる（同実測で 0 件）。ここから 2 つ:
 *
 * - MUST: scale の `amax / 127` は **`amax * INV_127` の乗算で書く**（`INV_127` は 1/127 の
 *   f32 リテラル）。本機のコンパイラは定数除数を同じ乗算へ畳むので生成コードは変わらないが、
 *   畳まないドライバでは 2.5 ULP の除算に落ちて `xs` が参照と割れる。**ソースの側で
 *   決定性を持たせる**。
 * - `q` を作る `x / s` は除数がデータ依存なので置き換えられない。したがって **`x/s` が
 *   丸め境界（半整数）の近傍にある要素では量子化値が ±1 段揺れうる**。GPU と TS 参照の
 *   atol=0 突合は「境界から十分離れたデータ」でのみ成立する契約で、テスト側が余裕を
 *   実測して門にする（tests/gpu_i8a8_test.ts）。実モデルでは ±1 段の揺れがそのまま
 *   数値差として乗る（E2E の tolerance を実測導出する理由の 1 つ）。
 */

import { IS_NAN_BITS_WGSL } from "../codegen/elementwise.ts";
import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";

/** 1 行を畳む workgroup の幅（行 reduce / softmax と同じ 256）。 */
export const QUANTIZE_ROWS_WORKGROUP_SIZE = 256;

/** 量子化の格子の端（**−128 は使わない** — 上の MUST）。 */
export const QUANTIZE_ROWS_ABS_MAX = 127;

/** f32 の最小 normal（`torch.finfo(float32).tiny`）。全ゼロ行の scale の床。 */
export const F32_TINY = 1.1754943508222875e-38;

/**
 * 1/127 の**倍精度リテラル**（WGSL 側は f32 へ正しく丸めて読む）。scale を除算ではなく
 * この定数との乗算で作るための正本（上の MUST）。
 */
export const INV_ABS_MAX = 1 / QUANTIZE_ROWS_ABS_MAX;

export const QUANTIZE_ROWS_KEY =
  `quantize_rows:v1:f32>i8:pertoken:wg${QUANTIZE_ROWS_WORKGROUP_SIZE}`;

const NAN_MAX_FN = `fn nan_max(a: f32, b: f32) -> f32 {
  return select(select(max(a, b), b, is_nan_bits(b)), a, is_nan_bits(a));
}`;

export const QUANTIZE_ROWS_WGSL: string =
  `// karume quantize_rows (行ごとの per-token symmetric i8: s = max(amax/127, tiny), q = clamp(round(x/s), ±127))
struct Params {
  rows: u32,
  dim: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> xq: array<u32>;
@group(0) @binding(3) var<storage, read_write> xs: array<f32>;

${IS_NAN_BITS_WGSL}

${NAN_MAX_FN}

var<workgroup> scratch: array<f32, ${QUANTIZE_ROWS_WORKGROUP_SIZE}>;

@compute @workgroup_size(${QUANTIZE_ROWS_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let dim = params.dim;
  // 呼び出し側が dim % 4 == 0 を保証する（i8 ペイロードは平坦添字 4 詰め）
  let quads = dim / 4u;
  var row = wid.x;
  while (row < params.rows) {
    let base = row * dim;

    // ① 行の絶対値最大（NaN はビット列判定で伝播 — 素の max は NaN を飲む）
    var acc = 0.0;
    var i = lid;
    while (i < dim) {
      acc = nan_max(acc, abs(x[base + i]));
      i = i + ${QUANTIZE_ROWS_WORKGROUP_SIZE}u;
    }
    scratch[lid] = acc;
    workgroupBarrier();
    var stride = ${QUANTIZE_ROWS_WORKGROUP_SIZE / 2}u;
    while (stride > 0u) {
      if (lid < stride) {
        scratch[lid] = nan_max(scratch[lid], scratch[lid + stride]);
      }
      workgroupBarrier();
      stride = stride / 2u;
    }
    let amax = scratch[0u];
    // 全ゼロ行は s = tiny → q = 0 → 0·tiny = 0 で厳密。NaN は max に飲まれるので外殻で通す。
    // MUST: 127 での除算ではなく **1/127 との乗算**（乗算だけが正しく丸められる — 上の MUST）
    let s = select(max(amax * ${INV_ABS_MAX}, ${F32_TINY}), amax, is_nan_bits(amax));
    if (lid == 0u) {
      xs[row] = s;
    }

    // ② 4 連続要素（quad）ごとに量子化して 1 語へ詰める
    let qbase = row * quads;
    var q = lid;
    while (q < quads) {
      let e = base + q * 4u;
      let v = vec4<f32>(x[e], x[e + 1u], x[e + 2u], x[e + 3u]) / s;
      let r = clamp(round(v), vec4<f32>(-${QUANTIZE_ROWS_ABS_MAX}.0), vec4<f32>(${QUANTIZE_ROWS_ABS_MAX}.0));
      xq[qbase + q] = pack4xI8(vec4<i32>(r));
      q = q + ${QUANTIZE_ROWS_WORKGROUP_SIZE}u;
    }

    // 次の行が scratch[lid] を上書きする前に scratch[0] の読み終わりを揃える
    workgroupBarrier();
    row = row + nwg.x;
  }
}
`;

/**
 * uniform の Params（`{rows, dim}`）。WGSL の uniform アドレス空間では struct の整列が
 * 16 バイトになるため、2 語ぶんの内容でも 16 バイト確保する MUST（softmax / reduce と同じ）。
 */
export const quantizeRowsParams = (rows: number, dim: number): Uint32Array<ArrayBuffer> => {
  assertU32Params("quantize_rows params", { rows, dim });
  if (dim < 1) {
    throw new CodegenError(`quantize_rows params: dim は正整数（${dim}）`);
  }
  if (dim % 4 !== 0) {
    throw new CodegenError(`quantize_rows params: dim は 4 の倍数（${dim}）`);
  }
  const params = new Uint32Array(4);
  params[0] = rows;
  params[1] = dim;
  return params;
};
