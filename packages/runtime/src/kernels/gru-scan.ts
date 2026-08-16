/**
 * GRU の**隠れ側スキャン**（op `gru_scan` / `gru_scan_reverse`・第 2 層 — ADR 0056）の
 * f32 固定カーネル。入力側 GEMM は呼び手が既存 `linear` で用意する（この op は持たない）。
 *
 * ```
 * gh      = W_hh · h + b_hh                     // [3H]（bias は last = GEMM の epilogue 形）
 * r       = sigmoid(gh_r + gi_r)
 * z       = sigmoid(gh_z + gi_z)
 * n       = tanh(gi_n + gh_n · r)
 * h'      = (h − n) · z + n
 * ```
 *
 * MUST: 演算の**並びと括り方は torch の GRU 分解の逐語**（ADR 0056 決定 3）。数学的に同値な
 * `(1 − z)·n + z·h` は f32 で別の丸め列になる（10 万要素中 44,345 件がビット不一致の実測）。
 * ゲートの足し順（隠れ側が第 1 引数）と `n` の積の順（`gh_n · r`）も分解形に合わせる。
 * MUST: bias は **last**（`(Σ W·h) + b`）。conv 系の bias-first（ADR 0024）を写すと丸めの
 * 並びが変わり、分解経路の `linear` とのビット同一が崩れる。
 * MUST: 縮約は `k` 昇順の逐次で、字面も `acc = acc + w * h`（GEMM と同じ形 — src/kernels/gemm.ts
 * の `accumulatorUpdate`）。GEMM は K タイル 16 で分割するが 1 出力あたりの加算順は k 昇順の
 * ままなので、この素の逐次ループと丸め列が一致する。
 * MUST: `sigmoid` の本体は elementwise codegen と**同じ文字列**を共有する（silu.ts と同じ理由 —
 * 書き写すと primitive と融合版で丸め列が割れうる）。`tanh` は WGSL 組込なので共有の問題は無い。
 *
 * ## 丸め障壁（workgroup memory 往復）
 *
 * `a * b + c` を 1 式で書くと **fma へ縮約される**（本ワークツリーの実 GPU 実測: 65,536 要素中
 * 15,371 件が mul / add 2 dispatch の結果とビット不一致。`let p = a * b;` と名前を付けても
 * 結果は同じで、`bitcast` の往復でも止まらない）。分解経路は `mul` と `add` が別 dispatch =
 * storage 往復なので縮約されない。そこで **積を workgroup memory へ書いて barrier 後に読み戻す**
 * ことで明示的な materialization 点を残す（silu.ts の技法・同実測で不一致 0）。
 *
 * **この丸め障壁は WGSL 仕様の保証ではない** — 仕様は fusion を許すだけで、workgroup memory
 * 往復を最適化障壁として尊重することは要求していない。ビット同一門
 * （tests/gpu_gru_scan_parity_test.ts）が割れたら、まずここを疑うこと。
 *
 * ## 走査の裁定（性能ではなく正しさの都合）
 *
 * バッチ要素どうしの再帰は独立なので **1 workgroup = 1 バッチ要素**（バッチ方向は
 * layer-norm と同じ grid-stride で dispatch 上限を跨ぐ）。**1 lane = 1 隠れユニット**に固定し、
 * `H ≤ {@link GRU_SCAN_MAX_HIDDEN}` を超える形は {@link gruScanParams} が fail loudly にする
 * （ADR 0056 決定 5）。時間ループと barrier 群は workgroup 一様な制御流の中だけに置く
 * （dims は uniform — layer-norm.ts の MUST と同じ）。
 *
 * MUST: 逆方向は**走査順だけ**が逆で、書き出しは常に順方向の時間添字（`flip` を挟まない形が
 * この op の存在理由の 1 つ — ADR 0056 決定 2）。
 */

import { SIGMOID_STABLE_WGSL } from "../codegen/elementwise.ts";
import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";

export const GRU_SCAN_WORKGROUP_SIZE = 256;

/**
 * 1 lane = 1 隠れユニットの割り当て上限（= workgroup サイズ）。
 *
 * MUST: 超過は {@link gruScanParams} が `CodegenError` で落とす（黙って縮退させない）。
 * 実測に出ている形は H = 128（vowel-detector の 2 層 BiGRU）だけで、上限を上げるには
 * workgroup 内 grid-stride と `h` の二重化が要る = 別の設計判断（ADR 0056 決定 5）。
 */
export const GRU_SCAN_MAX_HIDDEN = GRU_SCAN_WORKGROUP_SIZE;

/** 走査方向（`gru_scan` / `gru_scan_reverse` の 2 op に 1 対 1 で対応する）。 */
export type GruScanDirection = "forward" | "reverse";

const canonicalizeDirection = (direction: GruScanDirection): GruScanDirection => {
  if (direction === "forward" || direction === "reverse") return direction;
  throw new CodegenError(`gru_scan codegen: 走査方向が不正（${direction}）`);
};

/** MUST: WGSL を変えたらキーも上げる（パイプラインキャッシュは本文を見ない）。 */
export const gruScanKey = (direction: GruScanDirection): string =>
  `gru_scan:v1:f32:${
    canonicalizeDirection(direction)
  }:wg${GRU_SCAN_WORKGROUP_SIZE}:h${GRU_SCAN_MAX_HIDDEN}`;

export const gruScanWgsl = (direction: GruScanDirection): string => {
  const canonical = canonicalizeDirection(direction);
  // 逆方向は走査順だけを反転する（出力の時間添字 `t` はどちらも順方向の並び）。
  const timeIndex = canonical === "reverse" ? "dims.time - 1u - step" : "step";
  return `// karume gru_scan (${canonical}, f32, 1 workgroup = 1 バッチ要素 / 1 lane = 1 隠れユニット)
struct Dims {
  time: u32,
  batch: u32,
  hidden: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> gi: array<f32>;
@group(0) @binding(2) var<storage, read> h0: array<f32>;
@group(0) @binding(3) var<storage, read> w_hh: array<f32>;
@group(0) @binding(4) var<storage, read> b_hh: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

var<workgroup> h_shared: array<f32, ${GRU_SCAN_MAX_HIDDEN}>;
// 丸め障壁の中継（f32 のビット列をそのまま置く）。lane は自分の枠しか触らない。
var<workgroup> stage: array<u32, ${GRU_SCAN_MAX_HIDDEN}>;

${SIGMOID_STABLE_WGSL}

@compute @workgroup_size(${GRU_SCAN_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let hidden = dims.hidden;
  let gates = hidden * 3u;
  let lane_used = lid < hidden;
  // W_hh は [3H, H] の行優先。ゲートは r / z / n の順に H 行ずつ並ぶ。
  let row_r = lid * hidden;
  let row_z = (hidden + lid) * hidden;
  let row_n = (hidden * 2u + lid) * hidden;
  var item = wid.x;
  while (item < dims.batch) {
    if (lane_used) {
      h_shared[lid] = h0[item * hidden + lid];
    }
    workgroupBarrier();
    var step = 0u;
    while (step < dims.time) {
      let t = ${timeIndex};
      let gi_base = (t * dims.batch + item) * gates;
      var h_prev = 0.0;
      var gate_z = 0.0;
      if (lane_used) {
        h_prev = h_shared[lid];
        var acc_r = 0.0;
        var acc_z = 0.0;
        var acc_n = 0.0;
        for (var k = 0u; k < hidden; k = k + 1u) {
          let hk = h_shared[k];
          acc_r = acc_r + w_hh[row_r + k] * hk;
          acc_z = acc_z + w_hh[row_z + k] * hk;
          acc_n = acc_n + w_hh[row_n + k] * hk;
        }
        // bias は last（GEMM の epilogue と同じ）。ゲートの足し順は隠れ側が第 1 引数。
        let gate_r = sigmoid_stable((acc_r + b_hh[lid]) + gi[gi_base + lid]);
        gate_z = sigmoid_stable((acc_z + b_hh[hidden + lid]) + gi[gi_base + hidden + lid]);
        let gh_n = acc_n + b_hh[hidden * 2u + lid];
        // 丸め障壁 ①: 分解経路の mul は別 dispatch なので、fma 縮約させない
        stage[lid] = bitcast<u32>(gh_n * gate_r);
      }
      workgroupBarrier();
      var cand = 0.0;
      if (lane_used) {
        // n = tanh(i_n + h_n·r) — 入力側が第 1 引数
        cand = tanh(gi[gi_base + hidden * 2u + lid] + bitcast<f32>(stage[lid]));
        // 丸め障壁 ②: h' = (h − n)·z + n の mul と add の間
        stage[lid] = bitcast<u32>((h_prev - cand) * gate_z);
      }
      workgroupBarrier();
      if (lane_used) {
        let h_next = bitcast<f32>(stage[lid]) + cand;
        h_shared[lid] = h_next;
        out[(t * dims.batch + item) * hidden + lid] = h_next;
      }
      // 次ステップの縮約が h_shared を読む前に、この書き込みを揃える
      workgroupBarrier();
      step = step + 1u;
    }
    item = item + nwg.x;
  }
}
`;
};

/** gru_scan の幾何（params の唯一の入力型）。 */
export type GruScanDims = {
  /** 時間長 T（記号次元が束縛された後の実長）。 */
  readonly time: number;
  /** バッチ N。 */
  readonly batch: number;
  /** 隠れ幅 H。 */
  readonly hidden: number;
};

/**
 * 12 バイト + 整列で 16 バイトの uniform Dims（WGSL の uniform struct は 16 バイト整列 —
 * layer-norm.ts と同じ扱い）。
 *
 * MUST: `hidden` の上限をここで見る（契約検査と二重だが、カーネル直呼びの経路も通る門）。
 * 超過を黙って通すと `h_shared` の範囲外書き込みになり、例外なしに別ユニットの状態が壊れる。
 */
export const gruScanParams = (dims: GruScanDims): Uint32Array<ArrayBuffer> => {
  const { time, batch, hidden } = dims;
  assertU32Params("gru_scan params", { time, batch });
  if (!Number.isSafeInteger(hidden) || hidden < 1) {
    throw new CodegenError(`gru_scan params: hidden は正整数（${hidden}）`);
  }
  if (hidden > GRU_SCAN_MAX_HIDDEN) {
    throw new CodegenError(
      `gru_scan params: hidden ${hidden} が上限 ${GRU_SCAN_MAX_HIDDEN} を超える（1 lane = 1 隠れユニットの割り当て — ADR 0056 決定 5）`,
    );
  }
  return new Uint32Array([time, batch, hidden, 0]);
};
