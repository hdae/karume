/**
 * linear（`x[…,in] × W[out,in] + b[out]`、f32）。先行次元は平坦化して `[m,k] × [k,n]` の
 * 2 次元 GEMM に落とす（`m` = 先行次元の積、`k` = 入力特徴数、`n` = 出力特徴数）。
 *
 * 実体は GEMM 3 op 共通の 64×64 レジスタブロッキング骨格（src/kernels/gemm.ts）で、matmul /
 * bmm との違いは 3 点だけ:
 *
 * - **重みは転置レイアウト**（torch の `Linear.weight` は `[out, in]`）。連続方向が k なので、
 *   1 スレッドが k 連続 4 要素を読み、**共有メモリ側で転置して置く**（転置コピーは出さない）。
 * - 末尾で bias を 1 度だけ足す（accumulator に初期値として入れない — 縮約順序を matmul と
 *   同じ「k 昇順の逐次」に保つため。bias を先に積むと丸めの並びが変わる）。
 * - 重み格納の変種を持つ（`w=f32` / `w=f16` / `w=i8` / `w=i4` / `w=i2` — ADR 0018 / 0019 / 0069 /
 *   0097）。差は共有メモリへの読み出し式（と i8 / i4 / i2 の scale 束縛）だけで、i8 の scale は**共有タイルへの読み込み時**に
 *   要素ごとに掛かる（MAC ごとでも縮約の外でもない — ADR 0019）。
 */

import { assertU32Params } from "../codegen/params.ts";
import { type GemmCompute, gemmComputeKeyPart, gemmKeyPart, gemmParams, gemmWgsl } from "./gemm.ts";
import type { GemmGeometry } from "./gemm-geometry.ts";
import {
  i4GroupKeyPart,
  i4GroupShift,
  weightKeyPart,
  type WeightStorage,
} from "./weight-storage.ts";

export { LINEAR_SCALE_BINDING } from "./gemm.ts";

/**
 * `rows` は平坦化後の M（先行次元の積）。タイル幾何のバケット（src/kernels/gemm-geometry.ts の
 * `gemmGeometryForRows`）を決める形状由来の値で、MUST: キー・WGSL・dispatch に**同じ M** を通す。
 *
 * `groupSize` は i4 の group 長（i4 のとき必須）。shift を WGSL に焼くので**キーにも入れる**
 * （`:wi4g32` — 同一キー → バイト同一 WGSL の codegen 決定性）。
 *
 * `geometry` は明示の幾何（src/kernels/gemm.ts の `GemmSpec` — Session ではプロファイルの値・
 * 計測では掃引の候補）。渡すと `rows` のバケットより優先し、MUST: キーと WGSL へ**同じ値**を
 * 通す（`rows` と同じ規律）。
 */
export const linearKey = (
  weight: WeightStorage,
  v4: boolean,
  compute: GemmCompute = "f32",
  rows?: number,
  groupSize?: number,
  geometry?: GemmGeometry,
): string => {
  i4GroupShift("linear", weight, groupSize);
  // i8 の scale 読出しを保護した世代。未変更の格納形式のキーは維持する。
  return `linear:v${weight === "i8" ? 3 : 2}:f32:${gemmKeyPart(v4, rows, geometry)}${
    weightKeyPart(weight)
  }${i4GroupKeyPart(groupSize)}${gemmComputeKeyPart(compute)}`;
};

export const linearWgsl = (
  weight: WeightStorage,
  v4: boolean,
  compute: GemmCompute = "f32",
  rows?: number,
  groupSize?: number,
  geometry?: GemmGeometry,
): string =>
  gemmWgsl({
    op: "linear",
    v4,
    weight,
    compute,
    rows,
    weightGroupShift: i4GroupShift("linear", weight, groupSize),
    geometry,
  });

/**
 * uniform の Dims（m / n / k の u32 域は {@link gemmParams} が見る）。
 *
 * MUST: 重みの論理要素数 `n·k ≤ 2^32 − 1`。GEMM 経路は行頭を論理要素番号 `wcol·dims.k` として
 * u32 で先に作り、ローダが語位置へ変える（i2 は `>> 4`・i4 は `>> 3`）。積が先に折り返すと
 * 上位ビットは戻らず、別の出力チャネルの重みを例外なしに読む。i2 では格納が約 1 GiB で届くので
 * binding 上限や常駐計画の圧縮後バイト数の検査では止まらない（格納形を問わず添字は同じ式なので、
 * 検査も格納形で分けない）。
 */
export const linearParams = (m: number, n: number, k: number): Uint32Array<ArrayBuffer> => {
  const params = gemmParams("linear", m, n, k);
  assertU32Params("linear params", { "n * k": n * k });
  return params;
};
