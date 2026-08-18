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
 * - 重み格納の変種を持つ（`w=f32` / `w=f16` / `w=i8` — ADR 0018 / 0019）。差は共有メモリへの
 *   読み出し式（と i8 の scale 束縛）だけで、i8 の scale は**共有タイルへの読み込み時**に
 *   要素ごとに掛かる（MAC ごとでも縮約の外でもない — ADR 0019）。
 */

import { CodegenError } from "../codegen/errors.ts";
import { type GemmCompute, gemmComputeKeyPart, gemmKeyPart, gemmParams, gemmWgsl } from "./gemm.ts";
import { weightKeyPart, type WeightStorage } from "./weight-storage.ts";

export { LINEAR_SCALE_BINDING } from "./gemm.ts";

/**
 * i4 の group 長（キー断片と WGSL の shift の共通導出点）。
 *
 * MUST: i4 と 1 対 1（i4 なのに無い / i4 以外に付く、はどちらも結線バグで、黙って通すと
 * 「group 32 のパイプラインが group 64 の資産で走る」沈黙誤値になる）。2 冪 ≥ 16 は宣言層
 * （format/ir.ts）が保証済みだが、shift をここで導出する以上は言い直す。
 */
const i4GroupShift = (weight: WeightStorage, groupSize: number | undefined): number | undefined => {
  if ((weight === "i4") !== (groupSize !== undefined)) {
    throw new CodegenError(`linear: groupSize は重み i4 格納と対で渡す（weight=${weight}）`);
  }
  if (groupSize === undefined) return undefined;
  const shift = Math.log2(groupSize);
  if (!Number.isInteger(shift) || groupSize < 16) {
    throw new CodegenError(`linear: i4 の group_size ${groupSize} が 2 冪かつ 16 以上でない`);
  }
  return shift;
};

/**
 * `rows` は平坦化後の M（先行次元の積）。タイル幾何のバケット（src/kernels/gemm-geometry.ts の
 * `gemmGeometryForRows`）を決める形状由来の値で、MUST: キー・WGSL・dispatch に**同じ M** を通す。
 *
 * `groupSize` は i4 の group 長（i4 のとき必須）。shift を WGSL に焼くので**キーにも入れる**
 * （`:wi4g32` — 同一キー → バイト同一 WGSL の codegen 決定性）。
 */
export const linearKey = (
  weight: WeightStorage,
  v4: boolean,
  compute: GemmCompute = "f32",
  rows?: number,
  groupSize?: number,
): string => {
  i4GroupShift(weight, groupSize);
  const group = groupSize === undefined ? "" : `g${groupSize}`;
  return `linear:v2:f32:${gemmKeyPart(v4, rows)}${weightKeyPart(weight)}${group}${
    gemmComputeKeyPart(compute)
  }`;
};

export const linearWgsl = (
  weight: WeightStorage,
  v4: boolean,
  compute: GemmCompute = "f32",
  rows?: number,
  groupSize?: number,
): string =>
  gemmWgsl({
    op: "linear",
    v4,
    weight,
    compute,
    rows,
    weightGroupShift: i4GroupShift(weight, groupSize),
  });

export const linearParams = (m: number, n: number, k: number): Uint32Array<ArrayBuffer> =>
  gemmParams("linear", m, n, k);
