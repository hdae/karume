/**
 * dispatch 数の決め方。1 次元あたりの workgroup 数には上限（仕様既定 65535）があり、
 * 実モデルで実際に超える（16 head × 4096 token = 65536 行）。カーネル族ごとに超過時の
 * 振る舞いが違うため、判断をこの 2 関数に閉じ込める。
 */

import { DispatchLimitError } from "./errors.ts";

/**
 * grid-stride カーネルの dispatch 数。上限超過は縮退させる — カーネルが
 * `num_workgroups` から stride を導いて残りを回すため、少ない workgroup でも全域を覆う。
 *
 * MUST: `items + stride ≤ 2^32`（stride = 返す workgroup 数 × `workgroupSize`）。カーネルは
 * `i = i + stride` を u32 で加算して `i < n` を見るので、超えると末尾の加算が折り返して
 * ループが終わらない（GPU のハングで、例外にならない）。WGSL 側を引き算の終了判定に変えると
 * 全 grid-stride カーネルの WGSL が変わるので、門はここ 1 箇所に置く。
 */
export const gridStrideWorkgroups = (
  items: number,
  workgroupSize: number,
  limit: number,
): number => {
  const groups = Math.min(limit, Math.ceil(items / workgroupSize));
  const stride = groups * workgroupSize;
  if (items + stride > 2 ** 32) {
    throw new DispatchLimitError(
      `grid-stride: 要素数 ${items} + stride ${stride} が 2^32 を超える（u32 のループ添字が末尾の加算で折り返し、ループが終わらない）`,
    );
  }
  return groups;
};

/**
 * 1 workgroup = 1 タイルで全域を覆うカーネル（matmul）の dispatch 数。
 * MUST: 上限超過は fail loudly — 縮退させるとタイルが欠落し、例外なしに出力の一部が
 * 未書き込み（配り直しなら前の値）のまま残る。
 */
export const tiledWorkgroups = (
  items: number,
  tile: number,
  limit: number,
  where: string,
): number => {
  const groups = Math.ceil(items / tile);
  if (groups > limit) {
    throw new DispatchLimitError(
      `${where}: 必要 workgroup 数 ${groups} が device 上限 ${limit} を超える（${items} / タイル ${tile}）`,
    );
  }
  return groups;
};
