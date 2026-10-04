/**
 * Wan のグラフ宣言とグラフ出力を読む小道具（値の宣言・形の照合・出力の dtype・非有限値の走査）。
 *
 * 置き場がここなのは、`./pipeline.ts`（Wan2.1 の class）と共有の段（`./text-stage.ts` /
 * `./dit-loop.ts` / `./tile-decode.ts`）が同じ小道具を読むため（ADR 0121 決定 10）。`owner` は
 * 文言の接頭辞（Wan2.1 は `"WanPipeline"`）。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 */

import type { Tensor } from "@karume/runtime";

import type { GraphOwner } from "../hub/components.ts";

/** グラフの値の宣言（宣言の無い名前は fail loudly）。 */
export const valueOf = (
  owner: string,
  model: GraphOwner,
  name: string,
): GraphOwner["graph"]["values"][string] => {
  if (!Object.hasOwn(model.graph.values, name)) {
    throw new Error(`${owner}: グラフの値 '${name}' の宣言が無い`);
  }
  return model.graph.values[name];
};

/** グラフの値の形（宣言の無い名前は fail loudly）。 */
export const valueShape = (
  owner: string,
  model: GraphOwner,
  name: string,
): readonly (number | string)[] => valueOf(owner, model, name).shape;

/**
 * 宣言の形を rank と全軸で照合する（数は静的次元・文字列は記号次元の名前）。`expected` はホストが
 * 組む形そのもの。
 */
export const assertDims = (
  owner: string,
  dims: readonly (number | string)[],
  expected: readonly (number | string)[],
  where: string,
): void => {
  if (dims.length !== expected.length || dims.some((dim, axis) => dim !== expected[axis])) {
    const declared = dims.join(", ");
    const host = expected.join(", ");
    throw new Error(`${owner}: ${where} の形 [${declared}] がホストの組む [${host}] と違う`);
  }
};

export const asF32 = (tensor: Tensor, where: string): Float32Array => {
  if (tensor.dtype !== "f32") throw new Error(`${where}: f32 でない（${tensor.dtype}）`);
  return tensor.data;
};

/** 最初の非有限値の添字（無ければ -1）。 */
export const firstNonFinite = (values: Float32Array): number =>
  values.findIndex((value) => !Number.isFinite(value));
