/**
 * Small helpers that read Wan's graph declarations and graph outputs (value declarations, shape
 * matching, output dtype, scan for non-finite values).
 *
 * They live here because the shared stages (`./text-stage.ts` / `./dit-loop.ts` / `./tile-decode.ts`
 * — used by the Wan2.1 / 2.2 classes through `./family.ts`) read the same helpers (ADR 0121
 * decision 10). `owner` is the message prefix (`"WanPipeline"` for Wan2.1, `"WanTi2vPipeline"` for
 * Wan2.2).
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state —
 * CLAUDE.md).
 */

import type { Tensor } from "@karume/runtime";

import type { GraphOwner } from "../hub/components.ts";

/** The declaration of a graph value (an undeclared name fails loudly). */
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

/** The shape of a graph value (an undeclared name fails loudly). */
export const valueShape = (
  owner: string,
  model: GraphOwner,
  name: string,
): readonly (number | string)[] => valueOf(owner, model, name).shape;

/**
 * Matches a declared shape on rank and every axis (a number is a static dimension, a string the
 * name of a symbolic dimension). `expected` is the shape exactly as the host builds it.
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

export const asF32 = (tensor: Tensor, where: string): Float32Array<ArrayBuffer> => {
  if (tensor.dtype !== "f32") throw new Error(`${where}: f32 でない（${tensor.dtype}）`);
  return tensor.data;
};

/** Index of the first non-finite value (-1 when there is none). */
export const firstNonFinite = (values: Float32Array<ArrayBuffer>): number =>
  values.findIndex((value) => !Number.isFinite(value));
