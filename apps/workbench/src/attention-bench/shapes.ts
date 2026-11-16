/**
 * The shapes the attention bench entry times: the chosen ones of Wan2.2 TI2V-5B's attention (24 heads,
 * D 128), each on the row block production would dispatch first unless rows per dispatch is given.
 * Pure.
 */

import { type AttentionBenchShape, WAN22_ATTENTION_SHAPES } from "@karume/runtime/tune";

/** The ids of the bench's Wan2.2 shapes, in its order (the options of `shapes`). */
export const SHAPE_IDS: readonly string[] = WAN22_ATTENTION_SHAPES.map(({ id }) => id);

/**
 * The chosen shapes in the bench's order, with `rowsPerDispatch` applied: 0 keeps production's
 * split; a number is capped at each shape's query count.
 */
export const benchShapes = (
  ids: readonly string[],
  rowsPerDispatch: number,
): AttentionBenchShape[] =>
  WAN22_ATTENTION_SHAPES.filter(({ id }) => ids.includes(id)).map((shape) =>
    rowsPerDispatch === 0
      ? shape
      : { ...shape, rowsPerDispatch: Math.min(rowsPerDispatch, shape.queries) },
  );
