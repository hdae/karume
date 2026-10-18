/**
 * Reads the container asset `rope_base` (per-axis rope base tables).
 *
 * The base tables are an **asset** of the `transformer` container (role `rope-base` — ADR 0109
 * decision 4), and the payload bytes are **in safetensors format** (neither a graph nor weights).
 * Parsing the payload is left to `parseSafetensors` on the `@karume/runtime` public surface (a
 * strict reader — coverage of the data section, alignment and dtype are checked there); this layer
 * only holds the purpose-specific checks "fetch the F32 rank-2 table per axis and compare row
 * counts" (DECIDED: put on the runtime public surface to remove a duplicate implementation —
 * ADR 0008 addendum 2026-08-05).
 *
 * ## The rope table is "rearranged from base tables", not "computed"
 *
 * MUST NOT compute cos / sin in TS. `torch`'s f32 trigonometric functions can differ by 1 ulp from
 * the correctly rounded value (measured: of 8,192 position × frequency pairs, cos 472 / sin 231),
 * and JS `Math.cos` rounds correctly in f64, so the two **always disagree**. Since the static graph
 * has torch's values baked in, bit identity with the S-form holds only by "baking the per-axis base
 * tables and just rearranging them".
 *
 * The base tables **do not depend on the resolution** (a table of axis × position, so any H' / W'
 * can be assembled). The row count is the length of upstream's `seq = arange(max(max_size))` = the
 * ceiling of the model's position table itself.
 */

import { parseSafetensors } from "@karume/runtime";
import type { SafetensorsFile } from "@karume/runtime";

/** Tensor keys of the base tables (order is the block order t → h → w). */
const ROPE_AXES = ["t", "h", "w"] as const;

/**
 * Accepts F32 only (the base tables are real cos / sin tables; no other storage exists upstream).
 */
const F32_DTYPE = "F32";

/** Per-axis cos / sin base tables (row = position, column = that axis's block width). */
export type RopeBase = {
  /** Row count shared by all axes (= the ceiling of the model's position table). */
  readonly rows: number;
  /** Block width per axis (twice `t + h + w` is head_dim). */
  readonly widths: readonly [number, number, number];
  /** cos base tables in the order `[t, h, w]`. */
  readonly cos: readonly Float32Array<ArrayBuffer>[];
  /** Same for sin. */
  readonly sin: readonly Float32Array<ArrayBuffer>[];
};

type RawTable = {
  readonly data: Float32Array<ArrayBuffer>;
  readonly shape: readonly [number, number];
};

/**
 * Fetches one F32 rank-2 table (alignment is guaranteed by `parseSafetensors` — the view is
 * zero-copy).
 */
const tableOf = (file: SafetensorsFile, name: string): RawTable => {
  const view = file.tensors.get(name);
  if (view === undefined) throw new Error(`rope 素表に '${name}' が無い`);
  if (view.dtype !== F32_DTYPE) {
    throw new Error(`rope 素表 '${name}': 格納 dtype が ${view.dtype}（F32 が必要）`);
  }
  if (view.shape.length !== 2) {
    throw new Error(`rope 素表 '${name}': rank が 2 でない`);
  }
  const shape: readonly [number, number] = [view.shape[0], view.shape[1]];
  return {
    data: new Float32Array(file.buffer, view.byteOffset, shape[0] * shape[1]),
    shape,
  };
};

/**
 * Reads the payload (safetensors format) of the asset `rope_base`.
 *
 * MUST: check that the row count is the same on every axis. Otherwise a mix-up of the "read h's
 * rows from w's table" kind stays within range and passes silently.
 */
export const parseRopeBase = (buffer: ArrayBuffer): RopeBase => {
  const file = parseSafetensors(buffer);
  // MUST: do not hold this in a module-scope const (cross-cutting invariant "zero side effects in
  // every module = no execution at import" — the condition for tree-shaking through the barrel; the
  // same reason as png.ts's buildCrcTable). The base tables are only 6 tensors, so building it per
  // call is negligible.
  const expectedKeys = new Set(ROPE_AXES.flatMap((axis) => [`cos_${axis}`, `sin_${axis}`]));
  for (const name of file.tensors.keys()) {
    if (!expectedKeys.has(name)) throw new Error(`rope 素表に想定外のテンソル '${name}' がある`);
  }
  const cos: Float32Array<ArrayBuffer>[] = [];
  const sin: Float32Array<ArrayBuffer>[] = [];
  const widths: number[] = [];
  let rows: number | undefined;
  for (const axis of ROPE_AXES) {
    const cosTable = tableOf(file, `cos_${axis}`);
    const sinTable = tableOf(file, `sin_${axis}`);
    if (cosTable.shape[0] !== sinTable.shape[0] || cosTable.shape[1] !== sinTable.shape[1]) {
      throw new Error(`rope 素表 ${axis} の cos / sin で shape が違う`);
    }
    if (rows !== undefined && cosTable.shape[0] !== rows) {
      throw new Error(`rope 素表 ${axis} の行数 ${cosTable.shape[0]} が他軸の ${rows} と違う`);
    }
    rows = cosTable.shape[0];
    widths.push(cosTable.shape[1]);
    cos.push(cosTable.data);
    sin.push(sinTable.data);
  }
  return { rows: rows ?? 0, widths: [widths[0], widths[1], widths[2]], cos, sin };
};

/** Width of one row of the rope table (= attention head_dim). */
export const ropeWidth = (base: RopeBase): number =>
  2 * (base.widths[0] + base.widths[1] + base.widths[2]);

/** Number of axes (`ropeTables` uses it to build the read positions). */
export const ROPE_AXIS_COUNT = ROPE_AXES.length;
