/**
 * The host side of the 3D RoPE tables of Wan2.1's DiT — reading the container asset `rope_base`
 * (per-axis base tables) and reordering it into the S-shaped graph's inputs `rope_cos` /
 * `rope_sin [1,S,1,head_dim]` (ADR 0118 decision 3).
 *
 * The base tables are an **asset** of the `transformer` container (role `rope-base` — ADR 0109
 * decision 4) with a safetensors payload. The keys are `cos_t` / `sin_t` / `cos_h` / `sin_h` /
 * `cos_w` / `sin_w` (the same spelling and meaning as Anima's `rope_base` = an axis × position ×
 * frequency table), each an F32 `[rows, number of frequencies of that axis]`. The values are upstream
 * `WanRotaryPosEmbed`'s tables as they are (the exporter cuts them out of upstream's output —
 * `wan/dit_patch.py`).
 *
 * ## The tables are "reordered from base tables", not "computed"
 *
 * MUST: do **not** compute cos / sin in TS. Upstream computes the angles and trigonometric functions in
 * float64 and then drops to f32, so copying the formula into TS can split the last bit through
 * implementation differences in `pow` / division / `cos`. Only reordering the base tables is
 * bit-identical to upstream's tables (the same judgement as Anima's `rope-base.ts`).
 *
 * ## Layout (interleaved form)
 *
 * One row (one token) is the concatenation `[t block, h block, w block]`, and each block repeats the
 * value of every frequency twice (upstream's `repeat_interleave(2)` — a rotation pair is two adjacent
 * elements `(x[2i], x[2i+1])`). The order differs from Anima's half-split form
 * (`[t, h, w, t, h, w]`).
 */

import { parseSafetensors } from "@karume/runtime";
import type { SafetensorsFile } from "@karume/runtime";
import type { WanTokenGrid } from "./dit-tokens.ts";

/** The axes of the base tables (in block order t → h → w). */
const ROPE_AXES = ["t", "h", "w"] as const;

/** Only F32 is accepted (the base tables are real-valued cos / sin tables; upstream has no other storage form). */
const F32_DTYPE = "F32";

/** Per-axis cos / sin base tables (row = position, column = a frequency of that axis). */
export type WanRopeBase = {
  /** The row count shared by all axes (= the ceiling of upstream's position table, `rope_max_seq_len`). */
  readonly rows: number;
  /** The number of frequencies per axis (`2 × (t + h + w)` is head_dim). */
  readonly widths: readonly [number, number, number];
  /** The cos base tables in `[t, h, w]` order (row-major `[rows, widths[axis]]`). */
  readonly cos: readonly [
    Float32Array<ArrayBuffer>,
    Float32Array<ArrayBuffer>,
    Float32Array<ArrayBuffer>,
  ];
  /** The same for sin. */
  readonly sin: readonly [
    Float32Array<ArrayBuffer>,
    Float32Array<ArrayBuffer>,
    Float32Array<ArrayBuffer>,
  ];
};

type RawTable = {
  readonly data: Float32Array<ArrayBuffer>;
  readonly shape: readonly [number, number];
};

/** Fetches one F32 rank-2 table (`parseSafetensors` already guarantees alignment — the view is zero-copy). */
const tableOf = (file: SafetensorsFile, name: string): RawTable => {
  const view = file.tensors.get(name);
  if (view === undefined) throw new Error(`rope 素表に '${name}' が無い`);
  if (view.dtype !== F32_DTYPE) {
    throw new Error(`rope 素表 '${name}': 格納 dtype が ${view.dtype}（F32 が必要）`);
  }
  if (view.shape.length !== 2) throw new Error(`rope 素表 '${name}': rank が 2 でない`);
  const shape: readonly [number, number] = [view.shape[0], view.shape[1]];
  return { data: new Float32Array(file.buffer, view.byteOffset, shape[0] * shape[1]), shape };
};

/**
 * Reads the payload (safetensors format) of the asset `rope_base`.
 *
 * MUST: check that there are exactly 6 keys and that the row count is the same on every axis.
 * Otherwise a mix-up such as "reading h's rows from w's table" stays in range and passes silently.
 */
export const parseWanRopeBase = (buffer: ArrayBuffer): WanRopeBase => {
  const file = parseSafetensors(buffer);
  // MUST: not a module-scope const (cross-cutting invariant "every module is side-effect free").
  const expectedKeys = new Set(ROPE_AXES.flatMap((axis) => [`cos_${axis}`, `sin_${axis}`]));
  for (const name of file.tensors.keys()) {
    if (!expectedKeys.has(name)) throw new Error(`rope 素表に想定外のテンソル '${name}' がある`);
  }
  const tables = ROPE_AXES.map((axis) => {
    const cos = tableOf(file, `cos_${axis}`);
    const sin = tableOf(file, `sin_${axis}`);
    if (cos.shape[0] !== sin.shape[0] || cos.shape[1] !== sin.shape[1]) {
      throw new Error(`rope 素表 ${axis} の cos / sin で shape が違う`);
    }
    return { axis, cos, sin };
  });
  const rows = tables[0].cos.shape[0];
  for (const { axis, cos } of tables) {
    if (cos.shape[0] !== rows) {
      throw new Error(`rope 素表 ${axis} の行数 ${cos.shape[0]} が他軸の ${rows} と違う`);
    }
  }
  const [t, h, w] = tables;
  return {
    rows,
    widths: [t.cos.shape[1], h.cos.shape[1], w.cos.shape[1]],
    cos: [t.cos.data, h.cos.data, w.cos.data],
    sin: [t.sin.data, h.sin.data, w.sin.data],
  };
};

/** The width of one RoPE table row (= the attention's head_dim). */
export const wanRopeWidth = (base: WanRopeBase): number =>
  2 * (base.widths[0] + base.widths[1] + base.widths[2]);

/**
 * Builds `rope_cos` / `rope_sin` (`S·head_dim`, the flattened `[1,S,1,head_dim]`) from the base
 * tables.
 *
 * The row of token `(f, h, w)` (index `(f·H' + h)·W' + w`) is the base-table row at position f for
 * the t block, position h for the h block and position w for the w block, each frequency repeated
 * twice (module doc). MUST: reject a grid where any of F' / H' / W' exceeds the base tables' row
 * count (upstream cannot build past the ceiling of its position table either).
 */
export const wanRopeTables = (
  base: WanRopeBase,
  grid: WanTokenGrid,
): { readonly cos: Float32Array<ArrayBuffer>; readonly sin: Float32Array<ArrayBuffer> } => {
  const positions = [grid.frames, grid.rows, grid.cols];
  if (positions.some((extent) => extent > base.rows)) {
    throw new Error(
      `rope 素表の行数 ${base.rows} では F'=${grid.frames} / H'=${grid.rows} / W'=${grid.cols} を` +
        "組めない（上流の位置表の天井を超えている）",
    );
  }
  const rowWidth = wanRopeWidth(base);
  const cos = new Float32Array(grid.count * rowWidth);
  const sin = new Float32Array(grid.count * rowWidth);
  for (let frame = 0; frame < grid.frames; frame += 1) {
    for (let row = 0; row < grid.rows; row += 1) {
      for (let col = 0; col < grid.cols; col += 1) {
        const token = (frame * grid.rows + row) * grid.cols + col;
        const axisPositions = [frame, row, col];
        let at = token * rowWidth;
        for (let axis = 0; axis < ROPE_AXES.length; axis += 1) {
          const span = base.widths[axis];
          const from = axisPositions[axis] * span;
          for (let index = 0; index < span; index += 1) {
            const cosValue = base.cos[axis][from + index];
            const sinValue = base.sin[axis][from + index];
            cos[at] = cosValue;
            cos[at + 1] = cosValue;
            sin[at] = sinValue;
            sin[at + 1] = sinValue;
            at += 2;
          }
        }
      }
    }
  }
  return { cos, sin };
};
