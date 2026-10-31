/**
 * The host side of the MOSS-Image DiT's RoPE — reading the `dit` container's asset `rope_base`
 * (per-axis base tables) and reordering it into the graph inputs `rope_cos` / `rope_sin`
 * (`[1,1,S,128]` for `dit` / `dit_skip22`, `[1,1,T,128]` for `dit_text` — ADR 0138 decision 2,
 * stage 2).
 *
 * ## The tables are reordered base tables, never computed
 *
 * MUST: do **not** compute sin / cos in TS (ADR 0034 decision 3). Upstream's `_RoPE` computes the
 * phases and trigonometric functions in f32 on the CPU; copying the formula could split the last bit
 * through implementation differences. The asset holds upstream's own outputs, cut per axis by the
 * recipe (`tools/export-recipes/moss/dit_patch.py` — `rope_base_tables`), so reordering them gives
 * upstream's table bit for bit (the recipe's eager gate and `moss_dit_host_test.ts` check it).
 *
 * ## The asset (safetensors, role `rope-base`)
 *
 * Keys `sin_<axis>` / `cos_<axis>` for the four axes `r, f, h, w` (upstream's coordinate order),
 * each F32 `[rows, bands]` (16 bands at head_dim 128):
 *
 * - `r`: 1 row (every token has r = 0);
 * - `f`: `maxText + 1` rows, row = position (text tokens at f = 0 … T − 1, image tokens at f = T);
 * - `h` / `w`: `2·maxSide − 1` rows over the half-integer grid `±(maxSide − 1)/2` — row
 *   `2h + maxSide − H` is position `h − (H − 1)/2` of a side of H cells (rows hold half-integer
 *   positions because H − 1 is odd for even H).
 *
 * ## Layout (half-split form)
 *
 * Upstream's table row is `[sin r f h w | cos r f h w]` (64 + 64) and its `_rotate` repeats each half
 * to 128: `cos = cat(cos, cos)`. The graph inputs are those repeated halves, so one row of
 * `rope_cos` is `[cos_r, cos_f, cos_h, cos_w, cos_r, cos_f, cos_h, cos_w]` (16 values each). Rows
 * follow the joint sequence `[image (h-major), text]`: image token (h, w) at
 * `(r 0, f T, h − (H−1)/2, w − (W−1)/2)` — **its position depends on the text length T** — and
 * text token i at `(r 0, f i, h 0, w 0)`.
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state).
 */

import { parseSafetensors, type SafetensorsFile } from "@karume/runtime";

/** The axes in upstream's coordinate order (`_RoPE.forward`'s `(r, f, h, w)`). */
const ROPE_AXES = ["r", "f", "h", "w"] as const;

type RopeAxis = (typeof ROPE_AXES)[number];

/** Per-axis sin / cos base tables (row-major `[rows, bands]`). */
export type MossRopeBase = {
  /** Frequencies per axis (16 for head_dim 128: 4 axes × 2 halves × 16 = 128). */
  readonly bands: number;
  /** The longest text the `f` table covers (rows − 1 — 4,096). */
  readonly maxText: number;
  /** The longest side in latent cells the `h` / `w` tables cover ((rows + 1) / 2 — 4,096). */
  readonly maxSide: number;
  readonly sin: Readonly<Record<RopeAxis, Float32Array<ArrayBuffer>>>;
  readonly cos: Readonly<Record<RopeAxis, Float32Array<ArrayBuffer>>>;
};

/** One RoPE table pair (`S·width` elements each — the flattened `[1,1,S,width]`). */
export type MossRopeTables = {
  readonly cos: Float32Array<ArrayBuffer>;
  readonly sin: Float32Array<ArrayBuffer>;
};

const tableOf = (
  file: SafetensorsFile,
  name: string,
): { readonly data: Float32Array<ArrayBuffer>; readonly rows: number; readonly bands: number } => {
  const view = file.tensors.get(name);
  if (view === undefined) throw new Error(`The rope base table has no '${name}'`);
  if (view.dtype !== "F32") {
    throw new Error(`Rope base table '${name}': storage dtype is ${view.dtype} (F32 is required)`);
  }
  if (view.shape.length !== 2) throw new Error(`Rope base table '${name}': rank is not 2`);
  const [rows, bands] = view.shape;
  return { data: new Float32Array(file.buffer, view.byteOffset, rows * bands), rows, bands };
};

/**
 * Reads the payload (safetensors) of the asset `rope_base`.
 *
 * MUST: exactly the eight keys, one band count on every axis, one row for `r`, the same row count
 * for `h` and `w`, and an odd row count for them (the half-integer grid). Otherwise a mix-up such as
 * "reading h's rows from w's table" stays in range and passes silently.
 */
export const parseMossRopeBase = (buffer: ArrayBuffer): MossRopeBase => {
  const file = parseSafetensors(buffer);
  // MUST: not a module-scope const (every module is side-effect free).
  const expected = new Set(ROPE_AXES.flatMap((axis) => [`sin_${axis}`, `cos_${axis}`]));
  for (const name of file.tensors.keys()) {
    if (!expected.has(name)) throw new Error(`The rope base table has an unexpected '${name}'`);
  }
  const read = (kind: "sin" | "cos"): Record<RopeAxis, ReturnType<typeof tableOf>> => ({
    r: tableOf(file, `${kind}_r`),
    f: tableOf(file, `${kind}_f`),
    h: tableOf(file, `${kind}_h`),
    w: tableOf(file, `${kind}_w`),
  });
  const sin = read("sin");
  const cos = read("cos");
  const bands = sin.r.bands;
  for (const axis of ROPE_AXES) {
    for (const table of [sin[axis], cos[axis]]) {
      if (table.bands !== bands) {
        throw new Error(`Rope base table ${axis}: ${table.bands} bands (the r table has ${bands})`);
      }
    }
    if (sin[axis].rows !== cos[axis].rows) {
      throw new Error(`Rope base table ${axis}: sin and cos row counts differ`);
    }
  }
  if (sin.r.rows !== 1) throw new Error(`Rope base table r: ${sin.r.rows} rows (1 expected)`);
  const sideRows = sin.h.rows;
  if (sin.w.rows !== sideRows || sideRows % 2 !== 1) {
    throw new Error(
      `Rope base tables h / w: ${sideRows} / ${sin.w.rows} rows (one odd count — the half-integer grid)`,
    );
  }
  const pick = (tables: Record<RopeAxis, ReturnType<typeof tableOf>>) => ({
    r: tables.r.data,
    f: tables.f.data,
    h: tables.h.data,
    w: tables.w.data,
  });
  return {
    bands,
    maxText: sin.f.rows - 1,
    maxSide: (sideRows + 1) / 2,
    sin: pick(sin),
    cos: pick(cos),
  };
};

/** The width of one table row (= the attention head_dim: 4 axes × 2 halves × bands). */
export const mossRopeWidth = (base: MossRopeBase): number => 2 * ROPE_AXES.length * base.bands;

/** Writes one row: the four axis rows, then the same four again (the repeated half). */
const writeRow = (
  base: MossRopeBase,
  out: MossRopeTables,
  at: number,
  rows: Readonly<Record<RopeAxis, number>>,
): void => {
  const { bands } = base;
  const half = ROPE_AXES.length * bands;
  for (let axis = 0; axis < ROPE_AXES.length; axis += 1) {
    const name = ROPE_AXES[axis];
    const from = rows[name] * bands;
    const to = at + axis * bands;
    for (let band = 0; band < bands; band += 1) {
      const cos = base.cos[name][from + band];
      const sin = base.sin[name][from + band];
      out.cos[to + band] = cos;
      out.cos[to + half + band] = cos;
      out.sin[to + band] = sin;
      out.sin[to + half + band] = sin;
    }
  }
};

const assertTextLength = (base: MossRopeBase, textLength: number): void => {
  if (!Number.isInteger(textLength) || textLength < 1 || textLength > base.maxText) {
    throw new Error(`RoPE: text length ${textLength} outside 1 … ${base.maxText}`);
  }
};

/** The latent grid of an image in cells (`H/16 × W/16`). */
export type MossCells = { readonly rows: number; readonly cols: number };

/**
 * `rope_cos` / `rope_sin` of `dit` / `dit_skip22` for an image of `cells` and a text of
 * `textLength` tokens (S = rows·cols + T rows — module doc "Layout").
 *
 * MUST: refuse a side over `maxSide` or a text over `maxText` (no base-table row exists — the
 * acceptance check rejects such requests first, naming the limit).
 */
export const mossRopeTables = (
  base: MossRopeBase,
  cells: MossCells,
  textLength: number,
): MossRopeTables => {
  const { rows: height, cols: width } = cells;
  for (const side of [height, width]) {
    if (!Number.isInteger(side) || side < 1 || side > base.maxSide) {
      throw new Error(`RoPE: ${height}×${width} cells — a side outside 1 … ${base.maxSide}`);
    }
  }
  assertTextLength(base, textLength);
  const width128 = mossRopeWidth(base);
  const count = height * width + textLength;
  const out = { cos: new Float32Array(count * width128), sin: new Float32Array(count * width128) };
  const centre = base.maxSide - 1;
  for (let h = 0; h < height; h += 1) {
    for (let w = 0; w < width; w += 1) {
      writeRow(base, out, (h * width + w) * width128, {
        r: 0,
        f: textLength,
        h: 2 * h + base.maxSide - height,
        w: 2 * w + base.maxSide - width,
      });
    }
  }
  const image = height * width;
  for (let index = 0; index < textLength; index += 1) {
    writeRow(base, out, (image + index) * width128, { r: 0, f: index, h: centre, w: centre });
  }
  return out;
};

/**
 * `rope_cos` / `rope_sin` of `dit_text` (the text rows of the joint table — positions do not depend
 * on the image).
 */
export const mossTextRopeTables = (base: MossRopeBase, textLength: number): MossRopeTables => {
  assertTextLength(base, textLength);
  const width128 = mossRopeWidth(base);
  const out = {
    cos: new Float32Array(textLength * width128),
    sin: new Float32Array(textLength * width128),
  };
  const centre = base.maxSide - 1;
  for (let index = 0; index < textLength; index += 1) {
    writeRow(base, out, index * width128, { r: 0, f: index, h: centre, w: centre });
  }
  return out;
};
