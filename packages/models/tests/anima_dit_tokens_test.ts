// The host glue of the sequence-form (S tokens) DiT (patchify / unpatchify / rope tables) and the
// reader of the rope base tables.
//
// **Axis mix-ups belong to the class that "becomes an involution at the real shapes and matches by
// accident"**, so the cases here give the patch height / width / channel count all different
// values, a shape that passes only with a permutation of cycle length 3 or more. The check with
// real weights on a real GPU is the E2E test (P3, iteration 2).

import { test } from "vite-plus/test";
import { assertEquals, assertThrows } from "@std/assert";
import {
  ditPatchGeometry,
  patchifyLatents,
  ropeTables,
  tokenCount,
  unpatchifyTokens,
} from "../src/anima/dit-tokens.ts";
import { parseRopeBase, type RopeBase, ropeWidth } from "../src/anima/rope-base.ts";
import { buildSafetensors, f32Bytes } from "./helpers/safetensors.ts";

/** Geometry with a non-square patch (2×3) and 3 channels. None of C, ph and pw coincide. */
const TOKEN_GEOMETRY = { channels: 3, patchHeight: 2, patchWidth: 3 } as const;

test("ditPatchGeometry: derives the patch and the channel count from the graph's input and output widths", () => {
  // Anima's real shape (tokens 68 = 17·2·2 / output 64 = 2·2·16).
  assertEquals(ditPatchGeometry(68, 64), { channels: 16, patchHeight: 2, patchWidth: 2 });
  // Difference = patch volume, quotient = channel count (derived so that callers hold no literal).
  assertEquals(ditPatchGeometry(45, 36), { channels: 4, patchHeight: 3, patchWidth: 3 });
});

test("ditPatchGeometry: rejects widths it cannot derive from (does not silently assume a square patch)", () => {
  assertThrows(() => ditPatchGeometry(64, 68), Error, "Cannot derive");
  assertThrows(() => ditPatchGeometry(70, 64), Error, "Cannot derive");
  // The difference is not a square number = the premise of pt=1 and a square patch is broken.
  assertThrows(() => ditPatchGeometry(72, 64), Error, "perfect square");
});

test("tokenCount: the product of the grid of the latent divided by the patch (S)", () => {
  assertEquals(tokenCount([1, 3, 4, 9], TOKEN_GEOMETRY), 2 * 3);
  // 1024px (latent 128, patch 2) gives S = 4,096.
  assertEquals(
    tokenCount([1, 16, 128, 128], { channels: 16, patchHeight: 2, patchWidth: 2 }),
    4096,
  );
});

test("patchifyLatents: last dimension is (c, ih, iw), token index is h·W'+w, padding is all zero", () => {
  const { channels, patchHeight, patchWidth } = TOKEN_GEOMETRY;
  const [height, width] = [4, 9];
  const rows = height / patchHeight;
  const cols = width / patchWidth;
  const latents = Float32Array.from({ length: channels * height * width }, (_, i) => i + 1);

  const tokens = patchifyLatents(latents, [1, channels, height, width], TOKEN_GEOMETRY);

  const patchVolume = patchHeight * patchWidth;
  const tokenWidth = (channels + 1) * patchVolume;
  assertEquals(tokens.length, rows * cols * tokenWidth);
  for (let channel = 0; channel < channels; channel += 1) {
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        for (let innerH = 0; innerH < patchHeight; innerH += 1) {
          for (let innerW = 0; innerW < patchWidth; innerW += 1) {
            const at =
              channel * height * width +
              (row * patchHeight + innerH) * width +
              col * patchWidth +
              innerW;
            assertEquals(
              tokens[
                (row * cols + col) * tokenWidth +
                  channel * patchVolume +
                  innerH * patchWidth +
                  innerW
              ],
              latents[at],
              `c=${channel} h=${row}.${innerH} w=${col}.${innerW}`,
            );
          }
        }
      }
    }
  }
  // The padding channel (the last one) is always zero — because the mask the pipeline passes is
  // zero.
  for (let token = 0; token < rows * cols; token += 1) {
    for (let inner = 0; inner < patchVolume; inner += 1) {
      assertEquals(tokens[token * tokenWidth + channels * patchVolume + inner], 0);
    }
  }
});

test("unpatchifyTokens: last dimension is (ih, iw, c) — not the reverse order of patchify", () => {
  const { channels, patchHeight, patchWidth } = TOKEN_GEOMETRY;
  const [height, width] = [4, 9];
  const rows = height / patchHeight;
  const cols = width / patchWidth;
  const tokenWidth = patchHeight * patchWidth * channels;
  const tokens = Float32Array.from({ length: rows * cols * tokenWidth }, (_, i) => i + 1);

  const latents = unpatchifyTokens(tokens, [1, channels, height, width], TOKEN_GEOMETRY);

  assertEquals(latents.length, channels * height * width);
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      for (let innerH = 0; innerH < patchHeight; innerH += 1) {
        for (let innerW = 0; innerW < patchWidth; innerW += 1) {
          for (let channel = 0; channel < channels; channel += 1) {
            const at =
              channel * height * width +
              (row * patchHeight + innerH) * width +
              col * patchWidth +
              innerW;
            assertEquals(
              latents[at],
              tokens[
                (row * cols + col) * tokenWidth +
                  (innerH * patchWidth + innerW) * channels +
                  channel
              ],
              `c=${channel} h=${row}.${innerH} w=${col}.${innerW}`,
            );
          }
        }
      }
    }
  }
});

test("patchify / unpatchify: rejects mismatched shapes (no silently wrong values)", () => {
  const shape = [1, TOKEN_GEOMETRY.channels, 4, 9];
  assertThrows(
    () => patchifyLatents(new Float32Array(4), shape, TOKEN_GEOMETRY),
    Error,
    "element count",
  );
  assertThrows(
    () => patchifyLatents(new Float32Array(36), [1, 3, 5, 9], TOKEN_GEOMETRY),
    Error,
    "is not divisible by",
  );
  assertThrows(
    () => patchifyLatents(new Float32Array(36), [2, 3, 4, 9], TOKEN_GEOMETRY),
    Error,
    "assumes a [1,C,H,W]",
  );
  assertThrows(
    () => unpatchifyTokens(new Float32Array(7), shape, TOKEN_GEOMETRY),
    Error,
    "element count",
  );
});

/**
 * Base tables with a different width and different values per axis (a shape where a mix-up of block
 * order or read position always shows).
 */
const syntheticRopeBase = (rows: number): RopeBase => {
  const widths: readonly [number, number, number] = [1, 2, 3];
  const make = (axis: number, kind: number): Float32Array<ArrayBuffer> =>
    Float32Array.from(
      { length: rows * widths[axis] },
      (_, i) =>
        (axis + 1) * 1000 + kind * 100 + Math.floor(i / widths[axis]) * 10 + (i % widths[axis]),
    );
  return {
    rows,
    widths,
    cos: [make(0, 0), make(1, 0), make(2, 0)],
    sin: [make(0, 1), make(1, 1), make(2, 1)],
  };
};

test("ropeTables: one row is [t,h,w,t,h,w], positions are t=0 / h=row / w=column", () => {
  const base = syntheticRopeBase(8);
  const geometry = { channels: 2, patchHeight: 2, patchWidth: 2 } as const;
  const [height, width] = [6, 8];
  const rows = height / geometry.patchHeight;
  const cols = width / geometry.patchWidth;

  const { cos, sin } = ropeTables(base, [1, geometry.channels, height, width], geometry);

  const rowWidth = ropeWidth(base);
  assertEquals(rowWidth, 12, "the row width is 2·(1+2+3)");
  assertEquals(cos.length, rows * cols * rowWidth);
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const offsets = [0, row * base.widths[1], col * base.widths[2]];
      const expectedCos: number[] = [];
      const expectedSin: number[] = [];
      for (let half = 0; half < 2; half += 1) {
        for (let axis = 0; axis < 3; axis += 1) {
          const span = base.widths[axis];
          for (let index = 0; index < span; index += 1) {
            expectedCos.push(base.cos[axis][offsets[axis] + index]);
            expectedSin.push(base.sin[axis][offsets[axis] + index]);
          }
        }
      }
      const at = (row * cols + col) * rowWidth;
      assertEquals([...cos.slice(at, at + rowWidth)], expectedCos, `cos (${row}, ${col})`);
      assertEquals([...sin.slice(at, at + rowWidth)], expectedSin, `sin (${row}, ${col})`);
    }
  }
});

test("ropeTables: rejects a resolution beyond the base tables' row count (the ceiling of the model's position table)", () => {
  const base = syntheticRopeBase(3);
  const geometry = { channels: 2, patchHeight: 2, patchWidth: 2 } as const;

  assertThrows(() => ropeTables(base, [1, 2, 8, 4], geometry), Error, "ceiling");
});

const ropeTable = (rows: number, width: number, seed: number) => ({
  dtype: "F32",
  shape: [rows, width],
  data: f32Bytes(Array.from({ length: rows * width }, (_, i) => seed + i)),
});

test("parseRopeBase: reads the six base tables in axis order (t → h → w)", () => {
  const buffer = buildSafetensors([
    { name: "cos_t", ...ropeTable(4, 1, 100) },
    { name: "sin_t", ...ropeTable(4, 1, 200) },
    { name: "cos_h", ...ropeTable(4, 2, 300) },
    { name: "sin_h", ...ropeTable(4, 2, 400) },
    { name: "cos_w", ...ropeTable(4, 3, 500) },
    { name: "sin_w", ...ropeTable(4, 3, 600) },
  ]);
  const base = parseRopeBase(buffer);
  assertEquals(base.rows, 4);
  assertEquals(base.widths, [1, 2, 3]);
  assertEquals(ropeWidth(base), 12);
  // The axes are ordered t → h → w (a mix-up that reads cos_h at w's position breaks here).
  assertEquals([...base.cos[0]], [100, 101, 102, 103]);
  assertEquals([...base.cos[1]].slice(0, 2), [300, 301]);
  assertEquals([...base.sin[2]].slice(0, 3), [600, 601, 602]);
});

test("parseRopeBase: rejects base tables whose per-axis row counts differ (does not silently read another axis's rows)", () => {
  const buffer = buildSafetensors([
    { name: "cos_t", ...ropeTable(4, 1, 0) },
    { name: "sin_t", ...ropeTable(4, 1, 0) },
    { name: "cos_h", ...ropeTable(4, 2, 0) },
    { name: "sin_h", ...ropeTable(4, 2, 0) },
    // Only w has a different row count = a shape where the mix-up "read h's rows from w's table"
    // stays in range.
    { name: "cos_w", ...ropeTable(3, 2, 0) },
    { name: "sin_w", ...ropeTable(3, 2, 0) },
  ]);

  assertThrows(() => parseRopeBase(buffer), Error, "row count");
});

test("parseRopeBase: rejects base tables mixed with an unexpected tensor", () => {
  // One extra tensor (a shape one can hit when rebuilding the `karume-anima-extra` mirror).
  // Skipping it would open a path where "a table other than the base tables lives in the same asset
  // unnoticed".
  const buffer = buildSafetensors([
    { name: "cos_t", ...ropeTable(4, 1, 0) },
    { name: "sin_t", ...ropeTable(4, 1, 0) },
    { name: "cos_h", ...ropeTable(4, 2, 0) },
    { name: "sin_h", ...ropeTable(4, 2, 0) },
    { name: "cos_w", ...ropeTable(4, 3, 0) },
    { name: "sin_w", ...ropeTable(4, 3, 0) },
    { name: "cos_x", ...ropeTable(4, 1, 0) },
  ]);

  assertThrows(() => parseRopeBase(buffer), Error, "unexpected tensor 'cos_x'");
});

test("parseRopeBase: rejects cos / sin of the same axis with different column widths (row-count agreement does not catch it)", () => {
  // The row counts stay at 4 on every axis while only the h-axis pair gets different column widths.
  // It passes the row-count check, so without a per-pair shape comparison a table with "cos 2
  // columns / sin 3 columns" would pass silently.
  const buffer = buildSafetensors([
    { name: "cos_t", ...ropeTable(4, 1, 0) },
    { name: "sin_t", ...ropeTable(4, 1, 0) },
    { name: "cos_h", ...ropeTable(4, 2, 0) },
    { name: "sin_h", ...ropeTable(4, 3, 0) },
    { name: "cos_w", ...ropeTable(4, 3, 0) },
    { name: "sin_w", ...ropeTable(4, 3, 0) },
  ]);

  assertThrows(() => parseRopeBase(buffer), Error, "the shapes of cos / sin do not match");
});

test("parseRopeBase: rejects a missing table, a non-F32 table or a wrong rank", () => {
  assertThrows(
    () =>
      parseRopeBase(
        buildSafetensors([
          { name: "cos_t", ...ropeTable(2, 1, 0) },
          { name: "sin_t", ...ropeTable(2, 1, 0) },
        ]),
      ),
    Error,
    "has no 'cos_h'",
  );
  assertThrows(
    () =>
      parseRopeBase(
        buildSafetensors([{ name: "cos_t", dtype: "F16", shape: [2, 1], data: new Uint8Array(4) }]),
      ),
    Error,
    "F32 is required",
  );
  assertThrows(
    () =>
      parseRopeBase(
        buildSafetensors([{ name: "cos_t", dtype: "F32", shape: [2], data: f32Bytes([0, 0]) }]),
      ),
    Error,
    "rank is not 2",
  );
});
