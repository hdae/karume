/**
 * The vision tower of EmbeddingGemma 2 on the host: an RGB8 image → the patch rows and positions the
 * tower graph `vision` reads, the tower's graph contract, and one tower run → the image's soft rows.
 *
 * ## The preprocessing (upstream's `Gemma4ImageProcessor`, as the snapshot configures it)
 *
 * 1. The size: the aspect-preserving rule of the Gemma 4 family ({@link imageTargetSize} — the
 *    largest multiples of 48 px within `budget` soft tokens, in f64 in upstream's order, which also
 *    gives upstream's rounding edges: 1190 × 17 becomes 6672 × 48, 139 soft tokens).
 * 2. The resize: torch's uint8 antialiased bicubic (`image/torch-resample.ts`), skipped on an axis
 *    whose size already matches (upstream skips it too). It also upscales.
 * 3. The patches: the resized image cut into 16 × 16 patches, each one row of 768 values in
 *    (row, column, channel) order, each value `f32(u8 · f32(1/255))` (upstream's rescale; no
 *    normalisation — the tower applies `2 · (x − 0.5)` itself).
 *
 * **Only the real patches, in block order** (the io contract of the recipe,
 * `tools/export-recipes/embeddinggemma2/export_vision.py`): with the patch grid `W × H` (both
 * multiples of 3), the rows are the 3 × 3 blocks in raster order, and inside each block its 9 patches
 * in raster order — row `i` is block `b = i div 9` (`bx = b mod (W / 3)`, `by = b div (W / 3)`),
 * `(dy, dx) = divmod(i mod 9, 3)`, at `position_x = 3 · bx + dx`, `position_y = 3 · by + dy`. The
 * tower pools each block's 9 adjacent rows into one soft token, so soft token `j` is block `j` —
 * upstream's order of the image's placeholder rows. Upstream pads to `budget · 9` rows and masks them;
 * the real rows' values do not depend on them, so none is bound here.
 *
 * Byte-identical to the series' io inputs (`embeddinggemma2_image_input_test.ts`).
 *
 * MUST: modules are side-effect free (CLAUDE.md).
 */

import type { RunOutputs, Session } from "@karume/runtime";

import type { PipelineVisionConfig } from "../gemma/config.ts";
import { imageTargetSize } from "../gemma/vision.ts";
import { assertRgb8, type Rgb8Image } from "../image/preprocess.ts";
import { resizeRgb8TorchBicubic } from "../image/torch-resample.ts";
import type { GraphOwner } from "../hub/components.ts";

/** The message prefix (the class name). */
const OWNER = "embeddinggemma.Pipeline";

/** The side of one patch in pixels (upstream `patch_size`). */
const PATCH_SIDE = 16;

/** The side of one soft token's block in patches (upstream `pooling_kernel_size`). */
const POOLING = 3;

/** The patches of one soft token (`pooling_kernel_size²`). */
export const PATCHES_PER_SOFT_TOKEN = POOLING * POOLING;

/** The values of one patch row (16 · 16 · 3). */
export const PATCH_ELEMENTS = PATCH_SIDE * PATCH_SIDE * 3;

/** Upstream's `1 / 255` as the f32 it multiplies by. */
const RESCALE = Math.fround(1 / 255);

/** The tower graph's inputs, in order (`GRAPH_INPUTS` of the recipe's `export_vision.py`). */
const VISION_PIXELS = "pixels";
const VISION_POSITION_X = "position_x";
const VISION_POSITION_Y = "position_y";

/** One image as the tower reads it. */
export type ImagePatchRows = {
  /** The image's soft tokens (n). */
  readonly softTokens: number;
  /** `[9n, 768]`: the patch rows in block order. */
  readonly pixels: Float32Array<ArrayBuffer>;
  /** `[9n]`: each row's column in the resized image's patch grid. */
  readonly positionX: Int32Array<ArrayBuffer>;
  /** `[9n]`: each row's row in that grid. */
  readonly positionY: Int32Array<ArrayBuffer>;
};

/**
 * The preprocessing of one image (module doc).
 *
 * NOTE: the `export` is for the pipeline and its host tests (not on `mod.ts` / the subpath surface —
 * ADR 0008).
 */
export const imagePatchRows = (image: Rgb8Image, budget: number): ImagePatchRows => {
  assertRgb8(image);
  const target = imageTargetSize(image.height, image.width, budget);
  const data =
    target.width === image.width && target.height === image.height
      ? image.data
      : resizeRgb8TorchBicubic(image, target.width, target.height).data;
  const blockColumns = target.width / (PATCH_SIDE * POOLING);
  const blockRows = target.height / (PATCH_SIDE * POOLING);
  const softTokens = blockColumns * blockRows;
  const patches = softTokens * PATCHES_PER_SOFT_TOKEN;
  const pixels = new Float32Array(patches * PATCH_ELEMENTS);
  const positionX = new Int32Array(patches);
  const positionY = new Int32Array(patches);
  const rowStride = target.width * 3;
  let patch = 0;
  let at = 0;
  for (let by = 0; by < blockRows; by += 1) {
    for (let bx = 0; bx < blockColumns; bx += 1) {
      for (let dy = 0; dy < POOLING; dy += 1) {
        for (let dx = 0; dx < POOLING; dx += 1) {
          const column = POOLING * bx + dx;
          const row = POOLING * by + dy;
          positionX[patch] = column;
          positionY[patch] = row;
          patch += 1;
          for (let y = 0; y < PATCH_SIDE; y += 1) {
            const start = (row * PATCH_SIDE + y) * rowStride + column * PATCH_SIDE * 3;
            for (let k = 0; k < PATCH_SIDE * 3; k += 1) {
              pixels[at] = (data[start + k] ?? 0) * RESCALE;
              at += 1;
            }
          }
        }
      }
    }
  }
  return { softTokens, pixels, positionX, positionY };
};

/** What the pipeline reads of the tower graph (checked at admission). */
export type EmbeddingGemmaVisionContract = {
  /** The name of the one output (`[1, n, hiddenSize]`). */
  readonly output: string;
  /** The width of a soft token (= `pipelineConfig.vision.hiddenSize`). */
  readonly width: number;
  /** The patch positions the graph's tables cover (`0 … positions − 1` on each axis). */
  readonly positions: number;
};

/** Whether a declared shape is exactly `expected` (rank and every axis). */
const sameShape = (
  shape: readonly (number | string)[],
  expected: readonly (number | string)[],
): boolean =>
  shape.length === expected.length && shape.every((dim, axis) => dim === expected[axis]);

/**
 * Matches the tower graph against what the host binds and reads, and against
 * `pipelineConfig.vision`.
 *
 * - One symbol n (the soft-token count); exactly the inputs `pixels` f32 `[1, 9n, 768]`,
 *   `position_x` and `position_y` i32 `[1, 9n]`, in that order; one f32 output `[1, n, hiddenSize]`;
 *   no state.
 * - The positions the graph's tables cover: the fewest rows of the `embedding` tables gathered by a
 *   position input (the folded RoPE tables, 840 rows). MUST: it holds the longest side the resize
 *   rule gives at the declared budget (`3 · budget` patches — a side of 48 px leaves the other at
 *   most `budget · 48` px), or a long thin image would gather past a table's end.
 *
 * NOTE: the `export` is for the host tests (not on `mod.ts` / the subpath surface — ADR 0008).
 */
export const embeddingGemmaVisionContract = (
  tower: GraphOwner,
  vision: PipelineVisionConfig,
): EmbeddingGemmaVisionContract => {
  const { graph } = tower;
  if (graph.symbols.length !== 1) {
    throw new Error(
      `${OWNER}: the vision graph's symbolic dimensions are [${graph.symbols.join(", ")}] (one soft-token count is required)`,
    );
  }
  const [symbol] = graph.symbols;
  const rows = `${PATCHES_PER_SOFT_TOKEN}${symbol}`;
  const expected = [
    [VISION_PIXELS, "f32", [1, rows, PATCH_ELEMENTS]],
    [VISION_POSITION_X, "i32", [1, rows]],
    [VISION_POSITION_Y, "i32", [1, rows]],
  ] as const;
  const declared = graph.inputs.map(
    (input) => `${input.name} ${input.dtype} [${input.shape.join(", ")}]`,
  );
  if (
    graph.inputs.length !== expected.length ||
    expected.some(
      ([name, dtype, shape], index) =>
        graph.inputs[index]?.name !== name ||
        graph.inputs[index]?.dtype !== dtype ||
        !sameShape(graph.inputs[index]?.shape ?? [], shape),
    )
  ) {
    throw new Error(
      `${OWNER}: the vision graph's inputs are ${declared.join(" / ")}, not` +
        ` ${expected.map(([name, dtype, shape]) => `${name} ${dtype} [${shape.join(", ")}]`).join(" / ")}`,
    );
  }
  if (Object.keys(graph.states).length > 0) {
    throw new Error(`${OWNER}: the vision graph holds state slots (it runs one image per run)`);
  }
  if (graph.outputs.length !== 1) {
    throw new Error(
      `${OWNER}: the vision graph has ${graph.outputs.length} outputs (one is required)`,
    );
  }
  const [output] = graph.outputs;
  const value = Object.hasOwn(graph.values, output) ? graph.values[output] : undefined;
  if (
    value === undefined ||
    value.dtype !== "f32" ||
    !sameShape(value.shape, [1, symbol, vision.hiddenSize])
  ) {
    throw new Error(
      `${OWNER}: the vision graph's output '${output}' is ${value?.dtype ?? "undeclared"}` +
        ` [${value?.shape.join(", ") ?? ""}], expected f32 [1, ${symbol}, ${vision.hiddenSize}]` +
        " (pipelineConfig.vision.hiddenSize)",
    );
  }
  let positions: number | undefined;
  const gathered = new Set<string>();
  for (const node of graph.nodes) {
    const index = node.ins[1];
    if (node.op !== "embedding" || (index !== VISION_POSITION_X && index !== VISION_POSITION_Y)) {
      continue;
    }
    const table = Object.hasOwn(graph.values, node.ins[0]) ? graph.values[node.ins[0]] : undefined;
    const extent = table?.shape[0];
    if (typeof extent !== "number") {
      throw new Error(
        `${OWNER}: the vision graph's table '${node.ins[0]}' gathered by '${index}' has no static row count`,
      );
    }
    gathered.add(index);
    positions = positions === undefined ? extent : Math.min(positions, extent);
  }
  if (positions === undefined || gathered.size !== 2) {
    throw new Error(
      `${OWNER}: the vision graph gathers no table by both '${VISION_POSITION_X}' and` +
        ` '${VISION_POSITION_Y}', so the positions it covers cannot be read`,
    );
  }
  const longestSide = POOLING * vision.softTokenBudget;
  if (longestSide > positions) {
    throw new Error(
      `${OWNER}: pipelineConfig.vision.softTokenBudget ${vision.softTokenBudget} lets an image side` +
        ` reach ${longestSide} patches, but the vision graph's tables cover ${positions} positions`,
    );
  }
  return { output, width: vision.hiddenSize, positions };
};

/**
 * Runs one image's rows through the tower and returns its soft tokens `[n · width]` (the readback as
 * it came: the caller copies it into the text graph's soft rows before the Session's next run).
 *
 * MUST: the shape is checked — a tower that returned another row count would still fill rows, of
 * another image.
 */
export const runVisionTower = async (
  session: Pick<Session, "run">,
  rows: ImagePatchRows,
  contract: EmbeddingGemmaVisionContract,
): Promise<Float32Array<ArrayBuffer>> => {
  const patches = rows.softTokens * PATCHES_PER_SOFT_TOKEN;
  const outputs: RunOutputs = await session.run({
    [VISION_PIXELS]: { dtype: "f32", shape: [1, patches, PATCH_ELEMENTS], data: rows.pixels },
    [VISION_POSITION_X]: { dtype: "i32", shape: [1, patches], data: rows.positionX },
    [VISION_POSITION_Y]: { dtype: "i32", shape: [1, patches], data: rows.positionY },
  });
  const { output, width } = contract;
  if (!Object.hasOwn(outputs, output)) {
    throw new Error(
      `${OWNER}: the vision graph's output '${output}' is missing from the run results`,
    );
  }
  const tensor = outputs[output];
  if (
    tensor.dtype !== "f32" ||
    !sameShape(tensor.shape, [1, rows.softTokens, width]) ||
    tensor.data.length !== rows.softTokens * width
  ) {
    throw new Error(
      `${OWNER}: the vision graph returned ${tensor.dtype} [${tensor.shape.join(", ")}], not f32` +
        ` [1, ${rows.softTokens}, ${width}]`,
    );
  }
  return tensor.data;
};
