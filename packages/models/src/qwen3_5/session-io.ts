/**
 * Pure functions that build the inputs and read the output of the Qwen3.5 text tower graph on the
 * host (ADR 0137 decision 2; used by `Qwen3_5TextEncoder` in `./encoder.ts`).
 *
 * - Input: one id sequence `[T]` (the tokenizer's output — `Qwen3_5Tokenizer.tokenize` in
 *   `./tokenizer.ts`, after the chat template of `./chat.ts`) becomes the four graph inputs
 *   `input_ids` i32 `[1, T]`, `positions` f32 `[T]` (0 … T−1) and the RoPE rows `rope_cos` /
 *   `rope_sin` f32 `[1, T, 64]` (`./rope.ts`). T is the graph's one symbolic dimension, so every
 *   input carries the same T by construction.
 * - Output: checks the graph output `[1, T, width]` and reads it as `[T, width]` (row-major, f32)
 *   ({@link qwen3_5HiddenStates}). No padding and no mask — the graph runs on the valid length.
 *
 * A violation on either side is an assembly error, not an input error (a plain `Error` — the
 * classification of ADR 0107 decision 2): the length bound of user text is the tokenizer's
 * (`ModelInputError`, the maintainer's ruling 50), so an id sequence that reaches this module
 * outside the bounds skipped it.
 *
 * MUST: modules are side-effect free (CLAUDE.md).
 */

import type { Tensor } from "@karume/runtime";
import { buildQwen3_5Positions, buildQwen3_5RopeRows, type F32Tensor } from "./rope.ts";

/**
 * The graph input names, in the recipe's order (`INPUT_NAMES` in
 * `tools/export-recipes/qwen3_5/patch.py`).
 */
export const QWEN3_5_INPUT_IDS = "input_ids";
export const QWEN3_5_POSITIONS = "positions";
export const QWEN3_5_ROPE_COS = "rope_cos";
export const QWEN3_5_ROPE_SIN = "rope_sin";

/**
 * The bounds of the valid length T (`MIN_TOKENS` / `MAX_TOKENS` in the recipe's `patch.py` — the
 * export's `Dim("T", min, max)`). The upper bound is MOSS-Image's `max_sequence_length` and the
 * tokenizer asset's `maxLength` (4096); a caller that pairs the tokenizer with this encoder checks
 * that the two agree, as Wan does for umT5 (`src/wan/text-stage.ts`).
 */
export const QWEN3_5_MIN_TOKENS = 1;
export const QWEN3_5_MAX_TOKENS = 4096;

/** An i32 tensor (the type of `input_ids`). */
type I32Tensor = Extract<Tensor, { readonly dtype: "i32" }>;

/** The bundle of inputs passed to the Session. */
export type Qwen3_5SessionInputs = {
  readonly [QWEN3_5_INPUT_IDS]: I32Tensor;
  readonly [QWEN3_5_POSITIONS]: F32Tensor;
  readonly [QWEN3_5_ROPE_COS]: F32Tensor;
  readonly [QWEN3_5_ROPE_SIN]: F32Tensor;
};

/**
 * Builds the four graph inputs for one id sequence.
 *
 * `vocabularySize` is the row count of the graph's embedding table (`./graph-contract.ts`). An id
 * outside `0 … vocabularySize−1` fails here: on the GPU the embedding writes NaN rows for it (the
 * runtime's out-of-range ruling), which would surface only as a non-finite output with no pointer
 * to the id.
 *
 * `ids` is passed as is, without copying (Session inputs are borrowed — not modified until the run
 * settles; the contract is in the doc of `@karume/runtime`'s `Tensor`).
 */
export const qwen3_5SessionInputs = (
  ids: Int32Array<ArrayBuffer>,
  vocabularySize: number,
): Qwen3_5SessionInputs => {
  const length = ids.length;
  if (length < QWEN3_5_MIN_TOKENS || length > QWEN3_5_MAX_TOKENS) {
    throw new Error(
      `Qwen3.5 id sequence length ${length} is outside ${QWEN3_5_MIN_TOKENS}..${QWEN3_5_MAX_TOKENS}`,
    );
  }
  const outside = ids.findIndex((id) => id < 0 || id >= vocabularySize);
  if (outside !== -1) {
    throw new Error(
      `Qwen3.5 id ${ids[outside]} at index ${outside} is outside the embedding's ` +
        `${vocabularySize} rows`,
    );
  }
  const rope = buildQwen3_5RopeRows(length);
  return {
    [QWEN3_5_INPUT_IDS]: { dtype: "i32", shape: [1, length], data: ids },
    [QWEN3_5_POSITIONS]: buildQwen3_5Positions(length),
    [QWEN3_5_ROPE_COS]: rope.cos,
    [QWEN3_5_ROPE_SIN]: rope.sin,
  };
};

/**
 * Checks the graph output `[1, tokens, width]` and returns its contents as `[tokens, width]`
 * (row-major, f32) (no copy — the Session's output belongs to the caller).
 *
 * `tokens` is the length of the id sequence passed as input (the row count must match it — the
 * shape catches a mix-up with another run's output). `width` is the output width of the graph
 * declaration (`hiddenSize` in `./graph-contract.ts`).
 */
export const qwen3_5HiddenStates = (
  output: Tensor,
  tokens: number,
  width: number,
): Float32Array<ArrayBuffer> => {
  if (output.dtype !== "f32") {
    throw new Error(`Qwen3.5 output dtype ${output.dtype} is not f32`);
  }
  const [batch, length, channels] = output.shape;
  if (output.shape.length !== 3 || batch !== 1 || length !== tokens || channels !== width) {
    throw new Error(`Qwen3.5 output [${output.shape.join(",")}] is not [1,${tokens},${width}]`);
  }
  if (output.data.length !== tokens * width) {
    throw new Error(
      `Qwen3.5 output element count ${output.data.length} does not match [1,${tokens},${width}]`,
    );
  }
  return output.data;
};
