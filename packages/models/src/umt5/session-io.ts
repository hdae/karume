/**
 * Pure functions that build the inputs and read the outputs of the umT5-XXL graph (ADR 0119
 * decisions 3 and 4) on the host (used by the standalone `Umt5Encoder` (`./encoder.ts`) and by Wan's
 * text stage (`encodeWanPrompts` in `src/wan/text-stage.ts`)).
 *
 * - Input: bundles the tokenizer's id sequence `[L]` (`Umt5Tokenizer.tokenize` in `./tokenizer.ts`)
 *   and the bucket table `[L, L]` (`./relative-position.ts`) into the graph inputs token id `[1, L]`
 *   and bucket table. L is a dynamic dimension of one symbol, and this checks that both inputs have
 *   the same L.
 * - Output: checks the graph output `[1, L, width]` and reads it as `[L, width]` (row-major, f32)
 *   ({@link umt5HiddenStates}). To pass it to Wan's DiT, it is copied into the context input
 *   `encoder_hidden_states [1, rows, width]` and zero-padded past the valid length
 *   ({@link padUmt5Context} — the same as `new_zeros` in upstream's `_get_t5_prompt_embeds`, and the
 *   same rule as `padWanTextEmbedding` on the precomputed-asset path). There is no mask (decision 4).
 *
 * A mismatch on either side is an assembly error, not an input error (a plain `Error` — the
 * classification of ADR 0107 decision 2).
 *
 * MUST: modules are side-effect free (CLAUDE.md).
 */

import type { Tensor } from "@karume/runtime";
import { type I32Tensor, UMT5_MAX_LENGTH, UMT5_MIN_TOKENS } from "./relative-position.ts";

/**
 * The graph input names (`INPUT_NAMES` in the recipe `wan/umt5_patch.py` — the fixture test checks
 * that they match).
 */
export const UMT5_INPUT_IDS = "input_ids";
export const UMT5_RELATIVE_POSITION_BUCKETS = "relative_position_buckets";

/** The bundle of inputs passed to the Session (both i32). */
export type Umt5SessionInputs = {
  readonly [UMT5_INPUT_IDS]: I32Tensor;
  readonly [UMT5_RELATIVE_POSITION_BUCKETS]: I32Tensor;
};

/**
 * Bundles the id sequence and the bucket table into graph inputs.
 *
 * `ids` and `buckets.data` are passed as they are, without copying (Session inputs are borrowed —
 * not modified until the run settles; the contract is in the doc of `@karume/runtime`'s `Tensor`).
 */
export const umt5SessionInputs = (
  ids: Int32Array<ArrayBuffer>,
  buckets: I32Tensor,
): Umt5SessionInputs => {
  const length = ids.length;
  if (length < UMT5_MIN_TOKENS || length > UMT5_MAX_LENGTH) {
    throw new Error(
      `umT5 id sequence length ${length} is outside ${UMT5_MIN_TOKENS}..${UMT5_MAX_LENGTH}`,
    );
  }
  const [rows, columns] = buckets.shape;
  if (buckets.shape.length !== 2 || rows !== length || columns !== length) {
    const shape = buckets.shape.join(",");
    throw new Error(
      `umT5 bucket table [${shape}] is not a square of the id sequence length ${length}`,
    );
  }
  if (buckets.data.length !== length * length) {
    throw new Error(
      `umT5 bucket table element count ${buckets.data.length} does not match [${length},${length}]`,
    );
  }
  return {
    [UMT5_INPUT_IDS]: { dtype: "i32", shape: [1, length], data: ids },
    [UMT5_RELATIVE_POSITION_BUCKETS]: buckets,
  };
};

/**
 * Checks the graph output `[1, tokens, width]` and returns its contents as `[tokens, width]`
 * (row-major, f32) (no copy — the Session's output belongs to the caller).
 *
 * `tokens` is the length of the id sequence passed as input (the output's row count must match it —
 * the shape catches a mix-up with another run's output). `width` is the output width of the graph
 * declaration (`hiddenSize` in `./graph-contract.ts`).
 */
export const umt5HiddenStates = (
  output: Tensor,
  tokens: number,
  width: number,
): Float32Array<ArrayBuffer> => {
  if (output.dtype !== "f32") {
    throw new Error(`umT5 output dtype ${output.dtype} is not f32`);
  }
  const [batch, length, channels] = output.shape;
  if (output.shape.length !== 3 || batch !== 1 || length !== tokens || channels !== width) {
    throw new Error(`umT5 output [${output.shape.join(",")}] is not [1,${tokens},${width}]`);
  }
  if (output.data.length !== tokens * width) {
    throw new Error(
      `umT5 output element count ${output.data.length} does not match [1,${tokens},${width}]`,
    );
  }
  return output.data;
};

/**
 * Zero-pads the graph output `[1, tokens, width]` into the DiT's context `[rows, width]` (row-major,
 * f32) (checked by {@link umt5HiddenStates}).
 *
 * `rows` / `width` come from the declaration of `encoder_hidden_states` in the DiT graph (as in
 * `src/wan/text-stage.ts` — 512 / 4,096 are not copied in by hand).
 */
export const padUmt5Context = (
  output: Tensor,
  tokens: number,
  rows: number,
  width: number,
): Float32Array<ArrayBuffer> => {
  const states = umt5HiddenStates(output, tokens, width);
  if (tokens > rows) {
    throw new Error(`umT5 output row count ${tokens} exceeds the DiT context row count ${rows}`);
  }
  const padded = new Float32Array(rows * width);
  padded.set(states);
  return padded;
};
