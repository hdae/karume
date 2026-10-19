/**
 * Fixed-length greedy generation loop (**pipeline-independent shared code** — autoregressive language
 * models generally go through "feed the prompt in fixed-length chunks → step one token at a time"),
 * so it lives in `src/generation/` and not in a family directory (the same standing as `src/image/` /
 * `src/audio/`).
 *
 * The execution shapes are **only the two** of ADR 0066 decision 4: prefill pads the last chunk with
 * the physical chunk row count fixed at `M = chunkLength`, and decode is `M = queryLength = 1`. What
 * is cut is `queryLength`; the host does not hold the logical length (`pastLength`) — the context
 * advances it when a run succeeds (decision 6's ban on double bookkeeping). So the only thing this
 * loop remembers is "the next token to feed and its absolute position".
 *
 * MUST: no sampling here. The next token is read as is from the graph's argmax output (`[1,M,1]`
 * i32), and no temperature, top-k or RNG is kept on the host (sampling / RNG stay on the host as the
 * op-vocabulary ruling says, but the purpose of this function is **acceptance on a fixed token id
 * sequence**, so it is closed to a single deterministic path).
 */

import type {
  GenerationContext,
  GenerationContextSpec,
  RunInputs,
  RunOutputs,
  SymbolBindings,
  Tensor,
} from "@karume/runtime";

/** The surface the generation loop requires of a context — opaque; only its release is called. */
export type GenerationDisposable = { dispose(): Promise<void> };

/**
 * The surface the generation loop requires of a Session (narrow interface — DI lets a fake stand
 * in).
 *
 * MUST: not `Pick<Session, "run" | "createGenerationContext">`. `GenerationContext` is a **nominal
 * type** with `#` private fields that no GPU-free fake can satisfy however it is written, so with
 * Pick the unit tests would require a real GPU. The context type is threaded through as a type
 * parameter to pin in the types that "only the object create returned goes back into run";
 * collapsing it to `{ dispose }` would (through method bivariance) let passing another context to a
 * real Session pass the type check.
 *
 * MUST: that the real `Session` satisfies this surface is pinned by a type check on the test side
 * (detects spelling drift).
 */
export type GreedySession<C extends GenerationDisposable = GenerationContext> = {
  createGenerationContext(spec: GenerationContextSpec): Promise<C>;
  run(
    inputs: RunInputs,
    bindings: SymbolBindings | undefined,
    generation: { readonly context: C; readonly queryLength: number },
  ): Promise<RunOutputs>;
};

/** Specification of {@link generateGreedy}. */
export type GreedySpec<C extends GenerationDisposable = GenerationContext> = {
  readonly session: GreedySession<C>;
  /** Name of the graph input that receives the token id sequence (`[1,M]` i32). */
  readonly inputIds: string;
  /**
   * Producer of host-derived inputs (the minimal form of the same slot as `DerivedRunInputs` in
   * `program.ts`).
   *
   * It receives the token id sequence and the absolute position sequence **for the physical row
   * count** (prefill including pad rows, decode one row), with 0 in both for pad rows (the value
   * contract of ADR 0066 addendum 6). The return value is spread into the run's inputs as is, so
   * this is the only supplier of position-dependent graph inputs (gemma4's RoPE cos / sin —
   * `gemma/rope.ts`; `position_ids` of the MiniCPM5 decode series).
   *
   * MUST: a **synchronous pure function** (the same value for the same `(ids, positions)`).
   * `sequence.ts` is asynchronous because it needs a place to wait for GB-scale lazy loads (PLE
   * blocks); the purpose of this function is acceptance on a fixed token id sequence, so derived
   * inputs that wait are not passed through.
   */
  readonly derive?: (ids: Int32Array<ArrayBuffer>, positions: Int32Array<ArrayBuffer>) => RunInputs;
  /** Name of the graph output with the argmax token (`[1,M,1]` i32 — `[1,1,1]` for token-only). */
  readonly token: string;
  /**
   * Name of the **last_row input** (`[1]` i32) of the token-only exit (the form ADR 0068 decision 4
   * made the default; that default was superseded by ADR 0083 decision 6). When given, each run is
   * supplied with the index of the last valid row (prefill = `queryLength − 1` / decode = `0`) and
   * the `token` output is read as `[1,1,1]` (the one selected row). When omitted, the current logits
   * opt-in form (read the last valid row from the all-row argmax `[1,M,1]`).
   */
  readonly lastRow?: string;
  /** Rows of a fixed-length prefill chunk (ADR 0066 decision 4 — a plan-time context constant). */
  readonly chunkLength: number;
  /**
   * **Exclusive upper bound** of the absolute positions that can be looked up (positions are
   * `0..maxPosition-1` — the position limit the model declares. For series that ship with the RoPE
   * table baked in, the table's row count is the limit as is; the MiniCPM5 decode series has 512).
   *
   * MUST: not optional. A position beyond the limit raises no exception (the OOB gather of a series
   * with a baked table is non-finite by design — limitations; even a series without a table attends
   * at positions it was never trained on), and argmax **folds those logits into a plausible token
   * id**, so running without knowing the limit makes the second half of generation silently wrong
   * tokens (measured 2026-08-18 — all logits non-finite at position 512 → token 0). Failing here is
   * the only place to fail loudly.
   */
  readonly maxPosition: number;
  /** Symbol bindings of the state slot capacity (passed through to `createGenerationContext`). */
  readonly bindings?: SymbolBindings;
  /** The prompt's token id sequence (length 1 or more). */
  readonly prompt: readonly number[];
  /** Number of tokens to generate (1 or more). */
  readonly maxNewTokens: number;
};

/** The assignment of one prefill ({@link planPrefillChunks}). */
export type PrefillChunk = {
  /** Absolute position of the first valid row (= the number of tokens fed so far). */
  readonly position: number;
  /** Valid row count (`1..chunkLength` — only the last chunk can be shorter than chunkLength). */
  readonly queryLength: number;
};

const assertPositiveInteger = (value: number, where: string): void => {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new Error(`${where} ${value} is not a positive integer`);
  }
};

/**
 * Splits the prompt length into fixed-length chunks (ADR 0066 decision 4 — neither variable-length
 * chunks nor feeding everything at once).
 *
 * Only the last chunk has `queryLength < chunkLength`, and the difference is pad rows. It is a
 * **pure function** because split errors (a skipped position / an overlap / a dropped tail) belong
 * where they fail without running the GPU; the entry check of `generateGreedy` rides on this one
 * function too.
 */
export const planPrefillChunks = (
  promptLength: number,
  chunkLength: number,
): readonly PrefillChunk[] => {
  assertPositiveInteger(promptLength, "prompt length");
  assertPositiveInteger(chunkLength, "chunkLength");
  const chunks: PrefillChunk[] = [];
  for (let position = 0; position < promptLength; position += chunkLength) {
    chunks.push({ position, queryLength: Math.min(chunkLength, promptLength - position) });
  }
  return chunks;
};

/** One i32 input tensor (both the token id sequence and the absolute positions are `[1, rows]`). */
const i32Row = (rows: number, data: Int32Array<ArrayBuffer>): Tensor => ({
  dtype: "i32",
  shape: [1, rows],
  data,
});

/** The last_row input of the token-only form (`[1]` i32 — the index of the row to select). */
const lastRowInput = (row: number): Tensor => ({
  dtype: "i32",
  shape: [1],
  data: Int32Array.of(row),
});

const prefillInputs = <C extends GenerationDisposable>(
  spec: GreedySpec<C>,
  chunk: PrefillChunk,
): RunInputs => {
  // MUST: pad rows stay 0 (the value contract of ADR 0066 addendum 6 — non-finite pad-row values
  // would leak into valid rows through the "−inf add + exp" path, so the host fills them with 0).
  // The initial value of `Int32Array` is that contract as is, so only the first queryLength rows
  // need filling.
  const ids = new Int32Array(spec.chunkLength);
  const positions = new Int32Array(spec.chunkLength);
  for (let row = 0; row < chunk.queryLength; row += 1) {
    ids[row] = spec.prompt[chunk.position + row];
    positions[row] = chunk.position + row;
  }
  return {
    [spec.inputIds]: i32Row(spec.chunkLength, ids),
    // The token-only form selects the last **valid** row by index (the lm_head does not run on pad
    // rows — ADR 0068 decision 4).
    ...(spec.lastRow === undefined ? {} : { [spec.lastRow]: lastRowInput(chunk.queryLength - 1) }),
    ...(spec.derive === undefined ? {} : spec.derive(ids, positions)),
  };
};

const decodeInputs = <C extends GenerationDisposable>(
  spec: GreedySpec<C>,
  token: number,
  position: number,
): RunInputs => {
  const ids = Int32Array.of(token);
  const positions = Int32Array.of(position);
  return {
    [spec.inputIds]: i32Row(1, ids),
    ...(spec.lastRow === undefined ? {} : { [spec.lastRow]: lastRowInput(0) }),
    ...(spec.derive === undefined ? {} : spec.derive(ids, positions)),
  };
};

/**
 * Reads a token id from one row of the argmax output.
 *
 * MUST: check the dtype and shape here (fail loudly). **A mixed-up output name** that happens to
 * catch another output with a matching shape (logits, for example) returns a wrong token sequence
 * with no exception and no warning, so together with looking up the name, this checks that it has
 * "the shape of an argmax token sequence".
 */
const readToken = (outputs: RunOutputs, name: string, rows: number, row: number): number => {
  if (!Object.hasOwn(outputs, name)) throw new Error(`Graph output '${name}' is missing`);
  const tensor = outputs[name];
  if (tensor.dtype !== "i32") {
    throw new Error(
      `Graph output '${name}' is not i32 (${tensor.dtype}) — it does not point at the argmax token sequence`,
    );
  }
  if (
    tensor.shape.length !== 3 ||
    tensor.shape[0] !== 1 ||
    tensor.shape[1] !== rows ||
    tensor.shape[2] !== 1
  ) {
    throw new Error(
      `Graph output '${name}' has shape [${tensor.shape.join(",")}], not [1,${rows},1]`,
    );
  }
  return tensor.data[row];
};

/**
 * Generates `maxNewTokens` tokens greedily from the prompt (the two shapes of ADR 0066 decision 4).
 *
 * NOTE: **no EOS stop** — the purpose is parity acceptance on a fixed token id sequence, and a stop
 * condition would mix "where it stopped" into what is compared (practical stopping is the job of the
 * caller that looks at this return value).
 */
export const generateGreedy = async <C extends GenerationDisposable>(
  spec: GreedySpec<C>,
): Promise<number[]> => {
  const { session, prompt, chunkLength, maxNewTokens, maxPosition } = spec;
  const chunks = planPrefillChunks(prompt.length, chunkLength);
  assertPositiveInteger(maxNewTokens, "maxNewTokens");
  assertPositiveInteger(maxPosition, "maxPosition");
  prompt.forEach((id, index) => {
    // Writing to an `Int32Array` **silently** truncates non-integers and wraps values outside the
    // i32 range (e.g. 2^32+1 → 1 — it turns into another valid token id without an exception), so
    // fail at the entry.
    if (!Number.isSafeInteger(id) || id < 0 || id > 0x7fffffff) {
      throw new Error(`prompt[${index}] ${id} is not an integer in 0..2147483647`);
    }
  });
  // The largest absolute position touched = T + maxNewTokens - 2 at the last decode step (with zero
  // decodes it is prefill's T - 1, which folds into the same formula). Outside the table means
  // silently wrong tokens (the JSDoc of maxPosition), so fail here.
  const lastPosition = prompt.length + maxNewTokens - 2;
  if (lastPosition >= maxPosition) {
    throw new Error(
      `prompt ${prompt.length} + maxNewTokens ${maxNewTokens} reaches the last position ${lastPosition}` +
        ` (outside maxPosition ${maxPosition} — this asset can look up only positions 0..${maxPosition - 1})`,
    );
  }

  // MUST: the capacity symbol (a states-only symbol) is bound **only** at context creation and is
  // not passed to the run's bindings (ADR 0066 addendum 7 — only input-derived bindings take part
  // in the run's symbol resolution).
  const context = await session.createGenerationContext({
    bindings: spec.bindings,
    chunkLength,
  });
  try {
    // The token-only form outputs only the selected row (`[1,1,1]`), so both the shape and the
    // index read are fixed.
    const tokenOnly = spec.lastRow !== undefined;
    let token = 0;
    for (const chunk of chunks) {
      const outputs = await session.run(prefillInputs(spec, chunk), undefined, {
        context,
        queryLength: chunk.queryLength,
      });
      // Only **the last valid row of the last chunk** starts generation (there is at least one
      // chunk, so this initial value is always overwritten). Outputs of earlier chunks are dropped.
      token = tokenOnly
        ? readToken(outputs, spec.token, 1, 0)
        : readToken(outputs, spec.token, chunkLength, chunk.queryLength - 1);
    }

    const generated = [token];
    // Decode has the shape "placing `g_i` at position `T + i` yields `g_{i+1}`". Feeding the
    // previous step's return value straight into the next input is greedy's feedback, and this is
    // the only state.
    for (let index = 0; index + 1 < maxNewTokens; index += 1) {
      const outputs = await session.run(
        decodeInputs(spec, generated[index], prompt.length + index),
        undefined,
        { context, queryLength: 1 },
      );
      generated.push(readToken(outputs, spec.token, 1, 0));
    }
    return generated;
  } finally {
    // MUST: release it even after a failure midway (a failed context is poisoned and cannot be
    // reused — ADR 0066 addendum 3. Holding on to it piles up VRAM of the KV capacity per
    // generation).
    await context.dispose();
  }
};
