/**
 * Patching of the reference latent (left on the host — ADR 0047 decision 4).
 *
 * The input of the `speaker` graph is the patched `[1, tokens, latentDim×patchSize]`, and the patch
 * itself is a pure reshape, so there is no point in putting it into the IR.
 */

/**
 * Minimum of the symbolic dimension S (token count after patching) of the `speaker` graph. It is
 * exactly the export's `Dim(min=MIN_SYM_LENGTH)` (`tools/export-recipes/irodori/export.py` — the
 * default 2 that avoids torch.export's 0/1 specialization); the IR does not carry minimums, so the
 * host keeps a copy. A 1-token reference **cannot be fed to the graph**.
 */
export const SPEAKER_MIN_TOKENS = 2;

/** The reference latent after patching. */
type PatchedReference = {
  readonly data: Float32Array<ArrayBuffer>;
  readonly tokens: number;
  /** `latentDim × patchSize`. */
  readonly width: number;
};

/**
 * Folds the reference latent `[frames, latentDim]` into `[frames/patchSize, latentDim×patchSize]`.
 *
 * MUST: leftover frames are **discarded** (`usable` of upstream's `patch_sequence_with_mask`).
 * Rounding up and filling with 0 would feed nonexistent silent frames into the encoder as features
 * of the reference speaker.
 *
 * MUST: a reference that yields fewer than {@link SPEAKER_MIN_TOKENS} tokens fails (fewer than
 * patch width × 2 frames). This function also receives the output of the codec encoder (= a model
 * output), so a plain `Error` — on the path of a latent the caller passed, `patchRequestedLatent`
 * in `conditioning.ts` raises the thrown type to input-caused.
 */
export const patchReferenceLatent = (
  latent: Float32Array<ArrayBuffer>,
  latentDim: number,
  patchSize: number,
): PatchedReference => {
  if (latent.length % latentDim !== 0) {
    throw new Error(
      `Reference latent length ${latent.length} is not a multiple of latentDim ${latentDim}`,
    );
  }
  const frames = latent.length / latentDim;
  const tokens = Math.floor(frames / patchSize);
  if (tokens < SPEAKER_MIN_TOKENS) {
    throw new Error(
      `Reference latent frame count ${frames} is less than patch width ${patchSize} × ${SPEAKER_MIN_TOKENS}` +
        ` (${tokens} tokens — the minimum of the speaker symbolic dimension is ${SPEAKER_MIN_TOKENS} tokens)`,
    );
  }
  const width = latentDim * patchSize;
  // The prefix without the leftover is already laid out row-major (the patch is only a reshape).
  return { data: latent.slice(0, tokens * width), tokens, width };
};
