/**
 * The host side of the MOSS-Image DiT's S-form graph (S = N + T — one symbol) — the joint-sequence
 * inputs and the head rows (ADR 0138 decision 2, stage 2).
 *
 * Upstream concatenates `[video_embedder(image tokens), refined text]` and keeps the first N rows of
 * the head; the IR has neither a sum of symbols nor a symbolic slice, so `dit` takes S-row inputs
 * and selects per row with `where(is_image, video_embedder(latent_tokens), text_context)`, and the
 * head runs over all S rows. The host builds:
 *
 * - `latent_tokens [1,S,C]`: the patchified latents in the N image rows (upstream's
 *   `flatten(2).transpose(1, 2)` — h-major, token `h·W + w`, element `(token, c)` =
 *   `latents[c][h][w]`), zeros in the T text rows;
 * - `text_context [1,S,D]`: zeros in the image rows, the `dit_text` output in the text rows;
 * - `is_image [1,S,1]` (bool — u32 0 / 1): 1 on the N image rows;
 * - from the output `[1,S,C]`, upstream's sample `[C,H,W]` from the first N rows.
 *
 * These are `tools/export-recipes/moss/dit_patch.py`'s `dit_inputs` / `head_rows`, transcribed; the
 * host tests check them bit for bit against that recipe's goldens.
 *
 * The latents are `[C,H,W]` without a batch axis (B = 1 — CFG runs two forwards).
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state).
 */

import type { MossCells } from "./dit-rope.ts";

/** The image token count N of a latent grid. */
export const mossImageTokens = (cells: MossCells): number => cells.rows * cells.cols;

const assertLength = (
  values: ArrayLike<number>,
  expected: number,
  what: string,
  shape: string,
): void => {
  if (values.length !== expected) {
    throw new Error(`${what}: ${values.length} elements do not match ${shape}`);
  }
};

/**
 * `latent_tokens` (`S·channels`, the flattened `[1,S,channels]`) of latents `[channels, H, W]` for a
 * text of `textLength` tokens.
 */
export const mossLatentTokens = (
  latents: Float32Array<ArrayBuffer>,
  channels: number,
  cells: MossCells,
  textLength: number,
): Float32Array<ArrayBuffer> => {
  const image = mossImageTokens(cells);
  assertLength(
    latents,
    channels * image,
    "latent tokens",
    `[${channels}, ${cells.rows}, ${cells.cols}]`,
  );
  const tokens = new Float32Array((image + textLength) * channels);
  for (let channel = 0; channel < channels; channel += 1) {
    const plane = channel * image;
    for (let token = 0; token < image; token += 1) {
      tokens[token * channels + channel] = latents[plane + token];
    }
  }
  return tokens;
};

/** `is_image` (`S` elements, the flattened `[1,S,1]` — u32 0 / 1). */
export const mossIsImage = (cells: MossCells, textLength: number): Uint32Array<ArrayBuffer> => {
  const image = mossImageTokens(cells);
  const mask = new Uint32Array(image + textLength);
  mask.fill(1, 0, image);
  return mask;
};

/**
 * `text_context` (`S·width`, the flattened `[1,S,width]`) from the `dit_text` output `[T, width]`.
 */
export const mossTextContext = (
  refined: Float32Array<ArrayBuffer>,
  width: number,
  cells: MossCells,
  textLength: number,
): Float32Array<ArrayBuffer> => {
  assertLength(refined, textLength * width, "text context", `[${textLength}, ${width}]`);
  const image = mossImageTokens(cells);
  const context = new Float32Array((image + textLength) * width);
  context.set(refined, image * width);
  return context;
};

/**
 * Upstream's sample `[channels, H, W]` from the graph output (`S·channels`, `[1,S,channels]`): the
 * first N rows, transposed to channel-major.
 */
export const mossHeadRows = (
  output: Float32Array<ArrayBuffer>,
  channels: number,
  cells: MossCells,
  textLength: number,
): Float32Array<ArrayBuffer> => {
  const image = mossImageTokens(cells);
  assertLength(
    output,
    (image + textLength) * channels,
    "head rows",
    `[1, ${image + textLength}, ${channels}]`,
  );
  const sample = new Float32Array(channels * image);
  for (let token = 0; token < image; token += 1) {
    const row = token * channels;
    for (let channel = 0; channel < channels; channel += 1) {
      sample[channel * image + token] = output[row + channel];
    }
  }
  return sample;
};
