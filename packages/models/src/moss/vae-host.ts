/**
 * The host side of the MOSS-Image VAE stage — the three steps upstream runs around `vae.decode`
 * (`MOSSImagePipeline.__call__`, transcribed in `tools/export-recipes/moss/pipeline_ref.py` as
 * `denormalize` / `unpatchify`; ADR NNNN decision 8, plan step 3):
 *
 * 1. {@link mossDenormalizeLatents}: the latent BatchNorm undone with its running statistics,
 *    `z · std + mean` per DiT channel. `std = (running_var + eps).sqrt()` comes precomputed in the
 *    asset `latent_batch_norm` (upstream's expression, evaluated by the recipe — the host takes no
 *    square root), so the host does upstream's two f32 operations: a multiply, then an add (each
 *    rounded to f32 — `Math.fround` of a double product / sum of two f32 values is the f32 result).
 * 2. {@link mossUnpatchify}: `[4·Cz, h, w]` → `[Cz, 2h, 2w]` — upstream's
 *    `reshape(b, c/4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c/4, 2h, 2w)`:
 *    `out[c][2i + a][2j + b] = in[4c + 2a + b][i][j]` (data movement only).
 * 3. {@link mossClampImage}: `decoded.clamp(-1, 1)`, keeping NaN as NaN (torch's clamp propagates
 *    it; the stage then fails loudly on any non-finite value).
 *
 * The tests check all three Uint32-identical to the stage-0 records (`moss.vae_reference`).
 *
 * The tensors carry no batch axis (B = 1): latents `[128, H/16, W/16]`, the decode input
 * `[32, H/8, W/8]`, the image `[3, H, W]`, all row-major.
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state).
 */

import type { MossCells } from "./dit-rope.ts";
import type { MossLatentBatchNorm } from "./vae-contract.ts";

const OWNER = "MOSS-Image";

/** The DiT's 2×2 latent patch (one DiT cell is 2×2 VAE cells). */
const PATCH = 2;

/** Step 1 (module doc): `latents · std + mean` per channel, each operation rounded to f32. */
export const mossDenormalizeLatents = (
  latents: Float32Array<ArrayBuffer>,
  batchNorm: MossLatentBatchNorm,
  cells: MossCells,
): Float32Array<ArrayBuffer> => {
  const channels = batchNorm.mean.length;
  const plane = cells.rows * cells.cols;
  if (latents.length !== channels * plane) {
    throw new Error(
      `${OWNER}: ${latents.length} latents for [${channels}, ${cells.rows}, ${cells.cols}]`,
    );
  }
  const f32 = Math.fround;
  const out = new Float32Array(latents.length);
  for (let c = 0; c < channels; c += 1) {
    const std = batchNorm.std[c];
    const mean = batchNorm.mean[c];
    for (let at = c * plane; at < (c + 1) * plane; at += 1) {
      out[at] = f32(f32(latents[at] * std) + mean);
    }
  }
  return out;
};

/** Step 2 (module doc): `[4·Cz, h, w]` → `[Cz, 2h, 2w]`. */
export const mossUnpatchify = (
  latents: Float32Array<ArrayBuffer>,
  channels: number,
  cells: MossCells,
): Float32Array<ArrayBuffer> => {
  const { rows, cols } = cells;
  if (channels % (PATCH * PATCH) !== 0 || latents.length !== channels * rows * cols) {
    throw new Error(
      `${OWNER}: unpatchify of ${latents.length} values as [${channels}, ${rows}, ${cols}]`,
    );
  }
  const outChannels = channels / (PATCH * PATCH);
  const outRows = PATCH * rows;
  const outCols = PATCH * cols;
  const out = new Float32Array(latents.length);
  for (let c = 0; c < outChannels; c += 1) {
    for (let a = 0; a < PATCH; a += 1) {
      for (let b = 0; b < PATCH; b += 1) {
        const source = (c * PATCH * PATCH + a * PATCH + b) * rows * cols;
        for (let i = 0; i < rows; i += 1) {
          const target = (c * outRows + PATCH * i + a) * outCols + b;
          for (let j = 0; j < cols; j += 1) {
            out[target + PATCH * j] = latents[source + i * cols + j];
          }
        }
      }
    }
  }
  return out;
};

/** Step 3 (module doc): `clamp(-1, 1)` with NaN kept. */
export const mossClampImage = (decoded: Float32Array<ArrayBuffer>): Float32Array<ArrayBuffer> =>
  decoded.map((value) => (value > 1 ? 1 : value < -1 ? -1 : value));

/** Steps 1 and 2: the decode input `[Cz, 2h, 2w]` of the final DiT latents `[4·Cz, h, w]`. */
export const mossVaeInput = (
  latents: Float32Array<ArrayBuffer>,
  batchNorm: MossLatentBatchNorm,
  cells: MossCells,
): Float32Array<ArrayBuffer> =>
  mossUnpatchify(mossDenormalizeLatents(latents, batchNorm, cells), batchNorm.mean.length, cells);
