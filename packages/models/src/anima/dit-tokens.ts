/**
 * Host side of the DiT's **S-form graph** (one token-length symbol) — ADR 0034.
 *
 * ## Why this moves to the host
 *
 * The static-shape DiT carries three constants with the resolution baked in (the padding channel's
 * `[1,1,H,W]` and the rope cos / sin tables `[1,1,S,128]`). The IR's dimension language is "a
 * linear expression of one symbol per dimension", so `H·W` cannot be written, but **moving the
 * graph entry behind patchify** lets every node be written with the single token-length symbol `S`.
 * In exchange, patchify / unpatchify / building the rope tables come down here. The runtime does
 * not change by a line (the plan is per run; the pipeline key holds no shape).
 *
 * Per ADR 0038 §4 **the distributed transformer is S-form only**, so this path is the only path.
 */

import { ROPE_AXIS_COUNT, type RopeBase, ropeWidth } from "./rope-base.ts";

/**
 * Spatial compression ratio of the VAE (latent 1 → sample 8px).
 *
 * **Why it is a constant**: the S-form DiT holds no resolution at all, so "resolution → latent
 * side" cannot be read from the assets. A mix-up is caught on every run by the "output image size
 * == requested resolution" check at the end of generation — the VAE decoder's input/output shape is
 * the source of truth for the scale, and this is its copy.
 */
export const ANIMA_SPATIAL_COMPRESSION = 8;

/** patchify step and latent channel count (derived from the graph's input/output widths). */
export type DitPatchGeometry = {
  /** Latent channel count `C` (**excluding** the padding channel). */
  readonly channels: number;
  /** Patch height `ph`. */
  readonly patchHeight: number;
  /** Patch width `pw`. */
  readonly patchWidth: number;
};

/**
 * Derives the patch geometry from the graph's `tokens` input width and output width.
 *
 * The entry is `(C+1)·pt·ph·pw` (one extra padding channel) and the exit is `pt·ph·pw·C`, so the
 * difference is `P = pt·ph·pw` and the quotient is `C`. The derivation exists so that **callers do
 * not put 68 / 64 / 16 as literals** — the same rule as the tile geometry reading the scale from
 * the assets.
 *
 * MUST: assumes `pt = 1` (image) and `ph = pw` (square patch). Anima's `patch_size = (1,2,2)`
 * satisfies this, but a configuration that does not **cannot be derived**, so it fails instead of
 * continuing silently (exposed when `P` is not a perfect square).
 */
export const ditPatchGeometry = (tokenWidth: number, outputWidth: number): DitPatchGeometry => {
  const patchVolume = tokenWidth - outputWidth;
  if (patchVolume <= 0 || outputWidth <= 0 || outputWidth % patchVolume !== 0) {
    throw new Error(
      `Cannot derive the DiT patch geometry (tokens width ${tokenWidth} / output width ${outputWidth})`,
    );
  }
  const side = Math.round(Math.sqrt(patchVolume));
  if (side * side !== patchVolume) {
    throw new Error(
      `Patch volume ${patchVolume} is not a perfect square (the assumption of pt=1 and a square patch does not hold)`,
    );
  }
  return { channels: outputWidth / patchVolume, patchHeight: side, patchWidth: side };
};

/** Checks a `[1,C,H,W]` latent shape and returns `(H, W)`. */
const latentSides = (
  latentShape: readonly number[],
  geometry: DitPatchGeometry,
  where: string,
): readonly [number, number] => {
  if (latentShape.length !== 4 || latentShape[0] !== 1) {
    throw new Error(`${where}: assumes a [1,C,H,W] latent ([${latentShape}])`);
  }
  if (latentShape[1] !== geometry.channels) {
    throw new Error(
      `${where}: latent channel count ${latentShape[1]} does not match the graph's ${geometry.channels}`,
    );
  }
  const [, , height, width] = latentShape;
  if (height % geometry.patchHeight !== 0 || width % geometry.patchWidth !== 0) {
    throw new Error(
      `${where}: latent ${height}×${width} is not divisible by patch ${geometry.patchHeight}×${geometry.patchWidth}`,
    );
  }
  return [height, width];
};

/** Token length `S` for a latent `[1,C,H,W]`. */
export const tokenCount = (latentShape: readonly number[], geometry: DitPatchGeometry): number => {
  const [height, width] = latentSides(latentShape, geometry, "token length");
  return (height / geometry.patchHeight) * (width / geometry.patchWidth);
};

/**
 * `latents [1,C,H,W]` → `tokens [1,S,(C+1)·ph·pw]` (including the always-zero padding channel).
 *
 * The last dimension is ordered **`(c, ph, pw)`** (= `c·ph·pw + ih·pw + iw`), and the token index
 * is `h·W' + w`. The source of truth is the stage before proj in upstream's `CosmosPatchEmbed`;
 * this is its index form.
 *
 * MUST: the padding channel is **always zero** (the pipeline passes `latents.new_zeros`). Making it
 * a copy of the latent changes only the values while shape and length still match.
 */
export const patchifyLatents = (
  latents: Float32Array<ArrayBuffer>,
  latentShape: readonly number[],
  geometry: DitPatchGeometry,
): Float32Array<ArrayBuffer> => {
  const [height, width] = latentSides(latentShape, geometry, "patchify");
  const { channels, patchHeight, patchWidth } = geometry;
  if (latents.length !== channels * height * width) {
    throw new Error(`patchify: element count ${latents.length} does not match [${latentShape}]`);
  }
  const rows = height / patchHeight;
  const cols = width / patchWidth;
  const patchVolume = patchHeight * patchWidth;
  const tokenWidth = (channels + 1) * patchVolume;
  const tokens = new Float32Array(rows * cols * tokenWidth);
  for (let channel = 0; channel < channels; channel += 1) {
    const plane = channel * height * width;
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const base = (row * cols + col) * tokenWidth + channel * patchVolume;
        for (let innerH = 0; innerH < patchHeight; innerH += 1) {
          const line = plane + (row * patchHeight + innerH) * width + col * patchWidth;
          for (let innerW = 0; innerW < patchWidth; innerW += 1) {
            tokens[base + innerH * patchWidth + innerW] = latents[line + innerW];
          }
        }
      }
    }
  }
  // The padding channel (index `channels`) keeps Float32Array's initial 0 = always zero.
  return tokens;
};

/**
 * `tokens [1,S,ph·pw·C]` → `latents [1,C,H,W]`.
 *
 * MUST: this is **not the reverse of patchify**. The last dimension is ordered `(ih, iw, C)`
 * (= `(ih·pw + iw)·C + c`), different from patchify's `(c, ih, iw)` — as upstream's comment says;
 * "fixing it to the reverse order" scatters the pixels while the shape still matches.
 */
export const unpatchifyTokens = (
  tokens: Float32Array<ArrayBuffer>,
  latentShape: readonly number[],
  geometry: DitPatchGeometry,
): Float32Array<ArrayBuffer> => {
  const [height, width] = latentSides(latentShape, geometry, "unpatchify");
  const { channels, patchHeight, patchWidth } = geometry;
  const rows = height / patchHeight;
  const cols = width / patchWidth;
  const tokenWidth = patchHeight * patchWidth * channels;
  if (tokens.length !== rows * cols * tokenWidth) {
    throw new Error(
      `unpatchify: element count ${tokens.length} does not match [1,${rows * cols},${tokenWidth}]`,
    );
  }
  const latents = new Float32Array(channels * height * width);
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const token = (row * cols + col) * tokenWidth;
      for (let innerH = 0; innerH < patchHeight; innerH += 1) {
        for (let innerW = 0; innerW < patchWidth; innerW += 1) {
          const inner = (innerH * patchWidth + innerW) * channels;
          const at = (row * patchHeight + innerH) * width + col * patchWidth + innerW;
          for (let channel = 0; channel < channels; channel += 1) {
            latents[channel * height * width + at] = tokens[token + inner + channel];
          }
        }
      }
    }
  }
  return latents;
};

/**
 * Builds the `[1,1,S,head_dim]` cos / sin tables (images only, F'=1).
 *
 * One row is the block concatenation `[t, h, w, t, h, w]` (upstream's
 * `cat([emb_t, emb_h, emb_w] * 2, dim=-1)`), and the token index is `h·W' + w`. **cos / sin are
 * only copied from the base tables** (no trigonometric function is ever called — the
 * MUST in `rope-base.ts`).
 */
export const ropeTables = (
  base: RopeBase,
  latentShape: readonly number[],
  geometry: DitPatchGeometry,
): { readonly cos: Float32Array<ArrayBuffer>; readonly sin: Float32Array<ArrayBuffer> } => {
  const [height, width] = latentSides(latentShape, geometry, "rope table");
  const rows = height / geometry.patchHeight;
  const cols = width / geometry.patchWidth;
  if (rows > base.rows || cols > base.rows) {
    throw new Error(
      `The rope base table's ${base.rows} rows cannot build H'=${rows} / W'=${cols}` +
        " (exceeds the ceiling of the model's position table)",
    );
  }
  const rowWidth = ropeWidth(base);
  const cos = new Float32Array(rows * cols * rowWidth);
  const sin = new Float32Array(rows * cols * rowWidth);
  const [widthT, widthH, widthW] = base.widths;
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      // Read position per axis (t is always position 0 for an image).
      const offsets = [0, row * widthH, col * widthW];
      const spans = [widthT, widthH, widthW];
      const token = (row * cols + col) * rowWidth;
      for (let half = 0; half < 2; half += 1) {
        let at = token + half * (rowWidth / 2);
        for (let axis = 0; axis < ROPE_AXIS_COUNT; axis += 1) {
          const from = offsets[axis];
          const span = spans[axis];
          cos.set(base.cos[axis].subarray(from, from + span), at);
          sin.set(base.sin[axis].subarray(from, from + span), at);
          at += span;
        }
      }
    }
  }
  return { cos, sin };
};
