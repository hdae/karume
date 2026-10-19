/**
 * Spelling of the generation resolution and its **accepted set** (any resolution, non-square).
 *
 * ## Why the accepted set is held at the entry
 *
 * An unacceptable shape passed through fails deep down after the assets are loaded (rope base table
 * rows exceeded / tile layout impossible), and the diagnosis does not say "which knob is wrong".
 * The upper bound of `S` (`torch.export.Dim("S")` at export) does not survive in the IR, so it is
 * not checked deep down either. Failing **with the condition named** at the entry is what fail
 * loudly means here.
 *
 * MUST: this is **the entry's spelling check**, not the source of truth for the shape. Whether the
 * latent divides by the patch (`latentSides` in `dit-tokens.ts`), fits the rope base table's rows
 * (`ropeTables`, same file) and allows a tile layout (`planTileAxis` in `tiling.ts`) fail
 * separately on the actual values read from the assets. The constants here are **copies** of those;
 * when the assets change, that side is what follows.
 *
 * NOTE: the **number** of tiles is not checked at the entry. With the rounded equal-spacing layout
 * (ADR 0033 addendum 2026-09-02) it is `ceil((latent − 64) / 56) + 1`, only 5 even at this accepted
 * set's upper latent 256, so no side is rejected for it. Meeting the 512px lower bound always
 * allows a tile layout.
 *
 * MUST: do not write the accepted set into the manifest (ADR 0038 §2 "keep one derivation source")
 * — architecture constants belong to the pipeline implementation, the upper bound of `S` to the
 * `torch.export.Dim` declaration at export (not kept in the IR), and the existence of assets to the
 * weights / assets tables. **The constants of this module are the source of truth for the accepted
 * set**.
 */

import { ModelInputError } from "../errors.ts";
import { ANIMA_SPATIAL_COMPRESSION } from "./dit-tokens.ts";

/** Image size (pixels). Axis order is the same as the spelling, `width × height`. */
export type ImageSize = {
  readonly width: number;
  readonly height: number;
};

/**
 * Side of the DiT patch (Anima's `patch_size = (1,2,2)`).
 * MUST: the actual patch geometry is **derived from the graph's input/output widths**
 * (`ditPatchGeometry`). This is a copy only for stating the step at the entry; on a mismatch
 * `latentSides` fails on the asset-side value.
 */
const DIT_PATCH_SIDE = 2;

/**
 * Step of each side (= spatial compression 8 × patch 2). The smallest unit at which each latent
 * side divides by the patch.
 */
export const RESOLUTION_GRANULARITY = ANIMA_SPATIAL_COMPRESSION * DIT_PATCH_SIDE;

/**
 * Lower bound of each side. A shape that cannot hold even one **latent 64 of the VAE tile decoder**
 * (= 512px) cannot be laid out (`planTileAxis` fails with "the latent length is smaller than the
 * tile width").
 */
export const MIN_RESOLUTION_SIDE = 512;

/**
 * Upper bound of each latent side. The **row count of the per-axis rope base tables** (128 rows in
 * Anima = 128 tokens = latent 256 = 2048px) is the ceiling of the model's position table itself;
 * beyond it, upstream cannot build it either.
 *
 * NOTE (above 1920px are extrapolated positions outside the model's declaration — ruling of
 * 2026-09-04): the row count of the base tables is set by **the upper bound of the t axis**, and
 * the declaration of the h / w axes (`rope.max_size` = 120 tokens = 1920px) is lower. So a request
 * above 1920px up to 2048px can be read as table rows, but uses **positions outside** the h / w
 * range the model declared at training. The values are defined by the same formula as upstream, so
 * bit equality with the static shape holds; any difference is **quality only**, from positions
 * outside the training distribution (not measured). The accepted set is not narrowed for this
 * reason (the distribution's freedom is not cut without a measurement).
 */
const MAX_LATENT_SIDE = 256;

/** Upper bound of each side (pixels). The base tables' ceiling mapped to resolution. */
export const MAX_RESOLUTION_SIDE = MAX_LATENT_SIDE * ANIMA_SPATIAL_COMPRESSION;

/**
 * Upper bound of the token length `S` (`torch.export.Dim("S", max=16384)` when the S-form graph was
 * exported — ADR 0034; not kept in the IR).
 */
export const MAX_DIT_TOKENS = 16384;

/** Whether the size is square. */
const isSquare = (size: ImageSize): boolean => size.width === size.height;

/**
 * Spelling → size. Accepts `1344x768` and the square shorthand `512` (= `512x512`).
 *
 * MUST: fail an unacceptable shape **with the condition named**. "The resolution is wrong" alone
 * does not say which of step, lower bound, ceiling, tile count or the `S` bound was hit, and the
 * knobs end up being tried one by one.
 */
export const parseResolution = (raw: string): ImageSize => {
  const parts = raw.toLowerCase().split("x");
  const spelled = parts.length === 1 ? [parts[0], parts[0]] : parts;
  if (spelled.length !== 2 || spelled.some((part) => !/^\d+$/.test(part))) {
    throw new ModelInputError(
      `Resolution ${JSON.stringify(raw)} is neither WxH nor the square shorthand (e.g. 1344x768 / 512)`,
    );
  }
  const size = { width: Number(spelled[0]), height: Number(spelled[1]) };
  assertAcceptableResolution(size);
  return size;
};

/**
 * `1344x768` / the shorthand (`512`) for a square. The inverse of {@link parseResolution}; the
 * round trip is the identity.
 */
export const formatResolution = (size: ImageSize): string =>
  isSquare(size) ? `${size.width}` : `${size.width}x${size.height}`;

/** DiT token length `S` (= the product of the grid obtained by dividing each side by the step). */
const tokensOf = (size: ImageSize): number =>
  (size.width / RESOLUTION_GRANULARITY) * (size.height / RESOLUTION_GRANULARITY);

/**
 * Accepted-set check (4 conditions). Checked **in the same order** whatever the caller's spelling —
 * if the diagnosis of a shape that breaks several conditions changed per run, one could not tell
 * whether the fix led to another condition.
 *
 * NOTE: resolutions whose sides are not multiples of 32 px are accepted too, but for shapes where
 * `S` is not a multiple of 4 the default preset's (a8 + s16) self-attention stores ③PV and S in f32
 * (docs/limitations.md, the section on fused attention a8 / s16). The accepted set is not narrowed
 * for this reason.
 */
export const assertAcceptableResolution = (size: ImageSize): void => {
  const label = `${size.width}×${size.height}`;
  for (const [side, axis] of [
    [size.width, "width"],
    [size.height, "height"],
  ] as const) {
    if (!Number.isInteger(side)) {
      throw new ModelInputError(`Resolution ${label}: ${axis} ${side} is not an integer`);
    }
    if (side % RESOLUTION_GRANULARITY !== 0) {
      throw new ModelInputError(
        `Resolution ${label}: ${axis} ${side} is not a multiple of ${RESOLUTION_GRANULARITY}` +
          ` (the latent is not divisible by patch ${DIT_PATCH_SIDE})`,
      );
    }
    if (side < MIN_RESOLUTION_SIDE) {
      throw new ModelInputError(
        `Resolution ${label}: ${axis} ${side} is below the lower bound ${MIN_RESOLUTION_SIDE}` +
          ` (the VAE tile decoder's latent ${
            MIN_RESOLUTION_SIDE / ANIMA_SPATIAL_COMPRESSION
          } does not fit)`,
      );
    }
    if (side > MAX_RESOLUTION_SIDE) {
      throw new ModelInputError(
        `Resolution ${label}: ${axis} ${side} exceeds the upper bound ${MAX_RESOLUTION_SIDE}` +
          ` (the ceiling of the rope base table = latent ${MAX_LATENT_SIDE})`,
      );
    }
  }
  const tokens = tokensOf(size);
  // NOTE: with the current constants the side bound (2048 = grid 128) gives S ≤ 128² =
  // MAX_DIT_TOKENS, so no shape reaches this branch. It stays as insurance for when the constants
  // (side bound / step / S bound) move — the check on the constant relations
  // (`anima_resolution_test.ts`) pins that it is unreachable.
  if (tokens > MAX_DIT_TOKENS) {
    throw new Error(
      `Resolution ${label}: token length S=${tokens} exceeds the upper bound ${MAX_DIT_TOKENS}` +
        ' (the upper bound of Dim("S") when the S-form graph was exported — set by the area, separately from the per-side conditions)',
    );
  }
};
