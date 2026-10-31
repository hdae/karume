/**
 * The acceptance check of a MOSS-Image DiT request — image size and text embeddings (ADR 0138
 * decision 9, stage 2). Everything here is a `ModelInputError` (fixing the input makes it pass —
 * ADR 0107) and runs before any GPU work.
 *
 * - **Size**: width and height in pixels, positive multiples of 16 (upstream's own check — one latent
 *   cell is 16 px), and the longest side at most {@link MOSS_MAX_SIDE_PX} (65,536 px = 4,096 cells)
 *   — an explicit acceptance limit: the RoPE base tables' `h` / `w` axes hold one row per position of
 *   a side of up to 4,096 cells (2·4,096 − 1 rows; `./dit-rope.ts`). Upstream has no ceiling, but
 *   with the area cap of the VAE binding (ADR 0138 decision 8) a longer side would leave a short
 *   side under about 43 px. MUST: a longer side is rejected with an error that names the limit.
 * - **Area** ({@link assertMossImageArea}, ADR 0138 decision 8 — per device): the untiled VAE's
 *   largest intermediate must fit one storage binding, so the area is capped by the device's
 *   binding limit over the decoder's bytes per pixel (`./vae-contract.ts` derives both from the
 *   graphs — about 2.80M pixels at 2 GiB). MUST: a larger area is rejected with an error that names
 *   the limit. Whether the DiT pair and the VAE fit the device's VRAM is not an acceptance limit:
 *   each stage checks it before it opens (`fitsHeadroom`).
 * - **Text**: the internal entry takes encoder outputs only — a `Float32Array` `[T, 4096]` with
 *   1 ≤ T ≤ {@link MOSS_MAX_TEXT_TOKENS} (upstream requires at least one token and truncates at
 *   `max_sequence_length` 4,096), every value finite (a non-finite embedding would surface as
 *   non-finite latents after the whole DiT stage, blaming the wrong place — the Wan rule). The
 *   array type is checked at run time too: a caller outside the types (JavaScript, a cast) would
 *   otherwise pass this check and fail only inside the runtime, after the text stage was sized and
 *   opened, with an error that is not a `ModelInputError`.
 *
 * The family admission checks that the series' assets and graphs cover exactly these limits
 * (`./dit-contract.ts`), so an accepted request always has its base-table rows.
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state).
 */

import { ModelInputError } from "../errors.ts";
import type { MossCells } from "./dit-rope.ts";
import type { MossAreaLimit } from "./vae-contract.ts";

/** Pixels per latent cell (the VAE's 8× and the DiT's 2×2 patch — upstream's `height // 16`). */
export const MOSS_PIXELS_PER_CELL = 16;

/** The longest accepted side in latent cells (module doc). */
export const MOSS_MAX_SIDE_CELLS = 4096;

/** The longest accepted side in pixels (65,536). */
export const MOSS_MAX_SIDE_PX = MOSS_MAX_SIDE_CELLS * MOSS_PIXELS_PER_CELL;

/** The longest accepted text (upstream's `max_sequence_length`). */
export const MOSS_MAX_TEXT_TOKENS = 4096;

/** The width of one encoder output row (the DiT's `text_dim`). */
export const MOSS_TEXT_WIDTH = 4096;

const OWNER = "MOSS-Image";

/** One text-encoder output `[T, 4096]` (row-major). */
export type MossTextEmbeddings = {
  readonly data: Float32Array<ArrayBuffer>;
  readonly shape: readonly number[];
};

/**
 * Checks an image size (module doc "Size") and returns its latent grid.
 *
 * NOTE: `export` is for the family entry and its host tests (not on a public surface — the family
 * is not exported, ADR 0138 decision 12).
 */
export const assertMossImageSize = (width: number, height: number): MossCells => {
  for (const [name, side] of [
    ["width", width],
    ["height", height],
  ] as const) {
    if (!Number.isInteger(side) || side <= 0 || side % MOSS_PIXELS_PER_CELL !== 0) {
      throw new ModelInputError(
        `${OWNER}: ${name} ${side} px is not a positive multiple of ${MOSS_PIXELS_PER_CELL}`,
      );
    }
  }
  const longest = Math.max(width, height);
  if (longest > MOSS_MAX_SIDE_PX) {
    throw new ModelInputError(
      `${OWNER}: the longest side ${longest} px exceeds the accepted ${MOSS_MAX_SIDE_PX} px` +
        ` (${MOSS_MAX_SIDE_CELLS} latent cells — the rows of the RoPE base tables' h / w axes)`,
    );
  }
  return { rows: height / MOSS_PIXELS_PER_CELL, cols: width / MOSS_PIXELS_PER_CELL };
};

/**
 * Checks the image area against a device's area limit (module doc "Area"). Run after
 * {@link assertMossImageSize} (the sides are positive multiples of 16).
 */
export const assertMossImageArea = (width: number, height: number, limit: MossAreaLimit): void => {
  const area = width * height;
  if (area > limit.maxPixels) {
    throw new ModelInputError(
      `${OWNER}: ${width}×${height} px (${area} px) exceeds this device's area limit of` +
        ` ${limit.maxPixels} px — the untiled VAE's largest tensor (${limit.bytesPerPixel} B per pixel)` +
        ` must fit one storage binding of ${limit.bindingBytes} B`,
    );
  }
};

/**
 * Checks one text embedding (module doc "Text") and returns its token count T. `label` names it in
 * the messages (`"positive"` / `"negative"`).
 */
export const assertMossTextEmbeddings = (embeddings: MossTextEmbeddings, label: string): number => {
  const { data, shape } = embeddings;
  const where = `${OWNER}: the ${label} text embeddings`;
  // MUST: the run-time check of the declared type (module doc "Text").
  if (!(data instanceof Float32Array)) {
    throw new ModelInputError(`${where} are not a Float32Array (f32 values are required)`);
  }
  if (shape.length !== 2 || shape[1] !== MOSS_TEXT_WIDTH) {
    throw new ModelInputError(
      `${where} have shape [${shape.join(", ")}] (expected [T, ${MOSS_TEXT_WIDTH}])`,
    );
  }
  const [tokens] = shape;
  if (!Number.isInteger(tokens)) {
    throw new ModelInputError(`${where} have a token count ${tokens} that is not an integer`);
  }
  if (tokens < 1) {
    throw new ModelInputError(`${where} have ${tokens} tokens (at least 1 is required)`);
  }
  if (tokens > MOSS_MAX_TEXT_TOKENS) {
    throw new ModelInputError(
      `${where} have ${tokens} tokens, over the accepted ${MOSS_MAX_TEXT_TOKENS}` +
        " (upstream's max_sequence_length)",
    );
  }
  if (data.length !== tokens * MOSS_TEXT_WIDTH) {
    throw new ModelInputError(
      `${where} hold ${data.length} values for shape [${tokens}, ${MOSS_TEXT_WIDTH}]`,
    );
  }
  const broken = data.findIndex((value) => !Number.isFinite(value));
  if (broken !== -1) {
    throw new ModelInputError(
      `${where}: row ${Math.floor(broken / MOSS_TEXT_WIDTH)}, column ${broken % MOSS_TEXT_WIDTH}` +
        ` is non-finite (${data[broken]})`,
    );
  }
  return tokens;
};
