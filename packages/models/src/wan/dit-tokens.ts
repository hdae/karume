/**
 * The host side of Wan2.1's DiT **S-shaped graph** (one symbol for the token length) — patchify /
 * unpatchify (ADR 0118 decision 3).
 *
 * The graph's entry is after patchify and its exit before unpatchify, so the conversion between
 * latents `[C,F,H,W]` and the token sequence `[1,S,C·pt·ph·pw]` lives here. F / H / W never appear
 * inside the graph.
 *
 * ## The two layouts are different
 *
 * - The last dimension at the entry (patchify) is **`(c, pt, ph, pw)`** — the input order of the
 *   Linear that flattens the weight `[Cout, C, pt, ph, pw]` of upstream's patch embedding
 *   `Conv3d(k = s = patch)`.
 * - The last dimension at the exit (unpatchify) is **`(pt, ph, pw, c)`** — the order in which
 *   upstream reads the output of `proj_out` with `reshape(…, pt, ph, pw, -1)` (the exit of
 *   `transformer_wan.py`).
 *
 * Both use the token index `(f·H' + h)·W' + w`. MUST: do not "fix" unpatchify into the reverse of
 * patchify (the shapes still match and only the values scatter).
 *
 * The latents are `[C,F,H,W]` without a batch axis (B = 1 — CFG is two forwards. ADR 0118
 * decision 3).
 */

/** The patch strides (`(pt, ph, pw)`) and the latent channel count `C`. */
export type WanPatchGeometry = {
  readonly channels: number;
  readonly patchFrames: number;
  readonly patchHeight: number;
  readonly patchWidth: number;
};

/** The latent token grid (`F' = F/pt`, `H' = H/ph`, `W' = W/pw`) and the token length `S = F'·H'·W'`. */
export type WanTokenGrid = {
  readonly frames: number;
  readonly rows: number;
  readonly cols: number;
  readonly count: number;
};

/** The width of one token (`C·pt·ph·pw` — the same width at entry and exit; only the order differs). */
export const wanTokenWidth = (geometry: WanPatchGeometry): number =>
  geometry.channels * geometry.patchFrames * geometry.patchHeight * geometry.patchWidth;

/** Checks a `[C,F,H,W]` latent shape and returns `(F, H, W)`. */
const latentExtents = (
  latentShape: readonly number[],
  geometry: WanPatchGeometry,
  where: string,
): readonly [number, number, number] => {
  if (latentShape.length !== 4) {
    throw new Error(`${where}: 潜在は [C,F,H,W] 前提（[${latentShape}]）`);
  }
  const [channels, frames, height, width] = latentShape;
  if (channels !== geometry.channels) {
    throw new Error(
      `${where}: 潜在のチャネル数 ${channels} が patch 幾何の ${geometry.channels} と違う`,
    );
  }
  if (
    !(frames > 0 && height > 0 && width > 0) ||
    frames % geometry.patchFrames !== 0 ||
    height % geometry.patchHeight !== 0 ||
    width % geometry.patchWidth !== 0
  ) {
    throw new Error(
      `${where}: 潜在 ${frames}×${height}×${width} が patch ` +
        `${geometry.patchFrames}×${geometry.patchHeight}×${geometry.patchWidth} で割り切れない`,
    );
  }
  return [frames, height, width];
};

/** The token grid of latents `[C,F,H,W]`. */
export const wanTokenGrid = (
  latentShape: readonly number[],
  geometry: WanPatchGeometry,
): WanTokenGrid => {
  const [frames, height, width] = latentExtents(latentShape, geometry, "トークン格子");
  const grid = {
    frames: frames / geometry.patchFrames,
    rows: height / geometry.patchHeight,
    cols: width / geometry.patchWidth,
  };
  return { ...grid, count: grid.frames * grid.rows * grid.cols };
};

/**
 * Latents `[C,F,H,W]` → `tokens [1,S,C·pt·ph·pw]` (last dimension `(c, pt, ph, pw)`).
 *
 * Index: `tokens[token·width + c·V + (it·ph + ih)·pw + iw] = latents[c][f·pt+it][h·ph+ih][w·pw+iw]`
 * (`V = pt·ph·pw`, `token = (f·H' + h)·W' + w`).
 */
export const patchifyLatents = (
  latents: Float32Array<ArrayBuffer>,
  latentShape: readonly number[],
  geometry: WanPatchGeometry,
): Float32Array<ArrayBuffer> => {
  const [frames, height, width] = latentExtents(latentShape, geometry, "patchify");
  const { channels, patchFrames, patchHeight, patchWidth } = geometry;
  if (latents.length !== channels * frames * height * width) {
    throw new Error(`patchify: 要素数 ${latents.length} が [${latentShape}] と違う`);
  }
  const grid = wanTokenGrid(latentShape, geometry);
  const volume = patchFrames * patchHeight * patchWidth;
  const tokenWidth = channels * volume;
  const tokens = new Float32Array(grid.count * tokenWidth);
  for (let channel = 0; channel < channels; channel += 1) {
    for (let frame = 0; frame < grid.frames; frame += 1) {
      for (let row = 0; row < grid.rows; row += 1) {
        for (let col = 0; col < grid.cols; col += 1) {
          const token = (frame * grid.rows + row) * grid.cols + col;
          const base = token * tokenWidth + channel * volume;
          for (let innerT = 0; innerT < patchFrames; innerT += 1) {
            for (let innerH = 0; innerH < patchHeight; innerH += 1) {
              const line =
                ((channel * frames + frame * patchFrames + innerT) * height +
                  row * patchHeight +
                  innerH) *
                  width +
                col * patchWidth;
              const at = base + (innerT * patchHeight + innerH) * patchWidth;
              for (let innerW = 0; innerW < patchWidth; innerW += 1) {
                tokens[at + innerW] = latents[line + innerW];
              }
            }
          }
        }
      }
    }
  }
  return tokens;
};

/**
 * `tokens [1,S,pt·ph·pw·C]` → latents `[C,F,H,W]` (last dimension `(pt, ph, pw, c)`).
 *
 * Index: `latents[c][f·pt+it][h·ph+ih][w·pw+iw] = tokens[token·width + ((it·ph + ih)·pw + iw)·C + c]`.
 * MUST: this is not the reverse of patchify (module doc).
 */
export const unpatchifyTokens = (
  tokens: Float32Array<ArrayBuffer>,
  latentShape: readonly number[],
  geometry: WanPatchGeometry,
): Float32Array<ArrayBuffer> => {
  const [frames, height, width] = latentExtents(latentShape, geometry, "unpatchify");
  const { channels, patchFrames, patchHeight, patchWidth } = geometry;
  const grid = wanTokenGrid(latentShape, geometry);
  const tokenWidth = wanTokenWidth(geometry);
  if (tokens.length !== grid.count * tokenWidth) {
    throw new Error(
      `unpatchify: 要素数 ${tokens.length} が [1,${grid.count},${tokenWidth}] と違う`,
    );
  }
  const latents = new Float32Array(channels * frames * height * width);
  for (let frame = 0; frame < grid.frames; frame += 1) {
    for (let row = 0; row < grid.rows; row += 1) {
      for (let col = 0; col < grid.cols; col += 1) {
        const token = ((frame * grid.rows + row) * grid.cols + col) * tokenWidth;
        for (let innerT = 0; innerT < patchFrames; innerT += 1) {
          for (let innerH = 0; innerH < patchHeight; innerH += 1) {
            for (let innerW = 0; innerW < patchWidth; innerW += 1) {
              const inner =
                token + ((innerT * patchHeight + innerH) * patchWidth + innerW) * channels;
              const at =
                ((frame * patchFrames + innerT) * height + row * patchHeight + innerH) * width +
                col * patchWidth +
                innerW;
              for (let channel = 0; channel < channels; channel += 1) {
                latents[channel * frames * height * width + at] = tokens[inner + channel];
              }
            }
          }
        }
      }
    }
  }
  return latents;
};
