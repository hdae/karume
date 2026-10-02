/**
 * Wan2.1 の DiT の **S 形グラフ**（トークン長 1 シンボル）のホスト側 — patchify / unpatchify
 * （ADR 0118 決定 3）。
 *
 * グラフの入口は patchify の後・出口は unpatchify の前なので、潜在 `[C,F,H,W]` とトークン列
 * `[1,S,C·pt·ph·pw]` の行き来はここが持つ。グラフの中に F / H / W は 1 つも現れない。
 *
 * ## 2 つの並びは別物
 *
 * - 入口（patchify）の最終次元は **`(c, pt, ph, pw)`** — 上流の patch 埋め込み `Conv3d(k = s = patch)`
 *   の重み `[Cout, C, pt, ph, pw]` を平坦化した Linear の入力の並び。
 * - 出口（unpatchify）の最終次元は **`(pt, ph, pw, c)`** — 上流 `proj_out` の出力を
 *   `reshape(…, pt, ph, pw, -1)` で読む並び（`transformer_wan.py` の出口）。
 *
 * どちらもトークン添字は `(f·H' + h)·W' + w`。MUST: unpatchify を patchify の逆順に「直さない」
 * （shape は合ったまま値だけが散る）。
 *
 * 潜在はバッチ軸を持たない `[C,F,H,W]`（B = 1 — CFG は 2 回の forward。ADR 0118 決定 3）。
 */

/** patch の刻み（`(pt, ph, pw)`）と潜在のチャネル数 `C`。 */
export type WanPatchGeometry = {
  readonly channels: number;
  readonly patchFrames: number;
  readonly patchHeight: number;
  readonly patchWidth: number;
};

/** 潜在のトークン格子（`F' = F/pt`・`H' = H/ph`・`W' = W/pw`）とトークン長 `S = F'·H'·W'`。 */
export type WanTokenGrid = {
  readonly frames: number;
  readonly rows: number;
  readonly cols: number;
  readonly count: number;
};

/** 1 トークンの幅（`C·pt·ph·pw` — 入口と出口で同じ幅・並びだけが違う）。 */
export const wanTokenWidth = (geometry: WanPatchGeometry): number =>
  geometry.channels * geometry.patchFrames * geometry.patchHeight * geometry.patchWidth;

/** `[C,F,H,W]` の潜在形を検査して `(F, H, W)` を返す。 */
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

/** 潜在 `[C,F,H,W]` のトークン格子。 */
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
 * 潜在 `[C,F,H,W]` → `tokens [1,S,C·pt·ph·pw]`（最終次元は `(c, pt, ph, pw)`）。
 *
 * 添字: `tokens[token·width + c·V + (it·ph + ih)·pw + iw] = latents[c][f·pt+it][h·ph+ih][w·pw+iw]`
 * （`V = pt·ph·pw`・`token = (f·H' + h)·W' + w`）。
 */
export const patchifyLatents = (
  latents: Float32Array,
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
              const line = ((channel * frames + frame * patchFrames + innerT) * height +
                    row * patchHeight + innerH) * width + col * patchWidth;
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
 * `tokens [1,S,pt·ph·pw·C]` → 潜在 `[C,F,H,W]`（最終次元は `(pt, ph, pw, c)`）。
 *
 * 添字: `latents[c][f·pt+it][h·ph+ih][w·pw+iw] = tokens[token·width + ((it·ph + ih)·pw + iw)·C + c]`。
 * MUST: patchify の逆順ではない（モジュール doc）。
 */
export const unpatchifyTokens = (
  tokens: Float32Array,
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
              const inner = token +
                ((innerT * patchHeight + innerH) * patchWidth + innerW) * channels;
              const at = ((frame * patchFrames + innerT) * height + row * patchHeight + innerH) *
                  width + col * patchWidth + innerW;
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
