/**
 * DiT の **S 形グラフ**（トークン長 1 シンボル）のホスト側 — ADR 0034。
 *
 * ## なぜホストに出るのか
 *
 * 静的形の DiT は解像度が焼き込まれた 3 本の定数を抱えている（padding channel の
 * `[1,1,H,W]` と rope の cos / sin 表 `[1,1,S,128]`）。IR の次元言語は「1 次元 1 シンボルの
 * 一次式」なので `H·W` は書けないが、**グラフの入口を patchify の後ろへずらす**とトークン長
 * `S` の 1 シンボルだけで全ノードが書ける。その代わり patchify / unpatchify / rope 表の構築が
 * ここへ降りてくる。ランタイムは 1 行も変わらない（plan は run ごと・パイプラインキーに
 * shape が入らない）。
 *
 * ADR 0038 §4 により**配布される transformer は S 形のみ**なので、この経路が唯一の経路。
 */

import { ROPE_AXIS_COUNT, type RopeBase, ropeWidth } from "./rope-base.ts";

/**
 * VAE の空間圧縮率（latent 1 → sample 8px）。
 *
 * **なぜ定数として置くか**: S 形の DiT は解像度を 1 つも持たないので、「解像度 → latent の
 * 一辺」を資産から引けない。取り違えは生成の末尾の「出た画像の寸法 == 要求解像度」検査が
 * 実行のたびに閉じる — VAE decoder の入出力形が縮尺の正本で、ここはその写し。
 */
export const ANIMA_SPATIAL_COMPRESSION = 8;

/** patchify の刻みと latent のチャネル数（グラフの入出力幅から割り出す）。 */
export type DitPatchGeometry = {
  /** latent のチャネル数 `C`（padding channel を**含まない**）。 */
  readonly channels: number;
  /** patch の高さ `ph`。 */
  readonly patchHeight: number;
  /** patch の幅 `pw`。 */
  readonly patchWidth: number;
};

/**
 * グラフの `tokens` 入力幅と出力幅から patch 幾何を割り出す。
 *
 * 入口は `(C+1)·pt·ph·pw`（padding channel が 1 本増える）、出口は `pt·ph·pw·C` なので、
 * 差がそのまま `P = pt·ph·pw`、商が `C` になる。**呼び出し側に 68 / 64 / 16 を literal で
 * 置かない**ための割り出しで、タイル幾何が縮尺を資産から引くのと同じ規律。
 *
 * MUST: `pt = 1`（画像）・`ph = pw`（正方 patch）を仮定する。Anima の
 * `patch_size = (1,2,2)` はこれを満たすが、満たさない構成は**割り出せない**ので黙って
 * 続けずに落とす（`P` が平方数でなければ露見する）。
 */
export const ditPatchGeometry = (tokenWidth: number, outputWidth: number): DitPatchGeometry => {
  const patchVolume = tokenWidth - outputWidth;
  if (patchVolume <= 0 || outputWidth <= 0 || outputWidth % patchVolume !== 0) {
    throw new Error(
      `DiT の patch 幾何を割り出せない（tokens 幅 ${tokenWidth} / 出力幅 ${outputWidth}）`,
    );
  }
  const side = Math.round(Math.sqrt(patchVolume));
  if (side * side !== patchVolume) {
    throw new Error(
      `patch 体積 ${patchVolume} が平方数でない（pt=1・正方 patch の前提が崩れている）`,
    );
  }
  return { channels: outputWidth / patchVolume, patchHeight: side, patchWidth: side };
};

/** `[1,C,H,W]` の latent 形を検査して `(H, W)` を返す。 */
const latentSides = (
  latentShape: readonly number[],
  geometry: DitPatchGeometry,
  where: string,
): readonly [number, number] => {
  if (latentShape.length !== 4 || latentShape[0] !== 1) {
    throw new Error(`${where}: latent は [1,C,H,W] 前提（[${latentShape}]）`);
  }
  if (latentShape[1] !== geometry.channels) {
    throw new Error(
      `${where}: latent のチャネル数 ${latentShape[1]} がグラフの ${geometry.channels} と違う`,
    );
  }
  const [, , height, width] = latentShape;
  if (height % geometry.patchHeight !== 0 || width % geometry.patchWidth !== 0) {
    throw new Error(
      `${where}: latent ${height}×${width} が patch ${geometry.patchHeight}×${geometry.patchWidth} で割り切れない`,
    );
  }
  return [height, width];
};

/** latent `[1,C,H,W]` に対応するトークン長 `S`。 */
export const tokenCount = (
  latentShape: readonly number[],
  geometry: DitPatchGeometry,
): number => {
  const [height, width] = latentSides(latentShape, geometry, "トークン長");
  return (height / geometry.patchHeight) * (width / geometry.patchWidth);
};

/**
 * `latents [1,C,H,W]` → `tokens [1,S,(C+1)·ph·pw]`（恒常ゼロの padding channel 込み）。
 *
 * 最終次元の並びは **`(c, ph, pw)`**（= `c·ph·pw + ih·pw + iw`）、トークン添字は
 * `h·W' + w`。正本は上流 `CosmosPatchEmbed` の proj 前段で、こちらはその添字表現。
 *
 * MUST: padding channel は**恒常ゼロ**（パイプラインが `latents.new_zeros` を渡すため）。
 * ここを latent の写しにすると shape も長さも合ったまま値だけが変わる。
 */
export const patchifyLatents = (
  latents: Float32Array,
  latentShape: readonly number[],
  geometry: DitPatchGeometry,
): Float32Array<ArrayBuffer> => {
  const [height, width] = latentSides(latentShape, geometry, "patchify");
  const { channels, patchHeight, patchWidth } = geometry;
  if (latents.length !== channels * height * width) {
    throw new Error(`patchify: 要素数 ${latents.length} が [${latentShape}] と違う`);
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
  // padding channel（添字 `channels`）は Float32Array の初期値 0 のまま = 恒常ゼロ。
  return tokens;
};

/**
 * `tokens [1,S,ph·pw·C]` → `latents [1,C,H,W]`。
 *
 * MUST: **patchify の逆順ではない**。最終次元の並びは `(ih, iw, C)`（= `(ih·pw + iw)·C + c`）
 * で、patchify 側の `(c, ih, iw)` と別物 — 上流のコメントどおりで、「逆順に直す」と
 * shape は合ったまま画素が散る。
 */
export const unpatchifyTokens = (
  tokens: Float32Array,
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
      `unpatchify: 要素数 ${tokens.length} が [1,${rows * cols},${tokenWidth}] と違う`,
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
 * `[1,1,S,head_dim]` の cos / sin 表を組む（F'=1 の画像専用）。
 *
 * 1 行は `[t, h, w, t, h, w]` のブロック連結（上流の
 * `cat([emb_t, emb_h, emb_w] * 2, dim=-1)`）で、トークン添字は `h·W' + w`。
 * **cos / sin は素表からの写しだけ**（三角関数を 1 度も呼ばない — `rope-base.ts` の MUST）。
 */
export const ropeTables = (
  base: RopeBase,
  latentShape: readonly number[],
  geometry: DitPatchGeometry,
): { readonly cos: Float32Array<ArrayBuffer>; readonly sin: Float32Array<ArrayBuffer> } => {
  const [height, width] = latentSides(latentShape, geometry, "rope 表");
  const rows = height / geometry.patchHeight;
  const cols = width / geometry.patchWidth;
  if (rows > base.rows || cols > base.rows) {
    throw new Error(
      `rope 素表の行数 ${base.rows} では H'=${rows} / W'=${cols} を組めない` +
        "（モデル側の位置表の天井を超えている）",
    );
  }
  const rowWidth = ropeWidth(base);
  const cos = new Float32Array(rows * cols * rowWidth);
  const sin = new Float32Array(rows * cols * rowWidth);
  const [widthT, widthH, widthW] = base.widths;
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      // 軸ごとの読み出し位置（t は画像なので常に位置 0）。
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
