/**
 * BiRefNet の後処理の A/B（ADR 0125 — 裁定 4）の**純粋な計算**: 輪郭の帯・数値の列（K 列 / 総合列）・
 * 差の大きい窓・盲検の左右の割り付け・目視用の合成。GPU にも資産にも触らない（`analysis_test.ts` が
 * 直接縛る）。入出力と GPU は `main.ts`。
 *
 * ## 数値の列（記録だけ — 判定には使わない・ADR 0125）
 *
 * - **輪郭の帯**（{@link contourBand}）: どちらかの手順で `0 < α < 255` になる画素を、チェビシェフ距離で
 *   {@link BAND_RADIUS} 画素だけ膨張した範囲。内側（255）と外側（0）の大部分は 2 手順で同じなので、
 *   画像全体で平均すると差が薄まる — 測るのは輪郭だけにする。
 * - **K 列**（{@link kernelDifference}）: 帯の中での 2 手順の α の差（8bit の段で）。どちらの手順も
 *   四捨五入なので、差はカーネル（bicubic 対 bilinear）と量子化の位置（resize の前か後か）から出る。
 * - **総合列**（{@link errorAgainst}）: 擬似正解に対する帯の中の誤差。擬似正解は**目標が 2048² の
 *   画像に限り** 2048² モデルの σ（resize を通さない）。SAD = Σ|α − gt|、MSE = 平均 (α − gt)²、
 *   Gradient error = Σ(|∇α| − |∇gt|)²（α ∈ [0, 1]・勾配は σ = 1.4 のガウス微分 — 画像マッティングの
 *   評価の慣用の定義）。2048² モデルは輪郭が 1024² モデルより鋭く、鋭いカーネルに有利に振れうる
 *   （推測）ので、向きを見る補助に留める。
 */

/** 輪郭の帯の膨張の半径（画素・チェビシェフ距離 — ADR 0125 で固定）。 */
export const BAND_RADIUS = 2;

/** 8bit の α の最大値。 */
const ALPHA_MAX = 255;

/** 1 枚の 8bit の平面（行優先）。 */
export type Plane = {
  readonly data: Uint8Array<ArrayBufferLike>;
  readonly width: number;
  readonly height: number;
};

const assertSameShape = (left: Plane, right: Plane, where: string): void => {
  if (
    left.width !== right.width || left.height !== right.height ||
    left.data.length !== left.width * left.height || right.data.length !== left.data.length
  ) {
    throw new Error(
      `${where}: 寸法が揃わない（${left.width}×${left.height} / ${right.width}×${right.height}）`,
    );
  }
};

/**
 * 輪郭の帯（モジュール doc）。返すのは画素ごとの 0 / 1。膨張は横 → 縦の分離型の最大値フィルタ
 * （正方形の構造要素 = チェビシェフ距離 `radius` の膨張と同じ）。
 */
export const contourBand = (
  first: Plane,
  second: Plane,
  radius: number = BAND_RADIUS,
): Uint8Array<ArrayBuffer> => {
  assertSameShape(first, second, "輪郭の帯");
  const { width, height } = first;
  const soft = new Uint8Array(width * height);
  for (let index = 0; index < soft.length; index += 1) {
    const a = first.data[index];
    const b = second.data[index];
    soft[index] = (a > 0 && a < ALPHA_MAX) || (b > 0 && b < ALPHA_MAX) ? 1 : 0;
  }
  const horizontal = new Uint8Array(width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let hit = 0;
      for (
        let dx = Math.max(0, x - radius);
        dx <= Math.min(width - 1, x + radius) && !hit;
        dx += 1
      ) {
        hit = soft[y * width + dx];
      }
      horizontal[y * width + x] = hit;
    }
  }
  const band = new Uint8Array(width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let hit = 0;
      for (
        let dy = Math.max(0, y - radius);
        dy <= Math.min(height - 1, y + radius) && !hit;
        dy += 1
      ) {
        hit = horizontal[dy * width + x];
      }
      band[y * width + x] = hit;
    }
  }
  return band;
};

/** K 列（帯の中の 2 手順の差 — 8bit の段）。 */
export type KernelDifference = {
  /** 帯の画素数。 */
  readonly bandPixels: number;
  /** 帯の中で値が違う画素の数。 */
  readonly differing: number;
  /** 帯の中の |差| の平均（段）。 */
  readonly meanAbs: number;
  /** 帯の中の |差| の最大（段）。 */
  readonly maxAbs: number;
};

/** K 列（モジュール doc）。帯が空なら差も 0 と数える（比べる輪郭が無い）。 */
export const kernelDifference = (
  first: Plane,
  second: Plane,
  band: Uint8Array<ArrayBufferLike>,
): KernelDifference => {
  assertSameShape(first, second, "K 列");
  let bandPixels = 0;
  let differing = 0;
  let total = 0;
  let maxAbs = 0;
  for (let index = 0; index < band.length; index += 1) {
    if (band[index] === 0) continue;
    bandPixels += 1;
    const delta = Math.abs(first.data[index] - second.data[index]);
    if (delta > 0) differing += 1;
    total += delta;
    if (delta > maxAbs) maxAbs = delta;
  }
  return { bandPixels, differing, meanAbs: bandPixels === 0 ? 0 : total / bandPixels, maxAbs };
};

/** ガウス微分の σ（画像マッティングの Gradient error の慣用の値）。 */
const GRADIENT_SIGMA = 1.4;

/** ガウスの窓の裾の打ち切り（慣用の実装の `epsilon = 1e-2`）。 */
const GRADIENT_EPSILON = 1e-2;

/**
 * 勾配の大きさの地図（σ = {@link GRADIENT_SIGMA} のガウス微分 — x 方向の核は `g(y)·g'(x)` を L2 で
 * 正規化したもの・y 方向はその転置・端は複製）。慣用の実装（MATLAB の `gaussgradient`）と同じ核で、
 * 分離型で畳み込む。
 */
export const gradientMagnitude = (
  plane: Float64Array<ArrayBufferLike>,
  width: number,
  height: number,
): Float64Array<ArrayBuffer> => {
  const sigma = GRADIENT_SIGMA;
  const half = Math.ceil(
    sigma * Math.sqrt(-2 * Math.log(Math.sqrt(2 * Math.PI) * sigma * GRADIENT_EPSILON)),
  );
  const gauss = (u: number): number =>
    Math.exp(-(u * u) / (2 * sigma * sigma)) / (sigma * Math.sqrt(2 * Math.PI));
  const smooth = Array.from({ length: 2 * half + 1 }, (_, k) => gauss(k - half));
  const derive = Array.from(
    { length: 2 * half + 1 },
    (_, k) => (-(k - half) * gauss(k - half)) / (sigma * sigma),
  );
  const norm = Math.sqrt(smooth.reduce((sum, v) => sum + v * v, 0)) *
    Math.sqrt(derive.reduce((sum, v) => sum + v * v, 0));
  const clampX = (x: number): number => (x < 0 ? 0 : x >= width ? width - 1 : x);
  const clampY = (y: number): number => (y < 0 ? 0 : y >= height ? height - 1 : y);
  /** 横に `horizontal`、縦に `vertical` を当てる分離型の畳み込み（端は複製）。 */
  const separable = (horizontal: readonly number[], vertical: readonly number[]): Float64Array => {
    const middle = new Float64Array(width * height);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        let sum = 0;
        for (let k = -half; k <= half; k += 1) {
          sum += plane[y * width + clampX(x - k)] * horizontal[k + half];
        }
        middle[y * width + x] = sum;
      }
    }
    const out = new Float64Array(width * height);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        let sum = 0;
        for (let k = -half; k <= half; k += 1) {
          sum += middle[clampY(y - k) * width + x] * vertical[k + half];
        }
        out[y * width + x] = sum / norm;
      }
    }
    return out;
  };
  const gx = separable(derive, smooth);
  const gy = separable(smooth, derive);
  const magnitude = new Float64Array(width * height);
  for (let index = 0; index < magnitude.length; index += 1) {
    magnitude[index] = Math.hypot(gx[index], gy[index]);
  }
  return magnitude;
};

/** 総合列（擬似正解に対する帯の中の誤差 — α ∈ [0, 1]）。 */
export type MatteError = {
  readonly sad: number;
  readonly mse: number;
  readonly gradient: number;
};

/** 擬似正解（σ ∈ [0, 1] の f64 と、その勾配の大きさ — 手順ごとに勾配を計算し直さない）。 */
export type PseudoTruth = {
  readonly values: Float64Array<ArrayBufferLike>;
  readonly gradient: Float64Array<ArrayBufferLike>;
  readonly width: number;
  readonly height: number;
};

/** 擬似正解を組む（`values` は 2048² モデルの σ — モジュール doc の総合列）。 */
export const pseudoTruth = (
  values: Float64Array<ArrayBufferLike>,
  width: number,
  height: number,
): PseudoTruth => {
  if (values.length !== width * height) {
    throw new Error(`擬似正解の長さ ${values.length} が ${width}×${height} と違う`);
  }
  return { values, gradient: gradientMagnitude(values, width, height), width, height };
};

/** 総合列（モジュール doc）。 */
export const errorAgainst = (
  alpha: Plane,
  truth: PseudoTruth,
  band: Uint8Array<ArrayBufferLike>,
): MatteError => {
  const { width, height } = alpha;
  if (
    truth.width !== width || truth.height !== height || band.length !== width * height ||
    alpha.data.length !== width * height
  ) {
    throw new Error(
      `総合列: α ${width}×${height} と擬似正解 ${truth.width}×${truth.height}・帯 ${band.length} が揃わない`,
    );
  }
  const unit = new Float64Array(width * height);
  for (let index = 0; index < unit.length; index += 1) unit[index] = alpha.data[index] / ALPHA_MAX;
  const ours = gradientMagnitude(unit, width, height);
  const theirs = truth.gradient;
  let sad = 0;
  let squared = 0;
  let gradient = 0;
  let pixels = 0;
  for (let index = 0; index < unit.length; index += 1) {
    if (band[index] === 0) continue;
    pixels += 1;
    const delta = unit[index] - truth.values[index];
    sad += Math.abs(delta);
    squared += delta * delta;
    gradient += (ours[index] - theirs[index]) ** 2;
  }
  return { sad, mse: pixels === 0 ? 0 : squared / pixels, gradient };
};

/** 矩形（左上 `left`, `top` から `width × height`）。 */
export type Window = {
  readonly left: number;
  readonly top: number;
  readonly width: number;
  readonly height: number;
};

/**
 * 2 手順の |差| の和が最大の `size × size` の窓（同点は行優先で先・画像が窓より小さい軸は全体）。
 * 目視用の拡大の対象を選ぶ（差の無い場所を拡大しても比べられない）。
 */
export const largestDifferenceWindow = (first: Plane, second: Plane, size: number): Window => {
  assertSameShape(first, second, "差の大きい窓");
  const { width, height } = first;
  const windowWidth = Math.min(size, width);
  const windowHeight = Math.min(size, height);
  const table = new Float64Array((width + 1) * (height + 1));
  for (let y = 0; y < height; y += 1) {
    let row = 0;
    for (let x = 0; x < width; x += 1) {
      row += Math.abs(first.data[y * width + x] - second.data[y * width + x]);
      table[(y + 1) * (width + 1) + x + 1] = table[y * (width + 1) + x + 1] + row;
    }
  }
  const sum = (left: number, top: number): number => {
    const right = left + windowWidth;
    const bottom = top + windowHeight;
    return table[bottom * (width + 1) + right] - table[top * (width + 1) + right] -
      table[bottom * (width + 1) + left] + table[top * (width + 1) + left];
  };
  let best = { left: 0, top: 0, score: -1 };
  for (let top = 0; top + windowHeight <= height; top += 1) {
    for (let left = 0; left + windowWidth <= width; left += 1) {
      const score = sum(left, top);
      if (score > best.score) best = { left, top, score };
    }
  }
  return { left: best.left, top: best.top, width: windowWidth, height: windowHeight };
};

/**
 * 盲検の左右の割り付け（`true` = 1 つ目の手順を左に置く）。seed 固定の擬似乱数（mulberry32）なので、
 * 同じ seed と対の数なら同じ割り付けが再現する（対応表を作り直せる）。
 */
export const blindAssignment = (count: number, seed: number): readonly boolean[] => {
  if (!Number.isInteger(count) || count < 0 || !Number.isInteger(seed)) {
    throw new RangeError(`盲検の割り付け: 対の数 ${count} と seed ${seed} は整数`);
  }
  let state = seed >>> 0;
  const next = (): number => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return Array.from({ length: count }, () => next() < 0.5);
};

/** 合成の地。 */
export type Background = "checker" | "black" | "white";

/** 市松の 1 マスの辺（画素）と 2 色。 */
const CHECKER_CELL = 16;
const CHECKER_LIGHT = 192;
const CHECKER_DARK = 128;

const backgroundAt = (background: Background, x: number, y: number): number => {
  if (background === "black") return 0;
  if (background === "white") return 255;
  return ((Math.floor(x / CHECKER_CELL) + Math.floor(y / CHECKER_CELL)) % 2 === 0)
    ? CHECKER_LIGHT
    : CHECKER_DARK;
};

/** RGB8 の画像（行優先・画素あたり 3 バイト）。 */
export type Rgb = {
  readonly data: Uint8Array<ArrayBufferLike>;
  readonly width: number;
  readonly height: number;
};

/** 前景（ストレート α）を地へ合成した RGB8（`round(c·a + 地·(1 − a))`・a = α / 255）。 */
export const composite = (image: Rgb, alpha: Plane, background: Background): Rgb => {
  const { width, height } = image;
  if (
    alpha.width !== width || alpha.height !== height || image.data.length !== width * height * 3 ||
    alpha.data.length !== width * height
  ) {
    throw new Error(`合成: 画像 ${width}×${height} と α ${alpha.width}×${alpha.height} が揃わない`);
  }
  const data = new Uint8Array(width * height * 3);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      const a = alpha.data[index] / ALPHA_MAX;
      const ground = backgroundAt(background, x, y);
      for (let channel = 0; channel < 3; channel += 1) {
        data[index * 3 + channel] = Math.round(
          image.data[index * 3 + channel] * a + ground * (1 - a),
        );
      }
    }
  }
  return { data, width, height };
};

/** 矩形の切り出しを `factor` 倍の最近傍で拡大した RGB8（画素の段がそのまま見える形）。 */
export const zoomNearest = (image: Rgb, window: Window, factor: number): Rgb => {
  const width = window.width * factor;
  const height = window.height * factor;
  const data = new Uint8Array(width * height * 3);
  for (let y = 0; y < height; y += 1) {
    const sourceY = window.top + Math.floor(y / factor);
    for (let x = 0; x < width; x += 1) {
      const from = (sourceY * image.width + window.left + Math.floor(x / factor)) * 3;
      data.set(image.data.subarray(from, from + 3), (y * width + x) * 3);
    }
  }
  return { data, width, height };
};

/** 同じ寸法の画像の行列を `gap` 画素の隙間（灰色）を挟んで並べた RGB8（行優先の `rows[行][列]`）。 */
export const tile = (rows: readonly (readonly Rgb[])[], gap: number): Rgb => {
  const [first] = rows[0] ?? [];
  if (first === undefined) throw new Error("並べる画像が無い");
  const columns = rows[0].length;
  if (
    rows.some((row) =>
      row.length !== columns ||
      row.some((cell) => cell.width !== first.width || cell.height !== first.height)
    )
  ) {
    throw new Error("並べる画像の数か寸法が揃わない");
  }
  const width = columns * first.width + (columns - 1) * gap;
  const height = rows.length * first.height + (rows.length - 1) * gap;
  const data = new Uint8Array(width * height * 3).fill(96);
  rows.forEach((row, rowIndex) => {
    row.forEach((cell, columnIndex) => {
      const left = columnIndex * (first.width + gap);
      const top = rowIndex * (first.height + gap);
      for (let y = 0; y < cell.height; y += 1) {
        const from = y * cell.width * 3;
        data.set(cell.data.subarray(from, from + cell.width * 3), ((top + y) * width + left) * 3);
      }
    });
  });
  return { data, width, height };
};
