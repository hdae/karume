/**
 * Wan の動画 VAE の**タイル decode**（ホスト側 — ADR 0118 決定 2・段 5）。
 *
 * chunk グラフ（`vae-chunks.ts`）の空間は潜在 `t×t` の固定タイル（開いた資産の入力形）なので、
 * 全画面（832×480 = 潜在 60×104）はホストがタイルに切って chunk 列を回し、重なりをブレンドして
 * 貼り合わせる（常時タイル — ADR 0033 / 0038 §4 の動画版）。Python 側の正は
 * `tools/export-recipes/wan/vae_tiling.py`（GPU の照合が読む参照フィクスチャもそこが書く）。
 *
 * ## タイル幾何 = 丸め等間隔スナップ配置（上流からの意図的な逸脱）
 *
 * 上流（diffusers `AutoencoderKLWan.tiled_decode`）は `range(0, H, stride)` で走査するので最後の
 * タイルが短くなり、固定形の chunk グラフでは食えない。開始位置は 0 と `extent − tile` の間を
 * **丸めて等分**する（Anima と同じ規則 — ADR 0033 追記 P-3）。本数は「隣り合う対の重なりが
 * 下限以上」を満たす最小（{@link planWanVaeTileAxis}）。下限の正本は出力の 64 px で、潜在へは
 * 空間の圧縮で割って導く（{@link wanVaeMinTileOverlap} — ADR 0121 決定 6）。Wan2.1 は 64 ÷ 8 = 8 で、
 * 832×480 は行 0 / 14 / 28 × 列 0 / 24 / 48 / 72 の 12 枚。タイル辺と縮尺は資産の宣言から引く
 * （{@link planWanVaeTiles} — タイル辺の差し替えを配布物だけで済ませる。決定 2）。
 *
 * ## ブレンドは上流の線形ランプと同型・貼り付けは領域割り当て
 *
 * `blend_v` / `blend_h` と同じ式・同じ順序（**上のタイルとの縦ブレンドが先・左との横ブレンドが
 * 後**）・同じ sample 空間で、全フレーム・全チャネルに掛ける。上流はタイル配列を in-place に
 * 書き換えるので、隣に効くのはブレンド済みのタイル — ここでも同じ。貼り付けは「タイル i の担当 =
 * `[starts[i], starts[i+1])`・最後だけ末端まで」（上流の「stride 幅へ切り詰め + 全体 crop」の
 * スナップ版 — stride 幅で切り詰めると末端が欠ける。ADR 0033 決定 3）。
 *
 * ## タイルが外・chunk が内
 *
 * タイルごとに cache をゼロから作り直す（{@link decodeWanVaeTile} がタイルの頭でゼロに書く —
 * 上流 `tiled_decode` の `clear_cache` と同じ）。逆順だとタイルの枚数ぶんの cache を同時に持つ
 * （決定 2）。1 タイル = 1 batch・フレームだけを読み戻す手順は `vae-chunks.ts` の doc。
 *
 * ## 縮退（タイル 1 枚）は非タイルの chunk 列とビット同一 MUST
 *
 * 潜在がタイルちょうど（`extent === tile`）ならタイルは 1 枚で、ブレンドは 1 度も走らず担当領域が
 * 全体になる。この経路が chunk 列（{@link decodeWanVaeTile}）と 1 ビットでも違ったら切り出しか
 * 貼り付けの誤り — 実 GPU の e2e が Uint32 で門にする（ADR 0033 決定 4）。
 *
 * ## クランプ
 *
 * {@link decodeWanVaeTiled} の戻りは**クランプ前**（照合はクランプ前で行う — 飽和した要素の差を
 * 隠さない）。最終フレームは {@link clampWanVaeFrames} を通した値（上流と同じく貼り付けの後）。
 * unpatchify のある世代（Wan2.2）は、貼り合わせ（patchify 空間・クランプ前）→ {@link unpatchifyWanVaeFrames}
 * → クランプの順（上流の `_decode` / `tiled_decode` と同じ — ブレンドは patchify 空間で行う）。
 */

import type { GpuContext, Session } from "@karume/runtime";
import { decodeWanVaeTile, type WanVaeChunkCaches, type WanVaeChunkLayout } from "./vae-chunks.ts";

const f32 = Math.fround;

/**
 * 隣り合うタイルの出力画素での最小の重なり。上流の既定のブレンド幅
 * `tile_sample_min − tile_sample_stride`（256 − 192 = 64 px）と同じ。Python 側
 * `vae_tiling.MIN_OVERLAP_PX` と同じ値（潜在の重なりはここから {@link wanVaeMinTileOverlap} で導く —
 * 世代ごとに潜在の値を持たない）。
 */
const MIN_OVERLAP_PX = 64;

/**
 * 空間の圧縮（潜在 1 あたりの最終出力の画素数）= グラフの入出力の空間比 `scale` × ホストの
 * unpatchify の倍率 `patchSize`（ADR 0121 決定 6）。グラフの比は chunk グラフの宣言
 * （`sampleTile / tile`）、unpatchify の倍率は世代の記述子（`vaePatchSize`）から来る。
 *
 * MUST: 潜在の大きさ・タイルで覆えるかの門・重なりの式は、この 1 本から圧縮を取る（`tile-decode.ts`
 * の `wanSpatialCompression` と {@link wanVaeMinTileOverlap}）。圧縮をグラフの比だけで取ると、
 * unpatchify のある世代で潜在が倍の大きさになり、重なりも倍になる。
 */
export const wanVaeSpatialCompression = (scale: number, patchSize: number): number => {
  if (!Number.isInteger(scale) || scale < 1 || !Number.isInteger(patchSize) || patchSize < 1) {
    throw new Error(
      `空間の圧縮の因子が正の整数でない（グラフの比 ${scale}・unpatchify ${patchSize}）`,
    );
  }
  return scale * patchSize;
};

/**
 * 隣り合うタイルが潜在で重なる最小幅 = {@link MIN_OVERLAP_PX} ÷ 空間の圧縮
 * （{@link wanVaeSpatialCompression}）。Python 側 `vae_tiling.min_overlap_latent` と同じ式・同じ文言
 * （タイル計画の凍結表が両側で割れる）。
 *
 * MUST: 割り切れない圧縮は丸めずに落とす（丸めると重なりが 64 px から黙ってずれる）。
 */
export const wanVaeMinTileOverlap = (scale: number, patchSize: number): number => {
  const compression = wanVaeSpatialCompression(scale, patchSize);
  if (MIN_OVERLAP_PX % compression !== 0) {
    throw new Error(
      `重なり ${MIN_OVERLAP_PX} px が空間の圧縮 ${compression}` +
        `（グラフの比 ${scale} × unpatchify ${patchSize}）で割り切れない`,
    );
  }
  return MIN_OVERLAP_PX / compression;
};

/** 潜在の 1 軸ぶんのタイル配置。 */
export type WanVaeTileAxis = {
  /** この軸の潜在の全長。 */
  readonly extent: number;
  /** タイル 1 枚の潜在の幅（= chunk グラフの入力形）。 */
  readonly tile: number;
  /**
   * 各タイルの開始位置（潜在・昇順・末尾はちょうど `extent − tile`）。
   *
   * MUST: 開始位置の差（間隔）を欄として持たない — 対ごとに 1 潜在まで動くので、1 つの数に
   * 畳むと配置と黙って食い違う。要るところで `starts` から引く（{@link wanVaeBlendExtentAt}）。
   */
  readonly starts: readonly number[];
};

/** 潜在 `[C, F, H, W]` の 2 軸ぶんのタイル配置と、潜在 ↔ sample の縮尺。 */
export type WanVaeTilePlan = {
  /** 潜在のチャネル数（切り出しの平面数を決める）。 */
  readonly latentChannels: number;
  /**
   * chunk グラフの出口のチャネル数（貼り合わせでは、タイルの平面の数がこれの倍数であることの検査に
   * 使う — ブレンドと貼り付けは平面ごとに回すので、チャネルとフレームを分けない）。
   */
  readonly sampleChannels: number;
  /** 潜在 1 あたりの sample 画素数。 */
  readonly scale: number;
  /** 高さ軸。 */
  readonly rows: WanVaeTileAxis;
  /** 幅軸。 */
  readonly cols: WanVaeTileAxis;
};

/**
 * 1 軸ぶんの丸め等間隔スナップ配置（Python 側 `vae_tiling.plan_tile_axis` と同じ規則）。
 *
 * 本数は「重なりが `minOverlap` 以上」を満たす最小値 `ceil(span / (tile − minOverlap)) + 1`
 * （`span = extent − tile`）、開始位置は `round(i · span / (本数 − 1))`（`Math.round` = 0.5 は
 * 切り上げ — Python 側は同じ向きの整数式）。丸めの誤差が ±0.5 に収まるので間隔は 2 値にしか
 * ならない。
 *
 * `minOverlap` は潜在の単位で必須（既定値を置かない — 世代ごとの値は {@link wanVaeMinTileOverlap}
 * で導き、渡し忘れを Wan2.1 の値で黙って計画しない）。
 */
export const planWanVaeTileAxis = (
  extent: number,
  tile: number,
  minOverlap: number,
): WanVaeTileAxis => {
  if (!Number.isInteger(extent) || !Number.isInteger(tile) || !Number.isInteger(minOverlap)) {
    throw new Error(`タイル配置は整数で組む（extent=${extent} tile=${tile} 重なり=${minOverlap}）`);
  }
  if (tile < 1) throw new Error(`タイル幅 ${tile} が 1 未満`);
  if (extent < tile) {
    throw new Error(
      `潜在の全長 ${extent} がタイル幅 ${tile} より小さい（固定形の chunk グラフに食わせられない）`,
    );
  }
  if (minOverlap < 0 || minOverlap >= tile) {
    throw new Error(`最小の重なり ${minOverlap} が [0, ${tile}) の外（重なりはタイル幅未満）`);
  }
  const span = extent - tile;
  if (span === 0) {
    // 縮退: 1 枚で覆える。対が無いのでブレンドは走らず、貼り付けが素の写しになる（モジュール doc
    // の MUST — 非タイルの chunk 列とビット同一）。
    return { extent, tile, starts: [0] };
  }
  const count = Math.ceil(span / (tile - minOverlap)) + 1;
  const starts: number[] = [];
  for (let index = 0; index < count; index += 1) {
    starts.push(Math.round((index * span) / (count - 1)));
  }
  // 重なりの下限は本数の式から導けるが、導出は丸めの誤差評価に依っていて目で追えない。破れたら
  // 継ぎ目がランプで隠れなくなる（絵にしか出ない沈黙誤り）ので、構造で落とす。
  for (let index = 1; index < count; index += 1) {
    const overlap = tile - (starts[index] - starts[index - 1]);
    if (overlap < minOverlap) {
      throw new Error(
        `タイル ${index - 1}/${index} の重なり ${overlap} が下限 ${minOverlap} 未満` +
          `（潜在 ${extent} / タイル ${tile} / 開始位置 ${starts}）`,
      );
    }
  }
  return { extent, tile, starts };
};

/**
 * 対 `(index − 1, index)` のブレンド幅（**sample 空間** — 上流の `blend_height` / `blend_width` と
 * 同じ単位）。丸め等間隔なので対ごとに 1 潜在まで動く。
 */
export const wanVaeBlendExtentAt = (axis: WanVaeTileAxis, scale: number, index: number): number => {
  if (!Number.isInteger(index) || index < 1 || index >= axis.starts.length) {
    throw new RangeError(
      `ブレンド対 ${index} が範囲外（開始位置 ${axis.starts.length} 本 = 対 ${
        axis.starts.length - 1
      } 組）`,
    );
  }
  return (axis.tile - (axis.starts[index] - axis.starts[index - 1])) * scale;
};

/** タイル `index` の担当領域の長さ（潜在）= 次のタイルの開始まで・最後だけ末端まで。 */
const regionOf = (axis: WanVaeTileAxis, index: number): number =>
  (index + 1 < axis.starts.length ? axis.starts[index + 1] : axis.extent) - axis.starts[index];

/**
 * 潜在の空間 `height × width` のタイル計画。タイル辺・縮尺・チャネル数は chunk グラフの宣言から引く
 * （`layout.tile` と `layout.sampleTile / layout.tile` — 呼び手に 32 や 8 を literal で置かない）。
 * `minOverlap` は {@link planWanVaeTileAxis} と同じく潜在の単位で必須（生成要求の計画は
 * `tile-decode.ts` の `planWanGenerationTiles` を通す）。
 */
export const planWanVaeTiles = (
  layout: Pick<WanVaeChunkLayout, "latentChannels" | "tile" | "sampleTile" | "sampleChannels">,
  height: number,
  width: number,
  minOverlap: number,
): WanVaeTilePlan => {
  const scale = layout.sampleTile / layout.tile;
  if (!Number.isInteger(scale) || scale < 1) {
    throw new Error(`タイルの縮尺 ${layout.sampleTile} / ${layout.tile} が正の整数でない`);
  }
  return {
    latentChannels: layout.latentChannels,
    sampleChannels: layout.sampleChannels,
    scale,
    rows: planWanVaeTileAxis(height, layout.tile, minOverlap),
    cols: planWanVaeTileAxis(width, layout.tile, minOverlap),
  };
};

/** タイルの総枚数。 */
export const wanVaeTileCount = (plan: WanVaeTilePlan): number =>
  plan.rows.starts.length * plan.cols.starts.length;

/** 潜在 `[C, F, H, W]` のフレーム数 F（長さが計画と合わなければ fail loudly）。 */
const latentFrames = (plan: WanVaeTilePlan, latents: Float32Array): number => {
  const plane = plan.latentChannels * plan.rows.extent * plan.cols.extent;
  const frames = latents.length / plane;
  if (!Number.isInteger(frames) || frames < 1) {
    throw new Error(
      `潜在の要素数 ${latents.length} が [${plan.latentChannels},F,${plan.rows.extent},${plan.cols.extent}] でない`,
    );
  }
  return frames;
};

/**
 * 潜在 `[C, F, H, W]` から `(row, col)` のタイル `[C, F, t, t]` を切り出す（平面ごとのコピー —
 * `C·F` 枚の平面はどれも `H × W`）。
 */
export const wanVaeLatentTile = (
  plan: WanVaeTilePlan,
  latents: Float32Array,
  row: number,
  col: number,
): Float32Array<ArrayBuffer> => {
  const { rows, cols } = plan;
  if (row < 0 || row >= rows.starts.length || col < 0 || col >= cols.starts.length) {
    throw new RangeError(
      `タイル (${row}, ${col}) が範囲外（${rows.starts.length}×${cols.starts.length}）`,
    );
  }
  const planes = plan.latentChannels * latentFrames(plan, latents);
  const sourcePlane = rows.extent * cols.extent;
  const tilePlane = rows.tile * cols.tile;
  const top = rows.starts[row];
  const left = cols.starts[col];
  const out = new Float32Array(planes * tilePlane);
  for (let plane = 0; plane < planes; plane += 1) {
    for (let y = 0; y < rows.tile; y += 1) {
      const from = plane * sourcePlane + (top + y) * cols.extent + left;
      out.set(latents.subarray(from, from + cols.tile), plane * tilePlane + y * cols.tile);
    }
  }
  return out;
};

/**
 * 上のタイルとの線形ランプ合成（上流 `blend_v` と同型・**in-place**）。
 *
 * `b[y] = a[H−blend+y]·(1 − y/blend) + b[y]·(y/blend)`。MUST: 重み 2 本を先に f32 へ丸めてから
 * 掛ける（torch は Python float のスカラをテンソル dtype へ落として演算する）— 丸めの位置を
 * 動かすと参照と最終桁で割れる。
 */
const blendVertical = (
  above: Float32Array,
  current: Float32Array,
  planes: number,
  height: number,
  width: number,
  blend: number,
): void => {
  for (let y = 0; y < blend; y += 1) {
    const weightAbove = f32(1 - y / blend);
    const weightCurrent = f32(y / blend);
    for (let plane = 0; plane < planes; plane += 1) {
      const base = plane * height * width;
      const from = base + (height - blend + y) * width;
      const to = base + y * width;
      for (let x = 0; x < width; x += 1) {
        current[to + x] = f32(
          f32(above[from + x] * weightAbove) + f32(current[to + x] * weightCurrent),
        );
      }
    }
  }
};

/** 左のタイルとの線形ランプ合成（上流 `blend_h` と同型・**in-place**）。 */
const blendHorizontal = (
  left: Float32Array,
  current: Float32Array,
  planes: number,
  height: number,
  width: number,
  blend: number,
): void => {
  for (let x = 0; x < blend; x += 1) {
    const weightLeft = f32(1 - x / blend);
    const weightCurrent = f32(x / blend);
    for (let plane = 0; plane < planes; plane += 1) {
      const base = plane * height * width;
      for (let y = 0; y < height; y += 1) {
        const from = base + y * width + (width - blend + x);
        const to = base + y * width + x;
        current[to] = f32(f32(left[from] * weightLeft) + f32(current[to] * weightCurrent));
      }
    }
  }
};

/**
 * 貼り合わせの本体（**渡された配列の上で in-place にブレンドする** — 呼び手がその配列を所有して
 * いることが前提）。戻りは `[Cs, F', H·s, W·s]`（Cs は計画の `sampleChannels`・F' はタイルの
 * フレーム数）。
 */
const assembleOwnedTiles = (
  working: readonly Float32Array[],
  plan: WanVaeTilePlan,
): Float32Array<ArrayBuffer> => {
  const { rows, cols, scale } = plan;
  const rowCount = rows.starts.length;
  const colCount = cols.starts.length;
  if (working.length !== rowCount * colCount) {
    throw new Error(`タイル ${working.length} 枚が計画の ${rowCount}×${colCount} と違う`);
  }
  const tileHeight = rows.tile * scale;
  const tileWidth = cols.tile * scale;
  const tilePlane = tileHeight * tileWidth;
  const planes = working[0].length / tilePlane;
  if (!Number.isInteger(planes) || planes < 1 || planes % plan.sampleChannels !== 0) {
    throw new Error(
      `タイルの要素数 ${
        working[0].length
      } が [${plan.sampleChannels},F,${tileHeight},${tileWidth}] でない`,
    );
  }
  for (const [index, tile] of working.entries()) {
    if (tile.length !== planes * tilePlane) {
      throw new Error(
        `タイル ${index} の要素数 ${tile.length} が 1 枚目の ${planes * tilePlane} と違う`,
      );
    }
  }

  // MUST: 縦（上）→ 横（左）の順（上流 `tiled_decode` と同じ）。入れ替えると角の 4 枚が重なる
  // 領域で係数の積の順が変わる（単体テストの角の解析解が割れる）。
  // MUST: ブレンド幅は対ごとに引く（丸め等間隔なので対で 1 潜在まで違う）。
  for (let row = 0; row < rowCount; row += 1) {
    for (let col = 0; col < colCount; col += 1) {
      const current = working[row * colCount + col];
      if (row > 0) {
        blendVertical(
          working[(row - 1) * colCount + col],
          current,
          planes,
          tileHeight,
          tileWidth,
          wanVaeBlendExtentAt(rows, scale, row),
        );
      }
      if (col > 0) {
        blendHorizontal(
          working[row * colCount + col - 1],
          current,
          planes,
          tileHeight,
          tileWidth,
          wanVaeBlendExtentAt(cols, scale, col),
        );
      }
    }
  }

  const height = rows.extent * scale;
  const width = cols.extent * scale;
  const out = new Float32Array(planes * height * width);
  for (let row = 0; row < rowCount; row += 1) {
    // MUST: 幅は担当領域ちょうどにする。過大（常にタイル全幅）でも行優先の走査では後続タイルが
    // 上書きして値が変わらず、数値の門では検出できない（ADR 0033 の検出限界 1）— その無害さは
    // 走査順に依存するので、貼り付けを並列化・逆順化した瞬間に沈黙誤値へ変わる。
    const top = rows.starts[row] * scale;
    const spanRows = regionOf(rows, row) * scale;
    for (let col = 0; col < colCount; col += 1) {
      const left = cols.starts[col] * scale;
      const spanCols = regionOf(cols, col) * scale;
      const tile = working[row * colCount + col];
      for (let plane = 0; plane < planes; plane += 1) {
        for (let y = 0; y < spanRows; y += 1) {
          const from = plane * tilePlane + y * tileWidth;
          out.set(
            tile.subarray(from, from + spanCols),
            plane * height * width + (top + y) * width + left,
          );
        }
      }
    }
  }
  return out;
};

/**
 * decode 済みのタイル（行優先・各 `[Cs, F', s, s]`）をブレンドして `[Cs, F', H·s, W·s]` に貼り
 * 合わせる（クランプ前のまま・Cs は計画の `sampleChannels`）。
 *
 * MUST: 渡された配列を破壊しない（ブレンドは in-place なので写しの上で行う）。配列を自分で所有
 * する {@link decodeWanVaeTiled} は写さずに本体を呼ぶ。
 */
export const assembleWanVaeTiles = (
  tiles: readonly Float32Array[],
  plan: WanVaeTilePlan,
): Float32Array<ArrayBuffer> =>
  assembleOwnedTiles(tiles.map((tile) => Float32Array.from(tile)), plan);

/**
 * 全タイルを行優先で decode する（**タイルが外・chunk が内** — タイルごとに
 * {@link decodeWanVaeTile} が cache をゼロから回す）。戻りは各タイルのクランプ前の
 * `[Cs, 1 + 4(F−1), s, s]`（呼び手が所有する新しい配列）。
 *
 * `latents` は逆正規化済みの `[C, F, H, W]`。計画のタイル辺・縮尺・チャネル数（潜在と出口）が開いた
 * 資産（`caches.layout`）と食い違えば fail loudly。
 *
 * `onTile` はタイル 1 枚の decode が決着するたびに await する（`tile` は 1 始まり・行優先 — パイプラインの
 * 進捗と診断の口。投げれば残りのタイルを回さずに投げ直す）。
 *
 * MUST: 同じ device の別の batch・run を並行に発行しない（タイルごとの batch が device の区間
 * ロックを持つ — `decodeWanVaeTile`）。
 */
export const decodeWanVaeTiles = async (
  gpu: GpuContext,
  sessions: { readonly first: Session; readonly next: Session },
  caches: WanVaeChunkCaches,
  plan: WanVaeTilePlan,
  latents: Float32Array,
  onTile?: (tile: number) => void | Promise<void>,
): Promise<Float32Array<ArrayBuffer>[]> => {
  const { layout } = caches;
  if (
    plan.latentChannels !== layout.latentChannels || plan.rows.tile !== layout.tile ||
    plan.cols.tile !== layout.tile || plan.scale * layout.tile !== layout.sampleTile ||
    plan.sampleChannels !== layout.sampleChannels
  ) {
    throw new Error(
      `タイル計画（C ${plan.latentChannels}・タイル ${plan.rows.tile}×${plan.cols.tile}・縮尺 ${plan.scale}` +
        `・出口 ${plan.sampleChannels} チャネル）が資産（C ${layout.latentChannels}・タイル ${layout.tile}` +
        `・sample ${layout.sampleTile}・出口 ${layout.sampleChannels} チャネル）と違う`,
    );
  }
  // 潜在の長さは GPU を回す前に見る（タイルごとの切り出しでも落ちるが、それだと前のタイルの
  // decode を払ってから落ちる）。
  latentFrames(plan, latents);
  const decoded: Float32Array<ArrayBuffer>[] = [];
  for (let row = 0; row < plan.rows.starts.length; row += 1) {
    for (let col = 0; col < plan.cols.starts.length; col += 1) {
      decoded.push(
        await decodeWanVaeTile(gpu, sessions, caches, wanVaeLatentTile(plan, latents, row, col)),
      );
      await onTile?.(decoded.length);
    }
  }
  return decoded;
};

/**
 * タイル decode の本体: 全タイルを decode してブレンド・貼り付けした**クランプ前**の
 * `[Cs, 1 + 4(F−1), H·s, W·s]`。最終フレームは {@link clampWanVaeFrames} を通す。`onTile` は
 * {@link decodeWanVaeTiles} と同じ。
 */
export const decodeWanVaeTiled = async (
  gpu: GpuContext,
  sessions: { readonly first: Session; readonly next: Session },
  caches: WanVaeChunkCaches,
  plan: WanVaeTilePlan,
  latents: Float32Array,
  onTile?: (tile: number) => void | Promise<void>,
): Promise<Float32Array<ArrayBuffer>> =>
  assembleOwnedTiles(await decodeWanVaeTiles(gpu, sessions, caches, plan, latents, onTile), plan);

/**
 * patchify 空間 `[Cs, F, h, w]` → `[Cs/p², F, h·p, w·p]`（上流 diffusers `unpatchify` と同じ並び —
 * `view(c, r, q, f, h, w).permute(c, f, h, q, w, r)`・r = q = p）。添字の対応は
 * `out[c, f, y·p + dy, x·p + dx] = in[c·p² + dx·p + dy, f, y, x]`（チャネル内のずれは幅方向 dx が上位・
 * 高さ方向 dy が下位）。
 *
 * p = 1 は**入力をそのまま返す**（写さない — 上流の `if patch_size == 1: return x` と同じ）。p > 1 は
 * 新しい配列への写しだけで浮動小数の演算は無い（値はビット単位で保たれる）。
 *
 * 形・倍率・長さの食い違いは fail loudly（p = 1 でも見る）。
 */
export const unpatchifyWanVaeFrames = (
  frames: Float32Array<ArrayBuffer>,
  shape: readonly [number, number, number, number],
  patchSize: number,
): Float32Array<ArrayBuffer> => {
  if (!Number.isInteger(patchSize) || patchSize < 1) {
    throw new Error(`unpatchify の倍率 ${patchSize} が正の整数でない`);
  }
  if (shape.some((dim) => !Number.isInteger(dim) || dim < 1)) {
    throw new Error(`unpatchify の入力の形 [${shape.join(", ")}] の次元が正の整数でない`);
  }
  const [channels, count, height, width] = shape;
  const area = patchSize * patchSize;
  if (channels % area !== 0) {
    throw new Error(
      `unpatchify の入力のチャネル数 ${channels} が倍率 ${patchSize}² = ${area} で割り切れない`,
    );
  }
  const elements = channels * count * height * width;
  if (frames.length !== elements) {
    throw new Error(
      `unpatchify の入力の要素数 ${frames.length} が [${shape.join(", ")}]（${elements}）と違う`,
    );
  }
  if (patchSize === 1) return frames;
  const outChannels = channels / area;
  const plane = height * width;
  const out = new Float32Array(elements);
  // 出力の行が連続になる順（c → f → y → dy → x → dx）で回すので、書き込み先は 1 ずつ進む。
  let to = 0;
  for (let channel = 0; channel < outChannels; channel += 1) {
    for (let frame = 0; frame < count; frame += 1) {
      for (let y = 0; y < height; y += 1) {
        for (let dy = 0; dy < patchSize; dy += 1) {
          for (let x = 0; x < width; x += 1) {
            for (let dx = 0; dx < patchSize; dx += 1) {
              const source = channel * area + dx * patchSize + dy;
              out[to] = frames[(source * count + frame) * plane + y * width + x];
              to += 1;
            }
          }
        }
      }
    }
  }
  return out;
};

/**
 * フレームを `[-1, 1]` へ in-place にクランプする（上流 `torch.clamp(min=-1, max=1)` と同じ —
 * NaN は NaN のまま・`-0` は `-0` のまま）。
 */
export const clampWanVaeFrames = (frames: Float32Array): void => {
  for (let index = 0; index < frames.length; index += 1) {
    const value = frames[index];
    if (value > 1) frames[index] = 1;
    else if (value < -1) frames[index] = -1;
  }
};
