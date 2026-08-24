/**
 * 生成解像度の綴りと**受理集合**（任意解像度・非正方形）。
 *
 * ## なぜ受理集合を入口で持つのか
 *
 * 受理できない形をそのまま流すと、落ちるのは資産をロードし終えた後の深い場所（rope 素表の
 * 行数超過 / タイル配置の不成立 / `Dim("S")` の上限）で、診断も「どのノブが悪いのか」を
 * 言わない。入口で**条件を名指しして落とす**のが fail loudly。
 *
 * MUST: ここは**入口の綴り検査**であって、形の正本ではない。latent が patch で割り切れるか
 * （`dit-tokens.ts` の `latentSides`）・rope 素表の行数に収まるか（同 `ropeTables`）・タイルが
 * 組めるか（`tiling.ts` の `planTileAxis`）は、資産から引いた実際の値で別途落ちる。ここの
 * 定数はその**写し**で、資産を差し替えたときに追随するのは向こう側である。
 *
 * NOTE: 例外はタイルの**本数**で、これは写しを持たず `planTileAxis` をそのまま呼んで数える
 * （本数の決まり方は約数の乏しさに依存する非自明な式なので、二重に持つと必ず割れる）。
 * タイル幅と最小の重なりは向こう側の値を引く。
 *
 * MUST: 受理集合を manifest に書かない（ADR 0038 §2「導出元を一意に保つ」）— アーキ定数は
 * パイプライン実装、`S` の上限は dyn グラフの `Dim` 宣言、資産の実在は weights / assets 表。
 * **このモジュールの定数が受理集合の正本**である。
 */

import { ANIMA_SPATIAL_COMPRESSION } from "./dit-tokens.ts";
import { MIN_TILE_OVERLAP_LATENT, planTileAxis } from "./tiling.ts";

/** 画像の寸法（ピクセル）。軸の順は綴りと同じ `幅 × 高さ`。 */
export type ImageSize = {
  readonly width: number;
  readonly height: number;
};

/**
 * DiT の patch の一辺（Anima の `patch_size = (1,2,2)`）。
 * MUST: 実際の patch 幾何は**グラフの入出力幅から割り出す**（`ditPatchGeometry`）。ここは
 * 入口で刻みを言うためだけの写しで、食い違えば `latentSides` が資産側の値で落とす。
 */
const DIT_PATCH_SIDE = 2;

/** 各辺の刻み（= 空間圧縮 8 × patch 2）。latent の各辺が patch で割り切れる最小の単位。 */
export const RESOLUTION_GRANULARITY = ANIMA_SPATIAL_COMPRESSION * DIT_PATCH_SIDE;

/**
 * 各辺の下限。**VAE タイル decoder の latent 64**（= 512px）が 1 枚も入らない形は組めない
 * （`planTileAxis` が「latent 全長がタイル幅より小さい」で落とす）。
 */
export const MIN_RESOLUTION_SIDE = 512;

/**
 * latent の各辺の上限。rope の**軸別素表の行数**（Anima では 128 行 = トークン 128 =
 * latent 256 = 2048px 相当）がモデル側の位置表の天井そのもので、超えると上流でも組めない。
 */
const MAX_LATENT_SIDE = 256;

/** 各辺の上限（ピクセル）。素表の天井を解像度へ写したもの。 */
export const MAX_RESOLUTION_SIDE = MAX_LATENT_SIDE * ANIMA_SPATIAL_COMPRESSION;

/** トークン長 `S` の上限（S 形グラフの `Dim("S", max=16384)` — ADR 0034）。 */
export const MAX_DIT_TOKENS = 16384;

/** VAE タイル decoder の latent 1 辺（下限 512px の由来そのもの）。 */
const VAE_TILE_LATENT_SIDE = MIN_RESOLUTION_SIDE / ANIMA_SPATIAL_COMPRESSION;

/**
 * 1 辺あたりの VAE タイル本数の上限。
 *
 * `planTileAxis` は「重なりを満たす**最小本数**」を `extent − tile` の約数の中から選ぶので、
 * 約数の乏しい辺では本数が跳ねる（1456px = latent 182・span 118 = 2·59 → 60 本）。
 * `decodeTiled` は全タイルの decode 出力を抱えたまま貼り付けへ渡すため、両辺が跳ねると
 * ホスト確保は枚数の積で効く（1680px は 74×74 = 5,476 枚 ≈ 17GiB）。実質実行不能なのに
 * どこも fail loudly しないので、入口で名指しして落とす。
 *
 * 閾は実測の谷間に置く: 受理集合の他の辺は最大 8 本で、跳ねる 8 通りは 60 本以上と離れている
 * （`anima_resolution_test.ts` が本数の実測でこの谷間を固定する）。
 */
export const MAX_TILES_PER_AXIS = 10;

/** その辺を VAE タイルで覆うのに要る本数（式は `planTileAxis` と共有する）。 */
const tilesOf = (side: number): number =>
  planTileAxis(side / ANIMA_SPATIAL_COMPRESSION, VAE_TILE_LATENT_SIDE, MIN_TILE_OVERLAP_LATENT)
    .starts.length;

/**
 * 近傍の受理値（下側・上側それぞれ最も近い、本数の条件まで満たす辺）。
 *
 * 跳ねる辺は刻みの並びの中に飛び飛びで現れるので、「1 つ下の刻み」を機械的に案内すると
 * また跳ねる形を勧めうる。実際に数え直して通る値だけを返す。
 */
const nearbyAcceptableSides = (side: number): number[] => {
  const found: number[] = [];
  for (const step of [-RESOLUTION_GRANULARITY, RESOLUTION_GRANULARITY]) {
    for (
      let candidate = side + step;
      candidate >= MIN_RESOLUTION_SIDE && candidate <= MAX_RESOLUTION_SIDE;
      candidate += step
    ) {
      if (tilesOf(candidate) <= MAX_TILES_PER_AXIS) {
        found.push(candidate);
        break;
      }
    }
  }
  return found.sort((a, b) => a - b);
};

/** 正方か。 */
const isSquare = (size: ImageSize): boolean => size.width === size.height;

/**
 * 綴り → 寸法。`1344x768` と、正方の略記 `512`（= `512x512`）を受ける。
 *
 * MUST: 受理できない形は**条件を名指して**落とす。「解像度が変」だけだと、刻み・下限・
 * 天井・タイル本数・`S` の上限のどれに当たったのかが分からず、ノブを 1 つずつ試すことになる。
 */
export const parseResolution = (raw: string): ImageSize => {
  const parts = raw.toLowerCase().split("x");
  const spelled = parts.length === 1 ? [parts[0], parts[0]] : parts;
  if (spelled.length !== 2 || spelled.some((part) => !/^\d+$/.test(part))) {
    throw new Error(
      `解像度 ${JSON.stringify(raw)} が WxH でも正方の略記でもない（例: 1344x768 / 512）`,
    );
  }
  const size = { width: Number(spelled[0]), height: Number(spelled[1]) };
  assertAcceptableResolution(size);
  return size;
};

/** `1344x768` / 正方は略記（`512`）。{@link parseResolution} の逆で、往復は恒等。 */
export const formatResolution = (size: ImageSize): string =>
  isSquare(size) ? `${size.width}` : `${size.width}x${size.height}`;

/** DiT のトークン長 `S`（= 各辺の刻みで割った格子の積）。 */
const tokensOf = (size: ImageSize): number =>
  (size.width / RESOLUTION_GRANULARITY) * (size.height / RESOLUTION_GRANULARITY);

/**
 * 受理集合の検査（5 条件）。呼び出し側の綴りに依らず**同じ順**で見る — 複数破っている形で
 * 診断が run ごとに変わると、直した先でまた別の条件に当たったのかが分からない。
 */
export const assertAcceptableResolution = (size: ImageSize): void => {
  const label = `${size.width}×${size.height}`;
  for (const [side, axis] of [[size.width, "幅"], [size.height, "高さ"]] as const) {
    if (!Number.isInteger(side)) {
      throw new Error(`解像度 ${label} の${axis} ${side} が整数でない`);
    }
    if (side % RESOLUTION_GRANULARITY !== 0) {
      throw new Error(
        `解像度 ${label} の${axis} ${side} が ${RESOLUTION_GRANULARITY} の倍数でない` +
          `（latent が patch ${DIT_PATCH_SIDE} で割り切れない）`,
      );
    }
    if (side < MIN_RESOLUTION_SIDE) {
      throw new Error(
        `解像度 ${label} の${axis} ${side} が下限 ${MIN_RESOLUTION_SIDE} 未満` +
          `（VAE タイル decoder の latent ${
            MIN_RESOLUTION_SIDE / ANIMA_SPATIAL_COMPRESSION
          } が入らない）`,
      );
    }
    if (side > MAX_RESOLUTION_SIDE) {
      throw new Error(
        `解像度 ${label} の${axis} ${side} が上限 ${MAX_RESOLUTION_SIDE} 超` +
          `（rope 素表の天井 = latent ${MAX_LATENT_SIDE}）`,
      );
    }
    const tiles = tilesOf(side);
    if (tiles > MAX_TILES_PER_AXIS) {
      throw new Error(
        `解像度 ${label} の${axis} ${side} は VAE タイルが ${tiles} 本必要で上限 ` +
          `${MAX_TILES_PER_AXIS} 超（decode 出力を全枚数ぶん抱えるのでホスト RAM が破綻する）。` +
          `近傍の受理値: ${nearbyAcceptableSides(side).join(" / ")}`,
      );
    }
  }
  const tokens = tokensOf(size);
  // NOTE: 現行定数では各辺の上限（2048 = 格子 128）から S ≤ 128² = MAX_DIT_TOKENS なので、
  // この分岐に入る形は作れない。定数（辺の上限 / 刻み / S の上限）を動かしたときの保険として
  // 残す — 到達しないことは定数関係の門（`anima_resolution_test.ts`）が固定する。
  if (tokens > MAX_DIT_TOKENS) {
    throw new Error(
      `解像度 ${label} のトークン長 S=${tokens} が上限 ${MAX_DIT_TOKENS} 超` +
        '（S 形グラフの Dim("S") の上限 — 各辺の条件とは別に面積で決まる）',
    );
  }
};
