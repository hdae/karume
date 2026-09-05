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
 * NOTE: タイルの**本数**は入口で見ない。丸め等間隔配置（ADR 0033 追記 2026-09-02）では
 * `ceil((latent − 64) / 56) + 1` で、この受理集合の上限 latent 256 でも 5 本にしかならない
 * ため、跳ねる辺そのものが存在しない。下限 512px を満たせばタイルは必ず組める。
 *
 * MUST: 受理集合を manifest に書かない（ADR 0038 §2「導出元を一意に保つ」）— アーキ定数は
 * パイプライン実装、`S` の上限は dyn グラフの `Dim` 宣言、資産の実在は weights / assets 表。
 * **このモジュールの定数が受理集合の正本**である。
 */

import { ANIMA_SPATIAL_COMPRESSION } from "./dit-tokens.ts";

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
 *
 * NOTE（1920px 超はモデル宣言外の外挿位置 — 2026-09-04 裁定）: 素表の行数を決めているのは
 * **t 軸の上限**で、h / w 軸の宣言（`rope.max_size` = 120 トークン = 1920px）はそれより低い。
 * したがって 1920px 超〜2048px の要求は表の行としては引けるが、モデルが学習時に宣言した
 * h / w の範囲の**外側の位置**を使うことになる。数値は上流と同じ式で定義されるので静的形との
 * ビット一致は保たれ、差が出るとすれば学習分布外の位置に由来する**品質だけ**（未実測）。
 * 受理集合はこの理由では狭めない（実測なしに配布形の自由度を削らない）。
 */
const MAX_LATENT_SIDE = 256;

/** 各辺の上限（ピクセル）。素表の天井を解像度へ写したもの。 */
export const MAX_RESOLUTION_SIDE = MAX_LATENT_SIDE * ANIMA_SPATIAL_COMPRESSION;

/** トークン長 `S` の上限（S 形グラフの `Dim("S", max=16384)` — ADR 0034）。 */
export const MAX_DIT_TOKENS = 16384;

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
 * 受理集合の検査（4 条件）。呼び出し側の綴りに依らず**同じ順**で見る — 複数破っている形で
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
