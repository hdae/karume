/**
 * Wan2.2 TI2V-5B の VAE decoder の chunk グラフ（f16 系列 `wan2.2-ti2v-5b-f16-dyn`）のテストが共有するもの
 * （ADR 0121 段 4）: 系列の置き場・フィクスチャのケースの表・取り決めの期待値（cache の表を含む）・資産の在否。
 *
 * GPU 不要の突き合わせ（`wan_ti2v_vae_chunks_host_test.ts`）と実 GPU の chunk 列の照合
 * （`e2e_wan_ti2v_vae_chunks_test.ts`）が同じ表を読む — 表を 2 か所で育てると、片方だけに足したケースが黙って
 * 片方の門を外れる。
 *
 * ケースの正本は `tools/export-recipes/wan/export_vae.py` の `TI2V_FIXTURE_CASES`。cache の表の正本は上流の decoder の
 * forward の順で、recipe の `wan/tests/test_vae_patch_ti2v.py` の `TestCacheTableRealWeights.TABLE` が同じ値を凍結する
 * （二重凍結 — 片方だけ書き換えると、もう片方の側のテストが落ちる）。**列挙結果ではなくここで固定する**（生成を
 * 一部だけ流した環境でテストが黙って消える形にしない）。
 *
 * 段 5 のタイル参照 `vae_tiles.*`（書き手は `tools/export-recipes/wan/vae_tiling.py --model ti2v-5b` — ケースの正本は
 * その `TI2V_FIXTURE_CASES`）は、chunk グラフの組とは**別の在否の組**（{@link ti2vVaeTileAssets} — 3 本とも無ければ
 * 明示 SKIP・一部だけなら FAIL）。書き手が別の台本（`vae_tiling.py`）で、chunk の組とは別の時点に書かれるため。
 */

import { modelPresent } from "../../../runtime/tests/helpers/container-files.ts";
import { filePresent } from "./wan-ti2v-dit.ts";

export const WAN_TI2V_VAE_SERIES = "wan2.2-ti2v-5b-f16-dyn";
export const WAN_TI2V_VAE_ROOT = new URL(
  `../../../../outputs/series/${WAN_TI2V_VAE_SERIES}/`,
  import.meta.url,
);
/** 部品名（= 部品ディレクトリ名。グラフ名は系列のグラフ名の表 `series-graphs.ts` から引く）。 */
export const WAN_TI2V_VAE_FIRST = "vae_decoder_first";
export const WAN_TI2V_VAE_NEXT = "vae_decoder_next";
export const WAN_TI2V_VAE_MODEL_FILE = "model.krm";

/** SKIP 時にそのまま貼れる生成コマンド（容器 2 本とフィクスチャ 3 本を一組で据える）。 */
export const WAN_TI2V_VAE_GENERATE = "cd tools/export-recipes && uv run --group wan --inexact " +
  "python -m wan.export_vae --model ti2v-5b";

/** フィクスチャ 1 本（帯の決定用 1 本 + 受入れ 2 本 — 2.1 と同じ組み立て）。 */
export type Ti2vVaeCase = {
  readonly name: "band" | "accept" | "long";
  readonly role: "band" | "accept";
  readonly chunks: number;
};

/** `long` は 81 フレーム（1 + 4·20 — cache を 20 回持ち越す長さ）の chunk 列。 */
export const WAN_TI2V_VAE_CASES: readonly Ti2vVaeCase[] = [
  { name: "band", role: "band", chunks: 9 },
  { name: "accept", role: "accept", chunks: 5 },
  { name: "long", role: "accept", chunks: 21 },
];

/** 潜在タイルの辺（ADR 0121 決定 6 — 出力 256 px ÷ 空間の圧縮 16）。 */
export const WAN_TI2V_VAE_TILE = 16;
/** フレームの辺（patchify 空間 — 潜在 1 あたり 8）。 */
export const WAN_TI2V_VAE_SAMPLE_TILE = 128;
/** 潜在のチャネル数（上流 config の `z_dim`）。 */
export const WAN_TI2V_VAE_LATENT_CHANNELS = 48;
/** 出口のチャネル数（patchify 空間 `3·p²` — unpatchify はグラフの外）。 */
export const WAN_TI2V_VAE_SAMPLE_CHANNELS = 12;

/** cache の表の 1 行（名前 = `cache_` + 上流の `feat_idx` の 2 桁・形はタイル 16 のとき）。 */
export type Ti2vVaeCacheRow = {
  readonly name: string;
  readonly shape: readonly number[];
  /** upsample3d の `time_conv` の cache（first は持たない — 最初の chunk では走らない）。 */
  readonly timeConv: boolean;
};

const cacheRows = (
  from: number,
  to: number,
  shape: readonly number[],
  timeConv = false,
): Ti2vVaeCacheRow[] =>
  Array.from({ length: to - from }, (_, offset) => ({
    name: `cache_${String(from + offset).padStart(2, "0")}`,
    shape,
    timeConv,
  }));

/** next の cache の全体（入力の順 — recipe の表と同じ並び・同じ値）。 */
export const WAN_TI2V_VAE_CACHE_TABLE: readonly Ti2vVaeCacheRow[] = [
  ...cacheRows(0, 1, [48, 2, 16, 16]),
  ...cacheRows(1, 11, [1024, 2, 16, 16]),
  ...cacheRows(11, 12, [1024, 2, 16, 16], true),
  ...cacheRows(12, 18, [1024, 2, 32, 32]),
  ...cacheRows(18, 19, [1024, 2, 32, 32], true),
  ...cacheRows(19, 20, [1024, 2, 64, 64]),
  ...cacheRows(20, 25, [512, 2, 64, 64]),
  ...cacheRows(25, 26, [512, 2, 128, 128]),
  ...cacheRows(26, 32, [256, 2, 128, 128]),
];

/** `time_conv` の cache（next だけが持つ — 表の `timeConv` から）。 */
export const WAN_TI2V_VAE_TIME_CONV_CACHES: readonly string[] = WAN_TI2V_VAE_CACHE_TABLE
  .filter(({ timeConv }) => timeConv)
  .map(({ name }) => name);

/**
 * cache の常駐の合計バイト数（f32 — 116,940,800 要素 × 4。recipe の実重みのテストの要素数と同じ値で、表とは独立に
 * 書く — 表の組み立ての誤りをここで拾う）。
 */
export const WAN_TI2V_VAE_CACHE_BYTES = 467_763_200;
/**
 * first の cache の合計バイト数（f32 — 114,319,360 要素 × 4。recipe の同じテストの first の要素数と同じ値で、
 * 表とは独立に書く）。
 */
export const WAN_TI2V_VAE_FIRST_CACHE_BYTES = 457_277_440;

export const ti2vVaeModelUrl = (component: string): URL =>
  new URL(`${component}/${WAN_TI2V_VAE_MODEL_FILE}`, WAN_TI2V_VAE_ROOT);

export const ti2vVaeFixtureUrl = (name: Ti2vVaeCase["name"]): URL =>
  new URL(`vae_chunks.${name}.safetensors`, WAN_TI2V_VAE_ROOT);

/**
 * 一組の資産（容器 2 本 + フィクスチャ 3 本 — 書き手は一組で据える）の在否。呼ぶたびに見る（モジュールの評価では
 * ファイルに触らない）。
 */
export const ti2vVaeAssets = (): readonly {
  readonly path: string;
  readonly present: boolean;
}[] => [
  ...[WAN_TI2V_VAE_FIRST, WAN_TI2V_VAE_NEXT].map((component) => {
    const url = ti2vVaeModelUrl(component);
    return { path: url.pathname, present: modelPresent(url) };
  }),
  ...WAN_TI2V_VAE_CASES.map(({ name }) => {
    const url = ti2vVaeFixtureUrl(name);
    return { path: url.pathname, present: filePresent(url) };
  }),
];

// ---- タイル参照（段 5 — `vae_tiles.<case>.safetensors`）------------------------------------------------

/** SKIP 時にそのまま貼れるタイル参照の生成コマンド（chunk グラフの組とは別に書く）。 */
export const WAN_TI2V_VAE_TILES_GENERATE =
  "cd tools/export-recipes && uv run --group wan --inexact " +
  "python -m wan.vae_tiling --model ti2v-5b";

/** タイル参照 1 本（帯の決定用 1 本 + 受入れ 2 本 — 2.1 のタイル参照と同じ組み立て）。 */
export type Ti2vVaeTileCase = {
  readonly name: "band" | "accept" | "wide";
  readonly role: "band" | "accept";
  readonly chunks: number;
  /** 潜在の空間の縦（出力の画素 = × 16）。 */
  readonly height: number;
  /** 潜在の空間の横（出力の画素 = × 16）。 */
  readonly width: number;
  /** 非タイルの参照 `frames_full`（patchify 空間・クランプ前 — 観測）を持つ。 */
  readonly full: boolean;
  /** 上流の unpatchify → クランプを当てた RGB `frames_rgb [3, F, H, W]` を持つ（VAE 段の末尾の照合）。 */
  readonly rgb: boolean;
};

/**
 * - `band`: 832×480×81（潜在 30×52・21 chunk）— 帯を決める 1 本（ADR 0121 の検収表の段 5）。
 * - `accept`: 縦長 480×832×9（潜在 52×30・3 chunk）— タイルの位置も chunk 境界も band と違う。
 * - `wide`: 1280×704×5（潜在 44×80・2 chunk）— 対ごとにブレンド幅が違うのはこの寸法だけ。
 */
export const WAN_TI2V_VAE_TILE_CASES: readonly Ti2vVaeTileCase[] = [
  { name: "band", role: "band", chunks: 21, height: 30, width: 52, full: false, rgb: false },
  { name: "accept", role: "accept", chunks: 3, height: 52, width: 30, full: true, rgb: true },
  { name: "wide", role: "accept", chunks: 2, height: 44, width: 80, full: false, rgb: false },
];

export const ti2vVaeTileFixtureUrl = (name: Ti2vVaeTileCase["name"]): URL =>
  new URL(`vae_tiles.${name}.safetensors`, WAN_TI2V_VAE_ROOT);

/** タイル参照 3 本の在否（chunk グラフの組とは別の組）。呼ぶたびに見る（モジュールの評価ではファイルに触らない）。 */
export const ti2vVaeTileAssets = (): readonly {
  readonly path: string;
  readonly present: boolean;
}[] =>
  WAN_TI2V_VAE_TILE_CASES.map(({ name }) => {
    const url = ti2vVaeTileFixtureUrl(name);
    return { path: url.pathname, present: filePresent(url) };
  });
