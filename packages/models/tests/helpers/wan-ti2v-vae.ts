/**
 * What the tests of Wan2.2 TI2V-5B's VAE decoder chunk graphs share (ADR 0121 stage 4): the goldens of the f16 series
 * `wan2.2-ti2v-5b-f16-dyn` (fixtures, in the test-asset root — ADR 0139 decision 2), the table of fixture cases, the
 * expected contract values (including the cache table) and the fixtures' presence. The containers are the mirror's
 * (`models/karume-wan2.2/`, opened through the manifest with `openWanTi2vComponent` of `wan-ti2v-pipeline.ts` —
 * decision 1).
 *
 * GPU 不要の突き合わせ（`e2e_wan_ti2v_vae_chunks_host_test.ts`）と実 GPU の chunk 列の照合
 * （`e2e_wan_ti2v_vae_tiles_test.ts`）が同じ表を読む — 表を 2 か所で育てると、片方だけに足したケースが黙って
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

import { goldenDir } from "@karume/test-support/test-assets";
import { filePresent } from "./wan-ti2v-dit.ts";

/** The series the fixtures were exported with (their folder under `goldens/` of the test-asset root). */
const WAN_TI2V_VAE_SERIES = "wan2.2-ti2v-5b-f16-dyn";
/** The fixtures `vae_chunks.*` / `vae_tiles.*` / `vae_encoder.*` (the series' relative paths — decision 2). */
export const WAN_TI2V_VAE_GOLDEN_DIR = goldenDir(WAN_TI2V_VAE_SERIES);
/** Component names: the manifest's weights keys, which are also the containers' graph names (container-v1 §2.1). */
export const WAN_TI2V_VAE_FIRST = "vae_decoder_first";
export const WAN_TI2V_VAE_NEXT = "vae_decoder_next";

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
export const WAN_TI2V_VAE_TIME_CONV_CACHES: readonly string[] = WAN_TI2V_VAE_CACHE_TABLE.filter(
  ({ timeConv }) => timeConv,
).map(({ name }) => name);

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

export const ti2vVaeFixtureUrl = (name: Ti2vVaeCase["name"]): URL =>
  new URL(`vae_chunks.${name}.safetensors`, WAN_TI2V_VAE_GOLDEN_DIR);

/** The presence of the chunk fixtures `names` (all three by default). Checked on each call, never at module evaluation. */
export const ti2vVaeChunkFixtures = (
  names: readonly Ti2vVaeCase["name"][] = WAN_TI2V_VAE_CASES.map(({ name }) => name),
): readonly {
  readonly path: string;
  readonly present: boolean;
}[] =>
  names.map((name) => {
    const url = ti2vVaeFixtureUrl(name);
    return { path: url.pathname, present: filePresent(url) };
  });

// ---- タイル参照（段 5 — `vae_tiles.<case>.safetensors`）------------------------------------------------

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
  new URL(`vae_tiles.${name}.safetensors`, WAN_TI2V_VAE_GOLDEN_DIR);

/** タイル参照 3 本の在否（chunk グラフの組とは別の組）。呼ぶたびに見る（モジュールの評価ではファイルに触らない）。 */
export const ti2vVaeTileAssets = (): readonly {
  readonly path: string;
  readonly present: boolean;
}[] =>
  WAN_TI2V_VAE_TILE_CASES.map(({ name }) => {
    const url = ti2vVaeTileFixtureUrl(name);
    return { path: url.pathname, present: filePresent(url) };
  });

// ---- The 32-latent pair (ADR 0134 — weightless containers that borrow the 16-latent pair) ---------

/** The 32-latent pair's components (= component directories and graph names). */
export const WAN_TI2V_VAE_FIRST_32 = "vae_decoder_first_32";
export const WAN_TI2V_VAE_NEXT_32 = "vae_decoder_next_32";
/** The 32-latent pair's latent tile edge and frame edge (patchify space — 8 per latent). */
export const WAN_TI2V_VAE_TILE_32 = 32;
export const WAN_TI2V_VAE_SAMPLE_TILE_32 = 256;

/**
 * The tile references at tile 32 (ADR 0134 decision 6 — `vae_tiling.py --tile 32`). Only `wide`
 * (1280×704×5, latent 44×80 → 2×3 tiles) fits a 32-latent tile: `band` and `accept` are 30 latents
 * on one axis, so the writer skips them.
 */
export const WAN_TI2V_VAE_TILE32_CASES: readonly Ti2vVaeTileCase[] = WAN_TI2V_VAE_TILE_CASES.filter(
  ({ height, width }) => height >= WAN_TI2V_VAE_TILE_32 && width >= WAN_TI2V_VAE_TILE_32,
);

export const ti2vVaeTile32FixtureUrl = (name: Ti2vVaeTileCase["name"]): URL =>
  new URL(`vae_tiles.${name}.t32.safetensors`, WAN_TI2V_VAE_GOLDEN_DIR);

/**
 * Presence of the tile-32 references (none → explicit SKIP, some → FAIL — the convention of
 * {@link ti2vVaeTileAssets}). Checked on each call, never at module evaluation.
 */
export const ti2vVaeTile32TileAssets = (): readonly {
  readonly path: string;
  readonly present: boolean;
}[] =>
  WAN_TI2V_VAE_TILE32_CASES.map(({ name }) => {
    const url = ti2vVaeTile32FixtureUrl(name);
    return { path: url.pathname, present: filePresent(url) };
  });
