/**
 * Wan の世代の記述子 — 世代で違い、グラフ宣言から導けない値を世代ごとに 1 か所へ集める
 * （ADR 0121 決定 10）。潜在のチャネル数・タイル辺・グラフの入出力の空間比のようにグラフ宣言から
 * 導ける値はここに置かない（資産の宣言から読む）。
 *
 * NOTE: `export` はパイプラインの内部と、テスト・gpu-lab のため（`mod.ts` / サブパス面には出さない —
 * ADR 0008）。
 */

import {
  WAN22_LATENTS_MEAN,
  WAN22_LATENTS_STD,
  WAN_LATENTS_MEAN,
  WAN_LATENTS_STD,
  type WanLatentStats,
} from "./latents.ts";

/**
 * DiT のグラフ入力の形（ADR 0121 決定 3）。`"t2v"` = Wan2.1 の 5 本（`tokens`・`timesteps_proj`・
 * `encoder_hidden_states`・`rope_cos`・`rope_sin`）、`"ti2v"` = Wan2.2 TI2V の 7 本（5 本 +
 * `timesteps_proj_condition`・`condition_mask`）。
 */
export type WanDitInputForm = "t2v" | "ti2v";

export type WanGenerationDescriptor = {
  /**
   * VAE の前の逆正規化の統計（VAE の config にしか無い — IR にも配布資産にも入っていない・表は
   * `latents.ts`）。
   */
  readonly latents: WanLatentStats;
  /** 受理する寸法（要求の門と、全数をタイルで覆えるかを見る家族 admission の門が使う）。 */
  readonly acceptedSizes: readonly { readonly width: number; readonly height: number }[];
  /** 受理するフレーム数の範囲（両端を含む・4n+1 だけ — VAE の時間圧縮）。 */
  readonly minFrames: number;
  readonly maxFrames: number;
  /**
   * 要求が省いた寸法とフレーム数（受理集合の中）。配布形は宣言しない — step 数・guidance・shift の
   * 既定は manifest の `pipelineConfig`（`config.ts`）。
   */
  readonly defaults: { readonly frames: number; readonly width: number; readonly height: number };
  /** 出力のフレームレート（上流の世代の事実で、ノブではない — manifest に入れない・ADR 0121 決定 8）。 */
  readonly fps: number;
  /**
   * 上流の VAE の `patch_size`（ホストの unpatchify の倍率・無しは 1 — ADR 0121 決定 6）。空間の圧縮
   * （グラフの入出力の空間比 × この倍率）の因子。出口が RGB である前提なら VAE の出口のチャネル数
   * （3·p²）からも導けるが、上流の事実として持ち、admission で宣言との照合に使う
   * （`tile-decode.ts` の `assertWanVaeMatchesGeneration`）。
   */
  readonly vaePatchSize: number;
  /**
   * 上流の DiT の入力の形（TI2V の DiT は条件側の時刻と条件マスクを持つ）。`vaePatchSize` と同じく上流の
   * 事実として持ち、admission でグラフ宣言との照合に使う（`dit-loop.ts` の `ditContract`）。
   */
  readonly ditInputForm: WanDitInputForm;
};

/**
 * Wan2.1（T2V 1.3B）の記述子。
 *
 * 受理集合（ADR 0118 決定 7 — 832×480 / 480×832 × フレーム数 4n+1 の 5〜81）。検収したのは
 * 832×480 の 33 フレーム（段 3 / 5 / 6）と 81 フレーム（段 8）、480×832 の VAE（段 5）。
 *
 * 81 フレームの可否は DiT 段単独の VRAM だけで決まる — `pipeline.ts` の冒頭の NOTE の実測どおり DiT の
 * Session を畳んだ直後に確保が戻り、DiT 段と VAE 段は重ならない（切り替えの山 = DiT 段の山）。段 8 の
 * 実測（B570・81 フレーム 50 ステップの通し・fdinfo）で DiT 段の山は 7.31 GiB・VAE 段の山は 3.78 GiB
 * （ADR 0118「段 8 の結果」）。MUST: 拒む文言は上限（81）だけを言う（利用者に段の番号は意味を
 * 持たない）。
 *
 * 既定は最初の到達目標の 832×480・33 フレーム（受理集合の側の事実）。fps 16 と `patch_size` 無し（1）は
 * 上流の Wan2.1 の値。
 *
 * MUST: 受理集合を変えるときはモデルカード（`tools/export-recipes/wan/card.py` の `WAN_ACCEPTED_SIZES` /
 * `WAN_FRAMES`）と `tests/fixtures/wan-card-limits.json` も同じ値にする — カードは manifest に無い
 * この事実を写しで持つので、fixture を挟んだ両側のテスト（wan_pipeline_test.ts と recipe の
 * test_distribution.py）が片側だけの更新を赤にする。
 */
export const WAN21_GENERATION: WanGenerationDescriptor = {
  latents: { mean: WAN_LATENTS_MEAN, std: WAN_LATENTS_STD },
  acceptedSizes: [
    { width: 832, height: 480 },
    { width: 480, height: 832 },
  ],
  minFrames: 5,
  maxFrames: 81,
  defaults: { frames: 33, width: 832, height: 480 },
  fps: 16,
  vaePatchSize: 1,
  ditInputForm: "t2v",
};

/**
 * Wan2.2（TI2V 5B）の記述子（1280×704 / 704×1280 × フレーム数 4n+1 の 5〜33 だけ — ADR 0121 の
 * 「追記（2026-10-05）: 受理寸法を公式の 2 寸法へ」）。
 *
 * 受理寸法は公式実装の対応寸法（Wan2.2 の `SUPPORTED_SIZES` — 公式が 720P と呼ぶ 2 向き）だけ。pin した
 * Diffusers のパイプラインの既定 832×480 は公式実装の対応寸法の外で、実際に生成して視認すると 3 本（実用席
 * shift 3・参照席 shift 3・実用席 shift 5）とも絵が崩れた（太い輪郭線とベタ塗り）。1280×704 は実用席・shift 5
 * （同じ seed・プロンプト）で写実に出た。よって 832×480 / 480×832 は受理しない。段 5 の 832×480 系のタイル
 * 参照は受理寸法に依らない数値の照合として残る（タイル計画の入口 `planWanGenerationTiles` は受理集合を見ない）。
 *
 * 上限 33 フレームは開発機（B570・Deno の総確保の天井 約 9.4 GiB〈9,600 MiB〉）で 1280×704・50 ステップの
 * 完走を確かめた値。開発機での上限の引き上げ（49 / 57 フレーム）は実測を見て決める。公式実装の既定 121
 * フレームは、大きい GPU で完走を確かめてから広げる。
 *
 * 既定の 1280×704・33 フレームは仮置き（視認で確定する）。fps 24 と `patch_size` 2 は上流の Wan2.2 の値
 * （逆正規化の統計の表は `latents.ts`）。
 */
export const WAN22_TI2V_GENERATION: WanGenerationDescriptor = {
  latents: { mean: WAN22_LATENTS_MEAN, std: WAN22_LATENTS_STD },
  acceptedSizes: [
    { width: 1280, height: 704 },
    { width: 704, height: 1280 },
  ],
  minFrames: 5,
  maxFrames: 33,
  defaults: { frames: 33, width: 1280, height: 704 },
  fps: 24,
  vaePatchSize: 2,
  ditInputForm: "ti2v",
};
