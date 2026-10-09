/**
 * `WanTi2vPipeline` — テキスト（+ 条件画像）→ 動画（Wan2.2 TI2V 5B の T2V と I2V）の 1 本の面（ADR 0121 決定 10・11）。
 *
 * 段の順序（(encoder →) text → DiT → VAE を 1 段ずつ張って畳む）・Session の寿命・中断・公開型
 * （`WanGenerateRequest` / `GeneratedVideo` / `WanGenerateEvent` / 構築オプション）は Wan2.1 の
 * `WanPipeline` と同じ 1 本で、正本は `./pipeline.ts` 冒頭の doc。家族 admission・構築・生成の本体も
 * 共有の `family.ts` にあり、ここは class の殻（private のコンストラクタ・直列化鎖・`dispose`）と 2.2 だけの
 * 要求の型（{@link WanTi2vGenerateRequest} — 条件画像）を持ち、`WAN22_TI2V_FAMILY`（pipeline `wan-ti2v/1`・世代の
 * 記述子 `WAN22_TI2V_GENERATION`）を渡す。
 *
 * Wan2.1 との違いは全て世代の記述子と資産の宣言から来る（コードの分岐ではない）:
 *
 * - 受理集合は 1280×704 / 704×1280 × フレーム数 4n+1 の 5〜121（既定 1280×704・33 フレーム）
 * - 潜在は 48 チャネル（`[48, F', H/16, W/16]` — VAE のグラフの比 8 × unpatchify 2）・出力は 24 fps
 * - DiT の入力は 7 本（条件側の時刻と条件マスク — T2V では条件マスクは全て偽・条件側の時刻は生成側と同じ・I2V では
 *   先頭の潜在フレームのトークンが真で条件側の時刻は 0）
 * - I2V の VAE encoder（記号長の 3 グラフ — `vae-encoder.ts`）の部品を持つ（DiT の入力の形が `"ti2v"` の世代だけ）
 *
 * NOTE: 殻の重複（約 60 行）は受け入れている — 共有の基底 class や mixin は作らない（ADR 0121 段 6 の設計
 * §3.2。本体は `family.ts` の 1 本で、重複するのは private 状態と委譲だけ）。
 *
 * 配布形は `karume-wan2.2`（ADR 0121 段 8 — recipe `dist.py --pipeline wan-ti2v` が組む・HF には未公開）。入口は 2.1 と
 * 同じく配布形から取得する {@link WanTi2vPipeline.fromPretrained} と、取得済みのバイト列から組む
 * {@link WanTi2vPipeline.fromAssets} の 2 つ。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 */

import type { DistributionSource, HubRepoRef } from "@karume/hub";

import { createOperationChain } from "../concurrency/serial.ts";
import {
  generateWanVideo,
  loadWanFromAssets,
  loadWanFromPretrained,
  WAN22_TI2V_FAMILY,
  type WanState,
} from "./family.ts";
import type { Rgb8Image } from "../image/preprocess.ts";
import type {
  GeneratedVideo,
  WanAssets,
  WanFromPretrainedOptions,
  WanGenerateRequest,
  WanPipelineOptions,
} from "./pipeline.ts";
import type { WanI2vFit } from "./i2v-preprocess.ts";
import type { WanPrompt } from "./text-embeds.ts";

/**
 * Wan2.2 の 1 回の生成要求（{@link WanGenerateRequest} + I2V の条件画像）。`image` を省けばテキスト → 動画（T2V —
 * 画像の欄が無い要求と同じ経路・同じ値）。
 *
 * `image` を渡すと I2V（ADR 0121 決定 11 — 条件づけは diffusers の `WanImageToVideoPipeline` の `expand_timesteps` の形）:
 *
 * - 出力寸法: `width` / `height` を省けば、受理集合（1280×704 / 704×1280）から画像の縦横比に近い方を選ぶ（公式 Wan2.2 の
 *   比較式 `max(r / rc, rc / r)`・同点は横長 — 幅 ≥ 高さの画像は 1280×704）。片方か両方を明示したら、それに合う受理集合の
 *   寸法（無ければ `ModelInputError`）。公式は受理集合の外の寸法（16:9 の画像で 1248×704 など）も選ぶので、同じ画像でも
 *   公式よりクロップが多くなりうる。
 * - 前処理は RGB8 のまま（decode は呼び手 — 画素はそのまま使い、色空間の変換はしない）→ 寸法合わせ（`fit`）→
 *   `[-1, 1]` → VAE encoder（`vae_encoder` の段 — 最初の段）→ 潜在の正規化 → DiT の先頭の潜在フレームの条件。
 * - 出力の 1 枚目は条件画像を VAE で encode → decode した絵になる（先頭の潜在フレームが条件の潜在そのもの）。
 */
export type WanTi2vGenerateRequest = WanGenerateRequest & {
  /**
   * I2V の条件画像（RGB8・インターリーブ・`data.length = 3·width·height`）。寸法は任意（出力寸法へ合わせる）。
   * パイプラインは書き換えない。呼び手も `generate` が決着するまで `data` を書き換えない（画素は直列化鎖で順番が
   * 来てから読むので、待っている間の書き換えは生成に効く）。RGB8 の形でない値（`data` が `Uint8Array` でない・
   * `null` など）と、寸法と長さの食い違いは `ModelInputError`。
   */
  readonly image?: Rgb8Image;
  /**
   * 条件画像の寸法の合わせ方（`image` を渡したときだけ — 省けば `"crop"`）。
   *
   * - `"crop"`: 公式 Wan2.2 の I2V と同じ — 縦横比を保って出力を覆う大きさへ LANCZOS で伸縮し、中央を切り出す
   * - `"stretch"`: diffusers の `WanImageToVideoPipeline` と同じ — 出力寸法へ直接 LANCZOS で伸縮する（縦横比は保たない）
   *
   * `image` 無しで渡すと `ModelInputError`（効かないノブを黙って受けない）。綴り違いは素の `Error`（model / quant 名の
   * 綴り違いと同じ扱い — 打つ手は値の範囲を直すことではなく受理集合を引き直すこと・ADR 0107 決定 3）で、`image` の
   * 有無より先に見る（`image` 無しの綴り違いも素の `Error`）。
   */
  readonly fit?: WanI2vFit;
};

/**
 * Wan2.2 TI2V 5B のテキスト（+ 条件画像）→ 動画パイプライン（T2V と I2V — {@link WanTi2vGenerateRequest}）。
 *
 * 構築は {@link WanTi2vPipeline.fromPretrained}（配布形から取得）か {@link WanTi2vPipeline.fromAssets}
 * （取得済みバイト列）だけを入口にする（コンストラクタは private — manifest 検査と資産の突合を迂回した
 * 半端な状態を作らせない。ADR 0008）。
 */
export class WanTi2vPipeline {
  readonly #state: WanState;
  /** generate と dispose の直列化鎖。 */
  readonly #chain = createOperationChain();
  /** dispose の 1 本（undefined でないことが「dispose 済み」）。 */
  #disposal: Promise<void> | undefined;

  private constructor(state: WanState) {
    this.#state = state;
  }

  /**
   * 配布形から取得して組む（手順・`ref` の読み方は `WanPipeline.fromPretrained` と同じ 1 本 — `family.ts`）。取る部品は
   * 2.1 の部品に I2V の VAE encoder の 3 本（`vae_encoder_pre` / `vae_encoder_attn` / `vae_encoder_post` — 画像を渡さない
   * 使い方でも取る）を足したもの。manifest の pipeline は `wan-ti2v/1` を要る。
   *
   * NOTE: 配布形（`karume-wan2.2`）は HF に未公開で pin 定数も無い。手元のミラー（`models/karume-wan2.2`）は取得元
   * ハンドル（`localDirectory` / `@karume/hub/deno` の `denoDirectory`）で渡し、`"gpu"` の経路では umT5 の越境先
   * （`karume-umt5-xxl`）をその取得元の `crossRepo` に mapping で渡す（キーは manifest の `text_encoder` が宣言する repo —
   * 2.1 と同じ形）。
   */
  static async fromPretrained(
    ref: string | HubRepoRef | DistributionSource,
    options: WanFromPretrainedOptions = {},
  ): Promise<WanTi2vPipeline> {
    return new WanTi2vPipeline(await loadWanFromPretrained(WAN22_TI2V_FAMILY, ref, options));
  }

  /**
   * 取得済みの manifest + 資産から組む（{@link WanAssets}）。容器を開いて、取得面と同じ家族 admission と
   * 組み立て（`family.ts`）を通す — 2 面の違いは部品の供給口だけ。
   */
  static async fromAssets(
    input: WanAssets,
    options: WanPipelineOptions = {},
  ): Promise<WanTi2vPipeline> {
    return new WanTi2vPipeline(await loadWanFromAssets(WAN22_TI2V_FAMILY, input, options));
  }

  /**
   * テキスト埋め込み資産のプロンプトの一覧（資産のメタの並び — 埋め込みの値は持たない写し）。
   * 経路ごとの意味は `WanPipeline.prompts` と同じ（`"precomputed"` は受理集合そのもの・`"gpu"` は例示）。
   */
  get prompts(): readonly WanPrompt[] {
    return this.#state.textEmbeds.entries.map(({ name, role, prompt, normalized }) => ({
      name,
      role,
      prompt,
      normalized,
    }));
  }

  /**
   * プロンプト（と条件画像 — {@link WanTi2vGenerateRequest}）から動画を 1 本生成する。同じ要求（seed か latents・
   * ノブ・画像）なら同じ値が出る。並行に呼ばれた場合は待たされて順に走る（グラフの同時常駐を作らない —
   * `./pipeline.ts` 冒頭の doc）。
   */
  async generate(request: WanTi2vGenerateRequest): Promise<GeneratedVideo> {
    if (this.#disposal !== undefined) {
      throw new Error("WanTi2vPipeline: dispose 済みでは生成できない");
    }
    return await this.#chain(() => generateWanVideo(WAN22_TI2V_FAMILY, this.#state, request));
  }

  /**
   * 解放する。**内部で取得した GPU だけ**破棄する（`options.gpu` は呼び手の所有物）。
   * MUST: in-flight の生成の完了を待ってから破棄する（flush-before-destroy — 鎖に載せる）。
   */
  dispose(): Promise<void> {
    this.#disposal ??= this.#chain(() => {
      if (this.#state.ownsGpu) this.#state.gpu.destroy();
    });
    return this.#disposal;
  }

  /** `await using` 対応 — {@link dispose} の別名。 */
  [Symbol.asyncDispose](): Promise<void> {
    return this.dispose();
  }
}
