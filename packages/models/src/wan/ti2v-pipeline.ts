/**
 * `WanTi2vPipeline` — テキスト → 動画（Wan2.2 TI2V 5B の T2V）の 1 本の面（ADR 0121 決定 10）。
 *
 * 段の順序（text → DiT → VAE を 1 段ずつ張って畳む）・Session の寿命・中断・公開型
 * （`WanGenerateRequest` / `GeneratedVideo` / `WanGenerateEvent` / 構築オプション）は Wan2.1 の
 * `WanPipeline` と同じ 1 本で、正本は `./pipeline.ts` 冒頭の doc。家族 admission・構築・生成の本体も
 * 共有の `family.ts` にあり、ここは class の殻（private のコンストラクタ・直列化鎖・`dispose`）を持ち、
 * `WAN22_TI2V_FAMILY`（pipeline `wan-ti2v/1`・世代の記述子 `WAN22_TI2V_GENERATION`）を渡す。
 *
 * Wan2.1 との違いは全て世代の記述子と資産の宣言から来る（コードの分岐ではない）:
 *
 * - 受理集合は 1280×704 / 704×1280 × フレーム数 4n+1 の 5〜33（既定 1280×704・33 フレーム）
 * - 潜在は 48 チャネル（`[48, F', H/16, W/16]` — VAE のグラフの比 8 × unpatchify 2）・出力は 24 fps
 * - DiT の入力は 7 本（条件側の時刻と条件マスク — T2V では条件マスクは全て偽・条件側の時刻は生成側と同じ）
 *
 * NOTE: 殻の重複（約 60 行）は受け入れている — 共有の基底 class や mixin は作らない（ADR 0121 段 6 の設計
 * §3.2。本体は `family.ts` の 1 本で、重複するのは private 状態と委譲だけ）。
 *
 * NOTE: 配布形（`karume-wan2.2`）はまだ無い（ADR 0121 段 8）。それまでの入口は系列の容器から組んだ
 * manifest + 資産を渡す {@link WanTi2vPipeline.fromAssets}。
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
import type {
  GeneratedVideo,
  WanAssets,
  WanFromPretrainedOptions,
  WanGenerateRequest,
  WanPipelineOptions,
} from "./pipeline.ts";
import type { WanPrompt } from "./text-embeds.ts";

/**
 * Wan2.2 TI2V 5B のテキスト → 動画パイプライン。
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
   * 配布形から取得して組む（手順・取る部品・`ref` の読み方は `WanPipeline.fromPretrained` と同じ 1 本 —
   * `family.ts`）。manifest の pipeline は `wan-ti2v/1` を要る。
   *
   * NOTE: 配布形も、それを組む道具もまだ無い（モジュール doc — ADR 0121 段 8）。今この入口が意味を
   * 持つのは、manifest と容器を配布形の綴りどおりに手で並べたディレクトリを取得元ハンドル
   * （`localDirectory` / `@karume/hub/deno` の `denoDirectory`）で渡す形だけ。
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
   * プロンプトから動画を 1 本生成する。同じ要求（seed か latents・ノブ）なら同じ値が出る。
   * 並行に呼ばれた場合は待たされて順に走る（グラフの同時常駐を作らない — `./pipeline.ts` 冒頭の doc）。
   */
  async generate(request: WanGenerateRequest): Promise<GeneratedVideo> {
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
