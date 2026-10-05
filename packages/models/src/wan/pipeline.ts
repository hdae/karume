/**
 * `WanPipeline` — テキスト → 動画（Wan2.1 T2V 1.3B）の 1 本の面（ADR 0118 決定 7）。
 *
 * パイプライン（全段 Karume・torch 不使用）:
 *
 * 1. **text** — 経路は構築時に選ぶ（{@link WanPipelineOptions.textEncoder} — ADR 0119 決定 7・追記
 *    「段 10d の設計」B）:
 *    - `"gpu"`（既定）— umT5（重み i8 per-channel・活性 f32 — ADR 0119 決定 5）の Session を張り、
 *      positive と negative を 1 回ずつ回して畳む。プロンプトは `prompt_clean` の鏡像とトークナイザの門を
 *      通る任意の文字列（2〜512 トークン — ADR 0119 決定 1・2・4）。出力 `[1, L, 4096]` をホストが
 *      DiT の文脈の行数までゼロで詰める
 *    - `"precomputed"` — テキスト埋め込み資産からプロンプトの行を引くだけ（ADR 0118 決定 4 の第 1 段・
 *      GPU を使わない）。資産の集合に無い文字列は fail loudly
 * 2. **transformer** — S 形の DiT（`[1,S,64]` — 決定 3）を steps 回。CFG は uncond → cond の逐次 2 回
 *    （B = 1）で、合成 `uncond + g·(cond − uncond)` と UniPC の更新はホスト（決定 5 — `scheduler.ts`）
 * 3. **vae_decoder** — chunk グラフ 2 本（first / next）の**常時タイル** decode（決定 2 —
 *    `vae-tiles.ts`）→ `[-1, 1]` へクランプ
 *
 * 返すのはフレーム `[3, F, H, W]` の f32（値域 `[-1, 1]`）と fps（{@link GeneratedVideo}）。PNG への
 * 書き出しは呼び手（`wanFrameToRgba` + `encodePng`）。
 *
 * 段の本体は Wan2.1 / 2.2 の class が共有するモジュールにある（ADR 0121 決定 10 — 入口の門は
 * `plan.ts`・text 段は `text-stage.ts`・DiT 段は `dit-loop.ts`・VAE 段は `tile-decode.ts`）。家族
 * admission・構築・段の順序の本体も共有の `family.ts`（世代の値を `WanFamilySpec` で受ける 1 本）にあり、
 * ここは公開型と Wan2.1 の class の殻（直列化鎖・`dispose`）を持ち、`WAN21_FAMILY` を渡す。
 *
 * ## MUST: 段ごとに Session を張って畳む・text → DiT → VAE の順に 1 段ずつ
 *
 * 構築（{@link WanPipeline.fromPretrained} / {@link WanPipeline.fromAssets}）では Session を 1 本も
 * 張らない（コンテナを開くまで）。
 * `generate` の中で段ごとに Session を張り、畳んでから次の段を張る（決定 7 — anima の既定
 * `"per-stage"` と同じ）。text 段（umT5 の重み i8 5.30 GiB）と DiT 段（81 フレームで 7.31 GiB）の和は
 * B570 の天井 9,600 MiB を越える（ADR 0119 決定 11）ので、text 段は DiT 段を張る前に畳む。DiT 段
 * （33 フレームで約 5.7 GiB）と VAE 段（約 2.9 GiB）も同時に持たない。text 段を畳んだ直後の確保の残り
 * （DiT の後には初回だけ見えた）は未測（ADR 0119 未解決）。
 *
 * NOTE（解放待ちは入れていない — 決定 7 の「切り替えのピークを測り、足りなければ入れる」の結果）:
 * B570（Intel / wgpu）は `destroy()` の解放が次の device poll まで遅れうる（docs/known-issues.md
 * 「Intel Arc B570」節）が、2026-10-02 の実測（832×480・33 フレーム・2 ステップの通し・fdinfo の
 * `drm-total-vram0`）では、DiT 段の山 5.75 GiB が **DiT の Session を畳んだ直後**（`stage` の `end`・
 * VAE の段を張る前）に 0.43 GiB（最初の段の前 0.19 GiB）へ戻っていた。切り替えの山は DiT 段の山
 * そのもの（VAE 段の山 2.88 GiB と重ならない）。遅れが出ないのは、`Session.dispose` が未完了の仕事を
 * 決着させてから破棄し、破棄の時点で GPU が参照中のバッファが残らないため（推測 — 遅れの実測例は
 * 参照中のバッファを捨てる形だった）。
 *
 * MUST: 段の切り替えは公開 API 側でも守る — `generate` は直列化鎖に載せ（並行呼び出しは待たされて
 * 順に走る）、`dispose` はその完了を待ってから GPU を破棄する。
 *
 * ## 中断（`signal` — ADR 0119 決定 9・追記「段 10d の設計」G）
 *
 * 構築は {@link WanPipelineOptions.signal}（取得層と構築の段の境目 — anima / irodori と同じ形）、生成は
 * {@link WanGenerateRequest.signal}（段の境目・各 run の前〈text の 2 回・DiT の各 step〉・VAE のタイルの
 * 間 — `generation/sequence.ts` の形）。検査はイベントループへ 1 度譲ってから見る（`settleAbort`）。
 * 1 回の run は不可分で、中断は次の境目で効く。中断の例外は開いている Session を畳んでから
 * `signal.reason` を包まずに投げる（消費側が `error === controller.signal.reason` で識別できる —
 * ADR 0083 決定 5）。
 *
 * ## 配布形（`karume-wan2.1` — ADR 0118 段 7）
 *
 * 入口は配布形から取得する {@link WanPipeline.fromPretrained} と、取得済みのバイト列（manifest +
 * 資産）から組む {@link WanPipeline.fromAssets} の 2 つで、どちらも同じ家族 admission と組み立て
 * （`family.ts`）を通る。生成の既定（step 数・guidance・shift）は manifest の `pipelineConfig`
 * （`config.ts`）が持ち、UniPC の構造は上流の値（`WAN_UNIPC_CONFIG`）のまま。
 *
 * NOTE: 公開配布リポの対応表（`WAN_SOURCES`）はまだ無い — 公開リポを持たない家族は表を持たない
 * （ADR 0073 決定 1・vowel-detector と同じ）。HF 公開の pin 焼き込みの回に足す（docs/release-runbook.md）。
 *
 * ## MUST: 出力の「正しさ」はここでは担保されない
 *
 * 数値の正は参照との e2e（実 GPU — `packages/models/tests/e2e_wan_pipeline_test.ts`）が担保する。
 * seed 付き乱数は torch の `randn` とは別列（`random.ts`）なので、参照との照合は初期ノイズを
 * {@link WanGenerateRequest.latents} で注入して行う。
 */

import type { GpuContext, SessionDiagnostics } from "@karume/runtime";
import type { DistributionSource, HubRepoRef, Manifest } from "@karume/hub";

import { createOperationChain } from "../concurrency/serial.ts";
import type {
  FromPretrainedComponentOptions,
  FromPretrainedHubOptions,
} from "../hub/load-options.ts";
import {
  generateWanVideo,
  loadWanFromAssets,
  loadWanFromPretrained,
  WAN21_FAMILY,
  type WanState,
} from "./family.ts";
import type { WanPrompt } from "./text-embeds.ts";

/** 生成結果。`data` は `[3, frames, height, width]` の f32（値域 `[-1, 1]` — クランプ済み）。 */
export type GeneratedVideo = {
  readonly frames: number;
  readonly width: number;
  readonly height: number;
  /**
   * 出力のフレームレート（上流の世代の事実 — Wan2.1 は 16。manifest の宣言ではなくノブでもない —
   * ADR 0121 決定 8）。
   */
  readonly fps: number;
  readonly data: Float32Array<ArrayBuffer>;
};

/** `denoise-step` の `copyLatents()` が返す途中の潜在の写し（`[C, F', H/8, W/8]`）。 */
export type WanLatentSnapshot = {
  readonly data: Float32Array<ArrayBuffer>;
  readonly shape: readonly number[];
};

/**
 * {@link WanPipelineOptions.onRunDiagnostics} が受けるコンポーネント名（Session 1 本 = 1 名）。
 * `text_encoder` は `"gpu"` の経路の umT5（positive / negative の 1 回ずつ）。
 */
export type WanRunComponent =
  | "text_encoder"
  | "transformer"
  | "vae_decoder_first"
  | "vae_decoder_next";

/** {@link WanGenerateRequest.onEvent} が受ける生成イベント。 */
export type WanGenerateEvent =
  /**
   * 段の開始（`start` — Session を張る前）と終了（`end` — Session を畳んだ後）。
   * `text_encoder` は `"gpu"` の経路の umT5 の段（`"precomputed"` では出ない）。`vae_decoder` は
   * chunk グラフ 2 本の段。途中で落ちたら `end` は出ない。
   */
  | {
    readonly kind: "stage";
    readonly component: "text_encoder" | "transformer" | "vae_decoder";
    readonly at: "start" | "end";
  }
  | {
    readonly kind: "denoise-step";
    /** 完了した step 数（1 始まり）。 */
    readonly step: number;
    readonly steps: number;
    /** その step で DiT へ渡した timestep（整数）。 */
    readonly timestep: number;
    /** 呼んだときだけ途中の潜在を写して返す（UniPC の更新の後の潜在）。 */
    readonly copyLatents: () => WanLatentSnapshot;
  }
  /** VAE のタイル 1 枚の decode の完了（`tile` は 1 始まり）。 */
  | { readonly kind: "vae-tile"; readonly tile: number; readonly tiles: number };

/**
 * 1 回の生成要求。省いた step 数・guidance・shift は manifest の `pipelineConfig` の既定（配布形の値は
 * 参照の設定 — 50 ステップ・guide 5.0・shift 3.0）、寸法とフレーム数は 832×480・33 フレーム。
 */
export type WanGenerateRequest = {
  /**
   * プロンプト。受理集合は構築時の経路（{@link WanPipelineOptions.textEncoder}）で決まる:
   *
   * - `"gpu"`（既定）: 任意の文字列。上流の前処理（`prompt_clean`）の鏡像とトークナイザの門を通り、
   *   通らない入力は `ModelInputError`（ADR 0119 決定 1・2・4）— 語彙外の文字・Unicode 16.0.0 で
   *   未割り当ての文字・C1 制御文字・HTML の文字参照になりうる並び（`R&D` は `R & D` のように空白を挟めば
   *   通る）・文字化けに見える並び・本文中の特殊トークン・空白の直後の `▁`・空や空白だけ（1 トークン）・
   *   512 トークン超。拒否の文言は直し方まで言う。
   * - `"precomputed"`: **テキスト埋め込み資産の集合にある文字列だけ**（原文か正規化後の文字列の
   *   どちらかに完全一致 — {@link WanPipeline.prompts}）。集合の外は `ModelInputError`（ADR 0118
   *   決定 4）。
   */
  readonly prompt: string;
  /**
   * CFG の uncond 側のプロンプト（{@link WanGenerateRequest.prompt} と同じ受理集合）。省くと公式の
   * `sample_neg_prompt`（`"gpu"` は GPU で符号化した既定の文字列・`"precomputed"` は資産の `negative` の
   * 行 — 同じ原文）。`guidance` が 1（CFG を回さない）のときに渡すと `ModelInputError`（効かないノブを
   * 黙って受けない）。
   */
  readonly negativePrompt?: string;
  /** 初期ノイズの seed（既定 0 — {@link WanGenerateRequest.latents} とは排他）。 */
  readonly seed?: number;
  /**
   * 初期ノイズを外から渡す（`[16, F', H/8, W/8]` の f32・`F' = (frames − 1)/4 + 1`）。参照との照合で
   * torch の `randn` の列を注入する口（seed の生成器は torch とは別の列 — `random.ts`）。書き換えない。
   */
  readonly latents?: Float32Array<ArrayBuffer>;
  /**
   * denoise の step 数（1 以上・既定は `pipelineConfig.defaults.steps`）。`shift` との組で σ 列が
   * 狭義単調減少にならない値（数十万 step など）は `ModelInputError`。
   */
  readonly steps?: number;
  /**
   * CFG の強さ（1 以上で f32 に丸めても有限・既定は `pipelineConfig.defaults.guidance`）。1 なら
   * uncond 側を回さない（上流の `guidance_scale > 1` の判定と同じ）。1 未満は上流が CFG ごと切って値が
   * 効かないので `ModelInputError`。
   */
  readonly guidance?: number;
  /**
   * flow matching の shift（正・既定は `pipelineConfig.scheduler.shift`）。`steps` との組で σ 列が
   * 壊れる値（極端に大きい / 小さい shift）は `ModelInputError`。
   */
  readonly shift?: number;
  /** フレーム数（4n+1 の 5〜81・既定 33）。 */
  readonly frames?: number;
  /** 幅 × 高さ（832×480 か 480×832・既定 832×480）。 */
  readonly width?: number;
  readonly height?: number;
  /**
   * 生成イベントの観測席（{@link WanGenerateEvent}）。**await する**（発火の順が決定的になる）。
   * 例外は握らない — throw が step 粒度の中断になる（段の Session は畳まれる）。
   *
   * MUST: `onEvent` の中で同じパイプラインの `generate` / `dispose` を await しない（直列化鎖の
   * 自己デッドロック）。
   */
  readonly onEvent?: (event: WanGenerateEvent) => void | Promise<void>;
  /**
   * 生成の中断。段の境目（text → DiT → VAE）・各 run の前（text の positive / negative・DiT の各 step）・
   * VAE のタイルの間で、イベントループへ 1 度譲ってから検査する。1 回の run は不可分で、中断は次の
   * 境目で効く。中断したら開いている Session を畳んでから `signal.reason` を**そのまま**投げる（包まない
   * — `error === controller.signal.reason` で自分の中断を識別できる）。中断の後も同じパイプラインで
   * 次の `generate` を回せる。
   */
  readonly signal?: AbortSignal;
};

/** 構築オプション（{@link WanPipeline.fromAssets} / {@link WanPipeline.fromPretrained} 共通）。 */
export type WanPipelineOptions = {
  /** モデル（manifest の models のキー）。省略時は `defaultModel`。 */
  readonly model?: string;
  /** 実行構成（そのモデルの quants のキー）。省略時は `defaultQuant`。 */
  readonly quant?: string;
  /**
   * テキストエンコーダの経路（既定 `"gpu"` — ADR 0119 決定 7・追記「段 10d の設計」B）。取る部品が
   * 経路で変わるので構築時に決める（生成ごとには切り替えない）。
   *
   * - `"gpu"`: umT5（manifest の部品 `text_encoder` — 配布形では umT5 の配布リポへの越境参照・重み i8 で
   *   約 5.3 GiB）とトークナイザ資産（`umt5_tokenizer`）を取り、`generate` ごとに umT5 の段を張って畳む。
   *   任意のプロンプトを受ける。手元の配布形（`localDirectory` / `denoDirectory`）では、越境先を
   *   取得元の `crossRepo` の mapping で渡す（隣のディレクトリを推測しない — hub の `local.ts`）。
   * - `"precomputed"`: umT5 を取らず（1 バイトも読まない）、テキスト埋め込み資産のプロンプトだけを受ける
   *   軽い使い方。資産の埋め込みは bf16 の umT5 の出力で、`"gpu"`（i8 の重み・f32 の活性）とは値が
   *   一致しない — 同じ文字列でも経路を変えれば別の動画になる。
   *
   * 未知の綴りは素の `Error`（取得の前に落ちる）。
   */
  readonly textEncoder?: "gpu" | "precomputed";
  /**
   * 既存の GPU を共有する（渡した側が所有権を持つ — {@link WanPipeline.dispose} は破棄しない）。
   * 省くとパイプラインが `acquireGpu` し、`dispose` で破棄する。
   *
   * MUST: 計測（`acquireGpu({ gpuTiming: true })`）の device は渡せない — VAE の段は 1 タイル = 1 batch
   * で回り、runtime は計測の device で batch を開かない（`GpuContext.beginBatch` の MUST）。構築時に
   * 落とす（DiT の段を回し終えてから VAE の段で落ちる形にしない）。DiT の GPU 時間は DiT だけを回す
   * e2e（段 3）で測る。
   */
  readonly gpu?: GpuContext;
  /**
   * Session の診断の観測席（umT5 は 1 プロンプト = 1 回・DiT は 1 forward = 1 回・VAE は 1 タイル =
   * first / next の 1 回ずつ）。例外は握らない（fail loudly）。
   */
  readonly onRunDiagnostics?: (
    component: WanRunComponent,
    diagnostics: SessionDiagnostics,
  ) => void;
  /**
   * 構築の中断。{@link WanPipeline.fromPretrained} は同じ 1 本を取得層へも渡すので、取得（umT5 だけで
   * 約 5.3 GiB）と組み立てのどちらの最中でも効く。組み立ての段の境目（入口・容器を開いた後・資産の
   * 解析の前・GPU の取得の前後）でイベントループへ 1 度譲ってから検査する。中断の例外は
   * `signal.reason` を**そのまま**投げる（包まない — anima / irodori の構築と同じ形）。
   *
   * NOTE: 構築の `signal` は生成には持ち越さない（生成の中断は {@link WanGenerateRequest.signal}）—
   * 構築の寿命の値（`AbortSignal.timeout` など）が以後の生成を全部落とす形にしない。
   */
  readonly signal?: AbortSignal;
};

/**
 * {@link WanPipeline.fromPretrained} が追加で受ける取得層のオプション（hub へ透過する）。
 *
 * NOTE: `headers` / `fetch` / `caches` / `onRetry` が **HTTP 取得元専用**であることを含め、欄ごとの
 * 説明は {@link FromPretrainedHubOptions} に 1 本化してある。
 */
export type WanFromPretrainedOptions =
  & WanPipelineOptions
  & FromPretrainedHubOptions
  & FromPretrainedComponentOptions;

/**
 * 取得済みの manifest + 資産（hub の `fetchAssets` の返り値をそのまま渡せる形）。`assets` の部品は
 * 単一形 `krm` のキー（`transformer`）か part 列（`transformer[0]` / `transformer[1]` / … — part 0 から
 * 添字順）で、`transformer` / `vae_decoder_first` / `vae_decoder_next` の 3 本と、`"gpu"` の経路
 * （既定）では `text_encoder`（umT5）。加えて manifest の `assets` の `text_embeds`（埋め込み資産の
 * safetensors — 両方の経路）と、`"gpu"` の経路では `umt5_tokenizer`（トークナイザ資産の JSON）。
 */
export type WanAssets = {
  readonly manifest: Manifest;
  readonly assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>;
};

/**
 * Wan2.1 のテキスト → 動画パイプライン。
 *
 * 構築は {@link WanPipeline.fromPretrained}（配布形から取得）か {@link WanPipeline.fromAssets}（取得済み
 * バイト列）だけを入口にする（コンストラクタは private — manifest 検査と資産の突合を迂回した半端な
 * 状態を作らせない。ADR 0008）。
 */
export class WanPipeline {
  readonly #state: WanState;
  /** generate と dispose の直列化鎖。 */
  readonly #chain = createOperationChain();
  /** dispose の 1 本（undefined でないことが「dispose 済み」）。 */
  #disposal: Promise<void> | undefined;

  private constructor(state: WanState) {
    this.#state = state;
  }

  /**
   * 配布形から取得して組む（`loadManifest` → `resolveSelection` → **各部品の descriptor（part 0）
   * だけ**を取って admission → 重みの part を温める → 埋め込み資産・トークナイザ資産の `fetchAssets` →
   * 構築）。block は Session を組むその瞬間に part 順で読まれる（ADR 0109 — `src/hub/components.ts`）。
   * 取る部品は経路（{@link WanPipelineOptions.textEncoder}）で決まる — `"precomputed"` は umT5
   * （`text_encoder`）を取らない。部品を別リポの同じ役割で差し替えるときは
   * {@link FromPretrainedComponentOptions.components}（`transformer` / `vae_decoder_first` /
   * `vae_decoder_next`、`"gpu"` の経路では `text_encoder` も）。
   *
   * **`ref` は必須**（取得元に既定は無い — `src/hub/repo-ref.ts` の MUST。この家族は公開配布リポを
   * まだ持たないので pin 定数も無い）。文字列の `ref` は `{ repo }` と読む（= `main` 追従）。手元の
   * 配布形（`models/karume-wan2.1`）は**取得元ハンドル**で渡す（`localDirectory` / `@karume/hub/deno` の
   * `denoDirectory`）— HF の `owner/name` の綴りの門は通らず、network も CacheStorage も通らない
   * （{@link WanFromPretrainedOptions} の HTTP 専用ノブは効かない）。umT5 の越境先（umT5 の配布リポ）は
   * その取得元の `crossRepo` に mapping で渡す（例 `denoDirectory("models/karume-wan2.1", { crossRepo:
   * { "<owner>/karume-umt5-xxl": denoDirectory("models/karume-umt5-xxl") } })` — キーは manifest の
   * `text_encoder` が宣言する repo）。
   */
  static async fromPretrained(
    ref: string | HubRepoRef | DistributionSource,
    options: WanFromPretrainedOptions = {},
  ): Promise<WanPipeline> {
    return new WanPipeline(await loadWanFromPretrained(WAN21_FAMILY, ref, options));
  }

  /**
   * 取得済みの manifest + 資産から組む（{@link WanAssets}）。容器を開いて、取得面と同じ家族 admission と
   * 組み立て（`family.ts`）を通す — 2 面の違いは部品の供給口だけ。
   */
  static async fromAssets(
    input: WanAssets,
    options: WanPipelineOptions = {},
  ): Promise<WanPipeline> {
    return new WanPipeline(await loadWanFromAssets(WAN21_FAMILY, input, options));
  }

  /**
   * テキスト埋め込み資産のプロンプトの一覧（資産のメタの並び — 埋め込みの値は持たない写し）。
   * 経路で意味が変わる:
   *
   * - `"precomputed"`: **受理集合そのもの**。`prompt` / `negativePrompt` には各行の `prompt` か
   *   `normalized` を渡す。
   * - `"gpu"`（既定）: 例示（固定プロンプトの名前と原文）。受理集合は任意の文字列で、各行の `prompt` を
   *   渡すと GPU の umT5 で符号化する（資産の埋め込みは使わない）。
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
   * 並行に呼ばれた場合は待たされて順に走る（グラフの同時常駐を作らない — モジュール doc）。
   */
  async generate(request: WanGenerateRequest): Promise<GeneratedVideo> {
    if (this.#disposal !== undefined) throw new Error("WanPipeline: dispose 済みでは生成できない");
    return await this.#chain(() => generateWanVideo(WAN21_FAMILY, this.#state, request));
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
