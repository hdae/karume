/**
 * `AnimaPipeline` — テキスト → 画像（Anima）の 1 本の面。
 *
 * パイプライン（全段 Karume・torch 不使用）:
 *
 * 1. トークナイザ 2 本（Qwen2 BPE / T5 Unigram）でプロンプトを id 列にする
 * 2. `text_encoder`（Qwen3）→ `text_conditioner` → 512 ゼロ埋め
 * 3. `transformer`（S 形 DiT）を N step 回し、ホスト側で CFG 合成 + 更新（更新則は request の
 *    `sampler`、省略時は manifest の `pipelineConfig.scheduler.type` — Euler / DPM++ 2M）
 * 4. latent を per-channel 逆正規化 → `vae_decoder` を**常時タイル**で通す
 * 5. RGBA 化して返す（PNG 化は `encodePng` — パイプライン非依存の共通処理）
 *
 * ## MUST: 既定ではグラフは 1 本ずつ開いて閉じる（ADR 0016 / 0112）
 *
 * 既定の `residency: "per-stage"` では解放の**位置**に意味がある。テキスト経路（重み 1,396MiB）を
 * DiT ロードの**前**に、DiT を VAE ロードの**前**に解放する。DiT の重みは quant 席で倍違う
 * （既定席 `f16+dit8-a8-attn8-s16` = 1,875MiB・`f16` 席 = 3,733MiB — 2026-08-05 final-perf-bench
 * の VRAM 表）。VAE を常時タイルにした後は**チェーン最大は DiT 段**で、既定の VRAM の前提は
 * 「最大の段 1 本ぶん」。
 * したがって {@link AnimaPipeline.fromAssets} は **Session を 1 本も張らない** —
 * 開くのはコンテナ（`openContainer` = 2 文書の解析のみ）までで、GPU 常駐は
 * {@link AnimaPipeline.generate} の中で段ごとに張っては畳む。
 *
 * opt-in の `residency: "transformer"` は DiT の Session だけを generate を跨いで持ち続ける
 * （状態機械は `residency.ts`）。text / VAE の段は常駐 DiT の**上に**乗るので、チェーン最大は
 * 必ず上がる（既定席 1024² で常駐ぶん +2,646MiB — ADR 0112）。そこで常駐 DiT がある時点で段を張る前に
 * （text 段の前と、DiT 段の後・VAE 段の前の 2 点）、次の段が要る量を試し確保（runtime の
 * `fitsHeadroom`）で量り、入らなければ常駐 DiT を**先に**退避する（OOM を踏む前 — 重みのアップロードの
 * OOM は device を失わせうる: `residency.ts` の「退避の 2 本の線」）。見積りが外れて段が `GpuOutOfMemoryError` を投げたときは
 * 常駐 DiT を退避してその段を 1 回だけやり直す（2 本目の線）。どちらの退避の後も、この pipeline は
 * 常駐しない（格下げ）。常駐 DiT も最初の generate の DiT 段で作る（構築時には張らない）。
 *
 * MUST: この段取りは**公開 API 側でも**守る — `generate` は直列化鎖に載せ（並行呼び出しは
 * 待たされて順に走る）、`dispose` はその完了を待ってから GPU を破棄する。載せないと、並行
 * 呼び出し 2 本ぶんのグラフが同時常駐して VRAM の前提が崩れ、生成中の dispose が
 * flush-before-destroy を破る。
 *
 * ## MUST: DiT は S 形・VAE は常時タイル（ADR 0038 §4）
 *
 * 配布される transformer は解像度を 1 つも持たない S 形だけで、VAE decoder は latent 64×64 の
 * 固定タイル 1 本だけ。したがって**資産が解像度から独立**し、非タイル経路も静的形の分岐も
 * 存在しない（512px は 1 タイルに縮退し、非タイル decode とビット同一 — ADR 0033 の門）。
 * 受理集合の正本は `resolution.ts` の定数。
 *
 * ## MUST: 出力画像の「正しさ」はここでは担保されない
 *
 * 数値の正は参照フィクスチャとの E2E（実 GPU）が担保する。seed 付き乱数は **torch の `randn`
 * とは別列**なので、自由生成した絵を torch と比べることはできない（`random.ts` の doc）。
 */

import {
  acquireGpu,
  fitsHeadroom,
  type GpuContext,
  type Session,
  type SessionDiagnostics,
  type SessionOptions,
  type Tensor,
} from "@karume/runtime";
import {
  type DistributionSource,
  type GpuFeaturesSpec,
  type HubRepoRef,
  loadManifest,
  type Manifest,
  type ModelEntry,
  type Quant,
  resolveSelection,
} from "@karume/hub";

import {
  ANIMA_PIPELINE_MAJOR,
  ANIMA_PIPELINE_NAME,
  type AnimaPipelineConfig,
  type AnimaSamplerType,
  assertAnimaSamplerType,
  parseAnimaPipelineConfig,
} from "./config.ts";
import { assertAcceptableSteps, cfgEulerStep, sigmaSchedule, timestepsProj } from "./sampler.ts";
import {
  type DpmSolverMultistepInput,
  dpmSolverMultistepStep,
  needsUncond,
} from "../generation/dpm-solver-multistep.ts";
import { animaLatents, denormalizeLatents, padSequence } from "./latents.ts";
import { imageToRgba } from "./image.ts";
import { assertAcceptableResolution, formatResolution, type ImageSize } from "./resolution.ts";
import { blendExtentAt, decodeTiled, planVaeTiling, tileCount } from "./tiling.ts";
import {
  ANIMA_SPATIAL_COMPRESSION,
  type DitPatchGeometry,
  ditPatchGeometry,
  patchifyLatents,
  ropeTables,
  tokenCount,
  unpatchifyTokens,
} from "./dit-tokens.ts";
import { parseRopeBase, type RopeBase, ropeWidth } from "./rope-base.ts";
import { type AnimaTokenizers, createTokenizers } from "./text/tokenizer.ts";
import { Randn } from "./random.ts";
import { ModelInputError } from "../errors.ts";
import { assertAcceptableSeed } from "../request-gates.ts";
import { settleAbort } from "../concurrency/abort.ts";
import { createOperationChain } from "../concurrency/serial.ts";
import {
  assertGpuFeaturesGranted,
  assertRequiredLimitsBeforeDownload,
  assertRequiredLimitsSatisfied,
  sessionGpuFeatures,
  toAcquireGpuOptions,
} from "../session/gpu-features.ts";
import { type FamilySessionPolicy, resolveSessionOptions } from "../session/options.ts";
import { disposeSteps } from "../session/dispose-steps.ts";
import { settleReleasedMemory } from "../session/settle-released-memory.ts";
import { toManifestSource } from "../hub/repo-ref.ts";
import {
  type FromPretrainedComponentOptions,
  type FromPretrainedHubOptions,
  hubLoadOptions,
} from "../hub/load-options.ts";
import {
  assetComponentOpener,
  type ComponentOpener,
  type GraphOwner,
  loadContainerComponents,
  type ModelComponent,
} from "../hub/components.ts";
import { readAssetBuffer, readWholeAsset } from "../hub/asset-readers.ts";
import {
  type AnimaResidency,
  type AnimaResidencyAction,
  type AnimaResidencyReason,
  assertAnimaResidency,
  DEFAULT_ANIMA_RESIDENCY,
  generateMemoryNeed,
  type ResidencyNotice,
  stageMemoryNeed,
  TransformerResidency,
} from "./residency.ts";

/** manifest の weights / assets 表に現れる名前（ADR 0041 §3 の規約名）。 */
const TEXT_ENCODER = "text_encoder";
const TEXT_CONDITIONER = "text_conditioner";
const TRANSFORMER = "transformer";
const VAE_DECODER = "vae_decoder";
const TOKENIZER = "tokenizer";
const TOKENIZER_2 = "tokenizer_2";

/** `transformer` の容器が宣言する rope 素表の資産名（役割 `rope-base` — ADR 0109 決定 4）。 */
const ROPE_BASE = "rope_base";

/** この系列の weights 部品（差し替え席が受ける役割名でもある）。 */
const COMPONENT_KEYS = [TEXT_ENCODER, TEXT_CONDITIONER, TRANSFORMER, VAE_DECODER] as const;

/** 生成結果。`data` は RGBA 8bit（4 バイト / 画素・行優先）。 */
export type GeneratedImage = {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8Array<ArrayBuffer>;
};

/** 1 回の生成要求。未指定の欄は manifest の `pipelineConfig.defaults` が埋める。 */
export type AnimaGenerateRequest = {
  readonly prompt: string;
  /**
   * ネガティブプロンプト。`guidanceScale === 1` では uncond 側を 1 度も計算しないので、
   * **指定すると fail loudly**（効かないノブを黙って受けない — {@link needsUncond}）。
   */
  readonly negativePrompt?: string;
  readonly steps?: number;
  readonly guidanceScale?: number;
  readonly resolution?: ImageSize;
  /** 初期ノイズの seed（既定 0 — 同じ seed なら同じ画像）。 */
  readonly seed?: number;
  /**
   * denoise の更新則（{@link AnimaSamplerType}）。省略時は manifest の
   * `pipelineConfig.scheduler.type`（公式配布は `"euler"`）。
   *
   * DPM++ 2M は**選択肢の一つ**であって配布既定ではない（再裁定 2026-08-25）。更新則は資産に
   * 依らないホスト側の式なので、**どちらの値も配布物によらず常に有効** — 同じ資産・同じ seed
   * でも値が違えば出る画素は変わる（`euler` の指定は manifest 既定が euler の配布物とビット
   * 同一）。
   */
  readonly sampler?: AnimaSamplerType;
  /**
   * **この generate の後に** DiT の Session を持ち続けるか（{@link AnimaResidency}）。省略時は
   * {@link AnimaPipelineOptions.residency}。
   *
   * 開始時に常駐 DiT が既にあれば、値に関わらずそれを使う（読み直さない）。連続生成では
   * 途中を `"transformer"`、**最後の 1 枚だけ `"per-stage"`** にすると、その generate の後に
   * 常駐 DiT が解放される（`residency` イベントの `released` / `request`）。
   * 退避（OOM / 空き不足）で格下げ済みの pipeline では `"transformer"` を求めても持たず、
   * `released` / `downgraded` を名乗る。未知の綴りは GPU に触る前に `ModelInputError`。
   */
  readonly residency?: AnimaResidency;
  /**
   * 生成イベントの観測席（{@link AnimaGenerateEvent}）— 段の開始 / 終了・denoise の 1 step
   * 完了・VAE タイル 1 枚の完了・DiT 常駐の変化ごとに呼ばれる。
   *
   * **await する**（発火の順序が決定的になり、消費側で間引き / スロットリングができる）。
   * **例外は握らない**（`onRunDiagnostics` と同じ流儀 = fail loudly）— 副産物として
   * **throw が step 粒度の中断手段**になる（生成は reject し、段の Session は解放される）。
   * 持ち越した常駐 DiT を捨てるのは denoise ループ内（`denoise-step`）の throw だけで、そのとき
   * `residency` の `evicted` / `failure` を名乗る。`stage` / `vae-tile` / `residency` の通知での
   * throw では常駐 DiT は健全なので捨てない（ただし実効値が `"per-stage"` なら手放して
   * `released` / `request` を名乗る — 失敗した generate でも「この後に持たない」指示は効く）。
   * これらの通知で元と同じ例外を投げ直しても、生成が投げるのは元の例外 1 本のまま。
   *
   * NOTE: 毎 step の VAE プレビューは提供しない。既定の段ごと運転では VAE は DiT を解放した
   * **後**にしかロードできない（VRAM の MUST — モジュール doc）ので、途中結果として渡せるのは生 latent
   * （`copyLatents`）だけ。プレビューは `approximatePreview`（`@karume/models/anima`）が
   * この latent から近似する。
   *
   * MUST: `onEvent` の中で同じパイプラインの `generate` / `dispose` を
   * await してはならない（直列化鎖の自己デッドロック — 中断は throw で行う）。
   */
  readonly onEvent?: (event: AnimaGenerateEvent) => void | Promise<void>;
};

/**
 * {@link AnimaPipelineOptions.onRunDiagnostics} が受けるコンポーネント名（Session 1 本 = 1 名）。
 * `stage` イベント（{@link AnimaGenerateEvent}）の段名も同じ 4 名。
 */
export type AnimaRunComponent =
  | "text_encoder"
  | "text_conditioner"
  | "transformer"
  | "vae_decoder";

/** `denoise-step` の `copyLatents()` が返す途中 latent の写し。 */
export type AnimaLatentSnapshot = {
  readonly data: Float32Array<ArrayBuffer>;
  /** latent の形 `[1,C,H,W]`（DiT の plan が決めた値）。 */
  readonly shape: readonly number[];
};

/** {@link AnimaGenerateRequest.onEvent} が受ける生成イベント。 */
export type AnimaGenerateEvent =
  /**
   * 段の開始（`start`）と終了（`end`）。段ごと運転では Session 構築の**前**と解放の**後**に
   * 当たり、GB 級ロードの進捗が見える。持ち越した常駐 DiT を実効値 `"transformer"` で使う
   * `transformer` 段では start → end にロードも解放も入らない（段の本体だけ）。実効値
   * `"per-stage"` で使う段では解放（手放し）が、OOM で退避した段では解放と段ごとのロードが
   * start → end の中に入る。途中で落ちたら `end` は出ない。
   */
  | { readonly kind: "stage"; readonly component: AnimaRunComponent; readonly at: "start" | "end" }
  | {
    readonly kind: "denoise-step";
    /** 完了した step 数（1-based）。 */
    readonly step: number;
    readonly steps: number;
    /** その step で消費した sigma。 */
    readonly sigma: number;
    /** 呼んだときだけ途中 latent を写して返す（{@link latentSnapshot}）。 */
    readonly copyLatents: () => AnimaLatentSnapshot;
  }
  /** VAE タイル 1 枚の decode 完了（`tile` は 1-based）。 */
  | { readonly kind: "vae-tile"; readonly tile: number; readonly tiles: number }
  /**
   * DiT の常駐の変化（ADR 0112）。既定の段ごと運転で常駐と無関係な generate では出ない。
   *
   * - `retained` / `request`: DiT をこの generate の後も持ち続ける（`transformer` 段の end の前）。
   * - `released` / `request`: 持ち越した常駐 DiT を、実効値 `"per-stage"` に従って手放した
   *   （DiT 段の end の前 — generate が DiT 段の前で失敗したときは、その失敗の後に出る）。
   * - `released` / `downgraded`: 常駐を求められたが、退避（`out-of-memory` / `headroom`）で格下げ済み
   *   なので持たない。
   * - `evicted` / `headroom`: 次の段が要る量が常駐 DiT の上に入らない（試し確保が OOM）ので、**段を
   *   張る前に**常駐 DiT を捨てた。出る場所は 2 つ — `text_encoder` の `stage` start より前（text_encoder /
   *   text_conditioner の必要量で量った — 持ち越した DiT がある generate）か、`transformer` の `stage` end
   *   と `vae_decoder` の `stage` start の間（vae_decoder の必要量で量った — その時点で DiT が席に載って
   *   いるとき）。以後この pipeline は常駐しない（格下げ — OOM の退避と同じく戻さない）。text 段の前で
   *   退避し、かつ実効値が `"transformer"` なら、この generate の DiT 段の end の前に `released` /
   *   `downgraded` が続く（実効値 `"per-stage"` なら DiT 段は段ごと運転で何も名乗らない）。VAE 段の前で
   *   退避したときは、DiT 段は既に `retained` / `request` を名乗っている。
   * - `evicted` / `out-of-memory`: 常駐 DiT がある状態で段が `GpuOutOfMemoryError` を投げたので
   *   常駐 DiT を捨て、**その段を 1 回だけ最初からやり直す**（DiT 段なら段ごと運転で —
   *   `denoise-step` / `vae-tile` はこの後 1 から出直す）。以後この pipeline は常駐しない。DiT 段で
   *   退避するのは前の generate から持ち越した DiT だけ（この generate で作った DiT の OOM は段ごと
   *   運転と同じ VRAM 構成なので、退避せずにそのまま投げる — ADR 0112 決定 3）。
   * - `evicted` / `failure`: 持ち越した常駐 DiT を使った DiT 段が失敗した（denoise ループ内の
   *   `onEvent` の throw を含む）ので常駐 DiT を捨てた。
   */
  | {
    readonly kind: "residency";
    readonly component: "transformer";
    readonly action: AnimaResidencyAction;
    readonly reason: AnimaResidencyReason;
  };

/**
 * 途中 latent を返す口を作る（**lazy copy** — 呼ばれたときだけ写す）。
 *
 * 進捗だけを購読する消費側にコピー費用が一切かからず、内部の配列を渡さないので「次 step の
 * 入力を購読側に握られる」事故も構造的に起きない。
 *
 * MUST: `data` だけでなく `shape` も写す。実引数は `plan.latentShape` の**素の可変配列**で、
 * 参照を渡すと購読側の書き換えが次 step の patchify / rope の対応を崩す（要素数は変わらない
 * ので末尾の「出た画像の寸法 == 要求解像度」検査も通り、黙って別物が出る）。
 *
 * MUST: 呼ばれた時点ではなく**作った時点**の配列を写す（引数で束縛する）。denoise ループの
 * `current` は step ごとに**新しい配列へ差し替わる**ので、この束縛がそのまま「その step の
 * latent」になる。ループ変数を閉じ込めると、後から呼んだ購読側に別 step の latent が返る。
 *
 * NOTE: `export` は GPU 無しで独立性を縛るテストのため（`mod.ts` / サブパス面には出さない —
 * ADR 0008）。
 */
export const latentSnapshot = (
  latents: Float32Array<ArrayBuffer>,
  shape: readonly number[],
): () => AnimaLatentSnapshot =>
(): AnimaLatentSnapshot => ({ data: new Float32Array(latents), shape: [...shape] });

/**
 * Anima が受ける実行ノブ（manifest の quant 宣言と明示指定の両方）。優先順位・値域・組合せ・
 * 送出型の分類は全家族共通の `resolveSessionOptions` が持ち、ここは受理集合だけを決める。
 *
 * linear / 融合 attention の実行形 3 欄を受け、並列 GEMV と融合の 4 欄は受けない — 配布形が
 * 宣言し参照値で確かめてあるのは前者だけで、後者（Gemma の decode 向けに足したノブ）は
 * Anima のグラフでは効く席も数値も確かめていない組合せだから。
 *
 * NOTE: `export` は同値テストがミラーの全 quant を同じ表で通すため（`mod.ts` / サブパス面には
 * 出さない — ADR 0008）。
 */
export const ANIMA_SESSION_POLICY: FamilySessionPolicy = {
  linearCompute: true,
  attentionCompute: true,
  attentionScoreStorage: true,
  linearGemvReduce: false,
  stateAttentionReduce: false,
  fuseRmsNormAdd: false,
  fuseLinearStaticQuantize: false,
  packedStaticQuantize: false,
};

/** 構築オプション（{@link AnimaPipeline.fromAssets} / {@link AnimaPipeline.fromPretrained} 共通）。 */
export type AnimaPipelineOptions = {
  /**
   * 既存の GPU を共有する。**渡した側が所有権を持つ**ので {@link AnimaPipeline.dispose} は
   * 破棄しない。省略時はパイプラインが内部で `acquireGpu` し、`dispose()` で破棄する。
   */
  readonly gpu?: GpuContext;
  /** モデル（manifest の models のキー）。省略時は `defaultModel`。 */
  readonly model?: string;
  /** 実行構成（そのモデルの quants のキー）。省略時は `defaultQuant`。 */
  readonly quant?: string;
  /**
   * `Session.run` 1 回ごとの診断を受け取る観測席（DiT は 1 step = 1 回・VAE は 1 タイル =
   * 1 回）。op 別 GPU 時間（`lastRunTiming`）が要るときは `gpu` に
   * `acquireGpu({ gpuTiming: true })` を渡す（ADR 0021 — 既定は計測しない）。
   * コールバックの例外は握らない（fail loudly — 生成ごと落ちる）。
   */
  readonly onRunDiagnostics?: (
    component: AnimaRunComponent,
    diagnostics: SessionDiagnostics,
  ) => void;
  /**
   * 構築の中断。{@link AnimaPipeline.fromAssets} が段の境目（入口 / トークナイザ解釈 /
   * 各 `openContainer` の間 / GPU 取得の前後）で検査する。入口を除く各境目では**イベントループへ
   * 1 度譲ってから**検査するので、同期解析の最中に届いた中断も次の境目で効く
   * （`options.gpu` を渡して await が 1 つも無い経路でも同じ）。
   * {@link AnimaPipeline.fromPretrained} は同じ 1 本を取得層へも渡すので、**DL と組み立ての
   * どちらの最中でも**同じノブで中断できる（DL 完了後だけ中止ボタンが無反応、を作らない）。
   *
   * 中断の例外は `signal.reason` を**そのまま**投げる（包まない — 消費側が
   * `error === controller.signal.reason` で自分の中断を識別できる）。
   */
  readonly signal?: AbortSignal;
  /**
   * linear の実行形を quant の宣言より優先して指定する（省略時は quant の `session` の宣言 →
   * runtime 既定の順 — ADR 0058 追記 2026-09-26）。効くのは **DiT（transformer）の Session だけ**で、
   * text 系と VAE は quant の宣言どおり（f32）に走る（quant 席の `session` と同じ適用範囲）。`"f16"` は device の shader-f16 を要し、
   * 自前で取る GPU には要求を足す。共有 GPU（`gpu`）が持たなければ重みを取る前に落ちる。
   * 不正な値・組合せは `ModelInputError`（ADR 0107）。
   */
  readonly linearCompute?: SessionOptions["linearCompute"];
  /**
   * 融合 attention の実行形を quant の宣言より優先して指定する（優先順位と shader-f16 の扱いは
   * {@link AnimaPipelineOptions.linearCompute} と同じ）。`attentionScoreStorage: "f16"` との
   * 同時指定は runtime が受けない組合せなので `ModelInputError`。
   */
  readonly attentionCompute?: SessionOptions["attentionCompute"];
  /**
   * 融合 attention の S の格納形を quant の宣言より優先して指定する（優先順位は
   * {@link AnimaPipelineOptions.linearCompute} と同じ・shader-f16 は要らない）。
   */
  readonly attentionScoreStorage?: SessionOptions["attentionScoreStorage"];
  /**
   * generate の後に DiT の Session を持ち続けるかの既定（{@link AnimaResidency}・省略時
   * `"per-stage"` = 段ごとに張って畳む従来の挙動）。generate ごとに
   * {@link AnimaGenerateRequest.residency} で上書きできる。
   *
   * `"transformer"` は 2 回目以降の generate で DiT の読み直し・計画の導出・中間バッファの作り直しを
   * 消す代わりに、text / VAE の段が常駐 DiT の上に乗る（既定席 1024² で +2,646MiB・4 GB 級の
   * GPU では使えない — docs/limitations.md）。VRAM が足りずに OOM したら常駐 DiT を退避して
   * その段をやり直し、以後は常駐しない（`residency` イベントで名乗る — ADR 0112）。
   * 構築時には Session を張らない（常駐 DiT は最初の generate の DiT 段で作る）。
   * 未知の綴りは重みを取る前に `ModelInputError`。
   */
  readonly residency?: AnimaResidency;
};

/**
 * {@link AnimaPipeline.fromPretrained} が追加で受ける取得層のオプション（hub へ透過する）。
 * `signal` は構築側と共有なので {@link AnimaPipelineOptions} が持つ。
 *
 * NOTE: `headers` / `fetch` / `caches` / `onRetry` が **HTTP 取得元専用**であることを含め、
 * 欄ごとの説明は {@link FromPretrainedHubOptions} に 1 本化してある。
 */
export type AnimaFromPretrainedOptions =
  & AnimaPipelineOptions
  & FromPretrainedHubOptions
  & FromPretrainedComponentOptions;

/** 取得済み資産から直接組むときの入力（hub の `fetchAssets` の返り値をそのまま渡す）。 */
export type AnimaAssets = {
  readonly manifest: Manifest;
  readonly assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>;
};

/**
 * 取得済みバイト列を `openContainer` へ渡せる ArrayBuffer にする（門の本体は
 * {@link readAssetBuffer}）。
 *
 * MUST: `slice` で写さない — DiT は 1 本 3.7GiB あり、ホスト RAM のピークが倍になる。
 */
const assetBuffer = (
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  key: string,
): ArrayBuffer => readAssetBuffer("anima", "weights / assets", assets, key);

/**
 * 全量面（`fromAssets`）のコンポーネント供給口（受け口の実装は 8 家族共有 —
 * {@link assetComponentOpener}）。部品のキーは単一形 `krm` の 1 本（`transformer`）か、
 * 分割形の part 列（`transformer[0]` / `transformer[1]` / …）。
 */
const assetOpener = (assets: AnimaAssets["assets"]): Promise<ComponentOpener> =>
  assetComponentOpener("anima", assets, (key) => assetBuffer(assets, key), COMPONENT_KEYS);

const assetBytes = (
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  key: string,
): Uint8Array<ArrayBuffer> => {
  assetBuffer(assets, key);
  return assets[key];
};

/**
 * グラフ入力の**静的**次元を引く。
 * MUST: パイプライン側に literal を置かない — コンテナが正で、モデルを差し替えたら値も追随する。
 */
const graphInputShape = (
  model: GraphOwner,
  inputName: string,
): readonly (number | string)[] => {
  const spec = model.graph.inputs.find((input) => input.name === inputName);
  if (spec === undefined) throw new Error(`グラフ入力 '${inputName}' が無い`);
  return spec.shape;
};

const staticInputShape = (model: GraphOwner, inputName: string): readonly number[] =>
  graphInputShape(model, inputName).map((dim, axis) => {
    if (typeof dim !== "number") {
      throw new Error(`グラフ入力 '${inputName}' の軸 ${axis} が静的次元でない（${String(dim)}）`);
    }
    return dim;
  });

const graphOutputShape = (model: GraphOwner): readonly (number | string)[] =>
  model.graph.values[model.graph.outputs[0]].shape;

const staticOutputShape = (model: GraphOwner): readonly number[] => {
  const name = model.graph.outputs[0];
  return graphOutputShape(model).map((dim, axis) => {
    if (typeof dim !== "number") {
      throw new Error(`グラフ出力 '${name}' の軸 ${axis} が静的次元でない（${String(dim)}）`);
    }
    return dim;
  });
};

/**
 * 記号次元を許す形から**最終次元**（= 特徴幅）を引く。S 形の `tokens [1,S,68]` は軸 1 が
 * 記号なので {@link staticInputShape} では読めないが、patch 幾何を割り出すのに要るのは
 * 最終次元だけ。
 */
const featureWidth = (dims: readonly (number | string)[], where: string): number => {
  const last = dims.at(-1);
  if (typeof last !== "number") throw new Error(`${where} の最終次元が静的でない`);
  return last;
};

const asF32 = (tensor: Tensor, where: string): Float32Array => {
  if (tensor.dtype !== "f32") throw new Error(`${where}: f32 でない（${tensor.dtype}）`);
  return tensor.data;
};

const idsTensor = (values: Int32Array<ArrayBuffer>): Tensor => ({
  dtype: "i32",
  shape: [1, values.length],
  data: values,
});

/**
 * 段 1 本が常駐 DiT の上に要るバイト数（`residency.ts` の {@link stageMemoryNeed}）。
 *
 * device の上限 2 つは全段に渡す — VAE の中間は上限で行ブロックが決まり、無いと見積りが落ちる
 * （text 系にも同じ形で渡す — 段の実引数と同じ見積りにするため）。
 */
const stageNeed = (
  state: AnimaState,
  model: ModelComponent,
  bindings: Record<string, number>,
): number =>
  stageMemoryNeed(
    model.estimate({
      maxStorageBufferBindingSize: state.gpu.limits.maxStorageBufferBindingSize,
      maxBufferSize: state.gpu.limits.maxBufferSize,
      bindings,
    }),
    model.maxPartBytes,
  );

/**
 * text 段の前の量り（先回りの退避 — `residency.ts` の「退避の 2 本の線」）で使う、text_encoder /
 * text_conditioner が常駐 DiT の上に要るバイト数（段ごとの最大 — {@link generateMemoryNeed}）。
 *
 * 束縛はこの generate が段へ実際に渡す形から取る: text_encoder の `T` = qwen の id 列の長さ、
 * text_conditioner の `Tsrc` = その出力の長さ（= qwen の id 列の長さ）と `Ttgt` = T5 の id 列の長さ。
 * CFG の 2 本（正 / 負）は同じ Session で順に回るので長いほうで量る。
 */
const textStagesNeed = (
  state: AnimaState,
  prompts: readonly ReturnType<AnimaTokenizers["encode"]>[],
): number => {
  const qwen = Math.max(...prompts.map((ids) => ids.qwenIds.length));
  const t5 = Math.max(...prompts.map((ids) => ids.t5Ids.length));
  return generateMemoryNeed([
    stageNeed(state, state.textEncoder, { T: qwen }),
    stageNeed(state, state.textConditioner, { Tsrc: qwen, Ttgt: t5 }),
  ]);
};

/**
 * VAE 段の前の量り（DiT 段の後）で使う、vae_decoder が常駐 DiT の上に要るバイト数（静的形 — 束縛なし）。
 */
const vaeStageNeed = (state: AnimaState): number => stageNeed(state, state.vaeDecoder, {});

/**
 * 実際に uncond 側へ渡すネガティブプロンプトを決める（{@link AnimaPipeline.generate} の入口・
 * GPU に触れる前の純粋な検査）。
 *
 * MUST: 効かないノブを黙って受けない。guidance=1 は uncond 分岐を丸ごと計算しないので、
 * ネガティブプロンプトは 1 文字も使われない（指定できたように見えるのが最悪）。逆に uncond を
 * 計算する設定で綴りが無ければ、GPU へ入る前のここで落とす。
 *
 * uncond の要否は `guidance` だけの関数なので、判定は引数に取らずここで導く（呼び手の
 * `wantsUncond` と食い違う余地を作らない）。
 */
export const resolveNegativePrompt = (
  requested: string | undefined,
  fallback: string | undefined,
  guidance: number,
): string | undefined => {
  const wantsUncond = needsUncond(guidance);
  // どちらも `negativePrompt` と `guidanceScale` の**組合せ**の違反で、打つ手は要求の側にある
  // （ADR 0107 決定 3 の値域・型・組合せ）。同じ入口の `guidanceScale` 非有限と型を揃える。
  if (!wantsUncond && requested !== undefined) {
    throw new ModelInputError(
      `guidanceScale ${guidance} では uncond 側を計算しないので negativePrompt は効かない` +
        "（効かせるなら guidanceScale を 1 以外にする）",
    );
  }
  const negativePrompt = requested ?? fallback;
  if (wantsUncond && negativePrompt === undefined) {
    throw new ModelInputError(
      `guidanceScale ${guidance} は uncond 側を計算するので negativePrompt が要る` +
        "（manifest の pipelineConfig.defaults.negativePrompt か request で渡す）",
    );
  }
  return negativePrompt;
};

/**
 * 1 step ぶんの更新結果。`previousX0` は**そのまま次 step の入力へ持ち回る**状態で、
 * DPM++ 2M の 2 次項が読む唯一の履歴（Euler は履歴を持たないので常に `undefined`）。
 */
export type AnimaDenoiseUpdate = {
  readonly sample: Float32Array<ArrayBuffer>;
  readonly previousX0: Float32Array<ArrayBuffer> | undefined;
};

/**
 * サンプラ種別による更新則の選択点（denoise ループが 1 step ごとに呼ぶ**唯一の分岐**）。
 *
 * MUST: `"euler"` は `cfgEulerStep` と Δσ の綴りを 1 ビットも変えない — `scheduler.type` を
 * 持たない既存の配布 manifest は既定 `"euler"` に落ちる（`config.ts`）ので、**配布済みリポの
 * 出力画像がここでビット同一のまま**でなければならない。
 *
 * CFG 合成はどちらの更新則も自分の内側で行う（同じ式）。uncond 側 forward の要否は
 * {@link needsUncond} が `guidance` だけから決めるので、**種別に依存しない** — 呼び出し側の
 * 分岐は「何 step 回すか」も「uncond を計算するか」も共通のまま。
 *
 * NOTE: 入力の形は {@link DpmSolverMultistepInput} をそのまま借りる（Euler が読むのは
 * `sigmas` / `index` / `guidance` / 出力 2 本で、DPM++ 2M の入力の部分集合）。
 * NOTE: `export` は GPU 無しで選択を縛るテストのため（`mod.ts` / サブパス面には出さない —
 * ADR 0008）。
 */
export const denoiseStep = (
  type: AnimaSamplerType,
  input: DpmSolverMultistepInput,
): AnimaDenoiseUpdate => {
  switch (type) {
    case "euler": {
      const { sample, cond, uncond, guidance, sigmas, index } = input;
      return {
        sample: cfgEulerStep(
          sample,
          cond,
          uncond,
          Math.fround(sigmas[index + 1] - sigmas[index]),
          guidance,
        ),
        previousX0: undefined,
      };
    }
    case "dpmpp-2m": {
      const update = dpmSolverMultistepStep(input);
      return { sample: update.sample, previousX0: update.x0 };
    }
  }
};

/** S 形 DiT の step 間で変わらない材料（rope 表と patch 幾何は解像度だけの関数）。 */
type DynDitPlan = {
  readonly geometry: DitPatchGeometry;
  readonly latentShape: readonly number[];
  readonly tokenShape: readonly number[];
  readonly ropeShape: readonly number[];
  readonly cos: Float32Array<ArrayBuffer>;
  readonly sin: Float32Array<ArrayBuffer>;
};

/**
 * 解像度から S 形 DiT の材料を組む。denoise ループの外で 1 度だけ呼ぶ（毎 step 組み直すと
 * 1024px で 4MB×2 の無駄が step ごとに乗る）。
 */
const planDynDit = (
  model: GraphOwner,
  ropeBase: RopeBase,
  resolution: ImageSize,
): DynDitPlan => {
  const tokenWidth = featureWidth(graphInputShape(model, "tokens"), "グラフ入力 'tokens'");
  const geometry = ditPatchGeometry(
    tokenWidth,
    featureWidth(graphOutputShape(model), "グラフ出力"),
  );
  // MUST: latent の寸法は**解像度から**割り出す（S 形のグラフは解像度を持たない）。
  // 取り違えは generate 末尾の「出た画像の寸法 == 要求解像度」検査が閉じる。
  // MUST: 軸の順は `[1,C,H,W]`（綴りの WxH とは逆）。非正方でここを入れ替えると要素数は
  // 合ったまま latent が転置され、絵が黙って別物になる（正方では検出不能な取り違え）。
  const latentHeight = resolution.height / ANIMA_SPATIAL_COMPRESSION;
  const latentWidth = resolution.width / ANIMA_SPATIAL_COMPRESSION;
  if (!Number.isInteger(latentHeight) || !Number.isInteger(latentWidth)) {
    throw new Error(
      `解像度 ${resolution.width}×${resolution.height} が空間圧縮率 ${ANIMA_SPATIAL_COMPRESSION}` +
        " で割り切れない",
    );
  }
  const latentShape = [1, geometry.channels, latentHeight, latentWidth];
  const { cos, sin } = ropeTables(ropeBase, latentShape, geometry);
  return {
    geometry,
    latentShape,
    tokenShape: [1, tokenCount(latentShape, geometry), tokenWidth],
    ropeShape: [1, 1, tokenCount(latentShape, geometry), ropeWidth(ropeBase)],
    cos,
    sin,
  };
};

/** 段の本体が受ける run（出力 1 本を返す — 観測席への通知込み）。 */
type StageRun = (inputs: Record<string, Tensor>) => Promise<Tensor>;

/** Session 1 本の run を段の本体向けに包む（出力 1 本を取り出し、観測席へ診断を渡す）。 */
const stageRun = (
  session: Session,
  model: ModelComponent,
  observe: ((diagnostics: SessionDiagnostics) => void) | undefined,
): StageRun => {
  const outputName = model.graph.outputs[0];
  return async (inputs) => {
    const outputs = await session.run(inputs);
    if (observe !== undefined) observe(session.diagnostics());
    return outputs[outputName];
  };
};

/**
 * pipeline の `dispose` の本体: 常駐 DiT を畳んでから GPU を破棄する（`destroyGpu` は内部で取った
 * GPU のときだけ渡す — 共有 GPU は呼び出し側の所有物）。
 *
 * MUST: 常駐 DiT が先（flush-before-destroy）。畳むのに失敗しても GPU は返す（`disposeSteps` の doc —
 * 失敗は全部回した後に投げる）。
 *
 * NOTE: `export` は GPU 無しで順序と失敗の扱いを縛るテストのため（`mod.ts` / サブパス面には出さない
 * — ADR 0008）。実 GPU の e2e は「常駐 DiT が畳まれたこと」を観測する口を持たない（GpuContext に
 * live バイトの診断が無い）。
 */
export const disposeResidencyThenGpu = (
  residency: { readonly dispose: () => Promise<void> },
  destroyGpu: (() => void) | undefined,
): Promise<void> =>
  disposeSteps([
    () => residency.dispose(),
    ...(destroyGpu === undefined ? [] : [destroyGpu]),
  ]);

/**
 * 1 グラフぶんの Session を張り、使い終わったら必ず解放する。
 * MUST: `finally` で dispose する — 途中で落ちたときに VRAM が残ると、後続の段が確保に
 * 失敗して「最初の失敗とは別の場所」で落ちる。
 */
const withSession = async <T>(
  gpu: GpuContext,
  model: ModelComponent,
  sessionOptions: SessionOptions,
  observe: ((diagnostics: SessionDiagnostics) => void) | undefined,
  body: (run: StageRun) => Promise<T>,
): Promise<T> => {
  const session = await model.createSession(gpu, sessionOptions);
  let failure: { readonly error: unknown } | undefined;
  try {
    return await body(stageRun(session, model, observe));
  } catch (error) {
    failure = { error };
    throw error;
  } finally {
    // MUST: dispose の失敗で body の失敗を上書きしない（`disposeSteps` の doc）。
    await disposeSteps([
      () => {
        if (failure !== undefined) throw failure.error;
      },
      () => session.dispose(),
    ]);
  }
};

/** 観測席（{@link AnimaPipelineOptions.onRunDiagnostics}）へコンポーネント名を焼いて渡す。 */
const observer = (
  state: AnimaState,
  component: AnimaRunComponent,
): ((diagnostics: SessionDiagnostics) => void) | undefined => {
  const listener = state.onRunDiagnostics;
  return listener === undefined ? undefined : (diagnostics) => listener(component, diagnostics);
};

/** {@link AnimaPipeline} の内部状態（コンストラクタが private なので型は公開しない）。 */
type AnimaState = {
  readonly gpu: GpuContext;
  readonly ownsGpu: boolean;
  readonly config: AnimaPipelineConfig;
  readonly sessionOptions: SessionOptions;
  readonly tokenizers: AnimaTokenizers;
  readonly textEncoder: ModelComponent;
  readonly textConditioner: ModelComponent;
  readonly transformer: ModelComponent;
  readonly ropeBase: RopeBase;
  readonly vaeDecoder: ModelComponent;
  /** generate が request で上書きしなかったときの実効 residency（検査済み）。 */
  readonly residency: AnimaResidency;
  readonly onRunDiagnostics?: (
    component: AnimaRunComponent,
    diagnostics: SessionDiagnostics,
  ) => void;
};

/** 家族 admission（`AnimaPipeline.#admit`）が確定させる材料。 */
type AnimaAdmission = {
  readonly config: AnimaPipelineConfig;
  readonly quantName: string;
  readonly quant: Quant;
  /** 明示指定と quant 宣言を合成した実効設定（{@link ANIMA_SESSION_POLICY}）。 */
  readonly sessionOptions: SessionOptions;
  /** 実効設定が要る feature を quant 宣言へ足したもの（要求と検査の両方がこれを見る）。 */
  readonly gpuFeatures: GpuFeaturesSpec | undefined;
  /** 構築オプションの residency（検査済み・省略時は既定）。 */
  readonly residency: AnimaResidency;
};

/**
 * Anima のテキスト → 画像パイプライン。
 *
 * 構築は {@link AnimaPipeline.fromPretrained}（HF から取得）か
 * {@link AnimaPipeline.fromAssets}（取得済みバイト列）だけを入口にする — コンストラクタを
 * private にしてあるのは、manifest 検査と資産の突合を迂回した半端な状態を作れないようにする
 * ため（`createSession` / `acquireGpu` と同じ流儀 — ADR 0008）。
 */
export class AnimaPipeline {
  readonly #state: AnimaState;
  /** generate と dispose の直列化鎖（モジュール doc の「1 本ずつ」を公開 API 側で守る）。 */
  readonly #chain = createOperationChain();
  /**
   * dispose の 1 本。**undefined でないことが「dispose 済み」**（別に真偽値を持つと、独立に
   * 更新される派生状態になる）。
   */
  #disposal: Promise<void> | undefined;
  /**
   * DiT の常駐の席（ADR 0112）。pipeline の所有物 — `options.gpu` を共有していても `dispose` で
   * 畳む。鎖（{@link AnimaPipeline.#chain}）の内側からだけ触る。
   */
  readonly #residency: TransformerResidency<Session>;

  private constructor(state: AnimaState) {
    this.#state = state;
    this.#residency = new TransformerResidency<Session>({
      dispose: (session) => session.dispose(),
      // 退避（OOM / 空き不足）の後、解放が device に届くのを待つ（理由は settle-released-memory.ts の doc）。
      // 空の submit がここで安全なのは、退避が走るのは使用量が確保の線（予算の 97%）以下のとき（先回りの
      // 試し確保は解放まで待って返る・反応の退避は OOM で確保が拒まれた直後）か、device が既に死んでいる
      // ときだけだから — wgpu の submit 後の 99% 線の判定（超えると device を失う）を踏まない。
      // NOTE: B570 で pipeline の退避 → やり直し（evict-probe・ダミー 6 GiB）が device lost になったのは、
      // 段の OOM が `queue.writeBuffer` の staging 側で、その時点で device が無効化されていたため
      // （docs/research/2026-09-27-h35-oom-device-lost.md）— 解放待ちでは直らない。主線は先回りの退避
      // （`residency.ts` の「退避の 2 本の線」）。
      settleRelease: () => settleReleasedMemory(state.gpu),
    });
  }

  /**
   * 配布形から取得して組む（`loadManifest` → `resolveSelection` → **各部品の descriptor
   * （part 0）だけ**を取って admission → 重みの part を温める → 残り資産の `fetchAssets` →
   * 構築）。block は Session を組むその瞬間に part 順で読まれる（ADR 0109 —
   * `src/hub/components.ts`）。文字列の `ref` は `{ repo }` と読む（= `main` 追従）。
   * **`ref` は必須**（取得元に既定は無い — `src/hub/repo-ref.ts` の MUST）。
   *
   * 部品を別リポの同じ役割で差し替えるときは
   * {@link FromPretrainedComponentOptions.components}（`text_encoder` /
   * `text_conditioner` / `transformer` / `vae_decoder`）。
   *
   * 手元の配布形は**取得元ハンドル**で渡す（`localDirectory` / `@karume/hub/deno` の
   * `denoDirectory`）。HF の `owner/name` の綴りの門は通らず、network も CacheStorage も
   * 通らない（{@link AnimaFromPretrainedOptions} の HTTP 専用ノブは効かない）。
   */
  static async fromPretrained(
    ref: string | HubRepoRef | DistributionSource,
    options: AnimaFromPretrainedOptions = {},
  ): Promise<AnimaPipeline> {
    const source = toManifestSource(
      ref,
      "AnimaPipeline.fromPretrained",
      'ANIMA_SOURCES["anima"]（@karume/models/anima）',
    );
    const hubOptions = hubLoadOptions(options);
    const loaded = await loadManifest(source, hubOptions);
    const choice = {
      ...(options.model === undefined ? {} : { model: options.model }),
      ...(options.quant === undefined ? {} : { quant: options.quant }),
    };
    const selection = resolveSelection(loaded.manifest, choice);
    // signal は取得層と構築の**両方**へ渡す（DL が終わった瞬間に中断が効かなくなる窓を作らない）。
    const buildOptions: AnimaPipelineOptions = {
      ...(options.gpu === undefined ? {} : { gpu: options.gpu }),
      ...choice,
      ...(options.onRunDiagnostics === undefined
        ? {}
        : { onRunDiagnostics: options.onRunDiagnostics }),
      ...(options.signal === undefined ? {} : { signal: options.signal }),
      ...(options.linearCompute === undefined ? {} : { linearCompute: options.linearCompute }),
      ...(options.attentionCompute === undefined
        ? {}
        : { attentionCompute: options.attentionCompute }),
      ...(options.attentionScoreStorage === undefined
        ? {}
        : { attentionScoreStorage: options.attentionScoreStorage }),
      ...(options.residency === undefined ? {} : { residency: options.residency }),
    };
    // 家族の門は admission 席で通す（重みの part を取る前 — `src/hub/components.ts`）。
    const { admitted, assets, open } = await loadContainerComponents(
      "AnimaPipeline.fromPretrained",
      loaded,
      selection,
      COMPONENT_KEYS,
      async () => {
        const admitted = AnimaPipeline.#admit(loaded.manifest, buildOptions);
        // 配布形が宣言した `requiredLimits` は**重みの part を取る前**にここで見る
        // （ADR 0089 決定 5 — 共有 GPU ならその limits、自前で取る経路はアダプタ実測値）。
        await assertRequiredLimitsBeforeDownload(
          admitted.quant.requiredLimits,
          buildOptions.gpu,
          `AnimaPipeline: quant '${admitted.quantName}'`,
        );
        return admitted;
      },
      {
        ...hubOptions,
        ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
        ...(options.components === undefined ? {} : { components: options.components }),
      },
    );
    return await AnimaPipeline.#build(admitted, assets, open, buildOptions);
  }

  /**
   * この manifest を anima として実行できるかを見る（`src/hub/components.ts` の家族
   * admission 席 — 取得面では**重みの part を 1 バイトも取る前**に呼ばれる）。
   *
   * MUST: 家族の門はこの 1 本に集める。後段へ散らすと、取得面では GB 級の重みを落とした
   * **後**にしか落ちない（ADR 0070 決定 5 の文面より実装が狭くなる）。
   * MUST: manifest の契約違反は **GPU を取りに行く前**に落とす（他 6 家族と同じ順序）。
   *
   * NOTE: グラフを受け取らないのは、この家族が「グラフ宣言 × pipelineConfig」の突合を構築時に
   * 持たないため（patch 幾何と rope は解像度ごとに `planDynDit` が生成時に導く）。資産
   * （tokenizer 2 本 / rope 素表）の解析もこの席へは置けない — admission の時点で手元にあるのは
   * 資産の**宣言**（名前 → 役割・論理長）だけでバイト列はまだ無い（待つと重み prefetch より前
   * という位置が保てない）ので {@link AnimaPipeline.#build} に残る。
   */
  static #admit(
    manifest: Manifest,
    options: AnimaPipelineOptions,
  ): AnimaAdmission {
    // 中断の検査は**段の境目**に置く（各段は不可分 — 3.7GiB の部品を途中で畳む口は無い）。
    // 入口が最初の 1 本: 中断済みで呼ばれたら資産に 1 バイトも触らずに返す。
    options.signal?.throwIfAborted();
    const modelName = options.model ?? manifest.defaultModel;
    if (!Object.hasOwn(manifest.models, modelName)) {
      throw new Error(
        `AnimaPipeline: model '${modelName}' は manifest に無い` +
          `（利用可能: ${manifest.available.models.join(" / ")}）`,
      );
    }
    const entry: ModelEntry = manifest.models[modelName];
    const { name, major } = entry.pipeline;
    if (name !== ANIMA_PIPELINE_NAME) {
      throw new Error(
        `AnimaPipeline: manifest の pipeline が '${name}/${major}'` +
          `（'${ANIMA_PIPELINE_NAME}/${ANIMA_PIPELINE_MAJOR}' が必要）`,
      );
    }
    if (major !== ANIMA_PIPELINE_MAJOR) {
      // 「古い実装 × 新しいリポ」の沈黙劣化を止める唯一の門（ADR 0038 §6）。
      throw new Error(
        `AnimaPipeline: pipeline '${name}/${major}' の major に未対応` +
          `（この実装が読めるのは ${ANIMA_PIPELINE_NAME}/${ANIMA_PIPELINE_MAJOR}）`,
      );
    }
    const config = parseAnimaPipelineConfig(entry.pipelineConfig);

    const quantName = options.quant ?? entry.defaultQuant;
    if (!Object.hasOwn(entry.quants, quantName)) {
      throw new Error(
        `AnimaPipeline: quant '${quantName}' は manifest に無い` +
          `（利用可能: ${entry.available.quants.join(" / ")}）`,
      );
    }
    const quant = entry.quants[quantName];
    // 明示 > quant 宣言 > runtime 既定（全家族共通の 1 本）。未対応の宣言も誤った明示指定も
    // ここで落とす（重みの part を取る前）。
    const sessionOptions = resolveSessionOptions(
      ANIMA_SESSION_POLICY,
      quant.session,
      options,
      `AnimaPipeline: quant '${quantName}'`,
    );
    const gpuFeatures = sessionGpuFeatures(quant.gpuFeatures, sessionOptions);
    // residency も同じ席で見る（未知の綴りを重みの取得の後まで持ち越さない）。
    const residency = options.residency === undefined
      ? DEFAULT_ANIMA_RESIDENCY
      : assertAnimaResidency(options.residency, "AnimaPipeline: residency");

    // MUST: 共有 GPU の能力不足（feature / device limit）はこの席で落とす — 自前で取る場合と
    // 違って `acquireGpu` を待つ理由が無く、重みを落とす前に判る唯一の家族門（要求と検査の
    // 写像は `session/gpu-features.ts` の 1 本で、後段の検査も同じ関数を呼ぶ）。
    if (options.gpu !== undefined) {
      assertGpuFeaturesGranted(
        gpuFeatures,
        options.gpu,
        `AnimaPipeline: quant '${quantName}'`,
      );
      assertRequiredLimitsSatisfied(
        quant.requiredLimits,
        options.gpu.limits,
        `AnimaPipeline: quant '${quantName}'`,
      );
    }

    return { config, quantName, quant, sessionOptions, gpuFeatures, residency };
  }

  /**
   * admission を通った材料 + 資産から組む（{@link AnimaPipeline.fromAssets} と
   * `fromPretrained` が共有する 1 本）。2 面の違いは `open`（コンポーネントの供給口）だけで、
   * 検査も順序も同じものを通る。
   *
   * MUST: 資産の解析は **GPU を取りに行く前**（docstring の順序 MUST）。
   */
  static async #build(
    admitted: AnimaAdmission,
    assets: AnimaAssets["assets"],
    open: ComponentOpener,
    options: AnimaPipelineOptions,
  ): Promise<AnimaPipeline> {
    const { config, quantName, sessionOptions, gpuFeatures, residency } = admitted;

    // 資産の解析は GPU より前（docstring の順序 MUST）。3.7GiB の DiT を開くほうが device 生成
    // より重いが、壊れた配布形の真因を消さないほうを採る — GPU 無し環境では acquireGpu 自体が
    // 落ちるので、後ろに置くと「資産が無い」が永久に見えない。
    await settleAbort(options.signal);
    const tokenizers = createTokenizers(
      assetBytes(assets, TOKENIZER),
      assetBytes(assets, TOKENIZER_2),
    );
    await settleAbort(options.signal);
    const textEncoder = open(TEXT_ENCODER);
    await settleAbort(options.signal);
    const textConditioner = open(TEXT_CONDITIONER);
    await settleAbort(options.signal);
    const transformer = open(TRANSFORMER);
    // rope 素表は `transformer` の**容器の資産**（役割 `rope-base` — ADR 0109 決定 4）。
    // 66 KB なので全量で読む（区間読みが要るのは PLE のような行単位の表だけ）。
    const ropeBase = parseRopeBase(await readWholeAsset(transformer.asset(ROPE_BASE)));
    await settleAbort(options.signal);
    const vaeDecoder = open(VAE_DECODER);

    await settleAbort(options.signal);

    // MUST: 宣言された feature は device 作成時にしか要求できない（ADR 0028）。共有 GPU を
    // 渡された場合は要求できないので、能力が足りないことを名指しして落とす — 通すと Session
    // 構築まで進んでから落ちる（あるいは黙って別の経路へ縮退する）。共有 GPU は
    // {@link AnimaPipeline.#admit} が既に同じ 1 本で見ているが、自前で取った device は
    // ここが唯一の門。
    const gpu = options.gpu ?? await acquireGpu(toAcquireGpuOptions(gpuFeatures));
    const ownsGpu = options.gpu === undefined;
    try {
      // MUST: GPU 取得**後**の中断検査は try の中に置く — 外に出すと、内部で取った device を
      // 誰も解放できないまま抜ける（feature 検査と同じ後始末に乗せる）。
      // ここでもマクロタスクへ譲る: `acquireGpu` の await 解決はマイクロタスク継続なので、
      // 待機中に積まれたクリック由来の中断タスクはまだ実行されていない。
      await settleAbort(options.signal);
      assertGpuFeaturesGranted(gpuFeatures, gpu, `AnimaPipeline: quant '${quantName}'`);
      return new AnimaPipeline({
        gpu,
        ownsGpu,
        config,
        sessionOptions,
        tokenizers,
        textEncoder,
        textConditioner,
        transformer,
        ropeBase,
        vaeDecoder,
        residency,
        ...(options.onRunDiagnostics === undefined
          ? {}
          : { onRunDiagnostics: options.onRunDiagnostics }),
      });
    } catch (error) {
      // 内部で取った GPU は、構築に失敗したら誰も解放できなくなるのでここで返す。
      if (ownsGpu) gpu.destroy();
      throw error;
    }
  }

  /**
   * 取得済みの manifest + 資産から組む。**ここが核**で、以下を全て構築時に済ませる:
   *
   * - `model` の選択（未知のモデル名は利用可能な一覧つきで fail loudly — ADR 0041 §8）
   * - `pipeline` の契約名と major の検査（**未知 major は fail loudly** — 検査責務は
   *   models 側。ADR 0038 §1）
   * - `pipelineConfig` の手書きスキーマ検証（未知キーも fail loudly）
   * - quant の `session` → runtime `SessionOptions` の**明示写像**と `gpuFeatures` の解釈
   * - 全 weights の `openContainer` / rope 素表 / トークナイザ 2 本の解釈
   *
   * weights の部品キーは **2 形とも受ける** — 単一形 `krm` のバイト列 1 本（`transformer`）と、
   * 分割形の part 列（`transformer[0]` / `transformer[1]` / … — part 0 から添字順）。分割形は
   * バイト列を連結せず part 列のまま開く。添字の欠番と単一形キーとの混在は fail loudly
   * （受け口の実装は `src/hub/components.ts`）。rope 素表は `transformer` の容器の資産なので、
   * この面でも `assets` には並べない。
   *
   * MUST: manifest の契約違反と**資産の解析**は **GPU を取りに行く前**に落とす（他 6 家族と
   * 同じ順序）。順序がずれると、GPU の無い環境では別の例外に化けて「何が悪かったのか」が
   * 読み手に伝わらない。GPU 取得後に許される検査は GPU の能力（shader-f16）だけ。
   * MUST: Session は 1 本も張らない（VRAM の MUST — モジュール doc）。
   *
   * NOTE: 各段は不可分（3.7GiB の部品を途中で畳む口は無い）なので、
   * {@link AnimaPipelineOptions.signal} の検査は**段の境目**にだけ置き、そこで
   * イベントループへ 1 度譲ってから検査する（{@link settleAbort}）— 同期解析の最中に
   * 届いた中断は次の境目で効く（`options.gpu` 供給時も同様）。
   */
  static async fromAssets(
    input: AnimaAssets,
    options: AnimaPipelineOptions = {},
  ): Promise<AnimaPipeline> {
    const admitted = AnimaPipeline.#admit(input.manifest, options);
    return await AnimaPipeline.#build(
      admitted,
      input.assets,
      await assetOpener(input.assets),
      options,
    );
  }

  /**
   * プロンプトから画像 1 枚を生成する。
   *
   * 同じ seed・同じノブなら同じ画素が出る（乱数も丸めもホスト側で決定的 — `random.ts` /
   * `sampler.ts`）。
   *
   * 並行に呼ばれた場合は**待たされて順に**走る（グラフの同時常駐を作らない — モジュール doc）。
   */
  async generate(request: AnimaGenerateRequest): Promise<GeneratedImage> {
    // dispose 済みの判定は呼び出し時点で行う（鎖の中で見ると、dispose より前に受けた生成まで
    // 巻き添えで落ちる）。
    if (this.#disposal !== undefined) {
      throw new Error("AnimaPipeline: dispose 済みでは生成できない");
    }
    return await this.#chain(() => this.#generate(request));
  }

  async #generate(request: AnimaGenerateRequest): Promise<GeneratedImage> {
    const state = this.#state;
    const defaults = state.config.defaults;
    const steps = request.steps ?? defaults.steps;
    const guidance = request.guidanceScale ?? defaults.guidanceScale;
    const resolution = request.resolution ?? defaults.resolution;
    const seed = request.seed ?? 0;
    // 更新則は request が優先で、省略時だけ配布者の宣言に落ちる。未知の綴りは GPU に触る前に
    // 期待と実際を並べて落とす（TS 型を通らない JS 呼び出し向け — 既定への無言の縮退を作らない）。
    const sampler = request.sampler === undefined
      ? state.config.scheduler.type
      : assertAnimaSamplerType(request.sampler, "sampler");
    assertAcceptableResolution(resolution);
    // seed の検査も入口に置く。生成器を作るのは DiT の段（`new Randn(seed)`）で、そこは
    // text encoder / conditioner を回して DiT の重みを上げ終えた後なので、`Randn` 側だけに
    // 検査があると不正な seed が GB 級のロードの末に落ちる。
    assertAcceptableSeed(seed);
    // steps も同じ理由で入口に置く（受理集合の所有者は梯子側の `assertAcceptableSteps` 1 本）。
    assertAcceptableSteps(steps);
    if (!Number.isFinite(guidance)) {
      throw new ModelInputError(`guidanceScale ${guidance} が有限の数でない`);
    }
    // 実効 residency（request ?? 構築オプション）。意味は「この generate の**後に** DiT を持つか」
    // — 開始時の常駐 DiT は値に関わらず使う（`residency.ts` の `transformerSource`）。
    const residency = request.residency === undefined
      ? state.residency
      : assertAnimaResidency(request.residency, "residency");

    const wantsUncond = needsUncond(guidance);
    const negativePrompt = resolveNegativePrompt(
      request.negativePrompt,
      defaults.negativePrompt,
      guidance,
    );

    // 生成イベントの発火口。未購読なら何もしない 1 本に畳んで、発火点に分岐を置かない。
    const { onEvent } = request;
    const emit: (event: AnimaGenerateEvent) => Promise<void> = onEvent === undefined
      ? () => Promise.resolve()
      : async (event) => {
        await onEvent(event);
      };
    const notifyResidency = (notice: ResidencyNotice): Promise<void> =>
      emit({ kind: "residency", component: "transformer", ...notice });
    /** 先回りの退避の試し確保（`TransformerResidency.ensureHeadroom` の `probe`）。 */
    const probeHeadroom = (bytes: number): Promise<boolean> => fitsHeadroom(state.gpu, bytes);
    /**
     * DiT 以外の段 1 本を回す（`stage` を Session 構築の前と解放の後に挟む — 途中で落ちたら
     * `end` は出ない）。常駐 DiT がある状態の OOM は退避して段を 1 回だけやり直す
     * （`TransformerResidency.runStage`）。text 系 / VAE は quant の session を受けない（`{}`）。
     */
    const withStage = async <T>(
      component: Exclude<AnimaRunComponent, "transformer">,
      model: ModelComponent,
      body: (run: StageRun) => Promise<T>,
    ): Promise<T> => {
      await emit({ kind: "stage", component, at: "start" });
      const result = await this.#residency.runStage(
        () => withSession(state.gpu, model, {}, observer(state, component), body),
        notifyResidency,
      );
      await emit({ kind: "stage", component, at: "end" });
      return result;
    };

    // --- ① プロンプト層（GPU 不要・決定的）------------------------------------
    const positive = state.tokenizers.encode(request.prompt, "プロンプト");
    const negative = wantsUncond
      ? state.tokenizers.encode(negativePrompt as string, "ネガティブプロンプト")
      : undefined;
    const sigmas = sigmaSchedule(steps, state.config.scheduler.shift);

    // GPU に触る段（text 段以降）で generate が失敗したら、実効値 per-stage の「この generate の後に
    // 持たない」指示を効かせる（DiT 段より前の失敗 — text 段・`stage` の onEvent の throw 等 — でも
    // 持ち越した DiT を手放す）。DiT 段の中の失敗は `runTransformer` が決着させている。入力の検査
    // （ここより上）で落ちた要求は GPU にも常駐の席にも触らない。
    try {
      // --- ①' 先回りの退避（text 段の前 — 持ち越した常駐 DiT があるときだけ）-------------
      // text 系 2 段が常駐 DiT の上に入らなければ、OOM を踏む前に退避する（`residency.ts` の「退避の
      // 2 本の線」）。試し確保の throw（validation / device 消失）はそのまま下の失敗経路へ流す。
      // `holding` をここでも見るのは見積りの費用を段ごと運転に払わせないため（量るかどうかの判断は
      // `ensureHeadroom` が持つ）。
      if (this.#residency.holding) {
        await this.#residency.ensureHeadroom(
          textStagesNeed(state, negative === undefined ? [positive] : [positive, negative]),
          probeHeadroom,
          notifyResidency,
        );
      }

      // --- ② テキスト経路（DiT ロードの前に解放する）---------------------------
      const hidden = await withStage(
        "text_encoder",
        state.textEncoder,
        (run) =>
          Promise.all([
            run({ input_ids: idsTensor(positive.qwenIds) }),
            ...(negative === undefined ? [] : [run({ input_ids: idsTensor(negative.qwenIds) })]),
          ]),
      );
      const embeds = await withStage(
        "text_conditioner",
        state.textConditioner,
        (run) =>
          Promise.all([
            run({ source_hidden_states: hidden[0], target_input_ids: idsTensor(positive.t5Ids) }),
            ...(negative === undefined ? [] : [
              run({
                source_hidden_states: hidden[1],
                target_input_ids: idsTensor(negative.t5Ids),
              }),
            ]),
          ]),
      );

      // --- ③ denoise（DiT を N step。段ごと運転なら VAE ロードの前に解放する）---------
      // 低精度計算のノブ（quant の session）は **DiT の Session にだけ**効かせる —
      // text 系 / VAE は対象外（比較の軸を DiT 1 本に保つ）。
      // Session の出所（常駐 / 新規に常駐 / 段ごと）と段の後の扱いは `TransformerResidency` が
      // 決める（ADR 0112）。本体は OOM の退避後のやり直しで**最初から**呼び直されるので、denoise の
      // 状態（乱数・latent・DPM の履歴）は本体の中で作る。
      const observeTransformer = observer(state, "transformer");
      await emit({ kind: "stage", component: "transformer", at: "start" });
      const { latents, latentShape } = await this.#residency.runTransformer({
        effective: residency,
        open: () => state.transformer.createSession(state.gpu, state.sessionOptions),
        notify: notifyResidency,
        body: async (session) => {
          const model = state.transformer;
          const run = stageRun(session, model, observeTransformer);
          const plan = planDynDit(model, state.ropeBase, resolution);
          const [, projWidth] = staticInputShape(model, "timesteps_proj");
          const embedShape = staticInputShape(model, "encoder_hidden_states");
          const [, rows, width] = embedShape;
          if (embeds[0].shape[2] !== width) {
            throw new Error(`conditioner 出力の幅 ${embeds[0].shape[2]} が DiT の ${width} と違う`);
          }
          /**
           * latent 1 枚を DiT へ通す。入口で patchify、出口で unpatchify を挟むだけで、
           * 呼び出し側（CFG / Euler）は latent だけを見る。
           */
          const predict = async (
            current: Float32Array<ArrayBuffer>,
            proj: Tensor,
            embed: Float32Array<ArrayBuffer>,
          ): Promise<Float32Array> => {
            const output = await run({
              tokens: {
                dtype: "f32",
                shape: plan.tokenShape,
                data: patchifyLatents(current, plan.latentShape, plan.geometry),
              },
              timesteps_proj: proj,
              encoder_hidden_states: { dtype: "f32", shape: embedShape, data: embed },
              rope_cos: { dtype: "f32", shape: plan.ropeShape, data: plan.cos },
              rope_sin: { dtype: "f32", shape: plan.ropeShape, data: plan.sin },
            });
            return unpatchifyTokens(
              asF32(output, "DiT 出力（S 形）"),
              plan.latentShape,
              plan.geometry,
            );
          };
          const padded = embeds.map((embed) => padSequence(embed, rows));
          const elements = plan.latentShape.reduce((a, b) => a * b, 1);
          let current = new Randn(seed).normals(elements);
          // DPM++ 2M が読む唯一の履歴（step 0 では無い）。Euler 経路では常に undefined のまま。
          let previousX0: Float32Array<ArrayBuffer> | undefined;
          for (let index = 0; index < steps; index += 1) {
            const proj: Tensor = {
              dtype: "f32",
              shape: [1, projWidth],
              data: timestepsProj(
                sigmas[index],
                projWidth,
                state.config.scheduler.numTrainTimesteps,
              ),
            };
            const predictions: Float32Array[] = [];
            for (const embed of padded) predictions.push(await predict(current, proj, embed));
            const update = denoiseStep(sampler, {
              sample: current,
              cond: predictions[0],
              uncond: predictions[1],
              guidance,
              sigmas,
              index,
              previousX0,
            });
            current = update.sample;
            previousX0 = update.previousX0;
            await emit({
              kind: "denoise-step",
              step: index + 1,
              steps,
              sigma: sigmas[index],
              copyLatents: latentSnapshot(current, plan.latentShape),
            });
          }
          return { latents: current, latentShape: plan.latentShape };
        },
      });
      await emit({ kind: "stage", component: "transformer", at: "end" });

      // --- ③' 先回りの退避（VAE 段の前 — この時点で常駐 DiT があるときだけ）----------------
      // text 段の前の量りでは覆えない 2 つ — DiT をこの generate で作って席に載せた（最初の generate）
      // と、持ち越した DiT が新しい解像度で backing を育てた — をここで量る（`residency.ts` の「退避の
      // 2 本の線」）。DiT 段は常駐の恩恵を受け終えているので、退避はできるだけ遅いほうが得。
      if (this.#residency.holding) {
        await this.#residency.ensureHeadroom(vaeStageNeed(state), probeHeadroom, notifyResidency);
      }

      // --- ④ 逆正規化 → VAE decode（常時タイル — ADR 0038 §4）-------------------
      const { mean: latentsMean, std: latentsStd } = animaLatents();
      const denormalized = denormalizeLatents(latents, latentShape, latentsMean, latentsStd);
      const decoded = await withStage(
        "vae_decoder",
        state.vaeDecoder,
        async (run) => {
          const tileShape = staticInputShape(state.vaeDecoder, "latents");
          const sampleShape = staticOutputShape(state.vaeDecoder);
          const geometry = planVaeTiling(latentShape, tileShape, sampleShape);
          const tiles = tileCount(geometry);
          const blendsOf = (axis: typeof geometry.rows): number[] =>
            axis.starts.slice(1).map((_, index) => blendExtentAt(axis, geometry.scale, index + 1));
          let decodedTiles = 0;
          const pixels = await decodeTiled(denormalized, geometry, async (tile, row, col) => {
            const output = await run({ latents: { dtype: "f32", shape: tileShape, data: tile } });
            const sample = asF32(output, `VAE 出力（タイル ${row},${col}）`);
            decodedTiles += 1;
            await emit({ kind: "vae-tile", tile: decodedTiles, tiles });
            return sample;
          });
          // 寸法は**幾何**が正本（latent の全長 × 縮尺）。画素数から逆算しない — 非正方では
          // `3·H·W` の分解が一意でなく、逆算では縦横の取り違えが原理的に検出できない。
          return {
            pixels,
            width: geometry.cols.extent * geometry.scale,
            height: geometry.rows.extent * geometry.scale,
            tiles,
            // ブレンド幅は**対ごと**（丸め等間隔なので 1 latent まで動く）。診断に載るのは
            // 幾何そのもので、代表値へ畳むと食い違いの手掛かりが消える。
            blend: [blendsOf(geometry.rows), blendsOf(geometry.cols)] as const,
          };
        },
      );

      // MUST: 出た画像の寸法はノブではなく**資産**が決める。ここで食い違うなら開いた export が
      // 想定と違うので、黙って返さない。要素数との整合は `imageToRgba` が見る。
      if (decoded.width !== resolution.width || decoded.height !== resolution.height) {
        throw new Error(
          `VAE 出力が ${decoded.width}×${decoded.height} で要求解像度 ${
            formatResolution(resolution)
          } と違う（タイル ${decoded.tiles} 枚 / ブレンド 縦 [${decoded.blend[0]}] 横 [${
            decoded.blend[1]
          }]px）`,
        );
      }

      const rgba = imageToRgba(decoded.pixels, decoded.width, decoded.height);
      return {
        width: decoded.width,
        height: decoded.height,
        data: new Uint8Array(rgba.buffer),
      };
    } catch (error) {
      throw await this.#residency.releaseIfRequested(error, residency, notifyResidency);
    }
  }

  /**
   * 解放する。常駐 DiT（{@link AnimaPipelineOptions.residency}）は必ず畳み、**内部で取得した GPU
   * だけ**破棄する（`options.gpu` で渡された GpuContext は呼び出し側の所有物なので触らない）。
   *
   * `options.gpu` を共有していても dispose は必須 — 常駐 DiT（既定席 1024² で約 2.6 GiB）は共有
   * device の上に残り続ける。device が失われていると常駐 DiT の破棄（Session の flush）が失敗し、
   * dispose が reject しうる（内部で取った GPU はそれでも破棄する）。
   *
   * MUST: in-flight の生成の完了を待ってから破棄する（flush-before-destroy）— 破棄も鎖に
   * 載せることで、待ちと破棄の順序を 1 箇所で決める。2 度目以降も同じ完了を返す（先に返すと
   * 呼び出し側が「破棄済み」と見なして次へ進む）。
   */
  dispose(): Promise<void> {
    // MUST: 常駐 DiT を先に畳んでから GPU を破棄する（flush-before-destroy）。常駐 DiT は共有 GPU
    // （`options.gpu`）でも pipeline の所有物なので必ず畳む。畳むのに失敗しても内部で取った GPU は
    // 返す（`disposeSteps` の doc）。
    this.#disposal ??= this.#chain(() =>
      disposeResidencyThenGpu(
        this.#residency,
        this.#state.ownsGpu ? () => this.#state.gpu.destroy() : undefined,
      )
    );
    return this.#disposal;
  }

  /** `await using` 対応（Explicit Resource Management）— {@link dispose} の別名。 */
  [Symbol.asyncDispose](): Promise<void> {
    return this.dispose();
  }
}
