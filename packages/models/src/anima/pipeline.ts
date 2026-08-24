/**
 * `AnimaPipeline` — テキスト → 画像（Anima）の 1 本の面。
 *
 * パイプライン（全段 Karume・torch 不使用）:
 *
 * 1. トークナイザ 2 本（Qwen2 BPE / T5 Unigram）でプロンプトを id 列にする
 * 2. `text_encoder`（Qwen3）→ `text_conditioner` → 512 ゼロ埋め
 * 3. `transformer`（S 形 DiT）を N step 回し、ホスト側で CFG 合成 + Euler 更新
 * 4. latent を per-channel 逆正規化 → `vae_decoder` を**常時タイル**で通す
 * 5. RGBA 化して返す（PNG 化は `encodePng` — パイプライン非依存の共通処理）
 *
 * ## MUST: グラフは 1 本ずつ開いて閉じる（4 本同時常駐は VRAM で不成立 — ADR 0016）
 *
 * 解放の**位置**に意味がある。テキスト経路（1.4GB 級）を DiT ロードの**前**に、DiT
 * （3.7GB）を VAE ロードの**前**に解放する。実測機の GPUBuffer 総確保量の天井は 7,280MiB で、
 * テキスト経路と DiT が同時に生きると 5.2GB になり活性を乗せる余地が薄くなる。
 * したがって {@link AnimaPipeline.fromAssets} は **Session を 1 本も張らない** —
 * 開くのはコンテナ（`openModel` = ヘッダ解析のみ）までで、GPU 常駐は
 * {@link AnimaPipeline.generate} の中で段ごとに張っては畳む。
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
  createSession,
  type GpuContext,
  type KarumeModel,
  openModel,
  type Session,
  type SessionDiagnostics,
  type SessionOptions,
  type Tensor,
} from "@karume/runtime";
import {
  type AssetProgress,
  type CacheDiagnostic,
  fetchAssets,
  type HubRepoRef,
  loadManifest,
  type Manifest,
  type ModelEntry,
  resolveFiles,
} from "@karume/hub";

import {
  ANIMA_PIPELINE_MAJOR,
  ANIMA_PIPELINE_NAME,
  type AnimaPipelineConfig,
  parseAnimaPipelineConfig,
} from "./config.ts";
import { cfgEulerStep, needsUncond, sigmaSchedule, timestepsProj } from "./sampler.ts";
import {
  ANIMA_LATENTS_MEAN,
  ANIMA_LATENTS_STD,
  denormalizeLatents,
  padSequence,
} from "./latents.ts";
import { imageToRgba } from "./image.ts";
import { assertAcceptableResolution, formatResolution, type ImageSize } from "./resolution.ts";
import { blendExtent, decodeTiled, planVaeTiling, tileCount } from "./tiling.ts";
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
import { assertAcceptableSeed, Randn } from "./random.ts";
import { settleAbort } from "../concurrency/abort.ts";
import { createOperationChain } from "../concurrency/serial.ts";
import { assertGpuFeaturesGranted, toAcquireGpuOptions } from "../session/gpu-features.ts";
import { toSessionOptions } from "../session/options.ts";
import { toRepoRef } from "../hub/repo-ref.ts";

/** manifest の weights / assets 表に現れる取得キー（ADR 0041 §3 の規約名）。 */
const TEXT_ENCODER = "text_encoder";
const TEXT_CONDITIONER = "text_conditioner";
const TRANSFORMER = "transformer";
const TRANSFORMER_ROPE_BASE = "transformer.rope_base";
const VAE_DECODER = "vae_decoder";
const TOKENIZER = "tokenizer";
const TOKENIZER_2 = "tokenizer_2";

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
   * **指定すると fail loudly**（効かないノブを黙って受けない — `sampler.ts` の `needsUncond`）。
   */
  readonly negativePrompt?: string;
  readonly steps?: number;
  readonly guidanceScale?: number;
  readonly resolution?: ImageSize;
  /** 初期ノイズの seed（既定 0 — 同じ seed なら同じ画像）。 */
  readonly seed?: number;
  /**
   * 生成イベントの観測席（{@link AnimaGenerateEvent}）— 段の開始 / 終了・denoise の 1 step
   * 完了・VAE タイル 1 枚の完了ごとに呼ばれる。
   *
   * **await する**（発火の順序が決定的になり、消費側で間引き / スロットリングができる）。
   * **例外は握らない**（`onRunDiagnostics` と同じ流儀 = fail loudly）— 副産物として
   * **throw が step 粒度の中断手段**になる（生成は reject し、Session は `withSession` の
   * `finally` で解放される）。
   *
   * NOTE: 毎 step の VAE プレビューは提供しない。VAE は DiT を解放した**後**にしかロード
   * できない（VRAM の MUST — モジュール doc）ので、途中結果として渡せるのは生 latent
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
  /** 段の Session 構築の**前**（`start`）と解放の**後**（`end`）— GB 級ロードの進捗が見える。 */
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
  | { readonly kind: "vae-tile"; readonly tile: number; readonly tiles: number };

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
   * 各 `openModel` の間 / GPU 取得の前後）で検査する。入口を除く各境目では**イベントループへ
   * 1 度譲ってから**検査するので、同期解析の最中に届いた中断も次の境目で効く
   * （`options.gpu` を渡して await が 1 つも無い経路でも同じ）。
   * {@link AnimaPipeline.fromPretrained} は同じ 1 本を取得層へも渡すので、**DL と組み立ての
   * どちらの最中でも**同じノブで中断できる（DL 完了後だけ中止ボタンが無反応、を作らない）。
   *
   * 中断の例外は `signal.reason` を**そのまま**投げる（包まない — 消費側が
   * `error === controller.signal.reason` で自分の中断を識別できる）。
   */
  readonly signal?: AbortSignal;
};

/**
 * {@link AnimaPipeline.fromPretrained} が追加で受ける取得層のオプション（hub へ透過する）。
 * `signal` は構築側と共有なので {@link AnimaPipelineOptions} が持つ。
 */
export type AnimaFromPretrainedOptions = AnimaPipelineOptions & {
  /** `Authorization` 等。付けた取得は認証専用のキャッシュ名前空間へ隔離される。 */
  readonly headers?: HeadersInit;
  readonly onProgress?: (progress: AssetProgress) => void;
  readonly onCacheError?: (diagnostic: CacheDiagnostic) => void;
  /** `fetch` の差し替え（テスト・カスタム輸送用）。 */
  readonly fetch?: typeof globalThis.fetch;
  /** `CacheStorage` の差し替え（テスト用）。 */
  readonly caches?: CacheStorage;
};

/** 取得済み資産から直接組むときの入力（hub の `fetchAssets` の返り値をそのまま渡す）。 */
export type AnimaAssets = {
  readonly manifest: Manifest;
  readonly assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>;
};

/**
 * 取得済みバイト列を `openModel` へ渡せる ArrayBuffer にする。
 *
 * MUST: `slice` で写さない — DiT は 1 本 3.7GiB あり、ホスト RAM のピークが倍になる。hub は
 * buffer 全体を占める view を返す契約なので、崩れていたら**取得層の不変条件破れ**として落とす。
 */
const assetBuffer = (
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  key: string,
): ArrayBuffer => {
  if (!Object.hasOwn(assets, key)) {
    throw new Error(
      `anima: 資産 '${key}' が無い（manifest の weights / assets に ${key} が要る）` +
        `（揃っているキー: ${Object.keys(assets).join(" / ")}）`,
    );
  }
  const bytes = assets[key];
  if (bytes.byteOffset !== 0 || bytes.byteLength !== bytes.buffer.byteLength) {
    throw new Error(
      `anima: 資産 '${key}' の bytes が buffer 全体を占めていない` +
        `（byteOffset ${bytes.byteOffset} / byteLength ${bytes.byteLength} /` +
        ` buffer ${bytes.buffer.byteLength}）`,
    );
  }
  return bytes.buffer;
};

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
  model: KarumeModel,
  inputName: string,
): readonly (number | string)[] => {
  const spec = model.graph.inputs.find((input) => input.name === inputName);
  if (spec === undefined) throw new Error(`グラフ入力 '${inputName}' が無い`);
  return spec.shape;
};

const staticInputShape = (model: KarumeModel, inputName: string): readonly number[] =>
  graphInputShape(model, inputName).map((dim, axis) => {
    if (typeof dim !== "number") {
      throw new Error(`グラフ入力 '${inputName}' の軸 ${axis} が静的次元でない（${String(dim)}）`);
    }
    return dim;
  });

const graphOutputShape = (model: KarumeModel): readonly (number | string)[] =>
  model.graph.values[model.graph.outputs[0]].shape;

const staticOutputShape = (model: KarumeModel): readonly number[] => {
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
  if (!wantsUncond && requested !== undefined) {
    throw new Error(
      `guidanceScale ${guidance} では uncond 側を計算しないので negativePrompt は効かない` +
        "（効かせるなら guidanceScale を 1 以外にする）",
    );
  }
  const negativePrompt = requested ?? fallback;
  if (wantsUncond && negativePrompt === undefined) {
    throw new Error(
      `guidanceScale ${guidance} は uncond 側を計算するので negativePrompt が要る` +
        "（manifest の pipelineConfig.defaults.negativePrompt か request で渡す）",
    );
  }
  return negativePrompt;
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
  model: KarumeModel,
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

/**
 * 1 グラフぶんの Session を張り、使い終わったら必ず解放する。
 * MUST: `finally` で dispose する — 途中で落ちたときに VRAM が残ると、後続の段が確保に
 * 失敗して「最初の失敗とは別の場所」で落ちる。
 */
const withSession = async <T>(
  gpu: GpuContext,
  model: KarumeModel,
  sessionOptions: SessionOptions,
  observe: ((diagnostics: SessionDiagnostics) => void) | undefined,
  body: (
    run: (inputs: Record<string, Tensor>) => Promise<Tensor>,
    session: Session,
  ) => Promise<T>,
): Promise<T> => {
  const session = await createSession(gpu, model, sessionOptions);
  try {
    const outputName = model.graph.outputs[0];
    const run = async (inputs: Record<string, Tensor>): Promise<Tensor> => {
      const outputs = await session.run(inputs);
      if (observe !== undefined) observe(session.diagnostics());
      return outputs[outputName];
    };
    return await body(run, session);
  } finally {
    await session.dispose();
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
  readonly textEncoder: KarumeModel;
  readonly textConditioner: KarumeModel;
  readonly transformer: KarumeModel;
  readonly ropeBase: RopeBase;
  readonly vaeDecoder: KarumeModel;
  readonly onRunDiagnostics?: (
    component: AnimaRunComponent,
    diagnostics: SessionDiagnostics,
  ) => void;
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

  private constructor(state: AnimaState) {
    this.#state = state;
  }

  /**
   * HF リポジトリから取得して組む（`loadManifest` → `resolveFiles` → `fetchAssets` →
   * {@link AnimaPipeline.fromAssets} の糖衣）。文字列の `ref` は `{ repo }` と読む（= `main`
   * 追従）。**`ref` は必須**（取得元に既定は無い — `src/hub/repo-ref.ts` の MUST）。
   */
  static async fromPretrained(
    ref: string | HubRepoRef,
    options: AnimaFromPretrainedOptions = {},
  ): Promise<AnimaPipeline> {
    const repoRef = toRepoRef(
      ref,
      "AnimaPipeline.fromPretrained",
      "ANIMA_TURBO_CURRENT / ANIMA_CURRENT（@karume/models/anima）",
    );
    const hubOptions = {
      ...(options.signal === undefined ? {} : { signal: options.signal }),
      ...(options.headers === undefined ? {} : { headers: options.headers }),
      ...(options.onCacheError === undefined ? {} : { onCacheError: options.onCacheError }),
      ...(options.fetch === undefined ? {} : { fetch: options.fetch }),
      ...(options.caches === undefined ? {} : { caches: options.caches }),
    };
    const loaded = await loadManifest(repoRef, hubOptions);
    const selection = {
      ...(options.model === undefined ? {} : { model: options.model }),
      ...(options.quant === undefined ? {} : { quant: options.quant }),
    };
    const files = resolveFiles(loaded.manifest, selection);
    const assets = await fetchAssets(loaded, files, {
      ...hubOptions,
      ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
    });
    // signal は取得層と構築の**両方**へ渡す（DL が終わった瞬間に中断が効かなくなる窓を作らない）。
    return AnimaPipeline.fromAssets({ manifest: loaded.manifest, assets }, {
      ...(options.gpu === undefined ? {} : { gpu: options.gpu }),
      ...selection,
      ...(options.onRunDiagnostics === undefined
        ? {}
        : { onRunDiagnostics: options.onRunDiagnostics }),
      ...(options.signal === undefined ? {} : { signal: options.signal }),
    });
  }

  /**
   * 取得済みの manifest + 資産から組む。**ここが核**で、以下を全て構築時に済ませる:
   *
   * - `model` の選択（未知のモデル名は利用可能な一覧つきで fail loudly — ADR 0041 §8）
   * - `pipeline` の契約名と major の検査（**未知 major は fail loudly** — 検査責務は
   *   models 側。ADR 0038 §1）
   * - `pipelineConfig` の手書きスキーマ検証（未知キーも fail loudly）
   * - quant の `session` → runtime `SessionOptions` の**明示写像**と `gpuFeatures` の解釈
   * - 全 weights / assets の `openModel` / rope 素表 / トークナイザ 2 本の解釈
   *
   * MUST: manifest の契約違反と**資産の解析**は **GPU を取りに行く前**に落とす（他 6 家族と
   * 同じ順序）。順序がずれると、GPU の無い環境では別の例外に化けて「何が悪かったのか」が
   * 読み手に伝わらない。GPU 取得後に許される検査は GPU の能力（shader-f16）だけ。
   * MUST: Session は 1 本も張らない（VRAM の MUST — モジュール doc）。
   *
   * NOTE: 各段は不可分（3.7GiB の `openModel` を途中で畳む口は無い）なので、
   * {@link AnimaPipelineOptions.signal} の検査は**段の境目**にだけ置き、そこで
   * イベントループへ 1 度譲ってから検査する（{@link settleAbort}）— 同期解析の最中に
   * 届いた中断は次の境目で効く（`options.gpu` 供給時も同様）。
   */
  static async fromAssets(
    input: AnimaAssets,
    options: AnimaPipelineOptions = {},
  ): Promise<AnimaPipeline> {
    const { manifest, assets } = input;
    // 中断の検査は**段の境目**に置く（各段は不可分 — 3.7GiB の openModel を途中で畳む口は無い）。
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
    const sessionOptions = toSessionOptions(quant.session);

    // 資産の解析は GPU より前（docstring の順序 MUST）。3.7GiB の DiT を開くほうが device 生成
    // より重いが、壊れた配布形の真因を消さないほうを採る — GPU 無し環境では acquireGpu 自体が
    // 落ちるので、後ろに置くと「資産が無い」が永久に見えない。
    await settleAbort(options.signal);
    const tokenizers = createTokenizers(
      assetBytes(assets, TOKENIZER),
      assetBytes(assets, TOKENIZER_2),
    );
    await settleAbort(options.signal);
    const textEncoder = openModel(assetBuffer(assets, TEXT_ENCODER));
    await settleAbort(options.signal);
    const textConditioner = openModel(assetBuffer(assets, TEXT_CONDITIONER));
    await settleAbort(options.signal);
    const transformer = openModel(assetBuffer(assets, TRANSFORMER));
    const ropeBase = parseRopeBase(assetBuffer(assets, TRANSFORMER_ROPE_BASE));
    await settleAbort(options.signal);
    const vaeDecoder = openModel(assetBuffer(assets, VAE_DECODER));

    await settleAbort(options.signal);

    // MUST: 宣言された feature は device 作成時にしか要求できない（ADR 0028）。共有 GPU を
    // 渡された場合は要求できないので、能力が足りないことを**ここで**名指しして落とす — 通すと
    // Session 構築まで進んでから落ちる（あるいは黙って別の経路へ縮退する）。要求と検査の
    // 網羅表は `session/gpu-features.ts`（7 家族で 1 本）。
    const gpu = options.gpu ?? await acquireGpu(toAcquireGpuOptions(quant.gpuFeatures));
    const ownsGpu = options.gpu === undefined;
    try {
      // MUST: GPU 取得**後**の中断検査は try の中に置く — 外に出すと、内部で取った device を
      // 誰も解放できないまま抜ける（feature 検査と同じ後始末に乗せる）。
      // ここでもマクロタスクへ譲る: `acquireGpu` の await 解決はマイクロタスク継続なので、
      // 待機中に積まれたクリック由来の中断タスクはまだ実行されていない。
      await settleAbort(options.signal);
      assertGpuFeaturesGranted(quant.gpuFeatures, gpu, `AnimaPipeline: quant '${quantName}'`);
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
    assertAcceptableResolution(resolution);
    // seed の検査も入口に置く。生成器を作るのは DiT の段（`new Randn(seed)`）で、そこは
    // text encoder / conditioner を回して DiT の重みを上げ終えた後なので、`Randn` 側だけに
    // 検査があると不正な seed が GB 級のロードの末に落ちる。
    assertAcceptableSeed(seed);
    if (!Number.isInteger(steps) || steps < 2) {
      throw new Error(`steps ${steps} が 2 以上の整数でない（sigma の linspace が組めない）`);
    }
    if (!Number.isFinite(guidance)) throw new Error(`guidanceScale ${guidance} が有限の数でない`);

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
    /** 段 1 本を回す（`stage` を Session 構築の前と解放の後に挟む — 途中で落ちたら `end` は出ない）。 */
    const withStage = async <T>(
      component: AnimaRunComponent,
      model: KarumeModel,
      sessionOptions: SessionOptions,
      body: (
        run: (inputs: Record<string, Tensor>) => Promise<Tensor>,
        session: Session,
      ) => Promise<T>,
    ): Promise<T> => {
      await emit({ kind: "stage", component, at: "start" });
      const result = await withSession(
        state.gpu,
        model,
        sessionOptions,
        observer(state, component),
        body,
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

    // --- ② テキスト経路（DiT ロードの前に解放する）---------------------------
    const hidden = await withStage(
      "text_encoder",
      state.textEncoder,
      {},
      (run) =>
        Promise.all([
          run({ input_ids: idsTensor(positive.qwenIds) }),
          ...(negative === undefined ? [] : [run({ input_ids: idsTensor(negative.qwenIds) })]),
        ]),
    );
    const embeds = await withStage(
      "text_conditioner",
      state.textConditioner,
      {},
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

    // --- ③ denoise（DiT を N step。VAE ロードの前に解放する）-----------------
    // 低精度計算のノブ（quant の session）は **DiT の Session にだけ**効かせる —
    // text 系 / VAE は対象外（比較の軸を DiT 1 本に保つ）。
    const { latents, latentShape } = await withStage(
      "transformer",
      state.transformer,
      state.sessionOptions,
      async (run) => {
        const model = state.transformer;
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
          current = cfgEulerStep(
            current,
            predictions[0],
            predictions[1],
            Math.fround(sigmas[index + 1] - sigmas[index]),
            guidance,
          );
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
    );

    // --- ④ 逆正規化 → VAE decode（常時タイル — ADR 0038 §4）-------------------
    const denormalized = denormalizeLatents(
      latents,
      latentShape,
      ANIMA_LATENTS_MEAN,
      ANIMA_LATENTS_STD,
    );
    const decoded = await withStage(
      "vae_decoder",
      state.vaeDecoder,
      {},
      async (run) => {
        const tileShape = staticInputShape(state.vaeDecoder, "latents");
        const sampleShape = staticOutputShape(state.vaeDecoder);
        const geometry = planVaeTiling(latentShape, tileShape, sampleShape);
        const tiles = tileCount(geometry);
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
          blend: [
            blendExtent(geometry.rows, geometry.scale),
            blendExtent(geometry.cols, geometry.scale),
          ] as const,
        };
      },
    );

    // MUST: 出た画像の寸法はノブではなく**資産**が決める。ここで食い違うなら開いた export が
    // 想定と違うので、黙って返さない。要素数との整合は `imageToRgba` が見る。
    if (decoded.width !== resolution.width || decoded.height !== resolution.height) {
      throw new Error(
        `VAE 出力が ${decoded.width}×${decoded.height} で要求解像度 ${
          formatResolution(resolution)
        } と違う（タイル ${decoded.tiles} 枚 / ブレンド ${decoded.blend.join(",")}px）`,
      );
    }

    const rgba = imageToRgba(decoded.pixels, decoded.width, decoded.height);
    return {
      width: decoded.width,
      height: decoded.height,
      data: new Uint8Array(rgba.buffer),
    };
  }

  /**
   * 解放する。**内部で取得した GPU だけ**破棄する（`options.gpu` で渡された GpuContext は
   * 呼び出し側の所有物なので触らない）。
   *
   * MUST: in-flight の生成の完了を待ってから破棄する（flush-before-destroy）— 破棄も鎖に
   * 載せることで、待ちと破棄の順序を 1 箇所で決める。2 度目以降も同じ完了を返す（先に返すと
   * 呼び出し側が「破棄済み」と見なして次へ進む）。
   */
  dispose(): Promise<void> {
    this.#disposal ??= this.#chain(() => {
      if (this.#state.ownsGpu) this.#state.gpu.destroy();
    });
    return this.#disposal;
  }

  /** `await using` 対応（Explicit Resource Management）— {@link dispose} の別名。 */
  [Symbol.asyncDispose](): Promise<void> {
    return this.dispose();
  }
}
