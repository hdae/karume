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
  type SessionSpec,
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
import { Randn } from "./random.ts";

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
};

/** {@link AnimaPipeline.fromPretrained} だけが使う取得層のオプション（hub へ透過する）。 */
export type AnimaFromPretrainedOptions = AnimaPipelineOptions & {
  readonly signal?: AbortSignal;
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
 * manifest の `session`（3 キー固定の manifest 所有語彙）を runtime の `SessionOptions` へ
 * **1 キーずつ**写す。
 *
 * MUST: スプレッドで丸投げしない。ADR 0038 §3 の要点は「配布済み manifest を runtime 内部の
 * 綴りに釘付けしない」ことで、素通しにすると綴りが変わった瞬間に **runtime が未知キーを
 * 黙って無視して沈黙劣化する**（`s16` が名前だけになる）。写像を明示的に書くと、綴りが
 * 割れた時点で型検査が落ちる。
 *
 * NOTE: barrel には出さない（`export` はパッケージ内テストが写像そのものを叩くため — 綴りの
 * 契約は ADR 0038 §3 が正本で、写像の抜けは GPU を回さないと露見しない位置にある）。
 */
export const toSessionOptions = (spec: SessionSpec): SessionOptions => ({
  ...(spec.linearCompute === undefined ? {} : { linearCompute: spec.linearCompute }),
  ...(spec.attentionCompute === undefined ? {} : { attentionCompute: spec.attentionCompute }),
  ...(spec.attentionScoreStorage === undefined
    ? {}
    : { attentionScoreStorage: spec.attentionScoreStorage }),
});

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
  body: (
    run: (inputs: Record<string, Tensor>) => Promise<Tensor>,
    session: Session,
  ) => Promise<T>,
): Promise<T> => {
  const session = await createSession(gpu, model, sessionOptions);
  try {
    const outputName = model.graph.outputs[0];
    const run = async (inputs: Record<string, Tensor>): Promise<Tensor> =>
      (await session.run(inputs))[outputName];
    return await body(run, session);
  } finally {
    await session.dispose();
  }
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
  #disposed = false;

  private constructor(state: AnimaState) {
    this.#state = state;
  }

  /**
   * HF リポジトリから取得して組む（`loadManifest` → `resolveFiles` → `fetchAssets` →
   * {@link AnimaPipeline.fromAssets} の糖衣）。文字列の `ref` は `{ repo }` と読む。
   */
  static async fromPretrained(
    ref: string | HubRepoRef,
    options: AnimaFromPretrainedOptions = {},
  ): Promise<AnimaPipeline> {
    const repoRef: HubRepoRef = typeof ref === "string" ? { repo: ref } : ref;
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
    return AnimaPipeline.fromAssets({ manifest: loaded.manifest, assets }, {
      ...(options.gpu === undefined ? {} : { gpu: options.gpu }),
      ...selection,
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
   * MUST: Session は 1 本も張らない（VRAM の MUST — モジュール doc）。
   */
  static async fromAssets(
    input: AnimaAssets,
    options: AnimaPipelineOptions = {},
  ): Promise<AnimaPipeline> {
    const { manifest, assets } = input;
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
    const wantsShaderF16 = quant.gpuFeatures?.shaderF16 === true;

    // MUST: `shader-f16` は device 作成時にしか要求できない（ADR 0028）。共有 GPU を渡された
    // 場合は要求できないので、能力が足りないことを**ここで**名指しして落とす — 通すと
    // Session 構築まで進んでから落ちる（あるいは黙って別の経路へ縮退する）。
    const gpu = options.gpu ??
      await acquireGpu(wantsShaderF16 ? { shaderF16: true } : {});
    const ownsGpu = options.gpu === undefined;
    try {
      if (wantsShaderF16 && !gpu.shaderF16Enabled) {
        throw new Error(
          `AnimaPipeline: quant '${quantName}' は shader-f16 を要求するが、渡された` +
            " GpuContext で有効になっていない（acquireGpu({ shaderF16: true }) で取り直す）",
        );
      }
      return new AnimaPipeline({
        gpu,
        ownsGpu,
        config,
        sessionOptions: toSessionOptions(quant.session),
        tokenizers: createTokenizers(
          assetBytes(assets, TOKENIZER),
          assetBytes(assets, TOKENIZER_2),
        ),
        textEncoder: openModel(assetBuffer(assets, TEXT_ENCODER)),
        textConditioner: openModel(assetBuffer(assets, TEXT_CONDITIONER)),
        transformer: openModel(assetBuffer(assets, TRANSFORMER)),
        ropeBase: parseRopeBase(assetBuffer(assets, TRANSFORMER_ROPE_BASE)),
        vaeDecoder: openModel(assetBuffer(assets, VAE_DECODER)),
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
   */
  async generate(request: AnimaGenerateRequest): Promise<GeneratedImage> {
    if (this.#disposed) throw new Error("AnimaPipeline: dispose 済みでは生成できない");
    const state = this.#state;
    const defaults = state.config.defaults;
    const steps = request.steps ?? defaults.steps;
    const guidance = request.guidanceScale ?? defaults.guidanceScale;
    const resolution = request.resolution ?? defaults.resolution;
    const seed = request.seed ?? 0;
    assertAcceptableResolution(resolution);
    if (!Number.isInteger(steps) || steps < 2) {
      throw new Error(`steps ${steps} が 2 以上の整数でない（sigma の linspace が組めない）`);
    }
    if (!Number.isFinite(guidance)) throw new Error(`guidanceScale ${guidance} が有限の数でない`);

    const wantsUncond = needsUncond(guidance);
    // MUST: 効かないノブを黙って受けない。guidance=1 は uncond 分岐を丸ごと計算しないので、
    // ネガティブプロンプトは 1 文字も使われない（指定できたように見えるのが最悪）。
    if (!wantsUncond && request.negativePrompt !== undefined) {
      throw new Error(
        `guidanceScale ${guidance} では uncond 側を計算しないので negativePrompt は効かない` +
          "（効かせるなら guidanceScale を 1 以外にする）",
      );
    }
    const negativePrompt = request.negativePrompt ?? defaults.negativePrompt;
    if (wantsUncond && negativePrompt === undefined) {
      throw new Error(
        `guidanceScale ${guidance} は uncond 側を計算するので negativePrompt が要る` +
          "（manifest の pipelineConfig.defaults.negativePrompt か request で渡す）",
      );
    }

    // --- ① プロンプト層（GPU 不要・決定的）------------------------------------
    const positive = state.tokenizers.encode(request.prompt, "プロンプト");
    const negative = wantsUncond
      ? state.tokenizers.encode(negativePrompt as string, "ネガティブプロンプト")
      : undefined;
    const sigmas = sigmaSchedule(steps, state.config.scheduler.shift);

    // --- ② テキスト経路（DiT ロードの前に解放する）---------------------------
    const hidden = await withSession(state.gpu, state.textEncoder, {}, (run) =>
      Promise.all([
        run({ input_ids: idsTensor(positive.qwenIds) }),
        ...(negative === undefined ? [] : [run({ input_ids: idsTensor(negative.qwenIds) })]),
      ]));
    const embeds = await withSession(state.gpu, state.textConditioner, {}, (run) =>
      Promise.all([
        run({ source_hidden_states: hidden[0], target_input_ids: idsTensor(positive.t5Ids) }),
        ...(negative === undefined ? [] : [
          run({
            source_hidden_states: hidden[1],
            target_input_ids: idsTensor(negative.t5Ids),
          }),
        ]),
      ]));

    // --- ③ denoise（DiT を N step。VAE ロードの前に解放する）-----------------
    // 低精度計算のノブ（quant の session）は **DiT の Session にだけ**効かせる —
    // text 系 / VAE は対象外（比較の軸を DiT 1 本に保つ）。
    const { latents, latentShape } = await withSession(
      state.gpu,
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
    const decoded = await withSession(state.gpu, state.vaeDecoder, {}, async (run) => {
      const tileShape = staticInputShape(state.vaeDecoder, "latents");
      const sampleShape = staticOutputShape(state.vaeDecoder);
      const geometry = planVaeTiling(latentShape, tileShape, sampleShape);
      const pixels = await decodeTiled(denormalized, geometry, async (tile, row, col) => {
        const output = await run({ latents: { dtype: "f32", shape: tileShape, data: tile } });
        return asF32(output, `VAE 出力（タイル ${row},${col}）`);
      });
      // 寸法は**幾何**が正本（latent の全長 × 縮尺）。画素数から逆算しない — 非正方では
      // `3·H·W` の分解が一意でなく、逆算では縦横の取り違えが原理的に検出できない。
      return {
        pixels,
        width: geometry.cols.extent * geometry.scale,
        height: geometry.rows.extent * geometry.scale,
        tiles: tileCount(geometry),
        blend: [
          blendExtent(geometry.rows, geometry.scale),
          blendExtent(geometry.cols, geometry.scale),
        ] as const,
      };
    });

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
   * 呼び出し側の所有物なので触らない）。生成中の Session は段ごとに畳まれているので、
   * ここで残っているものは無い。
   */
  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    if (this.#state.ownsGpu) this.#state.gpu.destroy();
  }

  /** `using` 対応（Explicit Resource Management）— {@link dispose} の別名。 */
  [Symbol.dispose](): void {
    this.dispose();
  }
}
