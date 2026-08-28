/**
 * `BirefnetPipeline` — 画像 → α マット（背景抜き / salient object segmentation）の 1 本の面。
 *
 * パイプライン（全段 Karume・torch 不使用）:
 *
 * 1. ホスト: RGB8 の画素列を `pipelineConfig` の寸法へ resize（antialias 付き bilinear）
 * 2. ホスト: rescale + normalize を畳んだ形で通し、`[1, 3, S, S]` の f32 にする
 * 3. `matte` を 1 回回す → **sigmoid 前の logit** `[1, 1, S, S]`
 * 4. ホスト: `sigmoid` → **元画像の解像度へ戻す** → 8bit の α にする
 *
 * 3 と 4 の順序（sigmoid を先に、resize を後に）は上流の逐語 — 同梱 `handler.py` と
 * モデルカードの利用例がどちらも `preds.sigmoid()` してから `resize` する。logit のまま
 * 補間すると飽和域の外挿が効いて境界の α が変わる。
 *
 * ## MUST: Session は構築時に 1 本だけ張り、`dispose` まで持つ（Anima / Irodori と非対称）
 *
 * あちらが段ごとに張っては畳むのは、複数の巨大グラフを同時常駐させられないから（ADR 0016）。
 * こちらはグラフが 1 本（1024² で 964MB）で畳む相手がおらず、逆に「1 枚ごとに張っては畳む」
 * 形にすると**画像 1 枚あたり重みを丸ごとアップロードし直す**。背景抜きは何枚も通す使い方が
 * 普通なので、常駐が既定の形になる（SigLIP2 と同じ理由・同じ形）。
 *
 * MUST: それでも直列化鎖には載せる — `segment` の同時実行は 1 つの Session を 2 本の `run` で
 * 共有することになり、`dispose` は in-flight の完了を待たずに GPU を破棄する
 * （flush-before-destroy 違反）。
 *
 * ## 返すのは α マット **1 枚だけ**（RGBA 合成も PNG も持たない）
 *
 * {@link BirefnetPipeline.segment} が返すのは「入力と同じ寸法・画素あたり 1 バイトの α」で、
 * 前景を切り抜いた RGBA は返さない。
 *
 * - **合成の方針は用途で変わる**（ストレート α のまま canvas へ渡す / 白地や単色へ合成して
 *   JPEG にする / 色にじみ除去を挟む）。ここで 1 つ選ぶと、他の選び方をする口が消える。
 * - 合成そのものは `rgba[i * 4 + 3] = alpha[i]` の 1 行で、抽象化する規模でもない
 *   （下の「使い方」のとおり）。
 * - `encodePng` は truecolor RGB 専用で、**α が 255 でない画素を受け取らない**
 *   （`src/image/png.ts`）。RGBA を返す口を生やしても、それを書き出す口を同時に足さない限り
 *   Deno では使い道が無い — 「使いやすくする」ための 1 本が 2 本の機能追加になる。
 *
 * 使い方（白地へ合成して PNG にする場合）:
 *
 * ```ts
 * const matte = await pipeline.segment(image);
 * const rgba = new Uint8Array(matte.width * matte.height * 4);
 * for (let i = 0; i < matte.data.length; i += 1) {
 *   const a = matte.data[i] / 255;
 *   for (let c = 0; c < 3; c += 1) {
 *     rgba[i * 4 + c] = Math.round(image.data[i * 3 + c] * a + 255 * (1 - a));
 *   }
 *   rgba[i * 4 + 3] = 255;
 * }
 * await encodePng(rgba, matte.width, matte.height);
 * ```
 *
 * ## MUST: 数値の正はここでは担保されない
 *
 * 正はグラフ単位の golden E2E（`packages/runtime/tests/e2e_birefnet_test.ts` — 実重み 2 系列 ×
 * 合成 4 + 実画像 4 ケース）と、前処理のパリティ門
 * （`packages/models/tests/image_preprocess_test.ts` / 実画像は
 * `packages/models/tests/e2e_birefnet_real_test.ts`）が担保する。
 */

import {
  acquireGpu,
  type GpuContext,
  openModel,
  type Session,
  type SessionDiagnostics,
  type Tensor,
} from "@karume/runtime";
import {
  type AssetProgress,
  type CacheDiagnostic,
  type HubRepoRef,
  loadManifest,
  type Manifest,
  type ModelEntry,
  resolveFiles,
} from "@karume/hub";

import {
  BIREFNET_PIPELINE_MAJOR,
  BIREFNET_PIPELINE_NAME,
  type BirefnetPipelineConfig,
  parseBirefnetPipelineConfig,
} from "./config.ts";
import {
  normalizeToNchw,
  resizePlaneF32,
  resizeRgb8,
  type Rgb8Image,
} from "../image/preprocess.ts";
import { createOperationChain } from "../concurrency/serial.ts";
import { assertGpuFeaturesGranted, toAcquireGpuOptions } from "../session/gpu-features.ts";
import { toSessionOptions } from "../session/options.ts";
import { toRepoRef } from "../hub/repo-ref.ts";
import {
  type ComponentOpener,
  type GraphOwner,
  loadShardComponents,
  type ModelComponent,
  wholeComponent,
} from "../hub/components.ts";

/** manifest の weights 表に現れる取得キー（ADR 0041 §3 の規約名）。 */
const MATTE = "matte";

/** グラフ入力の名前（`export_birefnet.py` の `INPUT_NAME`）。 */
const PIXEL_VALUES = "pixel_values";

/** 入力のチャネル数（RGB）。batch と併せてグラフ側も静的 1 / 3 で焼かれている。 */
const CHANNELS = 3;

/** 出力のチャネル数（マットは 1 枚）。 */
const MATTE_CHANNELS = 1;

/** 8bit の α の最大値。 */
const ALPHA_MAX = 255;

/** 入力と同じ寸法の α マット（画素あたり 1 バイト・行優先）。 */
export type AlphaMatte = {
  readonly data: Uint8Array<ArrayBuffer>;
  readonly width: number;
  readonly height: number;
};

/** 構築オプション（{@link BirefnetPipeline.fromAssets} / {@link BirefnetPipeline.fromPretrained} 共通）。 */
export type BirefnetPipelineOptions = {
  /**
   * 既存の GPU を共有する。**渡した側が所有権を持つ**ので {@link BirefnetPipeline.dispose} は
   * 破棄しない。省略時はパイプラインが内部で `acquireGpu` し、`dispose()` で破棄する。
   */
  readonly gpu?: GpuContext;
  /** モデル（manifest の models のキー）。省略時は `defaultModel`。 */
  readonly model?: string;
  /** 実行構成（そのモデルの quants のキー）。省略時は `defaultQuant`。 */
  readonly quant?: string;
  /**
   * 実行 1 回ごとの診断を受け取る観測席（1 マット = `matte` 1 回）。op 別 GPU 時間
   * （`lastRunTiming`）が要るときは `gpu` に `acquireGpu({ gpuTiming: true })` を渡す
   * （ADR 0021 — 既定は計測しない）。
   *
   * NOTE: SigLIP2 と同じくコンポーネント名を渡さない — グラフが 1 本しかないので、名前が
   * 常に同じ 1 値になる（受け手が分岐できない引数を渡さない）。
   *
   * コールバックの例外は握らない（fail loudly — セグメンテーションごと落ちる）。
   */
  readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics) => void;
};

/** {@link BirefnetPipeline.fromPretrained} だけが使う取得層のオプション（hub へ透過する）。 */
export type BirefnetFromPretrainedOptions = BirefnetPipelineOptions & {
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
export type BirefnetAssets = {
  readonly manifest: Manifest;
  readonly assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>;
};

/**
 * 取得済みバイト列を `openModel` へ渡せる ArrayBuffer にする。
 *
 * MUST: `slice` で写さない — 1024² の配布形は 1 本 964MB あり、ホスト RAM のピークが倍になる。
 * hub は buffer 全体を占める view を返す契約なので、崩れていたら**取得層の不変条件破れ**として
 * 落とす。
 */
const assetBuffer = (
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  key: string,
): ArrayBuffer => {
  if (!Object.hasOwn(assets, key)) {
    throw new Error(
      `birefnet: 資産 '${key}' が無い（manifest の weights に ${key} が要る）` +
        `（揃っているキー: ${Object.keys(assets).join(" / ")}）`,
    );
  }
  const bytes = assets[key];
  if (bytes.byteOffset !== 0 || bytes.byteLength !== bytes.buffer.byteLength) {
    throw new Error(
      `birefnet: 資産 '${key}' の bytes が buffer 全体を占めていない` +
        `（byteOffset ${bytes.byteOffset} / byteLength ${bytes.byteLength} /` +
        ` buffer ${bytes.buffer.byteLength}）`,
    );
  }
  return bytes.buffer;
};

/**
 * 全量面（`fromAssets`）のコンポーネント供給口。取得済みバイト列を `openModel` で開き、Session は
 * 従来どおり全量面で組む（shard 面との違いは「どこからバイト列が来たか」だけ）。
 */
const assetOpener = (assets: BirefnetAssets["assets"]): ComponentOpener => (key) =>
  wholeComponent(openModel(assetBuffer(assets, key)));

/**
 * グラフ入力の 1 軸ぶんの**静的**次元が `pipelineConfig` の宣言と一致することを見る。
 *
 * MUST: 落とさない。前処理は宣言の寸法へ resize するので、グラフが別の解像度で焼かれていても
 * **ホスト側は最後まで通る**（落ちるのは Session の shape 検査で、そのときには「どちらの数が
 * 正しいのか」が読み手に伝わらない）。解像度違いの系列（1024² / 2048²）の資産取り違えは
 * ここが唯一の検出器。
 *
 * NOTE: `export` は GPU 無しで拒否経路を縛るテストのため（`mod.ts` / サブパス面には出さない —
 * ADR 0008）。
 */
export const assertStaticDim = (
  model: GraphOwner,
  inputName: string,
  axis: number,
  expected: number,
  where: string,
): void => {
  const spec = model.graph.inputs.find((input) => input.name === inputName);
  if (spec === undefined) throw new Error(`birefnet: グラフ入力 '${inputName}' が無い（${where}）`);
  const dim = spec.shape[axis];
  if (dim !== expected) {
    throw new Error(
      `birefnet: ${where} — グラフ入力 '${inputName}' の軸 ${axis} が ${String(dim)}、` +
        `pipelineConfig は ${expected}`,
    );
  }
};

/**
 * グラフ**出力**の形が `[1, 1, imageHeight, imageWidth]` であることを見る。
 *
 * MUST: 落とさない。後段は「要素数が H×W」しか見ない形にもできるが、それだと multi-scale
 * supervision の中間予測込みで焼かれたグラフ（`[1, 3, S, S]` 相当）が**別の値を α として**
 * 通り抜ける。
 *
 * NOTE: `export` は GPU 無しで拒否経路を縛るテストのため（`mod.ts` / サブパス面には出さない —
 * ADR 0008）。
 */
export const assertMatteShape = (
  model: GraphOwner,
  config: BirefnetPipelineConfig,
  where: string,
): void => {
  const name = model.graph.outputs[0];
  const value = model.graph.values[name];
  if (value === undefined) {
    throw new Error(`birefnet: グラフ出力 '${name}' の宣言が無い（${where}）`);
  }
  const expected = [1, MATTE_CHANNELS, config.imageHeight, config.imageWidth];
  if (
    value.shape.length !== expected.length ||
    value.shape.some((dim, axis) => dim !== expected[axis])
  ) {
    throw new Error(
      `birefnet: ${where} — グラフ出力 '${name}' の形が ` +
        `[${value.shape.map(String).join(", ")}]、期待は [${expected.join(", ")}]`,
    );
  }
};

const asF32 = (tensor: Tensor, where: string): Float32Array<ArrayBuffer> => {
  if (tensor.dtype !== "f32") throw new Error(`${where}: f32 でない（${tensor.dtype}）`);
  return tensor.data;
};

/**
 * RGB8 の画素列 → グラフへ渡す `pixel_values`（`[3, S, S]` を平らにしたもの）。
 *
 * 前処理そのものは `src/image/preprocess.ts` が持ち、ここがするのは `pipelineConfig` の
 * 定数で結線することだけ。**`resizeRgb8` は `(width, height)` の順**なので、宣言の 2 欄を
 * 取り違えると非正方の配布形で黙って転置される。
 *
 * NOTE: `export` は結線を直接叩くテストのため（`fromAssets` 経由で此処へ届くには実 IR
 * コンテナが要る）。`mod.ts` / サブパス面には出さない（ADR 0008）。
 */
export const preprocessPixelValues = (
  config: BirefnetPipelineConfig,
  image: Rgb8Image,
): Float32Array<ArrayBuffer> =>
  normalizeToNchw(
    resizeRgb8(image, config.imageWidth, config.imageHeight),
    config.imageMean,
    config.imageStd,
  );

/**
 * グラフの出す logit の地図 → 入力と同じ寸法の 8bit α（モジュール doc の段 4）。
 *
 * 段の順序（sigmoid → resize → 量子化）は上流の逐語。8bit へ落とすのは**この配布形の出力
 * 形式の決定**であって上流との一致の主張ではない（上流の利用例は `astype("uint8")` =
 * 切り捨てだが、こちらは最近傍へ丸める — α の 1 LSB は目に見えず、切り捨ては系統的に
 * 前景を薄くする）。
 *
 * NOTE: `export` はテストのため（{@link preprocessPixelValues} と同じ理由）。
 */
export const matteFromLogits = (
  logits: Float32Array,
  sourceWidth: number,
  sourceHeight: number,
  width: number,
  height: number,
): AlphaMatte => {
  const alpha = new Float32Array(logits.length);
  for (let index = 0; index < logits.length; index += 1) {
    const raw = logits[index];
    // MUST: 非有限値を黙って返さない。α の量子化（`Uint8Array` 代入）は NaN を 0 =
    // 「完全に透明な画素」という**正常に見えるマット**へ変換してしまう。resize は重み付き和で
    // NaN を伝播するだけなので、発生源に一番近いここで落とす（`anima/image.ts` と同じ検査点）。
    if (!Number.isFinite(raw)) {
      const x = index % sourceWidth;
      const y = Math.floor(index / sourceWidth);
      throw new Error(`birefnet: マットの logit (x=${x}, y=${y}) が非有限値`);
    }
    alpha[index] = 1 / (1 + Math.exp(-raw));
  }
  const scaled = resizePlaneF32(alpha, sourceWidth, sourceHeight, width, height);
  const data = new Uint8Array(scaled.length) as Uint8Array<ArrayBuffer>;
  for (let index = 0; index < scaled.length; index += 1) {
    data[index] = Math.round(scaled[index] * ALPHA_MAX);
  }
  return { data, width, height };
};

/** {@link BirefnetPipeline} の内部状態（公開面には出さない）。 */
type BirefnetState = {
  readonly gpu: GpuContext;
  readonly ownsGpu: boolean;
  readonly config: BirefnetPipelineConfig;
  readonly matte: ModelComponent;
  /** 構築時に張って `dispose` まで持つ 1 本（モジュール doc の MUST）。 */
  readonly session: Session;
  readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics) => void;
};

/**
 * manifest + 資産から実行状態を組む（{@link BirefnetPipeline.fromAssets} の中身）。
 *
 * MUST: manifest の契約違反と**資産の解析・グラフとの突合**は **GPU を取りに行く前**に落とす。
 * 順序がずれると、GPU の無い環境では別の例外に化けて「何が悪かったのか」が読み手に
 * 伝わらない。GPU 取得後に許される検査は GPU の能力（shader-f16）だけ（ADR 0028）。
 */
const openBirefnetState = async (
  input: BirefnetAssets,
  open: ComponentOpener,
  options: BirefnetPipelineOptions = {},
): Promise<BirefnetState> => {
  const { manifest } = input;
  const modelName = options.model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, modelName)) {
    throw new Error(
      `BirefnetPipeline: model '${modelName}' は manifest に無い` +
        `（利用可能: ${manifest.available.models.join(" / ")}）`,
    );
  }
  const entry: ModelEntry = manifest.models[modelName];
  const { name, major } = entry.pipeline;
  if (name !== BIREFNET_PIPELINE_NAME) {
    throw new Error(
      `BirefnetPipeline: manifest の pipeline が '${name}/${major}'` +
        `（'${BIREFNET_PIPELINE_NAME}/${BIREFNET_PIPELINE_MAJOR}' が必要）`,
    );
  }
  if (major !== BIREFNET_PIPELINE_MAJOR) {
    // 「古い実装 × 新しいリポ」の沈黙劣化を止める唯一の門（ADR 0038 §6）。
    throw new Error(
      `BirefnetPipeline: pipeline '${name}/${major}' の major に未対応` +
        `（この実装が読めるのは ${BIREFNET_PIPELINE_NAME}/${BIREFNET_PIPELINE_MAJOR}）`,
    );
  }
  const config = parseBirefnetPipelineConfig(entry.pipelineConfig);

  const quantName = options.quant ?? entry.defaultQuant;
  if (!Object.hasOwn(entry.quants, quantName)) {
    throw new Error(
      `BirefnetPipeline: quant '${quantName}' は manifest に無い` +
        `（利用可能: ${entry.available.quants.join(" / ")}）`,
    );
  }
  const quant = entry.quants[quantName];

  const matte = open(MATTE);
  // グラフの宣言と pipelineConfig の突合。入出力が 1 本ずつであることまで見るのは、
  // multi-scale supervision の中間予測込みで焼かれたグラフが混ざると、位置で引く後段が
  // 黙って別の値を α として読むため。
  if (matte.graph.inputs.length !== 1 || matte.graph.outputs.length !== 1) {
    throw new Error(
      `BirefnetPipeline: matte グラフの入出力が ${matte.graph.inputs.length} /` +
        ` ${matte.graph.outputs.length} 本（どちらも 1 本のマットが必要）`,
    );
  }
  assertStaticDim(matte, PIXEL_VALUES, 0, 1, "batch は静的 1");
  assertStaticDim(matte, PIXEL_VALUES, 1, CHANNELS, "RGB の 3 チャネル");
  assertStaticDim(matte, PIXEL_VALUES, 2, config.imageHeight, "imageHeight");
  assertStaticDim(matte, PIXEL_VALUES, 3, config.imageWidth, "imageWidth");
  assertMatteShape(matte, config, "マットの形");

  // MUST: 宣言された feature は device 作成時にしか要求できない（ADR 0028）。共有 GPU を
  // 渡された場合は要求できないので、能力が足りないことを**ここで**名指しして落とす。要求と
  // 検査の網羅表は `session/gpu-features.ts`（7 家族で 1 本）。
  const gpu = options.gpu ?? await acquireGpu(toAcquireGpuOptions(quant.gpuFeatures));
  const ownsGpu = options.gpu === undefined;
  try {
    assertGpuFeaturesGranted(quant.gpuFeatures, gpu, `BirefnetPipeline: quant '${quantName}'`);
    return {
      gpu,
      ownsGpu,
      config,
      matte,
      session: await matte.createSession(gpu, toSessionOptions(quant.session)),
      ...(options.onRunDiagnostics === undefined
        ? {}
        : { onRunDiagnostics: options.onRunDiagnostics }),
    };
  } catch (error) {
    // 内部で取った GPU は、構築に失敗したら誰も解放できなくなるのでここで返す。
    if (ownsGpu) gpu.destroy();
    throw error;
  }
};

/** 画像 1 枚を通す（前処理 → `matte` 1 回 → sigmoid → 元解像度の 8bit α）。 */
const segmentImage = async (state: BirefnetState, image: Rgb8Image): Promise<AlphaMatte> => {
  const { config } = state;
  const pixelValues: Tensor = {
    dtype: "f32",
    shape: [1, CHANNELS, config.imageHeight, config.imageWidth],
    data: preprocessPixelValues(config, image),
  };
  const outputs = await state.session.run({ [PIXEL_VALUES]: pixelValues });
  if (state.onRunDiagnostics !== undefined) state.onRunDiagnostics(state.session.diagnostics());
  const name = state.matte.graph.outputs[0];
  const tensor = outputs[name];
  if (tensor === undefined) throw new Error(`birefnet: グラフ出力 '${name}' が実行結果に無い`);
  const logits = asF32(tensor, "matte");
  const expected = config.imageWidth * config.imageHeight;
  if (logits.length !== expected) {
    throw new Error(
      `birefnet: マットの要素数 ${logits.length} が ` +
        `${config.imageWidth}×${config.imageHeight} と違う`,
    );
  }
  return matteFromLogits(
    logits,
    config.imageWidth,
    config.imageHeight,
    image.width,
    image.height,
  );
};

/**
 * BiRefNet 系の画像 → α マットのパイプライン（グラフ 1 本）。
 *
 * 構築は {@link BirefnetPipeline.fromPretrained}（HF から取得）か
 * {@link BirefnetPipeline.fromAssets}（取得済みバイト列）だけを入口にする — コンストラクタを
 * private にしてあるのは、manifest 検査と資産の突合を迂回した半端な状態を作れないようにする
 * ため（ADR 0008）。
 */
export class BirefnetPipeline {
  readonly #state: BirefnetState;
  /** segment と dispose の直列化鎖（「1 本ずつ」を公開 API 側で守る — モジュール doc）。 */
  readonly #chain = createOperationChain();
  /**
   * dispose の 1 本。**undefined でないことが「dispose 済み」**（別に真偽値を持つと、独立に
   * 更新される派生状態になる）。
   */
  #disposal: Promise<void> | undefined;

  private constructor(state: BirefnetState) {
    this.#state = state;
  }

  /**
   * HF リポジトリから取得して組む（`loadManifest` → `resolveFiles` → **各コンポーネントの
   * グラフ shard だけ**を取って `prepareModel` → 残り資産の `fetchAssets` → 構築）。重み shard は
   * Session を組むときに 1 本ずつ流れる（ADR 0070 — `src/hub/components.ts`）。文字列の
   * `ref` は `{ repo }` と読む（= `main` 追従）。**`ref` は必須**（取得元に既定は無い —
   * `src/hub/repo-ref.ts` の MUST。このファミリは公開配布リポを持たないので pin 定数も無い）。
   */
  static async fromPretrained(
    ref: string | HubRepoRef,
    options: BirefnetFromPretrainedOptions = {},
  ): Promise<BirefnetPipeline> {
    const repoRef = toRepoRef(ref, "BirefnetPipeline.fromPretrained");
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
    const { open, assets } = await loadShardComponents(
      "BirefnetPipeline.fromPretrained",
      loaded,
      files,
      [MATTE],
      {
        ...hubOptions,
        ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
      },
    );
    return new BirefnetPipeline(
      await openBirefnetState({ manifest: loaded.manifest, assets }, open, {
        ...(options.gpu === undefined ? {} : { gpu: options.gpu }),
        ...selection,
        ...(options.onRunDiagnostics === undefined
          ? {}
          : { onRunDiagnostics: options.onRunDiagnostics }),
      }),
    );
  }

  /**
   * 取得済みの manifest + 資産から組む。契約検査・資産の解釈・`openModel`・グラフとの突合を
   * 全てここで済ませ、**`matte` の Session を 1 本張って**返す（モジュール doc の MUST）。
   */
  static async fromAssets(
    input: BirefnetAssets,
    options: BirefnetPipelineOptions = {},
  ): Promise<BirefnetPipeline> {
    return new BirefnetPipeline(await openBirefnetState(input, assetOpener(input.assets), options));
  }

  /**
   * 画像 1 枚の α マットを出す。返るのは**入力と同じ寸法**の 8bit α（モジュール doc の
   * 「返すのは α マット 1 枚だけ」）。
   *
   * 入力は RGB8 の画素列（行優先・画素あたり 3 バイト）で、resize / rescale / normalize は
   * この中で配布形の定数どおりに掛かる。アスペクト比は保たない（上流の前処理が高さ・幅の
   * 対へそのまま伸縮する形で、crop も pad も無いため）。
   *
   * 並行に呼ばれた場合は**待たされて順に**走る（1 つの Session を 2 本の run で共有しない）。
   */
  async segment(image: Rgb8Image): Promise<AlphaMatte> {
    // dispose 済みの判定は呼び出し時点で行う（鎖の中で見ると、dispose より前に受けた
    // セグメンテーションまで巻き添えで落ちる）。
    if (this.#disposal !== undefined) {
      throw new Error("BirefnetPipeline: dispose 済みでは切り抜けない");
    }
    return await this.#chain(() => segmentImage(this.#state, image));
  }

  /**
   * 解放する。Session は必ず畳み、**内部で取得した GPU だけ**破棄する（`options.gpu` で
   * 渡された GpuContext は呼び出し側の所有物なので触らない）。
   *
   * MUST: in-flight の実行の完了を待ってから破棄する（flush-before-destroy）— 破棄も鎖に
   * 載せることで、待ちと破棄の順序を 1 箇所で決める。2 度目以降も同じ完了を返す（先に返すと
   * 呼び出し側が「破棄済み」と見なして次へ進む）。
   */
  dispose(): Promise<void> {
    this.#disposal ??= this.#chain(async () => {
      await this.#state.session.dispose();
      if (this.#state.ownsGpu) this.#state.gpu.destroy();
    });
    return this.#disposal;
  }

  /** `await using` 対応（Explicit Resource Management）— {@link dispose} の別名。 */
  [Symbol.asyncDispose](): Promise<void> {
    return this.dispose();
  }
}
