/**
 * `Siglip2Pipeline` — 画像 → 埋め込みベクトル（SigLIP2 の vision tower）の 1 本の面。
 *
 * パイプライン（全段 Karume・torch 不使用）:
 *
 * 1. ホスト: RGB8 の画素列を `pipelineConfig` の寸法へ resize（antialias 付き bilinear）
 * 2. ホスト: rescale + normalize を畳んだ形で通し、`[1, 3, H, W]` の f32 にする
 * 3. `vision` を 1 回回す → `pooler_output`（MAP head 経由の `[1, hidden]`）
 *
 * 前処理 2 段の実装と参照実装は `src/image/preprocess.ts`（`resizeRgb8` / `normalizeToNchw`）で、
 * ここはそれを `pipelineConfig` の定数で結線するだけ。**decode は karume の責務ではない** —
 * 入口は「RGB8 の画素列 + 幅 + 高さ」で、PNG / JPEG のデコーダは持たない（ブラウザなら
 * `createImageBitmap` + canvas、Deno なら任意のデコーダ）。
 *
 * ## MUST: Session は構築時に 1 本だけ張り、`dispose` まで持つ（他ファミリと非対称）
 *
 * Anima / Irodori が段ごとに張っては畳むのは、複数の巨大グラフを同時常駐させられないから
 * （ADR 0016）。こちらはグラフが 1 本（base 372MB / so400m 1.71GB）で畳む相手がおらず、
 * 逆に「1 枚ごとに張っては畳む」形にすると**画像 1 枚あたり重みを丸ごとアップロードし直す**。
 * 埋め込みは何枚も通す使い方が普通なので、常駐が既定の形になる。
 *
 * MUST: それでも直列化鎖には載せる — `embed` の同時実行は 1 つの Session を 2 本の `run` で
 * 共有することになり、`dispose` は in-flight の完了を待たずに GPU を破棄する
 * （flush-before-destroy 違反）。
 *
 * ## 出力は `pooler_output` **そのもの**（L2 正規化も cosine も持たない）
 *
 * SigLIP2 の pooler_output は上流でも正規化されていない生のベクトル（実測 L2 ノルム
 * 12.7〜13.1）で、
 *
 * - 正規化するかどうかは**下流の用途で変わる**（cosine 検索なら正規化・線形プローブの入力
 *   なら生のまま）。ここで畳むと、生のベクトルを得る口が消える。
 * - SigLIP 本来の類似度は text tower の `logit_scale` / `logit_bias` を通した値で、この
 *   配布形は vision tower しか持たない。**「正しい類似度」をここから出すことはできない**ので、
 *   それらしい `similarity()` を生やすと正しくない意味を約束することになる。
 * - 呼び出し側の実装は `v.map(x => x / norm)` の 2 行で、抽象化する規模でもない。
 *
 * したがって {@link Siglip2Pipeline.embed} は `pooler_output` を素で返す（ADR 0008 の「薄い面」
 * — 利用者ストーリーは「画像を埋め込む」まで）。
 *
 * ## MUST: 数値の正はここでは担保されない
 *
 * 正はグラフ単位の golden E2E（`packages/runtime/tests/e2e_siglip2_test.ts` — 実重み 2 系列 ×
 * 4 ケース）と、前処理のパリティ門（`packages/models/tests/image_preprocess_test.ts`）が
 * 担保する。
 */

import {
  acquireGpu,
  type GpuContext,
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
  type Quant,
  resolveFiles,
} from "@karume/hub";

import {
  parseSiglip2PipelineConfig,
  SIGLIP2_PIPELINE_MAJOR,
  SIGLIP2_PIPELINE_NAME,
  type Siglip2PipelineConfig,
} from "./config.ts";
import { normalizeToNchw, resizeRgb8, type Rgb8Image } from "../image/preprocess.ts";
import { createOperationChain } from "../concurrency/serial.ts";
import { assertGpuFeaturesGranted, toAcquireGpuOptions } from "../session/gpu-features.ts";
import { toSessionOptions } from "../session/options.ts";
import { toRepoRef } from "../hub/repo-ref.ts";
import {
  assetComponentOpener,
  type ComponentOpener,
  type GraphOwner,
  loadShardComponents,
  type ModelComponent,
} from "../hub/components.ts";

/** manifest の weights 表に現れる取得キー（ADR 0041 §3 の規約名）。 */
const VISION = "vision";

/** グラフ入力の名前（`export_siglip2.py` の `INPUT_NAME`）。 */
const PIXEL_VALUES = "pixel_values";

/** 入力のチャネル数（RGB）。batch と併せてグラフ側も静的 1 / 3 で焼かれている。 */
const CHANNELS = 3;

/** 構築オプション（{@link Siglip2Pipeline.fromAssets} / {@link Siglip2Pipeline.fromPretrained} 共通）。 */
export type Siglip2PipelineOptions = {
  /**
   * 既存の GPU を共有する。**渡した側が所有権を持つ**ので {@link Siglip2Pipeline.dispose} は
   * 破棄しない。省略時はパイプラインが内部で `acquireGpu` し、`dispose()` で破棄する。
   */
  readonly gpu?: GpuContext;
  /** モデル（manifest の models のキー）。省略時は `defaultModel`。 */
  readonly model?: string;
  /** 実行構成（そのモデルの quants のキー）。省略時は `defaultQuant`。 */
  readonly quant?: string;
  /**
   * 実行 1 回ごとの診断を受け取る観測席（1 埋め込み = `vision` 1 回）。op 別 GPU 時間
   * （`lastRunTiming`）が要るときは `gpu` に `acquireGpu({ gpuTiming: true })` を渡す
   * （ADR 0021 — 既定は計測しない）。
   *
   * NOTE: 他ファミリと違ってコンポーネント名を渡さない — グラフが 1 本しかないので、名前が
   * 常に同じ 1 値になる（受け手が分岐できない引数を渡さない）。
   *
   * コールバックの例外は握らない（fail loudly — 埋め込みごと落ちる）。
   */
  readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics) => void;
};

/** {@link Siglip2Pipeline.fromPretrained} だけが使う取得層のオプション（hub へ透過する）。 */
export type Siglip2FromPretrainedOptions = Siglip2PipelineOptions & {
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
export type Siglip2Assets = {
  readonly manifest: Manifest;
  readonly assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>;
};

/**
 * 取得済みバイト列を `openModel` へ渡せる ArrayBuffer にする。
 *
 * MUST: `slice` で写さない — so400m は 1 本 1.71GB あり、ホスト RAM のピークが倍になる。hub は
 * buffer 全体を占める view を返す契約なので、崩れていたら**取得層の不変条件破れ**として落とす。
 */
const assetBuffer = (
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  key: string,
): ArrayBuffer => {
  if (!Object.hasOwn(assets, key)) {
    throw new Error(
      `siglip2: 資産 '${key}' が無い（manifest の weights に ${key} が要る）` +
        `（揃っているキー: ${Object.keys(assets).join(" / ")}）`,
    );
  }
  const bytes = assets[key];
  if (bytes.byteOffset !== 0 || bytes.byteLength !== bytes.buffer.byteLength) {
    throw new Error(
      `siglip2: 資産 '${key}' の bytes が buffer 全体を占めていない` +
        `（byteOffset ${bytes.byteOffset} / byteLength ${bytes.byteLength} /` +
        ` buffer ${bytes.buffer.byteLength}）`,
    );
  }
  return bytes.buffer;
};

/**
 * 全量面（`fromAssets`）のコンポーネント供給口（受け口の実装は 7 家族共有 —
 * {@link assetComponentOpener}）。素の 1 本は `openModel` で開いて全量面で組み、shard 分割形
 * （`vision[0]` / `vision[1]` / …）は `fromPretrained` と同じ shard 逐次面へ流す。
 */
const assetOpener = (assets: Siglip2Assets["assets"]): ComponentOpener =>
  assetComponentOpener("siglip2", assets, (key) => assetBuffer(assets, key));

/**
 * グラフ入力の 1 軸ぶんの**静的**次元が `pipelineConfig` の宣言と一致することを見る。
 *
 * MUST: 落とさない。前処理は宣言の寸法へ resize するので、グラフが別の解像度で焼かれていても
 * **ホスト側は最後まで通る**（落ちるのは Session の shape 検査で、そのときには「どちらの数が
 * 正しいのか」が読み手に伝わらない）。base と so400m の資産取り違えはここが唯一の検出器。
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
  if (spec === undefined) throw new Error(`siglip2: グラフ入力 '${inputName}' が無い（${where}）`);
  const dim = spec.shape[axis];
  if (dim !== expected) {
    throw new Error(
      `siglip2: ${where} — グラフ入力 '${inputName}' の軸 ${axis} が ${String(dim)}、` +
        `pipelineConfig は ${expected}`,
    );
  }
};

/**
 * グラフ**出力**の 1 軸が宣言どおりの**静的**次元であることを見る。
 *
 * MUST: 落とさない。`hiddenDim` は前処理にも実行にも使われないので、ずれても埋め込みは
 * 何事もなく出る（読み手が「768 次元だ」と思って 1152 次元を配る形になる）。
 *
 * NOTE: `export` は GPU 無しで拒否経路を縛るテストのため（`mod.ts` / サブパス面には出さない —
 * ADR 0008）。
 */
export const assertOutputDim = (
  model: GraphOwner,
  axis: number,
  expected: number,
  where: string,
): void => {
  const name = model.graph.outputs[0];
  const value = model.graph.values[name];
  if (value === undefined) {
    throw new Error(`siglip2: グラフ出力 '${name}' の宣言が無い（${where}）`);
  }
  const dim = value.shape[axis];
  if (dim !== expected) {
    throw new Error(
      `siglip2: ${where} — グラフ出力 '${name}' の軸 ${axis} が ${String(dim)}、` +
        `pipelineConfig は ${expected}`,
    );
  }
};

const asF32 = (tensor: Tensor, where: string): Float32Array<ArrayBuffer> => {
  if (tensor.dtype !== "f32") throw new Error(`${where}: f32 でない（${tensor.dtype}）`);
  return tensor.data;
};

/**
 * RGB8 の画素列 → グラフへ渡す `pixel_values`（`[3, H, W]` を平らにしたもの）。
 *
 * 前処理そのものは `src/image/preprocess.ts` が持ち、ここがするのは `pipelineConfig` の
 * 定数で結線することだけ。**`resizeRgb8` は `(width, height)` の順**なので、宣言の 2 欄を
 * 取り違えると非正方の配布形で黙って転置される。
 *
 * NOTE: `export` は結線を直接叩くテストのため（`fromAssets` 経由で此処へ届くには実 IR
 * コンテナが要る）。`mod.ts` / サブパス面には出さない（ADR 0008）。
 */
export const preprocessPixelValues = (
  config: Siglip2PipelineConfig,
  image: Rgb8Image,
): Float32Array<ArrayBuffer> =>
  normalizeToNchw(
    resizeRgb8(image, config.imageWidth, config.imageHeight),
    config.imageMean,
    config.imageStd,
  );

/** {@link Siglip2Pipeline} の内部状態（公開面には出さない）。 */
type Siglip2State = {
  readonly gpu: GpuContext;
  readonly ownsGpu: boolean;
  readonly config: Siglip2PipelineConfig;
  readonly vision: ModelComponent;
  /** 構築時に張って `dispose` まで持つ 1 本（モジュール doc の MUST）。 */
  readonly session: Session;
  readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics) => void;
};

/** 家族 admission（{@link admitSiglip2}）が確定させる材料。 */
type Siglip2Admission = {
  readonly config: Siglip2PipelineConfig;
  readonly quantName: string;
  readonly quant: Quant;
  readonly vision: ModelComponent;
};

/**
 * この manifest とこのグラフを siglip2 として実行できるかを見る（`hub/components.ts` の
 * {@link FamilyAdmission} 席 — shard 面では**重み shard を 1 バイトも取る前**に呼ばれる）。
 *
 * MUST: 家族の門はこの 1 本に集める。後段へ散らすと、shard 面では GB 級の重みを落とした
 * **後**にしか落ちない（ADR 0070 決定 5 の文面より実装が狭くなる）。
 * MUST: manifest の契約違反と**グラフとの突合**は **GPU を取りに行く前**に落とす。順序が
 * ずれると、GPU の無い環境では別の例外に化けて「何が悪かったのか」が読み手に伝わらない。
 */
const admitSiglip2 = (
  manifest: Manifest,
  open: ComponentOpener,
  options: Siglip2PipelineOptions,
): Siglip2Admission => {
  const modelName = options.model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, modelName)) {
    throw new Error(
      `Siglip2Pipeline: model '${modelName}' は manifest に無い` +
        `（利用可能: ${manifest.available.models.join(" / ")}）`,
    );
  }
  const entry: ModelEntry = manifest.models[modelName];
  const { name, major } = entry.pipeline;
  if (name !== SIGLIP2_PIPELINE_NAME) {
    throw new Error(
      `Siglip2Pipeline: manifest の pipeline が '${name}/${major}'` +
        `（'${SIGLIP2_PIPELINE_NAME}/${SIGLIP2_PIPELINE_MAJOR}' が必要）`,
    );
  }
  if (major !== SIGLIP2_PIPELINE_MAJOR) {
    // 「古い実装 × 新しいリポ」の沈黙劣化を止める唯一の門（ADR 0038 §6）。
    throw new Error(
      `Siglip2Pipeline: pipeline '${name}/${major}' の major に未対応` +
        `（この実装が読めるのは ${SIGLIP2_PIPELINE_NAME}/${SIGLIP2_PIPELINE_MAJOR}）`,
    );
  }
  const config = parseSiglip2PipelineConfig(entry.pipelineConfig);

  const quantName = options.quant ?? entry.defaultQuant;
  if (!Object.hasOwn(entry.quants, quantName)) {
    throw new Error(
      `Siglip2Pipeline: quant '${quantName}' は manifest に無い` +
        `（利用可能: ${entry.available.quants.join(" / ")}）`,
    );
  }
  const quant = entry.quants[quantName];

  const vision = open(VISION);
  // グラフの宣言と pipelineConfig の突合。入出力が 1 本ずつであることまで見るのは、text tower
  // 込みの別グラフや golden 用の多出力版が混ざると、位置で引く後段が黙って別の値を読むため。
  if (vision.graph.inputs.length !== 1 || vision.graph.outputs.length !== 1) {
    throw new Error(
      `Siglip2Pipeline: vision グラフの入出力が ${vision.graph.inputs.length} /` +
        ` ${vision.graph.outputs.length} 本（どちらも 1 本の vision tower が必要）`,
    );
  }
  assertStaticDim(vision, PIXEL_VALUES, 0, 1, "batch は静的 1");
  assertStaticDim(vision, PIXEL_VALUES, 1, CHANNELS, "RGB の 3 チャネル");
  assertStaticDim(vision, PIXEL_VALUES, 2, config.imageHeight, "imageHeight");
  assertStaticDim(vision, PIXEL_VALUES, 3, config.imageWidth, "imageWidth");
  assertOutputDim(vision, 1, config.hiddenDim, "hiddenDim");

  // MUST: 共有 GPU の能力不足はこの席で落とす — 自前で取る場合と違って `acquireGpu` を
  // 待つ理由が無く、重みを落とす前に判る唯一の家族門（要求と検査の網羅表は
  // `session/gpu-features.ts` の 1 本で、後段の検査も同じ関数を呼ぶ）。
  if (options.gpu !== undefined) {
    assertGpuFeaturesGranted(
      quant.gpuFeatures,
      options.gpu,
      `Siglip2Pipeline: quant '${quantName}'`,
    );
  }

  return { config, quantName, quant, vision };
};

/**
 * admission を通った材料から実行状態を組む（GPU 取得と Session 構築だけ）。
 *
 * MUST: GPU 取得後に許される検査は GPU の能力（shader-f16）だけ（ADR 0028）。
 */
const openSiglip2State = async (
  admitted: Siglip2Admission,
  options: Siglip2PipelineOptions = {},
): Promise<Siglip2State> => {
  const { config, quant, quantName, vision } = admitted;
  // MUST: 宣言された feature は device 作成時にしか要求できない（ADR 0028）。共有 GPU を
  // 渡された場合は要求できないので、能力が足りないことを名指しして落とす（共有 GPU は
  // {@link admitSiglip2} が既に同じ 1 本で見ているが、自前で取った device はここが唯一の門）。
  const gpu = options.gpu ?? await acquireGpu(toAcquireGpuOptions(quant.gpuFeatures));
  const ownsGpu = options.gpu === undefined;
  try {
    assertGpuFeaturesGranted(quant.gpuFeatures, gpu, `Siglip2Pipeline: quant '${quantName}'`);
    return {
      gpu,
      ownsGpu,
      config,
      vision,
      session: await vision.createSession(gpu, toSessionOptions(quant.session)),
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

/** 画像 1 枚を通す（前処理 → `vision` 1 回 → `pooler_output`）。 */
const embedImage = async (
  state: Siglip2State,
  image: Rgb8Image,
): Promise<Float32Array<ArrayBuffer>> => {
  const { config } = state;
  const pixelValues: Tensor = {
    dtype: "f32",
    shape: [1, CHANNELS, config.imageHeight, config.imageWidth],
    data: preprocessPixelValues(config, image),
  };
  const outputs = await state.session.run({ [PIXEL_VALUES]: pixelValues });
  if (state.onRunDiagnostics !== undefined) state.onRunDiagnostics(state.session.diagnostics());
  const name = state.vision.graph.outputs[0];
  const tensor = outputs[name];
  if (tensor === undefined) throw new Error(`siglip2: グラフ出力 '${name}' が実行結果に無い`);
  const data = asF32(tensor, "pooler_output");
  if (data.length !== config.hiddenDim) {
    throw new Error(
      `siglip2: pooler_output の要素数 ${data.length} が hiddenDim ${config.hiddenDim} と違う`,
    );
  }
  return data;
};

/**
 * SigLIP2 の画像 → 埋め込みパイプライン（vision tower 1 本）。
 *
 * 構築は {@link Siglip2Pipeline.fromPretrained}（HF から取得）か
 * {@link Siglip2Pipeline.fromAssets}（取得済みバイト列）だけを入口にする — コンストラクタを
 * private にしてあるのは、manifest 検査と資産の突合を迂回した半端な状態を作れないようにする
 * ため（ADR 0008）。
 */
export class Siglip2Pipeline {
  readonly #state: Siglip2State;
  /** embed と dispose の直列化鎖（「1 本ずつ」を公開 API 側で守る — モジュール doc）。 */
  readonly #chain = createOperationChain();
  /**
   * dispose の 1 本。**undefined でないことが「dispose 済み」**（別に真偽値を持つと、独立に
   * 更新される派生状態になる）。
   */
  #disposal: Promise<void> | undefined;

  private constructor(state: Siglip2State) {
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
    options: Siglip2FromPretrainedOptions = {},
  ): Promise<Siglip2Pipeline> {
    const repoRef = toRepoRef(ref, "Siglip2Pipeline.fromPretrained");
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
    const buildOptions: Siglip2PipelineOptions = {
      ...(options.gpu === undefined ? {} : { gpu: options.gpu }),
      ...selection,
      ...(options.onRunDiagnostics === undefined
        ? {}
        : { onRunDiagnostics: options.onRunDiagnostics }),
    };
    // 家族の門は admission 席で通す（重み shard を取る前 — `hub/components.ts`）。
    const { admitted } = await loadShardComponents(
      "Siglip2Pipeline.fromPretrained",
      loaded,
      files,
      [VISION],
      (open) => admitSiglip2(loaded.manifest, open, buildOptions),
      {
        ...hubOptions,
        ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
      },
    );
    return new Siglip2Pipeline(await openSiglip2State(admitted, buildOptions));
  }

  /**
   * 取得済みの manifest + 資産から組む。契約検査・資産の解釈・`openModel`・グラフとの突合を
   * 全てここで済ませ、**`vision` の Session を 1 本張って**返す（モジュール doc の MUST）。
   *
   * 取得キーは `resolveFiles` の規約どおり **2 形とも受ける** — 素の 1 本（`vision`）と、
   * 1GiB 超のコンポーネントの shard 分割形（`vision[0]` / `vision[1]` / …）。分割形は
   * バイト列を連結せず `fromPretrained` と同じ shard 逐次面へ流す。添字の欠番と素キーとの混在は
   * fail loudly（受け口の実装は `src/hub/components.ts` の 1 本）。
   */
  static async fromAssets(
    input: Siglip2Assets,
    options: Siglip2PipelineOptions = {},
  ): Promise<Siglip2Pipeline> {
    const admitted = admitSiglip2(input.manifest, assetOpener(input.assets), options);
    return new Siglip2Pipeline(await openSiglip2State(admitted, options));
  }

  /**
   * 画像 1 枚を埋め込む。返るのは `pooler_output` **そのもの**（長さ `hiddenDim` の f32・
   * L2 正規化していない — モジュール doc の判断）。
   *
   * 入力は RGB8 の画素列（行優先・画素あたり 3 バイト）で、resize / rescale / normalize は
   * この中で配布形の定数どおりに掛かる。アスペクト比は保たない（SigLIP2 の前処理が高さ・幅の
   * 対へそのまま伸縮する形で、crop も pad も無いため）。
   *
   * 並行に呼ばれた場合は**待たされて順に**走る（1 つの Session を 2 本の run で共有しない）。
   */
  async embed(image: Rgb8Image): Promise<Float32Array<ArrayBuffer>> {
    // dispose 済みの判定は呼び出し時点で行う（鎖の中で見ると、dispose より前に受けた埋め込み
    // まで巻き添えで落ちる）。
    if (this.#disposal !== undefined) {
      throw new Error("Siglip2Pipeline: dispose 済みでは埋め込めない");
    }
    return await this.#chain(() => embedImage(this.#state, image));
  }

  /**
   * 解放する。Session は必ず畳み、**内部で取得した GPU だけ**破棄する（`options.gpu` で
   * 渡された GpuContext は呼び出し側の所有物なので触らない）。
   *
   * MUST: in-flight の埋め込みの完了を待ってから破棄する（flush-before-destroy）— 破棄も鎖に
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
