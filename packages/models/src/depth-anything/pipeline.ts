/**
 * `DepthAnythingPipeline` — 画像 → 相対深度マップ（単一画像の深度推定）の 1 本の面。
 *
 * パイプライン（全段 Karume・torch 不使用）:
 *
 * 1. ホスト: RGB8 の画素列を `pipelineConfig` の寸法へ resize（antialias 付き **bicubic**）
 * 2. ホスト: rescale + normalize を畳んだ形で通し、`[1, 3, S, S]` の f32 にする
 * 3. `depth` を 1 回回す → **相対深度** `[1, S, S]`（head 末尾の ReLU 済み・非負）
 * 4. ホスト: 元画像の解像度へ戻す
 *
 * 前処理 2 段の実装と参照実装は `src/image/preprocess.ts`（`resizeRgb8` / `normalizeToNchw`）で、
 * ここはそれを `pipelineConfig` の定数で結線するだけ。**decode は karume の責務ではない** —
 * 入口は「RGB8 の画素列 + 幅 + 高さ」で、PNG / JPEG のデコーダは持たない（ブラウザなら
 * `createImageBitmap` + canvas、Deno なら任意のデコーダ）。
 *
 * MUST: 補間は **bicubic**（上流 `preprocessor_config.json` の `"resample": 3`）。SigLIP2 /
 * BiRefNet の bilinear とは別のフィルタで、宣言の受理集合が `src/depth-anything/config.ts` で
 * 1 値に絞ってある（取り違えは `pixel_values` の最大 0.59 ずれ = uint8 1 LSB の 34 倍）。
 *
 * ## MUST: Session は構築時に 1 本だけ張り、`dispose` まで持つ（Anima / Irodori と非対称）
 *
 * あちらが段ごとに張っては畳むのは、複数の巨大グラフを同時常駐させられないから（ADR 0016）。
 * こちらはグラフが 1 本（Small で 99MB）で畳む相手がおらず、逆に「1 枚ごとに張っては畳む」
 * 形にすると**画像 1 枚あたり重みを丸ごとアップロードし直す**。深度推定は何枚も通す使い方が
 * 普通なので、常駐が既定の形になる（SigLIP2 / BiRefNet と同じ理由・同じ形）。
 *
 * MUST: それでも直列化鎖には載せる — `estimate` の同時実行は 1 つの Session を 2 本の `run` で
 * 共有することになり、`dispose` は in-flight の完了を待たずに GPU を破棄する
 * （flush-before-destroy 違反）。
 *
 * ## 返すのは **生の相対深度 f32**（`[0, 1]` へ畳まない・PNG も持たない）
 *
 * {@link DepthAnythingPipeline.estimate} が返すのは「入力と同じ寸法・画素あたり 1 つの f32」で、
 * 正規化も着色もしない。
 *
 * - **正規化は可視化の決定であって推論の一部ではない**。min-max で畳むと元の尺度が復元でき
 *   なくなり、2 枚の深度を同じ尺度で並べる / 閾値を跨いで使う口が消える（逆向きは 3 行で
 *   できる — 下の「使い方」）。BiRefNet が α マットを返して RGBA 合成を持たないのと同じ線。
 * - `min` / `max` を添えて返すこともしない — 返した配列から**導出できる**値で、独立に更新
 *   される席を作らない（CLAUDE.md の「導出可能な状態を二重に持たない」）。
 * - **相対深度には単位も原点も無い**（大きいほど手前）。`[0, 1]` に見える口を生やすと、
 *   絶対距離を約束していると読める — この重みは metric depth ではない
 *   （`depth_estimation_type` は `relative`・`karume.patch_depth_anything` が構成を検査する）。
 *
 * 使い方（可視化用に `[0, 1]` へ畳む場合）:
 *
 * ```ts
 * const depth = await pipeline.estimate(image);
 * let min = Infinity;
 * let max = -Infinity;
 * for (const value of depth.data) {
 *   if (value < min) min = value;
 *   if (value > max) max = value;
 * }
 * const span = max - min;
 * const normalized = depth.data.map((value) => (span > 0 ? (value - min) / span : 0));
 * ```
 *
 * ## NOTE: 元解像度へ戻す段は bilinear（上流の `post_process_depth_estimation` と別）
 *
 * 段 4 は `resizePlaneF32`（`src/image/preprocess.ts` — Pillow / torchvision と同じ台の
 * **antialias 付き bilinear**）を使う。上流 transformers の
 * `DPTImageProcessor.post_process_depth_estimation` は
 * `F.interpolate(mode="bicubic", align_corners=False)`（a = −0.75・antialias 無し）で、
 * **カーネルも縮小時の台の伸びも別**（焼かれた 518² より小さい画像へ戻すとき、こちらは
 * 面積平均が掛かる）。合わせていないのは、
 *
 * - モデルの答えは焼かれた解像度（518²）の `predicted_depth` そのもので、そこから先の拡大は
 *   情報を増やさない**表示解像度への引き伸ばし**だから。数値の正はその 518² の地図に対して
 *   `packages/runtime/tests/e2e_depth_anything_test.ts` が掛かっている。
 * - `post_process_depth_estimation` は transformers 側の任意の便宜関数で、チェックポイントの
 *   定義（`preprocessor_config.json`）には現れない。前処理の `resample` と違い、**配布形が
 *   宣言として持つ事実ではない**。
 *
 * 逆に言えば「上流の後処理とビット単位で揃える」主張は**していない**。揃える必要が出たら、
 * `resizePlaneF32` に `F.interpolate` 互換の三次枝（a = −0.75・antialias 無し）を足す設計に
 * なる（現状は bilinear 固定）。
 *
 * ## MUST: 数値の正はここでは担保されない
 *
 * 正はグラフ単位の golden E2E（`packages/runtime/tests/e2e_depth_anything_test.ts` — 実重み ×
 * 合成 4 + 実画像 4 ケース）と、前処理のパリティ門
 * （`packages/models/tests/image_preprocess_test.ts` / 実画像は
 * `packages/models/tests/e2e_depth_anything_real_test.ts`）が担保する。
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
  type DistributionSource,
  type HubRepoRef,
  loadManifest,
  type Manifest,
  type ModelEntry,
  type Quant,
  resolveFiles,
} from "@karume/hub";

import {
  DEPTH_ANYTHING_PIPELINE_MAJOR,
  DEPTH_ANYTHING_PIPELINE_NAME,
  type DepthAnythingPipelineConfig,
  parseDepthAnythingPipelineConfig,
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
import { toManifestSource } from "../hub/repo-ref.ts";
import {
  assetComponentOpener,
  type ComponentOpener,
  type GraphOwner,
  loadShardComponents,
  type ModelComponent,
} from "../hub/components.ts";

/** manifest の weights 表に現れる取得キー（ADR 0041 §3 の規約名）。 */
const DEPTH = "depth";

/** グラフ入力の名前（`export_depth_anything.py` の `INPUT_NAME`）。 */
const PIXEL_VALUES = "pixel_values";

/** 入力のチャネル数（RGB）。batch と併せてグラフ側も静的 1 / 3 で焼かれている。 */
const CHANNELS = 3;

/**
 * 前処理の補間フィルタ。`config.interpolation` は受理集合が `"bicubic"` の 1 値なので、
 * ここで分岐しない（宣言は「この配布形は bicubic を要求する」という主張で、選択肢ではない —
 * `config.ts` のモジュール doc）。
 */
const FILTER = "bicubic";

/** 入力と同じ寸法の相対深度マップ（画素あたり 1 つの f32・行優先・非負・大きいほど手前）。 */
export type DepthMap = {
  readonly data: Float32Array<ArrayBuffer>;
  readonly width: number;
  readonly height: number;
};

/**
 * 構築オプション（{@link DepthAnythingPipeline.fromAssets} /
 * {@link DepthAnythingPipeline.fromPretrained} 共通）。
 */
export type DepthAnythingPipelineOptions = {
  /**
   * 既存の GPU を共有する。**渡した側が所有権を持つ**ので
   * {@link DepthAnythingPipeline.dispose} は破棄しない。省略時はパイプラインが内部で
   * `acquireGpu` し、`dispose()` で破棄する。
   */
  readonly gpu?: GpuContext;
  /** モデル（manifest の models のキー）。省略時は `defaultModel`。 */
  readonly model?: string;
  /** 実行構成（そのモデルの quants のキー）。省略時は `defaultQuant`。 */
  readonly quant?: string;
  /**
   * 実行 1 回ごとの診断を受け取る観測席（1 深度地図 = `depth` 1 回）。op 別 GPU 時間
   * （`lastRunTiming`）が要るときは `gpu` に `acquireGpu({ gpuTiming: true })` を渡す
   * （ADR 0021 — 既定は計測しない）。
   *
   * NOTE: SigLIP2 / BiRefNet と同じくコンポーネント名を渡さない — グラフが 1 本しかないので、
   * 名前が常に同じ 1 値になる（受け手が分岐できない引数を渡さない）。
   *
   * コールバックの例外は握らない（fail loudly — 推定ごと落ちる）。
   */
  readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics) => void;
};

/**
 * {@link DepthAnythingPipeline.fromPretrained} だけが使う取得層のオプション（hub へ透過する）。
 *
 * NOTE: `headers` / `fetch` / `caches` は **HTTP 取得元専用**のノブで、取得元ハンドル
 * （`localDirectory` / `denoDirectory`）を渡した呼び出しでは 1 つも効かない — 手元の配布形は
 * network も CacheStorage も通らない。
 */
export type DepthAnythingFromPretrainedOptions = DepthAnythingPipelineOptions & {
  readonly signal?: AbortSignal;
  /**
   * `Authorization` 等。取得（revision 解決・ファイル）へそのまま透過する。
   *
   * NOTE: **キャッシュは credential で分けない**（by-design — キーにヘッダは入らないので、
   * 認証付きで取得したバイト列は以後の無認証呼び出しにもヒットする）。ADR 0080 決定 3 /
   * `docs/limitations.md` の「hub: キャッシュは credential で隔離しない」が正本。
   */
  readonly headers?: HeadersInit;
  readonly onProgress?: (progress: AssetProgress) => void;
  readonly onCacheError?: (diagnostic: CacheDiagnostic) => void;
  /** `fetch` の差し替え（テスト・カスタム輸送用）。 */
  readonly fetch?: typeof globalThis.fetch;
  /** `CacheStorage` の差し替え（テスト用）。 */
  readonly caches?: CacheStorage;
};

/** 取得済み資産から直接組むときの入力（hub の `fetchAssets` の返り値をそのまま渡す）。 */
export type DepthAnythingAssets = {
  readonly manifest: Manifest;
  readonly assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>;
};

/**
 * 取得済みバイト列を `openModel` へ渡せる ArrayBuffer にする。
 *
 * MUST: `slice` で写さない — 配布形は 1 本 99MB あり、ホスト RAM のピークが倍になる。hub は
 * buffer 全体を占める view を返す契約なので、崩れていたら**取得層の不変条件破れ**として落とす。
 */
const assetBuffer = (
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  key: string,
): ArrayBuffer => {
  if (!Object.hasOwn(assets, key)) {
    throw new Error(
      `depth-anything: 資産 '${key}' が無い（manifest の weights に ${key} が要る）` +
        `（揃っているキー: ${Object.keys(assets).join(" / ")}）`,
    );
  }
  const bytes = assets[key];
  if (bytes.byteOffset !== 0 || bytes.byteLength !== bytes.buffer.byteLength) {
    throw new Error(
      `depth-anything: 資産 '${key}' の bytes が buffer 全体を占めていない` +
        `（byteOffset ${bytes.byteOffset} / byteLength ${bytes.byteLength} /` +
        ` buffer ${bytes.buffer.byteLength}）`,
    );
  }
  return bytes.buffer;
};

/**
 * 全量面（`fromAssets`）のコンポーネント供給口（受け口の実装は 7 家族共有 —
 * {@link assetComponentOpener}）。素の 1 本は `openModel` で開いて全量面で組み、shard 分割形
 * （`depth[0]` / `depth[1]` / …）は `fromPretrained` と同じ shard 逐次面へ流す。
 */
const assetOpener = (assets: DepthAnythingAssets["assets"]): ComponentOpener =>
  assetComponentOpener("depth-anything", assets, (key) => assetBuffer(assets, key));

/**
 * グラフ入力の 1 軸ぶんの**静的**次元が `pipelineConfig` の宣言と一致することを見る。
 *
 * MUST: 落とさない。前処理は宣言の寸法へ resize するので、グラフが別の解像度で焼かれていても
 * **ホスト側は最後まで通る**（落ちるのは Session の shape 検査で、そのときには「どちらの数が
 * 正しいのか」が読み手に伝わらない）。DINOv2 の位置埋め込みはパッチ数に紐づいているので、
 * 事前学習解像度でないグラフはそもそも焼けない（`patch_depth_anything` の ②）— つまりここで
 * 落ちるのは常に**資産の取り違え**である。
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
  if (spec === undefined) {
    throw new Error(`depth-anything: グラフ入力 '${inputName}' が無い（${where}）`);
  }
  const dim = spec.shape[axis];
  if (dim !== expected) {
    throw new Error(
      `depth-anything: ${where} — グラフ入力 '${inputName}' の軸 ${axis} が ${String(dim)}、` +
        `pipelineConfig は ${expected}`,
    );
  }
};

/**
 * グラフ**出力**の形が `[1, imageHeight, imageWidth]` であることを見る。
 *
 * MUST: 落とさない。後段は「要素数が H×W」しか見ない形にもできるが、それだとチャネル軸を
 * 持ったまま焼かれたグラフ（`[1, 1, S, S]`）や、中間段まで出す別 export が**別の値を深度
 * として**通り抜ける。階数まで見るのは、`[1, S, S]` と `[1, 1, S, S]` が要素数では区別
 * できないため。
 *
 * NOTE: `export` は GPU 無しで拒否経路を縛るテストのため（`mod.ts` / サブパス面には出さない —
 * ADR 0008）。
 */
export const assertDepthShape = (
  model: GraphOwner,
  config: DepthAnythingPipelineConfig,
  where: string,
): void => {
  const name = model.graph.outputs[0];
  const value = model.graph.values[name];
  if (value === undefined) {
    throw new Error(`depth-anything: グラフ出力 '${name}' の宣言が無い（${where}）`);
  }
  const expected = [1, config.imageHeight, config.imageWidth];
  if (
    value.shape.length !== expected.length ||
    value.shape.some((dim, axis) => dim !== expected[axis])
  ) {
    throw new Error(
      `depth-anything: ${where} — グラフ出力 '${name}' の形が ` +
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
 * アスペクト比は保たない（グラフが正方 1 点でしか受け取らないので、伸縮以外に行き先が無い —
 * `config.ts` の NOTE）。
 *
 * NOTE: `export` は結線を直接叩くテストのため（`fromAssets` 経由で此処へ届くには実 IR
 * コンテナが要る）。`mod.ts` / サブパス面には出さない（ADR 0008）。
 */
export const preprocessPixelValues = (
  config: DepthAnythingPipelineConfig,
  image: Rgb8Image,
): Float32Array<ArrayBuffer> =>
  normalizeToNchw(
    resizeRgb8(image, config.imageWidth, config.imageHeight, FILTER),
    config.imageMean,
    config.imageStd,
  );

/**
 * グラフの出す深度地図 → 元画像の解像度（モジュール doc の段 4）。
 *
 * **値には触らない** — 正規化も clamp も掛けない（返すのは生の相対深度 — モジュール doc の
 * 「返すのは生の相対深度 f32」）。`resizePlaneF32` の重みは非負で総和 1 なので、非負という
 * 出力の性質（head 末尾の ReLU）は拡大しても保たれる。
 *
 * 恒等寸法（`sourceWidth === width && sourceHeight === height`）では値がそのまま通る
 * （`resizePlaneF32` の台が近傍 1 点へ縮む — 前処理層のテストが固定している）ので、
 * 「元解像度 = 焼かれた解像度」の呼び出しに丸め損失は入らない。
 *
 * MUST: 非有限値は resize の**前**（発生源に一番近い位置）で画素座標つきに落とす。
 * `resizePlaneF32` は重み付き和なので NaN を近傍へ広げてから返し、この doc が配っている
 * 可視化レシピの min / max 走査は NaN 比較が常に偽で素通りする（`span` が非正になり
 * **全画素 0 = 一様な真っ黒の深度地図**が正常値の顔で出る）。birefnet の `matteFromLogits`・
 * sbv2 の波形 gate・`anima/image.ts` と同じ検査点。
 *
 * NOTE: `export` はテストのため（{@link preprocessPixelValues} と同じ理由）。
 */
export const resampleDepth = (
  depth: Float32Array,
  sourceWidth: number,
  sourceHeight: number,
  width: number,
  height: number,
): DepthMap => {
  for (let index = 0; index < depth.length; index += 1) {
    if (!Number.isFinite(depth[index])) {
      const x = index % sourceWidth;
      const y = Math.floor(index / sourceWidth);
      throw new Error(`depth-anything: 深度値 (x=${x}, y=${y}) が非有限値`);
    }
  }
  return {
    data: resizePlaneF32(depth, sourceWidth, sourceHeight, width, height),
    width,
    height,
  };
};

/** {@link DepthAnythingPipeline} の内部状態（公開面には出さない）。 */
type DepthAnythingState = {
  readonly gpu: GpuContext;
  readonly ownsGpu: boolean;
  readonly config: DepthAnythingPipelineConfig;
  readonly depth: ModelComponent;
  /** 構築時に張って `dispose` まで持つ 1 本（モジュール doc の MUST）。 */
  readonly session: Session;
  readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics) => void;
};

/** 家族 admission（{@link admitDepthAnything}）が確定させる材料。 */
type DepthAnythingAdmission = {
  readonly config: DepthAnythingPipelineConfig;
  readonly quantName: string;
  readonly quant: Quant;
  readonly depth: ModelComponent;
};

/**
 * この manifest とこのグラフを depth-anything として実行できるかを見る（`hub/components.ts`
 * の家族 admission 席 — shard 面では**重み shard を 1 バイトも取る前**に呼ばれる）。
 *
 * MUST: 家族の門はこの 1 本に集める。後段へ散らすと、shard 面では GB 級の重みを落とした
 * **後**にしか落ちない（ADR 0070 決定 5 の文面より実装が狭くなる）。
 * MUST: manifest の契約違反と**グラフとの突合**は **GPU を取りに行く前**に落とす。順序が
 * ずれると、GPU の無い環境では別の例外に化けて「何が悪かったのか」が読み手に伝わらない。
 */
const admitDepthAnything = (
  manifest: Manifest,
  open: ComponentOpener,
  options: DepthAnythingPipelineOptions,
): DepthAnythingAdmission => {
  const modelName = options.model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, modelName)) {
    throw new Error(
      `DepthAnythingPipeline: model '${modelName}' は manifest に無い` +
        `（利用可能: ${manifest.available.models.join(" / ")}）`,
    );
  }
  const entry: ModelEntry = manifest.models[modelName];
  const { name, major } = entry.pipeline;
  if (name !== DEPTH_ANYTHING_PIPELINE_NAME) {
    throw new Error(
      `DepthAnythingPipeline: manifest の pipeline が '${name}/${major}'` +
        `（'${DEPTH_ANYTHING_PIPELINE_NAME}/${DEPTH_ANYTHING_PIPELINE_MAJOR}' が必要）`,
    );
  }
  if (major !== DEPTH_ANYTHING_PIPELINE_MAJOR) {
    // 「古い実装 × 新しいリポ」の沈黙劣化を止める唯一の門（ADR 0038 §6）。
    throw new Error(
      `DepthAnythingPipeline: pipeline '${name}/${major}' の major に未対応` +
        `（この実装が読めるのは ${DEPTH_ANYTHING_PIPELINE_NAME}/` +
        `${DEPTH_ANYTHING_PIPELINE_MAJOR}）`,
    );
  }
  const config = parseDepthAnythingPipelineConfig(entry.pipelineConfig);

  const quantName = options.quant ?? entry.defaultQuant;
  if (!Object.hasOwn(entry.quants, quantName)) {
    throw new Error(
      `DepthAnythingPipeline: quant '${quantName}' は manifest に無い` +
        `（利用可能: ${entry.available.quants.join(" / ")}）`,
    );
  }
  const quant = entry.quants[quantName];

  const depth = open(DEPTH);
  // グラフの宣言と pipelineConfig の突合。入出力が 1 本ずつであることまで見るのは、中間段の
  // 深度まで出す別 export が混ざると、位置で引く後段が黙って別の値を深度として読むため。
  if (depth.graph.inputs.length !== 1 || depth.graph.outputs.length !== 1) {
    throw new Error(
      `DepthAnythingPipeline: depth グラフの入出力が ${depth.graph.inputs.length} /` +
        ` ${depth.graph.outputs.length} 本（どちらも 1 本の深度地図が必要）`,
    );
  }
  assertStaticDim(depth, PIXEL_VALUES, 0, 1, "batch は静的 1");
  assertStaticDim(depth, PIXEL_VALUES, 1, CHANNELS, "RGB の 3 チャネル");
  assertStaticDim(depth, PIXEL_VALUES, 2, config.imageHeight, "imageHeight");
  assertStaticDim(depth, PIXEL_VALUES, 3, config.imageWidth, "imageWidth");
  assertDepthShape(depth, config, "深度地図の形");

  // MUST: 共有 GPU の能力不足はこの席で落とす — 自前で取る場合と違って `acquireGpu` を
  // 待つ理由が無く、重みを落とす前に判る唯一の家族門（要求と検査の網羅表は
  // `session/gpu-features.ts` の 1 本で、後段の検査も同じ関数を呼ぶ）。
  if (options.gpu !== undefined) {
    assertGpuFeaturesGranted(
      quant.gpuFeatures,
      options.gpu,
      `DepthAnythingPipeline: quant '${quantName}'`,
    );
  }

  return { config, quantName, quant, depth };
};

/**
 * admission を通った材料から実行状態を組む（GPU 取得と Session 構築だけ）。
 *
 * MUST: GPU 取得後に許される検査は GPU の能力（shader-f16）だけ（ADR 0028）。
 */
const openDepthAnythingState = async (
  admitted: DepthAnythingAdmission,
  options: DepthAnythingPipelineOptions = {},
): Promise<DepthAnythingState> => {
  const { config, quant, quantName, depth } = admitted;
  // MUST: 宣言された feature は device 作成時にしか要求できない（ADR 0028）。共有 GPU を
  // 渡された場合は要求できないので、能力が足りないことを名指しして落とす（共有 GPU は
  // {@link admitDepthAnything} が既に同じ 1 本で見ているが、自前で取った device は
  // ここが唯一の門）。
  const gpu = options.gpu ?? await acquireGpu(toAcquireGpuOptions(quant.gpuFeatures));
  const ownsGpu = options.gpu === undefined;
  try {
    assertGpuFeaturesGranted(quant.gpuFeatures, gpu, `DepthAnythingPipeline: quant '${quantName}'`);
    return {
      gpu,
      ownsGpu,
      config,
      depth,
      session: await depth.createSession(gpu, toSessionOptions(quant.session)),
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

/** 画像 1 枚を通す（前処理 → `depth` 1 回 → 元解像度へ戻す）。 */
const estimateImage = async (
  state: DepthAnythingState,
  image: Rgb8Image,
): Promise<DepthMap> => {
  const { config } = state;
  const pixelValues: Tensor = {
    dtype: "f32",
    shape: [1, CHANNELS, config.imageHeight, config.imageWidth],
    data: preprocessPixelValues(config, image),
  };
  const outputs = await state.session.run({ [PIXEL_VALUES]: pixelValues });
  if (state.onRunDiagnostics !== undefined) state.onRunDiagnostics(state.session.diagnostics());
  const name = state.depth.graph.outputs[0];
  const tensor = outputs[name];
  if (tensor === undefined) {
    throw new Error(`depth-anything: グラフ出力 '${name}' が実行結果に無い`);
  }
  const depth = asF32(tensor, "depth");
  const expected = config.imageWidth * config.imageHeight;
  if (depth.length !== expected) {
    throw new Error(
      `depth-anything: 深度地図の要素数 ${depth.length} が ` +
        `${config.imageWidth}×${config.imageHeight} と違う`,
    );
  }
  return resampleDepth(
    depth,
    config.imageWidth,
    config.imageHeight,
    image.width,
    image.height,
  );
};

/**
 * Depth Anything V2 の画像 → 相対深度マップのパイプライン（グラフ 1 本）。
 *
 * 構築は {@link DepthAnythingPipeline.fromPretrained}（HF から取得）か
 * {@link DepthAnythingPipeline.fromAssets}（取得済みバイト列）だけを入口にする — コンストラクタ
 * を private にしてあるのは、manifest 検査と資産の突合を迂回した半端な状態を作れないように
 * するため（ADR 0008）。
 */
export class DepthAnythingPipeline {
  readonly #state: DepthAnythingState;
  /** estimate と dispose の直列化鎖（「1 本ずつ」を公開 API 側で守る — モジュール doc）。 */
  readonly #chain = createOperationChain();
  /**
   * dispose の 1 本。**undefined でないことが「dispose 済み」**（別に真偽値を持つと、独立に
   * 更新される派生状態になる）。
   */
  #disposal: Promise<void> | undefined;

  private constructor(state: DepthAnythingState) {
    this.#state = state;
  }

  /**
   * 配布形から取得して組む（`loadManifest` → `resolveFiles` → **各コンポーネントの
   * グラフ shard だけ**を取って `prepareModel` → 残り資産の `fetchAssets` → 構築）。重み shard は
   * Session を組むときに 1 本ずつ流れる（ADR 0070 — `src/hub/components.ts`）。文字列の
   * `ref` は `{ repo }` と読む（= `main` 追従）。**`ref` は必須**（取得元に既定は無い —
   * `src/hub/repo-ref.ts` の MUST。このファミリは公開配布リポを持たないので pin 定数も無い）。
   *
   * 手元の配布形は**取得元ハンドル**で渡す（`localDirectory` / `@karume/hub/deno` の
   * `denoDirectory`）。HF の `owner/name` の綴りの門は通らず、network も CacheStorage も
   * 通らない（{@link DepthAnythingFromPretrainedOptions} の HTTP 専用ノブは効かない）。
   */
  static async fromPretrained(
    ref: string | HubRepoRef | DistributionSource,
    options: DepthAnythingFromPretrainedOptions = {},
  ): Promise<DepthAnythingPipeline> {
    const source = toManifestSource(ref, "DepthAnythingPipeline.fromPretrained");
    const hubOptions = {
      ...(options.signal === undefined ? {} : { signal: options.signal }),
      ...(options.headers === undefined ? {} : { headers: options.headers }),
      ...(options.onCacheError === undefined ? {} : { onCacheError: options.onCacheError }),
      ...(options.fetch === undefined ? {} : { fetch: options.fetch }),
      ...(options.caches === undefined ? {} : { caches: options.caches }),
    };
    const loaded = await loadManifest(source, hubOptions);
    const selection = {
      ...(options.model === undefined ? {} : { model: options.model }),
      ...(options.quant === undefined ? {} : { quant: options.quant }),
    };
    const files = resolveFiles(loaded.manifest, selection);
    const buildOptions: DepthAnythingPipelineOptions = {
      ...(options.gpu === undefined ? {} : { gpu: options.gpu }),
      ...selection,
      ...(options.onRunDiagnostics === undefined
        ? {}
        : { onRunDiagnostics: options.onRunDiagnostics }),
    };
    // 家族の門は admission 席で通す（重み shard を取る前 — `hub/components.ts`）。
    const { admitted } = await loadShardComponents(
      "DepthAnythingPipeline.fromPretrained",
      loaded,
      files,
      [DEPTH],
      (open) => admitDepthAnything(loaded.manifest, open, buildOptions),
      {
        ...hubOptions,
        ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
      },
    );
    return new DepthAnythingPipeline(await openDepthAnythingState(admitted, buildOptions));
  }

  /**
   * 取得済みの manifest + 資産から組む。契約検査・資産の解釈・`openModel`・グラフとの突合を
   * 全てここで済ませ、**`depth` の Session を 1 本張って**返す（モジュール doc の MUST）。
   *
   * 取得キーは `resolveFiles` の規約どおり **2 形とも受ける** — 素の 1 本（`depth`）と、
   * 1GiB 超のコンポーネントの shard 分割形（`depth[0]` / `depth[1]` / …）。分割形は
   * バイト列を連結せず `fromPretrained` と同じ shard 逐次面へ流す。添字の欠番と素キーとの混在は
   * fail loudly（受け口の実装は `src/hub/components.ts` の 1 本）。
   */
  static async fromAssets(
    input: DepthAnythingAssets,
    options: DepthAnythingPipelineOptions = {},
  ): Promise<DepthAnythingPipeline> {
    const admitted = admitDepthAnything(input.manifest, assetOpener(input.assets), options);
    return new DepthAnythingPipeline(await openDepthAnythingState(admitted, options));
  }

  /**
   * 画像 1 枚の相対深度を出す。返るのは**入力と同じ寸法**の生の f32（モジュール doc の
   * 「返すのは生の相対深度 f32」）— 大きいほど手前で、単位も原点も無い。
   *
   * 入力は RGB8 の画素列（行優先・画素あたり 3 バイト）で、resize / rescale / normalize は
   * この中で配布形の定数どおりに掛かる。アスペクト比は保たない（グラフが正方 1 点でしか
   * 受け取らないため — `config.ts` の NOTE）。
   *
   * 並行に呼ばれた場合は**待たされて順に**走る（1 つの Session を 2 本の run で共有しない）。
   */
  async estimate(image: Rgb8Image): Promise<DepthMap> {
    // dispose 済みの判定は呼び出し時点で行う（鎖の中で見ると、dispose より前に受けた
    // 推定まで巻き添えで落ちる）。
    if (this.#disposal !== undefined) {
      throw new Error("DepthAnythingPipeline: dispose 済みでは推定できない");
    }
    return await this.#chain(() => estimateImage(this.#state, image));
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
