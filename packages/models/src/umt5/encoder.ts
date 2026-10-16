/**
 * `Umt5Encoder` — テキスト → umT5-XXL の encoder の最終隠れ状態（`[L, 4096]`）の 1 本の面。
 *
 * パイプライン（全段 Karume・torch 不使用）:
 *
 * 1. ホスト: 配布形のトークナイザ資産（`tokenizer` — 形式 `karume-umt5-tokenizer/1`）で文字列を id 列 `[L]` に
 *    する（`./tokenizer.ts` の `Umt5Tokenizer` — 前処理も Unicode 正規化もしない・末尾に `</s>`・受理集合の外は
 *    `ModelInputError`）
 * 2. ホスト: 有効長 L だけで決まる相対位置のバケット表 `[L, L]` を作る（`./relative-position.ts`）
 * 3. `text_encoder` を 1 回回す → `[1, L, 4096]` の f32 を `[L, 4096]`（行優先）として返す
 *
 * パディングもマスクも持たない（グラフは有効長 L の記号次元で回る — ADR 0119 決定 4）。Wan のように固定長の
 * 文脈へ詰めるのは呼び手の責務。
 *
 * ## MUST: Session は構築時に 1 本だけ張り、`dispose` まで持つ
 *
 * 重みは i8 で約 5.3 GiB で、「1 回ごとに張っては畳む」形にすると**符号化 1 回ごとに重みを丸ごと
 * アップロードし直す**。埋め込みを何本も作る使い方が普通なので、siglip2 / depth-anything と同じく常駐が既定の
 * 形になる。Wan のように別の巨大グラフと交互に使うなら、Wan の text 段（`src/wan/text-stage.ts`）が段ごとに張って
 * 畳む — この class はその用途の面ではない。
 *
 * MUST: それでも直列化鎖には載せる — `encode` の同時実行は 1 つの Session を 2 本の `run` で共有することになり、
 * `dispose` は in-flight の完了を待たずに GPU を破棄する（flush-before-destroy 違反）。
 *
 * ## 中断（`signal`）
 *
 * 構築は {@link Umt5EncoderFromPretrainedOptions.signal}（取得層 — siglip2 / depth-anything と同じ置き場）、符号化は
 * {@link Umt5EncodeOptions.signal}（直列化鎖の順番を待った後・run の前で、イベントループへ 1 度譲ってから検査する —
 * 1 回の run は不可分）。中断の例外は `signal.reason` を**そのまま**投げる（包まない — 消費側が
 * `error === controller.signal.reason` で識別できる）。
 *
 * ## 配布形（`karume-umt5-xxl` — pipeline `umt5-encoder/1`）
 *
 * {@link Umt5Encoder.fromPretrained} の第 1 引数は必須で、このパッケージ版が検証した取得元は
 * {@link UMT5_SOURCES}（`./config.ts`）の `"umt5-xxl"`。手元の配布形は `@karume/hub/deno` の `denoDirectory` で渡す。
 *
 * ## MUST: 数値の正はここでは担保されない
 *
 * 正は実 GPU の e2e（`packages/models/tests/e2e_umt5_encoder_test.ts` — Wan の umT5 の移植の門
 * 〈`e2e_wan_umt5_test.ts`〉と同じ golden・同じ帯）と、トークナイザのパリティ（`umt5_tokenizer_test.ts`）・
 * バケット表のバイト一致（`umt5_relative_position_test.ts`）が担保する。
 */

import {
  acquireGpu,
  type GpuContext,
  type Session,
  type SessionDiagnostics,
  type SessionOptions,
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

import { settleAbort } from "../concurrency/abort.ts";
import { createOperationChain } from "../concurrency/serial.ts";
import { readAssetBuffer, readAssetJson } from "../hub/asset-readers.ts";
import {
  assetComponentOpener,
  type ComponentOpener,
  loadContainerComponents,
} from "../hub/components.ts";
import {
  type FromPretrainedComponentOptions,
  type FromPretrainedHubOptions,
  hubLoadOptions,
} from "../hub/load-options.ts";
import { toManifestSource } from "../hub/repo-ref.ts";
import { disposeSteps } from "../session/dispose-steps.ts";
import {
  assertGpuFeaturesGranted,
  assertRequiredLimitsBeforeDownload,
  assertRequiredLimitsSatisfied,
  sessionGpuFeatures,
  toAcquireGpuOptions,
} from "../session/gpu-features.ts";
import { type FamilySessionPolicy, resolveSessionOptions } from "../session/options.ts";
import { assertUmt5PipelineConfig, UMT5_PIPELINE_MAJOR, UMT5_PIPELINE_NAME } from "./config.ts";
import { type Umt5GraphContract, umt5GraphContract } from "./graph-contract.ts";
import { buildUmt5RelativePositionBuckets, UMT5_MAX_LENGTH } from "./relative-position.ts";
import { umt5HiddenStates, umt5SessionInputs } from "./session-io.ts";
import { parseUmt5TokenizerAsset, Umt5Tokenizer } from "./tokenizer.ts";

/** 文言の接頭辞（class 名）。 */
const OWNER = "Umt5Encoder";

/** manifest の weights 表に現れる部品名（差し替え席が受ける役割名でもある — ADR 0041 §3 の規約名）。 */
const TEXT_ENCODER = "text_encoder";

/** この系列の weights 部品。 */
const COMPONENT_KEYS = [TEXT_ENCODER] as const;

/**
 * トークナイザ資産のキー（manifest のモデル単位の `assets` — recipe `wan/umt5_distribution.py` の
 * `UMT5_TOKENIZER_ROLE`・形式 `karume-umt5-tokenizer/1`）。
 */
const TOKENIZER = "tokenizer";

/** 構築オプション（{@link Umt5Encoder.fromAssets} / {@link Umt5Encoder.fromPretrained} 共通）。 */
export type Umt5EncoderOptions = {
  /**
   * 既存の GPU を共有する。**渡した側が所有権を持つ**ので {@link Umt5Encoder.dispose} は破棄しない。省略時は
   * encoder が内部で `acquireGpu` し、`dispose()` で破棄する。
   */
  readonly gpu?: GpuContext;
  /** モデル（manifest の models のキー）。省略時は `defaultModel`。 */
  readonly model?: string;
  /** 実行構成（そのモデルの quants のキー）。省略時は `defaultQuant`。 */
  readonly quant?: string;
  /**
   * 実行 1 回ごとの診断を受け取る観測席（1 符号化 = `text_encoder` 1 回）。op 別 GPU 時間（`lastRunTiming`）が
   * 要るときは `gpu` に `acquireGpu({ gpuTiming: true })` を渡す（ADR 0021 — 既定は計測しない）。
   *
   * NOTE: コンポーネント名は渡さない — グラフが 1 本しかないので常に同じ 1 値になる（siglip2 と同じ）。
   *
   * コールバックの例外は握らない（fail loudly — 符号化ごと落ちる）。
   */
  readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics) => void;
};

/**
 * {@link Umt5Encoder.fromPretrained} だけが使う取得層のオプション（hub へ透過する）。部品差し替え席
 * （`components`）の役割は `text_encoder` 1 本。
 *
 * NOTE: `headers` / `fetch` / `caches` / `onRetry` が **HTTP 取得元専用**であることを含め、取得層のノブの説明は
 * {@link FromPretrainedHubOptions} に 1 本化してある。
 */
export type Umt5EncoderFromPretrainedOptions =
  & Umt5EncoderOptions
  & FromPretrainedHubOptions
  & FromPretrainedComponentOptions
  & {
    /** 取得（manifest と重み約 5.3 GiB・トークナイザ資産）の中断。構築の後には持ち越さない。 */
    readonly signal?: AbortSignal;
  };

/**
 * 取得済みの manifest + 資産から直接組むときの入力（hub の `fetchAssets` の返り値をそのまま渡せる形）。
 * `assets` は部品 `text_encoder`（単一形 `krm` のキー `text_encoder` か part 列 `text_encoder[0]` /
 * `text_encoder[1]` / …）と、manifest の `assets` のトークナイザ資産 `tokenizer`（JSON のバイト列）。
 */
export type Umt5EncoderAssets = {
  readonly manifest: Manifest;
  readonly assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>;
};

/** {@link Umt5Encoder.encode} の 1 回ぶんのオプション。 */
export type Umt5EncodeOptions = {
  /**
   * 符号化の中断。直列化鎖の順番を待った後・run の前で、イベントループへ 1 度譲ってから検査する（1 回の run は
   * 不可分）。中断したら `signal.reason` を**そのまま**投げる（包まない）。中断の後も同じ encoder で次の
   * `encode` を回せる。
   */
  readonly signal?: AbortSignal;
};

/** 符号化 1 回の結果。 */
export type Umt5Encoding = {
  /** トークナイザの id 列（i32・長さ `length`・末尾は `</s>` = 1）。 */
  readonly inputIds: Int32Array<ArrayBuffer>;
  /** encoder の最終隠れ状態 `[length, hiddenSize]`（f32・行優先 — 1 トークン 1 行）。 */
  readonly hiddenStates: Float32Array<ArrayBuffer>;
  /** トークン数 L（`</s>` を含む — 2〜512）。 */
  readonly length: number;
  /** 1 行の幅（umT5-XXL は 4,096 — グラフ宣言から読む）。 */
  readonly hiddenSize: number;
};

/**
 * umT5 の encoder が受ける実行ノブ（manifest の quant 宣言）。どのキーも受けない: 配布形の席は i8 の重みを f32 で
 * 回す 1 つだけで、実行の形は格納が決める（ADR 0119 決定 5 — `./graph-contract.ts` の格納の門）。宣言が現れたら、
 * 受ける前に参照値で確かめる — 黙って runtime へ流さない。
 *
 * NOTE: `export` は全家族の受理表の網羅と、配布形ミラーの同値を縛るテストのため（`mod.ts` / サブパス面には
 * 出さない — ADR 0008）。
 */
export const UMT5_ENCODER_SESSION_POLICY: FamilySessionPolicy = {
  linearCompute: false,
  attentionCompute: false,
  attentionScoreStorage: false,
  linearGemvReduce: false,
  stateAttentionReduce: false,
  fuseRmsNormAdd: false,
  fuseLinearStaticQuantize: false,
  packedStaticQuantize: false,
};

/** 家族 admission（{@link admitUmt5}）が確定させる材料。 */
type Umt5Admission = {
  readonly quantName: string;
  readonly quant: Quant;
  /** quant 宣言を受理表で通した実効設定（{@link UMT5_ENCODER_SESSION_POLICY}）。 */
  readonly sessionOptions: SessionOptions;
  /** 実効設定が要る feature を quant 宣言へ足したもの（要求と検査の両方がこれを見る）。 */
  readonly gpuFeatures: GpuFeaturesSpec | undefined;
  readonly contract: Umt5GraphContract;
};

/** {@link Umt5Encoder} の内部状態（公開面には出さない）。 */
type Umt5State = {
  readonly gpu: GpuContext;
  readonly ownsGpu: boolean;
  readonly tokenizer: Umt5Tokenizer;
  readonly contract: Umt5GraphContract;
  /** 構築時に張って `dispose` まで持つ 1 本（モジュール doc の MUST）。 */
  readonly session: Session;
  readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics) => void;
};

/**
 * 全量面（`fromAssets`）のコンポーネント供給口（受け口の実装は全家族共有 — {@link assetComponentOpener}）。
 *
 * MUST: `slice` で写さない — 重みは約 5.3 GiB で、写すとホスト RAM のピークが倍になる（{@link readAssetBuffer}）。
 */
const assetOpener = (assets: Umt5EncoderAssets["assets"]): Promise<ComponentOpener> =>
  assetComponentOpener(
    OWNER,
    assets,
    (key) => readAssetBuffer(OWNER, "weights / assets", assets, key),
    COMPONENT_KEYS,
  );

/**
 * この manifest とこのグラフを umT5 の encoder として実行できるかを見る（`hub/components.ts` の
 * {@link FamilyAdmission} 席 — 取得面では**重みの part を 1 バイトも取る前**に呼ばれる）。
 *
 * MUST: 家族の門はこの 1 本に集める（pipeline 名 / major・`pipelineConfig`・quant・トークナイザ資産の宣言・
 * quant の `session`・共有 GPU の能力・グラフ宣言 × ホストの取り決め）。後段へ散らすと、取得面では 5.3 GiB の
 * 重みを落とした**後**にしか落ちない（ADR 0070 決定 5）。
 * MUST: manifest の契約違反とグラフとの突合は **GPU を取りに行く前**に落とす（他の家族と同じ順序）。
 */
const admitUmt5 = (
  manifest: Manifest,
  open: ComponentOpener,
  options: Umt5EncoderOptions,
): Umt5Admission => {
  const modelName = options.model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, modelName)) {
    throw new Error(
      `${OWNER}: model '${modelName}' は manifest に無い` +
        `（利用可能: ${manifest.available.models.join(" / ")}）`,
    );
  }
  const entry: ModelEntry = manifest.models[modelName];
  const { name, major } = entry.pipeline;
  if (name !== UMT5_PIPELINE_NAME) {
    throw new Error(
      `${OWNER}: manifest の pipeline が '${name}/${major}'` +
        `（'${UMT5_PIPELINE_NAME}/${UMT5_PIPELINE_MAJOR}' が必要）`,
    );
  }
  if (major !== UMT5_PIPELINE_MAJOR) {
    // 「古い実装 × 新しいリポ」の沈黙劣化を止める唯一の門（ADR 0038 §6）。
    throw new Error(
      `${OWNER}: pipeline '${name}/${major}' の major に未対応` +
        `（この実装が読めるのは ${UMT5_PIPELINE_NAME}/${UMT5_PIPELINE_MAJOR}）`,
    );
  }
  assertUmt5PipelineConfig(entry.pipelineConfig);

  const quantName = options.quant ?? entry.defaultQuant;
  if (!Object.hasOwn(entry.quants, quantName)) {
    throw new Error(
      `${OWNER}: quant '${quantName}' は manifest に無い` +
        `（利用可能: ${entry.available.quants.join(" / ")}）`,
    );
  }
  const quant = entry.quants[quantName];
  // トークナイザ資産の中身は届いてから見る（{@link buildUmt5}）が、宣言が無ければ重み（約 5.3 GiB）を取る前に落とす。
  if (!Object.hasOwn(entry.assets, TOKENIZER)) {
    throw new Error(
      `${OWNER}: model '${modelName}' の manifest の assets にトークナイザ資産 '${TOKENIZER}' が無い` +
        "（単体の encoder は配布形のトークナイザで符号化する。トークナイザを持たない配布形〈取り込み由来の" +
        "手元の実験用ミラーなど〉は Wan の部品差し替え〈components の text_encoder〉でだけ使える）",
    );
  }
  // 未対応の宣言は部品を開く前（= 重みの part を取る前）に落とす（全家族共通の 1 本）。
  const sessionOptions = resolveSessionOptions(
    UMT5_ENCODER_SESSION_POLICY,
    quant.session,
    {},
    `${OWNER}: quant '${quantName}'`,
  );
  const gpuFeatures = sessionGpuFeatures(quant.gpuFeatures, sessionOptions);

  const contract = umt5GraphContract(open(TEXT_ENCODER), OWNER);

  // MUST: 共有 GPU の能力不足（feature / device limit）はこの席で落とす — 重みを落とす前に判る唯一の家族門
  // （要求と検査の写像は `session/gpu-features.ts` の 1 本で、後段の検査も同じ関数を呼ぶ）。
  if (options.gpu !== undefined) {
    assertGpuFeaturesGranted(gpuFeatures, options.gpu, `${OWNER}: quant '${quantName}'`);
    assertRequiredLimitsSatisfied(
      quant.requiredLimits,
      options.gpu.limits,
      `${OWNER}: quant '${quantName}'`,
    );
  }
  return { quantName, quant, sessionOptions, gpuFeatures, contract };
};

/**
 * admission を通った材料 + 資産から実行状態を組む（トークナイザ資産の解析 → GPU の取得 → Session の構築）。
 *
 * MUST: 資産の解析と突合は **GPU を取りに行く前**（壊れた資産の真因を GPU 無し環境の別の例外で消さない）。
 * MUST: GPU 取得後に許される検査は GPU の能力（feature — ADR 0028 / device limits — ADR 0089 決定 5）だけ。
 */
const buildUmt5 = async (
  admitted: Umt5Admission,
  assets: Umt5EncoderAssets["assets"],
  open: ComponentOpener,
  options: Umt5EncoderOptions,
): Promise<Umt5State> => {
  const { quantName, quant, sessionOptions, gpuFeatures, contract } = admitted;
  const tokenizer = new Umt5Tokenizer(
    parseUmt5TokenizerAsset(readAssetJson(OWNER, "weights / assets", assets, TOKENIZER), TOKENIZER),
  );
  // トークナイザの上限はグラフの記号次元の上限・バケット表の生成器の上限と同じ値（ADR 0119 決定 4）。違うと、
  // 上限の間の入力がトークナイザを通ってからバケット表の生成で落ちる（大きい側）か、上流が受ける長さを黙って
  // 拒む（小さい側）。
  if (tokenizer.maxLength !== UMT5_MAX_LENGTH) {
    throw new Error(
      `${OWNER}: トークナイザ資産の maxLength ${tokenizer.maxLength} が umT5 の有効長の上限 ` +
        `${UMT5_MAX_LENGTH} と違う`,
    );
  }
  // 供給口は admission が見たものと**同じ 1 本**（`open` は開いた部品を引き当てるだけ）。
  const textEncoder = open(TEXT_ENCODER);
  // MUST: 宣言された feature は device 作成時にしか要求できない（ADR 0028）。共有 GPU は {@link admitUmt5} が
  // 同じ 1 本で見ているが、自前で取った device はここが唯一の門。
  const gpu = options.gpu ?? await acquireGpu(toAcquireGpuOptions(gpuFeatures));
  const ownsGpu = options.gpu === undefined;
  try {
    assertGpuFeaturesGranted(gpuFeatures, gpu, `${OWNER}: quant '${quantName}'`);
    // 自前で取った device の limits もここで見る — `fromAssets` には取得前の事前判定の席が無い
    // （`fromPretrained` はアダプタ値で見ている）。共有 GPU は {@link admitUmt5} が見ている。
    if (ownsGpu) {
      assertRequiredLimitsSatisfied(
        quant.requiredLimits,
        gpu.limits,
        `${OWNER}: quant '${quantName}'`,
      );
    }
    return {
      gpu,
      ownsGpu,
      tokenizer,
      contract,
      session: await textEncoder.createSession(gpu, sessionOptions),
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

/** id 列 1 本を回す（バケット表 → `text_encoder` 1 回 → 出力の検査）。 */
const encodeIds = async (
  state: Umt5State,
  inputIds: Int32Array<ArrayBuffer>,
  signal: AbortSignal | undefined,
): Promise<Umt5Encoding> => {
  // 直列化鎖の順番待ちの間に届いた中断もここで効く（run は不可分 — 中断は run の前でだけ効く）。
  await settleAbort(signal);
  const { contract, session } = state;
  const outputs = await session.run(
    umt5SessionInputs(inputIds, buildUmt5RelativePositionBuckets(inputIds.length)),
  );
  state.onRunDiagnostics?.(session.diagnostics());
  if (!Object.hasOwn(outputs, contract.output)) {
    throw new Error(`${OWNER}: グラフ出力 '${contract.output}' が実行結果に無い`);
  }
  const hiddenStates = umt5HiddenStates(
    outputs[contract.output],
    inputIds.length,
    contract.hiddenSize,
  );
  // MUST: 非有限の出力は返さない（O(L·hiddenSize)）。下流の別のモデルへ渡ってから落ちると、文言が真因の段を
  // 指さない（Wan の text 段の同じ門 — `src/wan/text-stage.ts` の `encodeWanPrompts`）。
  const broken = hiddenStates.findIndex((value) => !Number.isFinite(value));
  if (broken !== -1) {
    throw new Error(
      `${OWNER}: umT5 の出力（${inputIds.length} トークン）の行 ` +
        `${Math.floor(broken / contract.hiddenSize)}・列 ${broken % contract.hiddenSize} が非有限` +
        `（${hiddenStates[broken]}）`,
    );
  }
  return { inputIds, hiddenStates, length: inputIds.length, hiddenSize: contract.hiddenSize };
};

/**
 * umT5-XXL のテキスト → encoder の最終隠れ状態（グラフ 1 本・Session 常駐）。
 *
 * 構築は {@link Umt5Encoder.fromPretrained}（配布形から取得）か {@link Umt5Encoder.fromAssets}（取得済み
 * バイト列）だけを入口にする — コンストラクタを private にしてあるのは、manifest 検査と資産の突合を迂回した
 * 半端な状態を作れないようにするため（ADR 0008）。
 */
export class Umt5Encoder {
  readonly #state: Umt5State;
  /** encode と dispose の直列化鎖（「1 本ずつ」を公開 API 側で守る — モジュール doc）。 */
  readonly #chain = createOperationChain();
  /**
   * dispose の 1 本。**undefined でないことが「dispose 済み」**（別に真偽値を持つと、独立に更新される派生状態に
   * なる）。
   */
  #disposal: Promise<void> | undefined;

  private constructor(state: Umt5State) {
    this.#state = state;
  }

  /**
   * 配布形から取得して組む（`loadManifest` → `resolveSelection` → **部品の descriptor（part 0）だけ**を取って
   * admission → 重みの part を温める → トークナイザ資産の `fetchAssets` → 構築）。block は Session を組むその瞬間に
   * part 順で読まれる（ADR 0109 — `src/hub/components.ts`）。部品を別リポの同じ役割で差し替えるときは
   * {@link FromPretrainedComponentOptions.components}（役割は `text_encoder` 1 本 — トークナイザは差し替えない）。
   *
   * **`ref` は必須**（取得元に既定は無い — `src/hub/repo-ref.ts` の MUST）。パッケージ版が検証した取得元は
   * {@link UMT5_SOURCES}（`./config.ts`）の `"umt5-xxl"` — 再現性を自分で固定するなら `{ repo, revision }` を
   * 呼び手が明示する。文字列の `ref` は `{ repo }` と読む（= `main` 追従）。手元の配布形
   * （`models/karume-umt5-xxl`）は**取得元ハンドル**で渡す（`localDirectory` / `@karume/hub/deno` の
   * `denoDirectory`）— HF の `owner/name` の綴りの門は通らず、network も CacheStorage も通らない
   * （{@link Umt5EncoderFromPretrainedOptions} の HTTP 専用ノブは効かない）。
   */
  static async fromPretrained(
    ref: string | HubRepoRef | DistributionSource,
    options: Umt5EncoderFromPretrainedOptions = {},
  ): Promise<Umt5Encoder> {
    const source = toManifestSource(
      ref,
      `${OWNER}.fromPretrained`,
      'UMT5_SOURCES["umt5-xxl"]（@karume/models/umt5）',
    );
    const hubOptions = hubLoadOptions(options);
    const loaded = await loadManifest(source, hubOptions);
    const choice = {
      ...(options.model === undefined ? {} : { model: options.model }),
      ...(options.quant === undefined ? {} : { quant: options.quant }),
    };
    const selection = resolveSelection(loaded.manifest, choice);
    const buildOptions: Umt5EncoderOptions = {
      ...(options.gpu === undefined ? {} : { gpu: options.gpu }),
      ...choice,
      ...(options.onRunDiagnostics === undefined
        ? {}
        : { onRunDiagnostics: options.onRunDiagnostics }),
    };
    // 家族の門は admission 席で通す（重みの part を取る前 — `hub/components.ts`）。
    const { admitted, assets, open } = await loadContainerComponents(
      `${OWNER}.fromPretrained`,
      loaded,
      selection,
      COMPONENT_KEYS,
      async (open) => {
        const admitted = admitUmt5(loaded.manifest, open, buildOptions);
        // 配布形が宣言した `requiredLimits` は**重みの part を取る前**にここで見る（ADR 0089 決定 5 — 共有 GPU なら
        // その limits、自前で取る経路はアダプタ実測値）。
        await assertRequiredLimitsBeforeDownload(
          admitted.quant.requiredLimits,
          buildOptions.gpu,
          `${OWNER}: quant '${admitted.quantName}'`,
        );
        return admitted;
      },
      {
        ...hubOptions,
        ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
        ...(options.components === undefined ? {} : { components: options.components }),
      },
    );
    return new Umt5Encoder(await buildUmt5(admitted, assets, open, buildOptions));
  }

  /**
   * 取得済みの manifest + 資産から組む（{@link Umt5EncoderAssets}）。容器を開いて、取得面と同じ家族 admission と
   * 組み立てを通し、**`text_encoder` の Session を 1 本張って**返す（モジュール doc の MUST）。
   */
  static async fromAssets(
    input: Umt5EncoderAssets,
    options: Umt5EncoderOptions = {},
  ): Promise<Umt5Encoder> {
    const open = await assetOpener(input.assets);
    const admitted = admitUmt5(input.manifest, open, options);
    return new Umt5Encoder(await buildUmt5(admitted, input.assets, open, options));
  }

  /**
   * トークナイザだけを通す（GPU に触らない — dispose の後も呼べる）。返るのは {@link encode} の `inputIds` と
   * 同じ id 列。受理集合の外は `ModelInputError`（拒否の規則は `src/umt5/tokenizer.ts` のモジュール doc —
   * 語彙外の文字・本文中の特殊トークン・空白の直後の `▁`・U+0020 以外の空白と末尾の空白・512 トークン超・
   * 空文字）。
   */
  tokenize(text: string): Int32Array<ArrayBuffer> {
    return this.#state.tokenizer.tokenize(text);
  }

  /**
   * 文字列 1 本を符号化する（トークナイザ → バケット表 → `text_encoder` 1 回）。返るのは id 列と最終隠れ状態
   * `[length, hiddenSize]`（{@link Umt5Encoding}）。受理集合の外の入力は GPU に触る前に `ModelInputError`
   * （{@link tokenize} と同じ規則）。
   *
   * 並行に呼ばれた場合は**待たされて順に**走る（1 つの Session を 2 本の run で共有しない）。
   */
  async encode(text: string, options: Umt5EncodeOptions = {}): Promise<Umt5Encoding> {
    // dispose 済みの判定は呼び出し時点で行う（鎖の中で見ると、dispose より前に受けた符号化まで巻き添えで落ちる）。
    if (this.#disposal !== undefined) {
      throw new Error(`${OWNER}: dispose 済みでは符号化できない`);
    }
    // 入力起因の拒否は順番を待たずに返す（トークナイザは状態を持たない純粋な処理）。
    const inputIds = this.#state.tokenizer.tokenize(text);
    return await this.#chain(() => encodeIds(this.#state, inputIds, options.signal));
  }

  /**
   * 解放する。Session は必ず畳み、**内部で取得した GPU だけ**破棄する（`options.gpu` で渡された GpuContext は
   * 呼び出し側の所有物なので触らない）。
   *
   * MUST: in-flight の符号化の完了を待ってから破棄する（flush-before-destroy）— 破棄も鎖に載せることで、待ちと
   * 破棄の順序を 1 箇所で決める。2 度目以降も同じ完了を返す。
   *
   * MUST: `Session.dispose()` が失敗しても `gpu.destroy()` は必ず通す（{@link disposeSteps}）。
   */
  dispose(): Promise<void> {
    this.#disposal ??= this.#chain(() =>
      disposeSteps([
        () => this.#state.session.dispose(),
        () => {
          if (this.#state.ownsGpu) this.#state.gpu.destroy();
        },
      ])
    );
    return this.#disposal;
  }

  /** `await using` 対応（Explicit Resource Management）— {@link dispose} の別名。 */
  [Symbol.asyncDispose](): Promise<void> {
    return this.dispose();
  }
}
