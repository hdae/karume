/**
 * Wan の家族 admission・構築・生成の本体（ADR 0121 決定 10 — Wan2.1 / 2.2 の class が共有する 1 本）。
 *
 * 公開 class（`./pipeline.ts` の `WanPipeline`・`./ti2v-pipeline.ts` の `WanTi2vPipeline`）は薄い殻
 * （private のコンストラクタ・直列化鎖・`dispose`）で、
 * manifest の門・部品の突合・GPU の取得・段の順序（(encoder →) text → DiT → VAE）の本体はここにある。段の順序と
 * Session の寿命・中断の MUST は `./pipeline.ts` 冒頭の doc が正本。世代ごとに違う値は全て
 * {@link WanFamilySpec} が運ぶ（文言の接頭辞・pipeline 名と major・世代の記述子）。I2V の encoder の部品の有無も
 * 世代の記述子（DiT の入力の形が `"ti2v"`）から決まる。
 *
 * NOTE: 内部だけ（`mod.ts` / サブパス面には出さない — ADR 0008）。公開型は `./pipeline.ts` と
 * `./ti2v-pipeline.ts` から `import type` だけで取る（値の import を持たない — 両方ともここの値を使うので、値の
 * 循環 import を作らない）。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 */

import {
  acquireGpu,
  type GpuContext,
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
import { type FamilySessionPolicy, resolveSessionOptions } from "../session/options.ts";
import {
  assertGpuFeaturesGranted,
  assertRequiredLimitsBeforeDownload,
  assertRequiredLimitsSatisfied,
  sessionGpuFeatures,
  toAcquireGpuOptions,
} from "../session/gpu-features.ts";
import { readAssetBuffer, readWholeAsset } from "../hub/asset-readers.ts";
import {
  assetComponentOpener,
  type ComponentOpener,
  loadContainerComponents,
  type ModelComponent,
} from "../hub/components.ts";
import { toManifestSource } from "../hub/repo-ref.ts";
import { hubLoadOptions } from "../hub/load-options.ts";
import {
  parseWanPipelineConfig,
  WAN_PIPELINE_MAJOR,
  WAN_PIPELINE_NAME,
  WAN_TI2V_PIPELINE_MAJOR,
  WAN_TI2V_PIPELINE_NAME,
  type WanPipelineConfig,
} from "./config.ts";
import { parseWanRopeBase, type WanRopeBase } from "./dit-rope.ts";
import {
  WAN21_GENERATION,
  WAN22_TI2V_GENERATION,
  type WanGenerationDescriptor,
} from "./descriptor.ts";
import {
  type DitContract,
  ditContract,
  ROPE_BASE,
  runWanDenoise,
  TRANSFORMER,
  type WanContexts,
  wanDitPatch,
} from "./dit-loop.ts";
import {
  type PlanLayout,
  planWanRequest,
  type WanGenerationKnobs,
  type WanGenerationPlan,
} from "./plan.ts";
import type {
  GeneratedVideo,
  WanAssets,
  WanFromPretrainedOptions,
  WanGenerateEvent,
  WanPipelineOptions,
  WanRunComponent,
} from "./pipeline.ts";
import type { WanTi2vGenerateRequest } from "./ti2v-pipeline.ts";
import {
  admitWanText,
  assertTextEncoderDeclared,
  assertTokenizerDeclared,
  encodeWanPrompts,
  gpuPromptGate,
  loadWanTextStage,
  precomputedContexts,
  precomputedPromptGate,
  TEXT_ENCODER,
  textEncoderRouteOf,
  type WanTextAdmission,
  type WanTextEncoderRoute,
  type WanTextStage,
} from "./text-stage.ts";
import type { WanTextEmbedding, WanTextEmbeds } from "./text-embeds.ts";
import type { WanPromptEncoder } from "./text/tokenizer.ts";
import {
  assertWanVaeMatchesGeneration,
  assertWanVaeTilesCover,
  decodeWanVaeStage,
  VAE_DECODER_FIRST,
  VAE_DECODER_NEXT,
} from "./tile-decode.ts";
import { type WanVaeChunkLayout, wanVaeChunkLayout } from "./vae-chunks.ts";
import {
  encodeWanImageStage,
  VAE_ENCODER_ATTN,
  VAE_ENCODER_POST,
  VAE_ENCODER_PRE,
  WAN_VAE_ENCODER_KEYS,
  type WanVaeEncoder,
  type WanVaeEncoderContract,
  wanVaeEncoderContract,
} from "./vae-encoder.ts";

/**
 * 世代と経路で取る部品（取得面の `componentKeys`・全量面の開く部品）。DiT の入力の形が `"ti2v"` の世代（Wan2.2）は
 * I2V の VAE encoder の 3 部品を、画像を渡さない使い方でも取る（構築の口を増やさない — 部品は配布形が常に持つ）。
 *
 * MUST: `"precomputed"` は umT5 の部品を開かない — 開くと umT5 の取得（i8 で約 5.3 GiB）が軽い使い方にも乗る
 * （決定 7 の「umT5 を取らない軽い使い方」が消える）。
 * MUST: 入力の形 `"t2v"` の世代（Wan2.1）の部品の並びは encoder の追加の前と同じ（2.1 の取得を変えない）。
 */
const wanComponentKeys = (
  generation: WanGenerationDescriptor,
  route: WanTextEncoderRoute,
): readonly string[] => [
  TRANSFORMER,
  VAE_DECODER_FIRST,
  VAE_DECODER_NEXT,
  ...(generation.ditInputForm === "ti2v" ? WAN_VAE_ENCODER_KEYS : []),
  ...(route === "gpu" ? [TEXT_ENCODER] : []),
];

/** 家族ごとに違う値（class が自分の 1 本を {@link loadWanFromPretrained} などへ渡す）。 */
export type WanFamilySpec = {
  /**
   * 文言の接頭辞（class 名）。共有の段（`text-stage.ts` / `dit-loop.ts` / `tile-decode.ts` /
   * `graph-io.ts`）へもこの値を渡す。
   */
  readonly owner: string;
  /** manifest の `pipeline` の名前と major（家族 admission の門 — ADR 0038 §6）。 */
  readonly pipeline: { readonly name: string; readonly major: number };
  /** 世代の記述子（受理集合・既定・fps・統計・VAE の照合・DiT の入力の形）。 */
  readonly generation: WanGenerationDescriptor;
};

/** Wan2.1 T2V 1.3B（`WanPipeline` — pipeline `wan/1`・{@link WAN21_GENERATION}）。 */
export const WAN21_FAMILY: WanFamilySpec = {
  owner: "WanPipeline",
  pipeline: { name: WAN_PIPELINE_NAME, major: WAN_PIPELINE_MAJOR },
  generation: WAN21_GENERATION,
};

/** Wan2.2 TI2V 5B（`WanTi2vPipeline` — pipeline `wan-ti2v/1`・{@link WAN22_TI2V_GENERATION}）。 */
export const WAN22_TI2V_FAMILY: WanFamilySpec = {
  owner: "WanTi2vPipeline",
  pipeline: { name: WAN_TI2V_PIPELINE_NAME, major: WAN_TI2V_PIPELINE_MAJOR },
  generation: WAN22_TI2V_GENERATION,
};

/**
 * 生成の要求を検査して計画にする（`generate` の入口・GPU に触る前の純粋な門 — 検査の本体と MUST は
 * {@link planWanRequest}）。`"precomputed"` の経路の入口で、プロンプトはテキスト埋め込み資産の集合で
 * 引く（{@link precomputedPromptGate}）。寸法とフレーム数の受理集合と既定は `generation`
 * （{@link WanFamilySpec.generation}）、拒否の文言の接頭辞は `owner`。
 *
 * NOTE: `export` は GPU 無しで門を縛るテストのため（`mod.ts` / サブパス面には出さない — ADR 0008）。
 */
export const planWanGeneration = (
  request: WanTi2vGenerateRequest,
  embeds: WanTextEmbeds,
  layout: PlanLayout,
  config: WanPipelineConfig,
  generation: WanGenerationDescriptor,
  owner: string,
): WanGenerationPlan<WanTextEmbedding> =>
  planWanRequest(request, precomputedPromptGate(embeds, owner), layout, config, generation);

/**
 * `"gpu"` の経路の入口（{@link planWanGeneration} と同じ検査の順・同じノブの門）。プロンプトは umT5 の
 * プロンプト層（`prompt_clean` の鏡像 → トークナイザ — {@link gpuPromptGate}）で id 列にする。拒否は
 * `ModelInputError`（前処理の拒否は派生の `PromptCleanError`）で、文言は直し方まで言う。
 * `negativePrompt` を省くと既定の negative（`text-stage.ts` の `WAN_DEFAULT_NEGATIVE_PROMPT`）を同じ門で
 * 符号化する。
 *
 * MUST: 符号化はここで済ませる — umT5 の重み（i8 で約 5.3 GiB）を上げてから語彙外で落ちる形にしない。
 *
 * NOTE: `export` は GPU 無しで門を縛るテストのため（`mod.ts` / サブパス面には出さない — ADR 0008）。
 */
export const planWanGpuGeneration = (
  request: WanTi2vGenerateRequest,
  encoder: WanPromptEncoder,
  layout: PlanLayout,
  config: WanPipelineConfig,
  generation: WanGenerationDescriptor,
): WanGenerationPlan<Int32Array<ArrayBuffer>> =>
  planWanRequest(request, gpuPromptGate(encoder), layout, config, generation);

/**
 * この家族が manifest の `session` と明示指定で受けるキー（受理表 — `session/options.ts`）。
 *
 * linear / 融合 attention の実行形 3 欄を受ける: 配布形の実用席 `f16+dit8-a8-attn8-s16` がこの 3 つを
 * 宣言する（ADR 0120 決定 1 / 7 — anima の同名の席と同じ宣言）。並列 GEMV と融合の 4 欄は受けない —
 * Wan の DiT のグラフでは効く席も数値も確かめていない組合せだから。受けないキーの宣言は重みを取る
 * 前に fail loudly（黙って既定で走らせない）。
 *
 * 席の選択は既存の `quant` だけで、利用者の明示指定の口は公開面に足さない（ADR 0120 決定 7 —
 * `resolveSessionOptions` の第 3 引数は空のまま。受理表は同じ 1 本なので、要求が出たら後から足せる）。
 *
 * NOTE: `export` は全家族の受理表の網羅を縛るテストのため（`mod.ts` / サブパス面には出さない —
 * ADR 0008）。
 */
export const WAN_SESSION_POLICY: FamilySessionPolicy = {
  linearCompute: true,
  attentionCompute: true,
  attentionScoreStorage: true,
  linearGemvReduce: false,
  stateAttentionReduce: false,
  fuseRmsNormAdd: false,
  fuseLinearStaticQuantize: false,
  packedStaticQuantize: false,
};

/** 家族 admission（{@link admitWan}）が確定させる材料。 */
type WanAdmission = {
  readonly config: WanPipelineConfig;
  readonly quantName: string;
  readonly quant: Quant;
  /** quant 宣言を受理表で通した実効設定（{@link WAN_SESSION_POLICY} — DiT の Session へ渡す）。 */
  readonly sessionOptions: SessionOptions;
  readonly gpuFeatures: GpuFeaturesSpec | undefined;
  readonly layout: WanVaeChunkLayout;
  readonly ropeBase: WanRopeBase;
  readonly dit: DitContract;
  /** 経路（`"gpu"` は umT5 のグラフの取り決めを伴う）。 */
  readonly textEncoder: WanTextAdmission;
  /** I2V の VAE encoder の 3 グラフの取り決め（DiT の入力の形が `"ti2v"` の世代だけ — 他は undefined）。 */
  readonly vaeEncoder: WanVaeEncoderContract | undefined;
};

/** Wan の class の内部状態（{@link loadWanFromPretrained} / {@link loadWanFromAssets} が組む）。 */
export type WanState = {
  readonly gpu: GpuContext;
  readonly ownsGpu: boolean;
  readonly config: WanPipelineConfig;
  readonly sessionOptions: SessionOptions;
  readonly transformer: ModelComponent;
  readonly vaeFirst: ModelComponent;
  readonly vaeNext: ModelComponent;
  /** I2V の VAE encoder（DiT の入力の形が `"ti2v"` の世代だけ — 他は undefined）。 */
  readonly vaeEncoder: WanVaeEncoder | undefined;
  readonly layout: WanVaeChunkLayout;
  readonly ropeBase: WanRopeBase;
  readonly dit: DitContract;
  /** 埋め込み資産（`"precomputed"` の受理集合・両経路の `prompts`）。 */
  readonly textEmbeds: WanTextEmbeds;
  readonly text: WanTextStage;
  readonly onRunDiagnostics?: (
    component: WanRunComponent,
    diagnostics: SessionDiagnostics,
  ) => void;
};

/**
 * 配布形から取得して組む（class の `fromPretrained` の本体 — 手順と MUST は `WanPipeline.fromPretrained`
 * の doc）。
 */
export const loadWanFromPretrained = async (
  spec: WanFamilySpec,
  ref: string | HubRepoRef | DistributionSource,
  options: WanFromPretrainedOptions,
): Promise<WanState> => {
  // 経路は取得の前に決める（取る部品が変わる）— 綴り違いは 1 バイトも取らずに落とす。
  const route = textEncoderRouteOf(options, spec.owner);
  const source = toManifestSource(ref, `${spec.owner}.fromPretrained`);
  // signal は取得層と構築の**両方**へ渡す（DL が終わった瞬間に中断が効かなくなる窓を作らない —
  // `hubLoadOptions` が写す・anima / irodori と同じ形）。
  const hubOptions = hubLoadOptions(options);
  const loaded = await loadManifest(source, hubOptions);
  assertTextEncoderDeclared(loaded.manifest, options, route, spec.owner);
  const choice = {
    ...(options.model === undefined ? {} : { model: options.model }),
    ...(options.quant === undefined ? {} : { quant: options.quant }),
  };
  const selection = resolveSelection(loaded.manifest, choice);
  const buildOptions: WanPipelineOptions = {
    ...(options.gpu === undefined ? {} : { gpu: options.gpu }),
    ...choice,
    textEncoder: route,
    ...(options.onRunDiagnostics === undefined
      ? {}
      : { onRunDiagnostics: options.onRunDiagnostics }),
    ...(options.signal === undefined ? {} : { signal: options.signal }),
  };
  // 家族の門は admission 席で通す（重みの part を取る前 — `src/hub/components.ts`）。
  const { admitted, assets, open } = await loadContainerComponents(
    `${spec.owner}.fromPretrained`,
    loaded,
    selection,
    wanComponentKeys(spec.generation, route),
    async (open) => {
      const admitted = await admitWan(spec, loaded.manifest, open, buildOptions);
      // 配布形が宣言した `requiredLimits` は**重みの part を取る前**にここで見る
      // （ADR 0089 決定 5 — 共有 GPU ならその limits、自前で取る経路はアダプタ実測値）。
      await assertRequiredLimitsBeforeDownload(
        admitted.quant.requiredLimits,
        buildOptions.gpu,
        `${spec.owner}: quant '${admitted.quantName}'`,
      );
      return admitted;
    },
    {
      ...hubOptions,
      ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
      ...(options.components === undefined ? {} : { components: options.components }),
    },
  );
  return await buildWan(spec, admitted, assets, open, buildOptions);
};

/**
 * 取得済みの manifest + 資産から組む（{@link WanAssets}）。容器を開いて、取得面と同じ家族 admission
 * （{@link admitWan}）と組み立て（{@link buildWan}）を通す — 2 面の違いは部品の供給口だけ。
 */
export const loadWanFromAssets = async (
  spec: WanFamilySpec,
  input: WanAssets,
  options: WanPipelineOptions,
): Promise<WanState> => {
  const route = textEncoderRouteOf(options, spec.owner);
  // 入口の検査（中断済みなら容器を 1 本も開かない — 開くのは GB 級の part 列の検証を含む）。
  await settleAbort(options.signal);
  assertTextEncoderDeclared(input.manifest, options, route, spec.owner);
  const buffer = (key: string): ArrayBuffer =>
    readAssetBuffer(spec.owner, "weights / assets", input.assets, key);
  const open = await assetComponentOpener(
    spec.owner,
    input.assets,
    buffer,
    wanComponentKeys(spec.generation, route),
  );
  const admitted = await admitWan(spec, input.manifest, open, options);
  return await buildWan(spec, admitted, input.assets, open, options);
};

/**
 * この manifest と部品のグラフ宣言を `spec` の世代（{@link WAN21_FAMILY} なら Wan2.1）として実行できるかを
 * 見る（家族 admission — 取得面では **重みの part を 1 バイトも取る前**に呼ばれる）。
 *
 * MUST: 家族の門はこの 1 本に集める（pipeline 名 / major・`pipelineConfig`・quant の `session`・
 * 共有 GPU の能力・計測の device・グラフ宣言 × ホストの取り決め）。後段へ散らすと、取得面では GB 級の
 * 重みを落とした**後**にしか落ちない（ADR 0070 決定 5）。
 * MUST: manifest の契約違反は **GPU を取りに行く前**に落とす（他の家族と同じ順序）。
 *
 * NOTE: RoPE の素表（`transformer` の容器の資産）はここで読む — 重み block と part を共有しない資産は
 * admission の席でも読める（`src/hub/components.ts`）。埋め込み資産・トークナイザ資産（manifest の
 * `assets`）はまだ届いていないので、中身の突合は {@link buildWan}（ここで見るのは宣言の有無）。
 */
const admitWan = async (
  spec: WanFamilySpec,
  manifest: Manifest,
  open: ComponentOpener,
  options: WanPipelineOptions,
): Promise<WanAdmission> => {
  // 段の境目（容器を開いた後・重みの part を取る前）。
  await settleAbort(options.signal);
  const route = textEncoderRouteOf(options, spec.owner);
  const modelName = options.model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, modelName)) {
    throw new Error(
      `${spec.owner}: model '${modelName}' は manifest に無い` +
        `（利用可能: ${manifest.available.models.join(" / ")}）`,
    );
  }
  const entry: ModelEntry = manifest.models[modelName];
  const { name, major } = entry.pipeline;
  if (name !== spec.pipeline.name) {
    throw new Error(
      `${spec.owner}: manifest の pipeline が '${name}/${major}'` +
        `（'${spec.pipeline.name}/${spec.pipeline.major}' が必要）`,
    );
  }
  if (major !== spec.pipeline.major) {
    // 「古い実装 × 新しいリポ」の沈黙劣化を止める唯一の門（ADR 0038 §6）。
    throw new Error(
      `${spec.owner}: pipeline '${name}/${major}' の major に未対応` +
        `（この実装が読めるのは ${spec.pipeline.name}/${spec.pipeline.major}）`,
    );
  }
  const config = parseWanPipelineConfig(entry.pipelineConfig);

  const quantName = options.quant ?? entry.defaultQuant;
  if (!Object.hasOwn(entry.quants, quantName)) {
    throw new Error(
      `${spec.owner}: quant '${quantName}' は manifest に無い` +
        `（利用可能: ${entry.available.quants.join(" / ")}）`,
    );
  }
  const quant = entry.quants[quantName];
  // `"gpu"` の経路はトークナイザ資産を読む — 宣言が無ければ umT5（約 5.3 GiB）を取る前に落とす
  // （中身は届いてから {@link buildWan} が見る）。
  assertTokenizerDeclared(entry, route, spec.owner);
  // 未対応の宣言は重みの part を取る前に落とす（全家族共通の 1 本）。
  const sessionOptions = resolveSessionOptions(
    WAN_SESSION_POLICY,
    quant.session,
    {},
    `${spec.owner}: quant '${quantName}'`,
  );
  const gpuFeatures = sessionGpuFeatures(quant.gpuFeatures, sessionOptions);

  if (options.gpu?.gpuTimingEnabled === true) {
    throw new Error(
      `${spec.owner}: gpuTiming が有効な device では VAE の段（1 タイル = 1 batch）を回せない` +
        "（runtime は計測の device で batch を開かない）— 計測なしの device を渡す",
    );
  }
  // MUST: 共有 GPU の能力不足（feature / device limit）はこの席で落とす — 重みを落とす前に判る
  // 唯一の家族門（後段の検査も同じ関数を呼ぶ）。
  if (options.gpu !== undefined) {
    assertGpuFeaturesGranted(gpuFeatures, options.gpu, `${spec.owner}: quant '${quantName}'`);
    assertRequiredLimitsSatisfied(
      quant.requiredLimits,
      options.gpu.limits,
      `${spec.owner}: quant '${quantName}'`,
    );
  }

  const transformer = open(TRANSFORMER);
  const layout = wanVaeChunkLayout(open(VAE_DECODER_FIRST), open(VAE_DECODER_NEXT));
  // 潜在のチャネル数は VAE の宣言から読み、記述子（統計の本数・unpatchify の倍率）と照合する。DiT の
  // `tokens` の幅との照合は、同じチャネル数で組んだ patch を ditContract が見る。
  assertWanVaeMatchesGeneration(layout, spec.generation, spec.owner);
  assertWanVaeTilesCover(layout, spec.generation, spec.owner);
  const ropeBase = parseWanRopeBase(await readWholeAsset(transformer.asset(ROPE_BASE)));
  const dit = ditContract(
    transformer,
    ropeBase,
    wanDitPatch(layout.latentChannels),
    spec.generation.ditInputForm,
    spec.owner,
  );
  const textEncoder = admitWanText(route, open, dit, spec.owner);
  // I2V の encoder は decoder の潜在と空間の圧縮に照らして見る（受理寸法の被覆を含む — `wanVaeEncoderContract`）。
  // 未知の DiT の入力の形は ditContract が先に落としている。
  const vaeEncoder = spec.generation.ditInputForm === "ti2v"
    ? wanVaeEncoderContract(
      {
        pre: open(VAE_ENCODER_PRE),
        attn: open(VAE_ENCODER_ATTN),
        post: open(VAE_ENCODER_POST),
      },
      layout,
      spec.generation,
      spec.owner,
    )
    : undefined;
  return {
    config,
    quantName,
    quant,
    sessionOptions,
    gpuFeatures,
    layout,
    ropeBase,
    dit,
    textEncoder,
    vaeEncoder,
  };
};

/**
 * admission を通った材料 + 資産から組む（{@link loadWanFromAssets} と {@link loadWanFromPretrained} が
 * 共有する 1 本）。埋め込み資産を解析して DiT の文脈入力と突き合わせ、GPU を取る（共有 GPU なら取らない）。
 *
 * MUST: 資産の解析と突合は **GPU を取りに行く前**（壊れた資産の真因を GPU 無し環境の別の例外で
 * 消さない — 他の家族と同じ順序）。Session は 1 本も張らない（VRAM の MUST — `./pipeline.ts` 冒頭の doc）。
 */
const buildWan = async (
  spec: WanFamilySpec,
  admitted: WanAdmission,
  assets: WanAssets["assets"],
  open: ComponentOpener,
  options: WanPipelineOptions,
): Promise<WanState> => {
  const { config, quantName, sessionOptions, gpuFeatures, layout, ropeBase, dit } = admitted;
  // 段の境目（資産が届いた後・解析の前）。
  await settleAbort(options.signal);
  const { textEmbeds, text } = await loadWanTextStage(
    admitted.textEncoder,
    assets,
    open,
    dit,
    options.signal,
    spec.owner,
  );

  await settleAbort(options.signal);
  const gpu = options.gpu ?? await acquireGpu(toAcquireGpuOptions(gpuFeatures));
  const ownsGpu = options.gpu === undefined;
  try {
    // MUST: GPU 取得**後**の中断検査は try の中に置く — 外に出すと、内部で取った device を誰も
    // 解放できないまま抜ける。`acquireGpu` の await 明けはマイクロタスクなので、ここでもマクロタスクへ
    // 譲る（待機中に積まれた中断のタスクはまだ実行されていない — anima と同じ）。
    await settleAbort(options.signal);
    // 宣言された feature は device 作成時にしか要求できない（ADR 0028）— 自前で取った device は
    // ここが唯一の門（共有 GPU は {@link admitWan} が同じ 1 本で見ている）。
    assertGpuFeaturesGranted(gpuFeatures, gpu, `${spec.owner}: quant '${quantName}'`);
    // 自前で取った device の limits もここで見る — `fromAssets` には取得前の事前判定の席が無く
    // （`fromPretrained` はアダプタ値で見ている）、見ないと宣言の不足が段ごとの Session の構築まで
    // 遅れる（VAE の段なら DiT の段を全部回した後）。共有 GPU は {@link admitWan} が見ている。
    if (ownsGpu) {
      assertRequiredLimitsSatisfied(
        admitted.quant.requiredLimits,
        gpu.limits,
        `${spec.owner}: quant '${quantName}'`,
      );
    }
    return {
      gpu,
      ownsGpu,
      config,
      sessionOptions,
      // 供給口は admission が見たものと**同じ 1 本**（`open` は開いた部品を引き当てるだけ）。
      transformer: open(TRANSFORMER),
      vaeFirst: open(VAE_DECODER_FIRST),
      vaeNext: open(VAE_DECODER_NEXT),
      vaeEncoder: admitted.vaeEncoder === undefined ? undefined : {
        pre: open(VAE_ENCODER_PRE),
        attn: open(VAE_ENCODER_ATTN),
        post: open(VAE_ENCODER_POST),
        contract: admitted.vaeEncoder,
      },
      layout,
      ropeBase,
      dit,
      textEmbeds,
      text,
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

/**
 * プロンプト（と I2V の条件画像）から動画を 1 本生成する（class の `generate` が直列化鎖に載せる本体 — 段の順序と
 * Session の寿命・中断は `./pipeline.ts` 冒頭の doc）。
 */
export const generateWanVideo = async (
  spec: WanFamilySpec,
  state: WanState,
  request: WanTi2vGenerateRequest,
): Promise<GeneratedVideo> => {
  const { dit } = state;
  const { onEvent, signal } = request;
  const emit = onEvent === undefined
    ? () => Promise.resolve()
    : async (event: WanGenerateEvent) => {
      await onEvent(event);
    };
  // 入口の門は経路ごと（どちらも GPU に触る前の純粋な検査）。プロンプトの文脈は経路ごとに作り、
  // encoder・DiT・VAE の段は同じ 1 本を通る。text の段は encoder の段の後に回すので、ここでは組み方だけを決める。
  let knobs: WanGenerationKnobs;
  let encodeContexts: () => Promise<WanContexts>;
  if (state.text.kind === "gpu") {
    const text = state.text;
    const plan = planWanGpuGeneration(
      request,
      text.encoder,
      state.layout,
      state.config,
      spec.generation,
    );
    knobs = plan;
    encodeContexts = async () => {
      // 段の境目（入口 / encoder → text）。直列化鎖の順番待ちの間に届いた中断もここで効く。
      await settleAbort(signal);
      return await encodeWanPrompts(state, text, plan, emit, signal, spec.owner);
    };
  } else {
    const plan = planWanGeneration(
      request,
      state.textEmbeds,
      state.layout,
      state.config,
      spec.generation,
      spec.owner,
    );
    knobs = plan;
    encodeContexts = () => Promise.resolve(precomputedContexts(plan, dit));
  }
  // I2V の encoder の段は最初（text の段より前 — ADR 0121 決定 11）。画像の無い要求は段を持たない。
  let condition: Float32Array<ArrayBuffer> | undefined;
  if (knobs.conditionImage !== undefined) {
    const { vaeEncoder } = state;
    if (vaeEncoder === undefined) {
      // 門（plan.ts）は入力の形 "ti2v" の世代でだけ画像を通し、その世代は構築で encoder を組むので、来たら配線の破れ。
      throw new Error(`${spec.owner}: 条件画像を受けたが VAE encoder の部品が組まれていない`);
    }
    // 段の境目（入口 → encoder）。
    await settleAbort(signal);
    condition = await encodeWanImageStage(
      { ...state, vaeEncoder },
      knobs.conditionImage,
      spec.generation,
      emit,
      signal,
      spec.owner,
    );
  }
  const contexts = await encodeContexts();
  // 段の境目（text → DiT）。
  await settleAbort(signal);
  const latents = await runWanDenoise(
    state,
    knobs,
    contexts,
    condition,
    emit,
    signal,
    spec.owner,
  );
  // 段の境目（DiT → VAE）。
  await settleAbort(signal);
  const data = await decodeWanVaeStage(
    state,
    knobs,
    latents,
    spec.generation,
    emit,
    signal,
    spec.owner,
  );
  return {
    frames: knobs.frames,
    width: knobs.width,
    height: knobs.height,
    fps: spec.generation.fps,
    data,
  };
};
