/**
 * irodori の**受理（admission）**— 配布形の宣言と製品グラフがこの実装で走れるかを見る門と、
 * 門が確定させた材料。取得キーと資産の読み口もここに置く（門が最初に読む口だから）。
 *
 * 10 段の説明（何をどの順で回すか）は {@link "./pipeline.ts"} 冒頭の doc が正本で、ここが持つ
 * のは段へ入る**前**に通す門だけ。Session も GPU も資産の所有権も持たない（構築と所有権は
 * `./pipeline.ts`）ので、門は実 GPU も実資産も無しで直接叩ける
 * （`tests/irodori_admission_test.ts`）。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 */

import type { SessionOptions } from "@karume/runtime";
import type { Manifest, ModelEntry, Quant } from "@karume/hub";

import {
  IRODORI_PIPELINE_MAJOR,
  IRODORI_PIPELINE_NAME,
  type IrodoriPipelineConfig,
  parseIrodoriPipelineConfig,
} from "./config.ts";
import type { IrodoriAssets, IrodoriPipelineOptions } from "./pipeline.ts";
import { settleAbort } from "../concurrency/abort.ts";
import {
  assertGpuFeaturesGranted,
  assertRequiredLimitsSatisfied,
} from "../session/gpu-features.ts";
import { toSessionOptions } from "../session/options.ts";
import {
  assetComponentOpener,
  type ComponentOpener,
  type GraphOwner,
  type ModelComponent,
} from "../hub/components.ts";
import { readAssetBuffer, readAssetJson } from "../hub/asset-readers.ts";
import { assertGraphInputDim } from "../hub/graph-gates.ts";

/** manifest の weights / assets 表に現れる取得キー（ADR 0041 §3 の規約名）。 */
export const BACKBONE = "backbone";
export const TEXT_PROJ = "text_proj";
export const CAPTION_PROJ = "caption_proj";
export const SPEAKER = "speaker";
export const DURATION = "duration";
export const DIT = "dit";
export const CODEC_DECODER = "codec_decoder";
export const CODEC_ENCODER = "codec_encoder";
export const TOKENIZER = "tokenizer";

/**
 * 取得済みバイト列を `openModel` へ渡せる ArrayBuffer にする（門の本体は
 * {@link readAssetBuffer}）。
 */
const assetBuffer = (
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  key: string,
): ArrayBuffer => readAssetBuffer("irodori", "weights / assets", assets, key);

/**
 * 全量面（`fromAssets`）のコンポーネント供給口（受け口の実装は 7 家族共有 —
 * {@link assetComponentOpener}）。素の 1 本は `openModel` で開いて全量面で組み、shard 分割形
 * （`dit[0]` / `dit[1]` / …）は `fromPretrained` と同じ shard 逐次面へ流す。
 */
export const assetOpener = (assets: IrodoriAssets["assets"]): ComponentOpener =>
  assetComponentOpener("irodori", assets, (key) => assetBuffer(assets, key));

/**
 * 資産 JSON を読む（decode / parse の門は {@link readAssetJson}）。
 *
 * NOTE: `export` は門を直接叩くテストのため（`fromAssets` 経由で此処へ届くには実 IR
 * コンテナ 8 本が要る）。`mod.ts` / サブパス面には出さない（ADR 0008）。
 */
export const assetJson = (
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  key: string,
): unknown => readAssetJson("irodori", "weights / assets", assets, key);

/**
 * グラフ入力の 1 軸ぶんの**静的**次元が `pipelineConfig` の宣言と一致することを見る。
 *
 * MUST: 落とさない。条件 state の宣言長や幅がホストの値とずれても、右 pad と行数計算は
 * そのまま通り（shape は合う）、**別の位置の条件を読んだ**結果が沈黙で出る。
 */
const assertStaticDim = (
  model: GraphOwner,
  inputName: string,
  axis: number,
  expected: number,
  where: string,
): void => assertGraphInputDim("irodori", model, inputName, axis, expected, where);

/**
 * グラフ**出力**の 1 軸が「記号 × 係数」の派生次元で、その係数が宣言と一致することを見る。
 *
 * MUST: 落とさない。decoder の出力倍率（1 latent フレーム → `hopLength` サンプル）がずれても
 * 出力は「それらしい長さの波形」になり、秒指定の切り出しと末尾トリムだけが別の位置を指す。
 */
const assertOutputScale = (
  model: GraphOwner,
  axis: number,
  expected: number,
  where: string,
): void => {
  const name = model.graph.outputs[0];
  const value = model.graph.values[name];
  if (value === undefined) {
    throw new Error(`irodori: グラフ出力 '${name}' の宣言が無い（${where}）`);
  }
  const symbol = model.graph.symbols[0];
  // 正準表記は `coeff·sym` で、**係数 1 は省略**する（`format/dims.ts` の `formatDim`）。
  // 綴りを合わせないと、倍率 1 の正しいグラフをここが誤って拒否する。
  const canonical = `${expected === 1 ? "" : expected}${symbol}`;
  const dim = value.shape[axis];
  if (symbol === undefined || dim !== canonical) {
    throw new Error(
      `irodori: ${where} — グラフ出力 '${name}' の軸 ${axis} が ${String(dim)}、` +
        `pipelineConfig からの期待は '${canonical}'`,
    );
  }
};

/**
 * グラフ**出力**の 1 軸が宣言どおりの**静的**次元であることを見る。
 *
 * MUST: 落とさない。encoder の latent 幅が `pipelineConfig` の `latentDim` とずれても、後段の
 * patch は「幅 × patchSize」で割り切れる限り通ってしまい、**別のチャネルを話者特徴として
 * 読んだ**結果が沈黙で出る。
 */
const assertOutputDim = (
  model: GraphOwner,
  axis: number,
  expected: number,
  where: string,
): void => {
  const name = model.graph.outputs[0];
  const value = model.graph.values[name];
  if (value === undefined) {
    throw new Error(`irodori: グラフ出力 '${name}' の宣言が無い（${where}）`);
  }
  const dim = value.shape[axis];
  if (dim !== expected) {
    throw new Error(
      `irodori: ${where} — グラフ出力 '${name}' の軸 ${axis} が ${String(dim)}、` +
        `pipelineConfig は ${expected}`,
    );
  }
};

/** 家族 admission（{@link admitIrodori}）が確定させる材料。 */
export type IrodoriAdmission = {
  readonly config: IrodoriPipelineConfig;
  readonly quantName: string;
  readonly quant: Quant;
  /**
   * `dit` の記号次元 S の名前（admission が唯一の導出点）。
   *
   * MUST: 実行時に `graph.symbols` から引き直さない — 「記号は 1 本」を確かめた席と使う席が
   * 離れると、経路によって検査が走ったり走らなかったりする（CLAUDE.md の「導出値は source of
   * truth から 1 度だけ導く」）。
   */
  readonly ditSymbol: string;
  readonly ditSessionOptions: SessionOptions;
  readonly backbone: ModelComponent;
  readonly textProj: ModelComponent;
  readonly captionProj: ModelComponent;
  readonly speaker: ModelComponent;
  readonly duration: ModelComponent;
  readonly dit: ModelComponent;
  readonly codecDecoder: ModelComponent;
  readonly codecEncoder: ModelComponent;
};

/**
 * この manifest とこのグラフを irodori として実行できるかを見る（`hub/components.ts` の
 * 家族 admission 席 — shard 面では**重み shard を 1 バイトも取る前**に呼ばれる）。
 *
 * MUST: 家族の門はこの 1 本に集める。後段へ散らすと、shard 面では GB 級の重みを落とした
 * **後**にしか落ちない（ADR 0070 決定 5 の文面より実装が狭くなる）。
 * MUST: manifest の契約違反と**グラフとの突合**は **GPU を取りに行く前**に落とす。順序が
 * ずれると、GPU の無い環境では別の例外に化けて「何が悪かったのか」が読み手に伝わらない。
 *
 * NOTE: 各段は不可分（`openModel` を途中で畳む口は無い）なので、
 * {@link IrodoriPipelineOptions.signal} の検査は**段の境目**にだけ置き、そこでイベントループへ
 * 1 度譲ってから検査する（{@link settleAbort}）— 同期解析の最中に届いた中断は次の境目で効く
 * （`options.gpu` 供給時も同様）。グラフとの突合（`assertStaticDim` 群）は開いたコンテナの
 * ヘッダを読むだけで、`openModel` 1 本より桁で軽いので境目を割らない。
 * NOTE: 資産（tokenizer）の解析はこの席へ置けない — admission の時点では extras をまだ
 * 取っていない（取ってからでは重み prefetch より前という位置が保てない）ので
 * {@link "./pipeline.ts"} の `buildIrodoriState` に残る。
 */
export const admitIrodori = async (
  manifest: Manifest,
  open: ComponentOpener,
  options: IrodoriPipelineOptions,
): Promise<IrodoriAdmission> => {
  // 中断の検査は**段の境目**に置く。入口が最初の 1 本: 中断済みで呼ばれたら manifest にも
  // 資産にも触らずに返す。
  options.signal?.throwIfAborted();
  const modelName = options.model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, modelName)) {
    throw new Error(
      `IrodoriPipeline: model '${modelName}' は manifest に無い` +
        `（利用可能: ${manifest.available.models.join(" / ")}）`,
    );
  }
  const entry: ModelEntry = manifest.models[modelName];
  const { name, major } = entry.pipeline;
  if (name !== IRODORI_PIPELINE_NAME) {
    throw new Error(
      `IrodoriPipeline: manifest の pipeline が '${name}/${major}'` +
        `（'${IRODORI_PIPELINE_NAME}/${IRODORI_PIPELINE_MAJOR}' が必要）`,
    );
  }
  if (major !== IRODORI_PIPELINE_MAJOR) {
    // 「古い実装 × 新しいリポ」の沈黙劣化を止める唯一の門（ADR 0038 §6）。
    throw new Error(
      `IrodoriPipeline: pipeline '${name}/${major}' の major に未対応` +
        `（この実装が読めるのは ${IRODORI_PIPELINE_NAME}/${IRODORI_PIPELINE_MAJOR}）`,
    );
  }
  const config = parseIrodoriPipelineConfig(entry.pipelineConfig);

  const quantName = options.quant ?? entry.defaultQuant;
  if (!Object.hasOwn(entry.quants, quantName)) {
    throw new Error(
      `IrodoriPipeline: quant '${quantName}' は manifest に無い` +
        `（利用可能: ${entry.available.quants.join(" / ")}）`,
    );
  }
  const quant = entry.quants[quantName];

  // 資産の解析は GPU より前（docstring の順序 MUST）。8 本の `openModel` はそれぞれ不可分なので、
  // 中断の検査はその境目に置く。
  await settleAbort(options.signal);
  const backbone = open(BACKBONE);
  await settleAbort(options.signal);
  const textProj = open(TEXT_PROJ);
  await settleAbort(options.signal);
  const captionProj = open(CAPTION_PROJ);
  await settleAbort(options.signal);
  const speaker = open(SPEAKER);
  await settleAbort(options.signal);
  const duration = open(DURATION);
  await settleAbort(options.signal);
  const dit = open(DIT);
  await settleAbort(options.signal);
  const codecDecoder = open(CODEC_DECODER);
  await settleAbort(options.signal);
  const codecEncoder = open(CODEC_ENCODER);

  // グラフの宣言と pipelineConfig の突合（ホストの式が読む数は全て config 由来）。
  assertStaticDim(dit, "x_t", 2, config.latentDim, "latentDim");
  assertStaticDim(dit, "t_embed", 1, config.timestepEmbedDim, "timestepEmbedDim");
  assertStaticDim(dit, "text_state", 1, config.maxTextLen, "maxTextLen");
  assertStaticDim(dit, "text_state", 2, config.textDim, "textDim");
  assertStaticDim(dit, "speaker_state", 1, config.speakerRows, "speakerRows");
  assertStaticDim(dit, "speaker_state", 2, config.speakerDim, "speakerDim");
  assertStaticDim(dit, "caption_state", 1, config.maxCaptionLen, "maxCaptionLen");
  assertStaticDim(dit, "caption_state", 2, config.captionDim, "captionDim");
  assertStaticDim(duration, "text_state", 2, config.textDim, "textDim");
  assertStaticDim(duration, "speaker_vec", 1, config.speakerDim, "speakerDim");
  assertStaticDim(duration, "caption_vec", 1, config.captionDim, "captionDim");
  // 参照 latent の patch 幅（latentDim × speakerPatchSize）が speaker の入力幅と一致する。
  assertStaticDim(
    speaker,
    "latent",
    2,
    config.latentDim * config.speakerPatchSize,
    "latentDim × speakerPatchSize",
  );
  // codec decoder は latent を 1 フレーム = hopLength サンプルへ展開する。入力幅と**出力の
  // 派生次元の係数**の両方を見る（係数だけがずれた資産は shape が合ったまま通り、切り出しと
  // 末尾トリムのサンプル位置だけが静かに別の場所を指す）。
  assertStaticDim(codecDecoder, "latent", 2, config.latentDim, "latentDim");
  assertOutputScale(codecDecoder, 2, config.hopLength, "hopLength");
  // encoder は逆向き（`[1,T,hopLength]` の波形 → `[1,T,latentDim]`）。入力のフレーム幅が
  // `hopLength` でないと、ホストが並べた波形が**1 フレームずつずれて**読まれる。
  assertStaticDim(codecEncoder, "wav", 2, config.hopLength, "hopLength");
  assertOutputDim(codecEncoder, 2, config.latentDim, "latentDim");
  // `dit` の記号次元は S の 1 本だけ（常駐経路は毎 enqueue この名前で束縛を渡す）。家族の門は
  // この 1 本に集める MUST の一部で、実行時に置くと ①GB 級の重みを落とした後にしか落ちない
  // ②ホスト経路（`gpuTiming` 有効 device / `onEvent` 購読）では走らず、同じ配布形が観測経路
  // ごとに違う文言で落ちる、の 2 つが起きる。
  const ditSymbols = dit.graph.symbols;
  if (ditSymbols.length !== 1) {
    throw new Error(`irodori: dit の記号次元が 1 本でない（[${ditSymbols.join(", ")}]）`);
  }
  const ditSymbol = ditSymbols[0];

  const ditSessionOptions = toSessionOptions(quant.session);

  // MUST: 共有 GPU の能力不足（feature / device limit）はこの席で落とす — 自前で取る場合と
  // 違って `acquireGpu` を待つ理由が無く、重みを落とす前に判る唯一の家族門（要求と検査の
  // 写像は `session/gpu-features.ts` の 1 本で、後段の検査も同じ関数を呼ぶ）。
  if (options.gpu !== undefined) {
    assertGpuFeaturesGranted(
      quant.gpuFeatures,
      options.gpu,
      `IrodoriPipeline: quant '${quantName}'`,
    );
    assertRequiredLimitsSatisfied(
      quant.requiredLimits,
      options.gpu.limits,
      `IrodoriPipeline: quant '${quantName}'`,
    );
  }

  return {
    config,
    quantName,
    quant,
    ditSymbol,
    ditSessionOptions,
    backbone,
    textProj,
    captionProj,
    speaker,
    duration,
    dit,
    codecDecoder,
    codecEncoder,
  };
};
