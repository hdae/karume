/**
 * Wan の text 段 — 経路（{@link WanTextEncoderRoute}）ごとの admission の部品・umT5 のグラフの取り決め・
 * 資産の読み込み・入口のプロンプトの門・DiT の文脈入力を組む段（`"gpu"` は umT5 の Session を張って
 * 畳む {@link encodeWanPrompts}・`"precomputed"` は資産の行を詰める {@link precomputedContexts}）。
 *
 * 段の順序と Session の寿命（text → DiT → VAE を 1 段ずつ張って畳む）は `./pipeline.ts` 冒頭の doc が
 * 正本で、ここは text 段の部品だけを持つ。Wan2.1 / 2.2 の class が共有する（umT5 は 2.1 と同じ —
 * ADR 0121 決定 8・10）。`owner` は文言の接頭辞（Wan2.1 は `"WanPipeline"`）。
 *
 * NOTE: 公開型（`WanPipelineOptions` / `WanGenerateEvent` / `WanRunComponent` / `WanAssets`）は
 * `./pipeline.ts` から `import type` で取る（型だけの参照は消去されるので循環 import にならない）。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 */

import { codecLayout, type GpuContext, type SessionDiagnostics } from "@karume/runtime";
import type { Manifest, ModelEntry } from "@karume/hub";

import { ModelInputError } from "../errors.ts";
import { settleAbort } from "../concurrency/abort.ts";
import { disposeSteps } from "../session/dispose-steps.ts";
import { readAssetBuffer, readAssetJson } from "../hub/asset-readers.ts";
import type { ComponentOpener, GraphOwner, ModelComponent } from "../hub/components.ts";
import type {
  WanAssets,
  WanGenerateEvent,
  WanPipelineOptions,
  WanRunComponent,
} from "./pipeline.ts";
import type { PromptGate, WanGenerationPlan } from "./plan.ts";
import { DIT_CONTEXT, type DitContract, type WanContexts } from "./dit-loop.ts";
import { assertDims, firstNonFinite, valueOf } from "./graph-io.ts";
import {
  findWanTextEmbedding,
  padWanTextEmbedding,
  parseWanTextEmbeds,
  type WanTextEmbedding,
  type WanTextEmbeds,
} from "./text-embeds.ts";
import { parseWanTokenizerAsset, WanPromptEncoder } from "./text/tokenizer.ts";
import { buildUmt5RelativePositionBuckets, WAN_UMT5_MAX_LENGTH } from "./umt5/relative-position.ts";
import {
  padUmt5Context,
  umt5SessionInputs,
  WAN_UMT5_INPUT_IDS,
  WAN_UMT5_RELATIVE_POSITION_BUCKETS,
} from "./umt5/session-io.ts";

/**
 * umT5 の部品（ADR 0119 追記「段 10d の設計」D — manifest では umT5 の配布リポへの越境参照）。取るのは
 * `"gpu"` の経路だけ。
 */
export const TEXT_ENCODER = "text_encoder";

/** テキストエンコーダの経路（{@link WanPipelineOptions.textEncoder}）。 */
export type WanTextEncoderRoute = NonNullable<WanPipelineOptions["textEncoder"]>;

/** 既定の経路（ADR 0119 決定 7 — 同じ文字列で経路が黙って変わらないよう、自動の切り替えは持たない）。 */
const DEFAULT_TEXT_ENCODER: WanTextEncoderRoute = "gpu";

/**
 * 構築オプションの経路の綴りを読む（省略は既定 `"gpu"`）。未知の綴りは素の `Error`（model / quant 名の
 * 綴り違いと同じ扱い — ADR 0107 決定 3）。MUST: 取得の前に呼ぶ — 経路で取る部品が変わる。
 */
export const textEncoderRouteOf = (
  options: Pick<WanPipelineOptions, "textEncoder">,
  owner: string,
): WanTextEncoderRoute => {
  const route: unknown = options.textEncoder ?? DEFAULT_TEXT_ENCODER;
  if (route !== "gpu" && route !== "precomputed") {
    throw new Error(
      `${owner}: textEncoder '${String(route)}' は 'gpu' / 'precomputed' のどちらでもない`,
    );
  }
  return route;
};

/**
 * `"gpu"` の経路で、選んだモデルが umT5 の部品を宣言しているかを見る（取得・容器を開く前）。
 *
 * WHY: 宣言が無いと、取得面・全量面の汎用の文言（「部品 'text_encoder' の容器が無い」）で落ち、
 * 既定が `"gpu"` であることも、umT5 を持たない配布形は `"precomputed"` で読めることも伝わらない。
 * 未知の model はここでは見ない（admission が利用可能な一覧つきで落とす）。
 */
export const assertTextEncoderDeclared = (
  manifest: Manifest,
  options: Pick<WanPipelineOptions, "model">,
  route: WanTextEncoderRoute,
  owner: string,
): void => {
  if (route !== "gpu") return;
  const modelName = options.model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, modelName)) return;
  if (!Object.hasOwn(manifest.models[modelName].weights, TEXT_ENCODER)) {
    throw new Error(
      `${owner}: model '${modelName}' の weights に umT5 の部品 '${TEXT_ENCODER}' が無い` +
        '（textEncoder の既定 "gpu" が取る。umT5 を持たない配布形を事前計算の埋め込みで回すなら ' +
        'textEncoder: "precomputed"）',
    );
  }
};

/**
 * テキスト埋め込み資産のキー（段 7 の manifest のモデル単位の `assets` — 決定 4）。両方の経路で読む —
 * `"precomputed"` の受理集合で、`"gpu"` でも `WanPipeline.prompts`（例示の一覧）の出所。
 */
const TEXT_EMBEDS = "text_embeds";

/**
 * umT5 のトークナイザ資産のキー（manifest のモデル単位の `assets` — ADR 0119 追記「段 10d の設計」C・
 * 形式 `karume-wan-umt5-tokenizer/1`）。読むのは `"gpu"` の経路だけ。
 */
const UMT5_TOKENIZER = "umt5_tokenizer";

/**
 * negative を省いたときの既定（公式 Wan2.1 の `wan_shared_cfg.sample_neg_prompt` の原文 — 全角の読点を
 * 含めて逐語。前処理が `,` へ畳む）。`"gpu"` の経路はこの文字列を GPU で符号化する（ADR 0119 追記
 * 「段 10d の設計」— 資産の行は使わない: 決定 7 の「positive も negative も GPU で作る」）。
 *
 * MUST: テキスト埋め込み資産の `negative` の行の原文とビット同一（`"precomputed"` の既定と同じ文字列を
 * 指す）。`wan_pipeline_test.ts` が recipe の固定プロンプト（`tools/export-recipes/wan/prompts.py` —
 * fixture `wan-text/parity.json` の `fixed-negative`）と資産の行の両方と突き合わせる。
 *
 * NOTE: `export` はその突き合わせのテストのため（`mod.ts` / サブパス面には出さない — ADR 0008）。
 */
export const WAN_DEFAULT_NEGATIVE_PROMPT: string =
  "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，" +
  "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，" +
  "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，" +
  "三条腿，背景人很多，倒着走";

/** 構築時に確かめた umT5 のグラフの取り決め（`"gpu"` の経路）。 */
type Umt5Contract = {
  /** グラフ出力の名前（`[1, L, width]` — width は DiT の文脈の幅）。 */
  readonly output: string;
};

/** umT5 のグラフ入力の宣言（無ければ fail loudly）。 */
const umt5Input = (
  owner: string,
  textEncoder: GraphOwner,
  name: string,
): GraphOwner["graph"]["inputs"][number] => {
  const spec = textEncoder.graph.inputs.find((input) => input.name === name);
  if (spec === undefined) {
    const declared = textEncoder.graph.inputs.map((input) => input.name).join(" / ");
    throw new Error(
      `${owner}: text_encoder のグラフ入力 '${name}' が無い（宣言: ${declared}）`,
    );
  }
  return spec;
};

/**
 * umT5 のグラフ宣言を、ホストが組む入力（id 列 `[1, L]` とバケット表 `[L, L]` の i32 — `umt5/session-io.ts`）
 * と DiT の文脈入力に突き合わせる（ADR 0119 決定 3・4・5）。
 *
 * MUST: 家族 admission（重みの part を取る前）で呼ぶ。ホストは自分の定数で入力を組み、出力を
 * `[1, L, contextWidth]` として DiT へ詰める（{@link padUmt5Context}）ので、宣言が違っていても落ちるのは
 * umT5 の重み（i8 で約 5.3 GiB）を上げた後の Session の shape 検査か、詰めの検査になる。
 *
 * - 入力はちょうど 2 本（`input_ids` `[1, L]`・`relative_position_buckets` `[L, L]`・どちらも i32）で、
 *   L は**同じ 1 つの記号次元**（有効長ごとに形を変えて渡す — 決定 4）。
 * - 出力は 1 本の f32 `[1, L, W]` で、L は入力と同じ記号・W は DiT の `encoder_hidden_states` の幅。
 * - 有効長の上限（{@link WAN_UMT5_MAX_LENGTH}）が DiT の文脈の行数に収まる。
 * - 格納は i8（`int8-sym` = per-channel）の重みと f32 の表だけ（決定 5 — GPU の移植の門で検証した
 *   組み合わせ。text 段の Session は quant の宣言を受けない〈`{}`〉ので、格納が実行の形を決める）。
 *   i8 を 1 本も持たない容器（全部 f32 など）も受けない。
 *
 * NOTE: `export` は家族 admission と、GPU 無しで門を縛るテストのため（`mod.ts` / サブパス面には
 * 出さない — ADR 0008）。
 */
export const umt5Contract = (
  textEncoder: GraphOwner,
  dit: Pick<DitContract, "contextRows" | "contextWidth">,
  owner: string,
): Umt5Contract => {
  const { graph } = textEncoder;
  const ids = umt5Input(owner, textEncoder, WAN_UMT5_INPUT_IDS);
  const buckets = umt5Input(owner, textEncoder, WAN_UMT5_RELATIVE_POSITION_BUCKETS);
  if (graph.inputs.length !== 2) {
    throw new Error(
      `${owner}: text_encoder のグラフ入力が ${graph.inputs.length} 本（` +
        `'${WAN_UMT5_INPUT_IDS}' と '${WAN_UMT5_RELATIVE_POSITION_BUCKETS}' の 2 本だけを組む）`,
    );
  }
  const length = ids.shape.at(1);
  if (typeof length !== "string") {
    throw new Error(
      `${owner}: text_encoder の '${WAN_UMT5_INPUT_IDS}' の軸 1 が記号次元でない` +
        `（${String(length)}）— ホストは有効長 L ごとに形を変えて渡す`,
    );
  }
  for (const [spec, expected] of [[ids, [1, length]], [buckets, [length, length]]] as const) {
    if (spec.dtype !== "i32") {
      throw new Error(
        `${owner}: text_encoder の '${spec.name}' の dtype ${spec.dtype} が i32 でない`,
      );
    }
    assertDims(owner, spec.shape, expected, `text_encoder の '${spec.name}'`);
  }
  if (graph.outputs.length !== 1) {
    throw new Error(
      `${owner}: text_encoder の出力が ${graph.outputs.length} 本（1 本の [1, L, W]）`,
    );
  }
  const [output] = graph.outputs;
  const value = valueOf(owner, textEncoder, output);
  if (value.dtype !== "f32") {
    throw new Error(
      `${owner}: text_encoder の出力 '${output}' の dtype ${value.dtype} が f32 でない`,
    );
  }
  assertDims(owner, value.shape, [1, length, dit.contextWidth], `text_encoder の出力 '${output}'`);
  if (WAN_UMT5_MAX_LENGTH > dit.contextRows) {
    throw new Error(
      `${owner}: umT5 の有効長の上限 ${WAN_UMT5_MAX_LENGTH} が DiT の文脈の行数 ${dit.contextRows} を超える`,
    );
  }
  let quantized = 0;
  for (const [name, initializer] of Object.entries(graph.initializers)) {
    if (initializer.storage === undefined) {
      // 共有の宣言（貸し手の Session の重みを借りる — 格納を持たない）。text 段に貸し手は居ない。
      throw new Error(
        `${owner}: text_encoder の initializer '${name}' が共有の宣言（text 段は重みを借りない）`,
      );
    }
    const layout = codecLayout(initializer.storage.codec);
    if (layout === "i8") quantized += 1;
    else if (layout !== "f32") {
      throw new Error(
        `${owner}: text_encoder の initializer '${name}' の格納 ${initializer.storage.codec} は受けない` +
          "（受けるのは i8 per-channel の重みと f32 の表だけ — ADR 0119 決定 5）",
      );
    }
  }
  if (quantized === 0) {
    throw new Error(
      `${owner}: text_encoder に i8 の重みが 1 本も無い（受けるのは i8 per-channel の重みと f32 の表 — ` +
        "ADR 0119 決定 5）",
    );
  }
  return { output };
};

/** 家族 admission が確定させる text 段の材料（`"gpu"` は umT5 のグラフの取り決めを伴う）。 */
export type WanTextAdmission =
  | { readonly route: "precomputed" }
  | { readonly route: "gpu"; readonly contract: Umt5Contract };

/**
 * `"gpu"` の経路はトークナイザ資産を読む — 宣言が無ければ umT5（約 5.3 GiB）を取る前に落とす（家族
 * admission で呼ぶ。中身は資産が届いてから {@link loadWanTextStage} が見る）。
 */
export const assertTokenizerDeclared = (
  entry: ModelEntry,
  route: WanTextEncoderRoute,
  owner: string,
): void => {
  if (route === "gpu" && !Object.hasOwn(entry.assets, UMT5_TOKENIZER)) {
    throw new Error(
      `${owner}: manifest の assets に umT5 のトークナイザ資産 '${UMT5_TOKENIZER}' が無い` +
        '（textEncoder: "gpu" の経路が読む。事前計算の埋め込みだけで回すなら textEncoder: "precomputed"）',
    );
  }
};

/**
 * 経路の admission（家族 admission で呼ぶ — 重みの part を取る前）。`"gpu"` は umT5 のグラフ宣言を DiT の
 * 文脈入力と突き合わせる（{@link umt5Contract}）。
 */
export const admitWanText = (
  route: WanTextEncoderRoute,
  open: ComponentOpener,
  dit: DitContract,
  owner: string,
): WanTextAdmission =>
  route === "gpu" ? { route, contract: umt5Contract(open(TEXT_ENCODER), dit, owner) } : { route };

/** 埋め込み資産の幅と有効長が DiT の文脈入力 `[1, rows, width]` に収まることを見る。 */
const assertEmbedsFitContext = (dit: DitContract, embeds: WanTextEmbeds, owner: string): void => {
  if (dit.contextWidth !== embeds.width) {
    throw new Error(
      `${owner}: '${DIT_CONTEXT}' の幅 ${dit.contextWidth} が埋め込み資産の幅 ${embeds.width} と違う`,
    );
  }
  const longest = Math.max(...embeds.entries.map((entry) => entry.tokens));
  if (longest > dit.contextRows) {
    throw new Error(
      `${owner}: 埋め込みの有効長 ${longest} が文脈の行数 ${dit.contextRows} を超える`,
    );
  }
};

/** text 段の材料（経路ごと）。 */
export type WanTextStage =
  /** 資産の埋め込みを引く（`textEmbeds` — Session を張らない）。 */
  | { readonly kind: "precomputed" }
  | {
    readonly kind: "gpu";
    /** プロンプト層（前処理 → トークナイザ — 入口の門）。 */
    readonly encoder: WanPromptEncoder;
    /** umT5 の部品（generate ごとに Session を張って畳む — 決定 11）。 */
    readonly component: ModelComponent;
    readonly contract: Umt5Contract;
  };

/**
 * text 段の資産を読んで組む（構築で呼ぶ — 資産が届いた後・GPU を取りに行く前）。埋め込み資産を解析して
 * DiT の文脈入力と突き合わせ、`"gpu"` の経路ではトークナイザ資産を解析して umT5 の部品を引き当てる。
 * 前後の段の境目の検査（`settleAbort`）は呼び手が置く。
 */
export const loadWanTextStage = async (
  textEncoder: WanTextAdmission,
  assets: WanAssets["assets"],
  open: ComponentOpener,
  dit: DitContract,
  signal: AbortSignal | undefined,
  owner: string,
): Promise<{ readonly textEmbeds: WanTextEmbeds; readonly text: WanTextStage }> => {
  const textEmbeds = parseWanTextEmbeds(
    readAssetBuffer(owner, "weights / assets", assets, TEXT_EMBEDS),
  );
  assertEmbedsFitContext(dit, textEmbeds, owner);
  let text: WanTextStage = { kind: "precomputed" };
  if (textEncoder.route === "gpu") {
    await settleAbort(signal);
    const encoder = new WanPromptEncoder(
      parseWanTokenizerAsset(
        readAssetJson(owner, "weights / assets", assets, UMT5_TOKENIZER),
        UMT5_TOKENIZER,
      ),
    );
    // トークナイザの上限はグラフの記号次元の上限・バケット表の生成器の上限と同じ値（決定 4）。
    // 違うと、上限の間のプロンプトがトークナイザを通ってからバケット表の生成で落ちる（大きい側）か、
    // 上流が受ける長さを黙って拒む（小さい側）。
    if (encoder.maxLength !== WAN_UMT5_MAX_LENGTH) {
      throw new Error(
        `${owner}: トークナイザ資産の maxLength ${encoder.maxLength} が umT5 の有効長の上限 ` +
          `${WAN_UMT5_MAX_LENGTH} と違う`,
      );
    }
    // 既定の negative はこの資産の門を通る MUST — 通らなければ資産の齟齬で、negativePrompt を省いた
    // 生成の入口で入力起因（`ModelInputError`）として落とすのは取り違え（呼び手の入力ではない）。
    try {
      encoder.encode(WAN_DEFAULT_NEGATIVE_PROMPT, "既定の negative");
    } catch (cause) {
      throw new Error(
        `${owner}: 既定の negative（公式の sample_neg_prompt）がトークナイザ資産の門を通らない`,
        { cause },
      );
    }
    text = {
      kind: "gpu",
      encoder,
      component: open(TEXT_ENCODER),
      contract: textEncoder.contract,
    };
  }
  return { textEmbeds, text };
};

/**
 * `"precomputed"` の経路のプロンプトの門 — テキスト埋め込み資産の集合で引く（集合の外は
 * `ModelInputError`）。
 */
export const precomputedPromptGate = (
  embeds: WanTextEmbeds,
  owner: string,
): PromptGate<WanTextEmbedding> => ({
  resolve: (text, what) => {
    const entry = findWanTextEmbedding(embeds, text);
    if (entry === undefined) {
      throw new ModelInputError(
        `${what} がテキスト埋め込み資産の集合に無い（事前計算の経路〈textEncoder: "precomputed"〉は ` +
          `${embeds.entries.length} 本だけを受ける: ${
            embeds.entries.map((entry) => entry.name).join(" / ")
          } — 原文か正規化後の文字列に完全一致させる。${owner}.prompts で引ける）`,
      );
    }
    return entry;
  },
  defaultNegative: () => {
    const defaults = embeds.entries.filter((entry) => entry.role === "negative");
    if (defaults.length !== 1) {
      throw new ModelInputError(
        `negativePrompt を省いたが、資産に negative の行が ${defaults.length} 本ある（1 本のときだけ既定にする）`,
      );
    }
    return defaults[0];
  },
});

/**
 * `"gpu"` の経路のプロンプトの門 — umT5 のプロンプト層（`prompt_clean` の鏡像 → トークナイザ —
 * {@link WanPromptEncoder.encode}）で id 列にする。`negativePrompt` を省くと
 * {@link WAN_DEFAULT_NEGATIVE_PROMPT} を同じ門で符号化する。
 */
export const gpuPromptGate = (encoder: WanPromptEncoder): PromptGate<Int32Array<ArrayBuffer>> => ({
  resolve: (text, what) => encoder.encode(text, what),
  defaultNegative: () => encoder.encode(WAN_DEFAULT_NEGATIVE_PROMPT, "negativePrompt（既定）"),
});

/** `"precomputed"` の経路の文脈入力（資産の行を DiT の文脈の行数までゼロで詰める — Session を張らない）。 */
export const precomputedContexts = (
  plan: WanGenerationPlan<WanTextEmbedding>,
  dit: DitContract,
): WanContexts => ({
  positive: padWanTextEmbedding(plan.positive, dit.contextRows, dit.contextWidth),
  negative: plan.negative === undefined
    ? undefined
    : padWanTextEmbedding(plan.negative, dit.contextRows, dit.contextWidth),
});

/** text 段が読む構築済みの材料（パイプラインの内部状態のうちこの段の分 — 構造で受ける）。 */
type WanTextEncodeState = {
  readonly gpu: GpuContext;
  readonly dit: DitContract;
  readonly onRunDiagnostics?: (
    component: WanRunComponent,
    diagnostics: SessionDiagnostics,
  ) => void;
};

/**
 * text の段（`"gpu"` の経路 — umT5 の Session を張り、positive → negative を 1 回ずつ回して畳む・`end` は
 * 畳んだ後）。出力 `[1, L, width]` を DiT の文脈の行数までゼロで詰めて返す（{@link padUmt5Context}）。
 *
 * MUST: DiT の段を張る前に畳む（ADR 0119 決定 11 — umT5 i8 5.30 GiB と DiT 段は B570 の天井に同居
 * できない）。畳む失敗で本体の失敗（run の失敗・中断・非有限の門）を上書きしない（DiT の段と同じ形）。
 * Session の実行オプションは `{}`（quant の `session` は DiT の Session だけが受ける — i8 の重みの実行は
 * 格納が決める・ADR 0119 決定 5）。
 *
 * MUST: 出力の有限性を見る（O(L·width)）。非有限の文脈を DiT へ渡すと、落ちるのは step 1 の潜在の門で、
 * 文言が DiT を指す（真因の段を取り違える）。
 */
export const encodeWanPrompts = async (
  state: WanTextEncodeState,
  text: Extract<WanTextStage, { readonly kind: "gpu" }>,
  plan: WanGenerationPlan<Int32Array<ArrayBuffer>>,
  emit: (event: WanGenerateEvent) => Promise<void>,
  signal: AbortSignal | undefined,
  owner: string,
): Promise<WanContexts> => {
  const { dit } = state;
  const observe = state.onRunDiagnostics;
  const prompts: readonly { readonly ids: Int32Array<ArrayBuffer>; readonly label: string }[] = [
    { ids: plan.positive, label: "prompt" },
    ...(plan.negative === undefined ? [] : [{ ids: plan.negative, label: "negativePrompt" }]),
  ];

  await emit({ kind: "stage", component: "text_encoder", at: "start" });
  const session = await text.component.createSession(state.gpu, {});
  const contexts: Float32Array<ArrayBuffer>[] = [];
  let failure: { readonly error: unknown } | undefined;
  try {
    for (const { ids, label } of prompts) {
      // 各 run の前（1 回の run は不可分 — 中断は次の run の前で効く）。
      await settleAbort(signal);
      const outputs = await session.run(
        umt5SessionInputs(ids, buildUmt5RelativePositionBuckets(ids.length)),
      );
      observe?.("text_encoder", session.diagnostics());
      if (!Object.hasOwn(outputs, text.contract.output)) {
        throw new Error(`${owner}: umT5 の出力 '${text.contract.output}' が無い`);
      }
      const context = padUmt5Context(
        outputs[text.contract.output],
        ids.length,
        dit.contextRows,
        dit.contextWidth,
      );
      const broken = firstNonFinite(context);
      if (broken !== -1) {
        throw new Error(
          `${owner}: umT5 の出力（${label}・${ids.length} トークン）の行 ` +
            `${Math.floor(broken / dit.contextWidth)}・列 ${broken % dit.contextWidth} が非有限` +
            `（${context[broken]}）— DiT の段へは渡さない`,
        );
      }
      contexts.push(context);
    }
  } catch (error) {
    failure = { error };
    throw error;
  } finally {
    await disposeSteps([
      () => {
        if (failure !== undefined) throw failure.error;
      },
      () => session.dispose(),
    ]);
  }
  await emit({ kind: "stage", component: "text_encoder", at: "end" });
  return { positive: contexts[0], negative: contexts.at(1) };
};
