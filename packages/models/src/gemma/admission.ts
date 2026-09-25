/**
 * gemma4 の**受理（admission）**— 配布形の宣言・製品グラフ・実行時ノブがこの実装で走れるかを
 * 見る門と、門が確定させた材料。
 *
 * MUST: 家族の門はここへ集める。{@link Gemma4Pipeline} 側へ散らすと、取得面では GB 級の重みを
 * 落とした**後**にしか落ちない — どの門がどの位置（GPU を取る前 / 資産を読む前 / 重みの part を
 * 取る前）で呼ばれるかは各関数の doc が名乗る。
 *
 * ここが持つのは受理集合の判定だけで、Session も GPU も資産の読み口も持たない（構築と所有権は
 * `./pipeline.ts`）。おかげで門は実 GPU も実資産も無しで直接叩ける（`tests/gemma4_config_test.ts`）。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 */

import { assertChunkBuckets } from "@karume/runtime";
import type { Manifest, ModelEntry, Quant } from "@karume/hub";

import { ModelInputError } from "../errors.ts";
import type { GraphOwner } from "../hub/components.ts";
import type { GenerationGraph } from "../generation/program.ts";
import {
  GEMMA4_PIPELINE_MAJOR,
  type Gemma4PipelineConfig,
  parseGemma4PipelineConfig,
} from "./config.ts";
import { assertGemma4QatModel, type Gemma4QatModel } from "./qat.ts";
import { GEMMA4_ROPE_LAYER_TYPES, GEMMA4_ROPE_PARTS, gemma4RopeInputName } from "./rope.ts";
import { admitGemma4Drafter, type Gemma4DrafterAdmission } from "./speculative.ts";

export type GemmaFamily = "gemma4" | "gemma4-qat";

/**
 * family ごとの入口の名前（例外の接頭辞と `where` の前半を**ここ 1 箇所**から出す）。
 *
 * MUST: 共通基底が投げる文言も family の名前を名乗る — `Gemma4QatPipeline.fromPretrained` を
 * 叩いた利用者が `Gemma4Pipeline: ...` を受け取ると、どの入口の話か辿れない。
 */
export const gemmaEntryName = (family: GemmaFamily): string =>
  family === "gemma4" ? "Gemma4Pipeline" : "Gemma4QatPipeline";

/**
 * family と、QAT だけが持つ model 名（{@link admitGemma4Qat} が**グラフから確定**させた値）。
 *
 * MUST: model 名を後段で再導出しない（判別規則が 2 実装に割れると、モデルが増えたときに
 * 片方だけが古い写像を使い続ける）。`Gemma4QatModel` に欄が増えれば型検査が欠落を教える。
 */
export type GemmaFamilyAdmission =
  | { readonly family: "gemma4" }
  | { readonly family: "gemma4-qat"; readonly model: Gemma4QatModel };

/**
 * 製品グラフの出口の本数（**順序が契約** — 出力 0 = 選んだ行の logits `[1,R,V]`・
 * 出力 1 = 同じ行の最終 norm 後 hidden `[1,R,H]`）。
 *
 * 名前ではなく順序で引く（`vocabSizeOf` / {@link buildGemma4Program}）— 出口の綴りは
 * 焼き手の内部名で、配布形ごとに動きうるためである。
 */
const GRAPH_OUTPUTS = 2;

/**
 * 家族 admission（GPU を取りに行く前・取得面では重み prefetch の前に通す門）が確定させる材料。
 *
 * NOTE: コンポーネントの実体はここに載せない — 供給口（`ComponentOpener`）が前段と後段へ
 * **同じ 1 本**で渡るので、開いた部品を材料へ写すと「前段が見たのと別の部品を後段が握る」形が
 * 書ける（差し替え席が入って以降はなおさら）。
 * NOTE: PLE loader も載せない — `wiring.derivedInputs.derive` の閉包が持つのが唯一の参照で、
 * 席を 2 つ作ると「片方だけ差し替えた」形が書ける。
 */
export type Gemma4Admission = {
  readonly config: Gemma4PipelineConfig;
  /** 最終行 logits 出口の語彙数（id 空間の相互照合の基準 — ADR 0085 決定 5）。 */
  readonly vocabSize: number;
  /** full スロットの容量記号（`createGenerationContext` の束縛点）。 */
  readonly capacitySymbol: string;
  /** 投機を指定したときだけ確定する drafter の突合結果。 */
  readonly drafter?: {
    readonly admission: Gemma4DrafterAdmission;
  };
};

/**
 * 選んだ行の logits 出口の語彙数をグラフから引く（`[1, R, V]` — ADR 0083 決定 6）。
 *
 * MUST: 呼び手に宣言させない。V は主 embedding の行数そのもので、宣言と食い違えば
 * PLE の索引との相互照合（ADR 0085 決定 5）が**間違った基準**で通ってしまう。形の検査は
 * `createGenerationProgram` が同じ値でもう一度行う。
 *
 * MUST: 出口は**2 本ちょうど**（出力 0 = logits・出力 1 = 最終 norm 後 hidden）。順序は IR の
 * 契約で、名前で引かないのは配布形の綴りに依存しないためである（`capacitySymbolOf` と同じ
 * 流儀）。出口 1 本の旧配布形は**ここで**落とす — 互換分岐を書くと「hidden の無い資産で
 * 投機が黙って組めない」形が残る。
 */
const vocabSizeOf = (graph: GenerationGraph, entry: string): number => {
  if (graph.outputs.length !== GRAPH_OUTPUTS) {
    throw new Error(
      `${entry}: グラフ出力が ${graph.outputs.length} 本` +
        `（製品グラフの出口は logits + hidden の ${GRAPH_OUTPUTS} 本 — ADR 0083 決定 6）`,
    );
  }
  const name = graph.outputs[0];
  if (!Object.hasOwn(graph.values, name)) {
    throw new Error(`${entry}: グラフ出力 '${name}' の値情報が無い`);
  }
  const shape = graph.values[name].shape;
  const vocab = shape[2];
  if (shape.length !== 3 || typeof vocab !== "number") {
    throw new Error(
      `${entry}: グラフ出力 '${name}' の shape [${shape.join(",")}] が [1,R,V] でない`,
    );
  }
  return vocab;
};

/**
 * 最終 norm 後 hidden 出口の幅をグラフから引く（`[1, R, H]` — 出力 1・ADR 0083 決定 6）。
 *
 * 呼ぶのは投機のときだけ（drafter の入力 `hidden` の幅がこれと一致する MUST）。本数と順序は
 * {@link vocabSizeOf} が既に見ている。
 */
const hiddenSizeOf = (graph: GenerationGraph, entry: string): number => {
  const name = graph.outputs[1];
  if (!Object.hasOwn(graph.values, name)) {
    throw new Error(`${entry}: グラフ出力 '${name}' の値情報が無い`);
  }
  const shape = graph.values[name].shape;
  const hidden = shape[2];
  if (shape.length !== 3 || typeof hidden !== "number") {
    throw new Error(
      `${entry}: グラフ出力 '${name}' の shape [${shape.join(",")}] が [1,R,H] でない`,
    );
  }
  return hidden;
};

/**
 * full スロットの容量記号をグラフから引く。
 *
 * 記号は「入力 shape から決まらないもの」がちょうど 1 本のはずで（chunk 長の記号は
 * `input_ids` の 2 次元目から決まる・容量記号は states にしか現れない）、それを
 * `createGenerationContext` の束縛点へ渡す（ADR 0066 追記 7）。綴りを定数で持たないのは、
 * 資産側の綴りが変わったときに**黙って束縛されない記号**が残るのを避けるため。
 */
const capacitySymbolOf = (graph: GenerationGraph, entry: string): string => {
  const fromInputs = new Set<string>();
  for (const input of graph.inputs) {
    for (const dim of input.shape) {
      if (typeof dim === "string") fromInputs.add(dim);
    }
  }
  const free = graph.symbols.filter((symbol) => !fromInputs.has(symbol));
  if (free.length !== 1) {
    throw new Error(
      `${entry}: 入力 shape から決まらない記号が ${free.length} 本` +
        `（[${free.join(", ")}] — full スロットの容量記号 1 本であること）`,
    );
  }
  return free[0];
};

/**
 * RoPE 派生入力 4 本の宣言形（`[1, M, headDim]`）と `pipelineConfig.rope.<層種>.headDim` の突合。
 *
 * MUST: setup で見られる配線は setup で見る。`createGenerationProgram` が見るのは派生入力の
 * **名前の被覆**だけなので、幅の食い違い（層種別の取り違え = sliding 256 と full 512 の引き違い。
 * exporter 側 `rope.py` が `head_dim` / `global_head_dim` の分岐で自認している間違い方）は、
 * ホストが渡す表を初 `run` が受けるまで落ちない — 3.7GiB のロードの**後**で、しかも文言は
 * 「要素数が shape と合わない」になる。焼く側の鏡像は `export_decode.py` の `assert_rope_inputs`。
 *
 * NOTE: 内部の口だが export してあるのは、この単位なら宣言の突合を実 GPU も実資産も無しで
 * 縛れるため（`tests/gemma4_config_test.ts` — siglip2 の `assertStaticDim` と同じ流儀）。
 */
export const assertRopeInputShapes = (
  graph: GenerationGraph,
  config: Gemma4PipelineConfig,
  // NOTE: 既定が通常 Gemma の名前なのは {@link assertChunkLength} と同じ理由（直接叩く検査は
  // 入口を持たない）。実経路（{@link admitGemma4}）は必ず family の名前を渡す。
  entry: string = gemmaEntryName("gemma4"),
): void => {
  for (const layerType of GEMMA4_ROPE_LAYER_TYPES) {
    const { headDim } = config.rope[layerType];
    for (const part of GEMMA4_ROPE_PARTS) {
      const name = gemma4RopeInputName(layerType, part);
      const input = graph.inputs.find((entry) => entry.name === name);
      if (input === undefined) {
        throw new Error(
          `${entry}: グラフ入力 '${name}' が無い（RoPE がホスト供給の資産でない）`,
        );
      }
      if (input.shape.length !== 3 || input.shape[2] !== headDim) {
        throw new Error(
          `${entry}: グラフ入力 '${name}' の shape [${input.shape.join(",")}] が` +
            ` pipelineConfig.rope.${layerType}.headDim ${headDim} と食い違う` +
            `（[1, M, ${headDim}] が要る）`,
        );
      }
    }
  }
};

/**
 * この製品グラフを gemma4 として実行できるかを見る（**重みの part を 1 バイトも取る前**）。
 *
 * MUST: 家族の門はこの 1 本に集める（他ファミリの `admit*` と同じ規律 — `hub/components.ts` の
 * {@link FamilyAdmission} 席で呼ばれる）。後段へ散らすと、取得面では GB 級の重みを落とした
 * **後**にしか落ちない。
 *
 * NOTE: tokenizer の解析はここに置けない — admission の時点では manifest の `assets` をまだ
 * 取っていない（取ってからでは重み prefetch より前という位置が保てない）ので、
 * {@link buildGemma4Program} に残る（anima の `#admit` と同じ分け方）。PLE の索引だけは
 * **容器の資産**なので、この席と同じ位置で読める（`./ple-index.ts` の `readGemma4PleIndex`）。
 *
 * NOTE: `config` **単体**の検査はここには無い — 2 つの入口が**どちらも**
 * {@link parseGemma4PipelineConfig} を通してから呼ぶ（値域・関係・未知キーの門はそこが正本で、
 * 同じ検査を 2 実装持たない）。ここが見るのは宣言**とグラフの突合**だけで、
 * {@link assertRopeInputShapes} がその 1 本である（グラフはこの席で初めて手に入る）。
 */
export const admitGemma4 = (
  component: GraphOwner,
  config: Gemma4PipelineConfig,
  drafter: GraphOwner | undefined,
  // MUST: family の入口名（{@link gemmaEntryName}）— 両家族がこの 1 本を通るので、配下の門の
  // 文言が固定の名前を名乗ると QAT の利用者に別の入口の話として届く。
  entry: string,
): Gemma4Admission => {
  const { graph } = component;
  assertRopeInputShapes(graph, config, entry);
  const vocabSize = vocabSizeOf(graph, entry);
  const capacitySymbol = capacitySymbolOf(graph, entry);
  return {
    config,
    vocabSize,
    capacitySymbol,
    // drafter の門は `./speculative.ts` が持つ（借り物スロット・共有 initializer の綴りは
    // 投機の知識で、target の門とは別の 1 本）。target の材料は**確定したもの**を渡す。
    ...(drafter === undefined ? {} : {
      drafter: {
        admission: admitGemma4Drafter(entry, drafter.graph, {
          graph,
          rope: config.rope,
          hiddenSize: hiddenSizeOf(graph, entry),
          capacitySymbol,
        }),
      },
    }),
  };
};

/**
 * 実行時ノブの `chunkLength` を検査して返す（{@link Gemma4PipelineOptions.chunkLength} の門）。
 *
 * MUST: 2 以上（グラフの chunk 記号は prefill 形の最小 2 で焼かれており、1 行の chunk は decode 形
 * として流れる）・配布形が宣言する `maxChunkLength` 以下・`maxPosition` 以下。宣言
 * （`parseGemma4PipelineConfig`）が同じ関係を既定値に対して見るので、ここが見るのは**呼び手が
 * 上書きした値**である。
 *
 * MUST: `maxChunkLength` の門は落とせない — 記号 `M` の trace 範囲は資産に残らない（IR の
 * `symbols` は名前の列だけ）ので、宣言だけが「この資産が受けられる chunk 行数」の出どころで
 * ある。門が無かった頃、上限 768 の資産に `chunkLength: 1024` を渡すと例外なしで走っていた
 * （2026-09-03 実測）— 保証の外で動く形は fail loudly にする（横断不変条件）。
 *
 * NOTE: 容量との関係（`chunkLength ≤ capacity`）はここでは見ない — 容量は sequence ごとに選ぶので、
 * 両者が揃う唯一の場所が `createGenerationSequence` である（同じ式を 2 箇所に持たない）。
 *
 * NOTE: `export` は門を直接叩くテストのため（{@link assertRopeInputShapes} と同じ扱い — 実経路は
 * `fromAssets` / `fromPretrained` / `estimateSessionMemory` の 3 つで、どれも妥当値しか渡さない）。
 * `mod.ts` / サブパス面には出さない（ADR 0008）。
 */
export const assertChunkLength = (
  chunkLength: number,
  config: Gemma4PipelineConfig,
  // NOTE: 既定が通常 Gemma の名前なのは、この門を**直接叩く検査**が入口を持たないため。
  // 実経路（`buildGemma4Program` / `estimateSessionMemory`）は必ず family の名前を渡す。
  entry: string = gemmaEntryName("gemma4"),
): number => {
  if (!Number.isSafeInteger(chunkLength) || chunkLength < 2) {
    throw new ModelInputError(
      `${entry}: chunkLength ${chunkLength} が 2 以上の整数でない`,
    );
  }
  if (chunkLength > config.maxChunkLength) {
    throw new ModelInputError(
      `${entry}: chunkLength ${chunkLength} が配布形の宣言 maxChunkLength` +
        ` ${config.maxChunkLength} を超えた（記号 M を焼いた trace 範囲の外）`,
    );
  }
  if (chunkLength > config.maxPosition) {
    throw new ModelInputError(
      `${entry}: chunkLength ${chunkLength} が maxPosition ${config.maxPosition} を超えた`,
    );
  }
  return chunkLength;
};

/**
 * 実行時ノブの `chunkBuckets` を検査して返す（{@link Gemma4PipelineOptions.chunkBuckets} の門）。
 *
 * MUST: 受理集合の規則（2 以上 `chunkLength` 未満・狭義昇順）は**写さない** — 正本は runtime の
 * `assertChunkBuckets` 1 本で、そこが拒否する指定を context 生成まで通さないためにここで先に
 * 通す。この層が足すのは入口の名前だけで、`Gemma4PipelineOptions` に渡した呼び手が
 * 「自分のどの指定が落ちたか」を読めるようにする（`assertChunkLength` と同じ流儀）。
 *
 * NOTE: `maxChunkLength` の門は要らない（バケットは `chunkLength` 未満で、その `chunkLength`
 * 自体が {@link assertChunkLength} の門を通っている）。
 */
export const assertGemma4ChunkBuckets = (
  chunkBuckets: readonly number[],
  chunkLength: number,
  // NOTE: 既定の理由は {@link assertChunkLength} と同じ（門を直接叩く検査のため）。
  entry: string = gemmaEntryName("gemma4"),
): readonly number[] => {
  try {
    assertChunkBuckets(chunkBuckets, chunkLength);
  } catch (cause) {
    throw new ModelInputError(
      `${entry}: ${cause instanceof Error ? cause.message : String(cause)}`,
      { cause },
    );
  }
  return chunkBuckets;
};

/**
 * この manifest を gemma4 として実行できるかを見る（**GPU も重みの part も触る前**）。
 *
 * MUST: 未知 major は fail loudly（ADR 0038 §1 — 「古い実装 × 新しいリポ」の沈黙劣化を止める
 * 唯一の門）。`quant` の実在検査は取得の前に済ませる（`resolveSelection` も同じことを見るが、
 * こちらは利用可能な一覧を添えて落とす）。
 *
 * MUST: 選ばれた `Quant` を**捨てずに返す** — `requiredLimits` の DL 前検査
 * （ADR 0089 決定 5）は呼び手（{@link Gemma4Pipeline.fromPretrained} の admission 閉包）が
 * 通す。ここで名前の実在だけ見て中身を落とすと、宣言された GPU 前提を誰も読まないまま
 * 3.7GiB を落とす形へ戻る。
 */
export const gemma4ManifestConfig = (
  manifest: Manifest,
  selection: { readonly model?: string; readonly quant?: string },
  family: GemmaFamily,
): {
  readonly config: Gemma4PipelineConfig;
  readonly quantName: string;
  readonly quant: Quant;
} => {
  const where = gemmaEntryName(family);
  const modelName = selection.model ?? manifest.defaultModel;
  if (family === "gemma4-qat") assertGemma4QatModel(modelName);
  if (!Object.hasOwn(manifest.models, modelName)) {
    throw new Error(
      `${where}: model '${modelName}' は manifest に無い` +
        `（利用可能: ${manifest.available.models.join(" / ")}）`,
    );
  }
  const entry: ModelEntry = manifest.models[modelName];
  const { name, major } = entry.pipeline;
  if (name !== family) {
    throw new Error(
      `${where}: manifest の pipeline が '${name}/${major}'` +
        `（'${family}/${GEMMA4_PIPELINE_MAJOR}' が必要）`,
    );
  }
  if (major !== GEMMA4_PIPELINE_MAJOR) {
    throw new Error(
      `${where}: pipeline '${name}/${major}' の major に未対応` +
        `（この実装が読めるのは ${family}/${GEMMA4_PIPELINE_MAJOR}）`,
    );
  }
  const quantName = selection.quant ?? entry.defaultQuant;
  if (!Object.hasOwn(entry.quants, quantName)) {
    throw new Error(
      `${where}: quant '${quantName}' は manifest に無い` +
        `（利用可能: ${entry.available.quants.join(" / ")}）`,
    );
  }
  return {
    config: parseGemma4PipelineConfig(entry.pipelineConfig),
    quantName,
    quant: entry.quants[quantName],
  };
};
