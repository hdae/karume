/**
 * `VowelDetectorPipeline` — 16kHz モノラル音声 → リップシンク用の母音系列（`.lab`）の 1 本の面。
 *
 * パイプライン（全段 Karume・torch 不使用）:
 *
 * 1. ホスト: 波形 → 80 次元 log-mel + DSP 3 次元 = `[T10, 83]`（`features.ts`・mel 基底は
 *    配布形の `assets` 席から受け取る）
 * 2. ホスト: 長さバケットの選択と**右ゼロ pad**（下の「長さバケット」）
 * 3. `crnn_t<長さ>` を 1 回回す → ロジット `[1, 長さ/2, 8]`（20ms 格子）
 * 4. ホスト: pad ぶんを捨てて log_softmax → Viterbi → 短区間マージ → cons 吸収 → `.lab`
 *    （`postprocess.ts`）
 *
 * **リサンプルは呼び出し側の責務**（上流 `@hdae/vowel-detector` の線をそのまま引く）。入口は
 * 「16kHz モノラルの `Float32Array`」で、WAV の decode も周波数変換も持たない（`decodeWav` は
 * barrel から別に出ている）。渡された配列が本当に 16kHz かは**観測できない** — 違う周波数を
 * 渡すと `detect` は最後まで走って別の母音列を出すので、そこは呼び出し側の不変条件になる。
 *
 * ## 長さバケット（グラフは長さごとに別物）
 *
 * `aten.gru.input` は `run_decompositions` が**時間方向へ完全展開**するので、T は動的軸に
 * できない（`Dim("T")` は `Specializations unexpectedly required (T)` で落ちる —
 * `export_vowel_detector.py` の docstring）。したがって配布形は**長さバケットごとに 1 本の
 * グラフ**を持ち、`detect` は入力長以上の最小のバケットを選んで右ゼロ pad する。
 *
 * MUST: **バケット上限を超える入力は fail loudly**（黙って切り詰めない）。切り詰めは音声の
 * 末尾を無言で捨てる = `.lab` が入力より短くなるだけで、呼び出し側からは正常な結果に見える。
 *
 * MUST: pad は**特徴空間のゼロ**（波形へ無音を継ぎ足して特徴を採り直さない）。発話内 z 化の
 * 統計が pad 側へ引かれて**先頭まで**壊れる（実測: 2.9 秒の発話に 5.1 秒の無音を足すと先頭の
 * 誤差が 0.024 → 3.31・argmax 一致率 0.796）。最終フレームの複製も同様に悪い（末尾の
 * max abs diff が zero の 5.8 に対し 10.5〜11.5）。
 *
 * ### pad が数値に与える影響（実測 — 承知した上でこの形を採っている）
 *
 * 逆方向 GRU が pad 側から状態を持ち帰るので、pad はロジットを O(1) 動かす。**pad 量に対して
 * 単調でも比例でもなく、2 フレーム（40ms）で既に飽和する**（実音声 4 本 × pad 10 段の実測:
 * 本体区間だけを比べた max abs diff が pad 2 で 2.2〜3.2、pad 1024 で 5.0〜6.0）。振幅は
 * 末尾に集中するが `|差| > 1e-2` は全フレームに届き、Viterbi が大域最適である以上、`.lab` の
 * 差は発話のどこにでも出る（実測の差は ①20ms の境界ずれ ②末尾に 40ms の `pau` が増減
 * ③発話中間に 40ms の `pau` が入る、の 3 型）。
 *
 * つまり**バケットの刻みを細かくしても品質は改善しない**（配布サイズだけが線形に増える）。
 * 刻みは「必要な最大長を覆う最少本数」で決める、というのが配布形の判断で、正本は
 * `tools/exporter/karume/dist.py` の母音検出節。数値の厳密さが要る用途（オフラインで
 * exporter を持てる場合）は、その音声の長さちょうどでグラフを焼く形がある — 実重み E2E
 * （`packages/runtime/tests/e2e_vowel_detector_test.ts`）が実際にその形で回っている。
 *
 * ## MUST: Session は `detect` ごとに張って畳む（SigLIP2 / BiRefNet と非対称）
 *
 * あちらがグラフ 1 本を常駐させるのは、入力に依らず**同じグラフ**を回すから。こちらは入力長で
 * グラフが変わるので、常駐させるならバケットの本数だけ Session を並べるか、直前のバケットを
 * 覚えておく cache を持つことになる。重みは 1 バケット 4〜15MB（重み本体は 2.66MB で共通）で
 * 再アップロードの実測は 30〜144ms なので、**張っては畳む**を既定にする（Irodori が巨大
 * グラフを 1 本ずつ張っては畳むのと同じ形 — `src/irodori/pipeline.ts` の `withSession`）。
 *
 * MUST: それでも直列化鎖には載せる — `detect` の同時実行は `dispose` が in-flight の完了を
 * 待たずに GPU を破棄する形（flush-before-destroy 違反）を作る。
 *
 * ## MUST: 数値の正はここでは担保されない
 *
 * 正は実重み E2E（`packages/runtime/tests/e2e_vowel_detector_test.ts` — 実音声 4 本の全鎖
 * `.lab` 完全一致 + 合成 golden 4 ケース × 4 系列のロジット突合）と、ホスト層のパリティ門
 * （`packages/models/tests/vowel_detector_host_test.ts` — Python 正本との突合）が担保する。
 */

import {
  acquireGpu,
  createSession,
  type GpuContext,
  type KarumeModel,
  openModel,
  parseSafetensors,
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
  type SessionSpec,
} from "@karume/hub";

import {
  parseVowelDetectorPipelineConfig,
  VOWEL_DETECTOR_PIPELINE_MAJOR,
  VOWEL_DETECTOR_PIPELINE_NAME,
  type VowelDetectorPipelineConfig,
} from "./config.ts";
import { extractFeatures, HOP, MEL_BINS, N_MELS, SAMPLE_RATE } from "./features.ts";
import { type LabSegment, logitsToSegments, toLab } from "./postprocess.ts";
import { createOperationChain } from "../concurrency/serial.ts";

/** manifest の assets 表に現れる取得キーと、その safetensors のテンソル名（`dist.py` と対）。 */
const MEL_BASIS = "mel_basis";

/** グラフ入力の名前（`export_vowel_detector.py` の `INPUT_NAME`）。 */
const INPUT_NAME = "features";

/** 出力の時間軸の刻み（入力 2 フレーム = 出力 1 フレーム — conv の stride 2）。 */
const TIME_STRIDE = 2;

/** 入力フレームの毎秒本数（hop 160 / 16kHz = 10ms 格子）— 秒で語る文言のためだけに使う。 */
const FRAMES_PER_SECOND = SAMPLE_RATE / HOP;

/** manifest の weights 表に現れる取得キー（長さバケット 1 本ぶん — `dist.py` の綴りと対）。 */
const graphRole = (frameLength: number): string => `crnn_t${frameLength}`;

/** {@link VowelDetectorPipeline.detect} の結果（上流 `@hdae/vowel-detector` の `DetectResult`）。 */
export type VowelDetectorResult = {
  /** 秒単位のセグメント列（a/i/u/e/o/N/pau — cons は後処理で吸収済み）。 */
  readonly segments: readonly LabSegment[];
  /** `.lab` 形式の本文（`開始 終了 ラベル` 行・秒 7 桁・末尾改行付き）。 */
  readonly lab: string;
};

/** 構築オプション（{@link VowelDetectorPipeline.fromAssets} / `fromPretrained` 共通）。 */
export type VowelDetectorPipelineOptions = {
  /**
   * 既存の GPU を共有する。**渡した側が所有権を持つ**ので
   * {@link VowelDetectorPipeline.dispose} は破棄しない。省略時はパイプラインが内部で
   * `acquireGpu` し、`dispose()` で破棄する。
   */
  readonly gpu?: GpuContext;
  /** モデル（manifest の models のキー）。省略時は `defaultModel`。 */
  readonly model?: string;
  /** 実行構成（そのモデルの quants のキー）。省略時は `defaultQuant`。 */
  readonly quant?: string;
  /**
   * 実行 1 回ごとの診断を受け取る観測席（1 detect = 選ばれたバケット 1 本の run）。op 別 GPU
   * 時間（`lastRunTiming`）が要るときは `gpu` に `acquireGpu({ gpuTiming: true })` を渡す
   * （ADR 0021 — 既定は計測しない）。
   *
   * NOTE: SigLIP2 / BiRefNet と同じくコンポーネント名を渡さない — 回るグラフは常に CRNN 1 本
   * で、違うのは長さだけ（受け手が分岐できる軸ではない）。
   *
   * コールバックの例外は握らない（fail loudly — 検出ごと落ちる）。
   */
  readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics) => void;
};

/** {@link VowelDetectorPipeline.fromPretrained} だけが使う取得層のオプション（hub へ透過する）。 */
export type VowelDetectorFromPretrainedOptions = VowelDetectorPipelineOptions & {
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
export type VowelDetectorAssets = {
  readonly manifest: Manifest;
  readonly assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>;
};

/**
 * manifest の `session`（3 キー固定の manifest 所有語彙）を runtime の `SessionOptions` へ
 * **1 キーずつ**写す。
 *
 * MUST: スプレッドで丸投げしない（ADR 0038 §3 — 素通しにすると綴りが変わった瞬間に runtime が
 * 未知キーを黙って無視して沈黙劣化する。写像を明示すると型検査が落ちる）。
 */
const toSessionOptions = (spec: SessionSpec): SessionOptions => ({
  ...(spec.linearCompute === undefined ? {} : { linearCompute: spec.linearCompute }),
  ...(spec.attentionCompute === undefined ? {} : { attentionCompute: spec.attentionCompute }),
  ...(spec.attentionScoreStorage === undefined
    ? {}
    : { attentionScoreStorage: spec.attentionScoreStorage }),
});

/**
 * 取得済みバイト列を `openModel` / `parseSafetensors` へ渡せる ArrayBuffer にする。
 *
 * MUST: `slice` で写さない（hub は buffer 全体を占める view を返す契約なので、崩れていたら
 * **取得層の不変条件破れ**として落とす — 他ファミリと同じ形）。
 */
const assetBuffer = (
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  key: string,
): ArrayBuffer => {
  if (!Object.hasOwn(assets, key)) {
    throw new Error(
      `vowel-detector: 資産 '${key}' が無い（manifest の weights / assets に ${key} が要る）` +
        `（揃っているキー: ${Object.keys(assets).join(" / ")}）`,
    );
  }
  const bytes = assets[key];
  if (bytes.byteOffset !== 0 || bytes.byteLength !== bytes.buffer.byteLength) {
    throw new Error(
      `vowel-detector: 資産 '${key}' の bytes が buffer 全体を占めていない` +
        `（byteOffset ${bytes.byteOffset} / byteLength ${bytes.byteLength} /` +
        ` buffer ${bytes.buffer.byteLength}）`,
    );
  }
  return bytes.buffer;
};

/**
 * mel 基底の資産（1 テンソルの f32 safetensors `[80, 257]`）を読む。
 *
 * 資産は外部境界なので、テンソル名・dtype・形を全て検査してから使う。**行列が転置していても
 * 要素数は同じ**（80×257 と 257×80 は別だが、`n_mels` を取り違えた別の基底なら要素数だけ
 * 合いうる）ので、形を 2 軸とも見る。
 *
 * NOTE: `export` はテストのため（`fromAssets` 経由でここへ届くには manifest 一式が要る）。
 * `mod.ts` / サブパス面には出さない（ADR 0008）。
 */
export const parseMelBasis = (buffer: ArrayBuffer): Float32Array<ArrayBuffer> => {
  const file = parseSafetensors(buffer);
  const view = file.tensors.get(MEL_BASIS);
  if (view === undefined) {
    throw new Error(
      `vowel-detector: mel 基底の資産にテンソル '${MEL_BASIS}' が無い` +
        `（入っているもの: ${[...file.tensors.keys()].join(" / ")}）`,
    );
  }
  if (view.dtype !== "F32") {
    throw new Error(`vowel-detector: mel 基底が F32 でない（${view.dtype}）`);
  }
  if (view.shape.length !== 2 || view.shape[0] !== N_MELS || view.shape[1] !== MEL_BINS) {
    throw new Error(
      `vowel-detector: mel 基底の形が [${view.shape.join(", ")}]、期待は [${N_MELS}, ${MEL_BINS}]`,
    );
  }
  return new Float32Array(file.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength));
};

/**
 * バケット 1 本ぶんのグラフが、宣言どおりの静的な形で焼かれていることを見る。
 *
 * MUST: 落とさない。長さだけが違うグラフは**入出力の名前も階数も同じ**なので、別のバケットの
 * 資産が同じ席に置かれていても、パイプラインは pad する長さを間違えたまま Session の
 * shape 検査まで進む（そのときには「どちらの数が正しいのか」が読み手に伝わらない）。
 */
const assertGraph = (
  model: KarumeModel,
  config: VowelDetectorPipelineConfig,
  frameLength: number,
): void => {
  const where = `バケット t${frameLength}`;
  if (model.graph.inputs.length !== 1 || model.graph.outputs.length !== 1) {
    throw new Error(
      `VowelDetectorPipeline: ${where} のグラフの入出力が ${model.graph.inputs.length} /` +
        ` ${model.graph.outputs.length} 本（どちらも 1 本の CRNN が必要）`,
    );
  }
  const [input] = model.graph.inputs;
  if (input.name !== INPUT_NAME) {
    throw new Error(
      `VowelDetectorPipeline: ${where} のグラフ入力が '${input.name}'（'${INPUT_NAME}' が必要）`,
    );
  }
  const expectedInput = [1, frameLength, config.featureDim];
  if (
    input.shape.length !== expectedInput.length ||
    input.shape.some((dim, axis) => dim !== expectedInput[axis])
  ) {
    throw new Error(
      `VowelDetectorPipeline: ${where} のグラフ入力の形が ` +
        `[${input.shape.map(String).join(", ")}]、期待は [${expectedInput.join(", ")}]`,
    );
  }
  const name = model.graph.outputs[0];
  const value = model.graph.values[name];
  if (value === undefined) {
    throw new Error(`VowelDetectorPipeline: ${where} のグラフ出力 '${name}' の宣言が無い`);
  }
  const expectedOutput = [1, frameLength / TIME_STRIDE, config.classes.length];
  if (
    value.shape.length !== expectedOutput.length ||
    value.shape.some((dim, axis) => dim !== expectedOutput[axis])
  ) {
    throw new Error(
      `VowelDetectorPipeline: ${where} のグラフ出力の形が ` +
        `[${value.shape.map(String).join(", ")}]、期待は [${expectedOutput.join(", ")}]`,
    );
  }
};

/**
 * 入力長（10ms フレーム数）→ 回すバケット。
 *
 * MUST: 上限超過は落とす（モジュール doc の MUST — 黙って切り詰めない）。
 *
 * NOTE: `export` はテストのため（境界の振る舞いを実 GPU 無しで名指しできるように）。
 */
export const pickFrameLength = (
  config: VowelDetectorPipelineConfig,
  frames: number,
): number => {
  for (const length of config.frameLengths) {
    if (length >= frames) return length;
  }
  const max = config.frameLengths[config.frameLengths.length - 1];
  throw new Error(
    `VowelDetectorPipeline: 音声が長すぎる（10ms フレーム ${frames} 本 = ` +
      `${(frames / FRAMES_PER_SECOND).toFixed(2)} 秒）— この配布形が持つ最大のバケットは ` +
      `${max} フレーム（${(max / FRAMES_PER_SECOND).toFixed(2)} 秒）。` +
      "切り詰めると末尾が黙って落ちるので、呼び出し側で区切って渡す",
  );
};

/** {@link VowelDetectorPipeline} の内部状態（公開面には出さない）。 */
type VowelDetectorState = {
  readonly gpu: GpuContext;
  readonly ownsGpu: boolean;
  readonly config: VowelDetectorPipelineConfig;
  /** 長さバケット → 開いたグラフ（Session はここには持たない — モジュール doc の MUST）。 */
  readonly graphs: ReadonlyMap<number, KarumeModel>;
  readonly sessionOptions: SessionOptions;
  readonly melBasis: Float32Array<ArrayBuffer>;
  readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics) => void;
};

/**
 * manifest + 資産から実行状態を組む（{@link VowelDetectorPipeline.fromAssets} の中身）。
 *
 * MUST: manifest の契約違反と**資産の解析・グラフとの突合**は **GPU を取りに行く前**に落とす。
 * 順序がずれると、GPU の無い環境では別の例外に化けて「何が悪かったのか」が読み手に
 * 伝わらない。GPU 取得後に許される検査は GPU の能力（shader-f16）だけ（ADR 0028）。
 */
const openVowelDetectorState = async (
  input: VowelDetectorAssets,
  options: VowelDetectorPipelineOptions = {},
): Promise<VowelDetectorState> => {
  const { manifest, assets } = input;
  const modelName = options.model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, modelName)) {
    throw new Error(
      `VowelDetectorPipeline: model '${modelName}' は manifest に無い` +
        `（利用可能: ${manifest.available.models.join(" / ")}）`,
    );
  }
  const entry: ModelEntry = manifest.models[modelName];
  const { name, major } = entry.pipeline;
  if (name !== VOWEL_DETECTOR_PIPELINE_NAME) {
    throw new Error(
      `VowelDetectorPipeline: manifest の pipeline が '${name}/${major}'` +
        `（'${VOWEL_DETECTOR_PIPELINE_NAME}/${VOWEL_DETECTOR_PIPELINE_MAJOR}' が必要）`,
    );
  }
  if (major !== VOWEL_DETECTOR_PIPELINE_MAJOR) {
    // 「古い実装 × 新しいリポ」の沈黙劣化を止める唯一の門（ADR 0038 §6）。
    throw new Error(
      `VowelDetectorPipeline: pipeline '${name}/${major}' の major に未対応` +
        `（この実装が読めるのは ` +
        `${VOWEL_DETECTOR_PIPELINE_NAME}/${VOWEL_DETECTOR_PIPELINE_MAJOR}）`,
    );
  }
  const config = parseVowelDetectorPipelineConfig(entry.pipelineConfig);

  const quantName = options.quant ?? entry.defaultQuant;
  if (!Object.hasOwn(entry.quants, quantName)) {
    throw new Error(
      `VowelDetectorPipeline: quant '${quantName}' は manifest に無い` +
        `（利用可能: ${entry.available.quants.join(" / ")}）`,
    );
  }
  const quant = entry.quants[quantName];
  const wantsShaderF16 = quant.gpuFeatures?.shaderF16 === true;

  const graphs = new Map<number, KarumeModel>();
  for (const frameLength of config.frameLengths) {
    const graph = openModel(assetBuffer(assets, graphRole(frameLength)));
    assertGraph(graph, config, frameLength);
    graphs.set(frameLength, graph);
  }
  const melBasis = parseMelBasis(assetBuffer(assets, MEL_BASIS));

  // MUST: `shader-f16` は device 作成時にしか要求できない（ADR 0028）。共有 GPU を渡された
  // 場合は要求できないので、能力が足りないことを**ここで**名指しして落とす。
  const gpu = options.gpu ?? await acquireGpu(wantsShaderF16 ? { shaderF16: true } : {});
  const ownsGpu = options.gpu === undefined;
  if (wantsShaderF16 && !gpu.shaderF16Enabled) {
    // 内部で取った GPU は、構築に失敗したら誰も解放できなくなるのでここで返す。
    if (ownsGpu) gpu.destroy();
    throw new Error(
      `VowelDetectorPipeline: quant '${quantName}' は shader-f16 を要求するが、渡された` +
        " GpuContext で有効になっていない（acquireGpu({ shaderF16: true }) で取り直す）",
    );
  }
  return {
    gpu,
    ownsGpu,
    config,
    graphs,
    sessionOptions: toSessionOptions(quant.session),
    melBasis,
    ...(options.onRunDiagnostics === undefined
      ? {}
      : { onRunDiagnostics: options.onRunDiagnostics }),
  };
};

/**
 * 選ばれたバケットの Session を張り、使い終わったら必ず解放する。
 * MUST: `finally` で dispose する — 途中で落ちたときに VRAM が残ると、後続の検出が確保に
 * 失敗して「最初の失敗とは別の場所」で落ちる。
 */
const runGraph = async (
  state: VowelDetectorState,
  model: KarumeModel,
  features: Tensor,
): Promise<Float32Array> => {
  const session: Session = await createSession(state.gpu, model, state.sessionOptions);
  try {
    const outputs = await session.run({ [INPUT_NAME]: features });
    if (state.onRunDiagnostics !== undefined) state.onRunDiagnostics(session.diagnostics());
    const name = model.graph.outputs[0];
    const tensor = outputs[name];
    if (tensor === undefined) {
      throw new Error(`vowel-detector: グラフ出力 '${name}' が実行結果に無い`);
    }
    // 判別子で絞る（Float32Array へのキャストは dtype がずれたときに黙って通る）。
    if (tensor.dtype !== "f32") {
      throw new Error(`vowel-detector: ロジットが f32 でない（${tensor.dtype}）`);
    }
    return tensor.data;
  } finally {
    await session.dispose();
  }
};

/** 音声 1 本を通す（特徴 → バケット選択 → 右ゼロ pad → run → pad を捨てて後処理）。 */
const detectAudio = async (
  state: VowelDetectorState,
  audio: Float32Array,
): Promise<VowelDetectorResult> => {
  const { config } = state;
  const features = extractFeatures(audio, state.melBasis);
  // 出力は 20ms 格子なので、奇数フレームの端数 1 本は落とす（**切り捨てであって pad ではない**
  // — 半端フレームを 1 本足すと、その 1 本だけが入力 1 フレーム分しか持たない出力になる）。
  const usable = features.frames - (features.frames % TIME_STRIDE);
  if (usable < TIME_STRIDE) {
    throw new Error(
      `VowelDetectorPipeline: 音声が短すぎる（10ms フレーム ${features.frames} 本）— ` +
        `出力 1 本を作るのに ${TIME_STRIDE} 本が要る`,
    );
  }
  const frameLength = pickFrameLength(config, usable);
  const rows = usable / TIME_STRIDE;

  // 右ゼロ pad（モジュール doc の MUST — 特徴空間のゼロであって無音波形ではない）。
  const padded = new Float32Array(frameLength * config.featureDim);
  padded.set(features.data.subarray(0, usable * config.featureDim));
  const graph = state.graphs.get(frameLength);
  if (graph === undefined) {
    throw new Error(`VowelDetectorPipeline: バケット t${frameLength} のグラフが無い`);
  }
  const logits = await runGraph(state, graph, {
    dtype: "f32",
    shape: [1, frameLength, config.featureDim],
    data: padded,
  });
  const expected = (frameLength / TIME_STRIDE) * config.classes.length;
  if (logits.length !== expected) {
    throw new Error(
      `vowel-detector: ロジットの要素数 ${logits.length} が ` +
        `${frameLength / TIME_STRIDE}×${config.classes.length} と違う`,
    );
  }
  // pad ぶんを捨ててから後処理へ渡す（`.lab` が入力より長くなるのを止める唯一の席）。
  const segments = logitsToSegments(logits.subarray(0, rows * config.classes.length), rows);
  return { segments, lab: toLab(segments) };
};

/**
 * 母音検出のパイプライン（音声 → `.lab`）。
 *
 * 構築は {@link VowelDetectorPipeline.fromPretrained}（HF から取得）か
 * {@link VowelDetectorPipeline.fromAssets}（取得済みバイト列）だけを入口にする —
 * コンストラクタを private にしてあるのは、manifest 検査と資産の突合を迂回した半端な状態を
 * 作れないようにするため（ADR 0008）。
 */
export class VowelDetectorPipeline {
  readonly #state: VowelDetectorState;
  /** detect と dispose の直列化鎖（「1 本ずつ」を公開 API 側で守る — モジュール doc）。 */
  readonly #chain = createOperationChain();
  /**
   * dispose の 1 本。**undefined でないことが「dispose 済み」**（別に真偽値を持つと、独立に
   * 更新される派生状態になる）。
   */
  #disposal: Promise<void> | undefined;

  private constructor(state: VowelDetectorState) {
    this.#state = state;
  }

  /**
   * HF リポジトリから取得して組む（`loadManifest` → `resolveFiles` → `fetchAssets` →
   * {@link VowelDetectorPipeline.fromAssets} の糖衣）。文字列の `ref` は `{ repo }` と読む。
   */
  static async fromPretrained(
    ref: string | HubRepoRef,
    options: VowelDetectorFromPretrainedOptions = {},
  ): Promise<VowelDetectorPipeline> {
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
    return VowelDetectorPipeline.fromAssets({ manifest: loaded.manifest, assets }, {
      ...(options.gpu === undefined ? {} : { gpu: options.gpu }),
      ...selection,
      ...(options.onRunDiagnostics === undefined
        ? {}
        : { onRunDiagnostics: options.onRunDiagnostics }),
    });
  }

  /**
   * 取得済みの manifest + 資産から組む。契約検査・資産の解釈・`openModel`・グラフとの突合を
   * 全てここで済ませる（Session はまだ張らない — モジュール doc の MUST）。
   */
  static async fromAssets(
    input: VowelDetectorAssets,
    options: VowelDetectorPipelineOptions = {},
  ): Promise<VowelDetectorPipeline> {
    return new VowelDetectorPipeline(await openVowelDetectorState(input, options));
  }

  /**
   * 音声 1 本の母音系列を出す。返るのはセグメント列と `.lab` 本文の両方（同じ区間割りの
   * 2 つの表現 — 上流 `@hdae/vowel-detector` の `detect` と同じ）。
   *
   * 入力は **16kHz モノラル**の `Float32Array`（`[-1, 1]` 尺度）。リサンプルは呼び出し側の
   * 責務で、周波数が違っても**この関数は落ちない**（モジュール doc）。
   *
   * 長さは配布形のバケットへ右ゼロ pad で丸められ、上限を超える入力は落ちる（同）。
   *
   * 並行に呼ばれた場合は**待たされて順に**走る（GPU の破棄と in-flight の run を交差させない）。
   */
  async detect(audio: Float32Array): Promise<VowelDetectorResult> {
    // dispose 済みの判定は呼び出し時点で行う（鎖の中で見ると、dispose より前に受けた検出まで
    // 巻き添えで落ちる）。
    if (this.#disposal !== undefined) {
      throw new Error("VowelDetectorPipeline: dispose 済みでは検出できない");
    }
    return await this.#chain(() => detectAudio(this.#state, audio));
  }

  /** 入力波形に要求するサンプリング周波数（配布形の宣言 — 呼び出し側のリサンプル先）。 */
  get sampleRate(): typeof SAMPLE_RATE {
    return this.#state.config.sampleRate;
  }

  /**
   * 解放する。**内部で取得した GPU だけ**破棄する（`options.gpu` で渡された GpuContext は
   * 呼び出し側の所有物なので触らない）。Session は `detect` ごとに畳んでいるので、ここで
   * 畳む相手は無い（モジュール doc の MUST）。
   *
   * MUST: in-flight の検出の完了を待ってから破棄する（flush-before-destroy）— 破棄も鎖に
   * 載せることで、待ちと破棄の順序を 1 箇所で決める。2 度目以降も同じ完了を返す（先に返すと
   * 呼び出し側が「破棄済み」と見なして次へ進む）。
   */
  dispose(): Promise<void> {
    this.#disposal ??= this.#chain(() => {
      if (this.#state.ownsGpu) this.#state.gpu.destroy();
      return Promise.resolve();
    });
    return this.#disposal;
  }

  /** `await using` 対応（Explicit Resource Management）— {@link dispose} の別名。 */
  [Symbol.asyncDispose](): Promise<void> {
    return this.dispose();
  }
}
