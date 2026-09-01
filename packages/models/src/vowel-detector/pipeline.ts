/**
 * `VowelDetectorPipeline` — 16kHz モノラル音声 → リップシンク用の母音系列（`.lab`）の 1 本の面。
 *
 * パイプライン（全段 Karume・torch 不使用）:
 *
 * 1. ホスト: 波形 → 80 次元 log-mel + DSP 3 次元 = `[T10, 83]`（`features.ts`・mel 基底は
 *    配布形の `assets` 席から受け取る）
 * 2. ホスト: 奇数フレームの端数 1 本を落とす（出力は 20ms 格子）
 * 3. `crnn` を 1 回回す → ロジット `[1, T10/2, 8]`（20ms 格子）
 * 4. ホスト: log_softmax → Viterbi → 短区間マージ → cons 吸収 → `.lab`（`postprocess.ts`）
 *
 * **リサンプルは呼び出し側の責務**（上流 `@hdae/vowel-detector` の線をそのまま引く）。入口は
 * 「16kHz モノラルの `Float32Array`」で、WAV の decode も周波数変換も持たない（`decodeWav` は
 * barrel から別に出ている）。渡された配列が本当に 16kHz かは**観測できない** — 違う周波数を
 * 渡すと `detect` は最後まで走って別の母音列を出すので、そこは呼び出し側の不変条件になる。
 *
 * ## 長さは記号軸（グラフは 1 本・pad も長さ別の変種も無い）
 *
 * グラフ入力は `features [1, 2T, 83]`、出力は `[1, T, 8]` で、**時間軸だけが記号** `T`
 * （20ms 格子）。`detect` は音声の実長のまま 1 回回し、`T` を束縛して渡す。
 * `nn.GRU` を `gru_scan` へ差し替えて記号長を通す形は ADR 0056、`2T` のような派生次元から
 * シンボルを束縛する形は ADR 0057（それ以前の長さ別グラフ + 右ゼロ pad は同 ADR に経緯がある）。
 *
 * MUST: **10ms フレーム数は偶数**（グラフが `2T` を宣言している）。奇数フレームの端数 1 本は
 * **切り捨て**る — 半端フレームは入力 1 本分しか持たない出力になり、後処理の
 * 「1 フレーム = 20ms」が末尾だけ崩れる。切り捨てを忘れると `bindSymbols` が
 * 「実測 285 が宣言 '2T' の形をしていない」で落ちる（黙って丸めない）。
 *
 * MUST: **`maxFrames` 超過は fail loudly**（黙って切り詰めない）。切り詰めは音声の末尾を
 * 無言で捨てる = `.lab` が入力より短くなるだけで、呼び出し側からは正常な結果に見える。
 * 上限はグラフを焼いたときの記号次元の上限そのもので、配布形の `pipelineConfig` が持つ
 * （IR は記号の値域を持たないので、ここでしか止められない — `config.ts`）。
 *
 * ### 右ゼロ pad をしない理由（実測 — ADR 0057 以前の形が抱えていた負債）
 *
 * 逆方向 GRU が pad 側から状態を持ち帰るので、pad はロジットを O(1) 動かす。**pad 量に対して
 * 単調でも比例でもなく、2 フレーム（40ms）で既に飽和する**（実音声 4 本 × pad 10 段の実測:
 * 本体区間だけを比べた max abs diff が pad 2 で 2.2〜3.2、pad 1024 で 5.0〜6.0）。振幅は
 * 末尾に集中するが `|差| > 1e-2` は全フレームに届き、Viterbi が大域最適である以上、`.lab` の
 * 差は発話のどこにでも出た（実測の差は ①20ms の境界ずれ ②末尾に 40ms の `pau` が増減
 * ③発話中間に 40ms の `pau` が入る、の 3 型）。記号長にした今、この 3 型は**消えている**:
 * 配布形経由の `.lab` は実重み E2E（`packages/models/tests/e2e_vowel_detector_chain_test.ts`）が
 * 固定している実長経路の `.lab` と完全一致する（`tests/e2e_vowel_detector_lab_test.ts`）。
 *
 * ## Session は `detect` ごとに張って畳む
 *
 * グラフが 1 本になったので SigLIP2 / BiRefNet と同じ常駐も採れるが、この波では**張っては
 * 畳む**を保つ（VRAM を検出の間だけに閉じる形 — Irodori が巨大グラフを 1 本ずつ張っては畳む
 * のと同じ。`src/irodori/pipeline.ts` の `withSession`）。重みは 2.66MB で再アップロードの
 * 実測は 30〜144ms なので、常駐化は**性能の判断**であって正しさの判断ではない。
 *
 * MUST: それでも直列化鎖には載せる — `detect` の同時実行は `dispose` が in-flight の完了を
 * 待たずに GPU を破棄する形（flush-before-destroy 違反）を作る。
 *
 * ## MUST: 数値の正はここでは担保されない
 *
 * 正は実重み E2E（合成 golden 4 ケース × 4 系列のロジット突合 =
 * `packages/runtime/tests/e2e_vowel_detector_test.ts`・実音声 4 本の全鎖 `.lab` 完全一致 =
 * `packages/models/tests/e2e_vowel_detector_chain_test.ts`）と、ホスト層のパリティ門
 * （`packages/models/tests/vowel_detector_host_test.ts` — Python 正本との突合）が担保する。
 */

import {
  acquireGpu,
  type GpuContext,
  parseSafetensors,
  type Session,
  type SessionDiagnostics,
  type SessionOptions,
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
  parseVowelDetectorPipelineConfig,
  VOWEL_DETECTOR_PIPELINE_MAJOR,
  VOWEL_DETECTOR_PIPELINE_NAME,
  type VowelDetectorPipelineConfig,
} from "./config.ts";
import { extractFeatures, HOP, MEL_BINS, N_MELS, SAMPLE_RATE } from "./features.ts";
import { type LabSegment, logitsToSegments, toLab } from "./postprocess.ts";
import { createOperationChain } from "../concurrency/serial.ts";
import {
  assertGpuFeaturesGranted,
  assertRequiredLimitsBeforeDownload,
  assertRequiredLimitsSatisfied,
  toAcquireGpuOptions,
} from "../session/gpu-features.ts";
import { toSessionOptions } from "../session/options.ts";
import { toManifestSource } from "../hub/repo-ref.ts";
import {
  assetComponentOpener,
  type ComponentOpener,
  type GraphOwner,
  loadShardComponents,
  type ModelComponent,
} from "../hub/components.ts";

/** manifest の assets 表に現れる取得キーと、その safetensors のテンソル名（`dist.py` と対）。 */
const MEL_BASIS = "mel_basis";

/** グラフ入力の名前（`export_vowel_detector.py` の `INPUT_NAME`）。 */
const INPUT_NAME = "features";

/** 出力の時間軸の刻み（入力 2 フレーム = 出力 1 フレーム — conv の stride 2）。 */
const TIME_STRIDE = 2;

/** 入力フレームの毎秒本数（hop 160 / 16kHz = 10ms 格子）— 秒で語る文言のためだけに使う。 */
const FRAMES_PER_SECOND = SAMPLE_RATE / HOP;

/** manifest の weights 表に現れる取得キー（CRNN 1 本 — `dist.py` の `VOWEL_DETECTOR_GRAPH_ROLE`）。 */
const GRAPH_ROLE = "crnn";

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
   * 実行 1 回ごとの診断を受け取る観測席（1 detect = CRNN 1 回の run）。op 別 GPU
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

/**
 * {@link VowelDetectorPipeline.fromPretrained} だけが使う取得層のオプション（hub へ透過する）。
 *
 * NOTE: `headers` / `fetch` / `caches` は **HTTP 取得元専用**のノブで、取得元ハンドル
 * （`localDirectory` / `denoDirectory`）を渡した呼び出しでは 1 つも効かない — 手元の配布形は
 * network も CacheStorage も通らない。
 */
export type VowelDetectorFromPretrainedOptions = VowelDetectorPipelineOptions & {
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
export type VowelDetectorAssets = {
  readonly manifest: Manifest;
  readonly assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>;
};

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
 * 全量面（`fromAssets`）のコンポーネント供給口（受け口の実装は 7 家族共有 —
 * {@link assetComponentOpener}）。素の 1 本は `openModel` で開いて全量面で組み、shard 分割形
 * （`crnn[0]` / `crnn[1]` / …）は `fromPretrained` と同じ shard 逐次面へ流す。
 */
const assetOpener = (assets: VowelDetectorAssets["assets"]): ComponentOpener =>
  assetComponentOpener("vowel-detector", assets, (key) => assetBuffer(assets, key));

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
 * グラフが**記号長**で焼かれていることと、`pipelineConfig` と噛み合うことを見る。
 * 返すのは時間軸の記号名（`detect` が束縛に使う 1 語）。
 *
 * MUST: 落とさない。長さを固定して焼いた古い形のグラフは**入出力の名前も階数も同じ**なので、
 * 同じ席に置かれても構築は通り、実行時に「shape が宣言と一致しない」だけが出る（そのときには
 * 「この配布形は 1 長でしか動かない」ことが読み手に伝わらない）。
 * MUST: 入力の時間軸が `2·T`・出力が `T` であることまで見る。倍率が抜けた配布形は、`.lab` の
 * 時間が 2 倍に伸びるだけで形は成立する。
 *
 * NOTE: `export` は GPU 無しで拒否経路を縛るテストのため（`mod.ts` / サブパス面には出さない —
 * ADR 0008）。
 */
export const assertGraph = (
  model: GraphOwner,
  config: VowelDetectorPipelineConfig,
): string => {
  if (model.graph.inputs.length !== 1 || model.graph.outputs.length !== 1) {
    throw new Error(
      `VowelDetectorPipeline: グラフの入出力が ${model.graph.inputs.length} /` +
        ` ${model.graph.outputs.length} 本（どちらも 1 本の CRNN が必要）`,
    );
  }
  if (model.graph.symbols.length !== 1) {
    throw new Error(
      `VowelDetectorPipeline: グラフの記号次元が [${model.graph.symbols.join(", ")}]` +
        "（時間軸 1 本だけが記号）",
    );
  }
  const [symbol] = model.graph.symbols;
  const [input] = model.graph.inputs;
  if (input.name !== INPUT_NAME) {
    throw new Error(
      `VowelDetectorPipeline: グラフ入力が '${input.name}'（'${INPUT_NAME}' が必要）`,
    );
  }
  // 正準表記は `coeff·sym`（係数 1 は省略 — `format/dims.ts` の `formatDim`）。
  const expectedInput = [1, `${TIME_STRIDE}${symbol}`, config.featureDim];
  if (
    input.shape.length !== expectedInput.length ||
    input.shape.some((dim, axis) => dim !== expectedInput[axis])
  ) {
    throw new Error(
      `VowelDetectorPipeline: グラフ入力の形が ` +
        `[${input.shape.map(String).join(", ")}]、期待は [${expectedInput.join(", ")}]`,
    );
  }
  const name = model.graph.outputs[0];
  const value = model.graph.values[name];
  if (value === undefined) {
    throw new Error(`VowelDetectorPipeline: グラフ出力 '${name}' の宣言が無い`);
  }
  const expectedOutput = [1, symbol, config.classes.length];
  if (
    value.shape.length !== expectedOutput.length ||
    value.shape.some((dim, axis) => dim !== expectedOutput[axis])
  ) {
    throw new Error(
      `VowelDetectorPipeline: グラフ出力の形が ` +
        `[${value.shape.map(String).join(", ")}]、期待は [${expectedOutput.join(", ")}]`,
    );
  }
  return symbol;
};

/**
 * 入力長（10ms フレーム数）が配布形の宣言する運用上限に収まっていることを見る。
 *
 * MUST: 上限超過は落とす（モジュール doc の MUST — 黙って切り詰めない）。上限はグラフを
 * 焼いたときの記号次元の上限で、IR は値域を持たないので**ここが唯一の門**。
 *
 * NOTE: `export` はテストのため（境界の振る舞いを実 GPU 無しで名指しできるように）。
 */
export const assertFrameLimit = (
  config: VowelDetectorPipelineConfig,
  frames: number,
): void => {
  if (frames <= config.maxFrames) return;
  const seconds = (value: number): string => (value / FRAMES_PER_SECOND).toFixed(2);
  throw new Error(
    `VowelDetectorPipeline: 音声が長すぎる（10ms フレーム ${frames} 本 = ` +
      `${seconds(frames)} 秒）— この配布形が焼かれている上限は ` +
      `${config.maxFrames} フレーム（${seconds(config.maxFrames)} 秒）。` +
      "切り詰めると末尾が黙って落ちるので、呼び出し側で区切って渡す",
  );
};

/** {@link VowelDetectorPipeline} の内部状態（公開面には出さない）。 */
type VowelDetectorState = {
  readonly gpu: GpuContext;
  readonly ownsGpu: boolean;
  readonly config: VowelDetectorPipelineConfig;
  /** 開いた CRNN グラフ（Session はここには持たない — モジュール doc）。 */
  readonly graph: ModelComponent;
  /** 時間軸の記号名（`assertGraph` がグラフから読んだもの — 束縛のキー）。 */
  readonly symbol: string;
  readonly sessionOptions: SessionOptions;
  readonly melBasis: Float32Array<ArrayBuffer>;
  readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics) => void;
};

/** 家族 admission（{@link admitVowelDetector}）が確定させる材料。 */
type VowelDetectorAdmission = {
  readonly config: VowelDetectorPipelineConfig;
  readonly quantName: string;
  readonly quant: Quant;
  readonly graph: ModelComponent;
  /** 時間軸の記号名（`assertGraph` がグラフから読んだもの）。 */
  readonly symbol: string;
};

/**
 * この manifest とこのグラフを vowel-detector として実行できるかを見る（`hub/components.ts`
 * の家族 admission 席 — shard 面では**重み shard を 1 バイトも取る前**に呼ばれる）。
 *
 * MUST: 家族の門はこの 1 本に集める。後段へ散らすと、shard 面では GB 級の重みを落とした
 * **後**にしか落ちない（ADR 0070 決定 5 の文面より実装が狭くなる）。
 * MUST: manifest の契約違反と**グラフとの突合**は **GPU を取りに行く前**に落とす。順序が
 * ずれると、GPU の無い環境では別の例外に化けて「何が悪かったのか」が読み手に伝わらない。
 *
 * NOTE: 資産（`mel_basis`）の解析はこの席へ置けない — admission の時点では extras を
 * まだ取っていない（取ってからでは重み prefetch より前という位置が保てない）。
 */
const admitVowelDetector = (
  manifest: Manifest,
  open: ComponentOpener,
  options: VowelDetectorPipelineOptions,
): VowelDetectorAdmission => {
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

  const graph = open(GRAPH_ROLE);
  const symbol = assertGraph(graph, config);

  // MUST: 共有 GPU の能力不足（feature / device limit）はこの席で落とす — 自前で取る場合と
  // 違って `acquireGpu` を待つ理由が無く、重みを落とす前に判る唯一の家族門（要求と検査の
  // 写像は `session/gpu-features.ts` の 1 本で、後段の検査も同じ関数を呼ぶ）。
  if (options.gpu !== undefined) {
    assertGpuFeaturesGranted(
      quant.gpuFeatures,
      options.gpu,
      `VowelDetectorPipeline: quant '${quantName}'`,
    );
    assertRequiredLimitsSatisfied(
      quant.requiredLimits,
      options.gpu.limits,
      `VowelDetectorPipeline: quant '${quantName}'`,
    );
  }

  return { config, quantName, quant, graph, symbol };
};

/**
 * admission を通った材料 + 資産から実行状態を組む。
 *
 * MUST: 資産の解析は **GPU を取りに行く前**に落とす。順序がずれると、GPU の無い環境では
 * 別の例外に化けて「何が悪かったのか」が読み手に伝わらない。GPU 取得後に許される検査は
 * GPU の能力（shader-f16）だけ（ADR 0028）。
 */
const openVowelDetectorState = async (
  admitted: VowelDetectorAdmission,
  assets: VowelDetectorAssets["assets"],
  options: VowelDetectorPipelineOptions = {},
): Promise<VowelDetectorState> => {
  const { config, quant, quantName, graph, symbol } = admitted;
  const melBasis = parseMelBasis(assetBuffer(assets, MEL_BASIS));

  // MUST: 宣言された feature は device 作成時にしか要求できない（ADR 0028）。共有 GPU を
  // 渡された場合は要求できないので、能力が足りないことを名指しして落とす（共有 GPU は
  // {@link admitVowelDetector} が既に同じ 1 本で見ているが、自前で取った device は
  // ここが唯一の門）。
  const gpu = options.gpu ?? await acquireGpu(toAcquireGpuOptions(quant.gpuFeatures));
  const ownsGpu = options.gpu === undefined;
  try {
    assertGpuFeaturesGranted(quant.gpuFeatures, gpu, `VowelDetectorPipeline: quant '${quantName}'`);
  } catch (error) {
    // 内部で取った GPU は、構築に失敗したら誰も解放できなくなるのでここで返す。
    if (ownsGpu) gpu.destroy();
    throw error;
  }
  return {
    gpu,
    ownsGpu,
    config,
    graph,
    symbol,
    sessionOptions: toSessionOptions(quant.session),
    melBasis,
    ...(options.onRunDiagnostics === undefined
      ? {}
      : { onRunDiagnostics: options.onRunDiagnostics }),
  };
};

/**
 * 1 検出ぶんの Session を張り、使い終わったら必ず解放する。
 * MUST: `finally` で dispose する — 途中で落ちたときに VRAM が残ると、後続の検出が確保に
 * 失敗して「最初の失敗とは別の場所」で落ちる。
 */
const runGraph = async (
  state: VowelDetectorState,
  features: Tensor,
  rows: number,
): Promise<Float32Array> => {
  const model = state.graph;
  const session: Session = await model.createSession(state.gpu, state.sessionOptions);
  try {
    // 記号 `T` は 20ms 格子の本数。入力軸は `2T` の派生次元なので実 shape からも解けるが
    // （ADR 0057）、ホストが数えた本数を明示で渡して食い違いを実行前に落とす。
    const outputs = await session.run({ [INPUT_NAME]: features }, { [state.symbol]: rows });
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

/** 音声 1 本を通す（特徴 → 端数の切り捨て → 上限検査 → 実長で run → 後処理）。 */
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
  assertFrameLimit(config, usable);
  const rows = usable / TIME_STRIDE;

  const logits = await runGraph(state, {
    dtype: "f32",
    shape: [1, usable, config.featureDim],
    // 端数を落とした実長ぶんだけを渡す（pad は 1 要素も無い — モジュール doc）。
    data: features.data.slice(0, usable * config.featureDim),
  }, rows);
  const expected = rows * config.classes.length;
  if (logits.length !== expected) {
    throw new Error(
      `vowel-detector: ロジットの要素数 ${logits.length} が ` +
        `${rows}×${config.classes.length} と違う`,
    );
  }
  const segments = logitsToSegments(logits, rows);
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
   * 配布形から取得して組む（`loadManifest` → `resolveFiles` → **各コンポーネントの
   * グラフ shard だけ**を取って `prepareModel` → 残り資産の `fetchAssets` → 構築）。重み shard は
   * Session を組むときに 1 本ずつ流れる（ADR 0070 — `src/hub/components.ts`）。文字列の
   * `ref` は `{ repo }` と読む（= `main` 追従）。**`ref` は必須**（取得元に既定は無い —
   * `src/hub/repo-ref.ts` の MUST。このファミリは公開配布リポを持たないので pin 定数も無い）。
   *
   * 手元の配布形は**取得元ハンドル**で渡す（`localDirectory` / `@karume/hub/deno` の
   * `denoDirectory`）。HF の `owner/name` の綴りの門は通らず、network も CacheStorage も
   * 通らない（{@link VowelDetectorFromPretrainedOptions} の HTTP 専用ノブは効かない）。
   */
  static async fromPretrained(
    ref: string | HubRepoRef | DistributionSource,
    options: VowelDetectorFromPretrainedOptions = {},
  ): Promise<VowelDetectorPipeline> {
    const source = toManifestSource(ref, "VowelDetectorPipeline.fromPretrained");
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
    const buildOptions: VowelDetectorPipelineOptions = {
      ...(options.gpu === undefined ? {} : { gpu: options.gpu }),
      ...selection,
      ...(options.onRunDiagnostics === undefined
        ? {}
        : { onRunDiagnostics: options.onRunDiagnostics }),
    };
    // 家族の門は admission 席で通す（重み shard を取る前 — `hub/components.ts`）。
    const { admitted, assets } = await loadShardComponents(
      "VowelDetectorPipeline.fromPretrained",
      loaded,
      files,
      [GRAPH_ROLE],
      async (open) => {
        const admitted = admitVowelDetector(loaded.manifest, open, buildOptions);
        // 配布形が宣言した `requiredLimits` は**重み shard を取る前**にここで見る
        // （ADR 0089 決定 5 — 共有 GPU ならその limits、自前で取る経路はアダプタ実測値）。
        await assertRequiredLimitsBeforeDownload(
          admitted.quant.requiredLimits,
          buildOptions.gpu,
          `VowelDetectorPipeline: quant '${admitted.quantName}'`,
        );
        return admitted;
      },
      {
        ...hubOptions,
        ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
      },
    );
    return new VowelDetectorPipeline(
      await openVowelDetectorState(admitted, assets, buildOptions),
    );
  }

  /**
   * 取得済みの manifest + 資産から組む。契約検査・資産の解釈・`openModel`・グラフとの突合を
   * 全てここで済ませる（Session はまだ張らない — モジュール doc の MUST）。
   *
   * 取得キーは `resolveFiles` の規約どおり **2 形とも受ける** — 素の 1 本（`crnn`）と、
   * 1GiB 超のコンポーネントの shard 分割形（`crnn[0]` / `crnn[1]` / …）。分割形は
   * バイト列を連結せず `fromPretrained` と同じ shard 逐次面へ流す。添字の欠番と素キーとの混在は
   * fail loudly（受け口の実装は `src/hub/components.ts` の 1 本）。
   */
  static async fromAssets(
    input: VowelDetectorAssets,
    options: VowelDetectorPipelineOptions = {},
  ): Promise<VowelDetectorPipeline> {
    const admitted = admitVowelDetector(input.manifest, assetOpener(input.assets), options);
    return new VowelDetectorPipeline(
      await openVowelDetectorState(admitted, input.assets, options),
    );
  }

  /**
   * 音声 1 本の母音系列を出す。返るのはセグメント列と `.lab` 本文の両方（同じ区間割りの
   * 2 つの表現 — 上流 `@hdae/vowel-detector` の `detect` と同じ）。
   *
   * 入力は **16kHz モノラル**の `Float32Array`（`[-1, 1]` 尺度）。リサンプルは呼び出し側の
   * 責務で、周波数が違っても**この関数は落ちない**（モジュール doc）。
   *
   * 長さは実長のまま記号次元 `T` に束縛される（pad は 1 要素も入らない）。10ms フレーム数が
   * 奇数なら端数 1 本を**切り捨てる**。`maxFrames` 超過は fail loudly で落ちる（同）。
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
