// `VowelDetectorPipeline` の**構築ガード**と、運用上限 / mel 基底資産の結線。GPU も実資産も
// 要らない範囲だけを見る（実 GPU の突合は `packages/runtime/tests/e2e_vowel_detector_test.ts`
//〈合成 golden のロジット〉と `e2e_vowel_detector_chain_test.ts`〈実音声の全鎖 → `.lab`〉が
// 持つ — 重複させない。ホスト層の Python 正本とのパリティは
// `packages/models/tests/vowel_detector_host_test.ts`）。
//
// 押さえるのは 6 点:
//
// ① `fromAssets` は **manifest の契約違反を GPU を取りに行く前**に落とす
//    （`src/vowel-detector/pipeline.ts` の `openVowelDetectorState` が掲げる MUST）。観測の
//    仕掛けは SigLIP2 / BiRefNet と同じ — **全ケースで容器は揃えて**おき、
//     - 契約違反ケースが「その違反の文言」で落ちる = 資産が揃っていても manifest の門が先
//     - 正しい manifest + 空の Record が `部品 'crnn' の容器が無い` で落ちる（受け口の診断）
//    の 2 つで門の順序そのものを縛る。
//    NOTE: 容器を開くのは admission の**前**（`assetComponentOpener` は同期の供給口を返すため
//    先に全部品を開く — ADR 0109 の継ぎ目）。
//
// ② グラフ宣言との突合（`assertGraph`）の**拒否経路**。`fromAssets` の中では実資産が
//    揃わないと踏めないので、門を直接叩く（`tests/helpers/stub-model.ts` が宣言だけの
//    `GraphOwner` を組む）。長さを固定して焼いた古い形は**入出力の名前も階数も同じ**なので、
//    門の綴りが `format/dims.ts` の正準表記からずれても正常系だけなら緑のまま通る。
//
// ③ `pipelineConfig` の**宣言 3 欄**（sampleRate / featureDim / classes）は受理集合が 1 値
//    きりで、外れた配布形はパース時に落ちる。とくに `classes` は**並びが id** なので、
//    置換された宣言が通ると `.lab` は成立したままラベルだけが入れ替わる。
//
// ④ **運用範囲**（`assertFrameFloor` / `assertFrameLimit`）— 下限未満も上限超過も
//    fail loudly。ここは「黙って切り詰めない・黙って pad しない」という配布形の約束そのもの
//    で、境界（ちょうどの長さ・1 フレーム外側）を両側とも名指しで踏む。範囲は配布形の宣言
//    なので、TS 側に定数を持たない（`config.ts` の MUST）。
//
// ⑤ **mel 基底資産**（`parseMelBasis`）— テンソル名・dtype・形の 3 つを見る。基底がずれても
//    特徴は「それらしい別の値」になるだけで、shape も値域も合ったまま最後まで通る。
//
// ⑥ **検出の実行経路の例外**（偽の Session）— run や診断コールバックの失敗の後に Session の
//    解放まで落ちても、呼び手の例外から最初の失敗を辿れる。

import { assert, assertEquals, assertRejects, assertStrictEquals, assertThrows } from "@std/assert";
import { parseManifest } from "@karume/hub";
import { ModelInputError } from "../src/errors.ts";
import {
  parseVowelDetectorPipelineConfig,
  type VowelDetectorPipelineConfig,
} from "../src/vowel-detector/config.ts";
import {
  assertFrameFloor,
  assertFrameLimit,
  assertGraph,
  parseMelBasis,
  VowelDetectorPipeline,
} from "../src/vowel-detector/pipeline.ts";
import {
  extractFeatures,
  FEATURE_DIM,
  featureFrameCount,
  HOP,
  MEL_BINS,
  N_FFT,
  N_MELS,
  SAMPLE_RATE,
} from "../src/vowel-detector/features.ts";
import { LIPSYNC_CLASSES } from "../src/vowel-detector/postprocess.ts";
import { writeSafetensors } from "./helpers/safetensors-write.ts";
import { type StubDim, stubModel } from "./helpers/stub-model.ts";
import { declaredContainer, partAssets, tensorlessContainer } from "./helpers/container-fixture.ts";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";

/**
 * 配布形が宣言する運用範囲
 * （`tools/export-recipes/vowel_detector/distribution.py` の `VOWEL_DETECTOR_MIN_FRAMES` /
 * `VOWEL_DETECTOR_MAX_FRAMES` と同じ数 — 記号 `T` の `Dim(min, max)` を 10ms 側の単位へ
 * 直したもの）。
 */
const MIN_FRAMES = 4;
const MAX_FRAMES = 60_000;

const fileRef = (path: string) => ({ path, size: 16, sha256: "a".repeat(64) });

/** 宣言だけの `crnn` 容器（家族の門が読む形まで再現し、実行はしない）。 */
const COMPONENT = partAssets(
  "crnn",
  await tensorlessContainer("crnn", {
    symbols: ["T"],
    inputs: [{ name: "features", shape: [1, "2T", FEATURE_DIM] }],
    output: { name: "logits", shape: [1, "T", LIPSYNC_CLASSES.length] },
  }),
);

/** `models/karume-vowel-detector/karume.json` の `pipelineConfig` 実物（5 欄）。 */
const PIPELINE_CONFIG: Record<string, unknown> = {
  sampleRate: SAMPLE_RATE,
  featureDim: FEATURE_DIM,
  classes: [...LIPSYNC_CLASSES],
  minFrames: MIN_FRAMES,
  maxFrames: MAX_FRAMES,
};

/** 配布形の骨格（検査に要る欄だけ）。`patch` は `models["crnn-epoch3"]` の中身を上書きする。 */
const manifestText = (patch: Record<string, unknown> = {}): string =>
  JSON.stringify({
    format: "karume/5",
    generator: "karume/0.2.2",
    defaultModel: "crnn-epoch3",
    models: {
      "crnn-epoch3": {
        pipeline: "vowel-detector/1",
        weights: {
          crnn: { f32: declaredContainer("crnn-epoch3/model.f32") },
        },
        assets: { mel_basis: fileRef("crnn-epoch3/features/mel-basis.safetensors") },
        quants: { f32: { weights: { crnn: "f32" }, session: {} } },
        defaultQuant: "f32",
        pipelineConfig: PIPELINE_CONFIG,
        ...patch,
      },
    },
  });

const emptyAssets = {} as Record<string, Uint8Array<ArrayBuffer>>;

Deno.test("fromAssets: 存在しない model は利用可能な一覧を添えて落とす", async () => {
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => VowelDetectorPipeline.fromAssets({ manifest, assets: COMPONENT }, { model: "nope" }),
    Error,
    "model 'nope' は manifest に無い",
  );
});

Deno.test("fromAssets: pipeline の契約名が vowel-detector でない manifest を落とす", async () => {
  const manifest = parseManifest(manifestText({ pipeline: "sbv2/1" }));
  await assertRejects(
    () => VowelDetectorPipeline.fromAssets({ manifest, assets: COMPONENT }),
    Error,
    "manifest の pipeline が 'sbv2/1'",
  );
});

Deno.test("fromAssets: 未知 major は fail loudly（検査責務は models 側 — ADR 0038 §1）", async () => {
  // 「古い実装 × 新しいリポ」の沈黙劣化を止める唯一の門。hub は major を検査しない。
  const manifest = parseManifest(manifestText({ pipeline: "vowel-detector/2" }));
  await assertRejects(
    () => VowelDetectorPipeline.fromAssets({ manifest, assets: COMPONENT }),
    Error,
    "major に未対応",
  );
});

Deno.test("fromAssets: 存在しない quant は利用可能な一覧を添えて落とす", async () => {
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => VowelDetectorPipeline.fromAssets({ manifest, assets: COMPONENT }, { quant: "nope" }),
    Error,
    "quant 'nope' は manifest に無い",
  );
});

Deno.test("fromAssets: pipelineConfig の未知キーは構築時に落ちる", async () => {
  // 綴り違い（`maxFrames` に対する `max_frames`）が黙って既定へ縮退する経路を作らない。
  const manifest = parseManifest(
    manifestText({ pipelineConfig: { ...PIPELINE_CONFIG, max_frames: 200 } }),
  );
  await assertRejects(
    () => VowelDetectorPipeline.fromAssets({ manifest, assets: COMPONENT }),
    Error,
    "pipelineConfig: 未知キー 'max_frames'",
  );
});

Deno.test("fromAssets: 部品の容器が無ければ 2 形の綴りつきで落ちる（受け口の診断）", async () => {
  // 上の 5 ケースの裏返し。容器が揃っていない Record では**部品の不在**で落ちる（manifest の
  // 文言では落ちない）= 上のケースが資産の不在に巻き添えられていないことの対偶。
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => VowelDetectorPipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "部品 'crnn' の容器が無い",
  );
});

Deno.test("fromAssets: 容器が揃っていれば資産（mel_basis）の不在まで進む", async () => {
  // 家族の門を全部通った先に残るのは manifest の `assets`（全量面）— ここで初めて
  // `mel_basis` の不在が見える（門の順序が資産の解析より前であることの対偶）。
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => VowelDetectorPipeline.fromAssets({ manifest, assets: COMPONENT }),
    Error,
    "資産 'mel_basis' が無い",
  );
});

// ---- グラフ宣言との突合（拒否経路）------------------------------------------

/** 実配布形と同じ pipelineConfig（この節は宣言の突合だけを見る）。 */
const graphConfig = parseVowelDetectorPipelineConfig(PIPELINE_CONFIG);

/** 記号長 1 グラフ（ADR 0056 / 0057）の宣言。`patch` で 1 点だけ壊す。 */
const crnnGraph = (
  patch: {
    readonly symbols?: readonly string[];
    readonly inputName?: string;
    readonly inputShape?: readonly StubDim[];
    readonly outputShape?: readonly StubDim[];
  } = {},
) =>
  stubModel({
    symbols: patch.symbols ?? ["T"],
    inputs: [{
      name: patch.inputName ?? "features",
      shape: patch.inputShape ?? [1, "2T", FEATURE_DIM],
    }],
    outputs: ["logits"],
    values: { logits: patch.outputShape ?? [1, "T", LIPSYNC_CLASSES.length] },
  });

Deno.test("assertGraph: 記号長で焼かれたグラフは時間軸の記号名を返す", () => {
  assertEquals(assertGraph(crnnGraph(), graphConfig), "T");
});

Deno.test("assertGraph: 長さを固定して焼いた古い形は記号次元で落ちる", () => {
  // 入出力の名前も階数も同じなので、同じ席に置かれても構築は通ってしまう形。
  assertThrows(
    () =>
      assertGraph(
        crnnGraph({
          symbols: [],
          inputShape: [1, 2000, FEATURE_DIM],
          outputShape: [1, 1000, LIPSYNC_CLASSES.length],
        }),
        graphConfig,
      ),
    Error,
    "VowelDetectorPipeline: グラフの記号次元が []",
  );
});

Deno.test("assertGraph: 入力の時間軸から 2 倍の係数が抜けたら落とす", () => {
  // 係数が抜けた配布形は `.lab` の時間が 2 倍に伸びるだけで、形は成立する。
  assertThrows(
    () => assertGraph(crnnGraph({ inputShape: [1, "T", FEATURE_DIM] }), graphConfig),
    Error,
    "VowelDetectorPipeline: グラフ入力の形が [1, T, 83]、期待は [1, 2T, 83]",
  );
});

Deno.test("assertGraph: 入力名が features でなければ落とす", () => {
  assertThrows(
    () => assertGraph(crnnGraph({ inputName: "input" }), graphConfig),
    Error,
    "VowelDetectorPipeline: グラフ入力が 'input'",
  );
});

Deno.test("assertGraph: 出力のクラス数が宣言と違えば落とす", () => {
  // クラス数がずれると `.lab` は成立したままラベルの割り当てだけが崩れる。
  assertThrows(
    () => assertGraph(crnnGraph({ outputShape: [1, "T", 5] }), graphConfig),
    Error,
    "VowelDetectorPipeline: グラフ出力の形が [1, T, 5]",
  );
});

// ---- pipelineConfig のスキーマ（宣言 3 欄 + 運用範囲 2 欄）--------------------

const config = (patch: Record<string, unknown> = {}): VowelDetectorPipelineConfig =>
  parseVowelDetectorPipelineConfig({ ...PIPELINE_CONFIG, ...patch });

Deno.test("pipelineConfig: 実物の 5 欄をそのまま読める", () => {
  const parsed = config();
  assertEquals(parsed.sampleRate, SAMPLE_RATE);
  assertEquals(parsed.featureDim, FEATURE_DIM);
  assertEquals([...parsed.classes], [...LIPSYNC_CLASSES]);
  assertEquals(parsed.minFrames, MIN_FRAMES);
  assertEquals(parsed.maxFrames, MAX_FRAMES);
});

Deno.test("pipelineConfig: 16kHz 以外の配布形は受理しない（リサンプラを持たない）", () => {
  assertThrows(
    () => config({ sampleRate: 22050 }),
    Error,
    "pipelineConfig.sampleRate: この実装が対応するのは 16000 だけ",
  );
});

Deno.test("pipelineConfig: 特徴次元が 83 でない配布形は受理しない", () => {
  assertThrows(
    () => config({ featureDim: 80 }),
    Error,
    "pipelineConfig.featureDim: この実装が対応するのは 83 だけ",
  );
});

Deno.test("pipelineConfig: クラスの**並び**が違う配布形は受理しない（並びが id）", () => {
  // 集合としては同じで並びだけが違う = ラベルが置換されるだけで .lab は完全に成立する形。
  const swapped = [...LIPSYNC_CLASSES];
  [swapped[0], swapped[1]] = [swapped[1], swapped[0]];
  assertThrows(
    () => config({ classes: swapped }),
    Error,
    "pipelineConfig.classes: この実装が対応するのは",
  );
});

Deno.test("pipelineConfig: 奇数・非整数・非正の上限宣言は落とす", () => {
  // グラフ入力は `2T` なので、奇数の上限は「その 1 本だけ通らない上限」= 意味が壊れている。
  for (const value of [251, 0, -2, 2.5, "2000"]) {
    assertThrows(
      () => config({ maxFrames: value }),
      Error,
      "pipelineConfig.maxFrames: 正の 2 の倍数でない",
      `maxFrames=${JSON.stringify(value)}`,
    );
  }
});

Deno.test("pipelineConfig: 上限の欠落は落とす（既定へ縮退しない）", () => {
  const { maxFrames: _dropped, ...rest } = PIPELINE_CONFIG;
  assertThrows(
    () => parseVowelDetectorPipelineConfig(rest),
    Error,
    "pipelineConfig.maxFrames: 無い",
  );
});

Deno.test("pipelineConfig: 奇数・非整数・非正の下限宣言は落とす", () => {
  // 下限も 10ms フレーム数（グラフ入力は `2T`）なので、上限と同じ刻みの検査が掛かる。
  for (const value of [3, 0, -2, 2.5, "4"]) {
    assertThrows(
      () => config({ minFrames: value }),
      Error,
      "pipelineConfig.minFrames: 正の 2 の倍数でない",
      `minFrames=${JSON.stringify(value)}`,
    );
  }
});

Deno.test("pipelineConfig: 下限の欠落は落とす（既定へ縮退しない）", () => {
  const { minFrames: _dropped, ...rest } = PIPELINE_CONFIG;
  assertThrows(
    () => parseVowelDetectorPipelineConfig(rest),
    Error,
    "pipelineConfig.minFrames: 無い",
  );
});

Deno.test("pipelineConfig: 下限が上限を超える宣言は落とす（受理集合が空になる）", () => {
  assertThrows(
    () => config({ minFrames: MAX_FRAMES + 2 }),
    Error,
    "受理できる入力長が 1 本も無い",
  );
});

// ---- 運用範囲（入力長の門・両側）---------------------------------------------

Deno.test("運用上限: ちょうどの長さは通り、1 フレーム超過で落ちる（境界は上側に閉じる）", () => {
  const parsed = config();
  assertFrameLimit(parsed, 2);
  assertFrameLimit(parsed, MAX_FRAMES);
  // 長さは呼び手が渡した波形の寸法なので入力起因（ADR 0107 決定 2 — 区切って渡せば通る）。
  const error = assertThrows(
    () => assertFrameLimit(parsed, MAX_FRAMES + 2),
    ModelInputError,
    "音声が長すぎる",
  );
  // 何秒までなら通るのかが文言に出ていること（切り詰めの代わりに呼び出し側が区切るため）。
  assert(error.message.includes(`${MAX_FRAMES} フレーム`), error.message);
  assert(error.message.includes("600.00 秒"), error.message);
});

Deno.test("運用下限: ちょうどの長さは通り、1 フレーム不足で落ちる（境界は下側に閉じる）", () => {
  const parsed = config();
  // 焼き込み下限ちょうど（10ms フレーム 4 本 = 記号 T が 2）は通る。
  assertFrameFloor(parsed, MIN_FRAMES);
  assertFrameFloor(parsed, MAX_FRAMES);
  for (const frames of [MIN_FRAMES - 1, MIN_FRAMES - 2]) {
    const error = assertThrows(
      () => assertFrameFloor(parsed, frames),
      ModelInputError,
      "音声が短すぎる",
      `frames=${frames}`,
    );
    // 何本あれば通るのかが文言に出ていること（pad の代わりに呼び出し側が長く渡すため）。
    assert(error.message.includes(`${MIN_FRAMES} フレーム`), error.message);
    assert(error.message.includes(`${frames} 本`), error.message);
  }
});

Deno.test("運用下限: 下限は配布形の宣言から来る（TS 側の定数ではない）", () => {
  // 同じ入力長が、宣言の下限が違う配布形では通ったり落ちたりする = 数の出所が manifest 側。
  const strict = config({ minFrames: 8 });
  assertThrows(() => assertFrameFloor(strict, 6), ModelInputError, "8 フレーム");
  assertFrameFloor(config({ minFrames: 2 }), 2);
});

// ---- mel 基底の資産 ----------------------------------------------------------

const melBasisBytes = (
  name = "mel_basis",
  shape: readonly number[] = [N_MELS, MEL_BINS],
): Uint8Array<ArrayBuffer> =>
  writeSafetensors(
    new Map([[name, {
      dtype: "F32" as const,
      shape,
      data: new Float32Array(shape.reduce((product, dim) => product * dim, 1)),
    }]]),
    {},
  );

Deno.test("mel 基底: [80, 257] の f32 を 1 テンソルだけ読む", () => {
  const basis = parseMelBasis(melBasisBytes().buffer);
  assertEquals(basis.length, N_MELS * MEL_BINS);
});

Deno.test("mel 基底: テンソル名が違えば入っているものを添えて落とす", () => {
  assertThrows(
    () => parseMelBasis(melBasisBytes("mel").buffer),
    Error,
    "テンソル 'mel_basis' が無い",
  );
});

Deno.test("mel 基底: 転置は要素数が同じでも落とす（形を 2 軸とも見る）", () => {
  assertThrows(
    () => parseMelBasis(melBasisBytes("mel_basis", [MEL_BINS, N_MELS]).buffer),
    Error,
    `期待は [${N_MELS}, ${MEL_BINS}]`,
  );
});

Deno.test("mel 基底: f32 でない格納形は落とす", () => {
  const bytes = writeSafetensors(
    new Map([["mel_basis", {
      dtype: "I32" as const,
      shape: [N_MELS, MEL_BINS],
      data: new Int32Array(N_MELS * MEL_BINS),
    }]]),
    {},
  );
  assertThrows(() => parseMelBasis(bytes.buffer), Error, "mel 基底が F32 でない");
});

// ---- 検出の実行経路（Session を偽物にして例外の運び方を見る）------------------
//
// `detect` は 1 検出ごとに Session を張って畳む。run や診断コールバックが落ちた後に Session の
// 解放まで落ちる（device lost では両方が起きる）とき、呼び手が受け取る例外から**最初の失敗**を
// 辿れることを縛る。Session も GPU も偽物で足りる（`createSession` までしか触らない）ので、
// 状態を直接組んで非公開のコンストラクタへ渡す（`wan_pipeline_test.ts` と同じ流儀）。

/** 偽の Session が投げるもの・呼び手が渡す診断コールバック。 */
type FakeRunPlan = {
  readonly runError?: Error;
  readonly disposeError?: Error;
  readonly onRunDiagnostics?: () => void;
  /** 既定は運用範囲 4..60,000 の配布形。 */
  readonly config?: VowelDetectorPipelineConfig;
  /** 既定は正しい寸法のゼロ基底。 */
  readonly melBasis?: Float32Array<ArrayBuffer>;
};

/** 1 秒の正弦波（10ms フレーム 97 本 → 端数を落として 96 本 = 運用範囲の内側）。 */
const ONE_SECOND = Float32Array.from(
  { length: SAMPLE_RATE },
  (_, index) => 0.5 * Math.sin((2 * Math.PI * 220 * index) / SAMPLE_RATE),
);

/** 偽の Session を張る CRNN で組んだパイプラインと、解放の呼び出し回数。 */
const pipelineWithFakeSession = (
  plan: FakeRunPlan,
): { readonly pipeline: VowelDetectorPipeline; readonly disposeCalls: () => number } => {
  let disposeCalls = 0;
  const state = {
    gpu: fakeGpuContext(fakeDevice()),
    ownsGpu: false,
    config: plan.config ?? graphConfig,
    graph: {
      graph: { outputs: ["logits"] },
      createSession: () =>
        Promise.resolve({
          run: () =>
            plan.runError === undefined
              ? Promise.resolve({
                logits: {
                  dtype: "f32",
                  shape: [1, 48, LIPSYNC_CLASSES.length],
                  data: new Float32Array(48 * LIPSYNC_CLASSES.length),
                },
              })
              : Promise.reject(plan.runError),
          diagnostics: () => ({}),
          dispose: () => {
            disposeCalls += 1;
            return plan.disposeError === undefined
              ? Promise.resolve()
              : Promise.reject(plan.disposeError);
          },
        }),
    },
    symbol: "T",
    sessionOptions: {},
    melBasis: plan.melBasis ?? new Float32Array(N_MELS * MEL_BINS),
    ...(plan.onRunDiagnostics === undefined ? {} : { onRunDiagnostics: plan.onRunDiagnostics }),
  };
  const pipeline: VowelDetectorPipeline = Reflect.construct(VowelDetectorPipeline, [state]);
  return { pipeline, disposeCalls: () => disposeCalls };
};

Deno.test("detect: run が落ちた後に Session の解放も落ちたら、run の例外を先頭に両方を運ぶ", async () => {
  const runFailure = new Error("run の途中で device lost");
  const disposeFailure = new Error("flush に失敗した");
  const { pipeline, disposeCalls } = pipelineWithFakeSession({
    runError: runFailure,
    disposeError: disposeFailure,
  });

  const error = await assertRejects(() => pipeline.detect(ONE_SECOND), AggregateError);

  // 解放の例外だけが届くと、最初に何が壊れたか（run の失敗）が消える。
  assertStrictEquals(error.errors[0], runFailure);
  assertStrictEquals(error.errors[1], disposeFailure);
  assertEquals(disposeCalls(), 1);
});

Deno.test("detect: 診断コールバックが落ちた後に Session の解放も落ちたら、コールバックの例外を先頭に運ぶ", async () => {
  const callbackFailure = new Error("診断コールバックが投げた");
  const disposeFailure = new Error("flush に失敗した");
  const { pipeline, disposeCalls } = pipelineWithFakeSession({
    disposeError: disposeFailure,
    onRunDiagnostics: () => {
      throw callbackFailure;
    },
  });

  const error = await assertRejects(() => pipeline.detect(ONE_SECOND), AggregateError);

  assertStrictEquals(error.errors[0], callbackFailure);
  assertStrictEquals(error.errors[1], disposeFailure);
  assertEquals(disposeCalls(), 1);
});

Deno.test("detect: run だけが落ちたら、その例外をそのまま運び Session は 1 回だけ解放する", async () => {
  const runFailure = new Error("run の途中で落ちた");
  const { pipeline, disposeCalls } = pipelineWithFakeSession({ runError: runFailure });

  const error = await assertRejects(() => pipeline.detect(ONE_SECOND));

  assertStrictEquals(error, runFailure);
  assertEquals(disposeCalls(), 1);
});

// ---- 運用上限は特徴を作る前に見る ------------------------------------------------
//
// フレーム数は波形の長さだけで決まる。上限を超える音声で FFT と mel 変換を全フレーム分
// 回してから拒否すると、600 秒級の音声で数秒同期で占有する（結果は同じ拒否）。観測点は
// 「寸法の壊れた mel 基底」— 特徴抽出に入れば基底の寸法で落ちるので、上限の拒否が先に
// 出ることが「抽出に入らなかった」ことの証拠になる。

/** 10ms フレームをちょうど `frames` 本作る長さの無音。 */
const silenceOfFrames = (frames: number): Float32Array<ArrayBuffer> =>
  new Float32Array(N_FFT + (frames - 1) * HOP);

Deno.test("フレーム数: 長さから数えた本数は特徴抽出の本数と一致する", () => {
  const melBasis = new Float32Array(N_MELS * MEL_BINS);
  for (const samples of [N_FFT, N_FFT + HOP - 1, N_FFT + HOP, SAMPLE_RATE, SAMPLE_RATE + 77]) {
    assertEquals(
      featureFrameCount(samples),
      extractFeatures(new Float32Array(samples), melBasis).frames,
      `samples=${samples}`,
    );
  }
  // 1 フレームも取れない長さは 1 未満（受理の判定は呼び手）。
  assert(featureFrameCount(N_FFT - 1) < 1);
});

Deno.test("detect: 上限を超える音声は特徴を作る前に入力エラーで落とす", async () => {
  const { pipeline } = pipelineWithFakeSession({
    config: config({ maxFrames: 8 }),
    melBasis: new Float32Array(0),
  });

  // 10 本（端数なし）は上限 8 を超える。抽出に入っていれば mel 基底の寸法で落ちる。
  await assertRejects(
    () => pipeline.detect(silenceOfFrames(10)),
    ModelInputError,
    "音声が長すぎる",
  );
});

Deno.test("detect: 端数を落として上限ちょうどになる音声は上限を通って特徴抽出へ進む", async () => {
  const { pipeline } = pipelineWithFakeSession({
    config: config({ maxFrames: 8 }),
    melBasis: new Float32Array(0),
  });

  // 9 本は端数 1 本を落として 8 本 = 上限ちょうど。上限の判定は切り捨て後の値で比べる。
  // NOTE: 境界を固定するテストで、退行は検出しない（上限の検査を消しても、mel 基底の寸法で
  // 同じ文言のまま落ちる）。退行を縛るのは上の「上限超過」のテスト。
  await assertRejects(() => pipeline.detect(silenceOfFrames(9)), Error, "mel 基底が 0 要素");
});
